"""
Fractal-memory language model (short-term memory path) on tinyshakespeare.

Encoder   : v = LN(x + W2 gelu(W1 x))            original + learned generalisation (bottleneck)
Memory    : tree of clusters (node 0 = virtual root). Each node = (key centre, value centre, count).
            Write = descend by best child; merge into it if cosine > tau[level], else start a new child.
            Every node on the merge path is updated as a running mean, so parents are coarser clusters.
Read      : beam search (k threads) down the tree; each thread = path of centres, general -> specific.
Decoder   : per thread, pooled path (learned level weights) + v -> next-char logits.
Mixture   : p = sum_t w_t p_t, w_t from match scores; a "null" thread (no memory) is always present.
Loss      : next-token NLL only.

Memory is reset for every sequence chunk (like a KV cache), and is treated as constant data
(no gradient into stored centres). Read-time scoring, path pooling, thread weights, decoder and
encoder all train through the loss.
"""
import argparse, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True  # faster matmuls on Ampere+; keep float32


# --------------------------------------------------------------------------------------
# Tree memory (pure tensor ops, no autograd)
# --------------------------------------------------------------------------------------
class TreeMemory:
    def __init__(self, B, N, dk, dv, taus, device, fanout=64):
        self.B, self.N, self.taus, self.D = B, N, taus, len(taus)
        self.fanout = fanout
        self.keys = torch.zeros(B, N, dk, device=device)
        self.vals = torch.zeros(B, N, dv, device=device)
        self.cnt = torch.zeros(B, N, device=device)
        self.parent = torch.full((B, N), -1, dtype=torch.long, device=device)
        self.children = torch.full((B, N, fanout), -1, dtype=torch.long, device=device)
        self.nn = torch.ones(B, dtype=torch.long, device=device)  # node 0 is the root
        self.ar = torch.arange(N, device=device)

    def reset(self):
        """Reset all memory buffers (call once per chunk, outside any CUDA graph)."""
        self.keys.zero_()
        self.vals.zero_()
        self.cnt.zero_()
        self.parent.fill_(-1)
        self.children.fill_(-1)
        self.nn.fill_(1)

    @torch.no_grad()
    def write(self, key, val):
        B, N = self.B, self.N
        ar = self.ar[:B]
        kn = key / (key.norm(dim=-1, keepdim=True) + 1e-6)
        cur = torch.zeros(B, dtype=torch.long, device=key.device)
        active = torch.ones(B, dtype=torch.bool, device=key.device)
        for l in range(self.D):
            # children of current node via child-index table (fan-out capped)
            ch = self.children[ar, cur]  # [B, fanout]
            ch_valid = (ch >= 0) & active[:, None]  # [B, fanout]
            ch_flat = ch.clamp(min=0)  # [B, fanout]
            ch_keys = self.keys.gather(1, ch_flat[:, :, None].expand(-1, -1, self.keys.shape[-1]))
            cos = torch.einsum('bd,bnd->bn', kn, ch_keys) / (ch_keys.norm(dim=-1) + 1e-6)
            cos = cos.masked_fill(~ch_valid, -2.0)
            best, bi = cos.max(-1)  # bi: index into fanout
            has = ch_valid.any(-1)
            # no force-merge at the last level: create a deeper child instead
            merge = active & has & (best > self.taus[l])
            create = active & ~merge & (self.nn < N)
            # merge: running-mean update on the best child (masked, no .nonzero sync)
            nodes = ch_flat.gather(1, bi[:, None]).squeeze(-1)  # [B] node id of best child
            c = self.cnt[ar, nodes] + 1
            newk = self.keys[ar, nodes] + (key - self.keys[ar, nodes]) / c[:, None]
            newv = self.vals[ar, nodes] + (val - self.vals[ar, nodes]) / c[:, None]
            self.keys[ar, nodes] = torch.where(merge[:, None], newk, self.keys[ar, nodes])
            self.vals[ar, nodes] = torch.where(merge[:, None], newv, self.vals[ar, nodes])
            self.cnt[ar, nodes] = torch.where(merge, c, self.cnt[ar, nodes])
            # create: start a new child at slot self.nn (masked)
            new = self.nn.clamp(max=N - 1)
            self.keys[ar, new] = torch.where(create[:, None], key, self.keys[ar, new])
            self.vals[ar, new] = torch.where(create[:, None], val, self.vals[ar, new])
            self.cnt[ar, new] = torch.where(create, torch.ones_like(self.cnt[ar, new]), self.cnt[ar, new])
            self.parent[ar, new] = torch.where(create, cur, self.parent[ar, new])
            # add the new child to the parent's child-index table (first empty slot)
            ch_cur = self.children[ar, cur]  # [B, fanout]
            first_empty = (ch_cur < 0).long().argmax(-1)  # [B]
            has_empty = (ch_cur < 0).any(-1)
            add = create & has_empty
            self.children[ar, cur, first_empty] = torch.where(add, new, self.children[ar, cur, first_empty])
            self.nn += create.long()  # in-place so CUDA-graph replay mutates the persistent buffer
            cur = torch.where(merge, nodes, cur)
            active = merge

    def write_learned(self, key, val, tau_w, tau_b, gate_k):
        """Differentiable write with a learned merge threshold.

        tau = sigmoid(w·v + b) per level; soft gate g = sigmoid(k·(sim - tau)).
        The stored key/value centres are updated by blending with the soft gate
        (gradient flows through the write within a chunk); the merge/create
        decision is hard (straight-through for 'create a new cluster').
        key:[B,d], val:[B,dv], tau_w:[D,dv], tau_b:[D], gate_k:[D].
        """
        B, N = self.B, self.N
        ar = self.ar[:B]
        valid = self.ar[None, :] < self.nn[:, None]
        knorm = self.keys.detach().norm(dim=-1) + 1e-6
        kn = key / (key.norm(dim=-1, keepdim=True) + 1e-6)
        cur = torch.zeros(B, dtype=torch.long, device=key.device)
        active = torch.ones(B, dtype=torch.bool, device=key.device)
        for l in range(self.D):
            childm = (self.parent == cur[:, None]) & valid & active[:, None]
            cos = torch.einsum('bd,bnd->bn', kn, self.keys.detach()) / knorm
            cos = cos.masked_fill(~childm, -2.0)
            best, bi = cos.max(-1)
            has = childm.any(-1)
            tau = torch.sigmoid(val @ tau_w[l] + tau_b[l])  # [B]
            g = torch.sigmoid(gate_k[l] * (best - tau))  # [B] soft merge gate
            merge = active & has & (best > tau.detach())
            create = active & ~merge & (self.nn < N)
            # differentiable value update (blend by gate g)
            nodes = bi
            c = self.cnt[ar, nodes] + 1
            newk = self.keys[ar, nodes] + g[:, None] * (key - self.keys[ar, nodes]) / c[:, None]
            newv = self.vals[ar, nodes] + g[:, None] * (val - self.vals[ar, nodes]) / c[:, None]
            keys = self.keys.clone()
            keys[ar, nodes] = torch.where(merge[:, None], newk, self.keys[ar, nodes])
            vals = self.vals.clone()
            vals[ar, nodes] = torch.where(merge[:, None], newv, self.vals[ar, nodes])
            # create a new cluster (straight-through: hard assignment)
            new = self.nn.clamp(max=N - 1)
            keys[ar, new] = torch.where(create[:, None], key, keys[ar, new])
            vals[ar, new] = torch.where(create[:, None], val, vals[ar, new])
            self.cnt[ar, new] = torch.where(create, torch.ones_like(self.cnt[ar, new]), self.cnt[ar, new])
            self.parent[ar, new] = torch.where(create, cur, self.parent[ar, new])
            ch_cur = self.children[ar, cur]
            first_empty = (ch_cur < 0).long().argmax(-1)
            has_empty = (ch_cur < 0).any(-1)
            add = create & has_empty
            self.children[ar, cur, first_empty] = torch.where(add, new, self.children[ar, cur, first_empty])
            self.nn += create.long()
            self.keys = keys
            self.vals = vals
            cur = torch.where(merge, bi, cur)
            active = merge

    def write_gated(self, key, val, gate):
        """Differentiable gated write for long-term memory. gate:[B] in [0,1].
        The gate scales the value update (gradient flows to the gate network)."""
        B, N = self.B, self.N
        ar = self.ar[:B]
        valid = self.ar[None, :] < self.nn[:, None]
        knorm = self.keys.detach().norm(dim=-1) + 1e-6
        kn = key / (key.norm(dim=-1, keepdim=True) + 1e-6)
        cur = torch.zeros(B, dtype=torch.long, device=key.device)
        active = torch.ones(B, dtype=torch.bool, device=key.device)
        for l in range(self.D):
            childm = (self.parent == cur[:, None]) & valid & active[:, None]
            cos = torch.einsum('bd,bnd->bn', kn, self.keys.detach()) / knorm
            cos = cos.masked_fill(~childm, -2.0)
            best, bi = cos.max(-1)
            has = childm.any(-1)
            merge = active & has & (best > self.taus[l])
            create = active & ~merge & (self.nn < N)
            nodes = bi
            c = self.cnt[ar, nodes] + 1
            newk = self.keys[ar, nodes] + gate[:, None] * (key - self.keys[ar, nodes]) / c[:, None]
            newv = self.vals[ar, nodes] + gate[:, None] * (val - self.vals[ar, nodes]) / c[:, None]
            keys = self.keys.clone()
            keys[ar, nodes] = torch.where(merge[:, None], newk, self.keys[ar, nodes])
            vals = self.vals.clone()
            vals[ar, nodes] = torch.where(merge[:, None], newv, self.vals[ar, nodes])
            new = self.nn.clamp(max=N - 1)
            keys[ar, new] = torch.where(create[:, None], key, keys[ar, new])
            vals[ar, new] = torch.where(create[:, None], val, vals[ar, new])
            self.cnt[ar, new] = torch.where(create, torch.ones_like(self.cnt[ar, new]), self.cnt[ar, new])
            self.parent[ar, new] = torch.where(create, cur, self.parent[ar, new])
            ch_cur = self.children[ar, cur]
            first_empty = (ch_cur < 0).long().argmax(-1)
            has_empty = (ch_cur < 0).any(-1)
            add = create & has_empty
            self.children[ar, cur, first_empty] = torch.where(add, new, self.children[ar, cur, first_empty])
            self.nn += create.long()
            self.keys = keys
            self.vals = vals
            cur = torch.where(merge, bi, cur)
            active = merge

    def detach_state(self):
        """Detach the differentiable memory state (stop-gradient across chunks)."""
        self.keys = self.keys.detach()
        self.vals = self.vals.detach()

    @torch.no_grad()
    def read(self, q, wk, K):
        """Beam search (tree walk). q: [B,dq] projected query, wk: [dq,d] key-projection weight.
        Each level scores only the children of the current beam nodes (child-index table),
        so cost scales with K*fanout per level, not with the total node count.
        Returns path node ids [B,K,D] (-1 = none)."""
        B, N, D, F = self.B, self.N, self.D, self.fanout
        dev = q.device
        beam = torch.zeros(B, K, dtype=torch.long, device=dev)
        alive = torch.zeros(B, K, dtype=torch.bool, device=dev)
        alive[:, 0] = True
        acc = torch.zeros(B, K, device=dev)
        path = torch.full((B, K, D), -1, dtype=torch.long, device=dev)
        ninf = float('-inf')
        for l in range(D):
            # children of the current beam nodes: [B, K, F]
            ch = self.children.gather(1, beam[:, :, None].expand(-1, -1, F))  # [B,K,F]
            ch_valid = (ch >= 0) & alive[:, :, None]  # [B,K,F]
            ch_flat = ch.clamp(min=0).reshape(B, K * F)  # [B, K*F]
            # score only the children (gather their keys, project, dot with query)
            ch_keys = self.keys.gather(1, ch_flat[:, :, None].expand(-1, -1, self.keys.shape[-1]))
            proj = ch_keys @ wk.t()  # [B, K*F, dq]
            sim = torch.einsum('bd,bnd->bn', q, proj) / math.sqrt(q.shape[-1])  # [B, K*F]
            sim = sim.reshape(B, K, F)
            cand = (acc[:, :, None] + sim).masked_fill(~ch_valid, ninf)  # [B,K,F]
            has_child = ch_valid.any(-1)
            stay = torch.where(alive & ~has_child, acc, torch.full_like(acc, ninf))  # [B,K]
            allc = torch.cat([cand.reshape(B, K * F), stay], -1)  # [B, K*F + K]
            top, idx = allc.reshape(B, -1).topk(K, dim=-1)
            is_stay = idx >= K * F
            src = torch.where(is_stay, idx - K * F, idx // F)  # source beam thread
            sel_child = ch_flat.gather(1, idx.clamp(max=K * F - 1))  # child node id
            sel = torch.where(is_stay, torch.full_like(idx, -1), sel_child)
            newnode = torch.where(is_stay, beam.gather(1, src), sel.clamp(max=N - 1))
            newpath = path.gather(1, src[:, :, None].expand(-1, -1, D)).clone()
            newpath[:, :, l] = torch.where(is_stay, torch.full_like(sel, -1), sel)
            beam, acc, path = newnode, top, newpath
            alive = torch.isfinite(top)
        return path

    def gather(self, path):
        B, K, D = path.shape
        mask = path >= 0
        nodes = path.clamp(min=0).reshape(B, K * D, 1)
        kp = self.keys.gather(1, nodes.expand(-1, -1, self.keys.shape[-1])).reshape(B, K, D, -1)
        vp = self.vals.gather(1, nodes.expand(-1, -1, self.vals.shape[-1])).reshape(B, K, D, -1)
        return kp * mask[..., None], vp * mask[..., None], mask


# --------------------------------------------------------------------------------------
# CUDA-graph token loop
# --------------------------------------------------------------------------------------
class GraphTokenLoop:
    """CUDA-graph accelerated token loop.

    Captures one write+read+gather step as a CUDA graph and replays it T times,
    copying each token's inputs into static buffers. The memory buffers are
    persistent and reset once per chunk (outside the graph). Falls back to a
    plain loop if capture fails.
    """
    def __init__(self, B, T, d, de, dq, K, D, taus, device, fanout=64):
        self.B, self.T, self.K, self.D = B, T, K, D
        self.device = device
        self.mem = TreeMemory(B, T + 1, d, de, taus, device, fanout=fanout)
        # static input buffers
        self.key_in = torch.zeros(B, d, device=device)
        self.val_in = torch.zeros(B, de, device=device)
        self.q_in = torch.zeros(B, dq, device=device)
        self.wk_in = torch.zeros(dq, d, device=device)
        # static output buffers
        self.kp_out = torch.zeros(B, K, D, d, device=device)
        self.vp_out = torch.zeros(B, K, D, de, device=device)
        self.m_out = torch.zeros(B, K, D, dtype=torch.bool, device=device)
        self.graph_wr = None
        self.graph_read = None
        self.ok = False
        self.log_growth = False
        self.growth = []
        try:
            self._capture()
            self.ok = True
        except Exception as e:
            print(f"[graph] capture failed, falling back to plain loop: {e}", flush=True)

    def _step_wr(self):
        self.mem.write(self.key_in, self.val_in)
        path = self.mem.read(self.q_in, self.wk_in, self.K)
        kp, vp, m = self.mem.gather(path)
        self.kp_out.copy_(kp); self.vp_out.copy_(vp); self.m_out.copy_(m)

    def _step_read(self):
        path = self.mem.read(self.q_in, self.wk_in, self.K)
        kp, vp, m = self.mem.gather(path)
        self.kp_out.copy_(kp); self.vp_out.copy_(vp); self.m_out.copy_(m)

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.mem.reset()
                self._step_wr()
                self._step_read()
        torch.cuda.current_stream().wait_stream(s)
        self.mem.reset()
        self.graph_wr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph_wr):
            self._step_wr()
        self.graph_read = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph_read):
            self._step_read()

    def run(self, vd, ex, q_all, wk_weight):
        """Run the token loop. vd:[B,T,d], ex:[B,T,de], q_all:[B,T,dq], wk_weight:[dq,d].
        Returns Kp, Vp, M as [B,T,K,D,*]. If log_growth, fills self.growth with the
        mean cluster count at every 32nd token."""
        self.mem.reset()
        self.wk_in.copy_(wk_weight)
        Kp, Vp, Mk = [], [], []
        self.growth = []
        self.q_in.copy_(q_all[:, 0])
        self.graph_read.replay()
        Kp.append(self.kp_out.clone()); Vp.append(self.vp_out.clone()); Mk.append(self.m_out.clone())
        if self.log_growth:
            self.growth.append(self.mem.nn.float().mean().item() - 1)
        for t in range(1, self.T):
            self.key_in.copy_(vd[:, t - 1])
            self.val_in.copy_(ex[:, t])
            self.q_in.copy_(q_all[:, t])
            self.graph_wr.replay()
            Kp.append(self.kp_out.clone()); Vp.append(self.vp_out.clone()); Mk.append(self.m_out.clone())
            if self.log_growth and t % 32 == 0:
                self.growth.append(self.mem.nn.float().mean().item() - 1)
        return torch.stack(Kp, 1), torch.stack(Vp, 1), torch.stack(Mk, 1)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class FractalLM(nn.Module):
    def __init__(self, V, d=96, de=32, win=8, dq=48, beams=4, taus=(0.3, 0.6, 0.85),
                 dec_hid=384, use_mem=True):
        super().__init__()
        self.V, self.d, self.de, self.win, self.dq = V, d, de, win, dq
        self.K, self.taus, self.D, self.use_mem = beams, list(taus), len(taus), use_mem
        self.emb = nn.Embedding(V + 1, de)  # last id = padding
        self.inp = nn.Linear(win * de, d)
        self.w1, self.w2 = nn.Linear(d, d // 2), nn.Linear(d // 2, d)  # compression -> growth
        self.ln = nn.LayerNorm(d)
        self.Wq, self.Wk = nn.Linear(d, dq, bias=False), nn.Linear(d, dq, bias=False)
        self.Wp = nn.Linear(d, d + de, bias=False)
        self.e_lvl = nn.Parameter(torch.zeros(self.D))
        self.s0 = nn.Parameter(torch.zeros(1))
        self.temp = nn.Parameter(torch.ones(1))
        # learned merge threshold: tau = sigmoid(w·v + b) per level, gate g = sigmoid(k·(sim - tau))
        self.tau_w = nn.Parameter(torch.randn(self.D, de) * 0.1)
        self.tau_b = nn.Parameter(torch.zeros(self.D))
        self.gate_k = nn.Parameter(torch.ones(self.D))
        # long-term memory: tag network + write gate (persistent TreeMemory set externally)
        self.lt_tag = nn.Linear(d, d, bias=False)
        self.lt_gate = nn.Linear(d, 1)
        self.lt_mem = None
        self.use_lt = False
        self.dec = nn.Sequential(nn.Linear(d + d + de, dec_hid), nn.GELU(), nn.Linear(dec_hid, V))
        self.graph_loop = None  # optional CUDA-graph token loop (set externally)
        self.learned_tau = False  # use differentiable learned merge threshold
        self.log_growth = False  # record cluster count vs token position (every 32 tokens)
        self.fanout = 64  # child-index table fan-out cap for the tree walk

    def encode(self, x):
        B, T = x.shape
        pad = torch.full((B, self.win - 1), self.V, dtype=x.dtype, device=x.device)
        w = torch.cat([pad, x], 1).unfold(1, self.win, 1)  # [B,T,win]
        h = self.inp(self.emb(w).reshape(B, T, -1))
        return self.ln(h + self.w2(F.gelu(self.w1(h))))

    def forward(self, x, y):
        B, T = x.shape
        d, de, K, D = self.d, self.de, self.K, self.D
        v = self.encode(x)
        stats = {}
        if self.use_mem:
            if self.learned_tau:
                # differentiable learned-threshold write (gradient flows through writes)
                mem = TreeMemory(B, T + 1, d, de, self.taus, x.device, fanout=self.fanout)
                vd = v.detach()
                q_all = self.Wq(vd)
                ex = self.emb(x).detach()
                Kp, Vp, Mk = [], [], []
                for t in range(T):
                    if t > 0:  # entry for step t-1: key = its encoding, value = the token that followed it
                        mem.write_learned(vd[:, t - 1], ex[:, t], self.tau_w, self.tau_b, self.gate_k)
                    with torch.no_grad():
                        path = mem.read(q_all[:, t], self.Wk.weight.detach(), K)
                    kp, vp, m = mem.gather(path)  # differentiable gather
                    Kp.append(kp); Vp.append(vp); Mk.append(m)
                Kp, Vp, M = torch.stack(Kp, 1), torch.stack(Vp, 1), torch.stack(Mk, 1)  # [B,T,K,D,*]
                stats['nodes'] = mem.nn.float().mean().item() - 1
            elif self.graph_loop is not None and self.graph_loop.ok:
                with torch.no_grad():
                    vd = v.detach()
                    q_all = self.Wq(vd)
                    ex = self.emb(x).detach()
                    Kp, Vp, M = self.graph_loop.run(vd, ex, q_all, self.Wk.weight.detach())
                    stats['nodes'] = self.graph_loop.mem.nn.float().mean().item() - 1
                    if self.log_growth:
                        stats['growth'] = list(self.graph_loop.growth)
            else:
                mem = TreeMemory(B, T + 1, d, de, self.taus, x.device, fanout=self.fanout)
                with torch.no_grad():
                    vd = v.detach()
                    q_all = self.Wq(vd)
                    ex = self.emb(x).detach()
                    Kp, Vp, Mk = [], [], []
                    for t in range(T):
                        if t > 0:  # entry for step t-1: key = its encoding, value = the token that followed it
                            mem.write(vd[:, t - 1], ex[:, t])
                        path = mem.read(q_all[:, t], self.Wk.weight.detach(), K)
                        kp, vp, m = mem.gather(path)
                        Kp.append(kp); Vp.append(vp); Mk.append(m)
                    Kp, Vp, M = torch.stack(Kp, 1), torch.stack(Vp, 1), torch.stack(Mk, 1)  # [B,T,K,D,*]
                    stats['nodes'] = mem.nn.float().mean().item() - 1
                    if self.log_growth:
                        stats['growth'] = [mem.nn.float().mean().item() - 1
                                           for t in range(0, T, 32)]
            if self.use_lt and self.lt_mem is not None:
                # long-term memory: persists across chunks, feeds the same mixture.
                # Causal: write entry for t-1 before reading at t (no future leak).
                self.lt_mem.detach_state()  # stop-gradient on older chunks
                vd = v.detach()
                ex = self.emb(x).detach()
                tags = self.lt_tag(vd)  # [B,T,d]
                gates = torch.sigmoid(self.lt_gate(vd))  # [B,T,1]
                q_all = self.Wq(vd)
                Kp_lt, Vp_lt, Mk_lt = [], [], []
                lt_fire = 0.0
                for t in range(T):
                    if t > 0:
                        self.lt_mem.write_gated(tags[:, t - 1], ex[:, t - 1], gates[:, t - 1, 0])
                        lt_fire += (gates[:, t - 1, 0] > 0.5).float().mean().item()
                    with torch.no_grad():
                        path = self.lt_mem.read(q_all[:, t], self.Wk.weight.detach(), K)
                    kp, vp, m = self.lt_mem.gather(path)  # differentiable gather
                    Kp_lt.append(kp); Vp_lt.append(vp); Mk_lt.append(m)
                stats['lt_gate_fire'] = lt_fire / max(1, T - 1)
                Kp_lt, Vp_lt, M_lt = torch.stack(Kp_lt, 1), torch.stack(Vp_lt, 1), torch.stack(Mk_lt, 1)
                Kp = torch.cat([Kp, Kp_lt], 2)
                Vp = torch.cat([Vp, Vp_lt], 2)
                M = torch.cat([M, M_lt], 2)
                K = K + K  # long-term adds K more threads
            q = self.Wq(v)
            sim = (q[:, :, None, None, :] * self.Wk(Kp)).sum(-1) / math.sqrt(self.dq)
            sim = sim * M
            tvalid = M.any(-1)
            s_thr = self.temp * sim.sum(-1) / M.sum(-1).clamp(min=1)
            logits_w = torch.cat([s_thr.masked_fill(~tvalid, -1e9), self.s0.expand(B, T, 1)], -1)
            logw = F.log_softmax(logits_w, -1)  # [B,T,K+1], null thread last
            C = torch.cat([Kp, Vp], -1)
            pl = (self.Wp(v)[:, :, None, None, :] * C).sum(-1) / math.sqrt(d + de) + self.e_lvl
            a = F.softmax(pl.masked_fill(~M, -1e9), -1) * M
            m = (a[..., None] * C).sum(-2)  # pooled path per thread
            m = torch.cat([m, torch.zeros(B, T, 1, d + de, device=x.device)], 2)
            Kn = K + 1
        else:
            logw = torch.zeros(B, T, 1, device=x.device)
            m = torch.zeros(B, T, 1, d + de, device=x.device)
            Kn = 1
        vin = v[:, :, None, :].expand(-1, -1, Kn, -1)
        logp = F.log_softmax(self.dec(torch.cat([vin, m], -1)), -1)  # [B,T,Kn,V]
        logmix = torch.logsumexp(logw[..., None] + logp, dim=2)  # [B,T,V]
        nll = -logmix.gather(-1, y[..., None]).squeeze(-1)  # [B,T]
        if self.use_mem:
            stats['null_w'] = logw[..., -1].exp()  # [B,T]
        return nll, stats


# --------------------------------------------------------------------------------------
# Data / train / eval
# --------------------------------------------------------------------------------------
def get_batch(data, B, T, gen, device, task='real', V=None):
    """Sample a batch. task='copy' generates synthetic copy sequences:
    each sequence of length T = a random T//2-char segment followed by the same
    segment repeated. Targets are x shifted by 1 (last target wraps to seg[0])."""
    if task == 'copy':
        half = T // 2
        seg = torch.randint(0, V, (B, half), generator=gen)  # CPU generator
        x = torch.cat([seg, seg], 1)  # [B, T]
        y = torch.cat([x[:, 1:], seg[:, :1]], 1)  # y[:, t] = x[:, t+1], last wraps
        return x.to(device), y.to(device)
    ix = torch.randint(0, len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, B, T, device, nb=20, seed=123, task='real', V=None):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    pos = torch.zeros(T, device=device); nulls = torch.zeros(T, device=device); nodes = []
    for _ in range(nb):
        x, y = get_batch(data, B, T, gen, device, task, V)
        nll, st = model(x, y)
        pos += nll.mean(0)
        if 'null_w' in st:
            nulls += st['null_w'].mean(0); nodes.append(st['nodes'])
    model.train()
    pos /= nb; nulls /= nb; q = T // 4
    r = dict(val=pos.mean().item(), first_q=pos[:q].mean().item(), last_q=pos[-q:].mean().item(),
             null_w_last_q=nulls[-q:].mean().item() if nodes else None,
             nodes=sum(nodes) / len(nodes) if nodes else None)
    if task == 'copy':
        h = T // 2
        r['first_half'] = pos[:h].mean().item()
        r['second_half'] = pos[h:].mean().item()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mem', type=int, default=1)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--d', type=int, default=96)
    ap.add_argument('--beams', type=int, default=4)
    ap.add_argument('--taus', type=str, default='0.1,0.3,0.5')
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='run')
    ap.add_argument('--data', type=str, default='input.txt')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--graph', type=int, default=0, help='use CUDA-graph token loop (cuda only)')
    ap.add_argument('--learned_tau', type=int, default=0, help='use differentiable learned merge threshold')
    ap.add_argument('--lt_mem', type=int, default=0, help='use long-term memory')
    ap.add_argument('--task', type=str, default='real', help="'real' or 'copy' (synthetic copy task)")
    ap.add_argument('--log_growth', type=int, default=0, help='log cluster count vs token position (every 32 tokens)')
    ap.add_argument('--fanout', type=int, default=64, help='child-index table fan-out cap for the tree walk')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = a.device

    text = open(a.data).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); tr, va = data[:n], data[n:]

    model = FractalLM(len(chars), d=a.d, beams=a.beams, taus=[float(t) for t in a.taus.split(',')],
                      use_mem=bool(a.mem)).to(device)
    model.learned_tau = bool(a.learned_tau)
    model.log_growth = bool(a.log_growth)
    model.fanout = a.fanout
    if a.lt_mem and a.mem:
        model.use_lt = True
        model.lt_mem = TreeMemory(a.B, 512, a.d, model.de, [float(t) for t in a.taus.split(',')], device, fanout=a.fanout)
        print(f"[lt] long-term memory on (cap 512)", flush=True)
    if a.graph and a.mem and not a.learned_tau and device.startswith('cuda'):
        model.graph_loop = GraphTokenLoop(a.B, a.T, a.d, model.de, model.dq, a.beams,
                                          len(model.taus), model.taus, device, fanout=a.fanout)
        model.graph_loop.log_growth = bool(a.log_growth)
        print(f"[graph] token loop: {'CUDA graph' if model.graph_loop.ok else 'plain loop'}", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  mem={a.mem}  device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    gen = torch.Generator().manual_seed(a.seed)
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        x, y = get_batch(tr, a.B, a.T, gen, device, a.task, len(chars))
        nll, st = model(x, y)
        loss = nll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            extra = f" nodes={st['nodes']:.0f}" if st else ""
            if a.task == 'copy':
                h = a.T // 2
                extra += f" first={nll[:, :h].mean().item():.3f} second={nll[:, h:].mean().item():.3f}"
            if a.log_growth and 'growth' in st:
                g = st['growth']
                extra += " growth=" + ",".join(f"{v:.0f}" for v in g)
            print(f"step {step} loss {loss.item():.4f}{extra} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(model, va, a.B, a.T, device, task=a.task, V=len(chars)); r['step'] = step; log.append(r)
            print("EVAL", json.dumps(r), flush=True)
            json.dump(log, open(f"{a.out}.json", 'w'))
    if a.learned_tau:
        with torch.no_grad():
            x, y = get_batch(va, a.B, a.T, gen, device)
            ex = model.emb(x)
            mean_tau = torch.sigmoid(model.tau_b + torch.einsum('btd,dl->bl', ex, model.tau_w.t())).mean(0)
            print("LEARNED_TAU", json.dumps({
                'tau_b': model.tau_b.detach().cpu().tolist(),
                'gate_k': model.gate_k.detach().cpu().tolist(),
                'mean_tau': mean_tau.detach().cpu().tolist(),
            }), flush=True)


if __name__ == '__main__':
    main()