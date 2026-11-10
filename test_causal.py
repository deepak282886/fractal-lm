"""Causality test: changing future tokens must not change the loss at earlier positions.

For a fixed batch, run the memory model in eval mode and record per-position nll[:, t].
Then corrupt x[:, t+1:] and the targets after t, rerun, and check nll[:, :t+1] is
unchanged (within 1e-5). Tests several t values for both the plain loop and the
CUDA-graph loop.
"""
import argparse, torch
from fractal_lm import FractalLM, GraphTokenLoop, TreeMemory, get_batch


def load(device, graph, B, T, lt=False):
    text = open('input.txt').read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data)); tr = data[:n]
    model = FractalLM(len(chars), d=96, beams=4, taus=[0.1, 0.3, 0.5], use_mem=True).to(device)
    if graph:
        model.graph_loop = GraphTokenLoop(B, T, 96, model.de, model.dq, 4, len(model.taus), model.taus, device)
    if lt:
        model.use_lt = True
        model.lt_mem = TreeMemory(B, 512, 96, model.de, [0.1, 0.3, 0.5], device)
    return model, tr


def max_diff_at(model, tr, t, device, B, T, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x, y = get_batch(tr, B, T, gen, device)
    model.eval()
    if model.lt_mem is not None:
        model.lt_mem.reset()  # start each run from the same (empty) long-term state
    with torch.no_grad():
        nll1, _ = model(x, y)
    # corrupt future tokens and targets
    x2 = x.clone(); y2 = y.clone()
    x2[:, t + 1:] = torch.randint(0, model.V, x2[:, t + 1:].shape, device=device)
    y2[:, t + 1:] = torch.randint(0, model.V, y2[:, t + 1:].shape, device=device)
    if model.lt_mem is not None:
        model.lt_mem.reset()
    with torch.no_grad():
        nll2, _ = model(x2, y2)
    diff = (nll1[:, :t + 1] - nll2[:, :t + 1]).abs().max().item()
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=16)
    ap.add_argument('--T', type=int, default=128)
    ap.add_argument('--ts', type=str, default='0,16,32,64,96,127')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    ts = [int(x) for x in a.ts.split(',')]

    for graph in (False, True):
        model, tr = load(a.device, graph, a.B, a.T)
        print(f"=== loop={'graph' if graph else 'plain'} ===", flush=True)
        ok = True
        for t in ts:
            d = max_diff_at(model, tr, t, a.device, a.B, a.T)
            status = 'OK' if d < 1e-5 else 'LEAK'
            if d >= 1e-5:
                ok = False
            print(f"  t={t:>3}  max|d nll[:{t + 1}]| = {d:.3e}  {status}", flush=True)
        print(f"  => {'PASS' if ok else 'FAIL'}", flush=True)

    # long-term memory path (causal interleaved write/read)
    model, tr = load(a.device, False, a.B, a.T, lt=True)
    print("=== loop=long-term ===", flush=True)
    ok = True
    for t in ts:
        d = max_diff_at(model, tr, t, a.device, a.B, a.T)
        status = 'OK' if d < 1e-5 else 'LEAK'
        if d >= 1e-5:
            ok = False
        print(f"  t={t:>3}  max|d nll[:{t + 1}]| = {d:.3e}  {status}", flush=True)
    print(f"  => {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == '__main__':
    main()
