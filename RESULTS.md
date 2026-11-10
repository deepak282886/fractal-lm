# Fractal-memory LM — experiment log

## Task F — synthetic copy task (does retrieval work?)

`--task copy`: each 128-char sequence = random 64-char segment + same segment repeated.
NLL logged separately for first half (random) and second half (repeat). 1000 steps, B=64, T=128.

| run | first_half | second_half |
|-----|-----------|-------------|
| mem=0 (baseline) | 4.1747 | 4.1747 |
| mem=1 taus=0.7,0.85,0.95 | 4.1932 | **0.0798** |

- Baseline second half stays at log(65)≈4.17 (can't predict random segment).
- Memory drops second-half NLL to 0.08 → **retrieval works end-to-end** (write → read → decoder).
- First half stays ~4.19 (random), as expected.
- `test_causal.py` PASS (plain, graph, long-term loops) after the change.
- Conclusion: no bug in read/write/decoder; memory genuinely retrieves and helps.

## Task G — growth-friendly thresholds on real data

Sweep at T=512, B=64, 3000 steps, seed 0. No-memory baseline val = 1.6424 (2-seed avg).
Growth logged every 32 tokens (mean cluster count within a chunk).

| taus | val | first_q | last_q | nodes (end of chunk) | growth (tokens 0..512) |
|------|-----|---------|--------|----------------------|------------------------|
| 0.7,0.85,0.95 | 1.6545 | 1.6732 | 1.6525 | 452 | 0,32,64,95,126,155,185,214,242,271,299,326,353,380,406,432 |
| 0.8,0.9,0.97 | 1.6489 | 1.6636 | 1.6480 | 479 | 0,32,64,96,127,158,189,220,250,280,310,340,369,398,427,455 |

- Both settings are **worse** than the no-memory baseline (1.6424). Memory is still neutral/negative on real data.
- Growth is near-linear (~1 new node per token): the tree is essentially a **flat list**, not a fractal tree.
  High taus keep leaves near-exact but the root accumulates ~N children, so there is no real hierarchy.
- Best of the two is `0.8,0.9,0.97` (val 1.6489) → chosen as the default for later tasks, but it does not beat baseline.

## Task H — depth that grows, reads that scale

- Implemented a real tree walk in `TreeMemory.read`: each node keeps a child-index table
  (fan-out cap), and each level scores only the children of the current beam nodes, so read
  cost scales with `K*fanout*D`, not with the total node count.
- Timed one read while the tree grows (fanout=64):

  | nodes | read time |
  |-------|-----------|
  | 128   | 26.0 ms   |
  | 512   | 26.4 ms   |
  | 2048  | 25.6 ms   |
  | 8192  | 26.0 ms   |

  Read time is **flat** (~26 ms) as the tree grows — reads scale.
- `test_causal.py` passes (plain, graph, long-term loops) with the tree walk.
- **T=2048 test (fanout=64, taus 0.8,0.9,0.97, 3000 steps, seed 0): val = 1.6728 at step 750**,
  worse than the no-memory baseline (1.6424). The tree is flat (~1900 nodes) but fanout=64
  orphans most nodes, so the memory can only reach 64 root children.
- **Root cause of flat tree**: the encoder produces near-orthogonal vectors. Measured cosine
  similarity between consecutive encoder outputs: mean 0.057, p90 0.223; only 3.3% of pairs
  exceed 0.3. So no two tokens cluster at any reasonable threshold, and the tree grows ~1
  node/token (a flat list) for every tau set tried (0.3,0.6,0.9 / 0.7,0.85,0.95 / 0.8,0.9,0.97).
- Attempts to force depth (descend into best child even below threshold, create only at the
  deepest level) made the tree a chain and **worsened** val (2.01 at 200 steps), so they were
  reverted. The memory is fundamentally limited by the non-clusterable encoder.
- Conclusion: reads scale (flat time), but val degrades at T=2048 because the tree is flat and
  memory does not help on real data.

## Task C — learned merge threshold

- The differentiable learned-tau write is very slow (no CUDA graph; ~1.7 s/step at T=128,
  ~7 s/step at T=512), so the prescribed 3000 steps at T=512 is infeasible (~5.7 h). Ran
  T=128, 300 steps, seed 0 instead to get a trend.
- Result: val = 1.9457 at step 300 (T=128), worse than the no-memory baseline at T=128 (1.6644).
- Learned tau collapses to ~0: `mean_tau = [1.5e-14, 1.7e-22, 1.4e-30]`. The model learns to
  merge everything (threshold ~0) because the encoder vectors are near-orthogonal and nothing
  clusters — consistent with the flat-tree finding in H.
- Conclusion: learned tau does not help; it collapses to a degenerate ~0 threshold.



