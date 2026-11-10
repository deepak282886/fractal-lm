"""Time one TreeMemory.read() as the tree grows to various sizes (Task H).

Grows the tree by writing random keys/values (high taus -> mostly new children),
then times a single read. Reports ms/read at each size.
"""
import argparse, time
import torch
from fractal_lm import TreeMemory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=str, default='128,512,2048,8192')
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--d', type=int, default=96)
    ap.add_argument('--dq', type=int, default=48)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--fanout', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(',')]
    taus = [0.7] * a.D
    for N in sizes:
        mem = TreeMemory(a.B, N, a.d, a.dq, taus, a.device, fanout=a.fanout)
        gen = torch.Generator().manual_seed(0)
        for i in range(N):
            key = torch.randn(a.B, a.d, device=a.device)
            val = torch.randn(a.B, a.dq, device=a.device)
            mem.write(key, val)
        q = torch.randn(a.B, a.dq, device=a.device)
        wk = torch.randn(a.dq, a.d, device=a.device)
        for _ in range(5):
            mem.read(q, wk, a.K)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            mem.read(q, wk, a.K)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 20
        print(f"N={N:>5}  nodes={mem.nn.float().mean().item() - 1:.0f}  read={dt * 1e3:.3f} ms", flush=True)


if __name__ == '__main__':
    main()
