"""Compare mem=0 vs mem=1 on the GPU, averaged over seeds.

Reads the final eval (last step) from each run's JSON and prints val, first_q,
last_q and null_w_last_q side by side, averaged over the two seeds.
"""
import argparse, json, os


def load_final(path):
    with open(path) as f:
        log = json.load(f)
    return log[-1]  # last eval (final step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, default='results')
    ap.add_argument('--seeds', type=str, default='0,1')
    ap.add_argument('--steps', type=int, default=600)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    def run(mem, seed):
        return load_final(os.path.join(a.dir, f'mem{mem}_s{seed}.json'))

    metrics = ['val', 'first_q', 'last_q', 'null_w_last_q']
    # collect per-mem, per-metric lists across seeds
    mem0 = {m: [] for m in metrics}
    mem1 = {m: [] for m in metrics}
    for s in seeds:
        r0, r1 = run(0, s), run(1, s)
        for m in metrics:
            mem0[m].append(r0.get(m))
            mem1[m].append(r1.get(m))

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float('nan')

    print(f"{'metric':<14}{'mem=0':>12}{'mem=1':>12}{'delta':>12}")
    print('-' * 50)
    for m in metrics:
        a0, a1 = avg(mem0[m]), avg(mem1[m])
        print(f"{m:<14}{a0:>12.4f}{a1:>12.4f}{a1 - a0:>12.4f}")

    # per-seed detail
    print('\nper-seed detail:')
    for s in seeds:
        r0, r1 = run(0, s), run(1, s)
        print(f"  seed {s}: mem0 val={r0['val']:.4f} last_q={r0['last_q']:.4f} | "
              f"mem1 val={r1['val']:.4f} last_q={r1['last_q']:.4f} null_w_last_q={r1.get('null_w_last_q')}")


if __name__ == '__main__':
    main()
