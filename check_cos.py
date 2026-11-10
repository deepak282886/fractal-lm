"""Quick diagnostic: how similar are encoder outputs to each other?"""
import torch, torch.nn.functional as F
from fractal_lm import FractalLM, get_batch

torch.manual_seed(0)
text = open('input.txt').read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)); tr = data[:n]

model = FractalLM(len(chars), d=96, beams=4, taus=[0.3, 0.6, 0.9], use_mem=True)
model.eval()
gen = torch.Generator().manual_seed(0)
x, y = get_batch(tr, 64, 512, gen, 'cpu', 'real', len(chars))
with torch.no_grad():
    v = model.encode(x)  # [B,T,d]
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    # cosine between consecutive positions and between random pairs
    c_consec = (v[:, 1:] * v[:, :-1]).sum(-1)
    # random pairs within a sequence
    i = torch.randint(0, 512, (64, 2000))
    j = torch.randint(0, 512, (64, 2000))
    c_rand = (v.gather(1, i[:, :, None].expand(-1, -1, 96)) * v.gather(1, j[:, :, None].expand(-1, -1, 96))).sum(-1)
    print(f"cos consecutive: mean={c_consec.mean().item():.3f} p90={c_consec.reshape(-1).quantile(0.9).item():.3f}")
    print(f"cos random    : mean={c_rand.mean().item():.3f} p90={c_rand.reshape(-1).quantile(0.9).item():.3f}")
    print(f"cos consecutive >0.3: {(c_consec > 0.3).float().mean().item():.3f}")
    print(f"cos consecutive >0.6: {(c_consec > 0.6).float().mean().item():.3f}")
    print(f"cos consecutive >0.9: {(c_consec > 0.9).float().mean().item():.3f}")
