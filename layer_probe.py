"""
layer_probe.py - layer-by-layer probe of three embedding / decision models:

    gemma : google/embeddinggemma-2            (text core of EmbeddingGemma 2)
    qwen  : Qwen/Qwen3-Embedding-0.6B          (28 layers, 1024-dim)
    laya  : convaiinnovations/laya             (ModernBERT-large + decision head; English checkpoint)

For a fixed set of sentences it records, at every layer:
  * how far each layer moves the token vectors (cosine + relative change + norm)
  * where paraphrases / negations / unrelated pairs separate
  * geometry (anisotropy, effective dimensionality, similarity to the final layer)
  * cross-model layer similarity (linear CKA)

Install:   pip install torch transformers sentence-transformers numpy matplotlib
Run:       python layer_probe.py extract            # GPU/CPU, loads one model at a time
           python layer_probe.py analyze            # numpy + matplotlib only
           python layer_probe.py analyze --standardize   # same, but z-scores each dimension (outlier-robust)
           python layer_probe.py analyze --standardize --cka-pool nosink   # choose the pooling used for CKA
           python layer_probe.py all
           python layer_probe.py extract --models qwen laya
           python layer_probe.py selftest           # synthetic data, no downloads, checks analysis code

Notes
  * Everything runs in float32 by default (all three fit in 8 GB, one at a time) so the
    activations are not distorted by half-precision rounding. Use --dtype bfloat16 if you
    run out of memory.
  * "hidden_states[0]" is the embedding-layer output, "hidden_states[-1]" is usually taken
    AFTER the model's final norm in HF, so the last step includes that norm.
  * Pooling: "mean" is used for all models (comparable). "nosink" skips the first token and "nomax"
    skips the single highest-norm token at each layer (both guard against massive-activation tokens).
    "native" is the model's own convention as far as I know: last-token for Qwen3-Embedding, mean for
    the other two (Gemma 2 / Laya native pooling is an assumption, check the model cards).
  * Laya is fed plain sentences here, not its option-marker format, so it is slightly
    out-of-distribution for it. The 'decision' sentences are in a question style to
    partially cover that.
"""
import os

# Laya's model card warns that transformers probing for TensorFlow can hang model loading.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
MODELS = {
    "gemma": dict(repo="google/embeddinggemma-2", label="EmbeddingGemma 2",
                  native_pool="mean", padding_side="right", loader="gemma"),
    "qwen": dict(repo="Qwen/Qwen3-Embedding-0.6B", label="Qwen3-Embedding-0.6B",
                 native_pool="last", padding_side="left", loader="hf"),
    "laya": dict(repo="convaiinnovations/laya", label="Laya (ModernBERT-large)",
                 native_pool="mean", padding_side="right", loader="laya"),
}
COLORS = {"gemma": "#1a73e8", "qwen": "#e8710a", "laya": "#188038"}

# --------------------------------------------------------------------------------------
# Data: (base, paraphrase, negation/antonym) triplets + decision-style prompts
# --------------------------------------------------------------------------------------
TRIPLETS = [
    ("The cat is sleeping on the sofa.", "A cat is napping on the couch.", "The cat is not sleeping on the sofa."),
    ("The meeting starts at nine tomorrow morning.", "Tomorrow's meeting begins at 9 a.m.", "The meeting does not start at nine tomorrow morning."),
    ("She passed the final exam with a high score.", "She scored well on her final exam and passed.", "She did not pass the final exam."),
    ("The server crashed after the latest deployment.", "Following the most recent release, the server went down.", "The server did not crash after the latest deployment."),
    ("He enjoys running along the river every evening.", "Every evening he likes to jog beside the river.", "He does not enjoy running along the river in the evening."),
    ("The company announced a new product yesterday.", "Yesterday the firm unveiled a new product.", "The company did not announce a new product yesterday."),
    ("This restaurant serves excellent vegetarian food.", "The vegetarian dishes at this restaurant are excellent.", "This restaurant does not serve good vegetarian food."),
    ("The train was delayed by about twenty minutes.", "The train ran roughly twenty minutes late.", "The train was not delayed."),
    ("They are planning to move to a bigger apartment.", "They intend to relocate to a larger flat.", "They are not planning to move to a bigger apartment."),
    ("The doctor recommended rest and plenty of water.", "The physician advised resting and drinking lots of water.", "The doctor did not recommend rest or water."),
    ("Our team won the championship last season.", "Last season our team took the title.", "Our team did not win the championship last season."),
    ("The package arrived earlier than expected.", "The parcel was delivered ahead of schedule.", "The package did not arrive earlier than expected."),
    ("The museum is open to visitors on Sundays.", "Visitors can enter the museum on Sundays.", "The museum is closed to visitors on Sundays."),
    ("The new law will take effect next month.", "Next month the new law comes into force.", "The new law will not take effect next month."),
    ("I forgot my umbrella at the office.", "I left my umbrella behind at work.", "I did not forget my umbrella at the office."),
    ("The software update fixed several security bugs.", "Several security vulnerabilities were patched in the update.", "The software update did not fix any security bugs."),
    ("The children are playing football in the park.", "Some kids are playing soccer at the park.", "The children are not playing football in the park."),
    ("The bank approved her loan application.", "Her application for a loan was approved by the bank.", "The bank rejected her loan application."),
    ("The conference will be held in Berlin this autumn.", "This autumn the conference takes place in Berlin.", "The conference will not be held in Berlin this autumn."),
    ("The soup needs a little more salt.", "A bit more salt should go into the soup.", "The soup does not need any more salt."),
    ("The flight to Mumbai leaves from gate twelve.", "Gate twelve is where the Mumbai flight departs.", "The flight to Mumbai does not leave from gate twelve."),
    ("He has been learning the piano for three years.", "For three years he has studied piano.", "He has not been learning the piano for three years."),
    ("The river flooded the village last spring.", "Last spring the village was flooded by the river.", "The river did not flood the village last spring."),
    ("The library extended its opening hours this week.", "This week the library is open for longer.", "The library did not extend its opening hours this week."),
]

DECISIONS = [
    "Email: 'Please reset my password, I am locked out.' Question: Is this a security-related request?",
    "Ticket: 'I was charged twice for the same order.' Question: Should this ticket be escalated to a human agent?",
    "Message: 'Great service, thank you so much!' Question: Is the sentiment positive?",
    "Log line: 'Disk usage at 97 percent on node-14.' Question: Does this require immediate action?",
    "Request: 'Delete all records older than one year.' Question: Is this action destructive?",
    "Email: 'Your account will be suspended unless you verify your details at this link.' Question: Is this message likely phishing?",
    "Alert: 'Login from a new device in another country.' Question: Should the user be asked for extra verification?",
    "Query: 'How do I update my billing address?' Question: Which team should handle this: billing, technical, or sales?",
    "Invoice: 'Total due 4,250 rupees, payment terms net thirty days.' Question: Is the amount above the approval threshold of 5,000 rupees?",
    "Chat: 'The app crashes every time I open the camera.' Question: Is this a bug report or a feature request?",
    "Comment: 'You people are useless and I hate this product.' Question: Does this message contain abusive language?",
    "Order: 'Ship two units to the Bengaluru warehouse by Friday.' Question: Is the deadline within three days?",
]


def build_dataset():
    sentences, groups = [], []
    for base, para, neg in TRIPLETS:
        sentences += [base, para, neg]
        groups += ["base", "para", "neg"]
    for d in DECISIONS:
        sentences.append(d)
        groups.append("decision")
    return sentences, groups


def pair_indices():
    n = len(TRIPLETS)
    base = np.array([3 * i for i in range(n)])
    return {
        "para": (base, base + 1),
        "neg": (base, base + 2),
        "unrel": (base, np.array([3 * ((i + 11) % n) for i in range(n)])),  # base vs a different base
    }


# --------------------------------------------------------------------------------------
# Stage 1: extraction (needs torch + transformers)
# --------------------------------------------------------------------------------------
def load_encoder_state(model, path):
    """Load the ModernBERT encoder weights out of a Laya checkpoint that also holds a decision head.

    The key prefix of the encoder inside the file is auto-detected from the token-embedding key,
    so it works whether the file was saved as `model.*`, `encoder.*`, `backbone.*` or bare.
    """
    from safetensors import safe_open

    anchor = "embeddings.tok_embeddings.weight"
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        hits = [k for k in keys if k.endswith(anchor)]
        if not hits:
            raise RuntimeError(
                f"no key ending in '{anchor}' in {path}. First keys in the file: " + ", ".join(keys[:30])
            )
        prefix = hits[0][: -len(anchor)]
        sd = {k[len(prefix):]: f.get_tensor(k) for k in keys if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  encoder prefix '{prefix}': loaded {len(sd) - len(unexpected)} tensors, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    bad = [k for k in missing if k.startswith(("layers.", "embeddings."))]
    if bad:
        raise RuntimeError(f"encoder weights missing from checkpoint, e.g. {bad[:5]}")
    return missing, unexpected


def load_laya(repo):
    """Laya ships a custom `laya` model type at the repo root, so AutoModel cannot build it.
    The repo does contain a plain ModernBERT config (encoder/), a tokenizer (tokenizer/) and
    the fine-tuned weights (root model.safetensors, encoder + decision head)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    config = ModernBertConfig.from_pretrained(repo, subfolder="encoder")
    model = ModernBertModel(config)
    load_encoder_state(model, hf_hub_download(repo, "model.safetensors"))
    return model, tok


def load_backbone(cfg, device, dtype):
    from transformers import AutoModel, AutoTokenizer

    repo = cfg["repo"]
    if cfg["loader"] == "laya":
        model, tok = load_laya(repo)
        tok.padding_side = cfg["padding_side"]
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model.to(device=device, dtype=dtype).eval(), tok
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = cfg["padding_side"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    backbone = None
    if cfg["loader"] == "gemma":
        # Prefer the sentence-transformers route: it is the loading path Google documents
        # and it picks the text tower of the multimodal checkpoint.
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(repo, device="cpu")
            first = st[0]
            backbone = getattr(first, "auto_model", None) or getattr(first, "model")
        except Exception as e:  # noqa: BLE001
            print(f"  sentence-transformers load failed ({e!r}); falling back to AutoModel")
    if backbone is None:
        backbone = AutoModel.from_pretrained(repo, trust_remote_code=True)

    backbone = backbone.to(device=device, dtype=dtype).eval()
    return backbone, tok


def leading_run(hs):
    """Split hidden states into (leading entries sharing the first entry's width, the rest).
    EmbeddingGemma 2 (loaded via AutoModel) returned 512-wide states followed by 768-wide ones."""
    d0 = hs[0].shape[-1]
    n = 0
    while n < len(hs) and hs[n].shape[-1] == d0:
        n += 1
    return hs[:n], hs[n:]


def describe_dims(hs):
    runs, last = [], None
    for h in hs:
        d = h.shape[-1]
        if runs and last == d:
            runs[-1][1] += 1
        else:
            runs.append([d, 1])
        last = d
    return " + ".join(f"{d}x{n}" for d, n in runs)


def forward_hidden(backbone, tok, texts, device, max_len):
    import torch

    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.inference_mode():
        out = backbone(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError(
            f"{type(backbone).__name__} did not return hidden_states "
            f"(output type {type(out).__name__}); this model needs a custom loader."
        )
    hs, dropped = leading_run(list(hs))
    if dropped and not getattr(forward_hidden, "warned", False):
        forward_hidden.warned = True
        print(f"  NOTE: model returned hidden states of mixed width {describe_dims(hs + dropped)}; "
              f"keeping the first {len(hs)} (width {hs[0].shape[-1]}) and dropping {len(dropped)} trailing ones. "
              f"If the dropped ones are part of the text stack, send me this line.")
    return torch.stack([h.float() for h in hs]), attn  # [L+1, B, T, D], [B, T]


def summarize_batch(hs, attn, native_pool):
    import torch
    import torch.nn.functional as F

    m = attn.to(hs.dtype)                                   # [B, T]
    denom = m.sum(1).clamp_min(1)                           # [B]
    mean = (hs * m[None, :, :, None]).sum(2) / denom[None, :, None]       # [L+1, B, D]
    if native_pool == "last":
        T = attn.shape[1]
        pos = (attn * torch.arange(T, device=attn.device)).argmax(1)     # last real token
        native = hs[:, torch.arange(hs.shape[1], device=hs.device), pos]  # [L+1, B, D]
    else:
        native = mean
    # Mean pool that skips the first real token. Many LLM-derived models park huge "attention sink"
    # activations on that token in the middle layers, which can swamp a plain mean over tokens.
    first = (attn.cumsum(1) == 1) & attn.bool()                         # first real token [B, T]
    m2 = m * (~first).to(m.dtype)
    nosink = (hs * m2[None, :, :, None]).sum(2) / m2.sum(1).clamp_min(1)[None, :, None]
    # Mean pool that drops the single highest-norm token at EACH layer (per sentence). Model-agnostic way to
    # remove a "massive activation" token wherever it sits (first token, end token, punctuation, ...).
    norms = hs.norm(dim=-1).masked_fill(attn[None] == 0, -1.0)                      # [L+1, B, T]
    top = F.one_hot(norms.argmax(2), num_classes=hs.shape[2]).to(hs.dtype)         # [L+1, B, T]
    w = m[None] * (1.0 - top)
    nomax = (hs * w[..., None]).sum(2) / w.sum(2).clamp_min(1)[..., None]           # [L+1, B, D]
    a, b = hs[1:], hs[:-1]
    step_cos = (F.cosine_similarity(a, b, dim=-1) * m[None]).sum(2) / denom[None]          # [L, B]
    step_rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6) * m[None]).sum(2) / denom[None]
    tok_norm = (hs.norm(dim=-1) * m[None]).sum(2) / denom[None]                              # [L+1, B]
    return [x.cpu().numpy() for x in (mean, native, step_cos, step_rel, tok_norm, nosink, nomax)]


def extract(keys, out_dir, batch_size, max_len, dtype_name, device):
    import gc

    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, dtype_name)
    sentences, _ = build_dataset()
    acts_dir = Path(out_dir) / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} dtype={dtype_name} sentences={len(sentences)}")

    for key in keys:
        cfg = MODELS[key]
        print(f"\n== {key}: {cfg['repo']}")
        backbone = tok = None
        try:
            backbone, tok = load_backbone(cfg, device, dtype)
            chunks = [[] for _ in range(7)]
            for start in range(0, len(sentences), batch_size):
                hs, attn = forward_hidden(backbone, tok, sentences[start:start + batch_size], device, max_len)
                if start == 0:
                    print(f"  hidden states: {hs.shape[0]} (embeddings + {hs.shape[0] - 1} layers), dim {hs.shape[-1]}")
                for c, x in zip(chunks, summarize_batch(hs, attn, cfg["native_pool"])):
                    c.append(x)
            mean, native, step_cos, step_rel, tok_norm, nosink, nomax = [np.concatenate(c, axis=1) for c in chunks]
            np.savez_compressed(
                acts_dir / f"{key}.npz", mean=mean, native=native, nosink=nosink, nomax=nomax, step_cos=step_cos,
                step_rel=step_rel, tok_norm=tok_norm, label=cfg["label"], native_pool=cfg["native_pool"],
            )
            print(f"  saved {acts_dir / (key + '.npz')}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED for {key}: {e!r}")
            if key == "laya":
                print("  hint: Laya is loaded as a plain ModernBERT encoder from encoder/ + tokenizer/ + the root "
                      "model.safetensors; send me the message above if the key names did not match.")
            if key == "gemma":
                print("  hint: the sentence-transformers route needs the model's processor class; try "
                      "`pip install -U transformers sentence-transformers pillow torchvision librosa` "
                      "(the AutoModel fallback is used otherwise).")
        finally:
            backbone = tok = None  # drop references so the next model gets the VRAM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Stage 2: analysis (numpy + matplotlib only)
# --------------------------------------------------------------------------------------
def load_acts(out_dir):
    acts = {}
    for key in MODELS:
        p = Path(out_dir) / "acts" / f"{key}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=False)
            acts[key] = {k: z[k] for k in ("mean", "native", "nosink", "nomax", "step_cos", "step_rel", "tok_norm") if k in z.files}
            acts[key]["label"] = str(z["label"])
    return acts


def _unit(Z):
    return Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9, None)


def layer_metrics(X, final, pairs):
    """X: [N, D] pooled vectors at one layer; final: [N, D] at the last layer."""
    X = np.asarray(X, dtype=np.float64)       # float64: float32 squares of huge activations can overflow
    final = np.asarray(final, dtype=np.float64)
    if not np.isfinite(X).all() or np.ptp(X, axis=0).max() == 0:  # degenerate layer: report NaN, not fake zeros
        names = [n + s for n in pairs for s in ("_raw", "_c")] + ["aniso", "pr", "cos_final", "cos_final_c"]
        return {n: float("nan") for n in names}
    out = {}
    for center in (False, True):
        Z = X - X.mean(0) if center else X
        Zn = _unit(Z)
        suffix = "_c" if center else "_raw"
        for name, (i, j) in pairs.items():
            out[name + suffix] = float((Zn[i] * Zn[j]).sum(1).mean())
    Xn = _unit(X)
    sim = Xn @ Xn.T
    n = len(X)
    out["aniso"] = float((sim.sum() - n) / (n * (n - 1)))
    if np.isfinite(X).all():
        lam = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
        out["pr"] = float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else float("nan")
    else:
        out["pr"] = float("nan")
    out["cos_final"] = float((Xn * _unit(final)).sum(1).mean())
    out["cos_final_c"] = float((_unit(X - X.mean(0)) * _unit(final - final.mean(0))).sum(1).mean())
    return out


def health_report(acts):
    """Flag layers whose pooled vectors are non-finite or have zero variance across sentences."""
    problems = []
    for k, a in acts.items():
        for pool in ("mean", "nosink", "nomax", "native"):
            if pool not in a:
                continue
            for l, X in enumerate(a[pool]):
                if not np.isfinite(X).all():
                    problems.append(f"{k}/{pool}: layer {l} has non-finite values")
                elif np.ptp(X, axis=0).max() == 0:
                    problems.append(f"{k}/{pool}: layer {l} is identical for every sentence (zero variance)")
    if problems:
        print("\nDATA HEALTH WARNINGS (metrics for these layers are NaN):")
        for p in problems[:40]:
            print("  -", p)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
    else:
        print("\nData health: all pooled vectors finite and non-degenerate.")


def pair_cos_arrays(X, pairs):
    """Per-pair mean-centered cosines {name: array[n_pairs]}; None for a degenerate layer."""
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all() or np.ptp(X, axis=0).max() == 0:
        return None
    Zn = _unit(X - X.mean(0))
    return {name: (Zn[i] * Zn[j]).sum(1) for name, (i, j) in pairs.items()}


def centered_gram(X):
    Xc = X - X.mean(0)
    return Xc @ Xc.T


def cka(K, L):
    return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum() + 1e-12))


def standardize_pooled(X):
    """Z-score every dimension across sentences, independently at every layer.
    X: [L+1, N, D]. Dimensions with (numerically) zero variance become 0."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    smax = sd.max(2, keepdims=True)
    keep = sd > 1e-6 * np.maximum(smax, 1e-30)
    return np.where(keep, (X - mu) / np.where(keep, sd, 1.0), 0.0)


def analyze(out_dir, standardize=False, cka_pool="auto"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    acts = load_acts(out_dir)
    if not acts:
        print(f"No activations found in {out_dir / 'acts'}. Run `extract` first.")
        return
    sentences, _ = build_dataset()
    pairs = pair_indices()
    keys = list(acts)
    for k in keys:
        assert acts[k]["mean"].shape[1] == len(sentences), f"{k}: sentence count mismatch, re-run extract"
    pools = [p for p in ("mean", "nosink", "nomax", "native") if all(p in acts[k] for k in keys)]
    for p_, what in (("nosink", "first-token-free"), ("nomax", "max-norm-token-free")):
        if p_ not in pools:
            print(f"NOTE: no {what} pooling ('{p_}') in the saved activations; re-run `extract` to add it.")
    auto_pool = "nomax" if "nomax" in pools else ("nosink" if "nosink" in pools else "mean")
    if cka_pool == "auto":
        cka_pool = auto_pool
    elif cka_pool not in pools:
        print(f"NOTE: CKA pooling '{cka_pool}' not available; using '{auto_pool}'.")
        cka_pool = auto_pool
    tag = lambda fig: fig.text(0.995, 0.005, "STANDARDIZED" if standardize else "raw", ha="right", va="bottom",
                               fontsize=8, color="gray")
    health_report(acts)
    if standardize:
        # Residual streams of LLM-derived models carry a few huge-magnitude dimensions that can dominate
        # cosine / dimensionality / CKA. Z-scoring each dimension across sentences removes that dominance.
        for k in keys:
            for pool in ("mean", "nosink", "nomax", "native"):
                if pool in acts[k]:
                    acts[k][pool] = standardize_pooled(acts[k][pool])
        out_dir = out_dir / "standardized"
        print(f"\nStandardized mode: every dimension z-scored across sentences per layer. Output: {out_dir}")

    rows, S = [], {}  # S[(key, pool, metric)] -> array over layers
    depth = {}
    for k in keys:
        a = acts[k]
        Lp1 = a["mean"].shape[0]
        depth[k] = np.arange(Lp1) / (Lp1 - 1)
        for pool in pools:
            X = a[pool]
            per_layer = [layer_metrics(X[l], X[-1], pairs) for l in range(Lp1)]
            for m in per_layer[0]:
                vals = np.array([d[m] for d in per_layer])
                S[(k, pool, m)] = vals
                rows += [dict(model=k, pool=pool, layer=l, depth=round(float(depth[k][l]), 4), metric=m, value=v)
                         for l, v in enumerate(vals)]
        for m, arr in (("step_cos", a["step_cos"]), ("step_rel", a["step_rel"]), ("tok_norm", a["tok_norm"])):
            vals = arr.mean(1)
            S[(k, "token", m)] = vals
            off = 0 if m == "tok_norm" else 1  # step metrics start at layer 1
            rows += [dict(model=k, pool="token", layer=l + off, depth=round(float(depth[k][l + off]), 4),
                          metric=m, value=float(v)) for l, v in enumerate(vals)]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "pool", "layer", "depth", "metric", "value"])
        w.writeheader()
        w.writerows(rows)

    # Paired bootstrap (over the triplets) of the paraphrase-minus-negation gap, per layer
    rng = np.random.default_rng(0)
    n_trip = len(TRIPLETS)
    boot = rng.integers(0, n_trip, size=(2000, n_trip))
    GAP, gap_rows = {}, []
    for k in keys:
        for pool in pools:
            Xp = acts[k][pool]
            g = np.full(Xp.shape[0], np.nan)
            lo, hi = g.copy(), g.copy()
            for l in range(Xp.shape[0]):
                d = pair_cos_arrays(Xp[l], pairs)
                if d is None:
                    continue
                diff = d["para"] - d["neg"]
                bm = diff[boot].mean(1)
                g[l], lo[l], hi[l] = diff.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)
            GAP[(k, pool)] = (g, lo, hi)
            gap_rows += [dict(model=k, pool=pool, layer=l, depth=round(float(depth[k][l]), 4),
                              gap=g[l], ci_lo=lo[l], ci_hi=hi[l]) for l in range(len(g))]
    with open(out_dir / "para_minus_neg_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "pool", "layer", "depth", "gap", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(gap_rows)

    # Fig 1: how much each layer rewrites the token vectors
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k in keys:
        L = len(depth[k]) - 1
        d_step = np.arange(1, L + 1) / L
        ax[0].plot(d_step, S[(k, "token", "step_cos")], label=acts[k]["label"], color=COLORS[k])
        ax[1].plot(d_step, S[(k, "token", "step_rel")], color=COLORS[k])
        ax[2].plot(depth[k], S[(k, "token", "tok_norm")], color=COLORS[k])
    ax[0].set(title="Direction change per layer (cos to previous layer)", xlabel="relative depth", ylabel="mean token cosine")
    ax[1].set(title="Size of update per layer (||Δ|| / ||h_prev||)", xlabel="relative depth", yscale="log")
    ax[2].set(title="Mean token norm", xlabel="relative depth", yscale="log")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_layer_steps.png", dpi=150)
    plt.close(fig)

    # Fig 2: where paraphrase / negation / unrelated separate (mean-centered cosine)
    fig, ax = plt.subplots(len(pools), len(keys), figsize=(5.2 * len(keys), 3.5 * len(pools)), squeeze=False)
    for r, pool in enumerate(pools):
        for c, k in enumerate(keys):
            for name, style in (("para", "-"), ("neg", "--"), ("unrel", ":")):
                ax[r, c].plot(depth[k], S[(k, pool, name + "_c")], style, label=name, color=COLORS[k])
            ax[r, c].set(title=f"{acts[k]['label']} [{pool} pooling]", xlabel="relative depth", ylabel="centered cosine")
            ax[r, c].axhline(0, color="gray", lw=0.5)
            if r == 0 and c == 0:
                ax[r, c].legend()
    tag(fig)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_separation.png", dpi=150)
    plt.close(fig)

    # Fig 2b: paraphrase minus negation, with 95% bootstrap CI over the 24 triplets
    fig, ax = plt.subplots(1, len(pools), figsize=(4.8 * len(pools), 4.2), squeeze=False, sharey=True)
    for c, pool in enumerate(pools):
        for k in keys:
            g, lo, hi = GAP[(k, pool)]
            ax[0, c].plot(depth[k], g, color=COLORS[k], label=acts[k]["label"])
            ax[0, c].fill_between(depth[k], lo, hi, color=COLORS[k], alpha=0.18, linewidth=0)
        ax[0, c].axhline(0, color="gray", lw=0.6)
        ax[0, c].set(title=f"{pool} pooling", xlabel="relative depth")
    ax[0, 0].set_ylabel("paraphrase − negation (centered cos)\nband = 95% bootstrap CI, 24 triplets")
    ax[0, 0].legend(fontsize=8)
    tag(fig)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_para_minus_neg.png", dpi=150)
    plt.close(fig)

    # Fig 3: geometry
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    styles = {"mean": "-", "nosink": "-.", "nomax": ":", "native": "--"}
    for k in keys:
        for pool in pools:
            lab = acts[k]["label"] if pool == pools[0] else None
            ax[0].plot(depth[k], S[(k, pool, "aniso")], styles[pool], color=COLORS[k], label=lab)
            ax[1].plot(depth[k], S[(k, pool, "pr")], styles[pool], color=COLORS[k])
            ax[2].plot(depth[k], S[(k, pool, "cos_final_c")], styles[pool], color=COLORS[k])
    ax[0].set(title="Anisotropy (mean pairwise cosine, all sentences)", xlabel="relative depth")
    ax[1].set(title="Effective dimensionality (participation ratio)", xlabel="relative depth")
    ax[2].set(title="Cosine to final-layer embedding (mean-centered)", xlabel="relative depth")
    ax[0].legend(title="solid = mean, dash-dot = no first token, dotted = no max-norm token, dashed = native",
                 fontsize=7, title_fontsize=7)
    tag(fig)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_geometry.png", dpi=150)
    plt.close(fig)

    # Fig 4: cross-model linear CKA (sink-free mean pooling when available, else plain mean)
    grams = {k: [centered_gram(np.asarray(acts[k][cka_pool][l], dtype=np.float64))
                 for l in range(acts[k][cka_pool].shape[0])] for k in keys}
    combos = list(itertools.combinations(keys, 2))
    if combos:
        fig, ax = plt.subplots(1, len(combos), figsize=(5.4 * len(combos), 4.8), squeeze=False)
        best_rows = []
        for c, (ka, kb) in enumerate(combos):
            M = np.array([[cka(Ka, Kb) for Kb in grams[kb]] for Ka in grams[ka]])
            im = ax[0, c].imshow(M, origin="lower", aspect="auto", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap="viridis")
            ax[0, c].set(xlabel=f"{acts[kb]['label']} (rel. depth)", ylabel=f"{acts[ka]['label']} (rel. depth)",
                         title=f"Linear CKA ({cka_pool} pooling)")
            fig.colorbar(im, ax=ax[0, c])
            for la in range(M.shape[0]):
                lb = int(M[la].argmax())
                best_rows.append(dict(model_a=ka, layer_a=la, depth_a=round(float(depth[ka][la]), 3),
                                      model_b=kb, best_layer_b=lb, depth_b=round(float(depth[kb][lb]), 3),
                                      cka=round(float(M[la, lb]), 4)))
        tag(fig)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_cka.png", dpi=150)
        plt.close(fig)
        with open(out_dir / "cka_best_match.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(best_rows[0]))
            w.writeheader()
            w.writerows(best_rows)

    # Console summary
    print(f"\nResults in {out_dir}/  (metrics.csv, para_minus_neg_ci.csv, cka_best_match.csv, fig_layer_steps.png, "
          f"fig_separation.png, fig_para_minus_neg.png, fig_geometry.png, fig_cka.png)")
    print(f"\nFinal layer, paraphrase vs negation (centered cosine; 95% bootstrap CI over {n_trip} triplets; "
          f"* = CI excludes 0)")
    for pool in pools:
        print(f"[{pool}]")
        for k in keys:
            g, lo, hi = GAP[(k, pool)]
            p_, n_ = S[(k, pool, "para_c")][-1], S[(k, pool, "neg_c")][-1]
            star = "*" if (lo[-1] > 0 or hi[-1] < 0) else " "
            print(f"  {acts[k]['label']:26s} para {p_:6.3f}  neg {n_:6.3f}  gap {g[-1]:+.3f} [{lo[-1]:+.3f}, {hi[-1]:+.3f}] {star}")
    for pool in pools:
        print(f"\n[{pool} pooling]  (cos_final is mean-centered; gap = centered paraphrase cos minus unrelated cos)")
        print(f"{'model':26s} {'layers':>6s} {'biggest update @':>17s} {'best gap @':>16s} {'max gap':>8s} {'cos_final @50%':>15s}")
        for k in keys:
            L = len(depth[k]) - 1
            big = int(np.argmax(S[(k, "token", "step_rel")])) + 1
            gap = np.nan_to_num(S[(k, pool, "para_c")] - S[(k, pool, "unrel_c")], nan=-9)
            g = int(np.argmax(gap))
            half = S[(k, pool, "cos_final_c")][int(round(0.5 * L))]
            print(f"{acts[k]['label']:26s} {L:6d} {f'layer {big} ({big / L:.0%})':>17s} "
                  f"{f'layer {g} ({g / L:.0%})':>16s} {gap[g]:8.3f} {half:15.3f}")


# --------------------------------------------------------------------------------------
# Self-test: synthetic activations, exercises the whole analysis path
# --------------------------------------------------------------------------------------
def selftest(out_dir):
    rng = np.random.default_rng(0)
    sentences, _ = build_dataset()
    N = len(sentences)
    d = Path(out_dir) / "selftest"
    (d / "acts").mkdir(parents=True, exist_ok=True)
    for key, (L, D) in {"gemma": (18, 768), "qwen": (28, 1024), "laya": (28, 1024)}.items():
        base = rng.normal(size=(N, D))
        layers = [base]
        for _ in range(L):
            layers.append(layers[-1] + 0.4 * rng.normal(size=(N, D)))
        X = np.stack(layers).astype(np.float32)
        np.savez_compressed(
            d / "acts" / f"{key}.npz", mean=X, native=X, nosink=X + 0.01, nomax=X + 0.02,
            step_cos=rng.uniform(0.5, 1, size=(L, N)),
            step_rel=rng.uniform(0.01, 1, size=(L, N)), tok_norm=rng.uniform(1, 100, size=(L + 1, N)),
            label=MODELS[key]["label"], native_pool="mean",
        )
    analyze(d)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["extract", "analyze", "all", "selftest"])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--out", default="probe_out")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto")
    p.add_argument("--standardize", action="store_true",
                   help="analyze: z-score each dimension across sentences per layer (writes to <out>/standardized)")
    p.add_argument("--cka-pool", default="auto", choices=["auto", "mean", "nosink", "nomax", "native"],
                   help="analyze: pooling used for the CKA heatmaps (auto = nomax, else nosink, else mean)")
    a = p.parse_args()

    if a.command in ("extract", "all"):
        extract(a.models, a.out, a.batch_size, a.max_len, a.dtype, a.device)
    if a.command in ("analyze", "all"):
        analyze(a.out, a.standardize, a.cka_pool)
    if a.command == "selftest":
        selftest(a.out)


if __name__ == "__main__":
    main()