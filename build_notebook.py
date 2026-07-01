"""Builds chronos2_probing.ipynb — a single self-contained notebook that reproduces
the entire layer-wise probing experiment (the four scripts consolidated).

Run: python build_notebook.py  ->  writes chronos2_probing.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===================================================================== #
md(r"""
# Layer-wise Probing of a Frozen Chronos-2 Encoder

**A single, runnable source for the whole experiment.** This notebook consolidates the
four development scripts (`extract_check.py`, `probe_pipeline.py`, `probe_improve.py`,
`probe_harden.py`) into one shareable artifact. Run it top to bottom and it reproduces every
figure and table. It is the executable companion to `probing_walkthrough.md`.

## The question

We test a two-part hypothesis — the **"tunnel effect"** applied to a time-series foundation
model. Keep the two parts separate; that distinction is the backbone of everything below.

- **Part 1.** For a frozen Chronos-2, the **middle** encoder layers carry more
  transfer-relevant information than the **last** layer, where "transfer-relevant" = how well a
  *linear* classifier reads a downstream label out of that layer's representation.
- **Part 2.** That middle-over-last advantage is **larger under distribution shift (OOD)** than
  in-distribution (ID).

**Why Chronos-2 specifically?** It is a *forecasting* model. Its final layer is specialized to
hand off to the forecasting decoder — a representation optimized for predicting future values,
not class discrimination. So the prediction is mechanistically natural: the last layer should
shed general discriminative structure that the middle layers still hold.

**Chronos-2 facts used throughout:** encoder-only, T5-style, ~120M params, **12 encoder layers**,
hidden size **d_model = 768**, patch-based (**patch_size = 16**). The model is **frozen** — never
trained or fine-tuned; we only fit tiny linear classifiers *on top of* its fixed representations.

> **Concept — linear probing.** Freeze the backbone. For each layer, take its fixed
> representation for each input and train one small **linear** classifier (logistic regression)
> to predict the label. The probe's test accuracy proxies "how linearly decodable is the task
> from this layer." A *linear* probe (not an MLP) is deliberate: it measures information that is
> *directly* available, so layer differences reflect the representations, not the probe's
> capacity to dig.

## How to run

1. Create the environment (see the next cell for the exact package list).
2. Run all cells. Features are cached to `./features_cache/`, so a second run — and any
   collaborator who receives the cache — is near-instant. The only real cost is the first
   extraction of each (dataset, split, corruption).
3. Everything stays **float32**, on **MPS** (Apple Silicon) / **CUDA** if present / else **CPU**,
   under `torch.no_grad()`, with fixed seeds.
""")

# ===================================================================== #
md(r"""
## 0 · Setup

Environment (create once, outside the notebook):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "numpy<2" "numba==0.60.0" "llvmlite==0.43.0"   # arm64-macOS wheels for aeon
uv pip install torch "chronos-forecasting>=2.1.0" aeon scikit-learn matplotlib "pandas[pyarrow]"
uv pip install nbformat nbconvert ipykernel                    # to run this notebook
```

The cell below defines all global constants and selects the device. Fixed seeds make every
result reproducible; each heavy analysis cell also resets its own bootstrap RNG so results do
not depend on cell execution order.
""")

code(r"""
import math, json, warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")  # silence aeon's load_classification deprecation noise

# ---- reproducibility & global config ----
SEED        = 0
NUM_LAYERS  = 12
MIDDLE_BAND = list(range(3, 9))    # a-priori "middle" = layers 3..8 inclusive (NOT data-selected)
LAST_LAYER  = NUM_LAYERS - 1       # = 11
BOOT_B      = 2000                 # bootstrap resamples
CACHE_DIR   = Path("./features_cache"); CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR     = Path(".")

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- device: CUDA -> MPS -> CPU (project target is MPS/CPU; CUDA added for collaborators) ----
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"torch {torch.__version__} | device = {DEVICE}")
""")

# ===================================================================== #
md(r"""
## 1 · Stage 1 — Can we extract *every* layer? (forward hooks)

The make-or-break engineering question. Chronos-2's public API `Chronos2Pipeline.embed()`
returns the **last encoder layer only** — useless for a per-layer study. The internal model also
does **not** support `output_hidden_states=True`. So forward hooks are the right (and only) path.

> **Concept — forward hook.** PyTorch lets you register a function on any module that fires every
> time the module produces output, capturing it without changing the model. We attach one hook to
> each of the 12 transformer blocks (`encoder.block[i]`); a single forward pass fires all 12 and
> we harvest all 12 layers' outputs at once.

**Engineering details that mattered:** force **float32** (Chronos can default to bfloat16, which
misbehaves on MPS); run under `eval()` + `no_grad()`; never touch the weights.
""")

code(r"""
from chronos import Chronos2Pipeline

_PIPELINE = None

def get_pipeline():
    '''Load amazon/chronos-2 once: float32, on DEVICE, eval, gradients off (frozen).'''
    global _PIPELINE
    if _PIPELINE is None:
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", torch_dtype=torch.float32)  # frozen, fp32
        try:
            pipe.model.to(DEVICE)
        except Exception as e:                      # e.g. an MPS op gap -> fall back to CPU
            warnings.warn(f"to({DEVICE}) failed: {e}; using CPU")
            pipe.model.to("cpu")
        pipe.model.eval()
        for p in pipe.model.parameters():
            p.requires_grad_(False)
        _PIPELINE = pipe
    return _PIPELINE

pipe = get_pipeline()
cfg  = pipe.model.config
PATCH_SIZE = pipe.model.chronos_config.input_patch_size
print(f"params on {next(pipe.model.parameters()).device}")
print(f"d_model={cfg.d_model}  num_layers={cfg.num_layers}  num_heads={cfg.num_heads}  patch_size={PATCH_SIZE}")
print(f"encoder layer stack: encoder.block = {type(pipe.model.encoder.block).__name__}"
      f" of {len(pipe.model.encoder.block)} x {type(pipe.model.encoder.block[0]).__name__}")
""")

md(r"""
### 1.1 · Sanity check — 12 hooks → 12 layers, with the shape arithmetic

For an input series of length `L`, each layer's hidden state is `(batch, P, 768)` with

$$P = \lceil L / \texttt{patch\_size} \rceil + 2.$$

The `+2` is two special positions: a **[REG] register token** (a learned summary slot, like CLS)
and a **masked-future patch** (the slot the model would forecast into). Confirming this
arithmetic is how we know the hooks capture real per-layer states. The layout (verified from
`Chronos2Model.encode`) is `[content_patches…, REG, masked_future]` — so the REG token sits at
index `num_context_patches`, **not** index 0.
""")

code(r"""
# Minimal one-sample demonstration (mirrors extract_check.py).
from aeon.datasets import load_classification

Xdemo, ydemo = load_classification("BasicMotions", split="train")
series = torch.tensor(np.asarray(Xdemo)[0, 0], dtype=torch.float32)   # channel 0 of case 0, length L
L = series.shape[0]
P_expected = math.ceil(L / PATCH_SIZE) + 2

captured = {i: [] for i in range(NUM_LAYERS)}
def _demo_hook(i):
    def hook(_m, _inp, out):
        hs = out.hidden_states if hasattr(out, "hidden_states") and out.hidden_states is not None else out[0]
        captured[i].append(hs.detach().to("cpu"))
    return hook

handles = [blk.register_forward_hook(_demo_hook(i)) for i, blk in enumerate(pipe.model.encoder.block)]
try:
    with torch.no_grad():
        _ = pipe.embed([series])     # embed runs the FULL encoder; we harvest from hooks, ignore its return
finally:
    for h in handles: h.remove()

shapes = [tuple(torch.cat(captured[i], 0).shape) for i in range(NUM_LAYERS)]
print(f"series length L={L}  ->  P_expected = ceil({L}/{PATCH_SIZE})+2 = {P_expected}")
print(f"layers captured: {len(shapes)}   per-layer shape: {shapes[0]}")
assert len(shapes) == NUM_LAYERS and shapes[0][1] == P_expected
print("OK — 12 hooks fired, P arithmetic matches. Extraction is mechanically sound.")
""")

# ===================================================================== #
md(r"""
## 2 · The reusable feature extractor (with caching)

This is the workhorse used by every experiment. For a `(dataset, split, corruption, pooling)`
request it returns `{layer_idx: features (n, c*768)}` and labels `y`, and caches to disk.

**Design decisions (each one you may be asked to defend):**

- **Per-channel, then concatenate.** UEA datasets are multivariate. We run **each channel through
  the encoder separately** and concatenate the per-channel 768-vectors (a 3-channel dataset →
  `3×768 = 2304` features/layer). This is faithful to Head2Toe's "use all representation," reuses
  the validated univariate path, and keeps dimensionality manageable. (Native group-ID
  multivariate encoding is a distinct future experiment.)
- **Pooling = collapse `(P, 768)` → `(768,)`.** Three variants computed in one pass and cached
  separately: `content` (mean over content patches — the default), `all` (mean over all `P`),
  `reg` (the REG token only). We drop the masked-future patch from `content` because that slot is
  the model in *forecasting* mode and would leak prediction-mode signal into a classification probe.

  > Because every layer collapses to a 768-d vector for any series length and dataset,
  > **cross-dataset comparison is mechanically free** — all datasets land on the same footing.
""")

code(r"""
# ---------- input-space corruptions (label-preserving; applied to TEST inputs only) ----------
def _gaussian(X, alpha, seed):
    '''Per-series, per-channel additive Gaussian noise, std = alpha * std(series, channel).'''
    rng = np.random.default_rng(seed)
    std = X.std(axis=-1, keepdims=True)
    return (X + rng.standard_normal(X.shape).astype(np.float32) * (alpha * std).astype(np.float32)).astype(np.float32)

def _timewarp(X, factor):
    '''Resample each series by `factor` via linear interpolation, then crop/pad back to length t.'''
    n, c, t = X.shape
    new_len = int(round(t * factor)); old = np.arange(t, dtype=np.float64); new = np.linspace(0, t-1, new_len)
    out = np.empty_like(X)
    for i in range(n):
        for ch in range(c):
            w = np.interp(new, old, X[i, ch])
            if   new_len == t: out[i, ch] = w
            elif new_len >  t: s = (new_len - t)//2; out[i, ch] = w[s:s+t]
            else:              pl = (t-new_len)//2;  out[i, ch] = np.pad(w, (pl, t-new_len-pl), mode="edge")
    return out.astype(np.float32)

def _drift(X, amplitude):
    '''Add a half-cycle sinusoid (low-frequency baseline drift), scaled by per-series-channel std.'''
    n, c, t = X.shape
    std = X.std(axis=-1, keepdims=True)
    phase = np.sin(np.linspace(0, np.pi, t)).astype(np.float32)
    return (X + (amplitude * std).astype(np.float32) * phase[None, None, :]).astype(np.float32)

def apply_corruption(X, corr):
    if corr is None:                return X
    if corr["kind"] == "gauss":     return _gaussian(X, corr["alpha"], corr["seed"])
    if corr["kind"] == "timewarp":  return _timewarp(X, corr["factor"])
    if corr["kind"] == "drift":     return _drift(X, corr["amplitude"])
    raise ValueError(corr["kind"])

def _corr_repr(corr):
    return "clean" if corr is None else ",".join(f"{k}={corr[k]}" for k in sorted(corr))

def _cache_path(ds, split, corr, pool):
    return CACHE_DIR / f"{ds}__{split}__{_corr_repr(corr)}__{pool}.npz"
""")

code(r"""
def extract_features(dataset_name, split, corruption=None, batch_size=64, pooling="content"):
    '''Hook-based per-layer extraction with on-disk caching.

    Returns ({layer_idx: (n, c*768) float32}, y). Prints whether it loaded or extracted.
    All three pooling variants are written on a cache miss, so later pooling requests are free.
    '''
    cache_path = _cache_path(dataset_name, split, corruption, pooling)
    if cache_path.exists():
        print(f"  [loaded ] {cache_path.name}")
        d = np.load(cache_path, allow_pickle=True)
        feats = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("layer_")}
        return feats, d["y"]

    print(f"  [extract] {dataset_name}/{split}  corruption={_corr_repr(corruption)}  pooling(all 3)")
    from aeon.datasets import load_classification
    X, y = load_classification(dataset_name, split=split)
    X = apply_corruption(np.asarray(X, dtype=np.float32), corruption)
    n, c, t = X.shape

    pipe = get_pipeline()
    num_context = math.ceil(t / PATCH_SIZE)
    P_expected  = num_context + 2

    per = {p: {ch: {} for ch in range(c)} for p in ("content", "all", "reg")}
    for ch in range(c):
        inputs = [torch.from_numpy(np.ascontiguousarray(X[i, ch])).to(torch.float32) for i in range(n)]
        cap = {i: [] for i in range(NUM_LAYERS)}
        def mk(i):
            def hook(_m, _inp, out):
                hs = out.hidden_states if hasattr(out, "hidden_states") and out.hidden_states is not None else out[0]
                cap[i].append(hs.detach().to("cpu"))   # off-device immediately to keep MPS memory low
            return hook
        handles = [blk.register_forward_hook(mk(i)) for i, blk in enumerate(pipe.model.encoder.block)]
        try:
            with torch.no_grad():
                _ = pipe.embed(inputs, batch_size=batch_size)   # robust to batched OR per-series forward
        finally:
            for h in handles: h.remove()
        for i in range(NUM_LAYERS):
            full = torch.cat(cap[i], 0)                  # (n, P, 768)
            assert full.shape[0] == n, f"hook batch {full.shape[0]} != n={n}"
            if ch == 0 and i == 0:
                assert full.shape[1] == P_expected, f"P {full.shape[1]} != expected {P_expected}"
            per["content"][ch][i] = full[:, :num_context, :].mean(1).numpy()
            per["all"][ch][i]     = full.mean(1).numpy()
            per["reg"][ch][i]     = full[:, num_context, :].numpy()

    for pool in ("content", "all", "reg"):
        feats = {i: np.concatenate([per[pool][ch][i] for ch in range(c)], axis=1) for i in range(NUM_LAYERS)}
        np.savez(_cache_path(dataset_name, split, corruption, pool),
                 y=y, **{f"layer_{i}": feats[i] for i in range(NUM_LAYERS)})
    feats = {i: np.concatenate([per[pooling][ch][i] for ch in range(c)], axis=1) for i in range(NUM_LAYERS)}
    return feats, y
""")

# ===================================================================== #
md(r"""
## 3 · Probe + statistics machinery

> **Concept — probe hygiene (correctness, not tuning).** `StandardScaler` fit on **train only**
> then applied to test (linear probes on transformer features need standardization; train-only
> fitting prevents leaking test statistics). `LogisticRegression(max_iter=2000)` so it converges
> on high-dim features. Canonical UEA splits, fixed seeds. The only knob touched is `max_iter`.

> **Concept — bootstrap CI.** A single accuracy hides how much it would wobble on a different
> test draw. Resample the test set **with replacement** B=2000 times, recompute accuracy each
> time, take the 2.5/97.5 percentiles. Captures test-set sampling variance.

> **Concept — paired bootstrap (for differences).** To ask "is layer A better than B?" resample
> the **same** test indices and apply to both layers' per-sample correctness vectors, then look at
> `acc_A − acc_B`. If that interval excludes 0, the difference is real. **Subtlety:** two
> accuracies can have *overlapping* single-layer CIs while their *paired difference* excludes 0 —
> because the layers succeed/fail on largely the same samples. The paired test, not "do error bars
> overlap," is the right tool.
""")

code(r"""
def fit_layerwise_probes(f_train, y_train):
    '''One StandardScaler + LogisticRegression(max_iter=2000) per layer.'''
    probes = {}
    for i in range(NUM_LAYERS):
        scaler = StandardScaler().fit(f_train[i])
        clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(scaler.transform(f_train[i]), y_train)
        probes[i] = {"scaler": scaler, "clf": clf}
    return probes

def score_correctness(probes, features, y_true):
    '''Per-layer per-sample correctness (float32 0/1) — the substrate for all bootstraps.'''
    y_true = np.asarray(y_true)
    return {i: (probes[i]["clf"].predict(probes[i]["scaler"].transform(features[i])) == y_true).astype(np.float32)
            for i in range(NUM_LAYERS)}

def bootstrap_ci(correct, B=BOOT_B, rng=None):
    '''(point, lo, hi) for an accuracy, from its correctness vector. No refit.'''
    rng = rng or np.random.default_rng(SEED)
    c = np.asarray(correct, np.float64); n = c.size
    means = c[rng.integers(0, n, size=(B, n))].mean(1)
    return float(c.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def paired_diff_ci(a, b, B=BOOT_B, rng=None):
    '''(point, lo, hi) for acc_a - acc_b, SAME resampled indices applied to both.'''
    rng = rng or np.random.default_rng(SEED)
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); n = a.size
    idx = rng.integers(0, n, size=(B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def excl0(lo, hi):   return (lo > 0) or (hi < 0)
""")

md(r"""
### 3.1 · The pre-registered metrics

> **Pre-registration discipline.** The "middle" region is fixed **a priori** as layers **3–8**.
> We report the late-layer drop two ways:
> `late_drop_band = mean(acc over L3..L8) − acc(L11)` (the principled **headline**), and
> `late_drop_argmax = best non-final layer − acc(L11)` (**selection-biased** — you picked the best
> layer then tested it on the same data — so reported but labeled). Declaring the metric before
> looking avoids the circular "find the best layer, then prove it's best."

For **Part 2**, the quantity that *is* the hypothesis is the **amplification**:
`(mid − last)|OOD − (mid − last)|ID`. (An easy mistake — which the first automated run made — is
to instead measure the vertical ID−OOD gap per layer; that's a cousin, not the hypothesis.)
""")

code(r"""
def band_correct(correct):
    '''Per-sample mean correctness across the a-priori middle band (L3..L8).'''
    return np.stack([correct[L] for L in MIDDLE_BAND], 0).mean(0)

def late_drop_band_ci(correct, rng):
    return paired_diff_ci(band_correct(correct), correct[LAST_LAYER], rng=rng)

def late_drop_argmax_ci(correct, rng):
    accs = np.array([correct[L].mean() for L in range(NUM_LAYERS - 1)])
    Lstar = int(np.argmax(accs))
    pt, lo, hi = paired_diff_ci(correct[Lstar], correct[LAST_LAYER], rng=rng)
    return pt, lo, hi, Lstar

def amplification_ci(correct_id, correct_ood, layer_mid, rng):
    '''(gap_OOD - gap_ID) at a fixed mid layer; same indices across all four vectors.'''
    n = correct_id[layer_mid].size; idx = rng.integers(0, n, size=(BOOT_B, n))
    g_id  = correct_id[layer_mid][idx].mean(1)  - correct_id[LAST_LAYER][idx].mean(1)
    g_ood = correct_ood[layer_mid][idx].mean(1) - correct_ood[LAST_LAYER][idx].mean(1)
    amp = g_ood - g_id
    pt = (correct_ood[layer_mid].mean() - correct_ood[LAST_LAYER].mean()) \
       - (correct_id[layer_mid].mean()  - correct_id[LAST_LAYER].mean())
    return float(pt), float(np.percentile(amp, 2.5)), float(np.percentile(amp, 97.5))

def amplification_band_ci(correct_id, correct_ood, rng):
    '''Band-mean version of amplification (the principled one).'''
    n = correct_id[LAST_LAYER].size; idx = rng.integers(0, n, size=(BOOT_B, n))
    bid, bood = band_correct(correct_id), band_correct(correct_ood)
    g_id  = bid[idx].mean(1)  - correct_id[LAST_LAYER][idx].mean(1)
    g_ood = bood[idx].mean(1) - correct_ood[LAST_LAYER][idx].mean(1)
    amp = g_ood - g_id
    pt = (bood.mean() - correct_ood[LAST_LAYER].mean()) - (bid.mean() - correct_id[LAST_LAYER].mean())
    return float(pt), float(np.percentile(amp, 2.5)), float(np.percentile(amp, 97.5))

def saturation_gate(accs):
    '''Flag a dataset as saturating if >= 3 layers reach acc >= 0.95 (no headroom to test H).'''
    sat = int((np.asarray(accs) >= 0.95).sum())
    return sat >= 3, sat
""")

code(r"""
# ---------- shared plotting helpers ----------
def plot_layer_curves(curves, chance=None, title="", ylabel="test accuracy", ylim=None,
                      save=None, annotate_argmax=False):
    '''curves: list of (label, point[L], lo[L]|None, hi[L]|None, color, kwdict|None).'''
    fig, ax = plt.subplots(figsize=(7, 4.2)); xs = np.arange(NUM_LAYERS)
    for label, pt, lo, hi, color, kw in curves:
        kw = dict(kw or {})
        if lo is not None:
            ax.fill_between(xs, lo, hi, alpha=0.18, color=color, linewidth=0)
        ax.plot(xs, pt, marker="o", color=color, label=label, **kw)
        if annotate_argmax:
            j = int(np.argmax(pt)); ax.annotate(f"argmax L{j}", (j, pt[j]),
                                                (j, min(pt[j]+0.04, 1.04)), ha="center", fontsize=8, color=color)
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", lw=1, label=f"chance ({chance:.3f})")
    ax.set_xlabel("encoder layer index"); ax.set_ylabel(ylabel); ax.set_xticks(xs)
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    if ylim: ax.set_ylim(*ylim)
    fig.tight_layout()
    if save: fig.savefig(OUT_DIR / save, dpi=140); print(f"  [saved] {save}")
    plt.show()

def acc_with_ci(correct, rng):
    '''Per-layer (accs, lo, hi) arrays from a correctness dict.'''
    a = np.zeros(NUM_LAYERS); lo = np.zeros(NUM_LAYERS); hi = np.zeros(NUM_LAYERS)
    for i in range(NUM_LAYERS):
        a[i], lo[i], hi[i] = bootstrap_ci(correct[i], rng=rng)
    return a, lo, hi
""")

code(r"""
# ---------- one convenience wrapper used by most experiments ----------
def probe_dataset(dataset, pooling="content", rng=None):
    '''Fit per-layer probes on clean train; return probes + clean-test correctness/accs/CIs.'''
    rng = rng or np.random.default_rng(SEED)
    f_tr, y_tr = extract_features(dataset, "train", pooling=pooling)
    f_te, y_te = extract_features(dataset, "test",  pooling=pooling)
    probes = fit_layerwise_probes(f_tr, y_tr)
    correct = score_correctness(probes, f_te, y_te)
    accs, lo, hi = acc_with_ci(correct, rng)
    classes = np.unique(y_tr)
    return dict(probes=probes, f_te=f_te, y_te=y_te, correct=correct,
                accs=accs, lo=lo, hi=hi, chance=1.0/len(classes),
                n_test=len(y_te), n_classes=len(classes))

def score_shift(probes, dataset, corruption, y_ref, rng=None):
    '''Score a FROZEN clean-train probe on a corrupted TEST split. Returns correctness/accs/CIs.'''
    rng = rng or np.random.default_rng(SEED)
    f_o, y_o = extract_features(dataset, "test", corruption=corruption, pooling="content")
    assert np.array_equal(y_o, y_ref), "labels changed under corruption"
    correct = score_correctness(probes, f_o, y_ref)
    accs, lo, hi = acc_with_ci(correct, rng)
    return dict(correct=correct, accs=accs, lo=lo, hi=hi)
""")

# ===================================================================== #
md(r"""
## 4 · Stage 2 — First experiment & the informative failure (Epilepsy)

> **Concept — saturation.** If a dataset is so easy that the probe hits ~100% at many layers, the
> accuracy-vs-layer curve is flat at the ceiling and there is no headroom for layer differences to
> appear. A saturated dataset validates the pipeline but **cannot test the hypothesis**. Our
> saturation gate flags ≥3 layers ≥0.95. **Epilepsy trips it** — which is exactly why it became the
> known-saturated reference and SCP1/Handwriting became the real test beds.
""")

code(r"""
rng = np.random.default_rng(SEED)
epi = probe_dataset("Epilepsy", rng=rng)
sat, sat_n = saturation_gate(epi["accs"])

print(f"Epilepsy: n_test={epi['n_test']}  n_classes={epi['n_classes']}  chance={epi['chance']:.3f}")
print(f"{'layer':>5} {'acc':>8}  95% CI")
for i in range(NUM_LAYERS):
    print(f"{i:>5} {epi['accs'][i]:>8.4f}  [{epi['lo'][i]:.4f}, {epi['hi'][i]:.4f}]")
print(f"argmax = L{int(np.argmax(epi['accs']))}  acc={epi['accs'].max():.4f}")
if sat:
    print(f"\n*** SATURATION WARNING: {sat_n}/12 layers >= 0.95 -> cannot test the hypothesis here.")
    print("    Epilepsy becomes the known-saturated reference; primary test beds are SCP1 & Handwriting.")

plot_layer_curves([("Epilepsy ID (clean test)", epi["accs"], epi["lo"], epi["hi"], "C0", None)],
                  chance=epi["chance"], title="Epilepsy: layer-wise linear probe (saturates)",
                  ylim=(0, 1.05), annotate_argmax=True, save="fig_id_epilepsy.png")
""")

md(r"""
### 4.1 · Synthetic shift on Epilepsy (first OOD hint)

> **Two different "OOD" designs — keep them crisp.**
> **(a) Synthetic input shift** (this section): one dataset, probe trained on clean train,
> evaluated on clean test (ID) *and* a label-preserving corruption of the test inputs (OOD). Probe
> and scaler stay **frozen** from ID; only the inputs shift. This is *true transfer-under-shift* —
> same task, same classes, directly comparable on one axis.
> **(b) Cross-dataset comparison** (§6): two datasets, each with its own probe — compares where the
> profile *peaks*. **Not** classifier transfer (different label spaces).

Even though Epilepsy saturates, the strong-noise end of the sweep gives the first hint that deep
layers degrade under corruption.
""")

code(r"""
rng = np.random.default_rng(SEED)
alphas = [0.1, 0.25, 0.5]
epi_ood = {a: score_shift(epi["probes"], "Epilepsy", {"kind":"gauss","alpha":a,"seed":SEED}, epi["y_te"], rng)
           for a in alphas}

curves = [("ID (clean test)", epi["accs"], epi["lo"], epi["hi"], "C0", {"linewidth":2})]
for j, a in enumerate(alphas):
    curves.append((f"OOD gauss α={a}", epi_ood[a]["accs"], epi_ood[a]["lo"], epi_ood[a]["hi"], f"C{j+1}", {"linestyle":"--"}))
plot_layer_curves(curves, chance=epi["chance"],
                  title="Epilepsy: ID vs Gaussian-noise OOD sweep (frozen probe)",
                  ylim=(0, 1.05), save="fig_ood_sweep_epilepsy.png")
""")

# ===================================================================== #
md(r"""
## 5 · Stage 3 — A real signal with honest uncertainty (SCP1 primary)

**SCP1 (SelfRegulationSCP1, EEG)** is the primary dataset: it has real headroom (~0.76–0.86) and
is a far-from-pretraining domain where the tunnel effect should be strongest. **Handwriting**
(26 classes) is the second headroom dataset. From here on, every accuracy carries a bootstrap CI.
""")

code(r"""
rng = np.random.default_rng(SEED)
scp = probe_dataset("SelfRegulationSCP1", rng=rng)

print(f"SCP1: n_test={scp['n_test']}  n_classes={scp['n_classes']}  chance={scp['chance']:.3f}")
print(f"{'layer':>5} {'ID acc':>20}")
for i in range(NUM_LAYERS):
    print(f"{i:>5}  {scp['accs'][i]:.4f} [{scp['lo'][i]:.4f}, {scp['hi'][i]:.4f}]")

ldb = late_drop_band_ci(scp["correct"], np.random.default_rng(SEED))
lda = late_drop_argmax_ci(scp["correct"], np.random.default_rng(SEED))
print(f"\nargmax = L{int(np.argmax(scp['accs']))}  acc={scp['accs'].max():.4f}")
print(f"late_drop_band   (headline)        = {ldb[0]:+.4f}  CI [{ldb[1]:+.4f}, {ldb[2]:+.4f}]  "
      f"{'EXCLUDES 0' if excl0(ldb[1],ldb[2]) else 'includes 0'}")
print(f"late_drop_argmax (selection-biased)= {lda[0]:+.4f}  CI [{lda[1]:+.4f}, {lda[2]:+.4f}]  (L*={lda[3]})")
""")

code(r"""
# SCP1 ID vs OOD sweep (frozen clean-train probe; only test inputs shift)
rng = np.random.default_rng(SEED)
scp_ood = {a: score_shift(scp["probes"], "SelfRegulationSCP1", {"kind":"gauss","alpha":a,"seed":SEED}, scp["y_te"], rng)
           for a in alphas}

plot_layer_curves(
    [("SCP1 ID (clean test)",        scp["accs"],         scp["lo"],         scp["hi"],         "C0", None),
     ("OOD (gauss α=0.25)",          scp_ood[0.25]["accs"],scp_ood[0.25]["lo"],scp_ood[0.25]["hi"],"C3", None)],
    chance=scp["chance"], title="SCP1: ID vs synthetic-noise OOD (95% bootstrap CI)",
    ylim=(0.35, 1.0), save="fig_scp1_id_ood.png")

sweep = [("SCP1 ID", scp["accs"], scp["lo"], scp["hi"], "C0", {"linewidth":2})]
for j, a in enumerate(alphas):
    sweep.append((f"OOD α={a}", scp_ood[a]["accs"], scp_ood[a]["lo"], scp_ood[a]["hi"], f"C{j+1}", {"linestyle":"--"}))
plot_layer_curves(sweep, chance=scp["chance"], title="SCP1: ID vs Gaussian-noise OOD sweep (95% CI)",
                  ylim=(0.35, 1.0), save="fig_scp1_sweep.png")
""")

md(r"""
### 5.1 · Handwriting — the textbook tunnel *shape*

26 classes, lots of headroom. The clean-ID curve shows the classic tunnel shape: a rise, a
plateau, then a late decline.
""")

code(r"""
rng = np.random.default_rng(SEED)
hw = probe_dataset("Handwriting", rng=rng)
hw_sat, hw_sat_n = saturation_gate(hw["accs"])
ldb_hw = late_drop_band_ci(hw["correct"], np.random.default_rng(SEED))

print(f"Handwriting: n_test={hw['n_test']}  n_classes={hw['n_classes']}  chance={hw['chance']:.4f}  "
      f"saturating={hw_sat} ({hw_sat_n}/12)")
print(f"argmax = L{int(np.argmax(hw['accs']))}  acc={hw['accs'].max():.4f}")
print(f"late_drop_band = {ldb_hw[0]:+.4f}  CI [{ldb_hw[1]:+.4f}, {ldb_hw[2]:+.4f}]  "
      f"{'EXCLUDES 0' if excl0(ldb_hw[1],ldb_hw[2]) else 'includes 0'}")

plot_layer_curves([("Handwriting ID (clean test)", hw["accs"], hw["lo"], hw["hi"], "C0", None)],
                  chance=hw["chance"], title="Handwriting: layer-wise probe (textbook tunnel shape)",
                  ylim=(0, 0.45), annotate_argmax=True, save="fig_handwriting_id_ood.png")
""")

md(r"""
### 5.2 · Cross-dataset comparison — raw accuracy (a presentation pitfall, fixed)

> **The pitfall we caught.** The first cross-dataset plot used **min–max normalization** within
> each dataset. On a dataset whose raw accuracy lives in 0.95–1.00, min–max stretches pure noise
> into a dramatic-looking "profile." **Lesson:** plot **raw accuracy** (and always show the range);
> normalize only when scales genuinely differ and you're comparing shape. This is a *layer-profile*
> comparison across domains — **not** classifier transfer (different label spaces, separate probes).
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 4.2)); xs = np.arange(NUM_LAYERS)
ax.fill_between(xs, epi["lo"], epi["hi"], alpha=0.15, color="C0", lw=0)
ax.plot(xs, epi["accs"], "o-", color="C0", label="Epilepsy ID")
ax.fill_between(xs, scp["lo"], scp["hi"], alpha=0.15, color="C3", lw=0)
ax.plot(xs, scp["accs"], "s-", color="C3", label="SelfRegulationSCP1 ID")
ax.axhline(epi["chance"], ls="--", color="C0", lw=1, alpha=0.6, label=f"Epilepsy chance ({epi['chance']:.3f})")
ax.axhline(scp["chance"], ls="--", color="C3", lw=1, alpha=0.6, label=f"SCP1 chance ({scp['chance']:.3f})")
for r, c, m in [(epi, "C0", "o"), (scp, "C3", "s")]:
    j = int(np.argmax(r["accs"])); ax.annotate(f"argmax L{j}", (j, r["accs"][j]),
                                               (j, r["accs"][j]+0.04), ha="center", fontsize=8, color=c)
ax.set_xlabel("encoder layer index"); ax.set_ylabel("test accuracy"); ax.set_xticks(xs)
ax.set_title("Layer-profile comparison (raw accuracy, NOT classifier transfer)")
ax.set_ylim(0, 1.08); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT_DIR / "fig_transfer_raw.png", dpi=140); print("  [saved] fig_transfer_raw.png")
plt.show()
""")

# ===================================================================== #
md(r"""
## 6 · Stage 4 — Trying to break it: five hardening tests

> A result you haven't tried to break is just a hopeful anecdote. Each test targets a specific way
> the finding could be fake.
""")

md(r"""
### Test 1 — Pooling ablation: *does the result depend on how we pooled?*

Re-run the probe with `content` / `all` / `reg` pooling (all cached → nearly free).
""")

code(r"""
rng = np.random.default_rng(SEED)
pool_rows = []
for ds in ("SelfRegulationSCP1", "Handwriting"):
    for pool in ("content", "all", "reg"):
        d = probe_dataset(ds, pooling=pool, rng=rng)
        pt, lo, hi = late_drop_band_ci(d["correct"], np.random.default_rng(SEED))
        pool_rows.append((ds, pool, pt, lo, hi, excl0(lo, hi)))

print(f"{'dataset':>20} {'pool':>8} {'late_drop_band':>16} {'95% CI':>22} {'excl0':>6}")
for ds, pool, pt, lo, hi, e in pool_rows:
    print(f"{ds:>20} {pool:>8} {pt:>+16.4f} [{lo:+.4f},{hi:+.4f}] {'YES' if e else 'no':>6}")

# grouped bar chart
fig, ax = plt.subplots(figsize=(7, 4.2))
dss = ["SelfRegulationSCP1", "Handwriting"]; pools = ["content","all","reg"]; x = np.arange(len(dss)); w = 0.25
for j, pool in enumerate(pools):
    pts = [next(r for r in pool_rows if r[0]==ds and r[1]==pool)[2] for ds in dss]
    los = [next(r for r in pool_rows if r[0]==ds and r[1]==pool)[3] for ds in dss]
    his = [next(r for r in pool_rows if r[0]==ds and r[1]==pool)[4] for ds in dss]
    ax.bar(x+(j-1)*w, pts, w, yerr=[np.array(pts)-los, np.array(his)-pts], capsize=4,
           label=pool, color=f"C{j}", alpha=0.85)
ax.axhline(0, color="gray", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels(dss)
ax.set_ylabel("late_drop_band = mean(L3..L8) - L11"); ax.legend(title="pooling")
ax.set_title("Test 1: pooling ablation (95% paired CI)"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT_DIR/"fig_pooling_ablation.png", dpi=140); print("  [saved] fig_pooling_ablation.png")
plt.show()
""")

md(r"""
**Reading it:** the deficit holds for `content` and `all` on both datasets, but the **REG token
flips sign on Handwriting** — there the last-layer REG representation is *better* than the
mid-band. Interpretation: the tunnel effect lives in the **distributed content-patch
representations**, while the register token behaves like a CLS aggregator that can keep improving
to the end. A genuine nuance — flag it, don't over-theorize (one dataset).
""")

md(r"""
### Test 2 — Replication: *is two datasets just luck?*

Run the ID probe + late-drop test on several more non-saturated UEA datasets (channels capped at
≤8). Saturated datasets are recorded but excluded from the Part-1 claim.

> **Concept — consistent direction vs. multiple comparisons.** Many *independent* tests pointing
> the *same* way is evidence *for* a real effect (replication) — the opposite of cherry-picking one
> significant hit out of many. Part 1's tally is the good kind.
""")

code(r"""
rng = np.random.default_rng(SEED)
PART2 = ["Epilepsy", "SelfRegulationSCP1", "Handwriting", "UWaveGestureLibrary",
         "EthanolConcentration", "SelfRegulationSCP2", "LSST", "Cricket"]
rep_rows, failures = [], []
for ds in PART2:
    try:
        f_tr, y_tr = extract_features(ds, "train", pooling="content")
        if f_tr[0].shape[1] // 768 > 8:
            failures.append((ds, f"too many channels ({f_tr[0].shape[1]//768})")); continue
        f_te, y_te = extract_features(ds, "test", pooling="content")
    except Exception as e:
        failures.append((ds, f"{type(e).__name__}: {e}")); continue
    probes = fit_layerwise_probes(f_tr, y_tr)
    correct = score_correctness(probes, f_te, y_te)
    accs = np.array([correct[L].mean() for L in range(NUM_LAYERS)])
    sat, sat_n = saturation_gate(accs)
    pt, lo, hi = late_drop_band_ci(correct, np.random.default_rng(SEED))
    rep_rows.append(dict(ds=ds, n_test=len(y_te), n_cls=len(np.unique(y_te)), chance=1/len(np.unique(y_te)),
                         sat=sat, argmax=int(np.argmax(accs)), pt=pt, lo=lo, hi=hi, excl=excl0(lo,hi)))

print(f"{'dataset':>20} {'n_test':>6} {'cls':>4} {'chance':>7} {'sat?':>5} {'argmax':>6} "
      f"{'late_drop_band':>15} {'95% CI':>20} {'excl0':>5}")
for r in rep_rows:
    print(f"{r['ds']:>20} {r['n_test']:>6} {r['n_cls']:>4} {r['chance']:>7.3f} {'YES' if r['sat'] else 'no':>5} "
          f"L{r['argmax']:<5} {r['pt']:>+15.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {'YES' if r['excl'] else 'no':>5}")
for ds, why in failures: print(f"  (skipped {ds}: {why})")

nonsat = [r for r in rep_rows if not r["sat"]]
K = len(nonsat); J = sum(1 for r in nonsat if r["excl"] and r["pt"] > 0)
print(f"\nHEADLINE: final layer significantly worse than middle band in {J} of {K} non-saturated datasets.")
""")

code(r"""
# Forest plot (non-saturated only)
ns = sorted(nonsat, key=lambda r: r["pt"])
fig, ax = plt.subplots(figsize=(7.5, max(3, 0.5*len(ns)+1.5)))
ys = np.arange(len(ns))
ax.errorbar([r["pt"] for r in ns], ys, xerr=[[r["pt"]-r["lo"] for r in ns],[r["hi"]-r["pt"] for r in ns]],
            fmt="o", color="black", markerfacecolor="white", capsize=3)
for y, r in zip(ys, ns):
    ax.add_patch(plt.Rectangle((r["lo"], y-0.15), r["hi"]-r["lo"], 0.30,
                 facecolor="C2" if r["excl"] else "C7", alpha=0.25, edgecolor="none"))
ax.set_yticks(ys); ax.set_yticklabels([r["ds"] for r in ns]); ax.axvline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("late_drop_band = mean(L3..L8) - L11  (95% paired CI)")
ax.set_title(f"Test 2: late-layer deficit, non-saturated datasets ({J}/{K} CIs exclude 0)")
ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
fig.savefig(OUT_DIR/"fig_dataset_forest.png", dpi=140); print("  [saved] fig_dataset_forest.png"); plt.show()
""")

md(r"""
**Reading it:** the deficit is significant (CI excludes 0, positive) in **5 of 6** non-saturated
datasets. The 6th, **SCP2**, is a *non-counterexample* — its probe sits at chance, so it cannot
resolve any per-layer difference (underpowered, not contradictory). LSST's effect is real but tiny
(its 2466-sample test set makes a small gap significant). Cricket saturates and is excluded.
""")

md(r"""
### Test 3 — Is Part 2 significant at all? (SCP1 / Gaussian α=0.25)

Fix `best_mid` = argmax of clean-ID accuracy *within the band*, use the same layer for ID and OOD,
and bootstrap the amplification.
""")

code(r"""
rng = np.random.default_rng(SEED)
mid_accs = np.array([scp["correct"][L].mean() for L in MIDDLE_BAND])
best_mid = MIDDLE_BAND[int(np.argmax(mid_accs))]
oc = score_shift(scp["probes"], "SelfRegulationSCP1", {"kind":"gauss","alpha":0.25,"seed":SEED}, scp["y_te"], rng)["correct"]

amp   = amplification_ci(scp["correct"], oc, best_mid, np.random.default_rng(SEED))
amp_b = amplification_band_ci(scp["correct"], oc, np.random.default_rng(SEED))
print(f"best_mid = L{best_mid}  (clean-ID acc {mid_accs.max():.4f})")
print(f"amplification @ best_mid (L{best_mid}) = {amp[0]:+.4f}  CI [{amp[1]:+.4f}, {amp[2]:+.4f}]  "
      f"{'EXCLUDES 0' if excl0(amp[1],amp[2]) else 'includes 0'}")
print(f"amplification @ band (L3..L8)         = {amp_b[0]:+.4f}  CI [{amp_b[1]:+.4f}, {amp_b[2]:+.4f}]  "
      f"{'EXCLUDES 0' if excl0(amp_b[1],amp_b[2]) else 'includes 0'}")
print("\n-> At the single best layer NOT significant; at the a-priori band it IS. On this one condition, Part 2 has support.")
""")

md(r"""
### Test 4 — Split stability: *is it an artifact of the one canonical split?*

Pool train+test, make 20 fresh stratified re-splits at the canonical test fraction, refit each
time (no re-extraction), and look at the late-drop distribution.

> **Concept — what the bootstrap does *not* cover.** Bootstrap CIs treat the train/test split as
> fixed. Re-splitting probes a *different* source of variability — which samples land in train vs
> test. Passing both is stronger than either.
""")

code(r"""
def multisplit(dataset, n_splits=20):
    f_tr, y_tr = extract_features(dataset, "train", pooling="content")
    f_te, y_te = extract_features(dataset, "test",  pooling="content")
    Xf = {L: np.concatenate([f_tr[L], f_te[L]], 0) for L in range(NUM_LAYERS)}
    yf = np.concatenate([np.asarray(y_tr), np.asarray(y_te)], 0)
    n = len(yf); test_frac = len(y_te)/n
    drops, argmaxes = [], []
    for s in range(n_splits):
        (tr, te), = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=s).split(np.zeros(n), yf)
        accs = np.zeros(NUM_LAYERS); corr = {}
        for L in range(NUM_LAYERS):
            sc = StandardScaler().fit(Xf[L][tr])
            clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(sc.transform(Xf[L][tr]), yf[tr])
            corr[L] = (clf.predict(sc.transform(Xf[L][te])) == yf[te]).astype(np.float32)
            accs[L] = corr[L].mean()
        drops.append(band_correct(corr).mean() - accs[LAST_LAYER])
        argmaxes.append(int(np.argmax(accs[:NUM_LAYERS-1])))
    return np.array(drops), argmaxes, test_frac

ms = {}
for ds in ("SelfRegulationSCP1", "Handwriting"):
    drops, argmaxes, tf = multisplit(ds)
    ms[ds] = drops
    hist = {L: argmaxes.count(L) for L in sorted(set(argmaxes))}
    print(f"{ds}: test_frac={tf:.3f}  late_drop_band mean={drops.mean():+.4f}  "
          f"pct[2.5,97.5]=[{np.percentile(drops,2.5):+.4f},{np.percentile(drops,97.5):+.4f}]  "
          f"frac>0={(drops>0).mean():.2f}  argmax-hist={hist}")

fig, ax = plt.subplots(figsize=(7, 4.2))
bp = ax.boxplot([ms[d] for d in ms], labels=list(ms), patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["C0","C3"]): patch.set_facecolor(c); patch.set_alpha(0.35)
for i, d in enumerate(ms):
    ax.scatter(np.full(len(ms[d]), i+1)+np.random.uniform(-0.06,0.06,len(ms[d])), ms[d], color="black", s=14, alpha=0.7, zorder=3)
ax.axhline(0, color="gray", ls="--", lw=1); ax.set_ylabel("late_drop_band over 20 re-splits")
ax.set_title("Test 4: re-split robustness (distinct from canonical-split headline)"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT_DIR/"fig_multisplit_stability.png", dpi=140); print("  [saved] fig_multisplit_stability.png"); plt.show()
""")

md(r"""
### Test 5 — Does Part 2 generalize across shift types? (the adjudicator)

Add two more label-preserving corruptions beyond Gaussian — **time-warp** and **baseline drift** —
and ask whether the mid-vs-last gap widens under *each*, across SCP1 and Handwriting (6 cells). We
sanity-check that each shift keeps `best_mid` accuracy well above chance (labels survive).
""")

code(r"""
rng = np.random.default_rng(SEED)
SHIFTS = [("gauss(α=0.25)", {"kind":"gauss","alpha":0.25,"seed":SEED}),
          ("timewarp(f=1.2)", {"kind":"timewarp","factor":1.2}),
          ("drift(amp=0.3)",  {"kind":"drift","amplitude":0.3})]

amp_table = {}  # (dataset, shift_label) -> (pt, lo, hi)
for base in (scp, hw):
    ds = "SelfRegulationSCP1" if base is scp else "Handwriting"
    mid_accs = np.array([base["correct"][L].mean() for L in MIDDLE_BAND])
    bm = MIDDLE_BAND[int(np.argmax(mid_accs))]
    print(f"\n[{ds}]  best_mid=L{bm}  chance={base['chance']:.4f}")
    for label, corr in SHIFTS:
        sh = score_shift(base["probes"], ds, corr, base["y_te"], rng)
        amp_b = amplification_band_ci(base["correct"], sh["correct"], np.random.default_rng(SEED))
        amp_table[(ds, label)] = amp_b
        print(f"  {label:>16}: best_mid acc clean={base['correct'][bm].mean():.3f} "
              f"shifted={sh['correct'][bm].mean():.3f}  |  amp@band={amp_b[0]:+.4f} "
              f"CI [{amp_b[1]:+.4f},{amp_b[2]:+.4f}] {'EXCL0' if excl0(amp_b[1],amp_b[2]) else 'incl0'}")
        if ds == "SelfRegulationSCP1" and label.startswith("timewarp"):
            scp_tw = sh
""")

code(r"""
# Amplification bar chart across shift types x datasets
fig, ax = plt.subplots(figsize=(8, 4.2))
labels = [s[0] for s in SHIFTS]; dss = ["SelfRegulationSCP1","Handwriting"]; x = np.arange(len(labels)); w = 0.38
for i, ds in enumerate(dss):
    pts = [amp_table[(ds,l)][0] for l in labels]; los = [amp_table[(ds,l)][1] for l in labels]; his=[amp_table[(ds,l)][2] for l in labels]
    ax.bar(x+(i-0.5)*w, pts, w, yerr=[np.array(pts)-los, np.array(his)-pts], capsize=4,
           color=("C3" if ds=="SelfRegulationSCP1" else "C1"), alpha=0.85, label=ds)
ax.axhline(0, color="gray", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("amplification = (band-L11)_OOD - (band-L11)_ID  (95% CI)")
ax.set_title("Test 5: does the mid-vs-last gap widen under MULTIPLE shift types?"); ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT_DIR/"fig_shift_amplification.png", dpi=140); print("  [saved] fig_shift_amplification.png"); plt.show()

# SCP1 ID vs time-warp OOD curve
plot_layer_curves(
    [("SCP1 ID (clean test)", scp["accs"], scp["lo"], scp["hi"], "C0", None),
     ("SCP1 OOD (timewarp f=1.2)", scp_tw["accs"], scp_tw["lo"], scp_tw["hi"], "C4", None)],
    chance=scp["chance"], title="SCP1: ID vs time-warp OOD (curves come out ~parallel)",
    ylim=(0.35, 1.0), save="fig_scp1_timewarp_idood.png")
""")

md(r"""
**Reading it:** **only 1 of 6 cells** (SCP1/Gaussian) is significant in the predicted direction;
4/6 include 0; one isolated cell (Handwriting/drift) is significant but stands alone.

> **Why time-warp produces no amplification — and why that's sensible.** The time-warp ID and OOD
> curves come out roughly **parallel**: the warp knocks a similar amount off *every* layer. A shift
> that degrades all layers uniformly produces no differential for the last layer to lose more of.
> Only the input-coupled Gaussian case created a per-layer gradient — and even that didn't
> reproduce on the second dataset.
""")

# ===================================================================== #
md(r"""
## 7 · Where it landed: one robust finding, one honest null

**Part 1 — supported; state it strongly.**
> *Linear-probe accuracy of frozen Chronos-2 peaks in the middle encoder layers and is
> significantly lower at the final layer — significant in 5 of 6 non-saturated UEA datasets (95%
> paired-bootstrap CIs exclude 0), robust to pooling (content & all-positions), and stable across
> 20 re-splits — consistent with the tunnel effect and the final layer's specialization to the
> forecasting objective.*

The strength is the **coherence**: every independent test points the same way. The two soft spots
are not counterexamples (SCP2 underpowered at chance; LSST significant but tiny).

**Part 2 — not established; report as a clean null.**
> *The prediction that the middle-vs-last advantage widens under distribution shift did not
> replicate. It held only for Gaussian noise on SCP1 and failed under time-warp, drift, and on the
> second dataset (1 of 6 conditions). Uniform shifts degrade all layers roughly equally, so the
> original effect appears shift-specific rather than a general property of distribution shift.*

**Integrity points.** Don't chase a seventh shift hoping one lands (manufactures a false positive);
don't feature the isolated Handwriting/drift hit (among ~6 tests you expect ~1 false positive). A
*differentiated* answer — one part robust, one part not, weakness found by us — is what good
experimental work looks like.

## 8 · Limitations & next steps

- **Synthetic shifts ≠ real domain shift.** The proper Part-2 test is true cross-domain transfer or
  a naturally-shifted dataset (main follow-up).
- **Linear probe only** (by design, matching Head2Toe). A nonlinear probe or Head2Toe's actual
  multi-layer feature selection could change the picture.
- **Layer-level only.** Module-level splits (probe MHSA vs FFN sub-outputs separately) are the next
  resolution increase.
- **Per-channel concatenation, not native multivariate.** Probing Chronos-2's jointly-encoded
  group-ID representation is a distinct experiment.
- **Single model.** Whether the tunnel profile generalizes to other TSFMs (MOMENT, TimesFM, Moirai)
  is open.

---
*This notebook is the executable companion to `probing_walkthrough.md`. All four original scripts
(`extract_check.py`, `probe_pipeline.py`, `probe_improve.py`, `probe_harden.py`) are consolidated
here; the `./features_cache/` directory makes re-runs instant and is the thing to share alongside
this notebook for a zero-recompute handoff.*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "TS_Experiment (.venv)", "language": "python", "name": "tsexp"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(nb, "chronos2_probing.ipynb")
print(f"wrote chronos2_probing.ipynb with {len(cells)} cells")
