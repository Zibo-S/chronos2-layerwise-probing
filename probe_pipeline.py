"""
Layer-wise linear-probing pipeline over a frozen Chronos-2 encoder.

Pipeline:
  Phase A  reusable hook-based extract_features(...) with on-disk caching
  Phase B  ID layer-wise probe accuracy on Epilepsy
  Phase C  synthetic-shift OOD (Gaussian noise on TEST only) reusing B's
           scalers + probes
  Phase D  cross-domain layer-profile comparison (Epilepsy vs SelfRegulationSCP1)

Chronos-2 weights are frozen throughout (eval, no_grad, requires_grad=False).
Everything stays in float32 on MPS (or CPU fallback).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SEED = 0
NUM_LAYERS = 12
CACHE_DIR = Path("./features_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = Path(".")


# ----------------------------------------------------------------------- #
# Model loading (reused from extract_check.py)
# ----------------------------------------------------------------------- #

_PIPELINE = None
_CFG = None


def get_pipeline():
    global _PIPELINE, _CFG
    if _PIPELINE is not None:
        return _PIPELINE, _CFG

    from chronos import Chronos2Pipeline

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[model] loading amazon/chronos-2 (float32) -> target device {device}")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", torch_dtype=torch.float32)
    inner = pipeline.model
    try:
        inner.to(device)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"to({device}) failed: {e}; falling back to CPU")
        inner.to("cpu")
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)
    print(f"[model] params on {next(inner.parameters()).device}, "
          f"d_model={inner.config.d_model}, num_layers={inner.config.num_layers}, "
          f"patch_size={inner.chronos_config.input_patch_size}")
    _PIPELINE = pipeline
    _CFG = inner.config
    return pipeline, _CFG


# ----------------------------------------------------------------------- #
# Phase A — extract_features
# ----------------------------------------------------------------------- #

def _corr_repr(corruption):
    if corruption is None:
        return "clean"
    parts = [f"{k}={corruption[k]}" for k in sorted(corruption.keys())]
    return ",".join(parts)


def _cache_path(dataset, split, corruption, pooling):
    return CACHE_DIR / f"{dataset}__{split}__{_corr_repr(corruption)}__{pooling}.npz"


def _all_pool_caches_exist(dataset, split, corruption):
    return all(_cache_path(dataset, split, corruption, p).exists()
               for p in ("content", "all", "reg"))


def _load_cache(path):
    d = np.load(path, allow_pickle=True)
    features = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("layer_")}
    y = d["y"]
    return features, y


def _apply_gaussian_noise(X, alpha, seed):
    """Per-series, per-channel Gaussian noise with std = alpha * std(series, channel)."""
    rng = np.random.default_rng(seed)
    std = X.std(axis=-1, keepdims=True)  # (n, c, 1)
    noise = rng.standard_normal(X.shape).astype(np.float32) * (alpha * std).astype(np.float32)
    return (X + noise).astype(np.float32)


def _apply_timewarp(X, factor):
    """Label-preserving time warp via linear interpolation.

    Each series of length t is resampled onto a new grid of length round(t*factor),
    then center-cropped or edge-padded back to t. factor=1 is identity.
    """
    n, c, t = X.shape
    new_length = int(round(t * factor))
    old_idx = np.arange(t, dtype=np.float64)
    new_idx = np.linspace(0.0, t - 1.0, new_length)
    out = np.empty_like(X)
    for i in range(n):
        for ch in range(c):
            warped = np.interp(new_idx, old_idx, X[i, ch])
            if new_length == t:
                out[i, ch] = warped
            elif new_length > t:
                start = (new_length - t) // 2
                out[i, ch] = warped[start:start + t]
            else:
                pad_left = (t - new_length) // 2
                pad_right = t - new_length - pad_left
                out[i, ch] = np.pad(warped, (pad_left, pad_right), mode="edge")
    return out.astype(np.float32)


def _apply_drift(X, amplitude):
    """Half-cycle sinusoid added to each (series, channel), scaled by per-series-channel std."""
    n, c, t = X.shape
    std = X.std(axis=-1, keepdims=True)  # (n, c, 1)
    phase = np.sin(np.linspace(0.0, np.pi, t)).astype(np.float32)
    drift = (amplitude * std).astype(np.float32) * phase[None, None, :]
    return (X + drift).astype(np.float32)


def _apply_corruption(X, corruption):
    if corruption is None:
        return X
    kind = corruption["kind"]
    if kind == "gauss":
        return _apply_gaussian_noise(X, alpha=corruption["alpha"], seed=corruption["seed"])
    if kind == "timewarp":
        return _apply_timewarp(X, factor=corruption["factor"])
    if kind == "drift":
        return _apply_drift(X, amplitude=corruption["amplitude"])
    raise ValueError(f"unknown corruption kind: {kind!r}")


def extract_features(dataset_name, split, corruption=None, batch_size=64, pooling="content"):
    """Hook-based per-layer feature extraction with on-disk caching.

    Layout note (verified from chronos.chronos2.model.Chronos2Model.encode):
        encoder input is [content_patches..., REG, masked_future_patch].
        For embed() prediction_length=0 -> num_output_patches=1, so
        P = num_context_patches + 2  where num_context_patches = ceil(t/patch_size).

    Pooling variants computed in one pass and cached separately:
        - pool_content : mean over [0 .. num_context_patches-1]
        - pool_all     : mean over all P positions
        - pool_reg     : the REG token at index num_context_patches
    """
    cache_path = _cache_path(dataset_name, split, corruption, pooling)
    if cache_path.exists():
        print(f"  [cache HIT]  {cache_path.name}")
        return _load_cache(cache_path)

    if _all_pool_caches_exist(dataset_name, split, corruption):
        print(f"  [cache HIT]  {cache_path.name} (sibling pools present, requested missing)")
        # extremely unlikely branch but defensive

    print(f"  [cache MISS] extracting {dataset_name}/{split}  corruption={_corr_repr(corruption)}")

    from aeon.datasets import load_classification
    X, y = load_classification(dataset_name, split=split)
    X = np.asarray(X, dtype=np.float32)
    X = _apply_corruption(X, corruption)
    n, c, t = X.shape
    print(f"     X.shape={X.shape}  classes={sorted(np.unique(y).tolist())}")

    pipeline, cfg = get_pipeline()
    patch_size = pipeline.model.chronos_config.input_patch_size
    num_context_patches = math.ceil(t / patch_size)
    P_expected = num_context_patches + 2
    print(f"     patch_size={patch_size}  num_context_patches={num_context_patches}  "
          f"P_expected={P_expected}")

    # per_channel[pool][ch][layer] = (n, 768) float32 numpy array
    per_channel = {p: {ch: {} for ch in range(c)} for p in ("content", "all", "reg")}

    for ch in range(c):
        # Build univariate input list (one 1D tensor per case)
        inputs = [torch.from_numpy(np.ascontiguousarray(X[i, ch])).to(torch.float32) for i in range(n)]

        captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}

        def make_hook(layer_idx):
            def hook(_m, _inp, output):
                # Chronos2EncoderBlockOutput is a HF ModelOutput; .hidden_states is set,
                # and [0] also returns it (matches encoder.forward's layer_outputs[0]).
                if hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hs = output.hidden_states
                elif isinstance(output, (tuple, list)):
                    hs = output[0]
                else:
                    hs = output[0]
                # Move off-device immediately to keep MPS memory low.
                captured[layer_idx].append(hs.detach().to("cpu"))
            return hook

        hooks = [blk.register_forward_hook(make_hook(i))
                 for i, blk in enumerate(pipeline.model.encoder.block)]
        try:
            with torch.no_grad():
                _ = pipeline.embed(inputs, batch_size=batch_size)
        finally:
            for h in hooks:
                h.remove()

        for i in range(NUM_LAYERS):
            full = torch.cat(captured[i], dim=0)  # (n, P, 768)
            assert full.shape[0] == n, f"hook batch dim {full.shape[0]} != n={n} (ch={ch}, layer={i})"
            P_actual = full.shape[1]
            if ch == 0 and i == 0:
                assert P_actual == P_expected, (
                    f"P mismatch: expected {P_expected} (= ceil({t}/{patch_size})+2), got {P_actual}"
                )
                print(f"     P verified: actual={P_actual} (matches expected)")
            per_channel["content"][ch][i] = full[:, :num_context_patches, :].mean(dim=1).numpy()
            per_channel["all"][ch][i]     = full.mean(dim=1).numpy()
            per_channel["reg"][ch][i]     = full[:, num_context_patches, :].numpy()

        del captured

    # Concat across channels and write all three pooling caches.
    for pool_name in ("content", "all", "reg"):
        features = {
            i: np.concatenate([per_channel[pool_name][ch][i] for ch in range(c)], axis=1)
            for i in range(NUM_LAYERS)
        }
        cp = _cache_path(dataset_name, split, corruption, pool_name)
        save_kwargs = {f"layer_{i}": features[i] for i in range(NUM_LAYERS)}
        save_kwargs["y"] = y
        np.savez(cp, **save_kwargs)
        print(f"  [saved]      {cp.name}  per-layer shape={features[0].shape}")

    # Return the requested pooling.
    features = {
        i: np.concatenate([per_channel[pooling][ch][i] for ch in range(c)], axis=1)
        for i in range(NUM_LAYERS)
    }
    return features, y


# ----------------------------------------------------------------------- #
# Probe utilities
# ----------------------------------------------------------------------- #

def fit_layerwise_probes(train_features, y_train):
    """Fit one StandardScaler + LogisticRegression per layer. Returns dict per layer."""
    classes = sorted(np.unique(y_train).tolist())
    print(f"     n_train={len(y_train)}  classes={classes}")
    probes = {}
    for i in range(NUM_LAYERS):
        X_tr = train_features[i]
        scaler = StandardScaler().fit(X_tr)
        Xs = scaler.transform(X_tr)
        clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xs, y_train)
        probes[i] = {"scaler": scaler, "clf": clf}
    return probes


def score_layerwise(probes, test_features, y_test):
    accs = np.zeros(NUM_LAYERS, dtype=np.float64)
    for i in range(NUM_LAYERS):
        Xs = probes[i]["scaler"].transform(test_features[i])
        accs[i] = probes[i]["clf"].score(Xs, y_test)
    return accs


def print_layer_table(label, accs, chance):
    print(f"\n  Per-layer test accuracy ({label}):   chance = {chance:.4f}")
    print(f"  {'layer':>6}  {'acc':>8}")
    for i, a in enumerate(accs):
        print(f"  {i:>6d}  {a:>8.4f}")
    print(f"  argmax = layer {int(np.argmax(accs))}  (acc={accs.max():.4f})")


# ----------------------------------------------------------------------- #
# Plotting helpers
# ----------------------------------------------------------------------- #

def plot_curve(path, curves, chance=None, title="", ylabel="test accuracy",
               annotate_argmax=False, ylim=None):
    """curves: list of (label, ndarray length 12, kwargs-dict-or-None)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = np.arange(NUM_LAYERS)
    for label, ys, kw in curves:
        kw = dict(kw or {})
        ax.plot(xs, ys, marker="o", label=label, **kw)
        if annotate_argmax:
            j = int(np.argmax(ys))
            ax.annotate(f"argmax L{j}", xy=(j, ys[j]),
                        xytext=(j, ys[j] + 0.03), ha="center", fontsize=8)
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", linewidth=1, label=f"chance ({chance:.3f})")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [saved] {path}")


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------ Phase B
    print("\n" + "=" * 72)
    print("PHASE B  --  ID layer-wise probe on Epilepsy")
    print("=" * 72)

    print("Extracting Epilepsy/train ...")
    f_tr, y_tr = extract_features("Epilepsy", split="train")
    print("Extracting Epilepsy/test  ...")
    f_te, y_te = extract_features("Epilepsy", split="test")

    n_classes_epi = len(np.unique(y_tr))
    chance_epi = 1.0 / n_classes_epi

    print("\nFitting per-layer probes (Epilepsy) ...")
    probes_epi = fit_layerwise_probes(f_tr, y_tr)
    accs_id_epi = score_layerwise(probes_epi, f_te, y_te)
    print_layer_table("Epilepsy ID", accs_id_epi, chance_epi)

    saturated_layers = int((accs_id_epi >= 0.95).sum())
    saturation_warning = saturated_layers >= 3
    if saturation_warning:
        print(f"\n  *** SATURATION WARNING ***")
        print(f"  {saturated_layers} layers reach acc >= 0.95 on Epilepsy.")
        print(f"  This dataset is saturating layer-wise probing; recommended swap: "
              f"Handwriting (150/850 cases, 26 classes).")
    else:
        print(f"\n  Saturation check: {saturated_layers} layers >= 0.95 (no warning).")

    plot_curve(
        OUT_DIR / "fig_id_epilepsy.png",
        curves=[("Epilepsy ID (clean test)", accs_id_epi, {"color": "C0"})],
        chance=chance_epi,
        title="Epilepsy: layer-wise linear probe (test accuracy)",
        annotate_argmax=True,
        ylim=(0.0, 1.05),
    )

    # ------------------------------------------------------------ Phase C
    print("\n" + "=" * 72)
    print("PHASE C  --  synthetic shift on Epilepsy (Gaussian noise on TEST only)")
    print("=" * 72)
    alphas = [0.1, 0.25, 0.5]
    accs_ood = {}
    for alpha in alphas:
        print(f"\nExtracting Epilepsy/test  corruption=gauss alpha={alpha} ...")
        f_te_n, y_te_n = extract_features(
            "Epilepsy", split="test",
            corruption={"kind": "gauss", "alpha": alpha, "seed": SEED},
        )
        assert np.array_equal(y_te_n, y_te), "test labels changed under corruption"
        accs_ood[alpha] = score_layerwise(probes_epi, f_te_n, y_te)
        print_layer_table(f"Epilepsy OOD alpha={alpha}", accs_ood[alpha], chance_epi)

    # Per-layer ID vs OOD(0.25) table + gap
    print(f"\n  Per-layer  ID  vs  OOD(0.25)  gap = ID - OOD")
    print(f"  {'layer':>6}  {'ID':>8}  {'OOD':>8}  {'gap':>8}")
    gap = accs_id_epi - accs_ood[0.25]
    for i in range(NUM_LAYERS):
        print(f"  {i:>6d}  {accs_id_epi[i]:>8.4f}  {accs_ood[0.25][i]:>8.4f}  {gap[i]:>+8.4f}")
    print(f"  smallest gap: layer {int(np.argmin(gap))} ({gap.min():+.4f})")
    print(f"  largest  gap: layer {int(np.argmax(gap))} ({gap.max():+.4f})")

    # Headline figure: ID vs OOD(0.25)
    plot_curve(
        OUT_DIR / "fig_ood_synthetic_epilepsy.png",
        curves=[
            ("Epilepsy ID (clean test)", accs_id_epi, {"color": "C0"}),
            ("OOD (Gaussian noise, alpha=0.25)", accs_ood[0.25], {"color": "C3"}),
        ],
        chance=chance_epi,
        title="Epilepsy: ID vs synthetic noise OOD (probes/scalers frozen from ID)",
        ylim=(0.0, 1.05),
    )

    # Sweep figure: ID + all three alphas
    sweep_curves = [("Epilepsy ID (clean test)", accs_id_epi, {"color": "C0", "linewidth": 2})]
    for j, alpha in enumerate(alphas):
        sweep_curves.append(
            (f"OOD alpha={alpha}", accs_ood[alpha], {"color": f"C{j+1}", "linestyle": "--"})
        )
    plot_curve(
        OUT_DIR / "fig_ood_sweep_epilepsy.png",
        curves=sweep_curves,
        chance=chance_epi,
        title="Epilepsy: ID vs Gaussian-noise OOD sweep",
        ylim=(0.0, 1.05),
    )

    # ------------------------------------------------------------ Phase D
    print("\n" + "=" * 72)
    print("PHASE D  --  cross-domain transfer (Epilepsy vs SelfRegulationSCP1)")
    print("=" * 72)
    print("Extracting SelfRegulationSCP1/train ...")
    f_tr_scp, y_tr_scp = extract_features("SelfRegulationSCP1", split="train")
    print("Extracting SelfRegulationSCP1/test  ...")
    f_te_scp, y_te_scp = extract_features("SelfRegulationSCP1", split="test")

    n_classes_scp = len(np.unique(y_tr_scp))
    chance_scp = 1.0 / n_classes_scp

    print("\nFitting per-layer probes (SCP1) ...")
    probes_scp = fit_layerwise_probes(f_tr_scp, y_tr_scp)
    accs_id_scp = score_layerwise(probes_scp, f_te_scp, y_te_scp)
    print_layer_table("SelfRegulationSCP1 ID", accs_id_scp, chance_scp)

    print(f"\n  Raw per-layer accuracy comparison:")
    print(f"  {'layer':>6}  {'Epilepsy':>10}  {'SCP1':>10}")
    for i in range(NUM_LAYERS):
        print(f"  {i:>6d}  {accs_id_epi[i]:>10.4f}  {accs_id_scp[i]:>10.4f}")
    print(f"  Epilepsy: chance={chance_epi:.4f}  argmax=layer {int(np.argmax(accs_id_epi))}")
    print(f"  SCP1:     chance={chance_scp:.4f}  argmax=layer {int(np.argmax(accs_id_scp))}")

    # Min-max normalize each dataset's curve to [0, 1] within its own range
    def minmax(a):
        lo, hi = a.min(), a.max()
        return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

    n_epi = minmax(accs_id_epi)
    n_scp = minmax(accs_id_scp)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = np.arange(NUM_LAYERS)
    ax.plot(xs, n_epi, marker="o", color="C0", label=f"Epilepsy (raw range {accs_id_epi.min():.2f}-{accs_id_epi.max():.2f})")
    ax.plot(xs, n_scp, marker="s", color="C3", label=f"SCP1 (raw range {accs_id_scp.min():.2f}-{accs_id_scp.max():.2f})")
    j_epi = int(np.argmax(accs_id_epi))
    j_scp = int(np.argmax(accs_id_scp))
    ax.annotate(f"Epilepsy argmax L{j_epi}", xy=(j_epi, n_epi[j_epi]),
                xytext=(j_epi, n_epi[j_epi] + 0.06), ha="center", fontsize=8, color="C0")
    ax.annotate(f"SCP1 argmax L{j_scp}", xy=(j_scp, n_scp[j_scp]),
                xytext=(j_scp, n_scp[j_scp] - 0.08), ha="center", fontsize=8, color="C3")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel("min-max normalized test accuracy")
    ax.set_xticks(xs)
    ax.set_title("Layer profile comparison across domains (not classifier transfer)")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_transfer_profiles.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] {OUT_DIR / 'fig_transfer_profiles.png'}")

    print("\n  NOTE: this figure compares layer PROFILES across two domains, "
          "not classifier transfer — each probe was trained on its own dataset.")

    # ------------------------------------------------------------ Final summary
    pngs = ["fig_id_epilepsy.png", "fig_ood_synthetic_epilepsy.png",
            "fig_ood_sweep_epilepsy.png", "fig_transfer_profiles.png"]
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for p in pngs:
        path = OUT_DIR / p
        status = "OK" if path.exists() else "MISSING"
        print(f"  {status}  {p}  ({path.stat().st_size if path.exists() else 0} bytes)")
    print(f"  Saturation verdict (Epilepsy): "
          f"{'SATURATING - swap to Handwriting' if saturation_warning else 'not saturating'}  "
          f"({saturated_layers} layers >= 0.95)")
    print(f"  argmax layers:")
    print(f"    Epilepsy ID            : layer {int(np.argmax(accs_id_epi))} "
          f"(acc={accs_id_epi.max():.4f})")
    for alpha in alphas:
        print(f"    Epilepsy OOD alpha={alpha}: layer {int(np.argmax(accs_ood[alpha]))} "
              f"(acc={accs_ood[alpha].max():.4f})")
    print(f"    SelfRegulationSCP1 ID  : layer {int(np.argmax(accs_id_scp))} "
          f"(acc={accs_id_scp.max():.4f})")


if __name__ == "__main__":
    main()
