"""Layerwise spectral geometry (effective rank) of Chronos-2 representations — label-free.

Complements the tunnel-loss analysis with a representation-compression view: for every depth
(Emb L0 + blocks L1..L12) compute effective rank / spectral entropy / PC1 fraction of the EXACT
matrix the linear probe receives, on ONE fixed sample of rows shared by all layers.

Readouts (--readout; the sampled rows/geometry differ, everything downstream is identical):
  content   PRIMARY comparison. The cached content-pooled (mean over content-patch tokens)
            hidden states -> (N, 768); one row per window. -> results/extended_v3_rolling/spectral/.
  fslot     v4 headline. The K=ceil(H/P) native forecast-slot states per window (extract_kout_features
            -> feats["fslot"], (N, K, 768)) STACKED to (N*K, 768): every forecast slot is one point
            in the 768-d cloud the shared-head probe reads. -> results/ext_v4_future_tokens/spectral/.

Representation locations (recorded in every output, never silently mixed):
  probe_input   PRIMARY. The cached per-layer features (== post_block: raw block outputs; L12 is
                therefore PRE-final_layer_norm, matching what the probe sees). CPU-only, reuses the
                feature caches — no model, no extraction. Note: the probe additionally applies its
                own StandardScaler internally; geometry here is on the matrix BEFORE that scaling.
  post_final_ln Mechanistic diagnostic (the native head's input). NOT cached for the rolling set —
                requires a small GPU extraction pass; requesting it fails loud rather than substitute.

Backbone conditions: the pretrained backbone is deterministic, so there is ONE curve per dataset
(backbone_condition="pretrained", backbone_seed=null) — probe seeds are irrelevant to representation
geometry. Uncertainty = repeated subsampling WITHOUT replacement (subsample_metrics; naive
with-replacement bootstrap deflates rank and is not offered). Future ft_seed / random_init conditions
get one curve per backbone seed, kept separate.

Outputs: one JSON record per dataset x readout x location (schema: per-layer metrics + full normalized
spectra + subsample distributions + sampled row ids + provenance), plus per-metric layerwise figures.
Layer indexing/labels match the tunnel figures so curves can later be aligned with test loss and the
validation-defined tunnel shading.

Run (login node OK for PT-ID; PT-OOD needs its Stage-2 caches to exist):
    python -m experiments.run_spectral                                # content, warm caches
    python -m experiments.run_spectral --readout fslot               # v4 shared forecast-token (stacked)
    python -m experiments.run_spectral --datasets m4_hourly --repr-sample-size 4096
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing import config
from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE
from probing.extraction import _cache_path, _idf_prefix, _load_cache
from probing.spectral_metrics import spectral_metrics, subsample_metrics
from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS, domain_status

PTID_SET = "extended_v3_rolling"
POOLING = "content"                 # the probe pipeline's primary pooled readout
LOCATION = "probe_input"            # see module docstring; post_final_ln not cached (fail-loud)
ALL_TAGS = PT_ID_TAGS + PT_OOD_TAGS
C, H = 512, 64
K = math.ceil(H / OUTPUT_PATCH_SIZE)                    # native forecast-slot count (H=64 -> 4)
V4_ROOT = config.REPO_ROOT / "results" / "ext_v4_future_tokens"   # mirrors run_ptood_probing_ftok.OUT_ROOT

# readout-mutable module state (set in main from --readout); content = byte-identical default
READOUT = "content"                 # "content" (pooled) | "fslot" (shared forecast-token, stacked)
CACHE_KEY = POOLING                 # feature-cache pooling key: "content" | f"K{K}_H{H}"
POOL_TAG = POOLING                  # human label for filenames/records: "content" | "fslot"

SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber",
         "m4_hourly": "M4", "wind_farms_hourly": "WindFarms",
         "sg_carpark": "SG Carpark", "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}

METRICS = (("effective_rank", "effective rank  exp(H)"),
           ("spectral_entropy", "spectral entropy  H (nats)"),
           ("pc1_fraction", "PC1 variance fraction"))


def _derive_dirs():
    global SPEC_DIR, FIG_DIR
    root = V4_ROOT if READOUT == "fslot" else config.ID_OUT_DIR   # content stays in the v3 namespace
    SPEC_DIR = root / "spectral"
    FIG_DIR = SPEC_DIR / "figures"
    for d in (SPEC_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _split_name(tag, split):
    """PT-OOD caches use the *_rolling split names (disjoint from the legacy eval-only cache)."""
    return f"{split}_rolling" if tag in PT_OOD_TAGS else split


def _cache_file(tag, split):
    return _cache_path(_idf_prefix(tag), _split_name(tag, split), None, CACHE_KEY)


def _load_features(tag, split):
    """{layer: (N', 768)} from the on-disk feature cache — no model, no window rebuild.
    content: N' = N pooled windows (via _load_cache, the single-pooling cache).
    fslot:   the K native forecast-slot states per window (extract_kout_features's K-slot cache),
             STACKED to N' = N*K rows — every forecast slot is one point in the 768-d cloud the
             shared-head probe reads (the locked geometry decision, not mean-over-K/per-slot).
             Keys 0..NUM_LAYERS-1 = PRE-final-LN block slots; key NUM_LAYERS (=13) = the POST-final-LN
             slots (fslot_final, the native-head input) — the extra readout point beyond L12."""
    path = _cache_file(tag, split)
    if not path.exists():
        raise FileNotFoundError(
            f"no feature cache {path.name} — PT-ID caches come from the rolling runs, PT-OOD from "
            "the stage-2 GPU eval; run run_ptood_probing"
            + ("_ftok" if READOUT == "fslot" else "") + " first")
    if READOUT != "fslot":
        feats, _ = _load_cache(path)
        return feats
    d = np.load(path, allow_pickle=True)          # K-slot cache: content_L*/reg_L*/fslot_L*/*_final/y

    def _stack(arr, where):                       # (N, K, 768) -> (N*K, 768)
        arr = np.asarray(arr)
        assert arr.ndim == 3 and arr.shape[1] == K, (
            f"{path.name}: {where} shape {arr.shape}, expected (N, {K}, 768)")
        return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])

    feats = {i: _stack(d[f"fslot_L{i}"], f"fslot_L{i}") for i in range(NUM_LAYERS)}
    feats[NUM_LAYERS] = _stack(d["fslot_final"], "fslot_final")   # key 13 = post-final-LN native-head input
    return feats


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=config.REPO_ROOT if hasattr(config, "REPO_ROOT") else None,
                              check=True).stdout.strip()
    except Exception:
        return None


def analyze_dataset(tag, split, sample_size, n_subsamples, subsample_frac):
    """One spectral record: fixed sample of rows shared by every layer, full-sample point estimates
    + subsampling distributions, full normalized spectra retained."""
    feats = _load_features(tag, split)
    n_avail = feats[0].shape[0]
    rng = np.random.default_rng(SEED)
    if n_avail > sample_size:
        ids = np.sort(rng.choice(n_avail, size=sample_size, replace=False))
    else:
        ids = np.arange(n_avail)                 # fewer available than requested: use all
    layers = []
    for i in sorted(feats):                      # data-driven: 13 (content) or 14 (fslot, +post-LN)
        X = feats[i][ids].astype(np.float64)     # SAME rows at every layer
        m = spectral_metrics(X, return_spectrum=True)
        sub = subsample_metrics(X, n_subsamples=n_subsamples, frac=subsample_frac, seed=SEED)
        er = sub["effective_rank"]
        layers.append({
            "layer": i,
            "effective_rank": m["effective_rank"], "spectral_entropy": m["spectral_entropy"],
            "pc1_fraction": m["pc1_fraction"], "numerical_rank": m["numerical_rank"],
            "n_samples": m["n_samples"], "feature_dim": m["feature_dim"],
            "spectrum": m["spectrum"],
            "subsample_mean": er["mean"], "subsample_std": er["std"], "subsample_ci": er["ci"],
            "subsample_dists": sub,              # full distributions, all three metrics
        })
        print(f"    L{i:>2}  erank={m['effective_rank']:7.2f}  H={m['spectral_entropy']:.3f}  "
              f"pc1={m['pc1_fraction']:.3f}  numrank={m['numerical_rank']}")
    fslot_note = (f"fslot = K={K} native forecast-slot states STACKED to (N*K, 768); layers "
                  f"0..{NUM_LAYERS - 1} = PRE-final-LN block slots (Emb, L1..L{NUM_LAYERS - 1}), layer "
                  f"{NUM_LAYERS} = POST-final-LN slots (fslot_final, the native-head input); geometry "
                  "computed BEFORE the probe's internal StandardScaler")
    content_note = ("probe_input = cached content-pooled block outputs (L12 pre-final-LN); "
                    "geometry computed BEFORE the probe's internal StandardScaler")
    rec = {
        "dataset": tag, "domain_status": domain_status(tag),
        "backbone_condition": "pretrained", "backbone_seed": None,
        "readout": READOUT, "pooling": POOL_TAG, "representation_location": LOCATION,
        "forecast_slot_count": (K if READOUT == "fslot" else None),
        "split": split, "dataset_set": PTID_SET,
        "sample_size": int(len(ids)), "sample_size_requested": int(sample_size),
        "n_available": int(n_avail), "sample_ids": ids.tolist(),
        "subsample_protocol": {"method": "without_replacement", "n_subsamples": n_subsamples,
                               "frac": subsample_frac, "seed": SEED},
        "layers": layers,
        "provenance": {
            "feature_cache": _cache_file(tag, split).name,
            "computed": datetime.date.today().isoformat(), "git_commit": _git_commit(),
            "note": fslot_note if READOUT == "fslot" else content_note,
        },
    }
    out = SPEC_DIR / f"spectral__{tag}__{POOL_TAG}__{LOCATION}__{split}.json"
    json.dump(rec, open(out, "w"), indent=2)
    print(f"  [saved] {out.name}")
    return rec


def _point_labels(n):
    """x-axis labels for n readout points: Emb, L1..L12, then the post-final-LN point (L12+LN)
    as the 14th (fslot only). Content stays 13 → Emb..L12."""
    return (["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + ["L12+LN"])[:n]


def make_figures(records, split):
    """Plots A/B/C: one per metric, all datasets overlaid, full-sample point + subsample band.
    Effective rank shown as the actual effective dimension (not normalized)."""
    n = len(records[0]["layers"])            # data-driven: 13 (content) or 14 (fslot, +post-LN)
    x = np.arange(n)
    readout_lbl = ("shared forecast-token (stacked slots)" if READOUT == "fslot"
                   else f"{POOL_TAG}-pooled")
    for key, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for rec in records:
            v = np.array([L[key] for L in rec["layers"]])
            sub = [L["subsample_dists"][key] for L in rec["layers"]]
            lo = np.array([s["ci"][0] for s in sub]); hi = np.array([s["ci"][1] for s in sub])
            line, = ax.plot(x, v, "o-", ms=3.5, label=SHORT.get(rec["dataset"], rec["dataset"]))
            ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15)
        ax.set_xticks(x); ax.set_xticklabels(_point_labels(n))
        ax.set_xlabel("layer"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split('  ')[0]} by layer — {readout_lbl} probe input, "
                     f"{split} windows\nband = subsampling (without replacement) 95% interval",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"{key}_by_layer__{POOL_TAG}__{LOCATION}__{split}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  [saved] {out.name}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--readout", default="content", choices=("content", "fslot"),
                   help="content = pooled (v3 namespace); fslot = shared forecast-token stacked "
                        "(N*K, 768) (v4 ext_v4_future_tokens namespace)")
    p.add_argument("--datasets", nargs="*", default=None, choices=list(ALL_TAGS),
                   help="default: every dataset whose feature cache exists")
    p.add_argument("--split", default="train", choices=("train", "val", "test"),
                   help="train (default) = the split the probes are fit on")
    p.add_argument("--repr-sample-size", type=int, default=4096)
    p.add_argument("--repr-subsamples", type=int, default=200)
    p.add_argument("--repr-subsample-frac", type=float, default=0.8)
    p.add_argument("--location", default=LOCATION, choices=("probe_input", "post_final_ln"))
    return p.parse_args(argv)


def main():
    args = _parse_args()
    if args.location == "post_final_ln":
        raise SystemExit(
            "post_final_ln states are not cached for the rolling set (the block hooks capture "
            "pre-final-LN outputs; only extract_kout_features stores post-LN states, and only "
            "for K-pass runs). Requires a short GPU extraction pass — not silently substituted.")
    global READOUT, CACHE_KEY, POOL_TAG
    READOUT = args.readout
    CACHE_KEY = f"K{K}_H{H}" if READOUT == "fslot" else POOLING
    POOL_TAG = "fslot" if READOUT == "fslot" else POOLING
    config.set_dataset_set(PTID_SET)     # roster + rolling windows + cache namespace (outputs by readout)
    _derive_dirs()
    tags = args.datasets
    if tags is None:
        tags = [t for t in ALL_TAGS if _cache_file(t, args.split).exists()]
        skipped = [t for t in ALL_TAGS if t not in tags]
        if skipped:
            print(f"[skip, no cache yet] {skipped}")
    print(f"[run_spectral] readout={READOUT}  {len(tags)} datasets  split={args.split}  "
          f"pool_tag={POOL_TAG}  cache_key={CACHE_KEY}  location={LOCATION}  "
          f"sample<={args.repr_sample_size}  subsamples={args.repr_subsamples}@"
          f"{args.repr_subsample_frac}")
    records = []
    for tag in tags:
        print(f"\n[{SHORT.get(tag, tag)}] ({domain_status(tag)['pretraining']})")
        records.append(analyze_dataset(tag, args.split, args.repr_sample_size,
                                       args.repr_subsamples, args.repr_subsample_frac))
    if records:
        make_figures(records, args.split)


if __name__ == "__main__":
    main()
