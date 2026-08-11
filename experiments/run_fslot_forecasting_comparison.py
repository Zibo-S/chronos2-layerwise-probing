"""ACTUAL forecasting comparison on ORIGINAL-scale targets — 7 methods, 4 PT-ID datasets (§9).

Where the tunnel/transfer drivers score probes in Chronos-2's arcsinh quantile-loss currency, this
driver answers the plain forecasting question: un-transformed to raw units, on the SAME test windows,
how do the trained probe readouts compare to the naive baselines and to the pretrained native model?

Datasets: the four PT-ID / Probe-ID sets (Electricity, Uber, M4, WindFarms). Methods, all evaluated on
identical test windows / horizon / seasonal period:

  1. last-value              — raw last context value, repeated H steps (point baseline)
  2. seasonal-naive          — raw last season tiled to H (m=24; point baseline)
  3. shared-linear @entrance — frozen linear fslot probe at its VALIDATION tunnel entrance (l_start)
  4. shared-linear @L12+LN   — the same probe at the post-final-LN reference point
  5. native-MLP @entrance    — frozen native-structure MLP fslot probe at ITS validation tunnel entrance
  6. native-MLP @L12+LN      — the same MLP probe at the reference point
  7. native-Chronos-2        — the pretrained backbone + its original trained native output head

Both entrances come from VALIDATION only (each family's first-crossing tunnel l_start from the
PT-ID mean-validation curve); neither is chosen on test. The probes are freshly-trained readouts —
the native-MLP is NOT the pretrained native head. The three probe seeds are averaged per-window
(§9.3), never as a ±1 std "CI".

Metrics (ORIGINAL scale, canonical inverse transform mu + s*sinh(z), leakage-free context stats):
  * MASE  — primary, scale-normalized (in-context seasonal-naive denom m=24, = run_id_forecasting)
  * median MAE — secondary point metric
  * WQL   — supplementary probabilistic metric (raw weighted quantile loss). Computed from each
            probe's OWN quantiles; for the point baselines the point is repeated across levels and
            flagged point_as_quantiles=True (they do NOT produce calibrated quantiles). Native WQL
            needs a multi-quantile native pass (--native-wql; else recorded null).

Uncertainty (§9.3): per-window metric per seed -> averaged across seeds -> ONE shared paired
series-level cluster bootstrap (probing.stats; same resample for every method, so all pairwise
differences are paired on identical target series). Reports mean, 95% CI, paired diff vs
seasonal-naive and vs native-Chronos-2.

Reuses run_id_forecasting (inverse transform, seasonal denom, cached native median) + the frozen
probes/tunnels from run_ptood_probing_ftok + run_fslot_transfer. Needs the linear AND native_mlp
PT-ID checkpoints + tunnels on disk (run_ptood_probing_ftok --fit-ptid/--tunnels-only for BOTH
families). Native forecasting hits the GPU on a cold cache; probe scoring is CPU over warm fslot caches.

Run (USER submits; GPU only if the native cache is cold):
    python -m experiments.run_fslot_forecasting_comparison                 # MASE + median MAE + probe/naive WQL
    python -m experiments.run_fslot_forecasting_comparison --native-wql    # + native WQL (extra GPU pass)
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import CACHE_DIR
from probing.probes import (QUANTILE_SETS, _apply_shared_head, _slot_transform, median_index,
                            validate_quantiles)
from probing.stats import ci_bounds, cluster_bootstrap_apply, cluster_bootstrap_counts
from probing.id_data import build_windows
from probing.tunnel import PT_ID_TAGS
from experiments.run_id_forecasting import (M_SEASON, _ctx_stats, _mase_denominator,
                                            native_median_forecast)
from experiments.run_ptood_probing_ftok import (C, H, K, LAYER_LABELS, OUT_ROOT, PROBE_FAMILIES,
                                                RUN_SEEDS, RUNS_TAG, SHORT, _fslot_feats)

BOOT_B = 5000
SEED = config.SEED
OUT_DIR = OUT_ROOT / "forecasting_comparison"
REF_LAYER = len(LAYER_LABELS) - 1                 # L12+LN post-final-LN reference (index 13)
# fixed method order (rows in the table + bootstrap; last_value stays computed for the table)
METHODS = ["last_value", "seasonal_naive", "shared_linear@entrance", "shared_linear@ref",
           "native_mlp@entrance", "native_mlp@ref", "native_chronos2"]
# figure omits last_value: its persistence MASE is several× the others and stretches the y-axis so
# the interesting methods collapse together. It remains in the CSV/JSON table.
FIGURE_METHODS = [m for m in METHODS if m != "last_value"]
PROBE_METHODS = {"shared_linear@entrance": ("shared_linear", "entrance"),
                 "shared_linear@ref": ("shared_linear", "ref"),
                 "native_mlp@entrance": ("native_mlp", "entrance"),
                 "native_mlp@ref": ("native_mlp", "ref")}


def _derive_dirs():
    for d in (OUT_DIR, OUT_DIR / "figures", OUT_DIR / "tables"):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# tunnels: each family's validation-selected ENTRANCE layer (l_start), never test
# --------------------------------------------------------------------------- #
def _entrance_layer(family_name, tag, qset):
    """The family's first-crossing tunnel entrance (l_start) for `tag`, from the PT-ID
    mean-VALIDATION curve. Fail loud if the tunnel record is missing (run --tunnels-only)."""
    p = PROBE_FAMILIES[family_name].tunnel_path(tag, qset)
    if not p.exists():
        raise FileNotFoundError(f"missing {family_name} tunnel {p} — run "
                                f"`run_ptood_probing_ftok --probe-family {family_name} --tunnels-only`")
    return int(json.load(open(p))["l_start"])


# --------------------------------------------------------------------------- #
# raw-scale forecasts
# --------------------------------------------------------------------------- #
def _raw_future(w, mu, s):
    """The ORIGINAL-scale future target y (mu + s*sinh(z)) — the exact inverse of the arcsinh label
    transform, using each window's own context stats (leakage-free)."""
    return mu[:, None] + s[:, None] * np.sinh(np.asarray(w["Y_test_traj"], np.float64))


def _last_value_raw(X_test):
    return np.repeat(np.asarray(X_test, np.float64)[:, -1:], H, axis=1)          # (n, H)


def _seasonal_naive_raw(X_test, m=M_SEASON):
    """Raw seasonal-naive: tile the last m context values across H (y_hat[h] = x[C-m + h mod m])."""
    X64 = np.asarray(X_test, np.float64)
    idx = X64.shape[1] - m + (np.arange(H) % m)
    return X64[:, idx]                                                            # (n, H)


def _probe_quantiles_raw(fitted, feats, layer, mu, s, quantiles, device="cpu"):
    """Full ORIGINAL-scale quantile forecast (n, Q, H) of a FROZEN probe at one layer: scale the
    slot features with the probe's stored scaler, apply the shared head, inverse-arcsinh each
    quantile. sinh is monotone, so quantile order is preserved. Never trains."""
    Q, P = len(quantiles), int(fitted[layer]["output_patch_size"])
    m = fitted[layer].get("linear") or fitted[layer]["head"]
    m.eval()
    sc = fitted[layer]["scaler"]
    Xt = torch.as_tensor(_slot_transform(sc, feats[layer]), dtype=torch.float32, device=device)
    with torch.no_grad():
        z = _apply_shared_head(m, Xt, Q, P, H).cpu().numpy().astype(np.float64)   # (n, Q, H) arcsinh
    return mu[:, None, None] + s[:, None, None] * np.sinh(z)


# --------------------------------------------------------------------------- #
# per-window metrics (ORIGINAL scale) — row means feed the cluster bootstrap
# --------------------------------------------------------------------------- #
def _mase_pw(y_raw, yhat_raw, denom):
    return (np.abs(y_raw - yhat_raw) / denom).mean(axis=1)                        # (n,)


def _mae_pw(y_raw, yhat_median_raw):
    return np.abs(y_raw - yhat_median_raw).mean(axis=1)                           # (n,)


def _wql_pw_parts(y_raw, quant_raw, quantiles):
    """Per-window weighted-quantile-loss NUMERATOR and DENOMINATOR (kept separate so the bootstrap
    forms the ratio WQL = sum(num)/sum(den) inside each replicate). num = 2*sum_{q,H} pinball;
    den = sum_H |y|. quant_raw: (n, Q, H)."""
    y = y_raw[:, None, :]                                                         # (n, 1, H)
    tau = np.asarray(quantiles, np.float64)[None, :, None]                        # (1, Q, 1)
    pinball = np.where(y >= quant_raw, tau * (y - quant_raw), (1 - tau) * (quant_raw - y))
    num = 2.0 * pinball.sum(axis=(1, 2))                                          # (n,)
    den = np.abs(y_raw).sum(axis=1)                                              # (n,)
    return num, den


# --------------------------------------------------------------------------- #
# series-level cluster bootstrap (one shared resample per dataset -> paired methods)
# --------------------------------------------------------------------------- #
def _series_group(sid):
    _uniq, inv = np.unique(np.asarray(sid, np.int64), return_inverse=True)
    return _uniq.size, inv


def _per_series(vec, inv, S):
    out = np.zeros(S, np.float64)
    np.add.at(out, inv, np.asarray(vec, np.float64))
    return out


def _boot_mean(M, vec, inv, S):
    """Bootstrap distribution (B,) of a per-window metric's MEAN, resampling whole series with counts M.
    cluster_bootstrap_apply wants per_series_sum as (S, 1) and returns (B, 1) -> squeeze."""
    ssum = _per_series(vec, inv, S)[:, None]                          # (S, 1)
    scount = _per_series(np.ones_like(vec, np.float64), inv, S)       # (S,) window counts
    return cluster_bootstrap_apply(M, ssum, scount)[:, 0]


def _boot_ratio(M, num, den, inv, S):
    """Bootstrap distribution (B,) of a ratio metric (WQL = sum num / sum den) under the same resample:
    pass the per-series numerator as the 'sum' and the per-series denominator as the 'count'."""
    nsum = _per_series(num, inv, S)[:, None]                          # (S, 1)
    dsum = _per_series(den, inv, S)                                   # (S,) denominator per series
    return cluster_bootstrap_apply(M, nsum, dsum)[:, 0]


# --------------------------------------------------------------------------- #
# one dataset: build windows once, forecast every method, bootstrap paired
# --------------------------------------------------------------------------- #
def evaluate_dataset(tag, qset, quantiles, native_wql, device):
    print(f"\n[forecasting] {SHORT[tag]} ({qset})")
    w = build_windows(tag)                                    # SAME test windows for every method
    X_test = np.asarray(w["X_test"], np.float64)
    n = int(w["meta"]["n_test"])
    mu, s = _ctx_stats(X_test, w["meta"]["sigma_eps"])
    y_raw = _raw_future(w, mu, s)
    denom = np.maximum(_mase_denominator(X_test), 1e-8)[:, None]
    qmid = median_index(quantiles)
    S, inv = _series_group(w["series_test"])
    M = cluster_bootstrap_counts(S, BOOT_B, SEED)            # ONE shared resample -> paired methods
    print(f"  windows={n}  series={S}  entrance layers from validation tunnels:")

    feats = _fslot_feats(tag, "test", w["X_test"], w["y_test"])     # shared fslot cache (both families)
    ent = {fam: _entrance_layer(fam, tag, qset) for fam in ("shared_linear", "native_mlp")}
    layer_of = {"entrance": None, "ref": REF_LAYER}
    for fam in ("shared_linear", "native_mlp"):
        print(f"    {fam:>13}: entrance {LAYER_LABELS[ent[fam]]}  ref {LAYER_LABELS[REF_LAYER]}")

    # ---- per-window metrics per method (probes averaged over the 3 seeds) ----
    pw_mase, pw_mae, pw_wql = {}, {}, {}
    method_layer = {}

    yhat_lv = _last_value_raw(X_test)
    pw_mase["last_value"] = _mase_pw(y_raw, yhat_lv, denom); pw_mae["last_value"] = _mae_pw(y_raw, yhat_lv)
    yhat_sn = _seasonal_naive_raw(X_test)
    pw_mase["seasonal_naive"] = _mase_pw(y_raw, yhat_sn, denom)
    pw_mae["seasonal_naive"] = _mae_pw(y_raw, yhat_sn)
    # point baselines: repeat the point across quantiles for a (flagged) WQL
    for name, yhat in (("last_value", yhat_lv), ("seasonal_naive", yhat_sn)):
        q_rep = np.repeat(yhat[:, None, :], len(quantiles), axis=1)
        pw_wql[name] = _wql_pw_parts(y_raw, q_rep, quantiles)

    for method, (fam, which) in PROBE_METHODS.items():
        layer = ent[fam] if which == "entrance" else REF_LAYER
        method_layer[method] = layer
        mase_seeds, mae_seeds, num_seeds, den_seeds = [], [], [], []
        for seed in RUN_SEEDS:
            fitted = PROBE_FAMILIES[fam].load_ckpt(tag, qset, seed, device=device)
            qr = _probe_quantiles_raw(fitted, feats, layer, mu, s, quantiles, device=device)
            med = qr[:, qmid, :]
            mase_seeds.append(_mase_pw(y_raw, med, denom)); mae_seeds.append(_mae_pw(y_raw, med))
            num, den = _wql_pw_parts(y_raw, qr, quantiles)
            num_seeds.append(num); den_seeds.append(den)
            del fitted
        pw_mase[method] = np.mean(mase_seeds, axis=0)        # average per-window metric over seeds
        pw_mae[method] = np.mean(mae_seeds, axis=0)
        pw_wql[method] = (np.mean(num_seeds, axis=0), np.mean(den_seeds, axis=0))

    native = native_median_forecast(tag, w["X_test"], H).astype(np.float64)      # raw median (cached)
    pw_mase["native_chronos2"] = _mase_pw(y_raw, native, denom)
    pw_mae["native_chronos2"] = _mae_pw(y_raw, native)
    if native_wql:
        qr = _native_quantiles_raw(tag, w["X_test"], quantiles, device)
        pw_wql["native_chronos2"] = _wql_pw_parts(y_raw, qr, quantiles)

    # ---- bootstrap every method under the SAME resample; paired diffs vs seasonal + native ----
    boot_mase, boot_mae, boot_wql = {}, {}, {}
    for method in METHODS:
        boot_mase[method] = _boot_mean(M, pw_mase[method], inv, S)
        boot_mae[method] = _boot_mean(M, pw_mae[method], inv, S)
        if method in pw_wql:
            num, den = pw_wql[method]
            boot_wql[method] = _boot_ratio(M, num, den, inv, S)

    rows = []
    for method in METHODS:
        layer = method_layer.get(method)
        wql_pt = float((pw_wql[method][0].sum() / max(pw_wql[method][1].sum(), 1e-12))) \
            if method in pw_wql else None
        rows.append({
            "dataset": tag, "method": method,
            "layer": (None if layer is None else int(layer)),
            "layer_label": (None if layer is None else LAYER_LABELS[layer]),
            "validation_selection": ("tunnel_entrance(l_start)/validation" if "@entrance" in method
                                     else ("post_final_ln_reference" if "@ref" in method else "n/a")),
            "mase": round(float(pw_mase[method].mean()), 6),
            "mase_ci_lo": round(float(ci_bounds(boot_mase[method])[0]), 6),
            "mase_ci_hi": round(float(ci_bounds(boot_mase[method])[1]), 6),
            "median_mae": round(float(pw_mae[method].mean()), 6),
            "mae_ci_lo": round(float(ci_bounds(boot_mae[method])[0]), 6),
            "mae_ci_hi": round(float(ci_bounds(boot_mae[method])[1]), 6),
            "wql": (None if wql_pt is None else round(wql_pt, 6)),
            "wql_ci_lo": (round(float(ci_bounds(boot_wql[method])[0]), 6) if method in boot_wql else None),
            "wql_ci_hi": (round(float(ci_bounds(boot_wql[method])[1]), 6) if method in boot_wql else None),
            "point_as_quantiles": method in ("last_value", "seasonal_naive"),
            "mase_minus_seasonal": round(float((boot_mase[method] - boot_mase["seasonal_naive"]).mean()), 6),
            "mase_minus_seasonal_ci_lo": round(float(ci_bounds(boot_mase[method]
                                              - boot_mase["seasonal_naive"])[0]), 6),
            "mase_minus_seasonal_ci_hi": round(float(ci_bounds(boot_mase[method]
                                              - boot_mase["seasonal_naive"])[1]), 6),
            "mase_minus_native": round(float((boot_mase[method] - boot_mase["native_chronos2"]).mean()), 6),
            "mase_minus_native_ci_lo": round(float(ci_bounds(boot_mase[method]
                                            - boot_mase["native_chronos2"])[0]), 6),
            "mase_minus_native_ci_hi": round(float(ci_bounds(boot_mase[method]
                                            - boot_mase["native_chronos2"])[1]), 6),
            "n_windows": n, "n_series": S, "seasonal_m": M_SEASON, "run_seeds": " ".join(map(str, RUN_SEEDS)),
        })
        m = rows[-1]
        print(f"    {method:>22}  MASE {m['mase']:.3f} [{m['mase_ci_lo']:.3f},{m['mase_ci_hi']:.3f}]  "
              f"MAE {m['median_mae']:.3f}")
    return rows, ent


def _native_quantiles_raw(tag, X_test, quantiles, device):
    """Multi-quantile native Chronos-2 forecast (n, Q, H) in RAW units (for native WQL). GPU on a cold
    cache; cached to features_cache alongside the median. Separate key so it never clobbers the median
    cache. The context-tail guard fails loud on a re-windowed dataset."""
    from probing.extraction import _idf_prefix, get_pipeline
    levels = [float(q) for q in quantiles]
    cache = CACHE_DIR / f"{_idf_prefix(tag)}__test__native_q{len(levels)}_H{H}.npz"
    X_test = np.asarray(X_test, dtype=np.float32)
    if cache.exists():
        d = np.load(cache)
        if d["ctx_tail"].shape[0] == len(X_test) and np.allclose(d["ctx_tail"], X_test[:, -8:]):
            print(f"  [cache HIT]  {cache.name}")
            return d["quant"].astype(np.float64)
        raise RuntimeError(f"stale native-quantile cache {cache.name}: contexts changed — delete it")
    pipeline, _ = get_pipeline()
    print(f"  [native] {len(X_test)} windows x {len(levels)} quantiles (H={H})")
    qt, _mean = pipeline.predict_quantiles(list(X_test), prediction_length=H, quantile_levels=levels)
    quant = np.stack([q.reshape(H, len(levels)).cpu().numpy() for q in qt]).transpose(0, 2, 1)  # (n,Q,H)
    np.savez(cache, quant=quant.astype(np.float32), ctx_tail=X_test[:, -8:])
    print(f"  [saved]      {cache.name}  shape={quant.shape}")
    return quant.astype(np.float64)


# --------------------------------------------------------------------------- #
# outputs — tidy table, grouped MASE figure, WQL/MAE supplement, auto-summary
# --------------------------------------------------------------------------- #
def _dump(stem, rows):
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(f"{stem}.json", "w"), indent=2)


def make_mase_figure(rows, qset):
    datasets = list(PT_ID_TAGS)
    by = {(r["dataset"], r["method"]): r for r in rows}
    x = np.arange(len(datasets)); w = 0.14
    fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 4, 5))
    for k, method in enumerate(FIGURE_METHODS):
        vals = [by[(d, method)]["mase"] for d in datasets]
        lo = [by[(d, method)]["mase"] - by[(d, method)]["mase_ci_lo"] for d in datasets]
        hi = [by[(d, method)]["mase_ci_hi"] - by[(d, method)]["mase"] for d in datasets]
        ax.bar(x + (k - (len(FIGURE_METHODS) - 1) / 2) * w, vals, w, yerr=[lo, hi], capsize=2, label=method)
    ax.axhline(1.0, color="0.4", ls=":", lw=1, label="MASE = 1 (seasonal-naive scale)")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in datasets])
    ax.set_ylabel("MASE (original scale; lower = better)")
    ax.set_title(f"Original-scale forecasting: {len(FIGURE_METHODS)} methods x 4 PT-ID datasets  "
                 f"[{qset}, {RUNS_TAG}]  (last-value baseline omitted for scale; see table)\n"
                 "probe @entrance = validation tunnel l_start; error bars = 95% paired cluster bootstrap",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = OUT_DIR / "figures" / f"mase_by_method__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  [saved] {out.name}")


def _get(rows, dataset, method, field):
    for r in rows:
        if r["dataset"] == dataset and r["method"] == method:
            return r[field]
    return None


def write_summary(rows, qset):
    """Automatic findings (§9.4): MLP-vs-linear at ref and at entrances; entrance-vs-final per family;
    native-vs-MLP; native-vs-seasonal; WindFarms seasonal dominance; whether the MLP removes the
    late-layer WindFarms degradation."""
    lines = [f"# Forecasting comparison summary ({qset}, {RUNS_TAG})", ""]
    for d in PT_ID_TAGS:
        sn = _get(rows, d, "seasonal_naive", "mase"); nat = _get(rows, d, "native_chronos2", "mase")
        le, lr = _get(rows, d, "shared_linear@entrance", "mase"), _get(rows, d, "shared_linear@ref", "mase")
        me, mr = _get(rows, d, "native_mlp@entrance", "mase"), _get(rows, d, "native_mlp@ref", "mase")
        lines += [
            f"## {SHORT[d]}",
            f"- MLP vs linear @L12+LN:   ΔMASE = {mr - lr:+.3f}  (MLP {'better' if mr < lr else 'worse'})",
            f"- MLP vs linear @entrance: ΔMASE = {me - le:+.3f}  (MLP {'better' if me < le else 'worse'})",
            f"- entrance vs final: linear {lr - le:+.3f}, MLP {mr - me:+.3f}  "
            f"(negative = entrance already as good/better)",
            f"- native vs MLP@ref: {mr - nat:+.3f}  (native better by this much MASE)",
            f"- native vs seasonal-naive: {nat - sn:+.3f}  "
            f"({'native beats seasonal' if nat < sn else 'seasonal DOMINATES native'})",
        ]
        if d == "wind_farms_hourly":
            lines += [
                f"- WindFarms seasonal dominance: {'YES — seasonal-naive <= native' if nat >= sn else 'no'}",
                f"- MLP removes linear's late-layer degradation? linear entrance→ref {lr - le:+.3f} vs "
                f"MLP {mr - me:+.3f} — {'YES' if (lr - le) > 0 >= (mr - me) else 'not clearly'}",
            ]
        lines.append("")
    txt = "\n".join(lines)
    (OUT_DIR / f"summary__{qset}__{RUNS_TAG}.md").write_text(txt)
    print("\n" + txt)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quantile-set", default="q9", choices=sorted(QUANTILE_SETS))
    p.add_argument("--native-wql", action="store_true",
                   help="also compute native-Chronos-2 WQL (extra multi-quantile GPU pass)")
    p.add_argument("--datasets", nargs="*", default=list(PT_ID_TAGS), choices=list(PT_ID_TAGS))
    return p.parse_args(argv)


def main():
    args = _parse_args()
    config.set_dataset_set("extended_v3_rolling")
    _derive_dirs()
    quantiles = validate_quantiles(QUANTILE_SETS[args.quantile_set])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_fslot_forecasting_comparison] qset={args.quantile_set}  {RUNS_TAG}  K={K} H={H}  "
          f"native_wql={args.native_wql}  device={device}")
    rows = []
    for tag in args.datasets:
        r, _ent = evaluate_dataset(tag, args.quantile_set, quantiles, args.native_wql, device)
        rows += r
    _dump(str(OUT_DIR / "tables" / f"forecasting_comparison__{args.quantile_set}__{RUNS_TAG}"), rows)
    make_mase_figure(rows, args.quantile_set)
    write_summary(rows, args.quantile_set)
    print(f"\n  [done] {len(rows)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
