"""HEADLINE TimesFM-3 experiment: layer-wise probing of the LAST REAL CONTEXT TOKEN, Q=9.

    At each layer, how linearly decodable is the native C=512 -> H=64 forecast from the
    EXACT context token the native TimesFM-3 forecasting head reads?

Each model is probed where its own forecasting head reads:

    Chronos-2  : K=4 native forecast-slot states for H=64   (shared linear head over slots)
    TimesFM-3  : ONE last-context state, token index 15     (this driver)

so the probe geometry respects the architectural difference instead of imposing one artificial
shared scheme across models.

    per representation point l in {Emb, L1..L20}:
        ONE independent  W_l = Linear(1280, 64*9)  on  h_{l,15} in R^1280
        trained + reported with the FULL native Q=9 mean pinball loss
        (at L20 that is exactly the native output head's hypothesis class)

    5% forecasting-tunnel entrance, on VALIDATION only:
        l_tun = min { l : L_val(l) <= 1.05 * L_val(L20) }      (2%/1%/10% also saved)

The 16-prefix / shared-origin run (experiments/run_timesfm3_probing.py, probing/timesfm3_probes.py,
job_timesfm3_probing.sh, cache tag tfm3-prefix-v1) is an ABLATION and is untouched -- separate
entry point, separate cache namespace, separate results namespace. No Chronos-2 file is modified.

Reused unchanged: id_data.build_windows (windows, within-series split, series ids, seed, budget);
probes.WD_GRID_V2 + validate_quantiles/median_index + the 80/20 carve + AdamW convention;
stats' series-level cluster bootstrap; tunnel.tunnel_start (the project's 5% first-crossing rule);
and run_timesfm3_probing's mase_denominator / per_window_mase / cluster_ci, imported so both
TimesFM-3 lines compute MASE and CIs from ONE definition.

    WHERE TO RUN: loads a 330M model and fits 21 x |wd grid| probes. sbatch/salloc on Narval --
    NEVER a login node. The contract tests (python -m tests.test_timesfm3_last_token_probe)
    ARE login-node safe.

    python -m experiments.run_timesfm3_last_token_probing --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ONE definition of MASE / cluster CIs for both TimesFM-3 lines (read-only import of the
# ablation driver; those helpers are geometry- and quantile-agnostic). The MASE denominator is
# byte-identical to run_id_forecasting._mase_denominator, which run_native_head_adapter uses for
# all seven Chronos-2 datasets.
from experiments.run_timesfm3_probing import (MASE_DEN_FLOOR, M_SEASON, cluster_ci,  # noqa: E402
                                              mase_denominator, per_window_mase)
from probing.config import SEED as CANON_SEED  # noqa: E402
from probing.timesfm3_last_token_probes import (WD_GRID_LAST_TOKEN,  # noqa: E402
                                                WD_NULL_BASELINE)
from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS  # noqa: E402  (the roster's source of truth)

# --------------------------------------------------------------------------- #
# dataset suites
# --------------------------------------------------------------------------- #
# The HEADLINE suite is the Chronos-2 paper's seven datasets, taken from the SAME sources the
# Chronos-2 drivers use -- nothing is re-declared or re-derived here:
#     roster        probing/tunnel.py   PT_ID_TAGS + PT_OOD_TAGS
#     PT-ID windows id_data.build_windows under dataset set "extended_v3_rolling"
#                   (-> _build_rolling_windows, budgets (1394, 262, 262), seed 0)
#     PT-OOD windows id_data.build_ood_rolling_windows(tag, C=512, H=64, seed=SEED)
#     display names experiments/make_id_paper_figures.py:314  SHORT
#     PT-ID/PT-OOD  experiments/run_native_head_adapter.py:72  DATASET_KIND
# The dispatch below is the exact twin of run_native_head_adapter._windows() (== 
# run_ft_specialization.target_windows()), so both models see the same x[1:512] -> x[513:576].
PTID_SET = "extended_v3_rolling"
PAPER7 = tuple(PT_ID_TAGS) + tuple(PT_OOD_TAGS)
KIND = {**{t: "PT-ID" for t in PT_ID_TAGS}, **{t: "PT-OOD" for t in PT_OOD_TAGS}}
SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber TLC",
         "m4_hourly": "M4", "wind_farms_hourly": "Wind Farms", "sg_carpark": "SG Carpark",
         "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}
# committed Chronos-2 artifacts for those seven datasets: per-window test series/cluster ids and
# the window counts. This is what makes "same windows" a CHECK rather than a claim.
CHRONOS_REF_ROOT = REPO_ROOT / "results" / "ext_v5_native_head_adapter"


def suite_tags(suite: str) -> list[str]:
    """The dataset roster of a suite. ``paper7`` = the Chronos-2 headline seven, PT-ID first."""
    if suite == "paper7":
        return list(PAPER7)
    from probing.id_data import ID_DATASET_SPECS
    if suite in ID_DATASET_SPECS:
        return list(ID_DATASET_SPECS[suite])
    raise SystemExit(f"unknown suite {suite!r}; known: 'paper7' (the headline seven) or any "
                     f"id_data.ID_DATASET_SPECS key {sorted(ID_DATASET_SPECS)}")


def windows_for(tag: str, suite: str, args):
    """Windows for one dataset -- the EXACT twin of run_native_head_adapter._windows().

    paper7:  PT-OOD -> build_ood_rolling_windows(tag, C=512, H=64, seed=SEED)
             PT-ID  -> config.set_dataset_set("extended_v3_rolling"); build_windows(tag)
    other :  the legacy auto-split path (build_windows with this driver's C/H/stride/seed).

    Non-default C/H/seed are REFUSED for paper7: the committed Chronos-2 windows are C=512,
    H=64, seed=0, and silently building different ones would break the whole point.
    """
    from probing import config
    from probing.id_data import build_ood_rolling_windows, build_windows
    if suite != "paper7":
        config.set_dataset_set(suite)
        return build_windows(tag, C=args.context_len, H=args.horizon, stride=args.stride,
                             seed=args.seed)
    bad = {k: v for k, v in (("--context-len", (args.context_len, 512)),
                             ("--horizon", (args.horizon, 64)),
                             ("--seed", (args.seed, CANON_SEED))) if v[0] != v[1]}
    if bad:
        raise SystemExit(
            "the paper7 suite reproduces the COMMITTED Chronos-2 windows, which are C=512, "
            f"H=64, seed={CANON_SEED}; refusing " +
            ", ".join(f"{k}={v[0]} (must be {v[1]})" for k, v in bad.items()))
    if tag in PT_OOD_TAGS:
        # evaluation-only rosters: the cluster id (carpark / station / metric-query) is the
        # bootstrap unit and travels in series_*, exactly as in the Chronos-2 run.
        return build_ood_rolling_windows(tag, C=512, H=64, seed=CANON_SEED)
    config.set_dataset_set(PTID_SET)
    return build_windows(tag)


# --------------------------------------------------------------------------- #
# window parity with the committed Chronos-2 run
# --------------------------------------------------------------------------- #

def chronos_reference(tag: str, root=CHRONOS_REF_ROOT):
    """The committed Chronos-2 artifacts for ``tag``: window counts + per-window test series ids.

    configs/native_head_adapter__<tag>__config.json  -> n_train / n_val / n_test / C / H / kind
    bootstrap_inputs/native_head_adapter__<tag>.npz  -> series_test, one id per test window
    Returns None when the dataset is not part of that committed run (e.g. KDD / pedestrian).
    """
    cfg = Path(root) / "configs" / f"native_head_adapter__{tag}__config.json"
    npz = Path(root) / "bootstrap_inputs" / f"native_head_adapter__{tag}.npz"
    if not (cfg.exists() and npz.exists()):
        return None
    c = json.loads(cfg.read_text())
    with np.load(npz, allow_pickle=False) as z:
        sid = np.asarray(z["series_test"], np.int64)
    return {"config": c, "series_test": sid,
            "paths": {"config": str(cfg.relative_to(REPO_ROOT)),
                      "bootstrap_inputs": str(npz.relative_to(REPO_ROOT))}}


def window_identity(tag, w):
    """The reportable identity of one dataset's windows (counts + the first few identifiers)."""
    sid = np.asarray(w["series_test"], np.int64)
    origins = (w["meta"].get("origins", {}) or {}).get("test")
    ids = ([f"s{int(a)}@t{int(b)}" for a, b in zip(sid[:6], origins[:6])] if origins
           else [f"s{int(a)}" for a in sid[:6]])
    return {"dataset": tag, "short": SHORT.get(tag, tag), "kind": KIND.get(tag, "unclassified"),
            "split_mode": w["meta"].get("split_mode"),
            "n_train_windows": int(len(w["X_train"])),
            "n_val_windows": int(len(w["X_val"])) if "X_val" in w else None,
            "n_test_windows": int(len(sid)),
            "n_test_series": int(len(np.unique(sid))),
            "cluster_unit": w["meta"].get("cluster_unit", "series"),
            "first_test_identifiers": ids,
            "test_series_first6": [int(x) for x in sid[:6]]}


def assert_window_parity(tag, w, ref, *, strict=True):
    """TimesFM-3 must receive the SAME x[1:512] -> x[513:576] windows as Chronos-2.

    Checks, against the committed artifacts: C, H, seasonal m, the train/val/test window COUNTS,
    and -- the decisive one -- the per-window test series/cluster ids ELEMENT-WISE. Raises on any
    mismatch unless ``strict=False`` (then the failures are reported and carried in the record).
    """
    ident = window_identity(tag, w)
    if ref is None:
        ident.update(chronos_parity="no_committed_reference", parity_ok=None)
        return ident
    c, m = ref["config"], w["meta"]
    sid = np.asarray(w["series_test"], np.int64)
    rsid = ref["series_test"]
    fails = []
    for name, got, want in (("C", m.get("C"), c["C"]), ("H", m.get("H"), c["H"]),
                            ("seasonal_m", m.get("m_season"), c["seasonal_m"]),
                            ("n_train", int(len(w["X_train"])), c["n_train"]),
                            ("n_test", int(len(sid)), c["n_test"]),
                            ("kind", KIND.get(tag), c["kind"])):
        if got != want:
            fails.append(f"{name}: TimesFM={got!r} vs Chronos-2={want!r}")
    if "X_val" in w and int(len(w["X_val"])) != c["n_val"]:
        fails.append(f"n_val: TimesFM={len(w['X_val'])} vs Chronos-2={c['n_val']}")
    if sid.shape != rsid.shape:
        fails.append(f"series_test shape: {sid.shape} vs {rsid.shape}")
    elif not np.array_equal(sid, rsid):
        d = int((sid != rsid).sum())
        first = int(np.flatnonzero(sid != rsid)[0])
        fails.append(f"series_test differs in {d}/{len(sid)} windows (first at index {first}: "
                     f"{int(sid[first])} vs {int(rsid[first])})")
    ident.update(chronos_parity="match" if not fails else "MISMATCH",
                 parity_ok=not fails, parity_failures=fails,
                 chronos_reference=ref["paths"],
                 chronos_counts={"n_train": c["n_train"], "n_val": c["n_val"],
                                 "n_test": c["n_test"], "dataset_set": c["dataset_set"]})
    if fails and strict:
        raise RuntimeError(
            f"WINDOW PARITY FAILED for {tag}: TimesFM-3 is not being evaluated on the same "
            f"windows as Chronos-2.\n    " + "\n    ".join(fails) +
            f"\n  Reference: {ref['paths']['config']} + {ref['paths']['bootstrap_inputs']}.\n"
            "  Fix the window construction -- do NOT proceed with mismatched windows.")
    return ident


def print_roster_audit(rows):
    """The per-dataset window audit (the spec's required diagnostic)."""
    def cut(x, n):
        x = str(x)
        return x if len(x) <= n else x[:n - 1] + "~"
    hdr = (f"  {'dataset':<26}{'kind':<7}{'split_mode':<34}{'n_train':>8}{'n_val':>7}"
           f"{'n_test':>7}{'n_unit':>7}  {'parity':<12} first test windows")
    print("\n  WINDOW AUDIT -- these must be the SAME windows Chronos-2 was evaluated on")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {cut(r['dataset'], 25):<26}{cut(r['kind'], 6):<7}"
              f"{cut(r['split_mode'], 33):<34}{r['n_train_windows']:>8}"
              f"{str(r['n_val_windows']):>7}{r['n_test_windows']:>7}"
              f"{r['n_test_series']:>7}  {cut(r.get('chronos_parity'), 11):<12}"
              f"{' '.join(r['first_test_identifiers'][:4])}")
    ok = [r for r in rows if r.get("parity_ok")]
    nor = [r for r in rows if r.get("parity_ok") is None]
    print(f"  -> {len(ok)}/{len(rows)} datasets match the committed Chronos-2 windows "
          f"(counts + element-wise test series ids)" +
          (f"; {len(nor)} have no committed reference" if nor else ""))


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #

def run_dataset(tag, args, geom, paths, model, device, windows=None, ident=None):
    from probing.timesfm3_last_token import (LAST_LAYER, LAYER_NAMES, NUM_LAYERS, NUM_QUANTILES,
                                             assert_target_roundtrip, build_last_token_targets,
                                             cached_last_token_features, denormalize,
                                             raw_future_from_arcsinh)
    from probing.timesfm3_last_token_probes import (last_token_layerwise, native_reference,
                                                    tunnel_entrance, tunnel_entrances)

    kind, short = KIND.get(tag, "unclassified"), SHORT.get(tag, tag)
    print(f"\n{'=' * 82}\n[{tag}]  {short}  ({kind})\n{'=' * 82}")
    t0 = time.time()
    w = windows_for(tag, args.suite, args) if windows is None else windows
    if ident is None:
        ident = assert_window_parity(tag, w, chronos_reference(tag),
                                     strict=not args.allow_window_mismatch)
    meta = w["meta"]
    has_val = "X_val" in w and len(w["X_val"]) > 0
    print(f"  windows: {ident['n_train_windows']} train / "
          f"{ident['n_val_windows'] if has_val else 'carve'} val / "
          f"{ident['n_test_windows']} test  ({meta['split_mode']}, "
          f"{ident['n_test_series']} test "
          f"{ident['cluster_unit']}{'' if ident['cluster_unit'].endswith('s') else 's'})  "
          f"[Chronos-2 window parity: {ident.get('chronos_parity')}]")

    layers = list(range(NUM_LAYERS)) if args.layers is None else sorted(set(args.layers))
    if LAST_LAYER not in layers:
        layers = sorted(layers + [LAST_LAYER])
        print(f"  [note] added L{LAST_LAYER} to --layers (native endpoint + tunnel reference)")
    partial = len(layers) < NUM_LAYERS
    if partial:
        print(f"  [note] PARTIAL layer set {layers}: the tunnel entrance is only meaningful "
              "over the full Emb..L20 curve")

    # ---- features: ONE full-context decode() pass per batch, token 15 only ----
    ex = dict(geom=geom, model=model, device=device, batch_size=args.extract_batch_size,
              layers=layers, feature_dtype=np.dtype(args.feature_dtype),
              detrend=not args.no_detrend, cache_dir=paths["cache"], suite=args.suite,
              checkpoint=args.checkpoint, seed=args.seed, force=args.force_extract,
              allow_sorted_reference=args.allow_sorted_reference,
              bypass_sorting=not args.no_sorting_bypass)
    tr = cached_last_token_features(tag, "train", w["X_train"], **ex)
    te = cached_last_token_features(tag, "test", w["X_test"], **ex)
    va = cached_last_token_features(tag, "val", w["X_val"], **ex) if has_val else None
    shapes = {LAYER_NAMES[L]: tuple(np.shape(te["feats"][L])) for L in layers}
    print(f"  feature shapes: {LAYER_NAMES[layers[0]]} .. {LAYER_NAMES[layers[-1]]} each "
          f"{shapes[LAYER_NAMES[layers[-1]]]}   (test split; train is "
          f"{tuple(np.shape(tr['feats'][layers[-1]]))})")

    # ---- targets in the native normalized forecasting space of token 15 ----
    Ztr = np.concatenate([w["X_train"],
                          raw_future_from_arcsinh(w["X_train"], w["Y_train_traj"],
                                                  meta["sigma_eps"])], axis=1)
    Zte = np.concatenate([w["X_test"],
                          raw_future_from_arcsinh(w["X_test"], w["Y_test_traj"],
                                                  meta["sigma_eps"])], axis=1)
    ptr = build_last_token_targets(Ztr, tr["mu"], tr["sd"], geom, detrend=not args.no_detrend)
    pte = build_last_token_targets(Zte, te["mu"], te["sd"], geom, detrend=not args.no_detrend)
    pva = None
    if has_val:
        Zva = np.concatenate([w["X_val"],
                              raw_future_from_arcsinh(w["X_val"], w["Y_val_traj"],
                                                      meta["sigma_eps"])], axis=1)
        pva = build_last_token_targets(Zva, va["mu"], va["sd"], geom,
                                       detrend=not args.no_detrend)
        assert_target_roundtrip(Zva, pva["targets"], pva["trend"], va["mu"], va["sd"], geom,
                                pva["valid"], rtol=args.roundtrip_rtol)
    rt_tr = assert_target_roundtrip(Ztr, ptr["targets"], ptr["trend"], tr["mu"], tr["sd"],
                                    geom, ptr["valid"], rtol=args.roundtrip_rtol)
    rt_te = assert_target_roundtrip(Zte, pte["targets"], pte["trend"], te["mu"], te["sd"],
                                    geom, pte["valid"], rtol=args.roundtrip_rtol)
    print(f"  target round-trip (raw -> transformed -> raw): relative max err "
          f"train {rt_tr:.2e} / test {rt_te:.2e}   [< {args.roundtrip_rtol}]")
    print(f"  detrending fires: train {ptr['active'].mean():.0%} / "
          f"test {pte['active'].mean():.0%}    usable windows: "
          f"train {ptr['valid'].mean():.4f} / test {pte['valid'].mean():.4f}")
    nc = te["native_check"] or tr["native_check"]
    fresh = ""
    if not args.no_recheck_native and (te.get("cache_hit") or nc is None):
        # the cached check was measured when the cache was BUILT; re-prove it on one fresh
        # batch of this run's test windows (one forward pass) so a library change cannot hide
        # behind a warm cache.
        from probing.timesfm3_last_token import extract_last_token_features
        nb = min(args.extract_batch_size, len(w["X_test"]))
        chk = extract_last_token_features(
            w["X_test"][:nb], geom=geom, model=model, device=device, batch_size=nb,
            layers=[LAST_LAYER], feature_dtype=np.dtype(args.feature_dtype),
            detrend=not args.no_detrend, verify=True, progress=False,
            allow_sorted_reference=args.allow_sorted_reference,
            bypass_sorting=not args.no_sorting_bypass)
        nc = dict(chk["native_check"], rechecked_on_cache_hit=True)
        fresh = "  [re-checked this run]"
    if nc is not None:
        print(f"  L20 -> native head -> all {NUM_QUANTILES} quantiles vs decode(): "
              f"max|d| = {nc['max_abs']:.3e} (relative {nc['relative']:.3e}, "
              f"median {nc['median_max_abs']:.3e}, wrong-layout control "
              f"{nc['transposed_layout_relative']:.2e}){fresh}"
              f"{'  [SORTED reference]' if nc.get('sorted_reference_used') else ''}")
    if te["dtype_check"] is not None:
        d = te["dtype_check"]
        print(f"  feature storage {d['dtype']}: worst layer {d['worst_layer']} max|d| "
              f"{d['max_abs']:.3e} = {d['max_relative_to_layer_std']:.2e} of that layer's std")

    # ---- probes ----
    scores, diag = last_token_layerwise(
        tr["feats"], ptr["targets"], ptr["valid"], te["feats"], pte["targets"], pte["valid"],
        val_feats=(va["feats"] if has_val else None),
        val_targets=(pva["targets"] if has_val else None),
        val_valid=(pva["valid"] if has_val else None),
        H=geom.H, epochs=args.probe_epochs, lr=args.probe_lr,
        wd_grid=None if args.no_wd_grid else tuple(args.wd_grid), device=device,
        null_wd=args.null_wd, batch_size=args.probe_batch_size, layers=layers,
        collect_history=args.collect_history)
    rows = diag["test_rows"]

    # ---- MASE in RAW units + the native Q=9 / median baseline ----
    y_raw = Zte[rows, geom.target_start:geom.target_end]
    den = mase_denominator(w["X_test"][rows])
    n_clamped = int((den < MASE_DEN_FLOOR).sum())
    den = np.maximum(den, MASE_DEN_FLOOR)
    mase_curve, mase_pw = [], []
    for i in layers:
        yhat = denormalize(diag["test_median_pred"][i], mu=te["mu"][rows], sd=te["sd"][rows],
                           trend=pte["trend"][rows])
        pw = per_window_mase(y_raw, yhat, den)
        mase_pw.append(pw)
        mase_curve.append(float(pw.mean()))
    mase_pw = np.stack(mase_pw)

    nat = native_reference(te["native"], te["mu"], te["sd"], pte["trend"], pte["targets"],
                           pte["valid"])
    if not np.array_equal(nat["rows"], rows):
        raise RuntimeError("native baseline and probe are scored on different test windows")
    nat_mase_pw = per_window_mase(y_raw, nat["median_raw"], den)

    # ---- uncertainty: series-level cluster bootstrap, paired across layers ----
    sid = np.asarray(w["series_test"], np.int64)[rows]
    ref = layers.index(LAST_LAYER)
    q9_pw = np.stack([diag["test_q9_window"][i] for i in layers])
    med_pw = np.stack([diag["test_median_window"][i] for i in layers])
    boot = {"q9_loss": cluster_ci(q9_pw, sid, args.boot_b, args.seed, ref),
            "median_loss": cluster_ci(med_pw, sid, args.boot_b, args.seed, ref),
            "mase_context": cluster_ci(mase_pw, sid, args.boot_b, args.seed, ref)}
    # paired L20-probe-vs-native gaps: row 0 = native, row 1 = probe, reference = probe,
    # so delta_vs_last[0] = probe - native (positive = the probe is worse).
    gap = {}
    for name, p_pw, n_pw in (("q9_loss", q9_pw[ref], nat["q9_window"]),
                             ("median_loss", med_pw[ref], nat["median_window"]),
                             ("mase_context", mase_pw[ref], nat_mase_pw)):
        c = cluster_ci(np.stack([n_pw, p_pw]), sid, args.boot_b, args.seed, 1)
        gap[name] = {"probe_L20": c["point"][1], "native": c["point"][0],
                     "gap": c["delta_vs_last"][0], "gap_ci_lo": c["delta_ci_lo"][0],
                     "gap_ci_hi": c["delta_ci_hi"][0],
                     "relative_gap": (c["delta_vs_last"][0] / c["point"][0]
                                      if c["point"][0] else float("nan"))}

    # ---- tunnel entrance: VALIDATION only ----
    val = [diag["val_loss"][i] for i in layers]
    tun = tunnel_entrance(val, layers, args.tunnel_tol)
    tun_all = tunnel_entrances(val, layers)
    star = int(layers[int(np.argmin(val))])
    si = layers.index(star)
    ti = layers.index(tun["layer"])

    entry = {
        "tag": tag, "short": short, "kind": kind, "suite": args.suite,
        "window_identity": ident, "val_source": diag["val_source"],
        "layers": layers, "layer_names": [LAYER_NAMES[i] for i in layers],
        "partial_layer_set": partial, "geometry": geom.as_dict(), "window_meta": meta,
        "quantiles": diag["quantiles"], "num_quantiles": diag["num_quantiles"],
        "objective": "mean pinball loss over H*Q terms (1/(H*Q) sum_t sum_q rho_tau)",
        "val_q9_loss": val,
        "test_q9_loss": [scores[i] for i in layers],
        "test_median_loss": [diag["test_median_loss"][i] for i in layers],
        "mase_context": mase_curve,
        "train_q9_loss": [diag["train_loss"][i] for i in layers],
        "test_per_quantile_loss": {str(i): diag["test_per_quantile"][i] for i in layers},
        "weight_decay": [diag["wd"][i] for i in layers],
        "wd_at_grid_edge": [diag["wd_at_grid_edge"][i] for i in layers],
        "wd_grid": [float(x) for x in (args.wd_grid if not args.no_wd_grid else [])],
        "nonfinite_wd_candidates": [diag["nonfinite_wd_candidates"].get(i, 0) for i in layers],
        "null_baseline": {"wd": float(args.null_wd),
                          "val_loss": [(diag["null_baseline"].get(i) or {}).get("val_loss")
                                       for i in layers],
                          "test_q9_loss": [(diag["null_baseline"].get(i) or {}).get("test_q9")
                                           for i in layers],
                          "test_median_loss": [(diag["null_baseline"].get(i) or {}
                                                ).get("test_median_loss") for i in layers],
                          "max_abs_weight": [(diag["null_baseline"].get(i) or {}
                                              ).get("max_abs_weight") for i in layers],
                          "selected": False} if args.null_wd else None,
        "wd_selection": {str(i): diag["selection"][i] for i in layers},
        "n_train_rows": [diag["n_train_rows"][i] for i in layers],
        "n_val_windows": diag["n_val_rows"], "n_test_windows_scored": diag["n_test_rows"],
        "val_selected_best_layer": star, "val_best_loss": float(val[si]),
        "tunnel": tun, "tunnel_by_tolerance": tun_all,
        "relative_to_last": [float(v / val[-1]) for v in val],
        "test_relative_to_last": [float(scores[i] / scores[LAST_LAYER]) for i in layers],
        "native": {"q9_loss": nat["q9_loss"], "median_loss": nat["median_loss"],
                   "mase_context": float(nat_mase_pw.mean()),
                   "per_quantile_loss": nat["per_quantile"]},
        "l20_vs_native": {
            "probe_q9_loss": scores[LAST_LAYER], "native_q9_loss": nat["q9_loss"],
            "q9_absolute_gap": scores[LAST_LAYER] - nat["q9_loss"],
            "q9_relative_gap": ((scores[LAST_LAYER] - nat["q9_loss"]) / nat["q9_loss"]
                                if nat["q9_loss"] else float("nan")),
            "probe_median_loss": diag["test_median_loss"][LAST_LAYER],
            "native_median_loss": nat["median_loss"],
            "median_absolute_gap": diag["test_median_loss"][LAST_LAYER] - nat["median_loss"],
            "probe_median_mase": mase_curve[ref], "native_median_mase": float(nat_mase_pw.mean()),
            "mase_absolute_gap": mase_curve[ref] - float(nat_mase_pw.mean()),
            "bootstrap": gap,
            "note": "same linear hypothesis class R^1280 -> R^(64x9); the probe is "
                    "independently optimized with regularization on finite data, so equality "
                    "is NOT expected"},
        "bootstrap": boot,
        "at_tunnel_entrance": {"layer": tun["layer"], "layer_name": tun["layer_name"],
                               "val_q9_loss": float(val[ti]),
                               "test_q9_loss": float(scores[tun["layer"]]),
                               "test_median_loss": diag["test_median_loss"][tun["layer"]],
                               "mase_context": mase_curve[ti],
                               "test_q9_ci": [boot["q9_loss"]["ci_lo"][ti],
                                              boot["q9_loss"]["ci_hi"][ti]],
                               "test_delta_vs_L20": boot["q9_loss"]["delta_vs_last"][ti],
                               "test_delta_ci": [boot["q9_loss"]["delta_ci_lo"][ti],
                                                 boot["q9_loss"]["delta_ci_hi"][ti]]},
        "detrend_fire_rate": {"train": float(ptr["active"].mean()),
                              "test": float(pte["active"].mean())},
        "target_roundtrip_relative_err": {"train": rt_tr, "test": rt_te},
        "native_head_reconstruction": nc,
        "feature_dtype_check": te["dtype_check"] or tr["dtype_check"],
        "quantile_sorting": te["sorting"] or tr["sorting"],
        "feature_shapes": {k: list(v) for k, v in shapes.items()},
        "n_denominator_clamped": n_clamped, "cache_hit": bool(te.get("cache_hit")),
        "seconds": round(time.time() - t0, 1),
    }
    if args.collect_history:
        entry["history"] = {str(i): diag["history"][i] for i in layers}

    save_bootstrap_inputs(tag, entry, sid, q9_pw, med_pw, mase_pw, nat, nat_mase_pw, layers,
                          paths, args)
    print(f"  tunnel (val, {args.tunnel_tol:.0%}): entrance {tun['layer_name']} "
          f"(val {tun['val_loss']:.5f} <= {tun['threshold']:.5f} = "
          f"{1 + args.tunnel_tol:g}x L20's {tun['reference_val_loss']:.5f}); "
          f"2% entrance {tun_all['0.02']['layer_name']}")
    print(f"  Q9 test: entrance {entry['at_tunnel_entrance']['test_q9_loss']:.5f}  "
          f"L20 {scores[LAST_LAYER]:.5f}  native {nat['q9_loss']:.5f}   |   "
          f"MASE entrance {mase_curve[ti]:.4f}  L20 {mase_curve[ref]:.4f}  "
          f"native {nat_mase_pw.mean():.4f}   [{entry['seconds']}s]")
    if any(entry["wd_at_grid_edge"]):
        edge = [LAYER_NAMES[i] for i, e in zip(layers, entry["wd_at_grid_edge"]) if e]
        print(f"  [warn] weight decay selected the GRID MAXIMUM ({max(args.wd_grid):g}) at "
              f"{edge} -- validation may still be improving past the grid; widen --wd-grid")
    return entry


def save_bootstrap_inputs(tag, entry, sid, q9_pw, med_pw, mase_pw, nat, nat_mase_pw, layers,
                          paths, args):
    """Per-window test metrics -> the cluster bootstrap can be redone post hoc, and any other
    tolerance (2%, 1%, ...) re-evaluated from the saved validation curve, without re-extracting
    a single feature. Readout name ``last_token`` keeps it distinguishable from the ablation's
    ``origin_last``."""
    arrays = {"series_test": sid,
              "window_q9_loss__last_token": np.asarray(q9_pw, np.float64),
              "window_median_loss__last_token": np.asarray(med_pw, np.float64),
              "window_mase_context__last_token": np.asarray(mase_pw, np.float64),
              "window_q9_loss__native": np.asarray(nat["q9_window"], np.float64),
              "window_median_loss__native": np.asarray(nat["median_window"], np.float64),
              "window_mase_context__native": np.asarray(nat_mase_pw, np.float64)}
    for k, a in arrays.items():
        if not np.isfinite(a).all():
            raise RuntimeError(f"{tag}: non-finite values in {k}")
    meta = {"model": "timesfm-3.0", "experiment": "last_token_q9", "last_token_only": True,
            "prefix_extraction": False, "tag": tag, "short": entry["short"],
            "kind": entry["kind"], "suite": entry["suite"],
            "val_source": entry["val_source"], "window_identity": entry["window_identity"],
            "C": entry["geometry"]["C"], "H": entry["geometry"]["H"],
            "selected_token_index": entry["geometry"]["selected_token_index"],
            "num_real_context_patches": entry["geometry"]["num_real_context_patches"],
            "num_layers": len(layers), "layers": layers, "layer_names": entry["layer_names"],
            "last_layer": layers[-1], "quantiles": entry["quantiles"],
            "num_quantiles": entry["num_quantiles"], "objective": entry["objective"],
            "split_mode": entry["window_meta"]["split_mode"],
            "n_test_windows": int(len(sid)), "n_test_series": int(len(np.unique(sid))),
            "primary_readouts": ["last_token"], "controlled_readouts": [],
            "mase_definition": f"mase_context: in-context seasonal-naive (m={M_SEASON}, "
                               f"floor {MASE_DEN_FLOOR})",
            "seasonal_m": M_SEASON, "boot_b": int(args.boot_b), "seed": int(args.seed),
            "val_q9_loss_by_layer": {"last_token": entry["val_q9_loss"]},
            "tunnel": entry["tunnel"], "tunnel_by_tolerance": entry["tunnel_by_tolerance"],
            "reported": {"q9_loss": {"last_token": entry["test_q9_loss"]},
                         "median_loss": {"last_token": entry["test_median_loss"]},
                         "mase_context": {"last_token": entry["mase_context"]},
                         "native_q9_loss": entry["native"]["q9_loss"],
                         "native_median_loss": entry["native"]["median_loss"],
                         "native_mase_context": entry["native"]["mase_context"]}}
    out = paths["boot"] / "inputs" / f"{tag}__C{meta['C']}_H{meta['H']}_tok15_q9.npz"
    np.savez(out, meta=json.dumps(meta, default=str), **arrays)
    print(f"  [saved]      {out.relative_to(paths['out'])}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _panels(ds, ylab, title, paths, name, draw):
    """One panel per dataset, PT-ID row(s) first then PT-OOD, at most 4 panels per row."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(ds)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow), squeeze=False)
    axes = [a for row in ax for a in row]
    for a, e in zip(axes, ds):
        draw(a, e)
        a.set_title(f"{e.get('short', e['tag'])}  ({e.get('kind', '?')})", fontsize=10)
        a.set_xlabel("representation point")
        a.set_xticks(e["layers"][::2])
        a.set_xticklabels(e["layer_names"][::2], rotation=45, fontsize=7)
        a.grid(alpha=0.3)
    for a in axes[n:]:
        a.set_visible(False)
    for r in range(nrow):
        ax[r][0].set_ylabel(ylab)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    p = paths["fig"] / name
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  [figure]     {p.relative_to(paths['out'])}")


def make_figures(summary, paths):
    # PT-ID panels first, then PT-OOD -- the Chronos-2 paper's ordering
    order = {"PT-ID": 0, "PT-OOD": 1}
    ds = sorted(summary["datasets"].values(),
                key=lambda e: (order.get(e.get("kind"), 2), list(PAPER7).index(e["tag"])
                               if e["tag"] in PAPER7 else 99))
    if not ds:
        return

    def tun_line(a, e, label=True):
        t = e["tunnel"]
        a.axvline(t["layer"], ls=":", c="k", lw=1.1,
                  label=(f"{t['tol']:.0%} tunnel entrance = {t['layer_name']}"
                         if label else None))

    # 1. the headline: Q=9 forecasting loss vs layer
    def f1(a, e):
        b = e["bootstrap"]["q9_loss"]
        a.plot(e["layers"], e["test_q9_loss"], "o-", ms=3, color="#1f77b4",
               label="last-token probe (test)")
        a.fill_between(e["layers"], b["ci_lo"], b["ci_hi"], alpha=0.2, color="#1f77b4",
                       label="95% cluster-bootstrap CI")
        a.plot(e["layers"], e["val_q9_loss"], "s--", ms=2.5, lw=1, color="#9467bd", alpha=0.8,
               label="validation (selects the entrance)")
        a.axhline(e["native"]["q9_loss"], ls="--", c="crimson", lw=1.2,
                  label="native TimesFM-3 (Q=9)")
        nb = e.get("null_baseline")
        if nb and any(v is not None for v in nb["test_q9_loss"]):
            a.plot(e["layers"], nb["test_q9_loss"], "-", lw=1, color="0.55", alpha=0.9,
                   label=f"null (wd={nb['wd']:g}, ~bias-only)")
        tun_line(a, e)
    _panels(ds, "mean pinball loss over the 9 native quantiles",
            "TimesFM-3 last-context token (index 15), C=512 -> H=64, native Q=9 objective",
            paths, "q9_loss_by_layer.png", f1)

    # 2. relative to L20, with the 1+tol threshold -- the tunnel, if any, is visible here
    def f2(a, e):
        a.plot(e["layers"], e["relative_to_last"], "s--", ms=3, color="#9467bd",
               label="validation / L20")
        a.plot(e["layers"], e["test_relative_to_last"], "o-", ms=3, color="#1f77b4",
               label="test / L20")
        a.axhline(1.0, c="k", lw=0.8)
        a.axhline(1.05, ls="--", c="darkorange", lw=1.1, label="1.05 (5% criterion)")
        a.axhline(1.02, ls=":", c="darkorange", lw=1.0, label="1.02 (2% criterion)")
        tun_line(a, e)
    _panels(ds, r"$\mathcal{L}_\ell\ /\ \mathcal{L}_{L20}$",
            "Relative Q=9 loss vs the final layer (tunnel entrance = first validation crossing)",
            paths, "q9_relative_to_last_layer.png", f2)

    # 3. median (tau=0.5) MASE vs layer
    def f3(a, e):
        b = e["bootstrap"]["mase_context"]
        a.plot(e["layers"], e["mase_context"], "o-", ms=3, color="#2ca02c",
               label=r"probe median ($\tau$=0.5)")
        a.fill_between(e["layers"], b["ci_lo"], b["ci_hi"], alpha=0.2, color="#2ca02c",
                       label="95% cluster-bootstrap CI")
        a.axhline(e["native"]["mase_context"], ls="--", c="crimson", lw=1.2,
                  label="native TimesFM-3 median")
        tun_line(a, e)
    _panels(ds, f"MASE (in-context seasonal-naive, m={M_SEASON})",
            r"Median-forecast MASE by layer ($\tau$=0.5 row of the Q=9 probe), raw units",
            paths, "median_mase_by_layer.png", f3)

    # 4. median-only pinball loss (diagnostic; never selects the entrance)
    def f4(a, e):
        b = e["bootstrap"]["median_loss"]
        a.plot(e["layers"], e["test_median_loss"], "o-", ms=3, color="#8c564b",
               label=r"probe $\rho_{0.5}$ = 0.5*MAE")
        a.fill_between(e["layers"], b["ci_lo"], b["ci_hi"], alpha=0.2, color="#8c564b")
        a.axhline(e["native"]["median_loss"], ls="--", c="crimson", lw=1.2, label="native")
        tun_line(a, e)
    _panels(ds, r"median-only pinball loss ($\tau$=0.5, normalized space)",
            "Median-only diagnostic (NOT the tunnel criterion in this experiment)",
            paths, "median_loss_by_layer.png", f4)

    # 5. paired delta vs L20 with CIs
    def f5(a, e):
        b = e["bootstrap"]["q9_loss"]
        a.plot(e["layers"], b["delta_vs_last"], "o-", ms=3, color="#ff7f0e")
        a.fill_between(e["layers"], b["delta_ci_lo"], b["delta_ci_hi"], alpha=0.2,
                       color="#ff7f0e", label="95% paired CI")
        a.axhline(0, c="k", lw=0.8)
        tun_line(a, e)
    _panels(ds, r"$\mathcal{L}(L20) - \mathcal{L}(\ell)$   (+ = earlier layer better)",
            "Paired delta vs L20, series-level cluster bootstrap", paths,
            "q9_delta_vs_last_layer.png", f5)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TimesFM-3 last-context-token layer-wise probing (C=512 -> H=64, native Q=9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("model / environment (Narval: point these at $SCRATCH)")
    g.add_argument("--checkpoint", default=os.environ.get("TIMESFM3_CHECKPOINT",
                                                          "google/timesfm-3.0-pytorch"),
                   help="HF repo id, or a LOCAL dir with config.json + model.safetensors "
                        "(use the local form on offline compute nodes)")
    g.add_argument("--hf-home", default=None, help="sets HF_HOME before anything is loaded")
    g.add_argument("--cache-dir", default=os.environ.get("TFM3_LT_CACHE_DIR", None),
                   help="feature cache root [default: <repo>/features_cache]. MUST NOT be the "
                        "shared-origin ablation's cache dir")
    g.add_argument("--out-root", default=os.environ.get("TFM3_LT_OUT_ROOT", None),
                   help="results root [default: <repo>/results]")
    g.add_argument("--device", default=os.environ.get("TFM3_DEVICE", None),
                   help="cuda | mps | cpu [default: auto]")

    g = p.add_argument_group("data")
    g.add_argument("--suite", "--dataset-set", dest="suite",
                   default=os.environ.get("TFM3_SUITE", "paper7"),
                   help="'paper7' = the Chronos-2 headline seven (PT-ID: m4_hourly, "
                        "monash_electricity_hourly, uber_tlc_hourly, wind_farms_hourly; PT-OOD: "
                        "sg_carpark, coastal_ts, boom_hourly), windowed exactly as the Chronos-2 "
                        "run. Any id_data.ID_DATASET_SPECS key also works (e.g. extended_v1, "
                        "which is the KDD/pedestrian implementation-validation set)")
    g.add_argument("--datasets", nargs="+", default=None,
                   help="subset of the suite's roster (default: all of it)")
    g.add_argument("--allow-window-mismatch", action="store_true",
                   help="report, instead of aborting on, a disagreement with the committed "
                        "Chronos-2 windows (NOT recommended: the two models would be scored on "
                        "different data)")
    g.add_argument("--audit-only", action="store_true",
                   help="build the windows, print the per-dataset audit + Chronos-2 parity, and "
                        "exit WITHOUT loading the model or fitting anything")
    g.add_argument("--context-len", type=int, default=512)
    g.add_argument("--horizon", type=int, default=64)
    g.add_argument("--stride", type=int, default=64)

    g = p.add_argument_group("extraction (ONE full-context pass per batch)")
    g.add_argument("--extract-batch-size", type=int, default=64)
    g.add_argument("--feature-dtype", default="float32", choices=["float32", "float16"],
                   help="feature-cache storage dtype. float32 (default) is lossless (~2 GB for "
                        "the four datasets). float16 halves it but is a measurably lossy cast "
                        "(~1.2e-2 of the layer std) -- opt in explicitly")
    g.add_argument("--force-extract", action="store_true")
    g.add_argument("--layers", type=int, nargs="+", default=None,
                   help="representation points to probe; default 0..20 (0 = Emb, 20 = L20)")
    g.add_argument("--no-detrend", action="store_true",
                   help="ablation: disable TimesFM-3's linear detrending in the BACKBONE and "
                        "in the targets")
    g.add_argument("--allow-sorted-reference", action="store_true",
                   help="accept a SORTED decode() output as the L20 reconstruction reference, "
                        "when the installed timesfm3 sorts quantiles and exposes no bypass. "
                        "Recorded explicitly in the results; off by default")
    g.add_argument("--no-sorting-bypass", action="store_true",
                   help="do NOT disable discovered quantile-sorting knobs (diagnostic)")
    g.add_argument("--no-recheck-native", action="store_true",
                   help="skip re-proving the L20 all-quantile native reconstruction on one "
                        "fresh batch when the feature cache HITS (NOT recommended)")
    g.add_argument("--roundtrip-rtol", type=float, default=1e-4,
                   help="abort threshold for the target inverse round-trip")

    g = p.add_argument_group("probe (Linear(1280, 64*9), native Q=9)")
    g.add_argument("--probe-epochs", type=int, default=300)
    g.add_argument("--probe-lr", type=float, default=1e-2)
    g.add_argument("--probe-batch-size", type=int, default=0,
                   help="0 = full batch (the Chronos-2 protocol)")
    g.add_argument("--wd-grid", type=float, nargs="+",
                   default=list(WD_GRID_LAST_TOKEN),
                   help="hyperparameter-SELECTION grid: probes.WD_GRID_V2 extended by 10/30. It "
                        "stops there on purpose -- at lr=1e-2, wd=100 (lr*wd=1) zeroes the "
                        "weight every step and wd=300 (lr*wd=3) makes the decoupled decay "
                        "unstable, so selecting either would report an optimizer artifact")
    g.add_argument("--null-wd", type=float, default=WD_NULL_BASELINE,
                   help="extreme-decay NULL BASELINE fitted per layer and reported alongside the "
                        "probe, never a selection candidate: the weight is zeroed each step, so "
                        "it is ~a bias-only (marginal-quantile) fit = the no-information floor. "
                        "0 disables it")
    g.add_argument("--no-wd-grid", action="store_true")
    g.add_argument("--collect-history", action="store_true")

    g = p.add_argument_group("evaluation")
    g.add_argument("--tunnel-tol", type=float, default=0.05,
                   help="the headline tunnel criterion; 0.01/0.02/0.05/0.10 are always saved")
    g.add_argument("--boot-b", type=int, default=5000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-figures", action="store_true")
    g.add_argument("--tag", default=None, help="extra suffix on the output directory")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from probing import config
    from probing.timesfm3_last_token import (CACHE_VERSION, LAST_LAYER, NUM_LAYERS,
                                             NUM_QUANTILES, LastTokenGeometry,
                                             assert_backbone_frozen, assert_native_geometry,
                                             assert_native_quantiles, get_model,
                                             timesfm_version)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    roster = suite_tags(args.suite)
    if args.suite == "paper7":
        # the PT-ID roster + rolling windows + budgets live in this dataset set; PT-OOD targets
        # are outside ID_DATASET_SPECS and come from their own loaders (run_native_head_adapter
        # sets the same set once, globally, for the same reason).
        config.set_dataset_set(PTID_SET)
    else:
        config.set_dataset_set(args.suite)
    tags = args.datasets or roster
    unknown = [t for t in tags if t not in roster]
    if unknown:
        raise SystemExit(f"unknown dataset tag(s) {unknown} for suite {args.suite!r}; "
                         f"roster = {roster}")

    out_root = Path(args.out_root).expanduser() if args.out_root else REPO_ROOT / "results"
    name = f"timesfm3_last_token_{args.suite}_q9" + (f"_{args.tag}" if args.tag else "")
    out = out_root / name
    paths = {"out": out, "boot": out / "bootstrap", "fig": out / "figures",
             "cache": (Path(args.cache_dir).expanduser() if args.cache_dir
                       else REPO_ROOT / "features_cache")}
    (paths["boot"] / "inputs").mkdir(parents=True, exist_ok=True)
    paths["fig"].mkdir(parents=True, exist_ok=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)

    geom = LastTokenGeometry(args.context_len, args.horizon)      # strict: asserts 512/64/16/15

    # windows FIRST, so a roster/window problem costs no GPU time at all
    print(f"TimesFM-3 last-token probing -- suite {args.suite!r}: {len(tags)} datasets")
    windows, idents = {}, []
    for tag in tags:
        windows[tag] = windows_for(tag, args.suite, args)
        idents.append(assert_window_parity(tag, windows[tag], chronos_reference(tag),
                                           strict=not args.allow_window_mismatch))
    print_roster_audit(idents)
    if args.audit_only:
        audit_path = out / "window_audit.json"
        out.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({"suite": args.suite, "datasets": idents}, indent=2,
                                         default=str))
        print(f"\n[saved] {audit_path}\n[audit-only] no model was loaded, nothing was fitted.")
        return {"config": {**vars(args)}, "window_audit": idents}

    model, device = get_model(args.checkpoint, args.device)
    n_par = assert_backbone_frozen(model)
    qinfo = assert_native_quantiles(model)
    ginfo = assert_native_geometry(model, geom)
    row = geom.audit_row()

    print("TimesFM-3 LAST-CONTEXT-TOKEN layer-wise probing (native Q=9)")
    print(f"  checkpoint : {args.checkpoint}  (timesfm {timesfm_version()}, device {device})")
    print(f"  backbone   : FROZEN, {n_par} parameter tensors, requires_grad=False everywhere")
    print(f"  quantiles  : {[round(x, 4) for x in qinfo['quantiles']]}  (median at index "
          f"{qinfo['median_index']}, strictly increasing, read from the loaded model; full "
          f"precision in the summary JSON)")
    print(f"  native head: Linear(1280, {qinfo['output_head_out_features']}) "
          f"= Linear(1280, {args.horizon}*{NUM_QUANTILES})  <- reads h_L20 at token "
          f"{geom.selected_token_index}")
    print(f"  raw_context_length        = {row['raw_context_length']}")
    print(f"  num_real_context_patches  = {row['num_real_context_patches']}")
    print(f"  selected_token_index      = {row['selected_token_index_0based']}  "
          f"(decode()'s own forecast index: {ginfo['native_forecast_indices']})")
    print(f"  tokens in the decode pass = {row['n_tokens_in_decode_pass']} "
          f"({row['num_real_context_patches']} real context + "
          f"{row['n_tokens_in_decode_pass'] - row['num_real_context_patches']} placeholders)")
    print(f"  target                    = x[{row['target_start_1based']}:"
          f"{row['target_end_1based']}] (1-based), shape {row['transformed_target_shape']}")
    print(f"  probe                     = {row['probe']} per point, "
          f"{NUM_LAYERS} points Emb..L{NUM_LAYERS - 1}")
    hot = [w for w in args.wd_grid if args.probe_lr * w >= 1.0]
    print(f"  wd grid    : {len(args.wd_grid)} candidates "
          f"{min(args.wd_grid):g}..{max(args.wd_grid):g} (max lr*wd = "
          f"{args.probe_lr * max(args.wd_grid):g}), selected per layer on VALIDATION")
    if args.null_wd:
        print(f"  null base  : wd={args.null_wd:g} (lr*wd="
              f"{args.probe_lr * args.null_wd:g}) fitted per layer and REPORTED, never selected "
              "-- weight zeroed each step, so ~a bias-only / marginal-quantile fit")
    if hot:
        raise SystemExit(
            f"--wd-grid contains {[float(w) for w in hot]}, for which lr*wd >= 1 at "
            f"lr={args.probe_lr:g}: AdamW's decoupled decay multiplies the weight by (1 - lr*wd) "
            "each step, so these are optimizer artifacts (bias-only at lr*wd=1, unstable beyond) "
            "rather than regularization strengths. Use --null-wd for the extreme-decay reference, "
            "or lower --probe-lr if you really mean to search there.")
    print(f"  cache      : {paths['cache']}  ({CACHE_VERSION})")
    print(f"  out        : {out}")

    summary = {"config": {**vars(args), "num_layers": NUM_LAYERS, "last_layer": LAST_LAYER,
                          "m_season": M_SEASON, "mase_den_floor": MASE_DEN_FLOOR,
                          "cache_version": CACHE_VERSION, "timesfm_version": timesfm_version(),
                          "device": device, "n_backbone_param_tensors": n_par,
                          "backbone_frozen": True, "geometry": geom.as_dict(),
                          "native_quantiles": qinfo, "native_geometry": ginfo,
                          "audit_row": row, "suite": args.suite, "roster": tags,
                          "pt_id_tags": list(PT_ID_TAGS), "pt_ood_tags": list(PT_OOD_TAGS),
                          "ptid_dataset_set": PTID_SET,
                          "chronos_reference_root": str(CHRONOS_REF_ROOT.relative_to(REPO_ROOT))},
               "window_audit": idents,
               "datasets": {}}
    for tag, ident in zip(tags, idents):
        summary["datasets"][tag] = run_dataset(tag, args, geom, paths, model, device,
                                               windows=windows.pop(tag), ident=ident)

    sp = out / "timesfm3_last_token_summary.json"
    sp.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] {sp}")
    if not args.no_figures:
        make_figures(summary, paths)

    print(f"\n{'=' * 100}\nSUMMARY - forecasting tunnel on the native Q=9 objective, and L20 vs "
          f"the native head\n{'=' * 100}")
    print(f"{'dataset':<16}{'kind':<8}{'tun5%':>7}{'tun2%':>7}{'L(tun)':>9}{'L(L20)':>9}"
          f"{'native':>9}{'gap':>8}{'rel':>7}{'MASE L20':>10}{'nat MASE':>10}")
    last_kind = None
    for tag, e in summary["datasets"].items():
        c, t = e["l20_vs_native"], e["tunnel"]
        if last_kind is not None and e["kind"] != last_kind:
            print("-" * 100)
        last_kind = e["kind"]
        print(f"{e['short']:<16}{e['kind']:<8}{t['layer_name']:>7}"
              f"{e['tunnel_by_tolerance']['0.02']['layer_name']:>7}"
              f"{e['at_tunnel_entrance']['test_q9_loss']:>9.5f}{c['probe_q9_loss']:>9.5f}"
              f"{c['native_q9_loss']:>9.5f}{c['q9_absolute_gap']:>8.4f}"
              f"{c['q9_relative_gap']:>6.0%}{c['probe_median_mase']:>10.4f}"
              f"{c['native_median_mase']:>10.4f}")
    print("\n  tun5%/tun2% are VALIDATION-selected first crossings of (1+tol)*L_val(L20);\n"
          "  L(tun)/L(L20)/native are TEST mean pinball losses over the 9 native quantiles.\n"
          "  PT-ID = in Chronos-2's pretraining corpus, PT-OOD = documented outside it "
          "(probing/tunnel.py).")
    return summary


if __name__ == "__main__":
    main()
