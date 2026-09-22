"""Phase-1 TimesFM-3 adapter: last REAL context token, Linear(1280, H*Q), 21 depths.

WHERE TIMESFM-3'S OWN HEAD READS. ``decode()``'s forecast index for (C=512, H=64) is the single
token 15 — the last REAL context patch — and ``assert_native_geometry`` proves that from the
model's own attributes rather than asserting a remembered number. The Phase-1 probe reads
exactly that state: one ``Linear(1280, H*Q)`` per representation point, reshaped horizon-major
to (B, Q, 64), which is the native head's own flat layout (``verify_native_head`` reproduces
decode()'s nine quantiles from those same weights, and shows the transposed layout does not).

THE OLD MULTI-ORIGIN LINE IS NOT REINTRODUCED. ``experiments/run_timesfm3_probing.py`` (16
independent causal prefixes, one shared Linear(1280, 64) across origins, Q=1) is a different
experiment with a disjoint cache namespace (``tfm3-prefix-v1``, rank-3 features). The loader
REFUSES it by metadata and array rank, and a Phase-1 contract test pins that nothing here
touches it.

REPRESENTATION POINTS (21): Emb (the pre_transformer_resblock output) and L1..L20. There is no
separate final-norm point: L20 is what the native quantile head reads directly, so it is both
the last probed depth and the tunnel reference.

Q=9 vs Q=1. ``last_token_layerwise`` has always taken ``quantiles=`` and resolves the median
index from the vector in use, so Q is a parameter, not a rewrite. Phase 1 passes the canonical
nine; ``--quantile-set q1`` stays reachable for later robustness work and writes to its own
namespace.

GEOMETRY. One row per window at d=1280 — the readout token state itself, no reshape. That is
4x fewer rows than Chronos-2's slot stacking at 1.7x the width, which is exactly why the biased
CKA floor is so much higher here (~0.83 at N=262 vs ~0.42 for Chronos-2) and why the Phase-1
headline estimator is the unbiased one.
"""

from __future__ import annotations

import numpy as np

from probing import phase1
from probing.config import SEED
from probing.phase1_metrics import raw_window_metrics

__all__ = ["MODEL", "EPOCHS", "LR", "WD_GRID", "NULL_WD", "geometry_for", "extract",
           "fit_layerwise", "raw_metrics", "native_reference", "geometry_blocks"]

MODEL = "timesfm3"
EPOCHS = 300
LR = 1e-2


#: THE PHASE-1 GRID — the SAME 13 candidates all three models use, for Q=9 and Q=1 alike.
#: ``timesfm3_last_token_probes.WD_GRID_LAST_TOKEN`` stays that LINE's own grid, untouched, so
#: the committed paper7 last-token results keep their exact protocol; Phase 1 asserts the new
#: grid is a strict superset of it rather than replacing it silently.
#:
#: WHY IT GREW, measured on the committed Phase-1 cell (timesfm3 x monash_electricity_hourly):
#: 16 of 21 depths selected the old maximum 30 and validation was STILL falling there
#: (L19: 0.12419 at wd=10 -> 0.11677 at wd=30). The wd=100 null beat wd=30 at only 3 of 21
#: depths, so the optimum sits INSIDE (30, 100) — which is exactly the interval the new
#: candidates 45/65/90 cover, and the largest interval AdamW's decoupled decay admits at lr=1e-2.
WD_GRID = phase1.PHASE1_WD_GRID
NULL_WD = phase1.PHASE1_WD_NULL


def _assert_superset():
    """The no-regression rule, checked lazily (the legacy grid lives behind a torch import)."""
    from probing.timesfm3_last_token_probes import WD_GRID_LAST_TOKEN
    if not phase1.wd_grid_is_superset_of(WD_GRID, WD_GRID_LAST_TOKEN):
        raise RuntimeError("the Phase-1 grid must retain every legacy TimesFM-3 candidate; "
                           f"{WD_GRID} is not a superset of {WD_GRID_LAST_TOKEN}")
    return True


def _spec():
    return phase1.model_spec(MODEL)


def geometry_for(C: int = phase1.PHASE1_C, H: int = phase1.PHASE1_H):
    """The validated single-readout layout. ``strict=True`` hard-asserts C=512 -> H=64."""
    from probing.timesfm3_last_token import LastTokenGeometry
    return LastTokenGeometry(C=C, H=H, strict=True)


# --------------------------------------------------------------------------- #
# extraction + targets
# --------------------------------------------------------------------------- #
def extract(tag: str, w: dict, *, geom, cache_dir, checkpoint=None, model=None, device=None,
            batch_size: int = 64, feature_dtype=np.float32, detrend: bool = True,
            suite: str = "paper14", seed: int = SEED, force: bool = False,
            splits=("train", "val", "test"), roundtrip_rtol: float = 1e-4) -> tuple[dict, dict]:
    """Token-15 states + the native forecast + the probe targets, per split.

    The target is built in TimesFM-3's OWN normalized forecasting space: the raw future is
    recovered from the Chronos-2-built arcsinh labels (``raw_future_from_arcsinh``, the exact
    inverse), concatenated to the context, then passed through ``build_last_token_targets``
    (linear detrend + the readout token's RevIN statistics). ``assert_target_roundtrip`` proves
    raw -> transformed -> raw closes to ``roundtrip_rtol`` on every split, so the target space
    is verified rather than trusted.
    """
    from probing.timesfm3_last_token import (assert_target_roundtrip, build_last_token_targets,
                                             cached_last_token_features, raw_future_from_arcsinh)
    meta = w["meta"]
    ex = {"cache_hits": {}, "feature_shapes": {}, "target_roundtrip": {}, "detrend": bool(detrend),
          "detrend_fires": {}, "usable_fraction": {}, "native_check": None, "dtype_check": None,
          "sorting": None}
    kw = dict(geom=geom, model=model, device=device, batch_size=batch_size,
              feature_dtype=np.dtype(feature_dtype), detrend=detrend, cache_dir=cache_dir,
              suite=suite, checkpoint=checkpoint, seed=seed, force=force)
    out = {}
    for split in splits:
        Xk, Yk, Sk = f"X_{split}", f"Y_{split}_traj", f"series_{split}"
        if Xk not in w:
            raise KeyError(f"{tag}: the window dict has no {Xk!r} — Phase-1 needs a dedicated "
                           "validation split")
        X = np.asarray(w[Xk], np.float32)
        f = cached_last_token_features(tag, split, X, **kw)
        Z = np.concatenate([X, raw_future_from_arcsinh(X, w[Yk], meta["sigma_eps"])], axis=1)
        t = build_last_token_targets(Z, f["mu"], f["sd"], geom, detrend=detrend)
        rt = assert_target_roundtrip(Z, t["targets"], t["trend"], f["mu"], f["sd"], geom,
                                     t["valid"], rtol=roundtrip_rtol)
        out[split] = {"feats": f["feats"], "mu": f["mu"], "sd": f["sd"], "native": f["native"],
                      "X": X, "Z": Z, "targets": t["targets"], "trend": t["trend"],
                      "valid": t["valid"], "series": np.asarray(w[Sk], np.int64),
                      # the raw-unit future the MASE/WQL metrics score against, sliced ONCE
                      # with decode()'s own target window (geom.target_start/target_end)
                      "y_raw": np.asarray(Z, np.float64)[:, geom.target_start:geom.target_end]}
        ex["cache_hits"][split] = bool(f.get("cache_hit", False))
        ex["target_roundtrip"][split] = float(rt)
        ex["detrend_fires"][split] = float(np.mean(t["active"]))
        ex["usable_fraction"][split] = float(np.mean(t["valid"]))
        last = max(f["feats"])
        ex["feature_shapes"][split] = list(np.shape(f["feats"][last]))
        for k in ("native_check", "dtype_check", "sorting"):
            if f.get(k) is not None and ex[k] is None:
                ex[k] = f[k]
    ex["representation_points"] = _spec().labels
    ex["readout_token_index"] = int(geom.selected_token_index)
    return out, ex


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def fit_layerwise(tag: str, data: dict, *, quantiles, median_idx: int, geom, device: str,
                  epochs: int = EPOCHS, lr: float = LR, wd_grid=None, null_wd=None,
                  seed: int = SEED, verbose: bool = True) -> dict:
    """Fit + score one probe per representation point, via the VALIDATED layerwise sweep.

    ``last_token_layerwise`` is called unchanged (with ``collect_probe=True``, an additive
    option that only records the frozen probe and its predictions). The explicit-validation
    protocol applies: scaler and Linear fit on FULL train, weight decay chosen on the dedicated
    val split, no refit, test read once.
    """
    from probing.timesfm3_last_token_probes import last_token_layerwise
    spec = _spec()
    _assert_superset()
    null_wd = NULL_WD if null_wd is None else null_wd
    wd_grid = phase1.assert_wd_grid(WD_GRID if wd_grid is None else wd_grid, lr, null_wd=null_wd)
    tr, va, te = data["train"], data["val"], data["test"]

    scores, diag = last_token_layerwise(
        tr["feats"], tr["targets"], tr["valid"],
        te["feats"], te["targets"], te["valid"],
        val_feats=va["feats"], val_targets=va["targets"], val_valid=va["valid"],
        H=geom.H, epochs=epochs, lr=lr, quantiles=quantiles, wd_grid=wd_grid,
        device=device, null_wd=null_wd, batch_size=0, layers=list(range(spec.n_points)),
        collect_probe=True, verbose=verbose)

    layers = list(range(spec.n_points))
    grid_max, grid_min = float(max(wd_grid)), float(min(wd_grid))
    res = {
        "labels": spec.labels, "quantiles": [float(x) for x in quantiles],
        "median_index": median_idx, "val_source": diag["val_source"],
        "train_loss": [float(diag["train_loss"][i]) for i in layers],
        "val_loss": [float(diag["val_loss"][i]) for i in layers],
        "test_loss": [float(scores[i]) for i in layers],
        "wd": [float(diag["wd"][i]) for i in layers],
        "wd_at_grid_max": [bool(float(diag["wd"][i]) == grid_max) for i in layers],
        "wd_at_grid_min": [bool(float(diag["wd"][i]) == grid_min) for i in layers],
        "selection": [{str(k): float(v) for k, v in
                       (diag["selection"][i] or {}).get("val_loss_by_wd", {}).items()}
                      for i in layers],
        "n_params": [int(diag["probe"][i]["weight"].size + diag["probe"][i]["bias"].size)
                     for i in layers],
        "per_quantile_test": [diag["test_per_quantile"][i] for i in layers],
        "null_baseline": {spec.labels[i]: diag["null_baseline"].get(i) for i in layers},
        "test_window_loss": np.stack([diag["test_q9_window"][i] for i in layers]),
        "val_window_loss": np.stack([diag["val_window_loss"][i] for i in layers]),
        "pred_test": {spec.labels[i]: diag["test_pred"][i] for i in layers},
        "pred_val": {spec.labels[i]: diag["val_pred"][i] for i in layers},
        "probe_weights": {spec.labels[i]: diag["probe"][i] for i in layers},
        "rows_test": np.asarray(diag["test_rows"], np.int64),
        "rows_val": np.asarray(diag["val_rows"], np.int64),
        "probe_description": f"Linear(1280, {geom.H}*{len(quantiles)})",
        "wd_grid": [float(w) for w in wd_grid],
    }
    # The closed-form no-information floor, beside the per-depth FITTED null at wd=100. The two
    # answer the same question two ways, which is why both are kept: the fitted null shows what
    # this optimizer does at the wall, the closed form shows where the wall actually is.
    res["constant_forecast_floor"] = phase1.constant_forecast_floor(
        np.asarray(tr["targets"])[np.asarray(tr["valid"], bool)],
        {"val": np.asarray(va["targets"])[np.asarray(va["valid"], bool)],
         "test": np.asarray(te["targets"])[np.asarray(te["valid"], bool)]},
        quantiles)
    # The sweep drops windows the target builder marked invalid; every downstream array must be
    # indexed by the SAME rows or the cluster ids would not align with the losses.
    res["target_test"] = np.asarray(te["targets"], np.float32)[res["rows_test"]]
    res["target_val"] = np.asarray(va["targets"], np.float32)[res["rows_val"]]
    return res


def raw_metrics(tag: str, data: dict, res: dict, quantiles, median_idx: int) -> dict:
    """MASE / MAE / WQL per depth in RAW units (denormalize = RevIN inverse + trend add-back)."""
    from probing.timesfm3_last_token import denormalize
    te = data["test"]
    rows = res["rows_test"]
    X = np.asarray(te["X"], np.float64)[rows]
    y_raw = np.asarray(te["y_raw"], np.float64)[rows]
    out = {"mase_pw": [], "mae_pw": [], "wql_num_pw": [], "wql_den_pw": None,
           "n_denominator_clamped": None}
    for label in res["labels"]:
        z = np.asarray(res["pred_test"][label], np.float64)              # (n, Q, H)
        qraw = denormalize(z, mu=np.asarray(te["mu"])[rows], sd=np.asarray(te["sd"])[rows],
                           trend=np.asarray(te["trend"])[rows])
        m = raw_window_metrics(tag, X, y_raw, qraw, quantiles, median_idx)
        out["mase_pw"].append(m["mase_pw"])
        out["mae_pw"].append(m["mae_pw"])
        out["wql_num_pw"].append(m["wql_num_pw"])
        out["wql_den_pw"] = m["wql_den_pw"]
        out["n_denominator_clamped"] = m["n_denominator_clamped"]
    for k in ("mase_pw", "mae_pw", "wql_num_pw"):
        out[k] = np.stack(out[k])
    return out


def native_reference(tag: str, data: dict, res: dict, quantiles, median_idx: int) -> dict:
    """TimesFM-3's OWN decode() forecast, on the probes' axis and in raw units.

    The native head always emits its nine quantiles; under q1 the SAME forecast is scored on
    the median column alone (never re-run, never re-fit). Comparison only — never part of the
    tunnel (spec O).
    """
    from probing.timesfm3_last_token import NATIVE_MEDIAN_IDX
    from probing.timesfm3_last_token_probes import native_reference as _nat
    te = data["test"]
    # The native head always emits its nine quantiles. Under a smaller probe set we score the
    # SAME forecast on the levels the probe is scored on, by selecting those columns.
    cols = (np.arange(len(quantiles)) if len(quantiles) == 9
            else np.array([NATIVE_MEDIAN_IDX]))
    nat = _nat(te["native"], te["mu"], te["sd"], te["trend"], te["targets"], te["valid"],
               quantiles=quantiles, native_columns=cols)
    if not np.array_equal(np.asarray(nat["rows"]), res["rows_test"]):
        raise RuntimeError(f"{tag}: the native baseline and the probes are scored on different "
                           "test windows — refusing to report a comparison")
    rows = res["rows_test"]
    X = np.asarray(te["X"], np.float64)[rows]
    y_raw = np.asarray(te["y_raw"], np.float64)[rows]
    nat_raw = np.asarray(te["native"], np.float64)[rows][:, :, cols]      # (n, H, Q)
    qraw = np.ascontiguousarray(nat_raw.transpose(0, 2, 1))               # -> (n, Q, H)
    m = raw_window_metrics(tag, X, y_raw, qraw, quantiles, median_idx)
    return {"available": True, "loss": float(nat["q9_loss"]), "loss_window": nat["q9_window"],
            "mase": float(m["mase_pw"].mean()), "mase_window": m["mase_pw"],
            "mae": float(m["mae_pw"].mean()), "mae_window": m["mae_pw"],
            "wql_num_window": m["wql_num_pw"], "wql_den_window": m["wql_den_pw"],
            "per_quantile": nat["per_quantile"],
            "source": "timesfm3 decode() (cached at extraction), scored on the probe axis"}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def geometry_blocks(data: dict, res: dict, *, splits=("test", "train"), seed: int = SEED,
                    null_floor_reps: int = 3) -> list[dict]:
    """CKA + effective rank on the RAW token-15 states: one row per window, (N, 1280).

    The rows are restricted to the SAME windows the probes were scored on for the split in
    question, so a geometry row and a loss row describe the same observation.
    """
    spec = _spec()
    rows_by_split = {"test": res["rows_test"], "train": None, "val": res["rows_val"]}
    blocks = []
    for split in splits:
        d = data[split]
        rows = rows_by_split.get(split)
        if rows is None:
            rows = np.flatnonzero(np.asarray(d["valid"], bool))
        mats = [np.asarray(d["feats"][i], np.float64)[rows] for i in range(spec.n_points)]
        blocks.append(phase1.geometry_block(
            mats, spec.labels, d=spec.d, split=split, variant="headline",
            point_types=spec.point_types,        # all block_depth: L20 IS the head input
            seed=seed, null_floor_reps=null_floor_reps,
            row_construction=spec.geometry_rows))
    return blocks
