"""Phase-1 Chronos-2 adapter: K=4 forecast slots, ONE shared Linear(768, Q*16), 14 depths.

WHERE CHRONOS-2'S OWN HEAD READS. The native output head consumes the K = ceil(H/P) = 4
forecast-slot token states of a ``num_output_patches=K`` encoder pass and emits, per slot, one
output patch of Q quantiles x P=16 steps, which are concatenated along the horizon. The Phase-1
probe is the strictly-linear analogue of exactly that: ONE ``Linear(768, Q*16)``, the SAME
weight tensor applied to all four slots, patches concatenated to (B, Q, 64).

Sharing is STRUCTURAL, not a convention — the probe is a single ``nn.Linear`` applied to an
(n, K, 768) tensor, so there is one weight tensor and it cannot differ between slots. This is
``probes.fit_shared_forecast_probe_explicit_val`` / ``_apply_shared_head``, unchanged: the
validated Chronos-2 code path, called with ``quantiles=`` the canonical nine.

REPRESENTATION POINTS (14): Emb (L0, the embedded token sequence entering block 1), L1..L12
(block outputs), and L12+LN (the encoder's final_layer_norm output — the actual tensor the
native head consumes). ``extraction.extract_kout_features`` returns exactly these as
``feats["fslot"][0..12]`` plus ``final["fslot"]``.

TWO NOTES ON THE TRAINING OBJECTIVE, both pinned by Phase-1 contract tests:
  * ``fit_shared_forecast_probe_explicit_val`` trains and selects with Chronos-2's own
    ``chronos2_quantile_loss``, which sums the pinball terms over quantiles instead of averaging
    them. That is exactly 2Q times the Phase-1 mean pinball, so the weight-decay argmin and the
    tunnel's ratio test are unchanged, and AdamW (gradient normalized by its own second moment,
    decay decoupled from the loss) fits the same probe either way.
  * every REPORTED number here is recomputed as the common mean pinball from the one saved
    (n, Q, H) prediction tensor, so nothing downstream depends on the training convention.

GEOMETRY. Row construction is ``probing.cka.stack_slots``: (n, 4, 768) -> (4n, 768), row-major,
row w*4+k = window w, slot k. That is the construction the committed Chronos-2 CKA analysis
(``run_cka_analysis --extv4-fslot``) already uses, so the Phase-1 matrices are directly
comparable with it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from probing import phase1
from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE, SEED
from probing.phase1_metrics import raw_window_metrics
from probing.probes import (WD_GRID_V2, _apply_shared_head, _slot_transform,
                            fit_shared_forecast_probe_explicit_val, validate_quantiles)
# The exact inverse of id_data._make_examples' label transform (mu + sigma*sinh). It lives in
# probing/timesfm3.py because the TimesFM-3 line needed to un-transform Chronos-2-built labels
# first; it is pure numpy, model-free, and a Phase-1 contract test pins it as the round-trip
# inverse of id_data's forward transform. Imported, never re-implemented.
from probing.timesfm3 import raw_future_from_arcsinh

__all__ = ["MODEL", "EPOCHS", "LR", "WD_GRID", "POINT_KEYS", "extract", "fit_layerwise",
           "geometry_blocks", "native_reference"]

MODEL = "chronos2"
EPOCHS = 300
LR = 1e-2
#: The Chronos-2 line's own shared grid (``probes.WD_GRID_V2``), untouched. It is deliberately
#: NOT the TimesFM-3 / TiRex grid: those were extended to 10/30 after MEASURING grid-max
#: clipping on their own much larger probes (737k and a rank-deficient recurrent state). The
#: Chronos-2 shared-slot probe is 768*144 + 144 = 110,736 parameters on 4*1394 = 5576 slot rows,
#: the most favourable ratio of the three, and the committed runs do not clip. The driver reports
#: clipping per (dataset, depth) regardless, so if it does clip here that is a finding to act on,
#: not something to paper over.
WD_GRID = WD_GRID_V2

#: Feature-dict keys of the 14 representation points, in depth order. Keys 0..12 are the
#: block-hook states (L0 = Emb, L1..L12 = block outputs); key 13 = NUM_LAYERS is the
#: post-final-LayerNorm state, the native head's actual input.
#:
#: Integer keys, NOT a mixed int/"final": ``fit_shared_forecast_probe_explicit_val`` iterates
#: ``sorted(train_feats)``, which a mixed-type key set makes a TypeError. 13 is also exactly the
#: convention the committed Chronos-2 fslot line already uses
#: (run_ptood_probing_ftok._fslot_feats), so the two lines index the same 14 points identically.
POINT_KEYS = list(range(NUM_LAYERS + 1))
FINAL_KEY = NUM_LAYERS


def _spec():
    return phase1.model_spec(MODEL)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def extract(tag: str, w: dict, *, horizon: int = phase1.PHASE1_H, batch_size: int = 128,
            splits=("train", "val", "test")) -> dict:
    """K-slot forecast states for each split, plus the raw-unit targets.

    Returns ``{split: {"feats": {point_key: (n, K, 768)}, "X": (n, C), "Y": (n, H) arcsinh,
    "y_raw": (n, H), "series": (n,)}}`` and an ``extraction`` record.

    ``extract_kout_features`` owns the cache (``features_cache/IDF_<tag>__<split>__clean__
    K<K>_H<H>.npz``) and FAILS LOUDLY if the cached window labels do not match these windows, so
    a re-windowing can never be silently reused.
    """
    from probing.extraction import extract_kout_features

    K = math.ceil(horizon / OUTPUT_PATCH_SIZE)
    meta = w["meta"]
    out, ex = {}, {"K": K, "output_patch_size": OUTPUT_PATCH_SIZE, "cache_hits": {},
                   "feature_shapes": {}}
    for split in splits:
        Xk, Yk, Sk = f"X_{split}", f"Y_{split}_traj", f"series_{split}"
        if Xk not in w:
            raise KeyError(f"{tag}: the window dict has no {Xk!r} — Phase-1 needs a dedicated "
                           "validation split (the rolling-origin protocol provides one)")
        X, Y = np.asarray(w[Xk], np.float32), np.asarray(w[Yk], np.float32)
        feats, final, _y = extract_kout_features(tag, split, X, Y, horizon,
                                                 batch_size=batch_size)
        pts = {i: np.asarray(feats["fslot"][i]) for i in range(NUM_LAYERS)}
        pts[FINAL_KEY] = np.asarray(final["fslot"])
        for key, arr in pts.items():
            if arr.ndim != 3 or arr.shape[1] != K or arr.shape[2] != _spec().d:
                raise RuntimeError(f"{tag}/{split}/{key}: forecast-slot features are "
                                   f"{arr.shape}, expected (n, {K}, {_spec().d})")
        out[split] = {"feats": pts, "X": X, "Y": Y,
                      "y_raw": raw_future_from_arcsinh(X, Y, meta["sigma_eps"]),
                      "series": np.asarray(w[Sk], np.int64)}
        ex["feature_shapes"][split] = list(pts[FINAL_KEY].shape)
    ex["representation_points"] = _spec().labels
    ex["note"] = ("one num_output_patches=K encoder pass per batch; the K forecast-slot token "
                  "states are the native head's own input")
    return out, ex


# --------------------------------------------------------------------------- #
# prediction from a fitted probe -- ONE tensor everything else derives from
# --------------------------------------------------------------------------- #
def predict_quantiles(fitted_point: dict, feats, H: int, quantiles, device) -> np.ndarray:
    """Frozen probe -> (n, Q, H) prediction in Chronos-2's normalized (arcsinh) target space.

    Deliberately the ONE place a Chronos-2 Phase-1 number comes from: the loss, the per-window
    loss, the median for MASE and the quantiles for WQL are all derived from this tensor, so
    they cannot disagree about which prediction they describe. It applies the validated
    ``_apply_shared_head`` to the validated ``_slot_transform`` of the train-fitted scaler; a
    contract test pins that the loss computed from it equals
    ``probes.predict_shared_forecast_probe``'s own loss divided by 2Q.
    """
    Q, P = len(quantiles), int(fitted_point["output_patch_size"])
    lin = fitted_point["linear"].eval()
    # Defensive prediction-device matching: put the input where the weights ALREADY are, rather
    # than trusting the passed ``device``. fit_shared_forecast_probe_explicit_val resolves a None
    # device to cuda internally, so a caller that also passes None here would otherwise build the
    # input on cpu and hit a CPU/CUDA addmm mismatch. The ``device`` argument is retained for
    # signature stability but the fitted weights' device is authoritative.
    dev = next(lin.parameters()).device
    X = torch.as_tensor(_slot_transform(fitted_point["scaler"], np.asarray(feats, np.float32)),
                        dtype=torch.float32, device=dev)
    with torch.no_grad():
        return _apply_shared_head(lin, X, Q, P, H).cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# the layer-wise sweep
# --------------------------------------------------------------------------- #
def fit_layerwise(tag: str, data: dict, *, quantiles, median_idx: int, device: str,
                  epochs: int = EPOCHS, lr: float = LR, wd_grid=WD_GRID, seed: int = SEED,
                  horizon: int = phase1.PHASE1_H, verbose: bool = True) -> dict:
    """Fit + score one shared-slot probe at every depth. Train fits, val selects, test once.

    The scaler and the Linear are both fit on FULL train; each weight-decay candidate is scored
    on the DEDICATED validation split and the chosen-wd full-train model is kept with no refit
    (``fit_quantile_probe_explicit_val``'s contract). Validation never touches the scaler or the
    weights, and the test split is read exactly once, after everything is frozen.
    """
    spec = _spec()
    # Resolve the device ONCE, concretely, so the fit and every predict call agree. Mirrors the
    # idiom in probes.fit_shared_forecast_probe_explicit_val (which resolves None -> cuda
    # internally); resolving here keeps the two from desyncing when --device is left unset.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q = validate_quantiles(quantiles)
    Q = len(q)
    tr, va, te = data["train"], data["val"], data["test"]

    fitted = fit_shared_forecast_probe_explicit_val(
        tr["feats"], tr["Y"], va["feats"], va["Y"], quantiles=q, epochs=epochs, lr=lr,
        wd_grid=tuple(wd_grid), device=device, init_seed=seed,
        output_patch_size=OUTPUT_PATCH_SIZE)

    res = {"labels": spec.labels, "quantiles": [float(x) for x in q], "median_index": median_idx,
           "train_loss": [], "val_loss": [], "test_loss": [], "wd": [], "selection": [],
           "n_params": [], "wd_at_grid_max": [], "wd_at_grid_min": [],
           "val_window_loss": [], "test_window_loss": [], "per_quantile_test": [],
           "pred_val": {}, "pred_test": {},
           "probe_weights": {}}
    grid_max, grid_min = float(max(wd_grid)), float(min(wd_grid))

    for key, label in zip(POINT_KEYS, spec.labels):
        f = fitted[key]
        preds = {s: predict_quantiles(f, d["feats"][key], horizon, q, device)
                 for s, d in (("train", tr), ("val", va), ("test", te))}
        losses = {s: phase1.mean_pinball_per_window(preds[s], d["Y"], q)
                  for s, d in (("train", tr), ("val", va), ("test", te))}
        res["train_loss"].append(float(losses["train"].mean()))
        res["val_loss"].append(float(losses["val"].mean()))
        res["test_loss"].append(float(losses["test"].mean()))
        res["val_window_loss"].append(losses["val"])
        res["test_window_loss"].append(losses["test"])
        res["per_quantile_test"].append(phase1.per_quantile_mean_loss(preds["test"], te["Y"], q))
        res["wd"].append(float(f["wd"]))
        res["wd_at_grid_max"].append(bool(float(f["wd"]) == grid_max))
        res["wd_at_grid_min"].append(bool(float(f["wd"]) == grid_min))
        res["selection"].append({str(k): float(v) for k, v in
                                 (f["selection"] or {}).get("val_loss_by_wd", {}).items()})
        res["n_params"].append(int(sum(p.numel() for p in f["linear"].parameters())))
        res["pred_val"][label] = preds["val"]
        res["pred_test"][label] = preds["test"]
        res["probe_weights"][label] = {
            "weight": f["linear"].weight.detach().cpu().numpy().astype(np.float32),
            "bias": f["linear"].bias.detach().cpu().numpy().astype(np.float32),
            "scaler_mean": np.asarray(f["scaler"].mean_, np.float64),
            "scaler_scale": np.asarray(f["scaler"].scale_, np.float64)}
        if verbose:
            print(f"    [probe] {label:<8s} wd={f['wd']:<8g} "
                  f"train={res['train_loss'][-1]:.5f} val={res['val_loss'][-1]:.5f} "
                  f"test={res['test_loss'][-1]:.5f}", flush=True)

    res["val_window_loss"] = np.stack(res["val_window_loss"])
    res["test_window_loss"] = np.stack(res["test_window_loss"])
    res["probe_description"] = f"Linear(768, {Q}*{OUTPUT_PATCH_SIZE}) shared across K=4 slots"
    return res


# --------------------------------------------------------------------------- #
# raw-unit metrics + the native baseline
# --------------------------------------------------------------------------- #
def raw_metrics(tag: str, data: dict, res: dict, quantiles, median_idx: int) -> dict:
    """MASE / MAE / WQL per depth, in RAW units. Un-transforms with mu + sigma*sinh."""
    te = data["test"]
    X, Y = te["X"], te["Y"]
    mu = np.asarray(X, np.float64).mean(axis=1)
    sd = np.maximum(np.asarray(X, np.float64).std(axis=1), 1e-6)
    out = {"mase_pw": [], "mae_pw": [], "wql_num_pw": [], "wql_den_pw": None,
           "n_denominator_clamped": None}
    for label in res["labels"]:
        z = np.asarray(res["pred_test"][label], np.float64)               # (n, Q, H) arcsinh
        qraw = mu[:, None, None] + sd[:, None, None] * np.sinh(z)         # sinh is monotone ->
        m = raw_window_metrics(tag, X, te["y_raw"], qraw, quantiles, median_idx)  # order kept
        out["mase_pw"].append(m["mase_pw"])
        out["mae_pw"].append(m["mae_pw"])
        out["wql_num_pw"].append(m["wql_num_pw"])
        out["wql_den_pw"] = m["wql_den_pw"]
        out["n_denominator_clamped"] = m["n_denominator_clamped"]
    for k in ("mase_pw", "mae_pw", "wql_num_pw"):
        out[k] = np.stack(out[k])
    return out


def native_reference(tag: str, w: dict, data: dict, quantiles, median_idx: int,
                     batch_size: int = 128) -> dict | None:
    """Chronos-2's OWN forecast on these exact test windows, for comparison only.

    Never part of tunnel selection (spec O). Uses the pipeline's ``predict_quantiles`` at the
    Phase-1 levels, scored with the SAME mean pinball in the SAME normalized space as the
    probes (raw -> arcsinh via the window's own mu/sigma) and the SAME raw-unit MASE/WQL.

    Returns None if the pipeline cannot be loaded (e.g. a CPU-only audit run); the cell is still
    valid, the native comparison is simply absent and recorded as such.
    """
    try:
        from probing.extraction import get_pipeline
        pipeline, _cfg = get_pipeline()
    except Exception as exc:                                  # pragma: no cover - env dependent
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    te = data["test"]
    X = np.asarray(te["X"], np.float32)
    H = int(te["Y"].shape[1])
    q = np.asarray(quantiles, np.float64)
    # Chronos2Pipeline.predict_quantiles returns (quantiles, mean): quantiles is a LIST with one
    # entry per input series, each of shape (n_variates, H, Q). Probing is univariate, so every
    # entry must be (1, H, Q) -- assert that per-series shape EXPLICITLY (or the (H, Q) it squeezes
    # to) rather than reshaping, so an unexpected variate count fails loudly instead of being
    # silently coerced. Pass numpy rows (list(X[...])) so the pipeline places them on its device.
    per_series = []
    Qn = len(q)
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            quantiles_out, _mean = pipeline.predict_quantiles(
                list(X[s:s + batch_size]), prediction_length=H,
                quantile_levels=[float(v) for v in q])
            for qt in quantiles_out:
                a = np.asarray(qt.cpu() if hasattr(qt, "cpu") else qt, np.float64)
                if a.shape == (1, H, Qn):
                    a = a[0]
                elif a.shape != (H, Qn):
                    raise RuntimeError(
                        f"native per-series forecast is {a.shape}, expected (1, {H}, {Qn}) or "
                        f"({H}, {Qn}) -- univariate Chronos-2 output; refusing to reshape it")
                per_series.append(a)
    raw = np.stack(per_series, axis=0)                    # (n, H, Q)
    if raw.shape != (len(X), H, Qn):
        raise RuntimeError(f"native forecast is {raw.shape}, expected ({len(X)}, {H}, {Qn})")
    qraw = np.ascontiguousarray(raw.transpose(0, 2, 1))   # (n, Q, H) -- the project's layout

    mu = np.asarray(X, np.float64).mean(axis=1)
    sd = np.maximum(np.asarray(X, np.float64).std(axis=1), 1e-6)
    z = np.arcsinh((qraw - mu[:, None, None]) / sd[:, None, None])
    pw = phase1.mean_pinball_per_window(z, te["Y"], q)
    m = raw_window_metrics(tag, X, te["y_raw"], qraw, q, median_idx)
    return {"available": True, "loss": float(pw.mean()), "loss_window": pw,
            "mase": float(m["mase_pw"].mean()), "mase_window": m["mase_pw"],
            "mae": float(m["mae_pw"].mean()), "mae_window": m["mae_pw"],
            "wql_num_window": m["wql_num_pw"], "wql_den_window": m["wql_den_pw"],
            "source": "Chronos2Pipeline.predict_quantiles at the Phase-1 quantile levels"}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def geometry_blocks(data: dict, *, splits=("test", "train"), seed: int = SEED,
                    null_floor_reps: int = 3) -> list[dict]:
    """CKA + effective rank on the RAW forecast-slot states, both splits, both estimators.

    Rows are the forecast slots stacked row-major, (n, 4, 768) -> (4n, 768) — the committed
    Chronos-2 construction. Both splits are produced because the two committed Chronos-2
    analyses disagree about which one is the headline (``run_cka_analysis`` uses test,
    ``run_spectral`` uses train); saving both means neither has to be re-run.
    """
    spec = _spec()
    blocks = []
    for split in splits:
        feats = data[split]["feats"]
        mats = [phase1.stack_rows(np.asarray(feats[k], np.float64)) for k in POINT_KEYS]
        blocks.append(phase1.geometry_block(
            mats, spec.labels, d=spec.d, split=split, variant="headline",
            point_types=spec.point_types,        # L12+LN is a head input, not a 13th depth
            seed=seed, null_floor_reps=null_floor_reps,
            row_construction=spec.geometry_rows))
    return blocks
