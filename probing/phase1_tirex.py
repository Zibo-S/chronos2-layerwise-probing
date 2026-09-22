"""Phase-1 TiRex adapter: two_pass, ONE shared Linear(512, Q*32) over both readouts, 14 depths.

WHERE TIREX'S OWN HEAD READS. The released ``tirex-ts`` package's DEFAULT inference path for
H=64 is TWO forward passes (``max_accelerated_rollout_steps=1``), each 64 tokens, each reading
its own token 63:

    pass 0  context x[T-512:T]              token 63 = the LAST REAL context patch  -> y[T:T+32]
    pass 1  context x[T-512:T] ++ 32 NaN    token 63 = an APPENDED MISSING patch    -> y[T+32:T+64]

``two_pass`` is therefore the primary Phase-1 definition (frozen 2026-09-20). ``single_pass``
stays available as the documented robustness mode; the measured comparison — identical tunnel
entrance at every tolerance, Spearman 0.991/1.000 over depths, bit-identical headline CKA —
is the evidence that the choice costs no conclusion.

TOKEN 63/64, NOT 15/16. ``_adjust_context_length`` forces every context to the checkpoint's
``train_ctx_len`` = 2048 by NaN LEFT-padding, so C=512 becomes 48 masked patches + 16 real ones
and the real patches sit at 48..63. Nothing here hardcodes that: ``TiRexGeometry`` derives it
and asserts ``readout_indices[0] == n_context_tokens - 1``.

THE PROBE. ONE shared ``Linear(512, Q*32)`` applied to BOTH forecast-producing states; the two
predicted patches concatenate to (B, Q, 64). Sharing is structural (a single ``nn.Linear`` on a
(B, K, 512) tensor) and mirrors ``output_patch_embedding``, which is likewise one head applied
to every token. ONE ``StandardScaler`` fit on the STACKED (2N, 512) TRAIN rows — a per-position
scaler would put a different affine map in front of each position and silently un-share the head.
At Q=9 the head's 288 outputs match the native head's own 288 = 9 x 32 width.

GEOMETRY — CONSTANT N, at every depth.
    HEADLINE  ``pos0``                  N rows (readout position 0), all 14 depths
    COMPANION ``all_positions_from_L1`` 2N rows (both states), L1..L12+RMS only
(Emb, position 1) is bit-identical across windows — the masked-future token reaches the patch
embedding as (values=0, mask=0) before any recurrence, so its embedding is a pure bias. A 2N-row
Emb would be N varying rows plus N copies of one point, which deflates its spectrum and distorts
its CKA row relative to every other depth. An N that changes with depth would also be
indistinguishable from a representational change at exactly the depth of interest, since both
the rank ceiling min(N-1, d) and the biased-CKA floor move with N. Hence two self-consistent
constant-N curves and no ``mixed`` curve at all.
"""

from __future__ import annotations

import numpy as np

from probing import phase1
from probing.config import SEED
from probing.phase1_metrics import raw_window_metrics

__all__ = ["MODEL", "EPOCHS", "LR", "WD_GRID", "ROLLOUT_MODE", "BACKEND", "extract",
           "fit_layerwise", "raw_metrics", "native_reference", "geometry_blocks"]

MODEL = "tirex"
EPOCHS = 300
LR = 1e-2
ROLLOUT_MODE = "two_pass"          # the released package's own default inference pathway
BACKEND = "torch"                  # xLSTM's `cuda` backend is a different kernel; pick ONE


#: THE PHASE-1 GRID — the SAME 13 candidates all three models use, for Q=9 and Q=1 alike.
#: ``tirex_probes.WD_GRID_TIREX`` stays that LINE's own grid, untouched; Phase 1 asserts the new
#: grid is a strict superset of it. On the committed Phase-1 cell (tirex x Electricity) TiRex was
#: the ONE model that did not clip — its optimum is interior at wd=10 — so for TiRex the three
#: new candidates are expected to stay unselected. That is the point: the same search space for
#: all three models means a model difference cannot be a grid difference.
WD_GRID = phase1.PHASE1_WD_GRID


def _assert_superset():
    """The no-regression rule, checked lazily (the legacy grid lives behind a torch import)."""
    from probing.tirex_probes import WD_GRID_TIREX
    if not phase1.wd_grid_is_superset_of(WD_GRID, WD_GRID_TIREX):
        raise RuntimeError("the Phase-1 grid must retain every legacy TiRex candidate; "
                           f"{WD_GRID} is not a superset of {WD_GRID_TIREX}")
    return True


def _spec():
    return phase1.model_spec(MODEL)


# --------------------------------------------------------------------------- #
# extraction + targets
# --------------------------------------------------------------------------- #
def extract(tag: str, w: dict, *, model, geom, cache_dir, checkpoint, backend=BACKEND,
            batch_size: int = 256, mode: str = ROLLOUT_MODE, seed: int = SEED,
            splits=("train", "val", "test"), verbose: bool = False) -> tuple[dict, dict]:
    """Readout states + the native forecast + the per-pass-normalized targets, per split.

    ``batch_size`` is a REPRODUCIBILITY parameter, not a speed knob, and it is part of the cache
    key: TiRex's sLSTM recurrence runs in bfloat16 and a small-batch GEMM kernel switch perturbs
    block 1 by ~1e-3 of its std, which 64 timesteps x 12 blocks then amplify (measured on MPS:
    batch 4 and 8 bitwise identical, batch 2 not). A differing batch size REJECTS the cache
    rather than mixing two slightly different representations.
    """
    from probing.tirex_model import build_targets, cached_features
    out, ex = {}, {"rollout_mode": mode, "backend": backend, "batch_size": int(batch_size),
                   "cache_hits": {}, "feature_shapes": {},
                   "batch_size_is_part_of_cache_key": True,
                   "batch_size_reason": ("sLSTM runs in bfloat16; a small-batch GEMM kernel "
                                         "switch measurably perturbs the representation, so "
                                         "features must not be reused across batch sizes")}
    for split in splits:
        Xk, Yk, Sk = f"X_{split}", f"Y_{split}_traj", f"series_{split}"
        if Xk not in w:
            raise KeyError(f"{tag}: the window dict has no {Xk!r} — Phase-1 needs a dedicated "
                           "validation split")
        X = np.asarray(w[Xk], np.float32)
        feats, native, meta, hit = cached_features(
            tag, split, X, model, geom, cache_dir=cache_dir, checkpoint=checkpoint,
            backend=backend, seed=seed, batch_size=batch_size, verbose=verbose, mode=mode)
        y_raw = _raw_future(w, split)
        # build_targets returns a (targets, loc, scale) TUPLE (its established contract, shared
        # with run_tirex_probing); destructure it -- indexing it like a dict raises TypeError.
        tgt, loc, scale = build_targets(X, y_raw, model, geom, mode=mode)
        out[split] = {"feats": feats, "native": native, "X": X, "y_raw": y_raw,
                      "targets": tgt, "loc": loc, "scale": scale,
                      "series": np.asarray(w[Sk], np.int64)}
        ex["cache_hits"][split] = bool(hit)
        ex["feature_shapes"][split] = list(np.shape(feats[_spec().labels[-1]]))
        ex.setdefault("cache_metadata", {})[split] = meta
    ex["representation_points"] = _spec().labels
    ex["readout_tokens"] = list(getattr(geom, "readout_indices", []) or [])
    return out, ex


def _raw_future(w, split):
    """The raw-unit future for this split, the exact inverse of id_data's label transform."""
    from probing.timesfm3 import raw_future_from_arcsinh
    return raw_future_from_arcsinh(w[f"X_{split}"], w[f"Y_{split}_traj"],
                                   w["meta"]["sigma_eps"])


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def fit_layerwise(tag: str, data: dict, *, quantiles, median_idx: int, geom, device: str,
                  epochs: int = EPOCHS, lr: float = LR, wd_grid=None, seed: int = SEED,
                  verbose: bool = True) -> dict:
    """Fit + score the shared patch probe at every depth. Train fits, val selects, test once."""
    from probing.tirex_probes import (constant_forecast_floor, fit_shared_patch_probe,
                                      predict_quantiles)
    spec = _spec()
    _assert_superset()
    wd_grid = phase1.assert_wd_grid(WD_GRID if wd_grid is None else wd_grid, lr,
                                    null_wd=phase1.PHASE1_WD_NULL)
    grid_max, grid_min = float(max(wd_grid)), float(min(wd_grid))
    q = np.asarray(quantiles, np.float64)
    tr, va, te = data["train"], data["val"], data["test"]

    res = {"labels": spec.labels, "quantiles": [float(x) for x in q], "median_index": median_idx,
           "train_loss": [], "val_loss": [], "test_loss": [], "wd": [], "selection": [],
           "n_params": [], "wd_at_grid_max": [], "wd_at_grid_min": [],
           "val_window_loss": [], "test_window_loss": [], "per_quantile_test": [],
           "pred_val": {}, "pred_test": {}, "probe_weights": {}}

    for label in spec.labels:
        fit = fit_shared_patch_probe(tr["feats"][label], tr["targets"],
                                     va["feats"][label], va["targets"],
                                     out_patch=geom.output_patch, wd_grid=wd_grid, epochs=epochs,
                                     lr=lr, device=device, init_seed=seed, quantiles=q)
        preds = {s: predict_quantiles(fit, d["feats"][label], device=device)
                 for s, d in (("val", va), ("test", te))}
        pw = {s: phase1.mean_pinball_per_window(preds[s], d["targets"], q)
              for s, d in (("val", va), ("test", te))}
        res["train_loss"].append(float(fit["train_loss"]))
        res["val_loss"].append(float(pw["val"].mean()))
        res["test_loss"].append(float(pw["test"].mean()))
        res["val_window_loss"].append(pw["val"])
        res["test_window_loss"].append(pw["test"])
        res["per_quantile_test"].append(
            phase1.per_quantile_mean_loss(preds["test"], te["targets"], q))
        res["wd"].append(float(fit["wd"]))
        res["wd_at_grid_max"].append(bool(float(fit["wd"]) == grid_max))
        res["wd_at_grid_min"].append(bool(float(fit["wd"]) == grid_min))
        res["selection"].append({str(k): float(v)
                                 for k, v in fit["selection"]["val_loss_by_wd"].items()})
        res["n_params"].append(int(fit["n_params"]))
        res["pred_val"][label] = preds["val"]
        res["pred_test"][label] = preds["test"]
        res["probe_weights"][label] = {
            "weight": fit["probe"].weight.detach().cpu().numpy().astype(np.float32),
            "bias": fit["probe"].bias.detach().cpu().numpy().astype(np.float32),
            "scaler_mean": np.asarray(fit["scaler"].mean_, np.float64),
            "scaler_scale": np.asarray(fit["scaler"].scale_, np.float64)}
        if verbose:
            print(f"    [probe] {label:<8s} wd={fit['wd']:<8g} "
                  f"train={res['train_loss'][-1]:.5f} val={res['val_loss'][-1]:.5f} "
                  f"test={res['test_loss'][-1]:.5f}", flush=True)

    res["val_window_loss"] = np.stack(res["val_window_loss"])
    res["test_window_loss"] = np.stack(res["test_window_loss"])
    res["probe_description"] = (f"Linear(512, {len(q)}*{geom.output_patch}) shared across the "
                                f"{geom.n_forecast_patches} native forecast-producing states")
    # The closed-form no-information floor (best CONSTANT forecast under this objective). Its
    # tau=0.5 form is the per-step train median; for Q>1 the optimum is the per-step train
    # QUANTILE, which is what this computes. No fit, so no optimizer artifact can reach it.
    res["wd_grid"] = [float(w) for w in wd_grid]
    res["constant_forecast_floor"] = phase1.constant_forecast_floor(
        tr["targets"], {"train": tr["targets"], "val": va["targets"], "test": te["targets"]}, q)
    if len(q) == 1:
        res["constant_forecast_floor_q1_reference"] = constant_forecast_floor(
            tr["targets"], te["targets"], va["targets"])
    return res


# --------------------------------------------------------------------------- #
# raw-unit metrics + native baseline
# --------------------------------------------------------------------------- #
def raw_metrics(tag: str, data: dict, res: dict, quantiles, median_idx: int, geom) -> dict:
    """MASE / MAE / WQL per depth in RAW units. Denormalization is PER OUTPUT PATCH.

    In two_pass mode each of the two output patches carries its OWN (loc, scale), so a single
    global rescale would silently mix two normalized spaces. ``tirex_model.denormalize`` does
    the per-patch inverse and REFUSES flat (n,) statistics.
    """
    from probing.tirex_model import denormalize
    te = data["test"]
    X, y_raw = te["X"], te["y_raw"]
    out = {"mase_pw": [], "mae_pw": [], "wql_num_pw": [], "wql_den_pw": None,
           "n_denominator_clamped": None}
    for label in res["labels"]:
        z = np.asarray(res["pred_test"][label], np.float64)               # (n, Q, H)
        qraw = np.stack([denormalize(z[:, k, :], te["loc"], te["scale"], geom)
                         for k in range(z.shape[1])], axis=1)             # per-quantile row
        m = raw_window_metrics(tag, X, y_raw, qraw, quantiles, median_idx)
        out["mase_pw"].append(m["mase_pw"])
        out["mae_pw"].append(m["mae_pw"])
        out["wql_num_pw"].append(m["wql_num_pw"])
        out["wql_den_pw"] = m["wql_den_pw"]
        out["n_denominator_clamped"] = m["n_denominator_clamped"]
    for k in ("mase_pw", "mae_pw", "wql_num_pw"):
        out[k] = np.stack(out[k])
    return out


def native_reference(tag: str, data: dict, quantiles, median_idx: int, geom) -> dict:
    """TiRex's OWN forecast on these windows, scored on the probes' axis and in raw units.

    The native path returns (n, H, Q) RAW quantiles; they are re-normalized PER OUTPUT PATCH
    (the exact inverse of ``denormalize``) so probe and baseline are never compared across two
    different spaces. Comparison only — never part of the tunnel (spec O).
    """
    from probing.tirex_model import normalize_raw
    te = data["test"]
    nq = np.asarray(te["native"], np.float64)                             # (n, H, Q_native)
    if nq.ndim != 3 or nq.shape[1] != geom.H:
        raise ValueError(f"native forecasts must be (n, {geom.H}, Q), got {nq.shape}")
    q = np.asarray(quantiles, np.float64)
    if len(q) == geom.num_quantiles:
        cols = np.arange(geom.num_quantiles)
    else:                                   # score the SAME forecast on the probe's levels
        cols = np.array([geom.median_index])
    qraw = np.ascontiguousarray(nq[:, :, cols].transpose(0, 2, 1))        # (n, Q, H)
    znorm = np.stack([normalize_raw(qraw[:, k, :], te["loc"], te["scale"], geom)
                      for k in range(qraw.shape[1])], axis=1)
    pw = phase1.mean_pinball_per_window(znorm, te["targets"], q)
    m = raw_window_metrics(tag, te["X"], te["y_raw"], qraw, q, median_idx)
    return {"available": True, "loss": float(pw.mean()), "loss_window": pw,
            "mase": float(m["mase_pw"].mean()), "mase_window": m["mase_pw"],
            "mae": float(m["mae_pw"].mean()), "mae_window": m["mae_pw"],
            "wql_num_window": m["wql_num_pw"], "wql_den_window": m["wql_den_pw"],
            "source": f"tirex native forecast ({ROLLOUT_MODE}), cached at extraction"}


# --------------------------------------------------------------------------- #
# geometry — headline pos0 (constant N, all depths) + companion (2N, from L1)
# --------------------------------------------------------------------------- #
def geometry_blocks(data: dict, *, splits=("test", "train"), seed: int = SEED,
                    null_floor_reps: int = 3) -> list[dict]:
    """Both constant-N variants, both splits, both estimators, on RAW readout states."""
    from probing.tirex_geometry import (VARIANT_ALL_FROM_L1, VARIANT_POS0, DEGENERACY_CAVEAT,
                                        degenerate_readouts, variant_spec)
    spec = _spec()
    blocks = []
    for split in splits:
        feats = data[split]["feats"]
        K = int(np.asarray(feats[spec.labels[0]]).shape[1])
        deg = degenerate_readouts(feats, spec.labels)
        for variant in (VARIANT_POS0, VARIANT_ALL_FROM_L1):
            names, positions = variant_spec(variant, spec.labels, K)
            # L12+RMS is a head-input diagnostic in BOTH variants; the companion additionally
            # drops Emb (whose second readout state is degenerate before any recurrence).
            types = [spec.points[spec.labels.index(n)].point_type for n in names]
            mats = [phase1.stack_rows(np.asarray(feats[n], np.float64)[:, list(positions), :])
                    for n in names]
            b = phase1.geometry_block(
                mats, names, d=spec.d, split=split, variant=variant, point_types=types,
                seed=seed, null_floor_reps=null_floor_reps,
                row_construction=(f"readout positions {list(positions)} stacked row-major "
                                  f"over depths {names[0]}..{names[-1]}"))
            b["is_headline"] = (variant == VARIANT_POS0)
            b["degeneracy"] = deg
            b["degeneracy_caveat"] = DEGENERACY_CAVEAT
            blocks.append(b)
    return blocks
