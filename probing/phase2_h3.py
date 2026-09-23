"""H3 — can an intermediate representation functionally REPLACE the final one?

One dataset-specific cell (model x dataset), every block depth, offline from the Phase-1 caches:

    native     f(h_L)                           the full model (horizontal reference)
    hard       f(h_l)                           raw state through the frozen native pathway
    noa        f(a_l(h_l)), a_l label-free      native-output alignment (PRIMARY)
    fl         f(a_l(h_l)), a_l supervised      forecast-loss alignment (Chronos-2, TiRex)
    ra         f(a_l(h_l)), a_l hidden ridge    diagnostic (representation R^2 in the hidden metric)
    probe      Phase-1 linear probe             recoverability reference (READ from Phase 1)

"hard" says whether the raw state is already usable by the pretrained pathway; "probe" says
whether the forecast is recoverable at all; "noa"/"fl" say whether a lightweight transformation
bridges the difference. A strong probe with a weak hard cut is RECOVERABILITY WITHOUT
REPLACEABILITY; alignment then tests whether the final representation's forecasting role can be
approximated without executing the remaining blocks.

Train fits, validation selects, test is read once. Nothing here loads a model or touches a cache:
the pathway and the data are injected (``probing.phase2_pathways``), so the whole cell runs
end-to-end on a mock pathway in the model-free contracts.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from probing import phase1
from probing.phase1_metrics import raw_window_metrics
from probing.phase2 import (ADAPTER_EPOCHS, ADAPTER_EVAL_EVERY, ADAPTER_LR, ADAPTER_WD_GRID,
                            BOOT_B, BUDGET, BUDGET_RULE, FL_MODELS, FL_TIMESFM3_THEOREM,
                            FRONTIER_BUDGETS, H3_FAMILIES, Phase1DependencyError, budget_depth,
                            HEADLINE_TOL, LOW_SKILL_RATIO, MEDIAN_INDEX, QUANTILES, RIDGE_KAPPAS,
                            SEED, TUNNEL_TOLS, array_sha256, cluster_replicates,
                            compatibility_tunnel, ratio_summary, standard_wql)
from probing.phase2_align import (ResidualAdapter, affine_to_adapter, anchored_ridge_path,
                                  fit_residual_adapter, mse_loss, pinball_mean, save_adapter)

__all__ = ["DEFAULT_FIT_CFG", "run_h3_cell", "save_h3_cell", "native_gate", "families_for"]

DEFAULT_FIT_CFG = {"epochs": ADAPTER_EPOCHS, "lr": ADAPTER_LR, "wd_grid": list(ADAPTER_WD_GRID),
                   "eval_every": ADAPTER_EVAL_EVERY, "kappas": list(RIDGE_KAPPAS), "seed": SEED,
                   "native_gate_rtol": 1e-4, "boot_b": BOOT_B, "boot_seed": SEED}

METRICS = ("loss", "mase", "mae", "wql")


def families_for(model: str, requested) -> tuple[list[str], dict]:
    """The families that exist for this model, and why any requested one was dropped."""
    fams, dropped = [], {}
    for f in requested:
        if f not in H3_FAMILIES:
            raise ValueError(f"unknown H3 family {f!r}; known: {H3_FAMILIES}")
        if f == "fl" and model not in FL_MODELS:
            dropped[f] = FL_TIMESFM3_THEOREM
            continue
        fams.append(f)
    return fams, dropped


def native_gate(ours_loss, ours_mase, phase1_arrays: dict, rtol: float) -> dict:
    """The offline native pathway must reproduce the Phase-1 native baseline on the SAME rows.

    Phase 1 scored each model's own forecast (Chronos-2 pipeline, TimesFM-3 decode(), TiRex's
    two-pass default path). Phase 2 recomputes it offline as f(h_L) from the cached states. The two
    are not bitwise (different GEMM shapes / batching; float32 vs float64 inverses) but must agree
    to ``rtol`` on the mean test loss and mean MASE, or the cache / pathway / rows are wrong. The
    per-window maxima are recorded -- measured, never assumed to be 0.0.
    """
    ref_l = phase1_arrays.get("native_loss_window")
    ref_m = phase1_arrays.get("native_mase_window")
    if ref_l is None or ref_m is None:
        return {"available": False, "passed": None,
                "reason": "the Phase-1 cell saved no native baseline (--no-native)"}
    ours_loss, ours_mase = np.asarray(ours_loss, np.float64), np.asarray(ours_mase, np.float64)
    ref_l, ref_m = np.asarray(ref_l, np.float64), np.asarray(ref_m, np.float64)
    if ours_loss.shape != ref_l.shape:
        return {"available": True, "passed": False,
                "reason": f"row mismatch {ours_loss.shape} vs {ref_l.shape}"}
    rl = abs(ours_loss.mean() - ref_l.mean()) / max(abs(ref_l.mean()), 1e-12)
    rm = abs(ours_mase.mean() - ref_m.mean()) / max(abs(ref_m.mean()), 1e-12)
    dl, dm = np.abs(ours_loss - ref_l), np.abs(ours_mase - ref_m)
    return {"available": True, "rtol": float(rtol),
            "test_loss_mean_ours": float(ours_loss.mean()),
            "test_loss_mean_phase1": float(ref_l.mean()), "test_loss_mean_rel_diff": float(rl),
            "test_loss_window_max_abs_diff": float(dl.max()),
            "test_mase_mean_ours": float(ours_mase.mean()),
            "test_mase_mean_phase1": float(ref_m.mean()), "test_mase_mean_rel_diff": float(rm),
            "test_mase_window_max_abs_diff": float(dm.max()),
            "test_mase_window_max_rel_diff": float((dm / np.maximum(np.abs(ref_m), 1e-12)).max()),
            "passed": bool(rl <= rtol and rm <= rtol)}


def _crossing_fraction(z) -> float:
    return float(np.mean(np.any(np.diff(np.asarray(z), axis=1) < 0, axis=1)))


def _flat(a):
    a = np.asarray(a)
    return a.reshape(-1, a.shape[-1])


def _check_probe_rows(pred: dict, split, name: str) -> None:
    """The Phase-1 probe predictions must be the H3 split's rows, in order: same normalized
    targets, same cluster ids. A mismatch is an error, never a silent re-alignment."""
    if not pred:
        return
    tgt = np.asarray(pred.get("target"), np.float64)
    ref = np.asarray(split.target, np.float64)
    if tgt.shape != ref.shape:
        raise Phase1DependencyError(f"Phase-1 {name} predictions cover {tgt.shape}, the H3 {name} "
                                    f"split {ref.shape}: different rows")
    cid = pred.get("cluster_ids")
    if cid is not None and not np.array_equal(np.asarray(cid), np.asarray(split.cluster_ids)):
        raise Phase1DependencyError(f"Phase-1 {name} predictions: cluster ids differ from H3's")
    if not np.allclose(tgt, ref, rtol=1e-4, atol=1e-5, equal_nan=True):
        raise Phase1DependencyError(f"Phase-1 {name} predictions: normalized targets differ from "
                                    "H3's (rows are not aligned)")


def run_h3_cell(pathway, data, phase1_arrays: dict, *, entrances: dict, families, depths=None,
                device=None, fit_cfg=None, adapter_dir=None, floor_val_loss=None,
                save_prediction_depths=(), phase1_predictions=None, log=print) -> dict:
    """Compute one H3 cell. Returns every array and record the cell writer persists.

    ``phase1_predictions`` = {"val": ..., "test": ...} from ``Phase1Cell.predictions``: the frozen
    probe's normalized forecasts, pushed through this pathway's inverse for the probe's
    VALIDATION MASE (the H4 budget rule selects on it). Without them the probe arm has no
    validation MASE and never qualifies for a budget."""
    cfg = {**DEFAULT_FIT_CFG, **(fit_cfg or {})}
    model, tag = data.model, data.tag
    spec = phase1.model_spec(model)
    L = data.reference_index
    labels = data.depth_labels
    n_dep = L + 1
    all_depths = list(range(n_dep))
    depths = sorted(set(all_depths if depths is None else [int(x) for x in depths]))
    if L not in depths:
        depths.append(L)                             # the reference is always evaluated
    fams, dropped = families_for(model, families)
    device = device or pathway.device
    q = QUANTILES
    qt = torch.as_tensor(q, dtype=torch.float32, device=device)
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    n_va, n_te = len(va.target), len(te.target)
    t_cell = time.time()
    pathway.assert_frozen()
    verification, fits, timings = {}, {fam: {} for fam in fams}, {}

    # ---------------------------------------------------------------- native
    nat = {}
    for s, sd in (("train", tr), ("val", va), ("test", te)):
        out, z = pathway.run(sd.feats[L])
        nat[s] = {"out": out, "z": z}
    nat_val_loss = phase1.mean_pinball_per_window(nat["val"]["z"], va.target, q)
    nat_test_loss = phase1.mean_pinball_per_window(nat["test"]["z"], te.target, q)
    nat_raw = pathway.to_raw(nat["test"]["z"], te)
    nat_m = raw_window_metrics(tag, te.X, te.y_raw, nat_raw, q, MEDIAN_INDEX)
    nat_val_mase = raw_window_metrics(tag, va.X, va.y_raw, pathway.to_raw(nat["val"]["z"], va),
                                      q, MEDIAN_INDEX)["mase_pw"]
    gate = native_gate(nat_test_loss, nat_m["mase_pw"], phase1_arrays, cfg["native_gate_rtol"])
    verification["native_gate"] = gate
    if gate.get("passed") is False:
        raise RuntimeError(f"{model}/{tag}: the offline native pathway does NOT reproduce the "
                           f"Phase-1 native baseline: {gate}. Cache integrity, rows or the pathway "
                           "is wrong -- no H3 number is computed on top of it.")
    if te.native_raw is not None:                    # TimesFM-3 / TiRex: vs the cached forecast
        ref = np.ascontiguousarray(np.asarray(te.native_raw).transpose(0, 2, 1))
        d = np.abs(nat_raw - ref)
        verification["native_raw_vs_phase1_cache"] = {
            "max_abs": float(d.max()),
            "max_rel_to_mean_abs": float(d.max() / max(np.abs(ref).mean(), 1e-12)),
            "note": "offline f(h_L) on the (n, R, d) readout vs the model's own full-sequence "
                    "forward: a different GEMM shape, so not bitwise; measured every run"}
    if te.head_input is not None:                    # Chronos-2 / TiRex: norm(L12) vs cached
        with torch.no_grad():
            o2 = pathway.head(torch.as_tensor(te.head_input, device=pathway.device)).float().cpu()
        verification["final_norm_vs_cached_head_input"] = {
            "max_abs": float(np.abs(o2.numpy() - nat["test"]["out"]).max()),
            "note": "head(norm(h_L)) vs head(cached post-norm state): the pathway applies the "
                    "checkpoint's own final norm to the pre-norm final block"}
    verification["native_quantile_crossing_fraction_test"] = _crossing_fraction(nat["test"]["z"])

    # ---------------------------------------------------------------- per-window arrays
    shape_va, shape_te = (n_dep, n_va), (n_dep, n_te)
    arr = {f: {"val_loss": np.full(shape_va, np.nan), "val_mase": np.full(shape_va, np.nan),
               "test_loss": np.full(shape_te, np.nan),
               "test_mase": np.full(shape_te, np.nan), "test_mae": np.full(shape_te, np.nan),
               "test_wql_num": np.full(shape_te, np.nan)} for f in fams}
    preds = {}
    save_depths = set(int(x) for x in save_prediction_depths) | {L}

    def score(fam, l, adapter):
        _, zv = pathway.run(va.feats[l], adapter)
        _, zt = pathway.run(te.feats[l], adapter)
        raw = pathway.to_raw(zt, te)
        m = raw_window_metrics(tag, te.X, te.y_raw, raw, q, MEDIAN_INDEX)
        a = arr[fam]
        a["val_loss"][l] = phase1.mean_pinball_per_window(zv, va.target, q)
        a["val_mase"][l] = raw_window_metrics(tag, va.X, va.y_raw, pathway.to_raw(zv, va), q,
                                             MEDIAN_INDEX)["mase_pw"]
        a["test_loss"][l] = phase1.mean_pinball_per_window(zt, te.target, q)
        a["test_mase"][l], a["test_mae"][l] = m["mase_pw"], m["mae_pw"]
        a["test_wql_num"][l] = m["wql_num_pw"]
        if l in save_depths:
            preds[f"{fam}__{labels[l]}__z"] = zt.astype(np.float32)
            preds[f"{fam}__{labels[l]}__raw"] = raw.astype(np.float32)
        return zt

    def store_adapter(fam, l, ad, record):
        rec = {**record}
        if adapter_dir is not None:
            meta = {"model": model, "dataset": tag, "family": fam, "depth_index": l,
                    "label": labels[l], "d": pathway.d,
                    **{k: record.get(k) for k in ("selected_wd", "selected_epoch", "selected",
                                                  "selected_kappa", "selected_lambda")}}
            rec["adapter_file"] = save_adapter(
                adapter_dir / model / tag / f"{fam}__{labels[l]}.npz", ad, meta)
        rec["adapter_sha256"] = array_sha256(ad.delta.detach().cpu().numpy(),
                                             ad.bias.detach().cpu().numpy())
        fits[fam][labels[l]] = rec

    def torch_rows(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

    for l in depths:
        t0 = time.time()
        lab = labels[l]
        log(f"    [{model}/{tag}] depth {lab}")
        z_hard = None
        if "hard" in fams:
            z_hard = score("hard", l, None)
        # ---- NOA: label-free native-output alignment -------------------------------------
        if "noa" in fams:
            if l == L:
                score("noa", l, None)
                fits["noa"][lab] = {"fitted": False, "reason": "reference depth: the native "
                                    "pathway itself (the optimum is exactly Delta = 0)"}
            elif pathway.linear_head() is not None:
                W, _c = pathway.linear_head()
                r = anchored_ridge_path(_flat(tr.feats[l]), _flat(tr.feats[L]), _flat(va.feats[l]),
                                        _flat(va.feats[L]), metric=W.T @ W, out_dim=W.shape[0],
                                        kappas=cfg["kappas"])
                ad = affine_to_adapter(r["A"], r["b"]).to(device)
                score("noa", l, ad)
                store_adapter("noa", l, ad, {k: v for k, v in r.items() if k not in ("A", "b")}
                              | {"fitted": True, "solver": "closed form, head-weighted metric "
                                 "W^T W (Sylvester-type, eigenbasis)"})
            else:
                res = fit_residual_adapter(
                    d=pathway.d, pathway_fn=pathway.outputs, train_x=torch_rows(tr.feats[l]),
                    train_y=torch_rows(nat["train"]["out"]), val_x=torch_rows(va.feats[l]),
                    val_y=torch_rows(nat["val"]["out"]), train_loss=mse_loss,
                    val_criterion=mse_loss, wd_grid=cfg["wd_grid"], epochs=cfg["epochs"],
                    lr=cfg["lr"], eval_every=cfg["eval_every"], device=device, seed=cfg["seed"],
                    log=log, label=f"noa {lab}")
                score("noa", l, res["adapter"])
                store_adapter("noa", l, res["adapter"],
                              {k: v for k, v in res.items() if k != "adapter"} | {"fitted": True})
        # ---- FL: supervised alignment through the same frozen pathway --------------------
        if "fl" in fams:
            def fl_path(h):
                return pathway.q9(pathway.outputs(h))

            def fl_loss(p, y):
                return pinball_mean(p, y, qt)
            res = fit_residual_adapter(
                d=pathway.d, pathway_fn=fl_path, train_x=torch_rows(tr.feats[l]),
                train_y=torch_rows(tr.target), val_x=torch_rows(va.feats[l]),
                val_y=torch_rows(va.target), train_loss=fl_loss, val_criterion=fl_loss,
                wd_grid=cfg["wd_grid"], epochs=cfg["epochs"], lr=cfg["lr"],
                eval_every=cfg["eval_every"], device=device, seed=cfg["seed"], log=log,
                label=f"fl {lab}")
            score("fl", l, res["adapter"])
            store_adapter("fl", l, res["adapter"],
                          {k: v for k, v in res.items() if k != "adapter"} | {"fitted": True})
        # ---- RA: hidden-space ridge (diagnostic) -------------------------------------------
        if "ra" in fams:
            if l == L:
                score("ra", l, None)
                fits["ra"][lab] = {"fitted": False, "reason": "reference depth: identity"}
            else:
                r = anchored_ridge_path(_flat(tr.feats[l]), _flat(tr.feats[L]), _flat(va.feats[l]),
                                        _flat(va.feats[L]), metric=None, kappas=cfg["kappas"])
                ad = affine_to_adapter(r["A"], r["b"]).to(device)
                score("ra", l, ad)
                rec = {k: v for k, v in r.items() if k not in ("A", "b")} | {"fitted": True}
                for s, sdat in (("val", va), ("test", te)):
                    Yh = _flat(sdat.feats[l]) @ r["A"].T + r["b"]
                    Y = _flat(sdat.feats[L]).astype(np.float64)
                    ss = ((Y - Y.mean(0)) ** 2).sum()
                    rec[f"{s}_representation_r2"] = float(1 - ((Yh - Y) ** 2).sum() / ss) \
                        if ss > 0 else float("nan")
                store_adapter("ra", l, ad, rec)
        timings[lab] = round(time.time() - t0, 2)

    # ---------------------------------------------------------------- offline identities
    probe_l = depths[0] if depths[0] != L else L
    if "hard" in fams:
        verification["hard_cut_at_final_depth_equals_native"] = bool(
            np.array_equal(arr["hard"]["test_loss"][L], nat_test_loss))
        _, z_id = pathway.run(te.feats[probe_l], ResidualAdapter(pathway.d).to(device).eval())
        _, z_h = pathway.run(te.feats[probe_l], None)
        verification["identity_adapter_equals_hard_cut_bitwise"] = {
            "depth": labels[probe_l], "bitwise": bool(np.array_equal(z_id, z_h))}
        if not verification["identity_adapter_equals_hard_cut_bitwise"]["bitwise"]:
            raise RuntimeError("the residual adapter at Delta=0, b=0 is NOT bitwise the hard cut")
    if pathway.linear_head() is not None:
        W, _c = pathway.linear_head()
        r = anchored_ridge_path(_flat(tr.feats[L]), _flat(tr.feats[L]), _flat(va.feats[L]),
                                _flat(va.feats[L]), metric=W.T @ W, out_dim=W.shape[0],
                                kappas=cfg["kappas"][:3], include_hard_cut=False)
        verification["closed_form_identity_at_final_depth"] = {
            "max_abs_A_minus_I": float(np.abs(r["A"] - np.eye(len(r["A"]))).max()),
            "max_abs_b": float(np.abs(r["b"]).max()),
            "note": "X = Y makes A = I, b = 0 the exact optimum for every lambda; the "
                    "eigenbasis reconstruction reproduces it to float64 round-off"}

    # ---------------------------------------------------------------- probe (Phase 1, no refit)
    dep_rows = spec.depth_indices
    probe = {}
    for key, dest in (("test_loss_window", "test_loss"), ("val_loss_window", "val_loss"),
                      ("test_mase_window", "test_mase"), ("test_mae_window", "test_mae"),
                      ("test_wql_num_window", "test_wql_num")):
        if key in phase1_arrays:
            probe[dest] = np.asarray(phase1_arrays[key], np.float64)[dep_rows]
    if phase1_predictions:
        pv, pt = phase1_predictions.get("val") or {}, phase1_predictions.get("test") or {}
        _check_probe_rows(pv, va, "val")
        _check_probe_rows(pt, te, "test")
        vm = np.full(shape_va, np.nan)
        for l in all_depths:
            z = pv.get(f"pred__{labels[l]}")
            if z is not None:
                vm[l] = raw_window_metrics(tag, va.X, va.y_raw,
                                           pathway.to_raw(np.asarray(z, np.float32), va), q,
                                           MEDIAN_INDEX)["mase_pw"]
        probe["val_mase"] = vm
        # the SAME inverse must reproduce Phase 1's own probe TEST MASE, or the validation MASE
        # above is on a different scale from everything it is compared with
        if pt and "test_mase" in probe:
            worst = 0.0
            for l in all_depths:
                z = pt.get(f"pred__{labels[l]}")
                if z is None:
                    continue
                mt = raw_window_metrics(tag, te.X, te.y_raw,
                                        pathway.to_raw(np.asarray(z, np.float32), te), q,
                                        MEDIAN_INDEX)["mase_pw"]
                ref = float(np.mean(probe["test_mase"][l]))
                worst = max(worst, abs(float(np.mean(mt)) - ref) / max(abs(ref), 1e-12))
            verification["probe_inverse_gate"] = {
                "max_rel_diff_mean_test_mase": worst, "rtol": cfg["native_gate_rtol"],
                "passed": bool(worst <= cfg["native_gate_rtol"])}
            if not verification["probe_inverse_gate"]["passed"]:
                raise RuntimeError(f"{model}/{tag}: the pathway inverse does not reproduce the "
                                   f"Phase-1 probe's test MASE ({worst:.2e} rel.); the probe's "
                                   "validation MASE would be on the wrong scale")
    wql_den = nat_m["wql_den_pw"]
    if "test_wql_den_window" in phase1_arrays:
        verification["wql_denominator_equals_phase1"] = bool(np.allclose(
            phase1_arrays["test_wql_den_window"], wql_den, rtol=1e-12, atol=0))

    # ---------------------------------------------------------------- bootstrap (paired)
    B, bs = int(cfg["boot_b"]), int(cfg["boot_seed"])
    arms = {f: arr[f] for f in fams}
    if probe:
        arms["probe"] = probe
    keys = [("native", None)] + [(f, l) for f in arms for l in all_depths]

    def rows_for(metric):
        R = [{"loss": nat_test_loss, "mase": nat_m["mase_pw"], "mae": nat_m["mae_pw"],
              "wql": nat_m["wql_num_pw"]}[metric]]
        src = {"loss": "test_loss", "mase": "test_mase", "mae": "test_mae", "wql": "test_wql_num"}
        for f, l in keys[1:]:
            a = arms[f].get(src[metric])
            R.append(np.full(n_te, np.nan) if a is None else a[l])
        return np.stack(R)

    cid = te.cluster_ids
    reps, points = {}, {}
    for metric in METRICS:
        Wm = rows_for(metric)
        if metric == "wql":
            num = cluster_replicates(np.nan_to_num(Wm), cid, B, bs, reduce="sum")
            den = cluster_replicates(wql_den[None, :], cid, B, bs, reduce="sum")[:, 0]
            reps[metric] = num / (len(q) * den[:, None])
            points[metric] = Wm.sum(axis=1) / (len(q) * wql_den.sum())
        else:
            reps[metric] = cluster_replicates(np.nan_to_num(Wm), cid, B, bs, reduce="mean")
            points[metric] = Wm.mean(axis=1)
        bad = ~np.isfinite(points[metric])
        reps[metric][:, bad] = np.nan

    def stat(metric, f, l):
        i = keys.index((f, l))
        p = points[metric][i]
        if not np.isfinite(p):
            return None
        return ratio_summary(p, points[metric][0], reps[metric][:, i], reps[metric][:, 0])

    def gap_closure(f, l, metric="mase"):
        """G = (M_hard - M_arm) / (M_hard - M_native); valid only if the hard-cut gap's CI
        excludes 0 (otherwise there is nothing to close)."""
        if "hard" not in arms or f == "hard":
            return None
        ih, ia = keys.index(("hard", l)), keys.index((f, l))
        ph, pa, pn = points[metric][ih], points[metric][ia], points[metric][0]
        if not (np.isfinite(ph) and np.isfinite(pa)):
            return None
        rh, ra_, rn = reps[metric][:, ih], reps[metric][:, ia], reps[metric][:, 0]
        gap = rh - rn
        glo, ghi = np.percentile(gap, [2.5, 97.5])
        valid = bool(glo > 0 or ghi < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (rh - ra_) / gap
        lo, hi = np.nanpercentile(g, [2.5, 97.5]) if valid else (np.nan, np.nan)
        return {"gap_closure": float((ph - pa) / (ph - pn)) if valid else None,
                "ci_lo": float(lo) if valid else None, "ci_hi": float(hi) if valid else None,
                "hard_gap_ci": [float(glo), float(ghi)], "valid": valid}

    # ---------------------------------------------------------------- tunnels
    complete = set(depths) == set(all_depths)
    val_nat = float(nat_val_loss.mean())
    tunnels = {"phase1_recoverability": {f"tol_{t:g}": entrances[t]["depth_axis_index"]
                                         for t in TUNNEL_TOLS},
               "phase1_recoverability_labels": {f"tol_{t:g}": entrances[t]["label"]
                                                for t in TUNNEL_TOLS},
               "native_val_loss": val_nat, "complete_depth_sweep": complete}
    if complete:
        for f in list(fams) + (["probe"] if "val_loss" in probe else []):
            curve = np.nanmean(arms[f]["val_loss"], axis=1)
            ct = compatibility_tunnel(curve, val_nat)
            ct["compatibility_lag_in_depth"] = {
                k: int(v - entrances[float(k.split("_")[1])]["depth_axis_index"])
                for k, v in ct["by_tolerance"].items()}
            tunnels[f] = ct
    else:
        tunnels["note"] = "depth subset (smoke): compatibility tunnels need every depth"

    # ---------------------------------------------------------------- the H4 frontier
    # How much accuracy survives each cut, and, per accuracy budget, the earliest cut chosen on
    # VALIDATION MASE only (BUDGET_RULE). Test is read once, at the frozen depth.
    nv = float(np.mean(nat_val_mase))
    frontier = {"rule": BUDGET_RULE, "complete_depth_sweep": complete,
                "native_val_mase": nv, "native_test_mase": float(points["mase"][0]),
                "depth_labels": list(labels), "reference_index": L, "arms": {}}
    if complete:
        for f in arms:
            if "val_mase" not in arms[f]:
                continue
            with np.errstate(invalid="ignore"):
                vcurve = np.array([np.mean(arms[f]["val_mase"][l]) for l in all_depths]) / nv
            tstats = [stat("mase", f, l) for l in all_depths]
            rec = {"val_mase_ratio_by_depth": [None if not np.isfinite(v) else float(v)
                                                for v in vcurve],
                   "test_mase_ratio_by_depth": [None if t is None else t["ratio"]
                                                 for t in tstats],
                   "budgets": {}}
            for eps in FRONTIER_BUDGETS:
                dsel = budget_depth(vcurve, eps)
                if dsel == L:
                    op = {"depth_index": L, "label": labels[L], "truncated": False,
                          "blocks_removed": 0, "operating_model": "native",
                          "val_mase_ratio": 1.0, "test_mase_ratio": 1.0,
                          "test_mase_ci": [1.0, 1.0], "test_wql_ratio": 1.0,
                          "within_budget_test": True}
                else:
                    sm, sw = stat("mase", f, dsel), stat("wql", f, dsel)
                    op = {"depth_index": dsel, "label": labels[dsel], "truncated": True,
                          "blocks_removed": L - dsel, "operating_model": f,
                          "val_mase_ratio": float(vcurve[dsel]),
                          "test_mase_ratio": sm["ratio"],
                          "test_mase_ci": [sm["ratio_ci_lo"], sm["ratio_ci_hi"]],
                          "test_wql_ratio": None if sw is None else sw["ratio"],
                          "within_budget_test": bool(sm["ratio"] - 1.0 <= eps)}
                rec["budgets"][f"eps_{eps:g}"] = op
            frontier["arms"][f] = rec
    else:
        frontier["note"] = "depth subset (smoke): the frontier and the budget rule need every depth"

    # ---------------------------------------------------------------- the ladder at l_rec
    ladder = {"operating_point": "frozen Phase-1 sustained 5% tunnel entrance (validation)",
              "by_tolerance": {}}
    for t in TUNNEL_TOLS:
        l = entrances[t]["depth_axis_index"]
        rungs = {}
        for f in arms:
            if not np.isfinite(points["mase"][keys.index((f, l))]):
                continue
            rungs[f] = {m: stat(m, f, l) for m in METRICS}
            rungs[f]["gap_closure_mase"] = gap_closure(f, l)
            rungs[f]["gap_closure_wql"] = gap_closure(f, l, "wql")
        ladder["by_tolerance"][f"tol_{t:g}"] = {
            "depth_index": l, "label": labels[l], "relative_depth": l / L,
            "blocks_removed": L - l, "rungs": rungs,
            "compatibility_gap_exists": (None if "hard" not in rungs else
                                         bool(rungs["hard"]["mase"]["ratio_ci_lo"] > 1.0))}
    ladder["headline"] = ladder["by_tolerance"][f"tol_{HEADLINE_TOL:g}"]
    ladder["native"] = {"test_loss": float(points["loss"][0]), "test_mase": float(points["mase"][0]),
                        "test_mae": float(points["mae"][0]), "test_wql": float(points["wql"][0]),
                        "val_loss": val_nat,
                        "wql_standard": standard_wql(nat_m["wql_num_pw"], wql_den)}

    # ---------------------------------------------------------------- depth table
    rows = []
    for f in arms:
        for l in all_depths:
            i = keys.index((f, l))
            if not np.isfinite(points["mase"][i]):
                continue
            s_m, s_w, s_l = stat("mase", f, l), stat("wql", f, l), stat("loss", f, l)
            g = gap_closure(f, l)
            fr = (fits.get(f) or {}).get(labels[l], {})
            rows.append({
                "model": model, "dataset": tag, "family": f, "depth_index": l,
                "label": labels[l], "relative_depth": l / L,
                "is_phase1_entrance": l == entrances[HEADLINE_TOL]["depth_axis_index"],
                "val_loss": float(np.nanmean(arms[f]["val_loss"][l])) if "val_loss" in arms[f]
                else "",
                "val_mase": float(np.mean(arms[f]["val_mase"][l]))
                if "val_mase" in arms[f] and np.all(np.isfinite(arms[f]["val_mase"][l])) else "",
                "val_mase_ratio_vs_native": float(np.mean(arms[f]["val_mase"][l])) / nv
                if "val_mase" in arms[f] and np.all(np.isfinite(arms[f]["val_mase"][l])) else "",
                "test_loss": float(points["loss"][i]), "test_mase": float(points["mase"][i]),
                "test_mae": float(points["mae"][i]), "test_wql": float(points["wql"][i]),
                "mase_ratio_vs_native": s_m["ratio"], "mase_ratio_ci_lo": s_m["ratio_ci_lo"],
                "mase_ratio_ci_hi": s_m["ratio_ci_hi"],
                "wql_ratio_vs_native": s_w["ratio"], "wql_ratio_ci_lo": s_w["ratio_ci_lo"],
                "wql_ratio_ci_hi": s_w["ratio_ci_hi"],
                "loss_ratio_vs_native": s_l["ratio"],
                "within_budget_mase": s_m["within_budget"],
                "budget_fragile_mase": s_m["budget_fragile"],
                "gap_closure_mase": "" if not g or g["gap_closure"] is None else g["gap_closure"],
                "gap_closure_valid": "" if not g else g["valid"],
                "selected_hard_cut": fr.get("selected_hard_cut", ""),
                "selected_wd": "" if fr.get("selected_wd") is None else fr.get("selected_wd"),
                "selected_epoch": fr.get("selected_epoch", ""),
                "selected_kappa": "" if fr.get("selected_kappa") is None
                else fr.get("selected_kappa"),
                "n_adapter_params": fr.get("n_params", pathway.d * pathway.d + pathway.d
                                            if fr.get("fitted") else 0)})

    low_skill = None
    if floor_val_loss:
        low_skill = {"native_val_over_floor": val_nat / float(floor_val_loss),
                     "flag": bool(val_nat >= LOW_SKILL_RATIO * float(floor_val_loss)),
                     "threshold": LOW_SKILL_RATIO}

    bootstrap_inputs = {"depth_labels": np.asarray(labels, dtype=object),
                        "families": np.asarray(list(arms), dtype=object),
                        "cluster_ids_test": np.asarray(te.cluster_ids, np.int64),
                        "cluster_ids_val": np.asarray(va.cluster_ids, np.int64),
                        "rows_test": te.rows, "rows_val": va.rows,
                        "native_val_loss": nat_val_loss, "native_val_mase": nat_val_mase,
                        "native_test_loss": nat_test_loss,
                        "native_test_mase": nat_m["mase_pw"], "native_test_mae": nat_m["mae_pw"],
                        "native_test_wql_num": nat_m["wql_num_pw"], "test_wql_den": wql_den,
                        "mase_denominator_test": nat_m["denominator"]}
    for f, a in arms.items():
        for k, v in a.items():
            bootstrap_inputs[f"{f}__{k}"] = np.asarray(v, np.float64)
    preds["native__z"] = nat["test"]["z"].astype(np.float32)
    preds["native__raw"] = nat_raw.astype(np.float32)
    preds["y_raw"] = np.asarray(te.y_raw, np.float64)
    preds["target"] = np.asarray(te.target, np.float32)
    preds["cluster_ids"] = np.asarray(te.cluster_ids, np.int64)
    preds["rows"] = te.rows
    timings["cell_total_s"] = round(time.time() - t_cell, 1)
    return {"model": model, "dataset": tag, "families": list(arms), "fitted_families": fams,
            "dropped_families": dropped, "depths_evaluated": depths, "depth_labels": labels,
            "reference_index": L, "verification": verification, "fits": fits,
            "tunnels": tunnels, "ladder": ladder, "frontier": frontier, "rows": rows,
            "low_skill": low_skill,
            "bootstrap_inputs": bootstrap_inputs, "predictions": preds, "timings": timings,
            "fit_cfg": cfg, "pathway": pathway.describe(), "budget": BUDGET}


# --------------------------------------------------------------------------- #
# persistence -- one staging directory, every artifact regenerable without a model
# --------------------------------------------------------------------------- #
def save_h3_cell(stage, res: dict, cell_cfg: dict, config_hash: str, dependency: dict,
                 provenance: dict, extra: dict | None = None) -> dict:
    """Write every required H3 artifact into ``stage`` (a Phase2CellStore staging dir)."""
    from pathlib import Path
    from probing.phase2 import atomic_write_json, write_csv
    stage = Path(stage)
    atomic_write_json(stage / "cell_config.json", {**cell_cfg, "config_hash": config_hash})
    write_csv(stage / "depth_metrics.csv", res["rows"])
    atomic_write_json(stage / "tunnels.json", res["tunnels"])
    atomic_write_json(stage / "ladder_at_tunnel.json", res["ladder"])
    atomic_write_json(stage / "verification.json", res["verification"])
    atomic_write_json(stage / "fits.json", res["fits"])
    atomic_write_json(stage / "frontier.json", res["frontier"])
    np.savez_compressed(stage / "bootstrap_inputs.npz", **res["bootstrap_inputs"])
    np.savez_compressed(stage / "predictions_selected.npz", **res["predictions"])
    head = res["ladder"]["headline"]
    rung_view = {f: {"mase_ratio": v["mase"]["ratio"], "mase_ci": [v["mase"]["ratio_ci_lo"],
                                                                   v["mase"]["ratio_ci_hi"]],
                     "wql_ratio": v["wql"]["ratio"], "within_budget_mase":
                         v["mase"]["within_budget"]}
                 for f, v in head["rungs"].items()}
    summary = {
        "schema": "phase2_h3_cell/v1", "model": res["model"], "dataset": res["dataset"],
        "question": ("Once forecasting is recoverable from an intermediate representation, can a "
                     "lightweight transformation make it usable by the model's ORIGINAL frozen "
                     "forecasting pathway, close to the full native model?"),
        "families": res["families"], "dropped_families": res["dropped_families"],
        "depths_evaluated": [res["depth_labels"][i] for i in res["depths_evaluated"]],
        "complete_depth_sweep": res["tunnels"]["complete_depth_sweep"],
        "phase1_tunnel_entrance": head["label"], "blocks_removed_at_entrance":
            head["blocks_removed"],
        "ladder_at_entrance": rung_view,
        "compatibility_gap_exists_at_entrance": head["compatibility_gap_exists"],
        "native": res["ladder"]["native"], "low_skill": res["low_skill"],
        "frontier_budgets": {f: {k: {"label": v["label"], "truncated": v["truncated"],
                                     "test_mase_ratio": v["test_mase_ratio"]}
                                 for k, v in rec["budgets"].items()}
                             for f, rec in res["frontier"]["arms"].items()},
        "verification": {k: (v.get("passed") if isinstance(v, dict) and "passed" in v else v)
                         for k, v in res["verification"].items()},
        "timings": res["timings"], "fit_cfg": res["fit_cfg"], "pathway": res["pathway"],
        "phase1_dependency": dependency, "provenance": provenance,
        "caveats": {
            "recoverability_vs_replaceability": (
                "A strong intermediate-layer probe establishes RECOVERABILITY but not "
                "REPLACEABILITY: the probe replaces the pretrained pathway. The hard cut and "
                "the aligned native pathway test replaceability; only they license H4."),
            "cross_model_loss": phase1.CROSS_MODEL_LOSS_CAVEAT,
            "wql": "WQL is reported with the standard 1/Q normalization (phase2.standard_wql)",
            "operating_point": "the frozen Phase-1 sustained 5% entrance; compatibility "
                               "tunnels are a secondary diagnostic"},
        **(extra or {})}
    atomic_write_json(stage / "summary.json", summary)
    return summary
