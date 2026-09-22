"""The Phase-1 cell artifact contract: what one ``model x dataset`` cell writes, and how.

Split out from the driver ON PURPOSE. Everything here is a pure function of numpy arrays and
plain dicts — no model, no GPU, no dataset — so the artifact contract, the resume logic and the
combined-table reconstruction are all testable on synthetic input, which is exactly what
contracts 42-47 of the pre-run suite do.

THE DURABILITY RULE. ``results/three_model_final/`` must contain enough to regenerate every
Phase-1 figure WITHOUT re-running a foundation model. That means, per cell:

    layer_metrics.csv        per-depth train/val/test loss + MASE/MAE/WQL + CIs + hyperparams
                             + val/test ratio to the final block and the suffix excursion
    tunnel.json              the validation-defined SUSTAINED entrance at every tolerance, with
                             the first-crossing statistic beside it as a named diagnostic
    wd_selection.csv         per depth: selected wd, the grid, min/max flags and EVERY
                             candidate's validation loss -> "is the grid still clipped?"
    bootstrap_inputs.npz     the PER-WINDOW metric arrays and the cluster ids -> any CI can be
                             recomputed, at any B, without the model
    predictions_{val,test}.npz   the full (n, Q, H) forecasts, the targets, the raw contexts and
                             the cluster ids -> calibration, per-quantile and per-window
                             distribution plots, and val-side raw metrics, are all derivable
    cka/*.npy + cka_metadata.json    every matrix, with labels/split/estimator/N/d/null floor
    spectral_metrics.npz     the full normalized spectra, not just the scalar rank
    probe_artifacts/         the frozen probe weights, bias and scaler per depth
    summary.json             everything else, including the caveats a reader must carry

Heavy RAW hidden representations deliberately do NOT live here — they are several GB per model
and belong in $SCRATCH. What lives here instead is the cache manifest (path, key, metadata) so
the provenance of every number is still recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probing import phase1, registry

__all__ = ["save_cell", "layer_metric_rows", "effective_rank_rows", "cka_index_rows",
           "read_cell_summary", "cell_dirs"]


def cell_dirs(root: Path, model: str, dataset: str) -> Path:
    return Path(root) / model / dataset


# --------------------------------------------------------------------------- #
# row builders (also used by the combined-table generator)
# --------------------------------------------------------------------------- #
def layer_metric_rows(model: str, tag: str, spec, res: dict, boot: dict, tun: dict,
                      raw: dict | None) -> list[dict]:
    """One row per PROBED point — head-input diagnostics included, and marked as such.

    ``point_type`` / ``include_in_main_depth_axis`` are what a plotting script filters on. A
    head-input diagnostic (Chronos-2 L12+LN, TiRex L12+RMS) has an EMPTY ``relative_depth`` and
    ``relative_position`` as well, so it cannot silently land on a numeric depth axis.
    ``is_reference_point`` marks the final BLOCK, which is what "final-depth probe loss" means.
    """
    ent = tun["headline"]["index"]
    # Per-tolerance entrance indices, sustained AND first-crossing, so a figure can mark either
    # without re-deriving a criterion. The keys are exactly the ones tunnel.json carries.
    ent_by_tol = {e["tolerance"]: e["index"] for e in tun["by_tolerance"].values()}
    fc_by_tol = {e["tolerance"]: e["index"] for e in tun.get("first_crossing", {}).values()}
    # R_l / R_L per DEPTH (spec section 9). Indexed by depth-axis position, so a head-input
    # diagnostic simply has no ratio -- it has no final-block reference to be a ratio against.
    dep = tun["depth_axis_indices"]
    val_ratio = dict(zip(dep, tun.get("val_ratio_by_depth", [])))
    test_ratio = dict(zip(dep, tun.get("test_ratio_by_depth", [])))
    val_exc = dict(zip(dep, tun.get("val_suffix_excursion_by_depth", [])))
    test_exc = dict(zip(dep, tun.get("test_suffix_excursion_by_depth", [])))
    rows = []
    for i, p in enumerate(spec.points):
        r = {"model": model, "dataset": tag, "display_name": registry.display_name(tag),
             "point_index": i, "layer": p.label, "kind": p.kind,
             "point_type": p.point_type,
             "include_in_main_depth_axis": p.on_depth_axis,
             "block_index": p.block_index,
             "relative_depth": "" if p.relative_depth is None else p.relative_depth,
             "relative_position": "" if p.relative_position is None else p.relative_position,
             "is_tunnel_entrance": bool(i == ent),
             "is_reference_point": bool(i == spec.reference_index),
             "is_native_readout": bool(i == spec.native_readout_index),
             "train_loss": res["train_loss"][i],
             "val_loss": res["val_loss"][i],
             "test_loss": res["test_loss"][i],
             # R_l / R_L on each split, and the suffix excursion the criterion actually reads.
             # VALIDATION selects; the TEST twin is descriptive and never enters a selection.
             "val_ratio_to_final": val_ratio.get(i, ""),
             "test_ratio_to_final": test_ratio.get(i, ""),
             "val_suffix_excursion": val_exc.get(i, ""),
             "test_suffix_excursion": test_exc.get(i, ""),
             "weight_decay": res["wd"][i],
             "wd_at_grid_max": res["wd_at_grid_max"][i],
             "wd_at_grid_min": res["wd_at_grid_min"][i],
             "n_probe_params": res["n_params"][i]}
        for t, idx in ent_by_tol.items():
            r[f"is_tunnel_entrance_{t:g}"] = bool(i == idx)
        for t, idx in fc_by_tol.items():
            r[f"is_first_crossing_{t:g}"] = bool(i == idx)
        for name, key in (("test_loss", "loss"), ("test_mase", "mase"),
                          ("test_mae", "mae"), ("test_wql", "wql")):
            b = boot.get(key)
            if b is None:
                continue
            r[f"{name}_point"] = b["point"][i]
            r[f"{name}_ci_lo"] = b["ci_lo"][i]
            r[f"{name}_ci_hi"] = b["ci_hi"][i]
            r[f"{name}_delta_vs_last"] = b["delta_vs_last"][i]
            r[f"{name}_delta_ci_lo"] = b["delta_ci_lo"][i]
            r[f"{name}_delta_ci_hi"] = b["delta_ci_hi"][i]
        if raw is not None:
            r["test_mase"] = float(np.asarray(raw["mase_pw"])[i].mean())
            r["test_mae"] = float(np.asarray(raw["mae_pw"])[i].mean())
            r["test_wql"] = float(np.asarray(raw["wql_num_pw"])[i].sum()
                                  / np.asarray(raw["wql_den_pw"]).sum())
        rows.append(r)
    return rows


def effective_rank_rows(model: str, tag: str, blocks) -> list[dict]:
    rows = []
    for b in blocks:
        er = b["effective_rank"]
        types = er.get("point_types", [phase1.BLOCK_DEPTH] * len(er["layer_names"]))
        for j, lab in enumerate(er["layer_names"]):
            on_axis = types[j] == phase1.BLOCK_DEPTH
            rows.append({
                "model": model, "dataset": tag, "display_name": registry.display_name(tag),
                "geometry_variant": b["variant"], "split": b["split"], "layer": lab,
                "point_type": types[j], "include_in_main_depth_axis": on_axis,
                "point_index": j, "n_rows": b["n_rows"], "feature_dim": b["feature_dim"],
                "effective_rank": er["effective_rank"][j],
                "normalized_effective_rank": er["normalized_effective_rank"][j],
                "effective_rank_over_max_possible": er["effective_rank_over_max_possible"][j],
                "spectral_entropy": er["spectral_entropy"][j],
                "pc1_fraction": er["pc1_fraction"][j],
                "numerical_rank": er["numerical_rank"][j],
                "max_possible_rank": er["max_possible_rank"],
                # a head-input diagnostic is never part of a headline depth curve
                "is_headline": bool(b.get("is_headline", b["variant"] == "headline")
                                    and on_axis)})
    return rows


def cka_index_rows(model: str, tag: str, blocks) -> list[dict]:
    """The machine-readable index telling plotting code where every CKA matrix lives."""
    rows = []
    for b in blocks:
        for est in b["estimators"]:
            fl = b["cka_null_floor"][est]
            base = {
                "model": model, "dataset": tag, "display_name": registry.display_name(tag),
                "geometry_variant": b["variant"], "split": b["split"], "estimator": est,
                "n_rows": b["n_rows"], "feature_dim": b["feature_dim"],
                "null_floor_mean": fl["mean"], "null_floor_std": fl["std"],
                "is_headline_estimator": bool(est == "unbiased"),
                "is_headline_variant": bool(b.get("is_headline", b["variant"] == "headline"))}
            # the MAIN matrix: depth axis only, safe to draw as a depth heatmap
            rows.append({**base, "axis": phase1.BLOCK_DEPTH,
                         "path": f"{model}/{tag}/cka/cka_{est}_{b['variant']}_{b['split']}.npy",
                         "n_layers": len(b["labels"]), "labels": "|".join(b["labels"]),
                         "mean_offdiagonal_cka": fl["mean_offdiagonal_cka"],
                         "mean_offdiagonal_above_null_floor":
                             fl["mean_offdiagonal_above_null_floor"],
                         "is_main_depth_axis": True})
            if "cka_with_head_input" in b:
                rows.append({
                    **base, "axis": "with_head_input",
                    "path": (f"{model}/{tag}/cka/cka_{est}_{b['variant']}_{b['split']}"
                             f"__with_head_input.npy"),
                    "n_layers": len(b["cka_with_head_input_labels"]),
                    "labels": "|".join(b["cka_with_head_input_labels"]),
                    "mean_offdiagonal_cka": "", "mean_offdiagonal_above_null_floor": "",
                    "is_main_depth_axis": False})
    return rows


# --------------------------------------------------------------------------- #
# the writer
# --------------------------------------------------------------------------- #
def save_cell(stage: Path, *, model: str, tag: str, spec, cfg: dict, config_hash: str,
              res: dict, raw: dict | None, native: dict | None, boot: dict, tun: dict,
              geom_blocks, ident: dict, extraction: dict, window_meta: dict,
              cluster_ids: dict, contexts: dict, targets: dict,
              save_predictions: str = "val+test", save_probe_artifacts: bool = True,
              timings: dict | None = None, warnings: list | None = None) -> dict:
    """Write every required artifact of one cell into ``stage``. Returns the summary dict.

    ``stage`` is the staging directory from ``CellStore.begin()``; nothing here touches the
    final location, so a failure leaves no partial cell behind.
    """
    stage = Path(stage)
    (stage / "cka").mkdir(parents=True, exist_ok=True)
    (stage / "probe_artifacts").mkdir(parents=True, exist_ok=True)
    prov = registry.provenance(tag, model)
    dspec = registry.spec(tag)

    # ---- config ---------------------------------------------------------- #
    phase1.atomic_write_json(stage / "cell_config.json",
                             {**cfg, "config_hash": config_hash})

    # ---- tunnel ---------------------------------------------------------- #
    phase1.atomic_write_json(stage / "tunnel.json", {
        "model": model, "dataset": tag, "display_name": dspec.display_name, **tun,
        "representation_points": [p.as_dict() for p in spec.points]})

    # ---- per-layer table -------------------------------------------------- #
    rows = layer_metric_rows(model, tag, spec, res, boot, tun, raw)
    phase1.write_csv(stage / "layer_metrics.csv", rows)

    # ---- hyperparameters -------------------------------------------------- #
    phase1.atomic_write_json(stage / "probe_hparams.json", {
        "model": model, "dataset": tag, "probe": res.get("probe_description", spec.probe),
        "quantiles": res["quantiles"], "median_index": res["median_index"],
        "epochs": cfg.get("probe_epochs"), "lr": cfg.get("probe_lr"),
        "wd_grid": cfg.get("wd_grid"), "seed": cfg.get("seed"),
        "by_layer": [{"layer": spec.labels[i], "weight_decay": res["wd"][i],
                      "at_grid_max": res["wd_at_grid_max"][i],
                      "at_grid_min": res["wd_at_grid_min"][i],
                      "n_params": res["n_params"][i],
                      "val_loss_by_wd": res["selection"][i]}
                     for i in range(spec.n_points)],
        "grid_clipping": _grid_clipping(spec, res, cfg),
        "null_baseline": res.get("null_baseline"),
        "constant_forecast_floor": res.get("constant_forecast_floor")})

    # ---- weight-decay selection table ------------------------------------ #
    # Its own artifact, and a REQUIRED one: "is the expanded grid still clipped?" is the first
    # question this rerun exists to answer, and it must be answerable from a CSV rather than by
    # parsing nested JSON. One row per probed point, every candidate's validation loss on it.
    wd_rows = phase1.wd_selection_rows(model, tag, spec, res, cfg.get("wd_grid") or [],
                                       lr=cfg.get("probe_lr"),
                                       floor=res.get("constant_forecast_floor"))
    phase1.write_csv(stage / "wd_selection.csv", wd_rows)

    # ---- geometry --------------------------------------------------------- #
    er_rows = effective_rank_rows(model, tag, geom_blocks)
    phase1.write_csv(stage / "effective_rank.csv", er_rows)

    spectra, cka_meta = {}, []
    for b in geom_blocks:
        for est, M in b["cka"].items():
            # DEPTH AXIS ONLY -- this is the file a main heatmap should load, and it simply has
            # no row or column for a normalization to be drawn as a depth.
            np.save(stage / "cka" / f"cka_{est}_{b['variant']}_{b['split']}.npy",
                    np.asarray(M, np.float64))
        for est, M in b.get("cka_with_head_input", {}).items():
            np.save(stage / "cka" /
                    f"cka_{est}_{b['variant']}_{b['split']}__with_head_input.npy",
                    np.asarray(M, np.float64))
        for lab, sp in b["effective_rank"]["spectrum"].items():
            spectra[f"{b['variant']}__{b['split']}__{lab}"] = np.asarray(sp, np.float64)
        cka_meta.append({k: v for k, v in b.items()
                         if k not in ("cka", "cka_with_head_input", "effective_rank")}
                        | {"effective_rank_summary": {
                            "layer_names": b["effective_rank"]["layer_names"],
                            "point_types": b["effective_rank"].get("point_types"),
                            "effective_rank": b["effective_rank"]["effective_rank"]}})
    np.savez_compressed(stage / "spectral_metrics.npz", **spectra)
    phase1.atomic_write_json(stage / "cka" / "cka_metadata.json", {
        "model": model, "dataset": tag, "blocks": cka_meta,
        "index": cka_index_rows(model, tag, geom_blocks),
        "headline_estimator": "unbiased",
        "main_axis": ("cka_<est>_<variant>_<split>.npy covers the BLOCK-DEPTH axis only; the "
                      "__with_head_input sibling adds the final-normalization point and is for "
                      "the native-head/alignment work, not for a depth figure"),
        "geometry_caveat": phase1.GEOMETRY_CAVEAT,
        "head_input_caveat": phase1.HEAD_INPUT_CAVEAT,
        "spectra_file": "spectral_metrics.npz",
        "spectra_keys": "<variant>__<split>__<layer label>"})

    # ---- bootstrap inputs -------------------------------------------------- #
    bi = {"labels": np.asarray(spec.labels, dtype=object),
          "cluster_ids_test": np.asarray(cluster_ids["test"], np.int64),
          "cluster_ids_val": np.asarray(cluster_ids["val"], np.int64),
          "test_loss_window": np.asarray(res["test_window_loss"], np.float64),
          "val_loss_window": np.asarray(res["val_window_loss"], np.float64)}
    if raw is not None:
        bi.update(test_mase_window=np.asarray(raw["mase_pw"], np.float64),
                  test_mae_window=np.asarray(raw["mae_pw"], np.float64),
                  test_wql_num_window=np.asarray(raw["wql_num_pw"], np.float64),
                  test_wql_den_window=np.asarray(raw["wql_den_pw"], np.float64),
                  mase_denominator_test=np.asarray(raw.get("denominator", []), np.float64))
    if native and native.get("available"):
        for k, dest in (("loss_window", "native_loss_window"),
                        ("mase_window", "native_mase_window"),
                        ("mae_window", "native_mae_window"),
                        ("wql_num_window", "native_wql_num_window"),
                        ("wql_den_window", "native_wql_den_window")):
            if native.get(k) is not None:
                bi[dest] = np.asarray(native[k], np.float64)
    np.savez_compressed(stage / "bootstrap_inputs.npz", **bi)

    # ---- predictions ------------------------------------------------------- #
    saved_preds = []
    want = {"none": (), "test": ("test",), "val+test": ("val", "test")}[save_predictions]
    for split in want:
        key = f"pred_{split}"
        if key not in res:
            continue
        arrs = {f"pred__{lab}": np.asarray(res[key][lab], np.float32) for lab in spec.labels}
        arrs["target"] = np.asarray(targets[split], np.float32)
        arrs["context_raw"] = np.asarray(contexts[split], np.float32)
        arrs["cluster_ids"] = np.asarray(cluster_ids[split], np.int64)
        arrs["labels"] = np.asarray(spec.labels, dtype=object)
        arrs["quantiles"] = np.asarray(res["quantiles"], np.float64)
        np.savez_compressed(stage / f"predictions_{split}.npz", **arrs)
        saved_preds.append(split)

    # ---- probe artifacts ---------------------------------------------------- #
    # The frozen probe (weight, bias, scaler) per depth. NOT required to regenerate any Phase-1
    # figure -- the saved predictions already cover every curve, calibration and per-window
    # plot -- but required to APPLY a probe to new windows without refitting it. It is also by
    # far the largest durable artifact (TimesFM-3: 21 x Linear(1280, 576) ~ 55 MB per cell),
    # which is why it is switchable.
    n_probe_files = 0
    if save_probe_artifacts:
        for lab in spec.labels:
            pw = res.get("probe_weights", {}).get(lab)
            if pw is None:
                continue
            np.savez_compressed(stage / "probe_artifacts" / f"probe__{_slug(lab)}.npz", **pw)
            n_probe_files += 1

    # ---- summary ------------------------------------------------------------ #
    summary = {
        "schema": "phase1_cell/v1",
        "protocol": phase1.PHASE1_PROTOCOL_VERSION,
        "model": model, "model_display_name": spec.display_name,
        "dataset": tag, "display_name": dspec.display_name,
        "config_hash": config_hash,
        "dataset_facts": dspec.as_dict(),
        "pretraining_provenance": {"model": model, **prov.as_dict()},
        "provenance_note": ("`not_listed` means absent from the documentation we have. It is "
                            "NOT OOD, NOT unseen and NOT held out; only `explicitly_held_out` "
                            "carries an author's own exclusion statement."),
        "model_spec": spec.as_dict(),
        "window_identity": ident, "window_meta": _jsonable_meta(window_meta),
        "extraction": extraction,
        "quantile_set": cfg.get("quantile_set"), "quantiles": res["quantiles"],
        "num_quantiles": len(res["quantiles"]), "median_index": res["median_index"],
        "C": cfg.get("C"), "H": cfg.get("H"),
        "objective": ("mean pinball loss over the Q*H terms of a window, then averaged over "
                      "windows (probing.phase1.mean_pinball_per_window)"),
        "point_types": spec.point_types,
        "depth_axis_labels": spec.depth_labels,
        "depth_axis_indices": spec.depth_indices,
        "head_input_diagnostic_labels": spec.diagnostic_labels,
        "head_input_caveat": phase1.HEAD_INPUT_CAVEAT,
        "final_depth_point": spec.reference_label,
        "final_depth_index": spec.reference_index,
        "final_depth_loss": {"train": res["train_loss"][spec.reference_index],
                             "val": res["val_loss"][spec.reference_index],
                             "test": res["test_loss"][spec.reference_index]},
        "loss_by_layer": {"train": res["train_loss"], "val": res["val_loss"],
                          "test": res["test_loss"]},
        "per_quantile_test_loss": res.get("per_quantile_test"),
        "tunnel": tun["headline"],
        "tunnel_definition": tun["definition"],
        "tunnel_definition_caveat": tun.get("definition_caveat"),
        "tunnel_all_tolerances": {k: {"tolerance": v["tolerance"], "label": v["label"],
                                      "index": v["index"],
                                      "depth_axis_index": v["depth_axis_index"],
                                      "relative_depth": v["relative_depth"]}
                                  for k, v in tun["by_tolerance"].items()},
        # The OLD first-crossing statistic, under a name that says what it is. It is NOT the
        # tunnel and no table may label it one.
        "first_crossing_diagnostic": {k: {"tolerance": v["tolerance"], "label": v["label"],
                                          "index": v["index"],
                                          "depth_axis_index": v["depth_axis_index"],
                                          "relative_depth": v["relative_depth"]}
                                      for k, v in tun.get("first_crossing", {}).items()},
        "generalization_at_entrance": tun.get("generalization_at_entrance"),
        "val_ratio_by_depth": tun.get("val_ratio_by_depth"),
        "test_ratio_by_depth": tun.get("test_ratio_by_depth"),
        "wd_grid": cfg.get("wd_grid"),
        "wd_selection": _grid_clipping(spec, res, cfg),
        "constant_forecast_floor": res.get("constant_forecast_floor"),
        "bootstrap": {k: {kk: vv for kk, vv in v.items() if kk != "boot"}
                      for k, v in boot.items()},
        "native_baseline": _native_summary(native),
        "mase_definition": registry.MASE_DEFINITION,
        "seasonal_m": dspec.seasonal_m,
        "n_denominator_clamped": (raw or {}).get("n_denominator_clamped"),
        "geometry": [{k: v for k, v in b.items() if k not in ("cka", "effective_rank")}
                     | {"effective_rank": b["effective_rank"]["effective_rank"],
                        "effective_rank_labels": b["effective_rank"]["layer_names"]}
                     for b in geom_blocks],
        "predictions_saved": saved_preds,
        "probe_artifacts_saved": n_probe_files,
        "regeneration_note": ("every Phase-1 figure is regenerable from this directory without "
                              "re-running a foundation model: layerwise curves from "
                              "layer_metrics.csv, the validation tunnel decision from "
                              "tunnel.json, uncertainty at any B from bootstrap_inputs.npz, "
                              "CKA heatmaps from cka/, effective rank and full spectra from "
                              "effective_rank.csv + spectral_metrics.npz, and calibration / "
                              "per-window distributions from predictions_*.npz"),
        "cross_model_loss_caveat": phase1.CROSS_MODEL_LOSS_CAVEAT,
        "geometry_caveat": phase1.GEOMETRY_CAVEAT,
        "timings": timings or {},
        "warnings": list(warnings or []),
    }
    phase1.atomic_write_json(stage / "summary.json", summary)
    return summary


def _grid_clipping(spec, res: dict, cfg: dict) -> dict:
    """Is the selected weight decay AT a grid edge, and does that edge still mean anything?

    Under the EXPANDED grid the maximum (90 at lr=1e-2) sits within ~1e-3 relative of the
    bias-only floor, so "at the grid maximum" no longer automatically means "the search was cut
    off". The two readings are told apart by the closed-form ``constant_forecast_floor``, which
    is carried here so the distinction is a measurement, not a hope.
    """
    n = spec.n_points
    floor = (res.get("constant_forecast_floor") or {}).get("val_loss")
    at_max = [spec.labels[i] for i in range(n) if res["wd_at_grid_max"][i]]
    out = {
        "n_layers_at_grid_max": int(sum(res["wd_at_grid_max"])),
        "n_layers_at_grid_min": int(sum(res["wd_at_grid_min"])),
        "fraction_at_grid_max": float(sum(res["wd_at_grid_max"])) / n,
        "fraction_at_grid_min": float(sum(res["wd_at_grid_min"])) / n,
        "layers_at_grid_max": at_max,
        "layers_at_grid_min": [spec.labels[i] for i in range(n) if res["wd_at_grid_min"][i]],
        "median_selected_wd": float(np.median(np.asarray(res["wd"], np.float64))),
        "selected_wd_by_layer": {spec.labels[i]: float(res["wd"][i]) for i in range(n)},
        "grid_max": float(max(cfg.get("wd_grid") or [float("nan")])),
        "grid_min": float(min(cfg.get("wd_grid") or [float("nan")])),
        "lr": cfg.get("probe_lr"),
        "meaning": ("a selected weight decay at the grid MAXIMUM is reported as a decision to "
                    "make, never silently accepted. Read it together with "
                    "val_loss_over_constant_forecast_floor below: a ratio near 1 means the "
                    "depth's validation optimum IS the no-information floor (a finding), a "
                    "ratio well below 1 means the search really was cut off (widen the grid)")}
    if floor:
        out["constant_forecast_floor_val_loss"] = float(floor)
        out["val_loss_over_constant_forecast_floor"] = {
            spec.labels[i]: float(res["val_loss"][i]) / float(floor) for i in range(n)}
        out["at_grid_max_val_over_floor"] = {
            lab: float(res["val_loss"][spec.labels.index(lab)]) / float(floor) for lab in at_max}
    return out


def _slug(label: str) -> str:
    return label.replace("+", "_").replace(" ", "_")


def _native_summary(native):
    if not native:
        return {"available": False, "reason": "not computed"}
    if not native.get("available"):
        return native
    return {k: v for k, v in native.items() if not isinstance(v, np.ndarray)}


def _jsonable_meta(meta: dict) -> dict:
    """Window metadata minus the bulky per-window lists (they live in the npz artifacts)."""
    drop = {"origins", "eligible_series_ids", "selected_series"}
    out = {k: v for k, v in (meta or {}).items() if k not in drop}
    for k in drop:
        if k in (meta or {}):
            v = meta[k]
            out[f"{k}__n"] = (len(v) if not isinstance(v, dict)
                              else {kk: len(vv) for kk, vv in v.items()})
    return out


def read_cell_summary(root: Path, model: str, dataset: str) -> dict | None:
    p = cell_dirs(root, model, dataset) / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
