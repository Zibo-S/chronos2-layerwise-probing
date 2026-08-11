"""Validation-defined TUNNEL ranges + tunnel-effect statistics (PT-ID / PT-OOD framing).

Domain status is defined relative to the BACKBONE, not to where a probe was trained:
  * pt_id  — dataset was part of Chronos-2 pretraining (our 4 extended_v3_rolling sources);
  * pt_ood — dataset documented outside the pretraining corpus (sg_carpark / coastal_ts /
             boom_hourly).
An orthogonal "adaptation" axis (ft_id / ft_ood, relative to a later fine-tuned backbone) is
reserved in the record but unused until the adaptation block exists — a dataset can be pt_id
yet ft_ood, so the two axes are never collapsed.

Tunnel criterion (per dataset, ON VALIDATION ONLY). A tunnel starts at the FIRST layer that
reaches 95% of the last layer's performance (first-crossing definition, tol=0.05):
      l_start = min { l : L_val(l) <= (1 + tol) * L_val(last) }
    "when does performance FIRST reach within tol of final-layer quality?" The criterion is
    one-sided (a layer that BEATS the last layer satisfies it) and never sees the test split.
    Unlike the earlier sustained-plateau rule, a single early crossing is enough: on U-shaped
    curves (mid-layers best, late-middle hump) the tunnel opens at the first dip and MAY then
    contain a hump that rises back above (1+tol)*L(last).
Post-hoc, the EXCURSION statistic M = max_{j >= l_start} (L(j)/L(last) - 1) reports the worst
saturation violation inside the tunnel. Under first-crossing it is NOT bounded by tol on either
split — it is informative on BOTH the validation curve (does the plateau hold after entrance?)
and the test curve (does the val-defined plateau hold out of sample?).

Tunnel-effect statistics (all on TEST loss, tunnel boundary frozen from validation):
    D(dataset; l_s)  = (L_test(last) - L_test(l_s)) / L_test(l_s)     # >0: last layer worse
    D_ID(s)          = D on source s's own test set at its own l_s
    D_OOD(s, t)      = D on PT-OOD target t's test set at SOURCE s's l_s
    Delta(s, t)      = D_OOD(s, t) - D_ID(s)                          # >0: degradation stronger PT-OOD
CIs: within one dataset the cluster bootstrap is PAIRED across layers (one shared count
matrix). Delta spans two disjoint test sets, so its CI subtracts INDEPENDENT replicate
vectors (same B); it is not — and cannot be — paired across datasets.
"""
from __future__ import annotations

import numpy as np

from probing.config import BOOT_B, LAST_LAYER, NUM_LAYERS, SEED
from probing.stats import ci_bounds, cluster_bootstrap_apply, cluster_bootstrap_counts

TUNNEL_TOL = 0.05   # first-crossing threshold: 95% performance = loss within 5% of the last layer

PT_ID_TAGS = ("monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly")
PT_OOD_TAGS = ("sg_carpark", "coastal_ts", "boom_hourly")


def domain_status(tag):
    """{"pretraining": "pt_id"|"pt_ood", "adaptation": None} — adaptation filled by the
    (future) fine-tuning block, never here."""
    if tag in PT_ID_TAGS:
        return {"pretraining": "pt_id", "adaptation": None}
    if tag in PT_OOD_TAGS:
        return {"pretraining": "pt_ood", "adaptation": None}
    raise ValueError(f"no documented pretraining status for {tag!r}; "
                     f"known pt_id={PT_ID_TAGS} pt_ood={PT_OOD_TAGS}")


def _validate_curve(losses):
    v = np.asarray(losses, dtype=np.float64)
    if v.ndim != 1 or v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError(f"need a finite 1-D per-layer loss vector, got shape {v.shape}")
    return v


def tunnel_start(val_losses, tol=TUNNEL_TOL):
    """Tunnel boundary = FIRST-CROSSING: the earliest layer l that reaches 95% of the last
    layer's performance, i.e. val[l] <= (1+tol)*val[last]. Forward scan from the input embedding,
    returning the first qualifying layer. VALIDATION losses only; the last layer always satisfies
    it, so l is well-defined. NOTE (vs the old sustained rule): a later hump may rise back above
    threshold, so the tunnel can be non-monotonic and max_excursion() is now informative on the
    VALIDATION curve too, not just on test."""
    v = _validate_curve(val_losses)
    thr = (1.0 + tol) * v[-1]
    for l in range(v.size):
        if v[l] <= thr:
            return int(l)
    return int(v.size - 1)   # unreachable: v[last] <= (1+tol)*v[last] always holds


def max_excursion(losses, l_start):
    """M = max_{j >= l_start} (loss[j]/loss[last] - 1): the worst violation of saturation
    inside the tunnel. Under first-crossing it is NOT bounded by tol on either split; small M =
    a genuine plateau after entrance, large M = the tunnel is non-monotonic (a post-entrance hump)."""
    t = _validate_curve(losses)
    return float((t[l_start:] / t[-1] - 1.0).max())


def check_tunnel_on_test(test_losses, l_start, tol=TUNNEL_TOL):
    """Does the validation-defined tunnel generalize: test[j] <= (1+tol)*test[last] for every
    j >= l_start? Returns (holds, margins) with margins[j] = test[j]/test[last] - 1 (per layer,
    <= tol inside a holding tunnel). Never redefines l_start."""
    t = np.asarray(test_losses, dtype=np.float64)
    margins = t / t[-1] - 1.0
    holds = bool(np.all(t[l_start:] <= (1.0 + tol) * t[-1]))
    return holds, margins


def tunnel_record(tag, val_losses, test_losses, tol=TUNNEL_TOL, val_split_kind=None,
                  extra=None):
    """Assemble the portable per-dataset tunnel record (JSON-serializable). The first-crossing
    boundary is computed from `val_losses` only; `test_losses` enter only the generalization check
    + excursion stat. `l_start`/`tunnel` are the boundary — downstream D statistics key off them."""
    v = np.asarray(val_losses, dtype=np.float64)
    t = np.asarray(test_losses, dtype=np.float64)
    ls = tunnel_start(v, tol)
    holds, margins = check_tunnel_on_test(t, ls, tol)
    rec = {
        "dataset": tag, "domain_status": domain_status(tag),
        "tolerance": float(tol), "tunnel_definition": "first_crossing_95",
        "val_split_kind": val_split_kind,
        "last_layer": int(v.size - 1),
        "val_loss_by_layer": [float(x) for x in v],
        "test_loss_by_layer": [float(x) for x in t],
        "final_layer_val_loss": float(v[-1]),
        "l_start": ls, "tunnel": [ls, int(v.size - 1)],           # first-crossing boundary
        "max_excursion_val": max_excursion(v, ls),                # NOT bounded by tol under first-crossing
        "max_excursion_test": max_excursion(t, ls),               # M where D is measured (informative)
        "test_criterion_holds": holds,                            # does the val plateau hold on test?
        "test_margins": [float(x) for x in margins],
    }
    if extra:
        rec.update(extra)
    return rec


def tunnel_record_multi(tag, val_by_run, test_by_run, run_seeds, run_type="probe_seed",
                        tol=TUNNEL_TOL, val_split_kind=None, extra=None):
    """Multi-run tunnel record: 3 independent runs, tunnel defined from the MEAN validation
    curve (never per-seed tunnel indices averaged), evaluated on the MEAN test curve.

    val_by_run / test_by_run : (n_runs, n_layers) — one full curve per run, retained verbatim
    so seed sensitivity stays plottable. `run_type` records what varied across runs
    ("probe_seed" for the frozen pretrained backbone; "ft_seed" / "random_init" for the later
    backbone conditions). D_ID and M_test are point statistics on the mean curves."""
    V = np.asarray(val_by_run, dtype=np.float64)
    T = np.asarray(test_by_run, dtype=np.float64)
    if V.ndim != 2 or V.shape != T.shape or V.shape[0] != len(run_seeds):
        raise ValueError(f"need matching (n_runs, n_layers) curves per split with one seed per "
                         f"run — got val {V.shape}, test {T.shape}, seeds {list(run_seeds)}")
    mv, mt = V.mean(axis=0), T.mean(axis=0)
    ls = tunnel_start(mv, tol)                       # MEAN val curve defines the first-crossing boundary
    holds, margins = check_tunnel_on_test(mt, ls, tol)
    rec = {
        "dataset": tag, "domain_status": domain_status(tag),
        "run_type": run_type, "run_seeds": [int(s) for s in run_seeds],
        "val_loss_by_run": V.tolist(), "test_loss_by_run": T.tolist(),
        "mean_val_loss_by_layer": mv.tolist(), "std_val_loss_by_layer": V.std(axis=0).tolist(),
        "mean_test_loss_by_layer": mt.tolist(), "std_test_loss_by_layer": T.std(axis=0).tolist(),
        "tolerance": float(tol), "tunnel_definition": "first_crossing_95",
        "last_layer": int(mv.size - 1),
        "l_start": ls, "tunnel": [ls, int(mv.size - 1)],          # first-crossing boundary
        "D_ID": float((mt[-1] - mt[ls]) / mt[ls]),
        "M_test": max_excursion(mt, ls),                          # informative (test not forced flat)
        "max_excursion_val": max_excursion(mv, ls),               # NOT bounded by tol under first-crossing
        "test_criterion_holds": holds,                            # does the val plateau hold on test?
        "test_margins": [float(x) for x in margins],
        "val_split_kind": val_split_kind,
    }
    if extra:
        rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
# D statistics with cluster-bootstrap CIs
# --------------------------------------------------------------------------- #
def _layer_mean_boot(window_loss, cluster_ids, B=BOOT_B, seed=SEED):
    """(point (L,), boot (B, L)) window-mean loss per layer under the series/cluster
    bootstrap. window_loss: (L, n) per-window losses; cluster_ids: (n,). One shared count
    matrix -> all layers paired within the dataset."""
    wl = np.asarray(window_loss, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    S, L = uniq.size, wl.shape[0]
    per_sum = np.zeros((S, L))
    for j in range(L):
        per_sum[:, j] = np.bincount(inv, weights=wl[j], minlength=S)
    per_cnt = np.bincount(inv, minlength=S).astype(np.float64)
    M = cluster_bootstrap_counts(S, B, seed)
    return wl.mean(axis=1), cluster_bootstrap_apply(M, per_sum, per_cnt)


def d_stat_boot(window_loss, cluster_ids, l_start, last=LAST_LAYER, B=BOOT_B, seed=SEED):
    """D = (loss[last] - loss[l_start]) / loss[l_start] on one dataset's test windows.

    Returns {"point", "ci": (lo, hi), "boot": (B,), "n_clusters", "n_windows"}. The ratio is
    computed INSIDE each paired replicate (shared count matrix), so the CI reflects the
    correlated layer losses, not a raw-CI/constant approximation."""
    point, boot = _layer_mean_boot(window_loss, cluster_ids, B=B, seed=seed)
    d = (point[last] - point[l_start]) / point[l_start]
    db = (boot[:, last] - boot[:, l_start]) / boot[:, l_start]
    lo, hi = ci_bounds(db)
    return {"point": float(d), "ci": (float(lo), float(hi)), "boot": db,
            "n_clusters": int(np.unique(np.asarray(cluster_ids)).size),
            "n_windows": int(np.asarray(window_loss).shape[1])}


def m_stat_boot(window_loss, cluster_ids, l_start, last=LAST_LAYER, B=BOOT_B, seed=SEED):
    """M_test = max_{j >= l_start} (loss[j]/loss[last] - 1) with a paired cluster-bootstrap CI.

    The boundary l_start stays FIXED during resampling; the max is taken INSIDE each paired
    replicate (shared count matrix). Note the max of a noisy ratio is biased upward under
    resampling — the CI describes replicate variability of the statistic, boundary fixed."""
    point, boot = _layer_mean_boot(window_loss, cluster_ids, B=B, seed=seed)
    m = float((point[l_start:last + 1] / point[last] - 1.0).max())
    mb = (boot[:, l_start:last + 1] / boot[:, last][:, None] - 1.0).max(axis=1)
    lo, hi = ci_bounds(mb)
    return {"point": m, "ci": (float(lo), float(hi)), "boot": mb}


def delta_stat(d_ood, d_id):
    """Delta(s,t) = D_OOD(s,t) - D_ID(s) from two d_stat_boot results. The two D's come from
    DISJOINT test sets, so the replicate vectors are independent — the difference CI is the
    percentile CI of (boot_ood - boot_id), an independent (unpaired) bootstrap difference."""
    db = d_ood["boot"] - d_id["boot"]
    lo, hi = ci_bounds(db)
    return {"point": float(d_ood["point"] - d_id["point"]),
            "ci": (float(lo), float(hi)), "boot": db}


def val_curve_from_selection(selection, num_layers=NUM_LAYERS):
    """Per-layer validation loss = min over the wd grid, from a fit's selection diag
    ({layer: {"val_loss_by_wd": {...}, "chosen_wd": ...}}) — the same collapse
    source_selected_layer uses, minus the argmin."""
    out = []
    for i in range(num_layers):
        sel = selection[i]
        if sel is None:
            raise ValueError(f"layer {i}: no wd-selection record (wd grid was off) — "
                             "cannot build a validation curve")
        out.append(float(min(sel["val_loss_by_wd"].values())))
    return out
