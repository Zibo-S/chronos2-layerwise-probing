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

TWO CRITERIA LIVE HERE, and which one a caller uses is part of that experiment's identity:
  * ``tunnel_start``            FIRST-CROSSING (above). The criterion every COMMITTED Chronos-2 /
                                TimesFM-3 / TiRex result was computed under; unchanged, so no
                                published number moves.
  * ``sustained_tunnel_start``  SUSTAINED ENTRY. The PHASE-1 HEADLINE from 2026-09-22: the
                                earliest layer after which the excursion never again exceeds tol,
                                i.e. M is bounded by tol BY DEFINITION inside the tunnel. An
                                isolated early dip no longer opens one.
Phase 1 reports both — the sustained entrance as "the tunnel", the first-crossing value beside it
under the explicitly diagnostic name ``first_crossing_<tol>``.

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

from probing import registry
from probing.config import BOOT_B, LAST_LAYER, NUM_LAYERS, SEED
from probing.stats import ci_bounds, cluster_bootstrap_apply, cluster_bootstrap_counts

TUNNEL_TOL = 0.05   # first-crossing threshold: 95% performance = loss within 5% of the last layer

# ---------------------------------------------------------------------------
# LEGACY, CHRONOS-2-RELATIVE ROSTERS. These are now DERIVED from probing.registry rather than
# hand-typed here (and in run_cka_analysis, run_compression_cost and make_erank_stability_figure,
# which each kept their own copy). Filtering PAPER7 by window builder reproduces the historical
# tuples in their historical ORDER, which committed figures index positionally.
#
# They are kept ONLY so the committed Chronos-2 experiment lines keep running unchanged. They
# must not be used to decide which window builder a dataset gets (that is registry.builder) or
# to group a figure by pretraining status (that is registry.provenance, which is per MODEL).
# A pt_id/pt_ood label is meaningful for Chronos-2 alone; TimesFM-3 and TiRex have different
# corpora, so there is no model-independent "is this dataset seen".
# ---------------------------------------------------------------------------
PT_ID_TAGS = tuple(t for t in registry.PAPER7 if registry.builder(t) == "rolling_within_series")
PT_OOD_TAGS = tuple(t for t in registry.PAPER7 if registry.builder(t) == "rolling_cluster")


def domain_status(tag):
    """LEGACY Chronos-2-relative label: {"pretraining": "pt_id"|"pt_ood", "adaptation": None}.

    DEPRECATED for anything new — use ``registry.provenance(tag, model)``, which reports
    ``status`` and ``evidence_scope`` separately and is defined per model. This function
    survives only for the committed Chronos-2 drivers that already emit the flat label.

    It raises for any dataset outside the original seven, ON PURPOSE. The failure mode being
    prevented is a new dataset silently acquiring "pt_ood" — i.e. absence from one model's
    documentation being reported as evidence of being unseen. ``not_listed`` is not ``OOD``.
    """
    if tag in PT_ID_TAGS:
        return {"pretraining": "pt_id", "adaptation": None}
    if tag in PT_OOD_TAGS:
        return {"pretraining": "pt_ood", "adaptation": None}
    known = registry.known(tag)
    raise ValueError(
        f"no Chronos-2 pt_id/pt_ood label for {tag!r}"
        + (" — it IS in the dataset registry, but this flat label is defined only for the "
           "original seven and only relative to Chronos-2. Use "
           f"registry.provenance({tag!r}, model) instead; it will not invent a status."
           if known else
           f" — and it is not in the dataset registry at all. Add it to probing/registry.py.")
    )


def pretraining_provenance(tag, model=None):
    """The record field describing where ``tag`` sits relative to a model's pretraining corpus.

    With ``model``, returns the 2-D registry cell (status x evidence_scope + citation). Without
    one, returns ``None`` plus the reason — because there is no model-independent answer, and
    guessing one is exactly what the old flat label did.
    """
    if model is None:
        return {"model": None, "status": None,
                "note": "pretraining provenance is model-relative; pass model= to record it "
                        "(registry.MODELS = " + str(registry.MODELS) + ")"}
    return {"model": model, **registry.provenance(tag, model).as_dict()}


def _provenance_fields(tag, model):
    """The provenance keys of a tunnel record.

    ``pretraining_provenance`` (2-D, model-relative) is ALWAYS written. The legacy flat
    ``domain_status`` is written ONLY for the original seven, so committed artifacts keep their
    exact shape while a newly added dataset can never be stamped with a Chronos-2-relative
    pt_id/pt_ood label it has no evidence for.
    """
    out = {"pretraining_provenance": pretraining_provenance(tag, model)}
    if tag in PT_ID_TAGS or tag in PT_OOD_TAGS:
        out["domain_status"] = domain_status(tag)
    return out


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


# --------------------------------------------------------------------------- #
# SUSTAINED-ENTRY criterion (Phase-1 headline from 2026-09-22 onward)
# --------------------------------------------------------------------------- #
# ``tunnel_start`` above is FIRST-CROSSING: the earliest layer that dips inside the band, even
# if later layers climb back out. That makes an isolated early dip on a non-monotone curve read
# as a tunnel entrance. The sustained rule below answers the stricter question the Phase-1
# headline now asks -- "after which depth does the representation STAY recoverable?" -- and is
# defined directly in terms of the suffix excursion that ``max_excursion`` already computes:
#
#     E_l = max_{j >= l} ( L(j) / L(last) - 1 )          (the worst violation from l onward)
#     l_tunnel(tol) = min { l : E_l <= tol }
#
# Both functions stay: the committed Chronos-2 / TimesFM-3 / TiRex lines keep calling
# ``tunnel_start`` and their published numbers do not move. Phase 1 calls
# ``sustained_tunnel_start`` and reports the first-crossing value beside it, under a name that
# says what it is.
SUSTAINED_TUNNEL_DEFINITION = "sustained_suffix_v1"


def suffix_excursion(losses):
    """The full E_l curve: ``E[l] = max_{j >= l} (loss[j]/loss[last] - 1)``, one entry per layer.

    NON-INCREASING in l by construction (the max is taken over a shrinking suffix), which is
    what makes ``sustained_tunnel_start`` well defined and makes its entrance MONOTONE in the
    tolerance: a larger tol can only admit an earlier (or equal) layer. ``E[last] == 0``, so a
    non-negative tolerance is always satisfied somewhere and the scan can never come up empty.

    ``max_excursion(losses, l) == suffix_excursion(losses)[l]`` exactly -- this is the same
    statistic, computed for every l at once rather than for one boundary.
    """
    v = _validate_curve(losses)
    r = v / v[-1] - 1.0
    return np.maximum.accumulate(r[::-1])[::-1]


def sustained_tunnel_start(val_losses, tol=TUNNEL_TOL):
    """Tunnel boundary = SUSTAINED ENTRY: the earliest layer l such that l AND EVERY LATER LAYER
    stay within ``tol`` of the last layer's loss.

        l_tunnel(tol) = min { l : max_{j >= l} (val[j]/val[last] - 1) <= tol }

    VALIDATION losses only. Strictly stronger than :func:`tunnel_start`: an isolated early dip
    that a later hump climbs back out of does NOT open a tunnel, so the returned layer is always
    >= the first-crossing layer. Inside the returned tunnel ``max_excursion`` is bounded by
    ``tol`` by definition -- which is precisely what first-crossing could not promise.
    """
    tol = float(tol)
    if tol < 0:
        raise ValueError(f"tunnel tolerance must be >= 0, got {tol}")
    v = _validate_curve(val_losses)
    # The SAME comparison ``tunnel_start`` makes -- ``v[j] <= (1 + tol) * v[last]`` -- and
    # deliberately NOT ``v[j]/v[last] - 1 <= tol``. The two are the same statement in exact
    # arithmetic but differ in float rounding exactly ON the boundary (1.05/1.0 - 1 is
    # 0.050000000000000044 > 0.05, while 1.05 <= 1.05 * 1.0 holds). Sharing the comparison is
    # what guarantees the invariant ``sustained >= first_crossing`` for EVERY curve, including
    # the boundary ones, and keeps the inclusive-at-(1+tol) rule test 34 pins.
    out = np.flatnonzero(v > (1.0 + tol) * v[-1])
    return int(out[-1] + 1) if out.size else 0


def assert_tolerance_monotone(entrances):
    """``entrances`` = {tol: layer}. Larger tolerance must give an earlier-or-equal entrance.

    A THEOREM under :func:`sustained_tunnel_start` (E_l is non-increasing in l, so the admissible
    set only grows with tol), so a violation is an implementation bug, not a property of the
    data. Asserted anyway, because that is exactly the bug worth catching.
    """
    items = sorted(((float(t), int(l)) for t, l in entrances.items()), reverse=True)
    for (t_hi, l_hi), (t_lo, l_lo) in zip(items, items[1:]):
        if l_hi > l_lo:
            raise RuntimeError(
                f"tunnel entrance is not monotone in the tolerance: tol={t_hi:g} gives layer "
                f"{l_hi} but the STRICTER tol={t_lo:g} gives the earlier layer {l_lo}. Under "
                "the sustained rule this cannot happen for any curve -- it is an implementation "
                f"bug. Full map: {dict(sorted(entrances.items()))}")
    return True


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
                  extra=None, model=None):
    """Assemble the portable per-dataset tunnel record (JSON-serializable). The first-crossing
    boundary is computed from `val_losses` only; `test_losses` enter only the generalization check
    + excursion stat. `l_start`/`tunnel` are the boundary — downstream D statistics key off them."""
    v = np.asarray(val_losses, dtype=np.float64)
    t = np.asarray(test_losses, dtype=np.float64)
    ls = tunnel_start(v, tol)
    holds, margins = check_tunnel_on_test(t, ls, tol)
    rec = {
        "dataset": tag, **_provenance_fields(tag, model),
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
                        tol=TUNNEL_TOL, val_split_kind=None, extra=None, model=None):
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
        "dataset": tag, **_provenance_fields(tag, model),
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
