"""Phase 2f -- source-dataset scaling of zero-shot adapter transfer (Chronos-2, truncated at L6).

    Does a label-free adapter fitted on MORE source datasets transfer better to an UNSEEN target
    dataset, and does it move closer to the dataset-specific H3 adapter?

A testable hypothesis, not an assumed result: nothing here presumes a monotone trend.

One cell = (target t, source count N, order seed s). The adapter is the Phase 2c pooled fit,
UNCHANGED (``phase2_lodo.run_lodo_cell`` -> ``fit_pooled_adapter`` -> the H3 NOA fit
``phase2_align.fit_residual_adapter``): residual affine adapter from the identity, AdamW lr 1e-3,
1500 full-batch epochs, decoupled decay on Delta only over the H3 grid, validation every 25 epochs,
epoch 0 (the hard cut) a candidate, the equal-weighted per-source objective and selection
criterion. The ONLY experimental variable is WHICH source datasets the fit reads:

    pool(t)      every paper-14 dataset except t (13 datasets: the four targets have no series
                 sibling, so this is exactly the Phase 2c main-protocol fold of t)
    order(t, s)  one fixed permutation of pool(t) per seed s in {0, 1, 2}
    sources      N = 1, 4, 8: the first N datasets of order(t, s), NESTED (a larger N adds
                 datasets, it never replaces one); N = 13: the whole pool, ONE cell per target
                 shared by all three seeds (the set is identical)

The cell then reads t's TEST split once, beside the full model, the hard cut at L6 and the committed
dataset-specific H3 adapter (re-scored, gated against the H3 per-window numbers). Every Phase 2c
gate runs inside the cell unchanged.

What the design does NOT isolate (reported, never silently "fixed"): every source contributes its
full 1,394 training windows (the Phase 2c per-source budget), so a larger N also means more training
windows and more optimization work per step. Phase 2f measures the practical effect of a larger
source pool, not a pure effect of diversity at fixed data volume. A fixed-total-window control is a
documented follow-up, not part of this run.

Series families: Electricity hourly and Electricity 15-minute share meters (``SERIES_SIBLINGS``).
Both are in the N = 13 pool, so N = 13 holds 12 independent dataset families, not 13. The orders
keep the two out of the same N <= 8 prefix (a design restriction fixed before any fit).
"""

from __future__ import annotations

import hashlib
import itertools
import os
from pathlib import Path

import numpy as np

from probing import phase2_lodo as lodo
from probing.phase2 import QUANTILES

__all__ = ["PROTOCOL_VERSION", "KIND", "MODEL", "DEPTH", "N_GRID", "ORDER_SEEDS", "ORDER_RULE",
           "TARGETS", "TARGET_SELECTION_RULE", "TARGET_RATIONALE", "DOMAIN_FAMILY", "TEST_SERIES",
           "FULL_TEST_BUDGET", "REQUIRED", "select_targets", "eligible_sources", "family_of",
           "source_orders", "plan_cells", "cell_id", "expected_cell_ids", "cell_config",
           "FIT_REPRO_HARD_RTOL", "FIT_REPRO_VAL_RTOL", "FIT_REPRO_MASE_RTOL", "fit_reproduction",
           "training_work", "pooled_forecasts", "save_extras", "PROTECTED_RESULT_TREES",
           "assert_isolated"]

PROTOCOL_VERSION = "phase2f/source-scaling-v1"
KIND = "source_scaling"
MODEL = lodo.MODEL                          # chronos2
DEPTH = 6                                   # L6: 6 of the 12 blocks kept (a Phase 2c depth)
N_GRID = (1, 4, 8, 13)
ORDER_SEEDS = (0, 1, 2)
ORDER_RULE = "phase2f/source-order/v1"

# --------------------------------------------------------------------------- #
# target selection: FROZEN 2026-11-13 from dataset METADATA, before any Phase 2f / 2c result
# --------------------------------------------------------------------------- #
#: in order of sampling frequency (5min, 30min, 1H, 1D); ``select_targets`` returns roster order
TARGETS = ("LOOP_SEATTLE_5T", "monash_london_smart_meters", "boom_hourly", "m5")

#: Coarse application-domain families. The registry's domain strings are finer (Loop Seattle is
#: "road speed", SG Carpark "transport / parking"); for diversity both are transport.
DOMAIN_FAMILY = {
    "m4_hourly": "mixed", "monash_electricity_hourly": "energy", "uber_tlc_hourly": "transport",
    "wind_farms_hourly": "energy", "sg_carpark": "transport", "coastal_ts": "nature",
    "boom_hourly": "cloud", "LOOP_SEATTLE_5T": "transport", "electricity_15min": "energy",
    "SZ_TAXI_15T": "transport", "monash_london_smart_meters": "energy", "m5": "retail",
    "wiki_daily_100k": "web", "monash_traffic": "transport"}
#: Test series (= bootstrap clusters) of the committed Phase-1 / H3 windows (cluster_ids_test).
TEST_SERIES = {
    "m4_hourly": 262, "monash_electricity_hourly": 262, "uber_tlc_hourly": 262,
    "wind_farms_hourly": 262, "sg_carpark": 354, "coastal_ts": 24, "boom_hourly": 354,
    "LOOP_SEATTLE_5T": 262, "electricity_15min": 262, "SZ_TAXI_15T": 156,
    "monash_london_smart_meters": 262, "m5": 262, "wiki_daily_100k": 262, "monash_traffic": 262}
FULL_TEST_BUDGET = 262                      # the rolling protocol's val/test series budget

TARGET_SELECTION_RULE = (
    "Applied to dataset metadata only (sampling frequency, domain family, Chronos-2 pretraining "
    "provenance, number of test series), never to any transfer or truncation result. "
    "R1: no Electricity dataset (series siblings), so every target's source pool is all 13 other "
    "datasets, as in the Phase 2c main protocol. "
    "R2: the test split reaches the full 262-series budget (excludes SZ Taxi, 156 series, and "
    "Coastal T-S, 24), so every target has comparable statistical power. "
    "R3: four distinct sampling frequencies, all three provenance groups (held out / not listed / "
    "listed), four distinct domain families. "
    "R4: ties go to more targets held out from Chronos-2 pretraining (the strictest zero-shot "
    "case: unseen by the backbone and by the adapter), then to roster order.")

TARGET_RATIONALE = {
    "LOOP_SEATTLE_5T": {
        "frequency": "5min", "seasonal_period": 288, "domain_family": "transport",
        "provenance": "held_out",
        "signal": "freeway loop-detector speeds: smooth, strongly daily-periodic",
        "why": "the only 5-minute dataset; held out from Chronos-2 pretraining (fev-bench)"},
    "monash_london_smart_meters": {
        "frequency": "30min", "seasonal_period": 48, "domain_family": "energy",
        "provenance": "listed",
        "signal": "single-household electricity use: noisy and spiky",
        "why": "the only 30-minute dataset; documented pretraining data; same domain family as "
               "the two Electricity sources (a near-domain target)"},
    "boom_hourly": {
        "frequency": "1H", "seasonal_period": 24, "domain_family": "cloud",
        "provenance": "not_listed",
        "signal": "cloud observability metrics: the lowest dynamic range in the roster, many "
                  "flat stretches (Phase-1 yield screen: 12% constant future windows)",
        "why": "the hourly slot must be 'not listed' (only hourly datasets are), and SG Carpark "
               "would repeat Loop Seattle's transport family"},
    "m5": {
        "frequency": "1D", "seasonal_period": 7, "domain_family": "retail",
        "provenance": "held_out",
        "signal": "daily retail unit sales: intermittent low counts",
        "why": "daily slot; held out from Chronos-2 pretraining (Benchmark II, fev-bench); "
               "preferred to Wiki Daily (listed) by R4"},
}


def _freq_class(tag: str) -> str:
    from probing import registry
    return registry.spec(tag).freq.split("(")[0]          # "1H(agg)" (SG Carpark) -> "1H"


def select_targets(roster, *, k: int = 4, provenance=None, domain=None, test_series=None,
                   freq=None) -> tuple[list[str], dict]:
    """The frozen rule (``TARGET_SELECTION_RULE``) applied to metadata. Returns (targets, audit).
    Deterministic: every admissible k-subset is enumerated and ranked."""
    roster = list(roster)
    provenance = provenance or {t: v[0] for t, v in lodo.CHRONOS2_PROVENANCE.items()}
    domain = domain or DOMAIN_FAMILY
    test_series = test_series or TEST_SERIES
    freq = freq or {t: _freq_class(t) for t in roster}
    r1 = [t for t in roster if t not in lodo.SERIES_SIBLINGS]
    r2 = [t for t in r1 if test_series[t] >= FULL_TEST_BUDGET]
    groups = set(lodo.PROVENANCE_GROUPS)
    admissible = []
    for combo in itertools.combinations(r2, k):
        if (len({freq[t] for t in combo}) == k and {provenance[t] for t in combo} == groups
                and len({domain[t] for t in combo}) == k):
            admissible.append(combo)
    if not admissible:
        raise RuntimeError("no target set satisfies the selection rule")
    best = min(admissible, key=lambda c: (-sum(provenance[t] == "held_out" for t in c),
                                          [roster.index(t) for t in c]))
    audit = {"rule": TARGET_SELECTION_RULE,
             "excluded_R1_series_siblings": [t for t in roster if t not in r1],
             "excluded_R2_test_series_below_budget": {t: test_series[t] for t in r1
                                                      if t not in r2},
             "admissible_sets": [list(c) for c in admissible],
             "selected": list(best)}
    return list(best), audit


# --------------------------------------------------------------------------- #
# source pools and nested orders
# --------------------------------------------------------------------------- #
def eligible_sources(target: str, roster) -> list[str]:
    """Every other paper-14 dataset, in roster order: the Phase 2c main-protocol fold."""
    if target in lodo.SERIES_SIBLINGS:
        raise ValueError(f"{target!r} has a series sibling; Phase 2f targets must not (the pool "
                         "would lose a dataset and differ from 'all other datasets')")
    return lodo.fold_sources(target, roster, lodo.MAIN)


def family_of(tag: str) -> str:
    """Series family: the Electricity pair maps to one family, every other dataset to itself."""
    return "+".join(sorted({tag, *lodo.SERIES_SIBLINGS.get(tag, ())}))


def _key(target: str, seed: int, draw: int, tag: str) -> str:
    return hashlib.sha256(f"{ORDER_RULE}/{target}/{int(seed)}/{int(draw)}/{tag}"
                          .encode()).hexdigest()


def _permutation(pool, target: str, seed: int, draw: int) -> list[str]:
    """A pseudo-random permutation that depends only on (rule, target, seed, draw, tag): sorting by
    a sha256 key, so it is identical on every machine and NumPy version."""
    return sorted(pool, key=lambda t: _key(target, seed, draw, t))


def source_orders(target: str, roster, *, seeds=ORDER_SEEDS, n_grid=N_GRID,
                  max_draws: int = 10_000) -> list[dict]:
    """One permutation of the target's pool per seed. Rejection sampling (deterministic: draw
    1, 2, ... of the same seed) enforces two restrictions fixed BEFORE any fit, both blind to
    performance:

      family   no two members of one series family (the Electricity pair) inside the longest
               proper prefix (N = 8), so every N <= 8 subset holds distinct families
      first    the N = 1 source differs across the seeds of a target (three different
               single-source fits instead of a repeated one)
    """
    pool = eligible_sources(target, roster)
    grid = sorted(int(n) for n in n_grid)
    if grid[-1] != len(pool) or grid[0] < 1:
        raise ValueError(f"n_grid {grid} must end at the pool size {len(pool)}")
    proper = [n for n in grid if n < len(pool)]
    limit = max(proper) if proper else 0
    orders, firsts = [], []
    for s in seeds:
        for draw in range(1, max_draws + 1):
            perm = _permutation(pool, target, s, draw)
            fams = [family_of(t) for t in perm[:limit]]
            if len(set(fams)) == len(fams) and perm[0] not in firsts:
                break
        else:
            raise RuntimeError(f"{target} seed {s}: no admissible order in {max_draws} draws")
        firsts.append(perm[0])
        orders.append({"seed": int(s), "order": perm, "draw": draw, "rule": ORDER_RULE,
                       "restrictions": {"family_distinct_within_first": limit,
                                        "first_source_distinct_across_seeds": True}})
    return orders


def cell_id(n: int, seed) -> str:
    return f"N{int(n):02d}_" + ("all" if seed is None else f"seed{int(seed)}")


def plan_cells(roster, *, targets=TARGETS, seeds=ORDER_SEEDS, n_grid=N_GRID,
               depth: int = DEPTH) -> list[dict]:
    """Every (target, N, seed) cell, in a dict ``lodo.run_lodo_cell`` accepts unchanged.

    ``sources`` (what the fit reads) is the selected subset in ROSTER order, the Phase 2c pooled
    data order: the fit is a function of the SET, and the N = 13 cell is exactly the Phase 2c main
    cell of the same target and depth. ``selection_order`` keeps the order in which the subset was
    drawn."""
    roster = list(roster)
    cells = []
    for t in targets:
        pool = eligible_sources(t, roster)
        orders = source_orders(t, roster, seeds=seeds, n_grid=n_grid)
        for n in sorted(int(x) for x in n_grid):
            if n == len(pool):
                picks = [(None, list(pool), [o["seed"] for o in orders], None)]
            else:
                picks = [(o["seed"], o["order"][:n], None, o) for o in orders]
            for seed, sel, shared, o in picks:
                src = sorted(sel, key=roster.index)
                fams = sorted({family_of(x) for x in src})
                cells.append({
                    "variant": lodo.MAIN, "target": t, "depth": int(depth), "sources": src,
                    "excluded": [x for x in roster if x not in src],
                    "phase": "2f", "n_sources": n, "order_seed": seed, "cell_id": cell_id(n, seed),
                    "selection_order": list(sel), "pool": list(pool),
                    "n_families": len(fams),
                    "series_family_members": [x for x in src if x in lodo.SERIES_SIBLINGS],
                    "shared_across_seeds": shared,
                    "order_draw": None if o is None else o["draw"]})
    return cells


def expected_cell_ids(targets=TARGETS, seeds=ORDER_SEEDS, n_grid=N_GRID, pool_size: int = 13):
    """{(target, cell_id)} of the full design: 4 x (3 x 3 + 1) = 40."""
    out = set()
    for t in targets:
        for n in n_grid:
            out |= ({(t, cell_id(n, None))} if n == pool_size
                    else {(t, cell_id(n, s)) for s in seeds})
    return out


def cell_config(cell: dict, deps: dict, h3_ref: dict, fit_cfg: dict) -> dict:
    """The Phase 2c cell identity (protocol, objective, sources, optimizer, gates, dependencies)
    plus what defines a Phase 2f cell. Its digest is the resume key."""
    base = lodo.cell_config(cell, deps, h3_ref, fit_cfg)
    return {**base, "phase2f_protocol": PROTOCOL_VERSION, "phase2f_kind": KIND,
            "n_sources": int(cell["n_sources"]), "order_seed": cell["order_seed"],
            "order_rule": ORDER_RULE, "selection_order": list(cell["selection_order"]),
            "cell_id": cell["cell_id"]}


REQUIRED = tuple(lodo.LODO_REQUIRED) + ("selection.json", "forecasts.npz")


# --------------------------------------------------------------------------- #
# with ONE source the pooled fit IS that source's dataset-specific H3 fit
# --------------------------------------------------------------------------- #
#: epoch-0 (hard-cut) criterion: the same states, teacher and code -> agreement to float noise
FIT_REPRO_HARD_RTOL = 1e-6
#: the selected validation criterion and the source's own test MASE (the V3 mean rule's 1e-4)
FIT_REPRO_VAL_RTOL = 1e-4
FIT_REPRO_MASE_RTOL = 1e-4


def _rel(a, b) -> float:
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-12)


def fit_reproduction(fit_record: dict, adapter_sha256: str, h3_noa: dict, *, source: str,
                     label: str, source_test: dict | None = None) -> dict:
    """Compare an N = 1 pooled fit with the committed H3 NOA fit of its single source at the same
    depth (``fits.json["noa"][label]``). Phase 2c contract 5 proves the two code paths are the same
    fit; this measures it on the REAL data in THIS job's environment.

    PASS needs: the epoch-0 criterion equal to 1e-6 (same inputs), the same selected decay and
    epoch, the selected validation criterion equal to 1e-4, and (when given) the source's own test
    MASE equal to H3's NOA test MASE to 1e-4. A bitwise-identical adapter is recorded, not
    required."""
    f = fit_record
    ours = {float(c["wd"]): dict((int(e), float(v)) for e, v in c["val_curve"])
            for c in f["candidates"]}
    ref = {float(c["wd"]): dict((int(e), float(v)) for e, v in c["val_curve"])
           for c in h3_noa.get("candidates", [])}
    diffs = [_rel(ours[w][e], ref[w][e]) for w in ours if w in ref
             for e in ours[w] if e in ref[w] and np.isfinite(ref[w][e])]
    checks = {
        "hard_cut_val": {"ours": float(f["hard_cut_val"]), "h3": float(h3_noa["hard_cut_val"]),
                         "rel_diff": _rel(f["hard_cut_val"], h3_noa["hard_cut_val"]),
                         "rtol": FIT_REPRO_HARD_RTOL},
        "selected_wd": {"ours": f["selected_wd"], "h3": h3_noa["selected_wd"]},
        "selected_epoch": {"ours": int(f["selected_epoch"]), "h3": int(h3_noa["selected_epoch"])},
        "selected_val": {"ours": float(f["selected_val"]), "h3": float(h3_noa["selected_val"]),
                         "rel_diff": _rel(f["selected_val"], h3_noa["selected_val"]),
                         "rtol": FIT_REPRO_VAL_RTOL}}
    checks["hard_cut_val"]["passed"] = checks["hard_cut_val"]["rel_diff"] <= FIT_REPRO_HARD_RTOL
    checks["selected_wd"]["passed"] = (
        (f["selected_wd"] is None and h3_noa["selected_wd"] is None)
        or (f["selected_wd"] is not None and h3_noa["selected_wd"] is not None
            and float(f["selected_wd"]) == float(h3_noa["selected_wd"])))
    checks["selected_epoch"]["passed"] = int(f["selected_epoch"]) == int(h3_noa["selected_epoch"])
    checks["selected_val"]["passed"] = checks["selected_val"]["rel_diff"] <= FIT_REPRO_VAL_RTOL
    if source_test is not None:
        checks["source_test_mase_vs_h3"] = source_test
    passed = all(bool(c.get("passed")) for c in checks.values())
    return {"source": source, "label": label, "passed": passed,
            "bitwise_adapter": adapter_sha256 == h3_noa.get("adapter_sha256"),
            "adapter_sha256": adapter_sha256, "h3_adapter_sha256": h3_noa.get("adapter_sha256"),
            "val_curve_points_compared": len(diffs),
            "val_curve_max_rel_diff": float(max(diffs)) if diffs else None,
            "checks": checks,
            "rule": "N = 1 pooled fit vs the committed H3 NOA fit of its source: same epoch-0 "
                    "criterion (1e-6), same selected (wd, epoch), selected val criterion and the "
                    "source's own test MASE to 1e-4"}


# --------------------------------------------------------------------------- #
# per-cell records
# --------------------------------------------------------------------------- #
def training_work(fit: dict, rows_per_window: int) -> dict:
    """Training data volume and optimization work of one pooled fit (``fit_pooled_adapter``
    output). Every step is full batch, so window_passes = steps x training windows."""
    f = fit["fit"]
    done = 0
    for c in f["candidates"]:
        if c["status"] == "completed":
            done += int(f["epochs"])
        elif c.get("train_curve"):
            done += int(c["train_curve"][-1][0])
    n_tr, n_va = int(fit["n_train_rows"]), int(fit["n_val_rows"])
    return {"n_train_windows": n_tr, "n_val_windows": n_va,
            "n_train_slot_states": n_tr * int(rows_per_window),
            "n_train_windows_per_source": {r["source"]: int(r["n_train"])
                                           for r in fit["per_source_val"]},
            "n_val_windows_per_source": {r["source"]: int(r["n_val"])
                                         for r in fit["per_source_val"]},
            "steps_planned": len(f["grid"]) * int(f["epochs"]), "steps_completed": done,
            "window_passes": done * n_tr, "fit_seconds": float(fit["fit_seconds"])}


def pooled_forecasts(pathway, prepared: dict, target: str, depth: int, adapter) -> dict:
    """The pooled adapter's test forecasts, normalized and in raw units: the same forward pass and
    inverse ``score_arm`` uses, re-run after the cell (no selection reads them)."""
    te = prepared[target].test
    _, z = pathway.run(te.feats[int(depth)], adapter)
    return {"pooled_q_norm": z.astype(np.float32),
            "pooled_q_raw": np.asarray(pathway.to_raw(z, te), np.float32),
            "rows_test": np.asarray(te.rows), "cluster_ids_test": np.asarray(te.cluster_ids),
            "quantiles": np.asarray(QUANTILES, np.float64)}


def _atomic_savez(path: Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def save_extras(stage, cell: dict, res: dict, forecasts: dict, *, work: dict,
                fit_repro: dict | None, phase2c_crosscheck: dict | None = None,
                job: str | None = None) -> dict:
    """The Phase 2f records beside the Phase 2c ones (``lodo.save_lodo_cell`` writes those)."""
    from probing.phase2 import atomic_write_json
    stage = Path(stage)
    _atomic_savez(stage / "forecasts.npz", **forecasts)
    t = cell["target"]
    sel = {"schema": "phase2f_selection/v1", "protocol": PROTOCOL_VERSION, "target": t,
           "depth": int(cell["depth"]), "label": lodo.depth_label(cell["depth"]),
           "n_sources": int(cell["n_sources"]), "order_seed": cell["order_seed"],
           "cell_id": cell["cell_id"], "selection_order": list(cell["selection_order"]),
           "sources_fit_order": list(cell["sources"]), "pool": list(cell["pool"]),
           "excluded": list(cell["excluded"]), "n_families": int(cell["n_families"]),
           "series_family_members": list(cell["series_family_members"]),
           "both_series_siblings": len(cell["series_family_members"]) > 1,
           "shared_across_seeds": cell["shared_across_seeds"], "order_rule": ORDER_RULE,
           "order_draw": cell["order_draw"], "target_rationale": TARGET_RATIONALE.get(t),
           "training_work": work, "fit_reproduction": fit_repro,
           "phase2c_crosscheck": phase2c_crosscheck,
           "timings": res["timings"], "job": job,
           "target_labels_used": "final evaluation of the frozen adapter only (test split, "
                                 "read once); the fit reads source states and the full model's "
                                 "own outputs"}
    atomic_write_json(stage / "selection.json", sel)
    return sel


# --------------------------------------------------------------------------- #
# isolation from every other experiment
# --------------------------------------------------------------------------- #
PROTECTED_RESULT_TREES = ("three_model_phase1_q9_expanded", "three_model_phase1_q1",
                          "three_model_final", "three_model_phase2", "three_model_phase2_lodo",
                          "three_model_phase2_lodo_smoke", "three_model_phase2d_pooled",
                          "three_model_phase2e_noncontig", "three_model_phase2_nested",
                          "three_model_phase2_smoke")


def _inside(a: Path, b: Path) -> bool:
    return a == b or b in a.parents


def assert_isolated(output_root, adapter_root, repo_root, scratch=None) -> None:
    """Refuse an output root equal to, inside, or containing another experiment's results tree,
    and an adapter root that overlaps the H3 / Phase 2c / 2d / 2e artifact directories."""
    out = Path(output_root).resolve()
    res = Path(repo_root).resolve() / "results"
    for name in PROTECTED_RESULT_TREES:
        p = (res / name).resolve()
        if _inside(out, p) or _inside(p, out):
            raise ValueError(f"Phase 2f output root {out} overlaps the {name} tree ({p})")
    scratch = Path(scratch or os.environ.get("SCRATCH", "/tmp")).resolve()
    ad = Path(adapter_root).resolve()
    for p in (scratch / "chronos2_phase2", scratch / "chronos2" / "phase2d_pooled_artifacts",
              scratch / "chronos2" / "phase2e_noncontig", scratch / "chronos2" / "features_cache",
              scratch / "chronos2" / "phase1_shared_cache",
              scratch / "chronos2" / "phase1_shared_windows"):
        p = p.resolve()
        if _inside(ad, p) or _inside(p, ad):
            raise ValueError(f"Phase 2f adapter root {ad} overlaps {p}")
