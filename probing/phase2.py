"""Phase-2 contract: H3 (functional replaceability) and H4 (physical truncation).

The paper's four stages, and where each one lives:

    H1  recoverable early          Phase 1  a linear probe reads a near-final forecast before the
                                            final block (the forecasting tunnel)
    H2  geometry still evolves     Phase 1  CKA / effective rank keep changing inside the tunnel
    H3  alignment makes early      THIS     can a LIGHTWEIGHT map make h_l usable by the model's
        states usable                       OWN frozen forecasting pathway?
    H4  remove later blocks and    THIS     do the later blocks physically go away for a real
        accelerate                          latency / throughput / memory gain?

H1 established RECOVERABILITY, but the probe REPLACES the pretrained pathway, so it does not
show that the backbone can stop at an intermediate block. H3 separates the two:

    forecast recoverability  !=  native-pathway compatibility

and measures both on the SAME windows, rows and depth axis as Phase 1:

    native full model                  the horizontal reference
    hard cut                           h_l -> original final norm (if any) -> frozen native head
    label-free native-output alignment h_l -> a_l(h_l) -> same frozen pathway   (NOA, primary)
    supervised alignment               same pathway, fitted on the forecast loss (FL; Chronos-2
                                       and TiRex only -- for TimesFM-3 it is provably the probe)
    fresh Phase-1 linear probe         the recoverability reference (reused, never refit)

WHAT THIS MODULE OWNS
  the Phase-2 constants (adapter grids, epochs, lr, tolerances, the accuracy budget);
  read-only access to frozen Phase-1 cells (whitelisted tunnel fields, dependency hashes);
  the standard WQL normalization (the shared ``wql_parts`` sums over quantiles: 9x literature);
  the compatibility tunnel (Phase-1's sustained operator with the NATIVE model as reference);
  the paired cluster bootstrap over arms x depths;
  the atomic Phase-2 cell store.

WHAT IT DELIBERATELY DOES NOT OWN
  the tunnel criterion      -> probing.tunnel.sustained_tunnel_start (called, never copied)
  MASE / WQL parts          -> probing.phase1_metrics.raw_window_metrics
  the Phase-1 loss          -> probing.phase1.mean_pinball_per_window
  model pathways / caches   -> probing.phase2_pathways
  adapters and their fits   -> probing.phase2_align
  physical truncation (H4)  -> probing.phase2_truncate

PHASE 1 IS READ-ONLY. Nothing here writes into a Phase-1 results tree or a Phase-1 feature cache;
the Phase-1 caches were written non-atomically (the TimesFM-3 x M4 race), so Phase 2 never shares
a writer with them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probing import phase1
from probing.config import REPO_ROOT
from probing.tunnel import assert_tolerance_monotone, suffix_excursion, sustained_tunnel_start

__all__ = [
    "PHASE2_PROTOCOL_VERSION", "H3_FAMILIES", "FL_MODELS", "ADAPTER_LR", "ADAPTER_EPOCHS",
    "ADAPTER_EVAL_EVERY", "ADAPTER_WD_GRID", "RIDGE_KAPPAS", "TUNNEL_TOLS", "HEADLINE_TOL",
    "BUDGET", "BOOT_B", "SEED", "QUANTILES", "MEDIAN_INDEX", "LOW_SKILL_RATIO",
    "DEFAULT_PHASE1_ROOT", "DEFAULT_OUT_ROOT", "FL_TIMESFM3_THEOREM",
    "Phase1DependencyError", "Phase1Cell", "resolve_phase1_root", "parse_overrides",
    "PHASE1_REROUTES",
    "assert_adapter_wd_grid", "standard_wql", "compatibility_tunnel", "cluster_replicates",
    "ratio_summary", "H4_OUTCOME_RULE", "H4_STATUSES", "h4_outcome",
    "FRONTIER_BUDGETS", "FRONTIER_ARMS", "LATENCY_KIND_FOR_ARM", "BUDGET_RULE", "budget_depth",
    "budget_operating_points",
    "Phase2CellStore", "clean_staging", "file_sha256", "array_sha256",
    "atomic_write_json", "write_csv",
]

# --------------------------------------------------------------------------- #
# the frozen Phase-2 protocol
# --------------------------------------------------------------------------- #
#: Bumped BY HAND when the scientific meaning of an H3 cell changes. Part of the config hash.
PHASE2_PROTOCOL_VERSION = "phase2/h3-v2"      # v2: validation MASE + the H4 frontier

QUANTILES = phase1.PHASE1_QUANTILES                  # the nine canonical levels, unchanged
MEDIAN_INDEX = 4                                     # tau = 0.5 inside QUANTILES

#: The H3 ladder families. ``probe`` is not a family here: it is read from Phase 1, never refit.
#:   hard  the raw intermediate state through the frozen native pathway (no parameters)
#:   noa   native-output alignment, LABEL-FREE (the primary alignment)
#:   fl    forecast-loss alignment through the same frozen pathway (supervised; C2 + TiRex)
#:   ra    hidden-space ridge alignment h_l -> h_L (label-free DIAGNOSTIC; representation R^2)
H3_FAMILIES = ("hard", "noa", "fl", "ra")
FL_MODELS = ("chronos2", "tirex")

FL_TIMESFM3_THEOREM = (
    "TimesFM-3's native head is linear, g(x) = W x + c with W in R^{576 x 1280} of full row rank "
    "576. For an affine adapter a(x) = A x + b the pathway is W A x + (W b + c). Because W has "
    "full row rank, {W A : A in R^{1280 x 1280}} is the set of ALL 576 x 1280 matrices, so a "
    "forecast-loss adapter followed by the frozen head spans exactly the function class of the "
    "Phase-1 linear probe Linear(1280, 576). It is therefore not a separate rung: the probe IS "
    "the supervised alignment for TimesFM-3 (tested in code: tests.test_phase2_h3, contract 9).")

# Iterative adapters (Chronos-2, TiRex). Full-batch AdamW at the Phase-1 probe learning rate.
ADAPTER_LR = 1e-2
ADAPTER_EPOCHS = 300
ADAPTER_EVAL_EVERY = 10
#: Decoupled decay on Delta only, i.e. toward the IDENTITY (A = I + Delta). 0 = no decay. The
#: Phase-1 ``lr * wd < 1`` wall applies: at lr = 1e-2 the largest lr * wd here is 0.1.
ADAPTER_WD_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

#: Closed-form ridge strengths RELATIVE to the problem's own scale, lambda = kappa * g_max * m_max
#: (largest eigenvalue of X^T X times that of the metric M). A scale-relative grid spans
#: "effectively unregularized" to "effectively the identity" for EVERY dataset and depth, which
#: an absolute grid cannot promise across hidden widths 512 / 768 / 1280.
RIDGE_KAPPAS = tuple(float(10.0 ** k) for k in range(-12, 3))

TUNNEL_TOLS = phase1.PHASE1_TUNNEL_TOLS              # (0.01, 0.02, 0.05, 0.10)
HEADLINE_TOL = 0.05                                  # the frozen Phase-1 headline tolerance
BUDGET = 0.05                                        # "within 5% of the full native model"
BOOT_B = phase1.PHASE1_BOOT_B                        # 5000, seed 0, the Phase-1 bootstrap
SEED = 0
#: Descriptive low-skill flag: the native model's validation loss sits at >= 90% of the
#: closed-form no-information floor (Phase-1 ``constant_forecast_floor``). Pre-registered.
LOW_SKILL_RATIO = 0.9

DEFAULT_PHASE1_ROOT = REPO_ROOT / "results" / "three_model_phase1_q9_expanded"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "three_model_phase2"


def assert_adapter_wd_grid(grid, lr: float) -> tuple:
    """Refuse a decay grid that leaves AdamW's decoupled regime (the Phase-1 wall, lr*wd < 1)."""
    g = [float(w) for w in grid]
    if not g:
        raise ValueError("the adapter weight-decay grid is empty")
    if any(w < 0 for w in g) or len(set(g)) != len(g):
        raise ValueError(f"adapter weight-decay grid must be non-negative and unique: {g}")
    bad = [w for w in g if w * float(lr) >= phase1.PHASE1_WD_DECOUPLED_LIMIT]
    if bad:
        raise ValueError(f"adapter decay candidates {bad} give lr*wd >= "
                         f"{phase1.PHASE1_WD_DECOUPLED_LIMIT} at lr={lr:g}; AdamW's decoupled "
                         "decay would zero or flip Delta every step")
    return tuple(sorted(g))


# --------------------------------------------------------------------------- #
# the standard WQL (F5 of the design)
# --------------------------------------------------------------------------- #
def standard_wql(num, den, n_quantiles: int = len(QUANTILES)) -> float:
    """Literature WQL from the SHARED per-window parts.

    ``phase1_metrics.wql_parts`` returns numerator_w = 2 * sum_{q,h} rho_q, i.e. it SUMS over the Q
    levels; the literature WQL (Chronos papers, fev-bench) is the MEAN over levels. So

        WQL = sum_w num_w / (Q * sum_w den_w)

    Per-window parts are untouched, so every ratio and every bootstrap is unaffected; only the
    absolute scale is brought to the published convention.
    """
    num = np.asarray(num, np.float64)
    den = np.asarray(den, np.float64)
    return float(num.sum() / (float(n_quantiles) * den.sum()))


# --------------------------------------------------------------------------- #
# frozen Phase-1 cells -- read-only, whitelisted
# --------------------------------------------------------------------------- #
class Phase1DependencyError(RuntimeError):
    """A Phase-1 input Phase 2 depends on is missing, failed, pending or inconsistent."""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def array_sha256(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class Phase1Cell:
    """One frozen Phase-1 ``model x dataset`` cell, opened READ-ONLY.

    Only a whitelisted set of fields is ever read from ``tunnel.json``: the sustained entrance
    index and label at each saved tolerance. That file also carries TEST-split curves, and nothing
    in Phase 2 that selects anything may see them -- so they are not merely unused, they are never
    loaded into a Phase-2 selection path (contract: tests.test_phase2_h3, contract 13).
    """
    root: Path
    model: str
    dataset: str

    @property
    def path(self) -> Path:
        return Path(self.root) / self.model / self.dataset

    def marker(self) -> dict:
        m = self.path / "COMPLETE"
        if not m.exists():
            raise Phase1DependencyError(
                f"Phase-1 cell {self.model}/{self.dataset} under {self.root} has no COMPLETE "
                "marker (failed, pending or superseded). Phase 2 never runs on an incomplete "
                "Phase-1 cell; finish Phase 1 first (or point --phase1-override at the rerun).")
        mk = json.loads(m.read_text())
        if mk.get("model") != self.model or mk.get("dataset") != self.dataset:
            raise Phase1DependencyError(f"COMPLETE marker of {self.path} names "
                                        f"{mk.get('model')}/{mk.get('dataset')}")
        return mk

    def config(self) -> dict:
        return json.loads((self.path / "cell_config.json").read_text())

    def entrances(self, tols=TUNNEL_TOLS) -> dict:
        """{tol: {"depth_axis_index", "label"}} -- the ONLY tunnel fields Phase 2 reads."""
        t = json.loads((self.path / "tunnel.json").read_text())
        if t.get("definition") != phase1.TUNNEL_DEFINITION_VERSION:
            raise Phase1DependencyError(
                f"{self.path}/tunnel.json is definition {t.get('definition')!r}, not the frozen "
                f"{phase1.TUNNEL_DEFINITION_VERSION!r}; rebuild the Phase-1 tunnels first")
        out = {}
        for tol in tols:
            key = f"tol_{float(tol):g}"
            if key not in t.get("by_tolerance", {}):
                raise Phase1DependencyError(f"{self.path}/tunnel.json has no entrance at {key}")
            e = t["by_tolerance"][key]
            out[float(tol)] = {"depth_axis_index": int(e["depth_axis_index"]),
                               "label": str(e["label"])}
        return out

    def entrance(self, tol: float = HEADLINE_TOL) -> int:
        return self.entrances((tol,))[float(tol)]["depth_axis_index"]

    def arrays(self) -> dict:
        with np.load(self.path / "bootstrap_inputs.npz", allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def predictions(self, split: str) -> dict:
        """The frozen probe's saved (n, Q, H) predictions in the model's NORMALIZED target space
        (``pred__<label>``), with the normalized ``target`` and ``cluster_ids`` used to prove row
        alignment. H3 pushes them through the pathway's own inverse to get the probe's raw-unit
        VALIDATION MASE -- the quantity the H4 budget rule selects on."""
        if split not in ("val", "test"):
            raise ValueError(split)
        p = self.path / f"predictions_{split}.npz"
        if not p.exists():
            raise Phase1DependencyError(f"{p} missing: the Phase-1 cell did not save {split} "
                                        "predictions (needed for the probe's validation MASE)")
        with np.load(p, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def floor_val_loss(self):
        """The closed-form no-information floor's VALIDATION loss (Phase-1 probe_hparams.json)."""
        p = self.path / "probe_hparams.json"
        if not p.exists():
            return None
        fl = (json.loads(p.read_text()).get("constant_forecast_floor") or {})
        v = fl.get("val_loss")
        return None if v is None else float(v)

    def dependency_record(self) -> dict:
        mk = self.marker()
        cfg = self.config()
        rec = {"root": str(self.root), "model": self.model, "dataset": self.dataset,
               "config_hash": mk.get("config_hash"), "fit_hash": mk.get("fit_hash"),
               "postprocess_hash": mk.get("postprocess_hash"),
               "completed_utc": mk.get("completed_utc"),
               "protocol": mk.get("protocol"), "tunnel_definition": mk.get("tunnel_definition"),
               "window_digest": cfg.get("window_digest"), "checkpoint": cfg.get("checkpoint"),
               "quantile_set": cfg.get("quantile_set")}
        for name in ("bootstrap_inputs.npz", "tunnel.json", "cell_config.json",
                     "predictions_val.npz", "predictions_test.npz"):
            p = self.path / name
            rec[f"sha256_{name}"] = file_sha256(p) if p.exists() else None
        if cfg.get("quantile_set") != "q9":
            raise Phase1DependencyError(f"{self.path} is a {cfg.get('quantile_set')!r} cell; "
                                        "Phase 2 reads the Q=9 headline cells only")
        return rec


def parse_overrides(items) -> dict:
    """``["TAG=ROOT", ...]`` -> {tag: Path}. Used for the LOOP Seattle lr=1e-3 rerun."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--phase1-override expects TAG=ROOT, got {it!r}")
        tag, root = it.split("=", 1)
        out[tag.strip()] = Path(root.strip())
    return out


#: Phase-1 cells SUPERSEDED by a targeted rerun. The default Phase-1 tree still holds the old
#: cells (COMPLETE, but their probes were fit at lr=1e-2 and clipped the weight-decay grid), so a
#: plain lookup would silently read the superseded entrance. They are rerouted to the rerun tree
#: by default, and an explicit --phase1-override always wins.
PHASE1_REROUTES = {
    "LOOP_SEATTLE_5T": REPO_ROOT / "results" / "three_model_phase1_q9_loop_lowlr",
}


def resolve_phase1_root(tag: str, default_root, overrides: dict | None = None) -> Path:
    if overrides and tag in overrides:
        return Path(overrides[tag])
    if tag in PHASE1_REROUTES and Path(default_root).resolve() == DEFAULT_PHASE1_ROOT.resolve():
        return PHASE1_REROUTES[tag]
    return Path(default_root)


# --------------------------------------------------------------------------- #
# the compatibility tunnel -- the SAME sustained operator, native reference
# --------------------------------------------------------------------------- #
def compatibility_tunnel(val_by_depth, val_native: float, tols=TUNNEL_TOLS) -> dict:
    """Earliest depth after which an arm's VALIDATION loss stays within tol of the NATIVE model.

        l_arm(tol) = min { l : max_{j >= l} V_arm(j) / V_native - 1 <= tol }

    The curve handed to ``probing.tunnel.sustained_tunnel_start`` is [V_arm(Emb..L-1), V_native]:
    the final-depth entry is REPLACED by the native value, which is exact for the hard cut and for
    the label-free alignment (both ARE the native pathway at the final block) and makes the
    reference unambiguous for the supervised arm. An entrance equal to the final index means "no
    earlier depth stays within tol": no truncation is licensed at that tolerance.

    A secondary diagnostic only -- the main operating point of H3/H4 is the frozen Phase-1 entrance.
    """
    v = np.asarray(list(val_by_depth)[:-1] + [float(val_native)], np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"compatibility curve has non-finite entries: {v.tolist()}")
    L = len(v) - 1
    ent = {float(t): int(sustained_tunnel_start(v, t)) for t in tols}
    assert_tolerance_monotone(ent)
    return {"reference": "native", "definition": phase1.TUNNEL_DEFINITION_VERSION,
            "criterion": "min { l : max_{j>=l} V_arm(j)/V_native - 1 <= tol }  (VALIDATION)",
            "curve": v.tolist(), "suffix_excursion": suffix_excursion(v).tolist(),
            "by_tolerance": {f"tol_{t:g}": e for t, e in ent.items()},
            "no_truncation_at": [f"tol_{t:g}" for t, e in ent.items() if e == L]}


# --------------------------------------------------------------------------- #
# the paired cluster bootstrap over arms x depths
# --------------------------------------------------------------------------- #
def cluster_replicates(per_window, cluster_ids, B: int = BOOT_B, seed: int = SEED,
                       reduce: str = "mean"):
    """(B, R) bootstrap replicates of R per-window rows under ONE shared count matrix.

    ``per_window`` is (R, n). ``reduce="mean"`` gives the window-mean of each row (MASE, the
    normalized loss); ``reduce="sum"`` gives the resampled SUM (a WQL numerator or denominator).
    ``probing.stats.cluster_bootstrap_counts(S, B, seed)`` is a pure function of (S, B, seed), so
    every call for the same dataset draws the SAME resampled clusters -- which is what keeps every
    arm-vs-arm and arm-vs-native comparison paired, across calls and across files.
    """
    from probing.stats import cluster_bootstrap_counts
    W = np.atleast_2d(np.asarray(per_window, np.float64))
    uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    S = len(uniq)
    sums = np.zeros((S, W.shape[0]))
    np.add.at(sums, inv, W.T)
    M = cluster_bootstrap_counts(S, B, seed)
    num = M @ sums
    if reduce == "sum":
        return num
    cnt = np.bincount(inv, minlength=S).astype(np.float64)
    return num / (M @ cnt)[:, None]


def ratio_summary(point_arm: float, point_ref: float, rep_arm, rep_ref) -> dict:
    """Point ratio arm/ref and its percentile 95% CI from paired replicates (ratio - 1 is the
    relative degradation)."""
    rep = np.asarray(rep_arm, np.float64) / np.asarray(rep_ref, np.float64)
    lo, hi = np.percentile(rep, [2.5, 97.5])
    r = float(point_arm) / float(point_ref)
    return {"ratio": r, "ratio_ci_lo": float(lo), "ratio_ci_hi": float(hi),
            "degradation": r - 1.0, "degradation_ci_lo": float(lo) - 1.0,
            "degradation_ci_hi": float(hi) - 1.0,
            "within_budget": bool(r - 1.0 <= BUDGET),
            "budget_fragile": bool(lo - 1.0 <= BUDGET < hi - 1.0)}


# --------------------------------------------------------------------------- #
# THE H4 outcome -- pre-registered 2026-09-23, before any H4 number exists
# --------------------------------------------------------------------------- #
#: One candidate depth per model x dataset, one test, and no second chance:
#:   candidate_depth       = the frozen Phase-1 sustained 5% tunnel entrance
#:   successful_truncation = the label-free ALIGNED truncation's (NOA) test-MASE degradation vs
#:                           the full native model <= BUDGET (point estimate)
#:   if it is false, THAT FAILURE IS THE RESULT: no other depth is searched, evaluated or
#:   substituted. The H3 compatibility tunnel is descriptive and is never an H4 operating point.
H4_OUTCOME_RULE = {
    "version": "h4-outcome/v1",
    "candidate_depth": "the frozen Phase-1 sustained tunnel entrance at tol 0.05",
    "decision_arm": "noa (label-free aligned truncation)",
    "decision_metric": "test MASE ratio vs the full native model minus 1 (point estimate)",
    "budget": BUDGET,
    "on_failure": "report the failure; never search for, evaluate or substitute another depth",
    "fragile": "the paired 95% cluster-bootstrap CI of the decision metric contains the budget",
    "reported_not_deciding": ["noa WQL", "fl", "hard", "probe_head", "chronos2_small"],
}
#:   success                 blocks removed, NOA verified (V3), degradation <= budget
#:   failure                 blocks removed, NOA verified (V3), degradation >  budget
#:   no_truncation_possible  the Phase-1 entrance IS the final block: nothing to remove, so the
#:                           cell is neither a success nor a failure (counted separately)
#:   invalid_v3              the physical NOA model does not reproduce its offline H3 forecast:
#:                           a pipeline error to fix, never reported as a finding
H4_STATUSES = ("success", "failure", "no_truncation_possible", "invalid_v3")


def h4_outcome(arms: dict, *, candidate_depth: int, phase1_entrance: int, n_blocks: int,
               label: str, low_skill=None) -> dict:
    """Apply ``H4_OUTCOME_RULE`` to ONE evaluated model x dataset cell.

    ``arms`` are ``run_phase2_truncation``'s per-arm records (``mase_vs_native`` /
    ``wql_vs_native`` from :func:`ratio_summary`, ``V3_passed``). Pure: it takes one depth and
    returns one verdict -- there is no depth argument to vary and nothing here iterates depths.
    """
    if int(candidate_depth) != int(phase1_entrance):
        raise ValueError(f"H4 candidate depth {candidate_depth} is not the frozen Phase-1 "
                         f"entrance {phase1_entrance}: H4 has exactly one operating point")
    if not 0 <= int(candidate_depth) <= int(n_blocks):
        raise ValueError(f"candidate depth {candidate_depth} outside 0..{n_blocks}")
    if "noa" not in arms:
        raise ValueError("the H4 outcome is decided by the label-free aligned truncation (NOA), "
                         "and this cell has no NOA arm")
    noa = arms["noa"]
    m, w = noa["mase_vs_native"], noa["wql_vs_native"]
    removed = int(n_blocks) - int(candidate_depth)
    if removed == 0:
        status, ok = "no_truncation_possible", None
    elif noa.get("V3_passed") is not True:
        status, ok = "invalid_v3", None
    else:
        ok = bool(m["degradation"] <= BUDGET)
        status = "success" if ok else "failure"
    secondary = {}
    for arm in ("fl", "hard", "probe_head"):
        if arm in arms:
            r = arms[arm]["mase_vs_native"]
            secondary[arm] = {"mase_degradation": r["degradation"],
                              "mase_degradation_ci": [r["degradation_ci_lo"],
                                                      r["degradation_ci_hi"]],
                              "within_budget": r["within_budget"],
                              "V3_passed": arms[arm].get("V3_passed")}
    return {"rule": H4_OUTCOME_RULE, "status": status, "successful_truncation": ok,
            "candidate_depth": {"label": str(label), "depth_index": int(candidate_depth),
                                "source": "frozen Phase-1 sustained tunnel entrance, tol 0.05"},
            "blocks_total": int(n_blocks), "blocks_removed": removed,
            "aligned_arm": "noa",
            "aligned_mase_degradation": m["degradation"],
            "aligned_mase_degradation_ci": [m["degradation_ci_lo"], m["degradation_ci_hi"]],
            "aligned_wql_degradation": w["degradation"],
            "aligned_wql_degradation_ci": [w["degradation_ci_lo"], w["degradation_ci_hi"]],
            "budget": BUDGET,
            "fragile": bool(m["budget_fragile"]) if status in ("success", "failure") else None,
            "low_skill": low_skill, "secondary_not_deciding": secondary,
            "depth_search_performed": False}


# --------------------------------------------------------------------------- #
# THE H4 main result -- the accuracy-compute frontier (pre-registered 2026-09-23, before any
# full Phase-2 result): how much accuracy survives a given cut, and, for an accuracy BUDGET, the
# earliest usable cut chosen on VALIDATION ONLY
# --------------------------------------------------------------------------- #
FRONTIER_BUDGETS = (0.02, 0.05, 0.10, 0.20)
#: The truncation arms on the frontier. ``ra`` (hidden-space ridge) is a diagnostic and is kept
#: in the CSVs only. Which arm gives the best frontier is NOT assumed -- it is reported per model.
FRONTIER_ARMS = ("hard", "noa", "fl", "probe")
#: The latency-harness configuration that times each arm at a given depth.
LATENCY_KIND_FOR_ARM = {"hard": "hard", "noa": "adapter", "fl": "adapter", "ra": "adapter",
                        "probe": "probe_head"}
BUDGET_RULE = {
    "version": "h4-frontier/v1",
    "question": "how much forecasting accuracy survives a given reduction in inference cost?",
    "operating_point": "per model x dataset x arm x budget eps: the SHALLOWEST block depth l < L "
                       "whose VALIDATION MASE ratio to the full native model is <= 1 + eps at "
                       "EVERY truncation depth j in [l, L-1] (sustained, the Phase-1 operator)",
    "fallback": "no qualifying depth -> no truncation: the full native model (speedup 1)",
    "budgets": list(FRONTIER_BUDGETS),
    "selection_data": "validation only; the selected depth is frozen, then test is read once",
    "entrance_test": "the frozen Phase-1 entrance test (H4_OUTCOME_RULE) is kept unchanged as "
                     "the stricter secondary hypothesis test",
}


def budget_operating_points(frontier: dict, arms=FRONTIER_ARMS) -> list[dict]:
    """The distinct truncated (arm, depth) operating points a cell's ``frontier.json`` selected
    (on validation) over all budgets -- exactly what H4 physically instantiates and verifies.
    Budgets that selected no cut (the full model) add no point."""
    if not frontier.get("complete_depth_sweep"):
        raise ValueError("the frontier needs a complete depth sweep (a smoke cell has none)")
    pts = {}
    for arm in arms:
        rec = frontier.get("arms", {}).get(arm)
        if not rec:
            continue
        for key, op in rec["budgets"].items():
            if not op["truncated"]:
                continue
            p = pts.setdefault((arm, int(op["depth_index"])),
                               {"arm": arm, "depth_index": int(op["depth_index"]),
                                "label": op["label"], "blocks_removed": op["blocks_removed"],
                                "budgets": []})
            p["budgets"].append(float(key.split("_")[1]))
    return [pts[k] for k in sorted(pts)]


def budget_depth(val_ratio, eps: float) -> int:
    """The H4 operating-point rule, for ONE arm of ONE cell (``BUDGET_RULE``).

    ``val_ratio[j]`` = the arm's mean VALIDATION MASE at block depth j / the full native model's,
    for j = 0..L (entry L is ignored: depth L is the full model, the fallback). Returns the
    shallowest l in [0, L-1] with ``val_ratio[j] <= 1 + eps`` for every j in [l, L-1], else L.
    A non-finite ratio never qualifies. Takes validation numbers only -- there is no test input.
    """
    r = np.asarray(val_ratio, np.float64)
    L = len(r) - 1
    if L < 1:
        raise ValueError("need at least one truncation depth")
    ok = np.isfinite(r[:L]) & (r[:L] <= 1.0 + float(eps))
    depth = L
    for j in range(L - 1, -1, -1):
        if not ok[j]:
            break
        depth = j
    return int(depth)


# --------------------------------------------------------------------------- #
# atomic, resumable Phase-2 cell store
# --------------------------------------------------------------------------- #
atomic_write_json = phase1.atomic_write_json
write_csv = phase1.write_csv

H3_REQUIRED = ("cell_config.json", "summary.json", "depth_metrics.csv", "tunnels.json",
               "ladder_at_tunnel.json", "bootstrap_inputs.npz", "verification.json",
               "fits.json", "predictions_selected.npz", "frontier.json")


class Phase2CellStore:
    """The Phase-1 invariant, for Phase-2 cells: a cell directory exists in its final location
    ONLY if it is complete. Built in ``<kind>/<model>/<dataset>.building-<pid>/``, validated,
    ``COMPLETE`` written inside the staging dir, then ``os.replace``d into place."""

    def __init__(self, root, kind: str, model: str, dataset: str, required=H3_REQUIRED):
        self.root, self.kind, self.model, self.dataset = Path(root), kind, model, dataset
        self.required = tuple(required)

    @property
    def final(self) -> Path:
        return self.root / self.kind / self.model / self.dataset

    def staging(self) -> Path:
        return self.root / self.kind / self.model / f"{self.dataset}.building-{os.getpid()}"

    def read_marker(self):
        m = self.final / "COMPLETE"
        if not m.exists():
            return None
        try:
            return json.loads(m.read_text())
        except Exception:
            return {"config_hash": "unreadable"}

    def status(self, config_hash: str) -> tuple[str, str]:
        mk = self.read_marker()
        if mk is None:
            return "pending", "no COMPLETE marker"
        if mk.get("config_hash") != config_hash:
            return "incompatible", (f"COMPLETE carries {mk.get('config_hash')} but this run is "
                                    f"{config_hash}")
        missing = [a for a in self.required if not (self.final / a).exists()]
        if missing:
            return "corrupt", f"COMPLETE but missing {missing}"
        return "complete", "validated"

    def begin(self) -> Path:
        st = self.staging()
        if st.exists():
            shutil.rmtree(st)
        st.mkdir(parents=True)
        return st

    def commit(self, stage: Path, config_hash: str, extra: dict | None = None) -> Path:
        stage = Path(stage)
        missing = [a for a in self.required if not (stage / a).exists()]
        if missing:
            raise RuntimeError(f"{self.kind}/{self.model}/{self.dataset}: refusing COMPLETE, "
                               f"missing {missing}")
        marker = {"config_hash": config_hash, "kind": self.kind, "model": self.model,
                  "dataset": self.dataset, "protocol": PHASE2_PROTOCOL_VERSION,
                  "completed_utc": datetime.datetime.now(datetime.timezone.utc)
                                   .isoformat(timespec="seconds"),
                  "artifacts": sorted(str(p.relative_to(stage))
                                      for p in stage.rglob("*") if p.is_file()),
                  **(extra or {})}
        (stage / "COMPLETE").write_text(json.dumps(marker, indent=2, default=str))
        final = self.final
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        os.replace(stage, final)
        return final

    def abandon(self, stage: Path) -> None:
        if Path(stage).exists():
            shutil.rmtree(stage, ignore_errors=True)


def clean_staging(root, kind: str, cells=None) -> list[str]:
    """Remove stale ``.building-*`` staging dirs left by a killed job.

    ``cells`` = the (model, dataset) pairs THIS job owns. A driver must always pass it: several
    jobs may share one output root (one job per model, or a model split across dataset halves),
    and an unscoped sweep at one job's start-up would delete the cell another job is building.
    ``cells=None`` sweeps everything and is for tests / manual cleanup with no job running."""
    removed = []
    base = Path(root) / kind
    if not base.exists():
        return removed
    pats = (["*/*.building-*"] if cells is None
            else [f"{m}/{t}.building-*" for m, t in cells])
    for pat in pats:
        for p in base.glob(pat):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p.relative_to(root)))
    return removed
