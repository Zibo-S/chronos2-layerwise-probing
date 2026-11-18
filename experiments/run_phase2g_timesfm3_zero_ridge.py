"""Phase 2g -- TimesFM-3: identity-anchored vs ZERO-anchored closed-form alignment, every depth.

The committed TimesFM-3 aligned pathway (H3 "noa") is the closed form

    min_{A,b}  sum_i || A x_i + b - y_i ||_M^2  +  lambda || A - I ||_F^2 ,     M = W^T W

with x_i = the token-15 state at depth l, y_i = the same window's final-block state and W the
frozen linear head, so ||v||_M^2 = ||W v||^2 is the squared error between the truncated and the
full model's normalized head outputs. Its ridge shrinks toward the IDENTITY, i.e. toward the hard
cut. This experiment changes the anchor and nothing else:

    identity   lambda || A - I ||_F^2    the committed fit, recomputed with the committed function
    zero       lambda || A ||_F^2        PRIMARY: same objective, metric, kappa grid and selection
    zero_out   lambda || W A ||_F^2      sensitivity: plain multi-output ridge of the teacher's
                                         head outputs on the states (lambda = kappa * g_max)

As lambda -> infinity the zero-anchored fits tend to A = 0, b = mean(y): the mean full-model
output on the training windows (a constant normalized forecast), an explicit candidate beside the
kappa grid exactly as the hard cut is for the identity anchor.

Unchanged from H3: the Phase-1 token-15 caches, the windows, rows and splits, the teacher (the
full model's own head outputs from the cached final-block state), the criterion, the kappa grid
(phase2.RIDGE_KAPPAS), selection on the VALIDATION distillation error only, test read once, the
frozen pathway and its inverse, the metrics and the paired series-cluster bootstrap. No future
value enters a fit or a selection.

The hard cut, the committed identity-anchored arm, the supervised linear readout (Phase-1 probe)
and the native model are READ from the committed H3 cells (results/three_model_phase2), never
rewritten. They are recomputed here only as gates: the run must reproduce the committed native,
hard-cut and identity-anchored numbers before a zero-anchored number is reported.

Closed form. Centre X, Y; X^T X = V diag(g) V^T; M = Q diag(m) Q^T restricted to its range
(rank 576). With T = V^T X^T Y Q and B~ = V^T A^T Q:

    identity   B~_ij = (T_ij m_j + lambda [V^T Q]_ij) / (g_i m_j + lambda)
    zero       B~_ij =  T_ij m_j / (g_i m_j + lambda)
    zero_out   B~_ij =  T_ij / (g_i + lambda)

Every kappa is a re-weighting of one pair of eigendecompositions. The fraction of the least-squares
solution kept in direction (i, j) is g_i m_j / (g_i m_j + lambda) for BOTH anchors; its sum over i
is the effective number of parameters of output direction j, reported as its mean over j, its
mean weighted by each direction's share of the teacher's output variance, and for the top
direction. (The committed ``effective_dof`` field is sum_i g_i / (g_i + lambda): the value for a
direction with m_j = 1, which is not an output direction of this head.)

    python -m experiments.run_phase2g_timesfm3_zero_ridge --plan        # login node, no model
    python -m experiments.run_phase2g_timesfm3_zero_ridge --selfcheck   # synthetic, seconds
    python -m experiments.run_phase2g_timesfm3_zero_ridge --device cpu  # COMPUTE NODE (sbatch)
    python -m experiments.run_phase2g_timesfm3_zero_ridge --tables-only # from artifacts

WHERE TO RUN. A compute node (CPU is enough: 280 closed-form fits + bootstraps, about an hour on
8 cores). Only --plan and --tables-only are login-node safe.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing.phase2 import (BOOT_B, FRONTIER_BUDGETS, HEADLINE_TOL, MEDIAN_INDEX,   # noqa: E402
                            QUANTILES, RIDGE_KAPPAS, SEED, TUNNEL_TOLS, Phase1Cell,
                            Phase2CellStore, atomic_write_json, budget_depth, clean_staging,
                            cluster_replicates, compatibility_tunnel, file_sha256,
                            resolve_phase1_root, write_csv)

MODEL = "timesfm3"
KIND = "zero_ridge"
PROTOCOL = "phase2g/timesfm3-zero-ridge-v1"
DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2g_timesfm3_zero_ridge"
FAMILIES = ("identity", "zero", "zero_out")
NEW_ARMS = ("zero", "zero_out")
ARMS = ("hard", "identity", "zero", "zero_out", "best_anchor", "probe")
PER_WINDOW = ("val_loss", "val_mase", "test_loss", "test_mase", "test_mae", "test_wql_num")
REQUIRED = ("cell_config.json", "summary.json", "depth_metrics.csv", "fits.json",
            "tunnels_budgets.json", "verification.json", "bootstrap_inputs.npz")
GATE_RTOL = 1e-4                     # the Phase-2 native-gate tolerance, unchanged


# --------------------------------------------------------------------------- #
# the closed form, in the eigenbases of X^T X and of the head metric
# --------------------------------------------------------------------------- #
def metric_range(metric, out_dim: int):
    """Eigen-directions of M = W^T W inside its range: (m_r (r,), Q_r (d, r)), r = out_dim."""
    M = np.asarray(metric, np.float64)
    m, Q = np.linalg.eigh(0.5 * (M + M.T))
    m = np.clip(m, 0.0, None)
    keep = m > m.max() * 1e-12
    if int(keep.sum()) != int(out_dim):
        raise RuntimeError(f"metric rank {int(keep.sum())} != head output dimension {out_dim}")
    return m[keep], Q[:, keep]


class ReducedRidge:
    """All three families on one decomposition of one (dataset, depth)."""

    def __init__(self, X_tr, Y_tr, m_r, Q_r, out_dim: int):
        X, Y = np.asarray(X_tr, np.float64), np.asarray(Y_tr, np.float64)
        if X.shape != Y.shape:
            raise ValueError(f"shapes: X {X.shape}, Y {Y.shape}")
        self.n, self.d = X.shape
        self.p = int(out_dim)
        self.mr, self.Qr = np.asarray(m_r, np.float64), np.asarray(Q_r, np.float64)
        self.muX, self.muY = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.muX, Y - self.muY
        g, V = np.linalg.eigh(Xc.T @ Xc)
        self.g, self.V = np.clip(g, 0.0, None), V
        Yq = Yc @ self.Qr
        self.T = V.T @ (Xc.T @ Yq)                       # (d, r) = V^T X^T Y Q_r
        self.C1 = V.T @ self.Qr                          # (d, r) = V^T I Q_r
        self.scale = float(self.g.max() * self.mr.max())
        self.tr = (Xc @ V, Yq)
        self.hard_train = self._hard(X, Y)
        var = (Yq * Yq).sum(0) * self.mr                 # teacher output variance per direction
        self.out_weight = var / var.sum()

    def project(self, X, Y):
        return ((np.asarray(X, np.float64) - self.muX) @ self.V,
                (np.asarray(Y, np.float64) - self.muY) @ self.Qr)

    def _hard(self, X, Y) -> float:
        E = (np.asarray(X, np.float64) - np.asarray(Y, np.float64)) @ self.Qr
        return float(((E * E) @ self.mr).sum() / (len(E) * self.p))

    def crit(self, Xt, Yt, Bt) -> float:
        """Mean squared error between the adapted and the full model's head outputs."""
        E = -Yt if Bt is None else Xt @ Bt - Yt
        return float(((E * E) @ self.mr).sum() / (len(Yt) * self.p))

    def lam(self, family: str, kappa: float) -> float:
        return float(kappa) * (float(self.g.max()) if family == "zero_out" else self.scale)

    def coef(self, family: str, lam: float):
        if family == "zero_out":
            return self.T / (self.g[:, None] + lam)
        gm = self.g[:, None] * self.mr[None, :]
        if family == "identity":
            return (self.T * self.mr + lam * self.C1) / (gm + lam)
        if family == "zero":
            return self.T * self.mr / (gm + lam)
        raise ValueError(family)

    def dof(self, family: str, lam: float) -> dict:
        if family == "zero_out":
            s = float((self.g / (self.g + lam)).sum())
            return {"dof_mean": s, "dof_weighted": s, "dof_top": s}
        gm = self.g[:, None] * self.mr[None, :]
        per = (gm / (gm + lam)).sum(0)                   # effective parameters per output direction
        return {"dof_mean": float(per.mean()), "dof_weighted": float(per @ self.out_weight),
                "dof_top": float(per[int(np.argmax(self.mr))])}

    def path(self, family: str, X_va, Y_va, kappas=RIDGE_KAPPAS) -> dict:
        Xv, Yv = self.project(X_va, Y_va)
        if family == "identity":
            end = {"candidate": "hard_cut", "val": self._hard(X_va, Y_va), "train": self.hard_train}
        else:
            end = {"candidate": "constant", "val": self.crit(Xv, Yv, None),
                   "train": self.crit(*self.tr, None)}
        end.update({"kappa": None, "lambda": None, "dof_mean": 0.0, "dof_weighted": 0.0,
                    "dof_top": 0.0})
        cands = [end]
        for k in kappas:
            lam = self.lam(family, k)
            Bt = self.coef(family, lam)
            cands.append({"candidate": f"kappa={float(k):g}", "kappa": float(k), "lambda": lam,
                          "val": self.crit(Xv, Yv, Bt), "train": self.crit(*self.tr, Bt),
                          **self.dof(family, lam)})
        finite = [c for c in cands if np.isfinite(c["val"])]
        if not finite:
            raise RuntimeError(f"{family}: every candidate has a non-finite validation error")
        # ties -> the larger lambda (closer to the anchor); the anchor itself counts as +inf
        best = min(finite, key=lambda c: (c["val"], -(np.inf if c["lambda"] is None
                                                      else c["lambda"])))
        ks = [c["kappa"] for c in cands if c["kappa"] is not None]
        return {"family": family, "selected": best["candidate"], "selected_kappa": best["kappa"],
                "selected_lambda": best["lambda"], "selected_val": best["val"],
                "selected_train": best["train"], "selected_endpoint": best["kappa"] is None,
                "at_grid_edge": best["kappa"] is not None and best["kappa"] in (min(ks), max(ks)),
                "dof_mean": best["dof_mean"], "dof_weighted": best["dof_weighted"],
                "dof_top": best["dof_top"], "endpoint_val": end["val"], "candidates": cands,
                "Bt": None if best["kappa"] is None else self.coef(family, best["lambda"])}

    def split_error(self, family: str, sel: dict, X, Y) -> tuple[float, float]:
        """(distillation error, R^2 against that split's own mean teacher output)."""
        if family == "identity" and sel["selected_endpoint"]:
            c = self._hard(X, Y)
        else:
            c = self.crit(*self.project(X, Y), sel["Bt"])
        Yq = np.asarray(Y, np.float64) @ self.Qr
        Yq = Yq - Yq.mean(0)
        ss = float(((Yq * Yq) @ self.mr).sum() / (len(Yq) * self.p))
        return c, (1.0 - c / ss if ss > 0 else float("nan"))

    def adapt(self, X, Bt):
        """a(x) = A x + b for a zero-anchored solution, in float64 (b = mean(y) - A mean(x))."""
        X = np.asarray(X, np.float64)
        if Bt is None:
            return np.tile(self.muY, (len(X), 1))
        return ((X - self.muX) @ self.V) @ Bt @ self.Qr.T + self.muY


# --------------------------------------------------------------------------- #
# synthetic self-check (no model, no cache, seconds)
# --------------------------------------------------------------------------- #
def selfcheck() -> int:
    from probing.phase2_align import anchored_ridge_path
    rng = np.random.default_rng(0)
    n, d, p, nv = 80, 24, 10, 40
    mix = rng.normal(size=(d, d)) / np.sqrt(d)
    Wh = rng.normal(size=(p, d))
    Tm = rng.normal(size=(d, d)) * 0.3

    def draw(k):
        X = rng.normal(size=(k, d)) @ mix * np.linspace(3.0, 0.05, d)
        return X, X @ Tm + 0.2 * rng.normal(size=(k, d)) + 1.5

    X, Y = draw(n)
    Xv, Yv = draw(nv)
    M = Wh.T @ Wh
    mr, Qr = metric_range(M, p)
    red = ReducedRidge(X, Y, mr, Qr, p)
    ok, lines = True, []

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")

    # (a) the reduced identity path IS the committed function, candidate by candidate
    ref = anchored_ridge_path(X, Y, Xv, Yv, metric=M, out_dim=p, kappas=RIDGE_KAPPAS)
    mine = red.path("identity", Xv, Yv)
    worst = 0.0
    for c in ref["candidates"]:
        m = next(x for x in mine["candidates"] if x["candidate"] == c["candidate"])
        for s in ("val", "train"):
            worst = max(worst, abs(m[s] - c[s]) / max(abs(c[s]), 1e-300))
    check("reduced identity path == committed anchored_ridge_path", worst < 1e-8
          and mine["selected"] == ref["selected"], f"(max rel {worst:.1e})")
    # (b) first-order optimality of both zero-anchored solutions
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    G, R = Xc.T @ Xc, Xc.T @ Yc
    for fam in NEW_ARMS:
        res = 0.0
        for k in (1e-6, 1e-3, 1e-1):
            lam = red.lam(fam, k)
            B = red.V @ red.coef(fam, lam) @ Qr.T
            pen = lam * B if fam == "zero" else lam * B @ M
            res = max(res, np.abs(G @ B @ M + pen - R @ M).max() / np.abs(R @ M).max())
        check(f"{fam}: normal equations hold", res < 1e-9, f"(max rel residual {res:.1e})")
    # (c) lambda -> 0: both anchors give the least-squares outputs (n > d)
    P = {f: red.tr[0] @ red.coef(f, red.lam(f, 1e-14)) for f in FAMILIES}
    diff = max(np.abs(P[f] - P["identity"]).max() for f in NEW_ARMS) / np.abs(P["identity"]).max()
    check("lambda -> 0: the three families coincide", diff < 1e-6, f"(max rel {diff:.1e})")
    # (d) lambda -> infinity: the zero anchors reach the constant candidate
    for fam in NEW_ARMS:
        c_inf = red.crit(*red.tr, red.coef(fam, red.lam(fam, 1e14)))
        c_end = red.crit(*red.tr, None)
        check(f"{fam}: lambda -> inf is the mean teacher output", abs(c_inf - c_end) < 1e-8 * c_end)
    # (e) the criterion is the output-space MSE of W a(x) against W y, on adapted STATES
    sel = red.path("zero", Xv, Yv)
    out = red.adapt(Xv, sel["Bt"]) @ Wh.T
    mse = float(((out - Yv @ Wh.T) ** 2).sum() / (nv * p))
    check("criterion == MSE of head outputs on adapted states",
          abs(mse - sel["selected_val"]) < 1e-9 * max(mse, 1e-300),
          f"({mse:.6g} vs {sel['selected_val']:.6g})")
    # (f) dof limits
    check("dof: 0 at the anchor side, rank at lambda -> 0",
          red.dof("zero", 1e30)["dof_mean"] < 1e-6
          and abs(red.dof("zero", 1e-30)["dof_mean"] - int((red.g > 0).sum())) < 1e-6)
    print("SELFCHECK (synthetic)")
    print("\n".join(lines))
    print("SELFCHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
# committed H3 cell (read-only)
# --------------------------------------------------------------------------- #
H3_FILES = ("bootstrap_inputs.npz", "fits.json", "depth_metrics.csv", "summary.json")


def load_committed(h3_root, tag: str) -> dict:
    cell = Path(h3_root) / "h3" / MODEL / tag
    if not (cell / "COMPLETE").exists():
        raise phase2.Phase1DependencyError(f"no COMPLETE committed H3 cell at {cell}")
    with np.load(cell / "bootstrap_inputs.npz", allow_pickle=True) as z:
        arr = {k: z[k] for k in z.files}
    return {"path": str(cell), "arrays": arr,
            "fits": json.loads((cell / "fits.json").read_text()),
            "summary": json.loads((cell / "summary.json").read_text()),
            "sha256": {n: file_sha256(cell / n) for n in H3_FILES}}


def _arm_from_committed(arr: dict, fam: str) -> dict:
    return {k: np.asarray(arr[f"{fam}__{k}"], np.float64) for k in PER_WINDOW
            if f"{fam}__{k}" in arr}


def _flat(a):
    a = np.asarray(a)
    return a.reshape(-1, a.shape[-1])


def _rel_mean(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(abs(a.mean() - b.mean()) / max(abs(b.mean()), 1e-12))


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #
def run_cell(tag: str, pathway, data, committed: dict, entrances: dict, *, boot_b: int,
             seed: int, kappas=RIDGE_KAPPAS, depths=None, log=print) -> dict:
    from probing.phase1_metrics import raw_window_metrics
    from probing.phase2_align import affine_to_adapter, anchored_ridge_path
    q = QUANTILES
    L, labels = data.reference_index, list(data.depth_labels)
    n_dep = L + 1
    tr, va, te = (data.splits[s] for s in ("train", "val", "test"))
    c = committed["arrays"]
    device = pathway.device
    pathway.assert_frozen()
    verification = {}

    # ---- rows: the committed H3 cell and this run describe the same windows ------------------
    rows_ok = (np.array_equal(np.asarray(c["cluster_ids_test"]), np.asarray(te.cluster_ids))
               and np.array_equal(np.asarray(c["cluster_ids_val"]), np.asarray(va.cluster_ids))
               and np.array_equal(np.asarray(c["rows_test"]), np.asarray(te.rows))
               and np.array_equal(np.asarray(c["rows_val"]), np.asarray(va.rows)))
    if not rows_ok:
        raise phase2.Phase1DependencyError(f"{tag}: rows / cluster ids differ from the committed "
                                           "H3 cell; refusing to pair numbers across row sets")
    verification["rows_equal_committed_h3"] = True

    def score(states_val, states_test, adapter=None) -> dict:
        _, zv = pathway.run(states_val, adapter)
        _, zt = pathway.run(states_test, adapter)
        m = raw_window_metrics(tag, te.X, te.y_raw, pathway.to_raw(zt, te), q, MEDIAN_INDEX)
        vm = raw_window_metrics(tag, va.X, va.y_raw, pathway.to_raw(zv, va), q, MEDIAN_INDEX)
        return {"val_loss": np.asarray(phase1.mean_pinball_per_window(zv, va.target, q), np.float64),
                "val_mase": np.asarray(vm["mase_pw"], np.float64),
                "test_loss": np.asarray(phase1.mean_pinball_per_window(zt, te.target, q),
                                        np.float64),
                "test_mase": np.asarray(m["mase_pw"], np.float64),
                "test_mae": np.asarray(m["mae_pw"], np.float64),
                "test_wql_num": np.asarray(m["wql_num_pw"], np.float64)}

    # ---- native: recomputed, gated against the committed arrays ------------------------------
    nat = score(va.feats[L], te.feats[L])
    g_nat = {k: _rel_mean(nat[k], c[f"native_{k}"]) for k in ("test_mase", "test_loss", "val_loss")}
    verification["native_vs_committed"] = {
        "mean_rel_diff": g_nat, "rtol": GATE_RTOL,
        "test_mase_window_max_abs_diff": float(np.abs(nat["test_mase"]
                                                      - c["native_test_mase"]).max()),
        "passed": bool(max(g_nat.values()) <= GATE_RTOL)}
    if not verification["native_vs_committed"]["passed"]:
        raise RuntimeError(f"{tag}: the recomputed native pathway does not reproduce the committed "
                           f"H3 native arrays: {verification['native_vs_committed']}")

    W, _c = pathway.linear_head()
    M, p = W.T @ W, W.shape[0]
    mr, Qr = metric_range(M, p)
    shape = {"val": (n_dep, len(va.target)), "test": (n_dep, len(te.target))}
    new = {a: {k: np.full(shape[k.split("_")[0]], np.nan) for k in PER_WINDOW}
           for a in NEW_ARMS + ("identity_recomputed", "hard_recomputed")}
    fits, timings = {}, {}
    fit_depths = [l for l in (range(L) if depths is None else sorted(set(depths))) if l != L]
    gate_rows = []

    for l in fit_depths:
        t0 = time.time()
        lab = labels[l]
        Xtr, Ytr = _flat(tr.feats[l]), _flat(tr.feats[L])
        Xva, Yva = _flat(va.feats[l]), _flat(va.feats[L])
        Xte, Yte = _flat(te.feats[l]), _flat(te.feats[L])

        # hard cut, recomputed (gate only)
        s = score(va.feats[l], te.feats[l])
        for k in PER_WINDOW:
            new["hard_recomputed"][k][l] = s[k]

        # identity anchor: the COMMITTED function, unchanged (reproduction gate)
        r_id = anchored_ridge_path(Xtr, Ytr, Xva, Yva, metric=M, out_dim=p, kappas=kappas)
        ad = affine_to_adapter(r_id["A"], r_id["b"]).to(device)
        s = score(va.feats[l], te.feats[l], ad)
        for k in PER_WINDOW:
            new["identity_recomputed"][k][l] = s[k]
        cf = (committed["fits"].get("noa") or {}).get(lab, {})
        gate = {"label": lab,
                "identity_test_mase_rel_diff": _rel_mean(s["test_mase"], c["noa__test_mase"][l]),
                "hard_test_mase_rel_diff": _rel_mean(new["hard_recomputed"]["test_mase"][l],
                                                     c["hard__test_mase"][l]),
                "kappa_recomputed": r_id["selected_kappa"],
                "kappa_committed": cf.get("selected_kappa"),
                "kappa_equal": r_id["selected_kappa"] == cf.get("selected_kappa"),
                "selected_val_rel_diff": (abs(r_id["selected_val"] - cf["selected_val"])
                                          / max(abs(cf["selected_val"]), 1e-300)
                                          if cf.get("selected_val") is not None else None)}

        # the three families on one decomposition
        red = ReducedRidge(Xtr, Ytr, mr, Qr, p)
        rec = {"n_train_rows": int(red.n), "d": int(red.d), "out_dim": int(p),
               "g_max": float(red.g.max()), "g_min": float(red.g.min()),
               "m_max": float(mr.max()), "m_min_in_range": float(mr.min()),
               "identity_committed": {k: cf.get(k) for k in
                                      ("selected", "selected_kappa", "selected_val",
                                       "hard_cut_val", "at_grid_edge")}}
        sels = {}
        for fam in FAMILIES:
            sel = red.path(fam, Xva, Yva, kappas)
            sels[fam] = sel
            e_te, r2_te = red.split_error(fam, sel, Xte, Yte)
            _, r2_tr = red.split_error(fam, sel, Xtr, Ytr)
            _, r2_va = red.split_error(fam, sel, Xva, Yva)
            rec[fam] = {k: v for k, v in sel.items() if k != "Bt"} | {
                "distill_train": sel["selected_train"], "distill_val": sel["selected_val"],
                "distill_test": e_te, "r2_train": r2_tr, "r2_val": r2_va, "r2_test": r2_te}
        # the reduced identity path against the committed function, candidate by candidate
        worst = 0.0
        for cc in r_id["candidates"]:
            mm = next(x for x in sels["identity"]["candidates"] if x["candidate"] == cc["candidate"])
            worst = max(worst, abs(mm["val"] - cc["val"]) / max(abs(cc["val"]), 1e-300))
        gate["reduced_identity_vs_committed_function_max_rel_val"] = worst
        gate["reduced_identity_same_selection"] = sels["identity"]["selected"] == r_id["selected"]
        gate_rows.append(gate)

        for fam in NEW_ARMS:
            sv = red.adapt(Xva, sels[fam]["Bt"]).astype(np.float32)[:, None, :]
            st = red.adapt(Xte, sels[fam]["Bt"]).astype(np.float32)[:, None, :]
            s = score(sv, st)
            for k in PER_WINDOW:
                new[fam][k][l] = s[k]
        rec["best_anchor"] = ("zero" if sels["zero"]["selected_val"] < r_id["selected_val"]
                              else "identity")
        rec["best_anchor_rule"] = ("smaller VALIDATION distillation error of the two selected "
                                   "fits; ties -> identity (post hoc union of the two families)")
        fits[lab] = rec
        timings[lab] = round(time.time() - t0, 2)
        log(f"    [{tag}] {lab:>4}  identity k={r_id['selected_kappa']} val {r_id['selected_val']:.4g}"
            f" | zero k={sels['zero']['selected_kappa']} val {sels['zero']['selected_val']:.4g}"
            f" | zero_out k={sels['zero_out']['selected_kappa']} val "
            f"{sels['zero_out']['selected_val']:.4g}  ({timings[lab]} s)")

    # ---- gates ---------------------------------------------------------------------------------
    worst_id = max((g["identity_test_mase_rel_diff"] for g in gate_rows), default=0.0)
    worst_hard = max((g["hard_test_mase_rel_diff"] for g in gate_rows), default=0.0)
    verification["identity_reproduction"] = {
        "max_rel_diff_mean_test_mase": worst_id, "rtol": GATE_RTOL,
        "depths_with_a_different_kappa": [g["label"] for g in gate_rows if not g["kappa_equal"]],
        "passed": bool(worst_id <= GATE_RTOL)}
    verification["hard_cut_reproduction"] = {
        "max_rel_diff_mean_test_mase": worst_hard, "rtol": GATE_RTOL,
        "passed": bool(worst_hard <= GATE_RTOL)}
    verification["reduced_identity_path"] = {
        "max_rel_diff_val_vs_committed_function": max(
            (g["reduced_identity_vs_committed_function_max_rel_val"] for g in gate_rows),
            default=0.0),
        "same_selection_at_every_depth": all(g["reduced_identity_same_selection"]
                                             for g in gate_rows)}
    verification["per_depth"] = gate_rows
    gates_passed = bool(verification["identity_reproduction"]["passed"]
                        and verification["hard_cut_reproduction"]["passed"]
                        and verification["reduced_identity_path"]["same_selection_at_every_depth"])
    verification["gates_passed"] = gates_passed

    # ---- arms: committed numbers stay the reference; only the zero arms are new ---------------
    native = {k: np.asarray(c[f"native_{k}"], np.float64) for k in PER_WINDOW
              if f"native_{k}" in c}
    wql_den = np.asarray(c["test_wql_den"], np.float64)
    arms = {"hard": _arm_from_committed(c, "hard"), "identity": _arm_from_committed(c, "noa"),
            "zero": new["zero"], "zero_out": new["zero_out"],
            "probe": _arm_from_committed(c, "probe")}
    for a in NEW_ARMS:                                   # reference depth: the native pathway
        for k in PER_WINDOW:
            arms[a][k][L] = native[k]
    best = {k: np.array(arms["identity"][k], copy=True) for k in PER_WINDOW}
    for l in fit_depths:
        if fits[labels[l]]["best_anchor"] == "zero":
            for k in PER_WINDOW:
                best[k][l] = arms["zero"][k][l]
    arms["best_anchor"] = best
    arms = {a: arms[a] for a in ARMS}
    eval_depths = sorted(set(fit_depths) | {L})

    # ---- paired series-cluster bootstrap (one count matrix for every arm and depth) -----------
    cid = np.asarray(te.cluster_ids)
    keys = [("native", None)] + [(a, l) for a in ARMS for l in eval_depths]
    idx = {k: i for i, k in enumerate(keys)}
    src = {"mase": "test_mase", "loss": "test_loss", "wql": "test_wql_num"}
    reps, points = {}, {}
    for metric, key in src.items():
        Wm = np.stack([native[key]] + [arms[a][key][l] for a, l in keys[1:]])
        if metric == "wql":
            num = cluster_replicates(np.nan_to_num(Wm), cid, boot_b, seed, reduce="sum")
            den = cluster_replicates(wql_den[None, :], cid, boot_b, seed, reduce="sum")[:, 0]
            reps[metric] = num / (len(q) * den[:, None])
            points[metric] = Wm.sum(axis=1) / (len(q) * wql_den.sum())
        else:
            reps[metric] = cluster_replicates(np.nan_to_num(Wm), cid, boot_b, seed, reduce="mean")
            points[metric] = Wm.mean(axis=1)

    def ratio(metric, a, l, ref=("native", None)):
        i, j = idx[(a, l)], idx[ref]
        pa, pb = points[metric][i], points[metric][j]
        if not (np.isfinite(pa) and np.isfinite(pb)):
            return None
        lo, hi = np.percentile(reps[metric][:, i] / reps[metric][:, j], [2.5, 97.5])
        return float(pa / pb), float(lo), float(hi)

    nv_mase = float(np.mean(native["val_mase"]))
    nv_loss = float(np.mean(native["val_loss"]))
    ent = int(entrances[HEADLINE_TOL]["depth_axis_index"])
    rows = []
    for a in ARMS:
        for l in eval_depths:
            rm, rw, rl = ratio("mase", a, l), ratio("wql", a, l), ratio("loss", a, l)
            if rm is None:
                continue
            row = {"model": MODEL, "dataset": tag, "arm": a, "depth_index": l, "label": labels[l],
                   "relative_depth": l / L, "is_phase1_entrance": l == ent,
                   "val_loss": float(np.nanmean(arms[a]["val_loss"][l])),
                   "val_loss_ratio_vs_native": float(np.nanmean(arms[a]["val_loss"][l])) / nv_loss,
                   "val_mase_ratio_vs_native": (float(np.mean(arms[a]["val_mase"][l])) / nv_mase
                                                if "val_mase" in arms[a] else ""),
                   "test_mase": float(points["mase"][idx[(a, l)]]),
                   "mase_ratio_vs_native": rm[0], "mase_ratio_ci_lo": rm[1],
                   "mase_ratio_ci_hi": rm[2],
                   "wql_ratio_vs_native": "" if rw is None else rw[0],
                   "loss_ratio_vs_native": "" if rl is None else rl[0]}
            for ref in ("identity", "probe"):
                rr = ratio("mase", a, l, (ref, l))
                row[f"mase_ratio_vs_{ref}"] = "" if rr is None else rr[0]
                row[f"mase_ratio_vs_{ref}_ci_lo"] = "" if rr is None else rr[1]
                row[f"mase_ratio_vs_{ref}_ci_hi"] = "" if rr is None else rr[2]
            fam = {"identity": "identity", "zero": "zero", "zero_out": "zero_out",
                   "best_anchor": None, "hard": None, "probe": None}[a]
            fr = fits.get(labels[l], {})
            if a == "best_anchor" and fr:
                fam = fr["best_anchor"]
                row["anchor_chosen"] = fam
            f = fr.get(fam, {}) if fam else {}
            for k in ("selected", "selected_kappa", "at_grid_edge", "selected_endpoint",
                      "dof_mean", "dof_weighted", "dof_top", "distill_train", "distill_val",
                      "distill_test", "r2_train", "r2_val", "r2_test"):
                v = f.get(k)
                row[k] = "" if v is None else v
            row.setdefault("anchor_chosen", "")
            rows.append(row)

    # ---- native-referenced tunnel, budget rule, frozen Phase-1 entrance (post hoc readings) ---
    tb = {"phase1_entrance": {"depth_index": ent, "label": labels[ent]},
          "native_val_loss": nv_loss, "native_val_mase": nv_mase, "arms": {}}
    complete = eval_depths == list(range(n_dep))
    for a in ARMS:
        rec = {}
        re_ = ratio("mase", a, ent) if ent in eval_depths else None
        rec["at_phase1_entrance"] = (None if re_ is None else
                                     {"mase_ratio": re_[0], "ci": [re_[1], re_[2]],
                                      "within_5pct": bool(re_[0] - 1.0 <= 0.05)})
        if complete:
            ct = compatibility_tunnel(np.nanmean(arms[a]["val_loss"], axis=1), nv_loss)
            rec["native_referenced_tunnel"] = ct["by_tolerance"]
            if "val_mase" in arms[a]:
                vcurve = np.array([np.mean(arms[a]["val_mase"][l]) for l in range(n_dep)]) / nv_mase
                rec["budgets"] = {}
                for eps in FRONTIER_BUDGETS:
                    d_sel = budget_depth(vcurve, eps)
                    rt = (1.0, 1.0, 1.0) if d_sel == L else ratio("mase", a, d_sel)
                    rec["budgets"][f"eps_{eps:g}"] = {
                        "depth_index": int(d_sel), "label": labels[d_sel],
                        "truncated": bool(d_sel != L), "val_mase_ratio": float(vcurve[d_sel]),
                        "test_mase_ratio": rt[0], "within_budget_test": bool(rt[0] - 1.0 <= eps)}
        tb["arms"][a] = rec

    boot = {"depth_labels": np.asarray(labels, dtype=object),
            "arms": np.asarray(list(ARMS), dtype=object),
            "cluster_ids_test": np.asarray(te.cluster_ids, np.int64),
            "cluster_ids_val": np.asarray(va.cluster_ids, np.int64),
            "rows_test": te.rows, "rows_val": va.rows}
    for a in NEW_ARMS + ("identity_recomputed", "hard_recomputed"):
        for k in PER_WINDOW:
            boot[f"{a}__{k}"] = np.asarray(new[a][k] if a not in arms else arms[a][k], np.float64)
    for k in PER_WINDOW:
        boot[f"native_recomputed__{k}"] = nat[k]
    timings["fit_total_s"] = round(sum(v for v in timings.values()), 1)
    return {"rows": rows, "fits": fits, "tunnels_budgets": tb, "verification": verification,
            "bootstrap_inputs": boot, "timings": timings, "gates_passed": gates_passed,
            "depths_evaluated": eval_depths, "complete_depth_sweep": complete}


# --------------------------------------------------------------------------- #
# tables (artifacts only)
# --------------------------------------------------------------------------- #
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def make_tables(out) -> int:
    out = Path(out)
    cells = sorted(p.parent for p in (out / KIND / MODEL).glob("*/COMPLETE"))
    if not cells:
        print("no COMPLETE cells: no tables")
        return 1
    rows, tbs, gates = [], {}, {}
    for c in cells:
        with open(c / "depth_metrics.csv", newline="") as fh:
            rows += list(csv.DictReader(fh))
        tbs[c.name] = json.loads((c / "tunnels_budgets.json").read_text())
        gates[c.name] = json.loads((c / "verification.json").read_text())
    comb = out / "combined"
    write_csv(comb / "depth_metrics.csv", rows)
    tags = sorted({r["dataset"] for r in rows})
    labels = {int(r["depth_index"]): r["label"] for r in rows}
    L = max(labels)
    by = {(r["dataset"], r["arm"], int(r["depth_index"])): r for r in rows}

    def col(arm, l, key):
        return [_f(by[(t, arm, l)][key]) for t in tags if (t, arm, l) in by]

    depth_rows, fit_rows = [], []
    for l in sorted(labels):
        if l == L:
            continue
        row = {"depth_index": l, "label": labels[l], "n_datasets": len(col("zero", l, "test_mase"))}
        for a in ARMS:
            d = [x - 1.0 for x in col(a, l, "mase_ratio_vs_native")]
            row[f"dmase_median_{a}"] = _med(d)
            row[f"within5_{a}"] = int(sum(x <= 0.05 for x in d))
            row[f"within10_{a}"] = int(sum(x <= 0.10 for x in d))
            row[f"dwql_median_{a}"] = _med([x - 1.0 for x in col(a, l, "wql_ratio_vs_native")])
        for a, ref in (("zero", "identity"), ("zero_out", "identity"), ("zero", "probe"),
                       ("zero_out", "probe"), ("best_anchor", "probe"), ("identity", "probe")):
            hi = col(a, l, f"mase_ratio_vs_{ref}_ci_hi")
            lo = col(a, l, f"mase_ratio_vs_{ref}_ci_lo")
            row[f"{a}_better_than_{ref}"] = int(sum(x < 1.0 for x in hi))
            row[f"{a}_worse_than_{ref}"] = int(sum(x > 1.0 for x in lo))
            row[f"{a}_over_{ref}_median"] = _med(col(a, l, f"mase_ratio_vs_{ref}"))
        row["best_anchor_is_zero"] = int(sum(by[(t, "best_anchor", l)]["anchor_chosen"] == "zero"
                                             for t in tags if (t, "best_anchor", l) in by))
        depth_rows.append(row)
        for fam in FAMILIES:
            k = [x for x in col(fam, l, "selected_kappa") if np.isfinite(x)]
            tr_, va_, te_ = (col(fam, l, f"distill_{s}") for s in ("train", "val", "test"))
            fit_rows.append({
                "depth_index": l, "label": labels[l], "family": fam,
                "kappa_log10_median": float(np.median(np.log10(k))) if k else "",
                "kappa_min": min(k) if k else "", "kappa_max": max(k) if k else "",
                "n_endpoint_selected": int(sum(str(by[(t, fam, l)]["selected_endpoint"]) == "True"
                                               for t in tags if (t, fam, l) in by)),
                "n_at_grid_edge": int(sum(str(by[(t, fam, l)]["at_grid_edge"]) == "True"
                                          for t in tags if (t, fam, l) in by)),
                "dof_weighted_median": _med(col(fam, l, "dof_weighted")),
                "dof_mean_median": _med(col(fam, l, "dof_mean")),
                "dof_top_median": _med(col(fam, l, "dof_top")),
                "distill_train_median": _med(tr_), "distill_val_median": _med(va_),
                "distill_test_median": _med(te_),
                "val_over_train_median": _med([v / t for v, t in zip(va_, tr_) if t > 0]),
                "r2_val_median": _med(col(fam, l, "r2_val")),
                "r2_test_median": _med(col(fam, l, "r2_test")),
                "n_r2_test_negative": int(sum(x < 0 for x in col(fam, l, "r2_test")))})
    write_csv(comb / "summary_by_depth.csv", depth_rows)
    write_csv(comb / "fit_by_depth.csv", fit_rows)

    eb = []
    for t in tags:
        for a in ARMS:
            rec = tbs[t]["arms"].get(a, {})
            e = rec.get("at_phase1_entrance") or {}
            r = {"dataset": t, "arm": a, "phase1_entrance": tbs[t]["phase1_entrance"]["label"],
                 "mase_ratio_at_entrance": e.get("mase_ratio", ""),
                 "within_5pct_at_entrance": e.get("within_5pct", "")}
            for k, v in (rec.get("native_referenced_tunnel") or {}).items():
                r[f"native_tunnel_{k}"] = v
            for k, v in (rec.get("budgets") or {}).items():
                r[f"budget_{k}_depth"] = v["depth_index"]
                r[f"budget_{k}_test_ratio"] = v["test_mase_ratio"]
                r[f"budget_{k}_within"] = v["within_budget_test"]
            eb.append(r)
    write_csv(comb / "entrance_budget.csv", eb)

    def pct(x):
        return "n/a" if not np.isfinite(x) else f"{100 * x:+.1f}%"

    md = [f"# TimesFM-3: identity- vs zero-anchored closed-form alignment ({len(tags)} of 14 "
          "datasets)", "",
          "Median test MASE change vs the full model over datasets. `identity`, `hard` and `probe` "
          "are the committed H3 numbers; `zero` (primary) and `zero_out` are new.", "",
          "| depth | hard | identity | zero | zero_out | best anchor | probe | zero < identity "
          "(CI) | zero vs probe better / worse (CI) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in depth_rows:
        md.append(f"| {r['label']} | {r['dmase_median_hard'] + 1:.0f}x | "
                  + " | ".join(pct(r[f"dmase_median_{a}"]) for a in
                               ("identity", "zero", "zero_out", "best_anchor", "probe"))
                  + f" | {r['zero_better_than_identity']} / {r['n_datasets']} | "
                    f"{r['zero_better_than_probe']} / {r['zero_worse_than_probe']} |")
    md += ["", "Fit diagnostics (median over datasets; distillation error = MSE between the "
           "adapted and the full model's normalized head outputs).", "",
           "| depth | family | log10 kappa | dof (variance-weighted) | train | val | test | "
           "val / train | R2 test | R2 test < 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in fit_rows:
        k = r["kappa_log10_median"]
        md.append(f"| {r['label']} | {r['family']} | {'' if k == '' else f'{k:.1f}'} | "
                  f"{r['dof_weighted_median']:.0f} | {r['distill_train_median']:.4g} | "
                  f"{r['distill_val_median']:.4g} | {r['distill_test_median']:.4g} | "
                  f"{r['val_over_train_median']:.1f} | {r['r2_test_median']:.3f} | "
                  f"{r['n_r2_test_negative']} |")
    failed = [t for t in tags if not gates[t].get("gates_passed")]
    md += ["", f"Reproduction gates passed for {len(tags) - len(failed)} of {len(tags)} datasets."
           + (f" FAILED: {failed}" if failed else "")]
    (out / "report").mkdir(parents=True, exist_ok=True)
    (out / "report" / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_root)
    if out.resolve() == Path(args.h3_root).resolve():
        raise SystemExit("--output-root must not be the committed H3 tree")
    if args.selfcheck:
        return selfcheck()
    if args.tables_only:
        return make_tables(out)
    tags = [t for t in registry.roster("paper14") if not args.datasets or t in args.datasets]

    if args.plan:
        print(f"\nPHASE 2g PLAN  {len(tags)} datasets -> {out / KIND / MODEL}")
        bad = 0
        for tag in tags:
            root = resolve_phase1_root(tag, args.phase1_root, {})
            notes = []
            try:
                Phase1Cell(root, MODEL, tag).marker()
            except Exception as e:
                notes.append(f"phase1 NOT READY ({type(e).__name__})")
            if not (Path(args.h3_root) / "h3" / MODEL / tag / "COMPLETE").exists():
                notes.append("no committed H3 cell")
            if not (Path(args.window_cache) / f"windows__{args.suite}__{tag}.npz").exists():
                notes.append("no window cache")
            n_cache = len(list(Path(args.cache_root).glob(f"*__{args.suite}__{tag}__*")))
            if n_cache < 3:
                notes.append(f"{n_cache}/3 last-token cache roots")
            bad += bool(notes)
            print(f"  {tag:<28} {'ready' if not notes else '; '.join(notes)}")
        return 1 if bad else 0

    import torch
    from experiments.run_phase2_h3 import windows_readonly
    from probing.phase2_env import environment_record, set_precision_flags
    from probing.phase2_pathways import TimesFM3Pathway, load_timesfm3
    from probing.timesfm3_last_token import get_model
    from probing.window_reference import window_digest
    numerics = set_precision_flags(deterministic=True)
    (out / KIND).mkdir(parents=True, exist_ok=True)
    clean_staging(out, KIND, cells=[(MODEL, t) for t in tags])
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    run_dir = out / KIND / "runs" / job
    atomic_write_json(run_dir / "run_manifest.json", {
        "schema": "phase2g_zero_ridge_run/v1", "protocol": PROTOCOL, "job": job,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "phase1_root": str(args.phase1_root), "h3_root": str(args.h3_root),
        "cache_root": str(args.cache_root), "window_cache": str(args.window_cache),
        "datasets": tags, "kappas": list(RIDGE_KAPPAS), "boot_b": args.boot_b, "seed": args.seed,
        "numerics": numerics, "environment": environment_record()})
    state, failures, gate_fail, pathway = {"cells": {}}, [], [], None

    for tag in tags:
        key = f"{MODEL}/{tag}"
        print(f"\n{'#' * 86}\n# {tag}  ({registry.display_name(tag)})\n{'#' * 86}")
        try:
            root = resolve_phase1_root(tag, args.phase1_root, {})
            p1 = Phase1Cell(root, MODEL, tag)
            dep = p1.dependency_record()
            committed = load_committed(args.h3_root, tag)
            h3_dep = committed["summary"].get("phase1_dependency", {})
            if h3_dep.get("fit_hash") != dep["fit_hash"]:
                raise phase2.Phase1DependencyError(
                    f"{key}: the committed H3 cell was built on Phase-1 fit {h3_dep.get('fit_hash')}"
                    f", the Phase-1 cell on disk is {dep['fit_hash']}")
            ccfg = {"protocol": PROTOCOL, "model": MODEL, "dataset": tag,
                    "phase1_fit_hash": dep["fit_hash"], "window_digest": dep["window_digest"],
                    "h3_sha256": committed["sha256"], "kappas": list(RIDGE_KAPPAS),
                    "families": list(FAMILIES), "boot_b": args.boot_b, "seed": args.seed,
                    "depths": None if not args.depths else [int(d) for d in args.depths],
                    "gate_rtol": GATE_RTOL}
            chash = phase1._digest(ccfg)
            store = Phase2CellStore(out, KIND, MODEL, tag, required=REQUIRED)
            status, reason = store.status(chash)
            if status == "complete" and not args.force_recompute:
                print(f"[skip] {key}: COMPLETE ({reason})")
                state["cells"][key] = {"status": "complete"}
                continue
            if status == "incompatible" and not args.force_recompute:
                raise RuntimeError(f"{key} exists under a different configuration ({reason})")
            w = windows_readonly(tag, args)
            if window_digest(w) != dep["window_digest"]:
                raise phase2.Phase1DependencyError(f"{key}: window digest differs from Phase 1")
            if pathway is None:
                t0 = time.time()
                mdl, _dev = get_model(p1.config()["checkpoint"], args.device)
                pathway = TimesFM3Pathway.from_model(mdl)
                print(f"  [model] TimesFM-3 head ready on {pathway.device} in {time.time() - t0:.1f}s")
            t0 = time.time()
            data = load_timesfm3(tag, w, p1.arrays(), p1.config(), cache_dir=args.cache_root)
            print(f"  [data] loaded in {time.time() - t0:.1f}s; rows train/val/test = "
                  f"{[len(data.splits[s].target) for s in ('train', 'val', 'test')]}")
            res = run_cell(tag, pathway, data, committed, p1.entrances(), boot_b=args.boot_b,
                           seed=args.seed, depths=args.depths or None)
            after = {n: file_sha256(Path(committed["path"]) / n) for n in H3_FILES}
            res["verification"]["committed_h3_files_unchanged"] = after == committed["sha256"]
            stage = store.begin()
            try:
                atomic_write_json(stage / "cell_config.json", {**ccfg, "config_hash": chash})
                write_csv(stage / "depth_metrics.csv", res["rows"])
                atomic_write_json(stage / "fits.json", res["fits"])
                atomic_write_json(stage / "tunnels_budgets.json", res["tunnels_budgets"])
                atomic_write_json(stage / "verification.json", res["verification"])
                np.savez_compressed(stage / "bootstrap_inputs.npz", **res["bootstrap_inputs"])
                atomic_write_json(stage / "summary.json", {
                    "schema": "phase2g_zero_ridge_cell/v1", "model": MODEL, "dataset": tag,
                    "gates_passed": res["gates_passed"],
                    "complete_depth_sweep": res["complete_depth_sweep"],
                    "depths_evaluated": res["depths_evaluated"], "timings": res["timings"],
                    "phase1_dependency": dep, "committed_h3": {"path": committed["path"],
                                                               "sha256": committed["sha256"]},
                    "provenance": data.provenance, "pathway": pathway.describe()})
                final = store.commit(stage, chash, extra={"gates_passed": res["gates_passed"]})
            except Exception:
                store.abandon(stage)
                raise
            v = res["verification"]
            print(f"  gates: native {v['native_vs_committed']['passed']}, hard "
                  f"{v['hard_cut_reproduction']['passed']} "
                  f"({v['hard_cut_reproduction']['max_rel_diff_mean_test_mase']:.1e}), identity "
                  f"{v['identity_reproduction']['passed']} "
                  f"({v['identity_reproduction']['max_rel_diff_mean_test_mase']:.1e}; different "
                  f"kappa at {v['identity_reproduction']['depths_with_a_different_kappa']})")
            print(f"  COMPLETE {final}")
            state["cells"][key] = {"status": "complete", "gates_passed": res["gates_passed"],
                                   "timings": res["timings"]}
            if not res["gates_passed"]:
                gate_fail.append(tag)
            del data
        except Exception as exc:
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append((tag, f"{type(exc).__name__}: {exc}"))
            state["cells"][key] = {"status": "failed", "reason": str(exc)[:500]}
        atomic_write_json(run_dir / "cells.json", state)

    print(f"\n{'=' * 86}\nPHASE 2g SUMMARY  "
          f"{sum(v['status'] == 'complete' for v in state['cells'].values())} complete, "
          f"{len(failures)} failed, reproduction gate failed for {gate_fail}")
    for t, why in failures:
        print(f"  {t}: {why}")
    return 1 if failures else (3 if gate_fail else 0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    p.add_argument("--phase1-root", default=str(phase2.DEFAULT_PHASE1_ROOT))
    p.add_argument("--h3-root", default=str(phase2.DEFAULT_OUT_ROOT),
                   help="the committed Phase-2 tree (READ-ONLY)")
    p.add_argument("--cache-root", default=os.environ.get(
        "PHASE1_CACHE", str(scratch / "chronos2" / "phase1_shared_cache")))
    p.add_argument("--window-cache", default=os.environ.get(
        "PHASE1_WINDOWS", str(scratch / "chronos2" / "phase1_shared_windows")))
    p.add_argument("--suite", default="paper14")
    p.add_argument("--allow-window-rebuild", action="store_true")
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--depths", nargs="+", type=int, default=None,
                   help="depth-axis indices (default: every block depth; a subset writes no "
                        "tunnel / budget reading)")
    p.add_argument("--boot-b", type=int, default=BOOT_B)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cpu")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--tables-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
