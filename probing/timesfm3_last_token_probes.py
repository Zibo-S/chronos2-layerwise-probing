"""Per-layer linear probe on TimesFM-3's LAST REAL CONTEXT token.  Q = 9 native quantiles.

One INDEPENDENT ``Linear(1280, 64*9)`` per representation point l in {Emb, L1..L20}:

    (N, 1280) -> Linear(1280, 64*9) -> (N, 576) -> (N, 64, 9) -> (N, 9, 64)
                                                    ^^^^^^^^^
                                       the NATIVE head's flat layout is HORIZON-major
                                       (index = t*Q + q); proven, not assumed, by
                                       timesfm3_last_token.verify_native_head

At L20 this probe and TimesFM-3's own output head are therefore the SAME hypothesis class,
R^1280 -> R^(64x9), which makes L20 a clean endpoint. The trained probe may still differ from
the native head: it is independently optimized, regularized, and fit on finite data.

Objective (train AND report) -- the spec's mean pinball loss over ALL 64 steps and ALL 9 native
quantiles, NOT Chronos-2's sum-over-quantiles convention:

    L = 1/(H*Q) sum_t sum_q rho_{tau_q}(y_t - yhat_{t,q}),    rho_tau(u) = max(tau*u, (tau-1)*u)

``pinball_loss`` implements that formula directly (it is TimesFM-specific only in its
(B, Q, H) shape contract -- the quantile VALUES are arguments). The test module proves it
equals ``probes.mean_pinball_loss`` and ``probes.chronos2_quantile_loss_per_window / (2Q)``, so
the two model lines' metrics are provably one metric up to that documented constant.

Protocol reused from the Chronos-2 / shared-origin pipeline so the lines stay comparable:
``SEED``, the train/validation split protocol, StandardScaler fit on train only, AdamW with
weight decay on the WEIGHT only, full-batch 300 epochs. The selection grid is
``WD_GRID_LAST_TOKEN`` = ``probes.WD_GRID_V2`` extended by 10/30 (see the constant), plus a
separate, never-selected ``WD_NULL_BASELINE`` reference fit.

Why the SAME weight-decay grid transfers despite a different objective SCALE: this objective is
Chronos-2's / (2Q), but AdamW's decay is DECOUPLED -- the update is
-lr * (mhat/(sqrt(vhat)+eps) + wd*p), whose first term Adam normalizes to O(1) whatever the
loss scale, so the data-vs-decay balance at a given wd is unchanged. (A coupled-L2 optimizer,
or plain SGD, WOULD need the grid rescaled by 2Q.) The driver still flags any layer that
selects the grid maximum, which is the observable symptom if that reasoning ever fails. The 5% tunnel entrance reuses ``probing.tunnel.tunnel_start`` -- the project's
existing first-crossing rule -- so the criterion is identical to the Chronos-2 line's.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from probing.config import SEED
from probing.probes import WD_GRID_V2, median_index, validate_quantiles
from probing.timesfm3_last_token import (LAYER_NAMES, MODEL_DIMS, NATIVE_MEDIAN_IDX,
                                         NATIVE_QUANTILES, NUM_LAYERS, NUM_QUANTILES,
                                         safe_sigma)
from probing.tunnel import tunnel_start

# SELECTION GRID: the Chronos-2 shared grid extended upward by two stronger candidates. Defined
# as a superset so the lineage is explicit and probes.WD_GRID_V2 (the Chronos-2 line's grid) stays
# untouched:
#     WD_GRID_V2          = 1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3
#     WD_GRID_LAST_TOKEN  = ... + 10 30
# Why extend at all: this probe is Linear(1280, 576) = 737k parameters fit on 1394 train rows, so
# the optimum can sit above the old ceiling of 3 -- and a run whose selected wd is the grid MAXIMUM
# is a clipped grid, not a converged selection (the driver warns when that happens).
#
# Why the grid STOPS at 30, at the default lr=1e-2. AdamW's decay is DECOUPLED: each step
# multiplies the weight by (1 - lr*wd), so beyond lr*wd ~ 1 the optimizer is no longer doing
# regularized fitting. Measured on synthetic (60 x 1280) features, 30 epochs:
#     wd=10  (lr*wd=0.1)  max|W| 3.1e-2   val 0.071      normal
#     wd=30  (lr*wd=0.3)  max|W| 2.1e-2   val 0.049      normal   <- selection ceiling
#     wd=100 (lr*wd=1)    max|W| 1.0e-2   val 0.072      weight zeroed EVERY step; the fit
#                                                        collapses toward a bias-only predictor
#                                                        (for a pinball loss: the marginal
#                                                        quantile forecast)
#     wd=300 (lr*wd=3)    max|W| 3.8e+7   val 5.5e+8     |1 - lr*wd| > 1 -> the decay term alone
#                                                        amplifies the weight with alternating
#                                                        sign; unstable by construction
# 100 and 300 are therefore NOT hyperparameter candidates: selecting one would report an optimizer
# artifact as a probe. 100 is kept -- separately and explicitly labeled -- as the NULL BASELINE
# below; 300 and larger appear only in the tests, as failure-handling fixtures.
WD_GRID_LAST_TOKEN = tuple(WD_GRID_V2) + (10.0, 30.0)

# NULL BASELINE (diagnostic only, NEVER in the selection grid). At lr=1e-2 this zeroes the weight
# matrix at every step, so the fit keeps only about one Adam step of weight (max|W| ~ lr) and
# predicts essentially from the bias -- i.e. the marginal quantiles of the training targets, using
# no layer information. Its per-layer loss is the "no linearly decodable information" floor every
# probe curve should sit below; the fitted max|W| is recorded with it so the label stays a
# measurement rather than an assumption.
WD_NULL_BASELINE = 100.0

# --------------------------------------------------------------------------- #
# quantile sets -- the cross-model probe objective
# --------------------------------------------------------------------------- #
# The 2026-09-21 benchmark decision froze Q=1 / tau=0.5 as the CROSS-MODEL setting: Chronos-2
# has q1/q9/q21, TiRex is natively Q=1, and TimesFM-3 was hard-wired to its native Q=9. A
# cross-model tunnel comparison has to read one objective, so TimesFM-3 gains q1 here.
#
# Q=9 is NOT removed. It stays the default of this module and is retained as the
# TimesFM-3-specific native-distribution appendix -- it is the only one of the three models
# whose full native quantile vector a linear probe can be asked to reproduce.
QUANTILE_SETS = {
    "q9": NATIVE_QUANTILES,                                   # TimesFM-3's native vector
    "q1": np.array([0.5], dtype=np.float64),                  # cross-model setting
}

#: Which columns of the model's OWN (n, H, 9) native forecast a set scores against. The native
#: head always emits 9 quantiles; scoring it on q1 means selecting its median column, not
#: re-running the model.
NATIVE_COLUMNS = {
    "q9": np.arange(NUM_QUANTILES),
    "q1": np.array([NATIVE_MEDIAN_IDX]),
}


def quantile_set(name: str):
    """``(quantile vector, native column indices, median index WITHIN the set)``.

    Fails loudly on an unknown name -- there is no default quantile set at a call site.
    """
    if name not in QUANTILE_SETS:
        raise ValueError(f"unknown quantile set {name!r}; known: {sorted(QUANTILE_SETS)}")
    q = QUANTILE_SETS[name]
    mid = median_index(q)
    if mid is None:                       # MASE reads the median; a set without 0.5 cannot serve
        raise ValueError(f"quantile set {name!r} has no exact 0.5 level, so no median forecast "
                         f"and therefore no MASE could be computed from it")
    return q, NATIVE_COLUMNS[name], mid


__all__ = ["NATIVE_QUANTILES", "NUM_QUANTILES", "QUANTILE_SETS", "NATIVE_COLUMNS",
           "quantile_set", "WD_GRID_LAST_TOKEN", "WD_NULL_BASELINE",
           "make_probe",
           "reshape_prediction",
           "pinball_loss", "pinball_loss_per_window", "median_pinball_per_window",
           "per_quantile_loss", "fit_last_token_probe", "last_token_layerwise",
           "native_reference", "tunnel_entrance", "tunnel_entrances"]


# --------------------------------------------------------------------------- #
# the probe and its output layout
# --------------------------------------------------------------------------- #

def make_probe(H: int, Q: int = NUM_QUANTILES, device="cpu", seed: int = SEED) -> torch.nn.Linear:
    """THE probe: exactly one ``Linear(1280, H*Q)``, i.e. Linear(1280, 64*9) at H=64, Q=9."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(MODEL_DIMS, H * Q).to(device)
    if (lin.in_features, lin.out_features) != (MODEL_DIMS, H * Q):
        raise RuntimeError(f"probe must be Linear({MODEL_DIMS}, {H}*{Q}={H * Q}), got "
                           f"Linear({lin.in_features}, {lin.out_features})")
    return lin


def reshape_prediction(raw, H: int, Q: int = NUM_QUANTILES) -> torch.Tensor:
    """(B, H*Q) raw probe output -> (B, Q, H), using the NATIVE head's horizon-major layout.

    The native TimesFM-3 head's flat vector is read as (output_patch_len, Q): flat index
    t*Q + q. ``verify_native_head`` proves this by reproducing decode()'s own 9-quantile
    output from the same weights (and proves the transposed layout does NOT reproduce it).
    The (B, Q, H) orientation is what every loss helper in this project consumes.
    """
    raw = torch.as_tensor(raw)
    if raw.ndim != 2 or raw.shape[-1] != H * Q:
        raise RuntimeError(f"raw probe output must be (B, {H}*{Q}={H * Q}), got "
                           f"{tuple(raw.shape)}")
    out = raw.reshape(-1, H, Q).transpose(1, 2)
    if out.shape[1:] != (Q, H):
        raise RuntimeError(f"reshaped prediction is {tuple(out.shape)}, expected (B, {Q}, {H})")
    return out


def _check(pred, target, q) -> None:
    """Shape contract, asserted at every loss evaluation: pred (B, Q, H), target (B, H)."""
    if pred.ndim != 3:
        raise RuntimeError(f"pred must be (B, Q, H), got {tuple(pred.shape)}")
    if pred.shape[-2] != q.numel():
        raise RuntimeError(f"pred has {pred.shape[-2]} quantile rows but {q.numel()} quantile "
                           "levels were given")
    if target.ndim != 2 or pred.shape[0] != target.shape[0] \
            or pred.shape[-1] != target.shape[-1]:
        raise RuntimeError(f"pred {tuple(pred.shape)} is incompatible with target "
                           f"{tuple(target.shape)} -- need (B, Q, H) vs (B, H)")


def pinball_loss_per_window(pred, target, q) -> torch.Tensor:
    """Per-window mean pinball loss: mean over the Q*H terms of ONE window. Returns (B,).

    rho_tau(u) = max(tau*u, (tau-1)*u) with u = y_t - yhat_{t,q}; the SAME raw y_t is compared
    against every predicted quantile at step t. Its ``.mean()`` is the reported scalar, which
    is what lets the series-level cluster bootstrap resample test windows post hoc.
    """
    _check(pred, target, q)
    u = target.unsqueeze(1) - pred                              # (B, Q, H)
    qv = q.view(1, -1, 1)
    return torch.maximum(qv * u, (qv - 1.0) * u).mean(dim=(1, 2))


def pinball_loss(pred, target, q) -> torch.Tensor:
    """The training objective and the headline metric: mean over batch, quantiles, horizon."""
    return pinball_loss_per_window(pred, target, q).mean()


def median_pinball_per_window(pred, target, median_idx: int = NATIVE_MEDIAN_IDX) -> torch.Tensor:
    """Median-only diagnostic: rho_0.5(u) = 0.5*|u|, so this is 0.5 * MAE per window. (B,).

    Reported SEPARATELY. It never selects the tunnel entrance in this experiment -- the
    headline criterion is the full native Q=9 objective.
    """
    if pred.ndim != 3:
        raise RuntimeError(f"pred must be (B, Q, H), got {tuple(pred.shape)}")
    if not 0 <= median_idx < pred.shape[1]:
        raise RuntimeError(f"median index {median_idx} outside 0..{pred.shape[1] - 1}")
    return 0.5 * (target - pred[:, median_idx, :]).abs().mean(dim=1)


def per_quantile_loss(pred, target, q) -> list[float]:
    """mean_{batch, horizon} rho_{tau_q} for each quantile separately (diagnostic)."""
    _check(pred, target, q)
    u = target.unsqueeze(1) - pred
    qv = q.view(1, -1, 1)
    return torch.maximum(qv * u, (qv - 1.0) * u).mean(dim=(0, 2)).detach().cpu().tolist()


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def fit_last_token_probe(Xtr, ytr, q, H, weight_decay, epochs, lr, device,
                         Xval=None, yval=None, history=None, init_seed: int = SEED,
                         batch_size: int = 0, model_for_grad_check=None) -> torch.nn.Linear:
    """Fit ONE Linear(1280, H*Q) with the Q=9 mean pinball objective.

    Same optimizer convention as the rest of the project: AdamW, weight decay on the WEIGHT
    only (the pinball-optimal bias IS the target's quantile vector, so decaying it would shrink
    every predicted quantile toward 0), re-seeded per call so every layer and every weight-decay
    candidate starts from the same init. ``batch_size=0`` = full batch (the Chronos-2 protocol).
    """
    Q = q.numel()
    lin = make_probe(H, Q, device, init_seed)
    opt = torch.optim.AdamW(
        [{"params": [lin.weight], "weight_decay": weight_decay},
         {"params": [lin.bias], "weight_decay": 0.0}], lr=lr)
    n = Xtr.shape[0]
    full = batch_size <= 0 or batch_size >= n
    gen = torch.Generator(device="cpu").manual_seed(init_seed)

    def loss_of(X, y):
        return pinball_loss(reshape_prediction(lin(X), H, Q), y, q)

    def ev(X, y):
        with torch.no_grad():
            return float(loss_of(X, y).item())

    lin.train()
    for _ in range(epochs):
        if full:
            loss = loss_of(Xtr, ytr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if model_for_grad_check is not None:
                from probing.timesfm3_last_token import assert_no_backbone_grads
                assert_no_backbone_grads(model_for_grad_check)
            opt.step()
        else:
            perm = torch.randperm(n, generator=gen).to(Xtr.device)
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                loss = loss_of(Xtr[b], ytr[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if history is not None:
            history["train"].append(ev(Xtr, ytr))
            if Xval is not None:
                history["val"].append(ev(Xval, yval))
    lin.eval()
    return lin


def _carve(n_train: int, seed: int = SEED):
    """The project's 80/20 carve of TRAIN WINDOWS (never of rows built from them)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    n_val = max(1, int(0.2 * n_train))
    return np.sort(perm[:n_val]), np.sort(perm[n_val:])          # (val_windows, train_windows)


def last_token_layerwise(train_feats, train_targets, train_valid,
                         test_feats, test_targets, test_valid, *, H,
                         val_feats=None, val_targets=None, val_valid=None,
                         quantiles=NATIVE_QUANTILES, epochs: int = 300, lr: float = 1e-2,
                         wd_grid=WD_GRID_LAST_TOKEN, weight_decay: float = 1e-3,
                         null_wd: float = WD_NULL_BASELINE, device=None,
                         batch_size: int = 0, layers=None, collect_history: bool = False,
                         verbose: bool = True, model_for_grad_check=None):
    """One independent probe per representation point. Returns ({layer: test Q9 loss}, diag).

    TWO validation protocols, chosen by whether the dataset HAS a dedicated validation split:

    EXPLICIT (``val_feats`` given) -- the rolling-origin sets (id_data.ROLLING_SETS and the
    PT-OOD rolling builder), where val is a dedicated LATER forecast origin per series. Per
    layer, the StandardScaler AND the Linear are fit on the FULL train split (validation never
    touches the scaler or the weights), each weight-decay candidate is scored on val, and the
    chosen-wd full-train model is KEPT -- no refit, because it is already trained on all of
    train. This is exactly ``probes.fit_quantile_probe_explicit_val``'s contract, so the
    Chronos-2 and TimesFM-3 lines select weight decay and the tunnel layer the same way on the
    same windows.

    CARVE (``val_feats`` omitted) -- the legacy auto-split sets (e.g. extended_v1), where there
    is no dedicated val split: select on the seed-based 80/20 carve of TRAIN WINDOWS (scaler and
    probe fit on the 80% only), then REFIT scaler + probe on all valid train windows.

    Either way ``diag["val_loss"]`` is the curve the 5% tunnel entrance is computed from and it
    never sees test; ``diag["val_source"]`` records which protocol ran. Test scoring is
    identical: Q=9 loss (+ per-window for the bootstrap), median-only loss, the median
    prediction (for MASE, un-transformed by the driver), per-quantile losses.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q_np = validate_quantiles(quantiles)
    Q = len(q_np)
    # The median's position depends on the quantile vector actually in use (index 4 of the
    # native 9, index 0 of the Q=1 set). Reading it from q_np is what makes --quantile-set q1
    # score the right column instead of indexing past the end of a (B, 1, H) prediction.
    med_idx = median_index(q_np)
    q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
    train_targets = np.asarray(train_targets, np.float32)
    test_targets = np.asarray(test_targets, np.float32)
    if train_targets.ndim != 2 or train_targets.shape[1] != H:
        raise ValueError(f"train targets must be (n, {H}), got {train_targets.shape}")
    if test_targets.ndim != 2 or test_targets.shape[1] != H:
        raise ValueError(f"test targets must be (n, {H}), got {test_targets.shape}")
    train_valid = np.asarray(train_valid, bool).reshape(-1)
    test_valid = np.asarray(test_valid, bool).reshape(-1)
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(set(layers))

    all_tr = np.flatnonzero(train_valid)
    te_rows = np.flatnonzero(test_valid)
    explicit = val_feats is not None
    if explicit:
        if val_targets is None:
            raise ValueError("val_feats was given without val_targets")
        val_targets = np.asarray(val_targets, np.float32)
        if val_targets.ndim != 2 or val_targets.shape[1] != H:
            raise ValueError(f"val targets must be (n, {H}), got {val_targets.shape}")
        val_valid = (np.ones(len(val_targets), bool) if val_valid is None
                     else np.asarray(val_valid, bool).reshape(-1))
        sel_tr = all_tr                      # selection fits on the FULL train split
        va_rows = np.flatnonzero(val_valid)
        val_source = "explicit_temporal_split"
    else:
        va_w, tr_w = _carve(len(train_targets))
        sel_tr = tr_w[train_valid[tr_w]]
        va_rows = va_w[train_valid[va_w]]
        val_feats, val_targets = train_feats, train_targets
        val_source = "carve_80_20"
    for name, rows in (("train(selection)", sel_tr), ("validation", va_rows),
                       ("train(final)", all_tr), ("test", te_rows)):
        if rows.size == 0:
            raise RuntimeError(f"no valid window left in the {name} split")

    yte = torch.as_tensor(test_targets[te_rows], device=device)
    ytr_all = torch.as_tensor(train_targets[all_tr], device=device)
    ytr_sel = torch.as_tensor(train_targets[sel_tr], device=device)
    yva_sel = torch.as_tensor(val_targets[va_rows], device=device)

    out: dict[int, float] = {}
    diag: dict = {"wd": {}, "selection": {}, "val_loss": {}, "wd_at_grid_edge": {},
                  "nonfinite_wd_candidates": {}, "null_baseline": {},
                  "history": {}, "train_loss": {}, "test_q9": {}, "test_q9_window": {},
                  "test_median_loss": {}, "test_median_window": {}, "test_median_pred": {},
                  "test_per_quantile": {}, "n_train_rows": {}, "n_val_rows": int(va_rows.size),
                  "n_test_rows": int(te_rows.size), "test_rows": te_rows,
                  "val_source": val_source, "val_rows": va_rows,
                  "selection_train_windows": sel_tr, "validation_windows": va_rows,
                  "layers": layers, "layer_names": [LAYER_NAMES[i] for i in layers],
                  "quantiles": q_np.tolist(), "num_quantiles": Q}

    for i in layers:
        F_tr, F_te, F_va = train_feats[i], test_feats[i], val_feats[i]
        for nm, F in (("train", F_tr), ("test", F_te), ("val", F_va)):
            F = np.asarray(F)
            if F.ndim != 2 or F.shape[1] != MODEL_DIMS:
                raise RuntimeError(f"layer {i} {nm} features are {F.shape}; this experiment "
                                   f"needs (n, {MODEL_DIMS}) last-token states")

        # ---- 1. weight-decay selection (scaler fit on the SELECTION train rows only) ----
        Xs = np.asarray(F_tr[sel_tr], np.float32)
        sc_sel = StandardScaler().fit(Xs)
        Xtr_s = torch.as_tensor(sc_sel.transform(Xs), dtype=torch.float32, device=device)
        Xva_s = torch.as_tensor(sc_sel.transform(np.asarray(F_va[va_rows], np.float32)),
                                dtype=torch.float32, device=device)
        best_m = None
        if wd_grid is None:
            wd, sel = weight_decay, None
            m = fit_last_token_probe(Xtr_s, ytr_sel, q, H, wd, epochs, lr, device,
                                     batch_size=batch_size)
            with torch.no_grad():
                val = float(pinball_loss(reshape_prediction(m(Xva_s), H, Q), yva_sel, q).item())
            best_m = m
        else:
            best, wd, sel, n_bad = float("inf"), wd_grid[0], {}, 0
            for cand in wd_grid:
                m = fit_last_token_probe(Xtr_s, ytr_sel, q, H, cand, epochs, lr, device,
                                         batch_size=batch_size)
                with torch.no_grad():
                    v = float(pinball_loss(reshape_prediction(m(Xva_s), H, Q),
                                           yva_sel, q).item())
                sel[float(cand)] = v
                if not np.isfinite(v):
                    # expected for lr*wd > 2 (see WD_GRID_LAST_TOKEN): decoupled decay alone
                    # amplifies the weight each step. Recorded, never selected, never silently
                    # turned into a finite number.
                    n_bad += 1
                    continue
                if v < best:
                    best, wd, best_m = v, cand, m
            if best_m is None:
                raise RuntimeError(
                    f"layer {LAYER_NAMES[i]}: EVERY weight-decay candidate gave a non-finite "
                    f"validation loss {sel}. At lr={lr} a candidate with lr*wd > 2 makes AdamW's "
                    "decoupled decay divergent -- shrink --wd-grid or --probe-lr; nothing here "
                    "will invent a usable probe.")
            diag["nonfinite_wd_candidates"][i] = n_bad
            val = best
        diag["wd"][i] = float(wd)
        diag["val_loss"][i] = float(val)
        diag["selection"][i] = (None if sel is None
                                else {"val_loss_by_wd": sel, "chosen_wd": float(wd)})
        diag["wd_at_grid_edge"][i] = (False if wd_grid is None
                                      else bool(float(wd) == float(max(wd_grid))))
        if collect_history:
            hist = {"train": [], "val": []}
            fit_last_token_probe(Xtr_s, ytr_sel, q, H, wd, epochs, lr, device, Xval=Xva_s,
                                 yval=yva_sel, history=hist, batch_size=batch_size)
            diag["history"][i] = hist

        # ---- 1b. NULL BASELINE: fitted, reported, and EXCLUDED from selection ----
        null_m = None
        if null_wd and null_wd > 0:
            if float(null_wd) in {float(c) for c in (wd_grid or ())}:
                raise ValueError(
                    f"null_wd={null_wd} is also a SELECTION candidate -- the null baseline must "
                    "stay outside the grid, otherwise an optimizer artifact can win the "
                    "hyperparameter search")
            null_m = fit_last_token_probe(Xtr_s, ytr_sel, q, H, null_wd, epochs, lr, device,
                                          batch_size=batch_size)
            with torch.no_grad():
                nv = float(pinball_loss(reshape_prediction(null_m(Xva_s), H, Q), yva_sel,
                                        q).item())
            diag["null_baseline"][i] = {
                "wd": float(null_wd), "val_loss": nv, "selected": False,
                "max_abs_weight": float(null_m.weight.detach().abs().max()),
                "note": "extreme decay: weight zeroed each step, so this is ~ a bias-only "
                        "(marginal-quantile) fit -- the no-information floor, never a candidate"}
        del Xva_s

        # ---- 2. the final model ----
        if explicit:
            # the selection fit ALREADY used the full train split, so the chosen-wd model IS
            # the final model (probes.fit_quantile_probe_explicit_val's "no refit" contract)
            sc, lin, Xtr = sc_sel, best_m, Xtr_s
            if model_for_grad_check is not None:
                from probing.timesfm3_last_token import assert_no_backbone_grads
                assert_no_backbone_grads(model_for_grad_check)
        else:
            del Xtr_s
            Xa = np.asarray(F_tr[all_tr], np.float32)
            sc = StandardScaler().fit(Xa)
            Xtr = torch.as_tensor(sc.transform(Xa), dtype=torch.float32, device=device)
            lin = fit_last_token_probe(Xtr, ytr_all, q, H, wd, epochs, lr, device,
                                       batch_size=batch_size,
                                       model_for_grad_check=model_for_grad_check)
        diag["n_train_rows"][i] = int(Xtr.shape[0])
        with torch.no_grad():
            diag["train_loss"][i] = float(
                pinball_loss(reshape_prediction(lin(Xtr), H, Q), ytr_all, q).item())
        del Xtr

        # ---- 3. score on test ----
        with torch.no_grad():
            Xte = torch.as_tensor(sc.transform(np.asarray(F_te[te_rows], np.float32)),
                                  dtype=torch.float32, device=device)
            pred = reshape_prediction(lin(Xte), H, Q)                 # (n, Q, H)
            pw = pinball_loss_per_window(pred, yte, q)
            out[i] = float(pw.mean().item())
            diag["test_q9"][i] = out[i]
            diag["test_q9_window"][i] = pw.cpu().numpy().astype(np.float64)
            mpw = median_pinball_per_window(pred, yte, med_idx)
            diag["test_median_loss"][i] = float(mpw.mean().item())
            diag["test_median_window"][i] = mpw.cpu().numpy().astype(np.float64)
            diag["test_median_pred"][i] = pred[:, med_idx, :].cpu().numpy(
                ).astype(np.float32)
            diag["test_per_quantile"][i] = per_quantile_loss(pred, yte, q)
            if null_m is not None:
                # same scaler the null was TRAINED with (sc_sel); in the explicit-val protocol
                # that IS sc, so this is one transform, not two
                Xte_n = (Xte if sc_sel is sc else
                         torch.as_tensor(sc_sel.transform(np.asarray(F_te[te_rows], np.float32)),
                                         dtype=torch.float32, device=device))
                npred = reshape_prediction(null_m(Xte_n), H, Q)
                npw = pinball_loss_per_window(npred, yte, q)
                diag["null_baseline"][i].update(
                    test_q9=float(npw.mean().item()),
                    test_median_loss=float(median_pinball_per_window(
                        npred, yte, med_idx).mean().item()))
                del Xte_n, npred
            del Xte, pred
        if verbose:
            nb = diag["null_baseline"].get(i)
            print(f"    [{LAYER_NAMES[i]:>3}] wd={wd:<6g} rows={diag['n_train_rows'][i]:>5}  "
                  f"train={diag['train_loss'][i]:.5f}  val={val:.5f}  "
                  f"test(Q9)={out[i]:.5f}  test(median)={diag['test_median_loss'][i]:.5f}"
                  + (f"  null(wd={nb['wd']:g})={nb['test_q9']:.5f}" if nb else ""),
                  flush=True)
    return out, diag


# --------------------------------------------------------------------------- #
# native baseline, scored on the probes' own axis
# --------------------------------------------------------------------------- #

def native_reference(native_raw, mu, sd, trend, targets, valid, quantiles=NATIVE_QUANTILES,
                     device=None, native_columns=None) -> dict:
    """The native TimesFM-3 forecast, normalized into the probe target space and scored there.

    decode() returns RAW units; dividing out the SAME (trend, mu, sigma) the probe targets use
    puts all 9 native quantiles on the probes' axis, so the native Q=9 loss is directly
    comparable to every layer's. Returns the losses, the per-window vectors (for the cluster
    bootstrap) and the raw median forecast (for MASE).
    """
    device = device or "cpu"
    q_np = validate_quantiles(quantiles)
    med_idx = median_index(q_np)
    # The native head always emits its 9 quantiles. Under a smaller probe quantile set we score
    # the SAME forecast on the same levels the probe is scored on, by selecting those columns --
    # never by re-running or re-fitting anything.
    if native_columns is not None:
        native_raw = np.asarray(native_raw)[:, :, np.asarray(native_columns, int)]
    q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
    rows = np.flatnonzero(np.asarray(valid, bool).reshape(-1))
    nat = np.asarray(native_raw, np.float64)[rows]                       # (n, H, Q)
    if nat.ndim != 3 or nat.shape[-1] != len(q_np):
        raise ValueError(f"native forecasts must be (n, H, {len(q_np)}), got {nat.shape}")
    mu = np.asarray(mu, np.float64).reshape(-1)[rows]
    sd = np.asarray(sd, np.float64).reshape(-1)[rows]
    tr = np.asarray(trend, np.float64)[rows]                             # (n, H)
    z = (nat - tr[:, :, None] - mu[:, None, None]) / safe_sigma(sd)[:, None, None]
    pred = torch.as_tensor(np.ascontiguousarray(z.transpose(0, 2, 1)), dtype=torch.float32,
                           device=device)                                # (n, Q, H)
    y = torch.as_tensor(np.asarray(targets, np.float32)[rows], device=device)
    with torch.no_grad():
        pw = pinball_loss_per_window(pred, y, q).cpu().numpy().astype(np.float64)
        mpw = median_pinball_per_window(pred, y, med_idx).cpu().numpy().astype(
            np.float64)
        per_q = per_quantile_loss(pred, y, q)
    return {"q9_loss": float(pw.mean()), "q9_window": pw,
            "median_loss": float(mpw.mean()), "median_window": mpw,
            "per_quantile": per_q, "median_raw": nat[:, :, med_idx],
            "rows": rows}


# --------------------------------------------------------------------------- #
# tunnel entrance (VALIDATION only)
# --------------------------------------------------------------------------- #

def tunnel_entrance(val_losses, layers, tol: float = 0.05) -> dict:
    """l_tun = min { l : L_val(l) <= (1+tol) * L_val(L20) }, computed on VALIDATION only.

    Delegates the first-crossing scan to ``probing.tunnel.tunnel_start`` (the project's
    existing 5% rule, shared with the Chronos-2 line) and maps the returned POSITION in the
    supplied curve back to a representation-point index. The reference is the LAST element of
    the curve, so the caller must pass layers in increasing order ending at L20 -- asserted.
    """
    v = np.asarray(val_losses, dtype=np.float64)
    layers = [int(x) for x in layers]
    if v.ndim != 1 or v.size != len(layers):
        raise ValueError(f"val curve {v.shape} does not match {len(layers)} layers")
    if layers != sorted(layers):
        raise ValueError(f"layers must be increasing, got {layers}")
    if layers[-1] != NUM_LAYERS - 1:
        raise ValueError(f"the tunnel reference must be the last layer L{NUM_LAYERS - 1}; the "
                         f"curve ends at L{layers[-1]}")
    pos = int(tunnel_start(v, tol))
    return {"tol": float(tol), "position_in_curve": pos, "layer": layers[pos],
            "layer_name": LAYER_NAMES[layers[pos]],
            "val_loss": float(v[pos]), "reference_layer": layers[-1],
            "reference_val_loss": float(v[-1]),
            "threshold": float((1.0 + tol) * v[-1]),
            "ratio_at_entrance": float(v[pos] / v[-1]),
            "max_excursion_after_entrance": float(np.max(v[pos:] / v[-1] - 1.0))}


def tunnel_entrances(val_losses, layers, tols=(0.01, 0.02, 0.05, 0.10)) -> dict:
    """The 5% headline plus every other tolerance, so a 2% criterion needs no re-extraction."""
    return {f"{t:g}": tunnel_entrance(val_losses, layers, t) for t in tols}
