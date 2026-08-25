"""Pluggable probe methods — the extension point for new probing techniques.

A *probe* answers: "given the per-layer representations of a dataset, how much
task-relevant / structural information does each layer carry?" The reference probe is a
linear (logistic-regression) classifier; collaborators add alternative probes
(effective-rank, entropy, epiplexity, ...) here WITHOUT touching extraction or the
experiment loops.

--------------------------------------------------------------------------------------
PROBE CONTRACT
--------------------------------------------------------------------------------------
    probe(train_feats, train_labels, test_feats, test_labels) -> dict[int, float]

    train_feats / test_feats : {layer_idx: np.ndarray of shape (n_samples, d)}
                               d = n_channels * 768, produced by extract_window_features(...)
    train_labels / test_labels : 1-D array of labels aligned with the feature rows
    returns : {layer_idx: float}  — one scalar score per layer
              (higher = "more information", by convention)

- SUPERVISED probes (e.g. the linear probe) fit on the train split and evaluate on test.
- UNSUPERVISED measures (effective-rank, entropy) may ignore the labels and/or the train
  split and compute directly on `test_feats`. They still receive all four arguments so
  every probe has one uniform signature and slots into the same driver loop.
- EXCEPTION: `quantile_probe` requires 2-D (n, H) trajectory labels (id_data's Y_*_traj),
  not the 1-D scalar labels — it raises ValueError on 1-D input. A driver looping PROBES
  generically must special-case which label array it hands each probe.

To register a new probe: implement it below and add it to the PROBES dict at the bottom.
See README > "Adding a new probe".
--------------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE
from probing.extraction import fit_layerwise_probes
from probing.heads import build_head, head_param_count, wd_param_groups, NATIVE_D_FF


def score_layerwise_correctness(probes, features, y_true):
    """Returns dict {layer_idx: float32 correctness array of shape (n_test,)}."""
    y_true = np.asarray(y_true)
    out = {}
    for i in range(NUM_LAYERS):
        Xs = probes[i]["scaler"].transform(features[i])
        y_pred = probes[i]["clf"].predict(Xs)
        out[i] = (y_pred == y_true).astype(np.float32)
    return out


# ============================ reference probe (linear) ============================ #

def linear_probe(train_feats, train_labels, test_feats, test_labels):
    """Reference probe: per-layer StandardScaler + LogisticRegression, test accuracy.

    Numerically identical to the original pipeline: it simply composes the unchanged
    ``fit_layerwise_probes`` (fit on train only) and ``score_layerwise_correctness``,
    then reduces each layer's per-sample correctness to a mean accuracy.

    Returns {layer_idx: test_accuracy}.
    """
    probes = fit_layerwise_probes(train_feats, train_labels)
    correct = score_layerwise_correctness(probes, test_feats, test_labels)
    # reduce in float64 (matches how the pipeline's bootstrap_ci computes point accuracy)
    return {i: float(np.mean(correct[i], dtype=np.float64)) for i in range(NUM_LAYERS)}


# ===================== ID forecasting probes (Phase 0, linear) ==================== #
# These consume windowed forecasting features (see probing.id_data / extract_window_features)
# whose labels are the normalized H-step future mean. Both stay strictly LINEAR so the
# tunnel-effect diagnostic (linear readout) is preserved.

def ridge_regression_probe(train_feats, train_labels, test_feats, test_labels):
    """Per-layer linear Ridge regression on the (continuous) normalized future-mean label.

    Alpha is chosen per layer from a small grid on a held-out validation split carved from
    train (fixed-seed random 80/20). Score = R^2 on the test split. R^2 can be negative
    (worse than predicting the mean); that is reported as-is, never clipped.
    """
    y_tr = np.asarray(train_labels, dtype=np.float64)
    y_te = np.asarray(test_labels, dtype=np.float64)
    alphas = [0.1, 1.0, 10.0, 100.0]
    rng = np.random.default_rng(SEED)
    n = len(y_tr)
    perm = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    va, tr = perm[:n_val], perm[n_val:]

    out = {}
    for i in range(NUM_LAYERS):
        Xtr = train_feats[i]
        # alpha selection on the carved validation split
        sc = StandardScaler().fit(Xtr[tr])
        best_a, best_r2 = alphas[0], -np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(sc.transform(Xtr[tr]), y_tr[tr])
            r2v = r2_score(y_tr[va], m.predict(sc.transform(Xtr[va])))
            if r2v > best_r2:
                best_r2, best_a = r2v, a
        # refit on FULL train with the selected alpha, score on test
        sc_full = StandardScaler().fit(Xtr)
        m = Ridge(alpha=best_a).fit(sc_full.transform(Xtr), y_tr)
        out[i] = float(r2_score(y_te, m.predict(sc_full.transform(test_feats[i]))))
    return out


def binned_future_probe(train_feats, train_labels, test_feats, test_labels, n_bins=5):
    """Per-layer logistic regression on the future-mean label discretized into K quantile bins.

    Bin edges are computed on TRAIN LABELS ONLY. Score = accuracy per layer — an accuracy-scale
    curve directly comparable to the UEA classification curves, and robust to any last-layer
    triviality of raw regression. This is the PRIMARY tunnel-signature readout for the ID
    condition; ridge R^2 is secondary.
    """
    y_tr = np.asarray(train_labels, dtype=np.float64)
    y_te = np.asarray(test_labels, dtype=np.float64)
    edges = np.quantile(y_tr, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])  # K-1 interior edges
    ytr_b = np.digitize(y_tr, edges)
    yte_b = np.digitize(y_te, edges)

    out = {}
    for i in range(NUM_LAYERS):
        scaler = StandardScaler().fit(train_feats[i])
        clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(
            scaler.transform(train_feats[i]), ytr_b)
        pred = clf.predict(scaler.transform(test_feats[i]))
        out[i] = float(np.mean((pred == yte_b), dtype=np.float64))
    return out


# =============== Chronos-2-native quantile probe (forecasting currency) ============ #
# Predicts all 21 Chronos-2 quantiles at each of H horizon steps from each layer's
# pooled 768-d state, trained AND scored with Chronos-2's own quantile (pinball) loss.
# Labels are the arcsinh future trajectories (id_data.Y_*_traj). Score = test loss;
# LOWER is better, so the tunnel signature is the argmin dip (not a peak).

# Verified from amazon/chronos-2 config.json (chronos_config.quantiles).
CHRONOS2_QUANTILES = np.array(
    [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], dtype=np.float32)

# Named quantile configurations for the probe-capacity ablation. The probe head is
# Linear(d, Q*H), so Q directly scales the readout's parameter count (q1 = ~21x fewer
# params than q21 at the same d/H). "q21" IS CHRONOS2_QUANTILES — selecting it reproduces
# the committed numbers exactly (same default code path, seed 0, deterministic refit).
QUANTILE_SETS = {
    "q1":  np.array([0.5], dtype=np.float32),
    "q9":  np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
    "q21": CHRONOS2_QUANTILES,
}

# ------------------------------------------------------------------------------------------------ #
# Shared forecasting-probe protocol (q1/q9 rerun) — ONE source of truth imported by every fslot
# fitting driver (run_ptood_probing_ftok / run_fslot_transfer / run_ft_specialization /
# run_task_shift) so the grid + protocol identity can never drift between experiments.
#
# WD_GRID_V2 widens the legacy narrow grid (1e-5..1e-1): the q9 diagnostic showed many layers
# selecting the old maximum 1e-1 while validation loss was STILL improving (grid clipped), so three
# stronger candidates are added. PROBE_PROTOCOL_VERSION stamps every new result's path + metadata so
# a legacy narrow-grid q9 result can NEVER satisfy the skip logic of a new wide-grid q9 run — a fresh
# probe-protocol identity, disjoint from the committed q9 numbers.
WD_GRID_V2 = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0)
PROBE_PROTOCOL_VERSION = "v2"     # shared-linear fslot readout, wide WD grid


def validate_quantiles(quantiles):
    """Validate a quantile vector (non-empty, all strictly inside (0,1), strictly increasing
    — which also rejects duplicates) and return it as a float32 array."""
    q = np.asarray(quantiles, dtype=np.float32)
    assert q.ndim == 1 and len(q) > 0, f"quantiles must be a non-empty 1-D vector, got shape {q.shape}"
    assert np.all((q > 0.0) & (q < 1.0)), f"quantile levels must lie strictly in (0, 1), got {q.tolist()}"
    assert np.all(np.diff(q) > 0), f"quantile levels must be strictly increasing, got {q.tolist()}"
    return q


def median_index(quantiles):
    """Index of the EXACT 0.5 level in `quantiles`, or None when absent. Median-based
    metrics (MASE) must be skipped when this is None — never substitute a neighbor."""
    idx = np.flatnonzero(np.isclose(np.asarray(quantiles, dtype=np.float64), 0.5))
    return int(idx[0]) if len(idx) else None


def _check_pred_shape(pred, target, q):
    """Prediction-layout contract, asserted at every loss evaluation: (B, Q, H) — quantiles
    on dim -2, horizon on dim -1 (Chronos-2's own layout, 'b q (n p)')."""
    assert pred.ndim == 3, f"pred must be (B, Q, H), got {tuple(pred.shape)}"
    assert pred.shape[-2] == q.numel(), (
        f"pred has {pred.shape[-2]} quantile rows but {q.numel()} quantile levels were given")
    assert target.ndim == 2 and pred.shape[-1] == target.shape[-1], (
        f"pred horizon {pred.shape[-1]} != target horizon {tuple(target.shape)}")


def chronos2_quantile_loss(pred, target, q):
    """Chronos-2's quantile (pinball) loss, formula + reduction verbatim from
    chronos/chronos2/model.py:551,564.

        pred   : (B, Q, H) predicted quantiles
        target : (B, H)    true trajectory   (broadcast to (B, 1, H))
        q      : (Q,)      quantile levels    (broadcast to (1, Q, 1))

    loss = 2*|(y - q̂)(1[y<=q̂] - τ)|, reduced mean(horizon) -> sum(quantiles) -> mean(batch).

    NOTE: sum over quantiles (Chronos-2's convention) makes raw values grow with Q, so
    losses are NOT comparable across quantile sets — use mean_pinball_loss (= this / (2Q))
    for cross-set comparisons. This function stays the training objective for every set so
    the q21 configuration reproduces the committed numbers exactly.
    """
    _check_pred_shape(pred, target, q)
    target = target.unsqueeze(1)                                       # (B, 1, H)
    qv = q.view(1, -1, 1)                                              # (1, Q, 1)
    ql = 2.0 * torch.abs((target - pred) * ((target <= pred).to(pred.dtype) - qv))
    return ql.mean(dim=-1).sum(dim=-1).mean()


def chronos2_quantile_loss_per_window(pred, target, q):
    """Per-window Chronos-2 quantile loss: identical formula and mean(horizon) -> sum(quantiles)
    reduction as ``chronos2_quantile_loss``, but WITHOUT the final batch mean. Returns (B,);
    its ``.mean()`` is the reported scalar loss (same op chain), which is what lets the
    series-level cluster bootstrap resample test windows post hoc without refitting anything."""
    _check_pred_shape(pred, target, q)
    target = target.unsqueeze(1)                                       # (B, 1, H)
    qv = q.view(1, -1, 1)                                              # (1, Q, 1)
    ql = 2.0 * torch.abs((target - pred) * ((target <= pred).to(pred.dtype) - qv))
    return ql.mean(dim=-1).sum(dim=-1)


def mean_pinball_loss(pred, target, q):
    """Plain pinball loss, mean over batch x time steps x quantiles — the metric that IS
    comparable across quantile sets (q1/q9/q21), unlike Chronos-2's sum-over-quantiles
    convention. Elementwise max(τe, (τ-1)e) with e = y - q̂ equals half the Chronos-2
    elementwise term, so this = chronos2_quantile_loss / (2*Q) up to reduction-order
    rounding. Evaluation only — never the training objective (rescaling the objective
    would shift the AdamW weight-decay balance and move the committed q21 numbers).
    At q = [0.5] this equals 0.5 * MAE."""
    _check_pred_shape(pred, target, q)
    error = target.unsqueeze(1) - pred                                 # (B, Q, H)
    qv = q.view(1, -1, 1)                                              # (1, Q, 1)
    return torch.maximum(qv * error, (qv - 1.0) * error).mean()


def _fit_quantile_linear(Xtr, ytr, q, weight_decay, epochs, lr, device,
                         Xval=None, yval=None, history=None, init_seed=SEED):
    """Fit one strictly-linear map (d -> Q*H) with Chronos-2 loss; return the trained module.
    Re-seeded each call so every layer / weight_decay candidate starts from the same init.
    `init_seed` selects that init (the ONLY randomness in the full-batch deterministic fit) —
    it is what makes "3 independent probe runs" independent; default SEED keeps every existing
    call byte-identical.

    AdamW with decay on the WEIGHT only: the pinball-optimal bias is the target's quantile
    vector itself, so decaying the bias would shrink every predicted quantile toward 0 (and
    plain Adam's weight_decay is warped by the adaptive scaling, making grid values
    incomparable across layers).

    If `history` (a dict with "train"/"val" lists) is passed, per-epoch train loss (and val
    loss when Xval/yval are given) is appended BEFORE each update, plus once more after the
    loop -> length epochs+1 (init ... converged), for the training-curve diagnostic.
    history=None leaves the original loop exactly unchanged (behavior-preserving)."""
    Q, H = len(q), ytr.shape[1]
    torch.manual_seed(init_seed)
    lin = torch.nn.Linear(Xtr.shape[1], Q * H).to(device)
    opt = torch.optim.AdamW(
        [{"params": [lin.weight], "weight_decay": weight_decay},
         {"params": [lin.bias],   "weight_decay": 0.0}],
        lr=lr)
    lin.train()
    for _ in range(epochs):
        loss = chronos2_quantile_loss(lin(Xtr).view(-1, Q, H), ytr, q)
        if history is not None:
            history["train"].append(loss.item())
            if Xval is not None:
                with torch.no_grad():
                    history["val"].append(
                        chronos2_quantile_loss(lin(Xval).view(-1, Q, H), yval, q).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if history is not None:                       # final converged point (epoch = epochs)
        with torch.no_grad():
            history["train"].append(chronos2_quantile_loss(lin(Xtr).view(-1, Q, H), ytr, q).item())
            if Xval is not None:
                history["val"].append(chronos2_quantile_loss(lin(Xval).view(-1, Q, H), yval, q).item())
    lin.eval()
    return lin


def quantile_probe(train_feats, train_labels, test_feats, test_labels,
                   quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                   weight_decay=1e-3, wd_grid=None, device=None, collect_history=False,
                   collect_test_median=False, collect_test_window_loss=False):
    """Per-layer LINEAR quantile probe, trained + scored with Chronos-2's own quantile loss.

    train_labels / test_labels : (n, H) arcsinh future trajectories (id_data.Y_*_traj) -- NOT
                                 the scalar `y`. Raises if handed 1-D labels.
    Returns {layer: test_loss}. LOWER = better (tunnel signature = argmin over layers).

    weight_decay : fixed L2 used when `wd_grid is None` (debug / fast iteration).
    wd_grid      : if given (e.g. (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)), select per layer on a
                   seed-based 80/20 train/val carve (lowest val Chronos-2 loss), then refit
                   on FULL train. The carve matches ridge_regression_probe; test is untouched.
    collect_history : if True, also return a per-layer diagnostics dict
        {"wd":       {layer: selected_wd},
         "selection": {layer: {"val_loss_by_wd": {wd: val}, "chosen_wd": wd} | None},
         "history":   {layer: {"train": [...], "val": [...]}}}
      history/val come from a carve-fit (train on the 80%, val = held-out 20%) with the
      selected wd -- for the training-curve figures. The reported test loss in `out` is
      UNCHANGED (still the full-train refit). Returns (out, diag) when True, else out.
    collect_test_median : if True, diag also carries
        {"test_median": {layer: np.float32 (n_test, H)}}
      = the q=0.5 row of the FINAL (full-train refit) probe's test predictions, still in the
      arcsinh label space -- the driver un-transforms it for the MASE comparison against
      native Chronos-2. Returns (out, diag) whenever any collect_* flag is True.
    collect_test_window_loss : if True, diag also carries
        {"test_window_loss": {layer: np.float64 (n_test,)}}
      = the per-window (unreduced-over-batch) test loss of the same final refit; its mean
      equals out[layer]. Consumed by the series-level cluster bootstrap (run_bootstrap).

    Whenever a diag is returned it also carries {"test_mean_pinball": {layer: float}} — the
    quantile-count-normalized test pinball loss (mean over batch/steps/quantiles), the
    metric comparable ACROSS quantile sets (out[layer] uses Chronos-2's sum-over-quantiles
    convention and is only comparable within one set).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    Yte = np.asarray(test_labels, dtype=np.float32)
    if Ytr.ndim != 2:
        raise ValueError(f"quantile_probe needs (n, H) trajectory labels, got shape {Ytr.shape} "
                         "-- pass Y_*_traj, not the scalar y")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    Q, H = len(quantiles), Ytr.shape[1]
    ytr = torch.as_tensor(Ytr, device=device)
    yte = torch.as_tensor(Yte, device=device)

    # seed-based 80/20 carve of TRAIN for weight_decay selection (mirrors ridge probe)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(Ytr.shape[0])
    n_val = max(1, int(0.2 * Ytr.shape[0]))
    va_np, tr_np = perm[:n_val], perm[n_val:]
    va = torch.as_tensor(va_np, dtype=torch.long, device=device)
    tr = torch.as_tensor(tr_np, dtype=torch.long, device=device)

    out = {}
    diag = ({"wd": {}, "selection": {}, "history": {}, "test_median": {}, "test_window_loss": {},
             "test_mean_pinball": {}}
            if (collect_history or collect_test_median or collect_test_window_loss) else None)
    q_mid = median_index(quantiles)                       # exact 0.5 row, or None
    if collect_test_median and q_mid is None:
        raise ValueError(
            f"collect_test_median needs the 0.5 level in the quantile set, got "
            f"{quantiles.tolist()} — median/MASE metrics are unavailable for this set; "
            "skip them instead of substituting a neighboring quantile")
    for i in range(NUM_LAYERS):
        if wd_grid is None:
            wd = weight_decay
            if collect_history:
                diag["selection"][i] = None                       # no grid searched
        else:
            # selection: scaler AND probe fit on the 80% carve ONLY (mirrors ridge probe's
            # sc-on-carve); the validation rows never touch the scaler or the fit.
            sc_sel = StandardScaler().fit(train_feats[i][tr_np])
            Xtr_s = torch.as_tensor(sc_sel.transform(train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xva_s = torch.as_tensor(sc_sel.transform(train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            best_wd, best_val, sel = wd_grid[0], float("inf"), {}
            for cand in wd_grid:
                m = _fit_quantile_linear(Xtr_s, ytr[tr], q, cand, epochs, lr, device)
                with torch.no_grad():
                    v = chronos2_quantile_loss(m(Xva_s).view(-1, Q, H), ytr[va], q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd = v, cand
            wd = best_wd
            if collect_history:
                diag["selection"][i] = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                                        "chosen_wd": float(best_wd)}

        # training-curve carve-fit: train on the 80%, record train+val each epoch (diagnostic
        # only -- val here is the held-out 20%, never the test split, never used for selection).
        if collect_history:
            sc_h = StandardScaler().fit(train_feats[i][tr_np])
            Xh_tr = torch.as_tensor(sc_h.transform(train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xh_va = torch.as_tensor(sc_h.transform(train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            hist = {"train": [], "val": []}
            _fit_quantile_linear(Xh_tr, ytr[tr], q, wd, epochs, lr, device,
                                 Xval=Xh_va, yval=ytr[va], history=hist)
            diag["wd"][i] = float(wd)
            diag["history"][i] = hist

        # final: scaler + probe refit on FULL train with the selected wd, scored on test.
        # Train loss is logged so an under-optimized layer (train >> converged) is visible.
        sc = StandardScaler().fit(train_feats[i])
        Xtr = torch.as_tensor(sc.transform(train_feats[i]), dtype=torch.float32, device=device)
        Xte = torch.as_tensor(sc.transform(test_feats[i]), dtype=torch.float32, device=device)
        m = _fit_quantile_linear(Xtr, ytr, q, wd, epochs, lr, device)
        with torch.no_grad():
            train_loss = chronos2_quantile_loss(m(Xtr).view(-1, Q, H), ytr, q).item()
            pred_te = m(Xte).view(-1, Q, H)
            out[i] = float(chronos2_quantile_loss(pred_te, yte, q).item())
            if diag is not None:
                diag["test_mean_pinball"][i] = float(mean_pinball_loss(pred_te, yte, q).item())
            if collect_test_median:
                diag["test_median"][i] = pred_te[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred_te, yte, q).cpu().numpy().astype(np.float64)
        print(f"    [quantile] L{i:>2}  wd={wd:g}  train={train_loss:.3f}  test={out[i]:.3f}")
    return (out, diag) if diag is not None else out


# ---- Frozen fit/predict split of the pooled quantile probe (cross-dataset transfer) ---- #
# quantile_probe() trains AND scores in ONE call and returns only per-layer test losses — it
# never hands back the trained (scaler, weights, wd), so a probe can't be re-applied to a
# DIFFERENT dataset. fit_quantile_probe + predict_quantile_probe split that call into a
# training half returning the FROZEN per-layer probe and an evaluation half that scores
# arbitrary features. The training half mirrors quantile_probe's path EXACTLY (same SEED 80/20
# carve, same wd grid + selection, same StandardScaler + Linear refit on FULL train), so
# predict_quantile_probe(fit_quantile_probe(tr), te) reproduces quantile_probe(tr, te) on the
# same device (asserted in tests/test_ood_transfer.py). quantile_probe stays UNCHANGED; the
# committed numbers are untouched. Used by the strict cross-dataset OOD transfer driver
# (experiments/run_ood_transfer.py): fit once on a SOURCE, freeze, evaluate on every TARGET.

def fit_quantile_probe(train_feats, train_labels, quantiles=CHRONOS2_QUANTILES,
                       epochs=300, lr=1e-2, weight_decay=1e-3, wd_grid=None, device=None):
    """Train the per-layer linear quantile probe and RETURN the frozen fitted probe (not scores).

    train_labels : (n, H) arcsinh trajectory labels (id_data.Y_train_traj) — NOT the scalar y.
    Returns {layer: {"scaler": StandardScaler (fit on FULL train), "linear": nn.Linear(d, Q*H)
                     in eval mode on `device`, "wd": float, "selection": {...}|None,
                     "in_features": int, "out_features": int, "device": str}}.
    The training path is identical to quantile_probe's up to the omitted test scoring — the
    collect_history carve-fit that quantile_probe optionally runs between selection and the
    final refit does NOT affect that refit (each _fit_quantile_linear re-seeds torch), so a
    probe fit here and scored with predict_quantile_probe reproduces quantile_probe's losses.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    if Ytr.ndim != 2:
        raise ValueError(f"fit_quantile_probe needs (n, H) trajectory labels, got shape {Ytr.shape} "
                         "-- pass Y_train_traj, not the scalar y")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    Q, H = len(quantiles), Ytr.shape[1]
    ytr = torch.as_tensor(Ytr, device=device)

    rng = np.random.default_rng(SEED)                     # SAME seed-based 80/20 carve as quantile_probe
    perm = rng.permutation(Ytr.shape[0])
    n_val = max(1, int(0.2 * Ytr.shape[0]))
    va_np, tr_np = perm[:n_val], perm[n_val:]
    tr = torch.as_tensor(tr_np, dtype=torch.long, device=device)
    va = torch.as_tensor(va_np, dtype=torch.long, device=device)

    fitted = {}
    for i in range(NUM_LAYERS):
        if wd_grid is None:
            wd, selection = weight_decay, None
        else:
            # selection: scaler AND probe fit on the 80% carve ONLY (mirrors quantile_probe)
            sc_sel = StandardScaler().fit(train_feats[i][tr_np])
            Xtr_s = torch.as_tensor(sc_sel.transform(train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xva_s = torch.as_tensor(sc_sel.transform(train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            best_wd, best_val, sel = wd_grid[0], float("inf"), {}
            for cand in wd_grid:
                m = _fit_quantile_linear(Xtr_s, ytr[tr], q, cand, epochs, lr, device)
                with torch.no_grad():
                    v = chronos2_quantile_loss(m(Xva_s).view(-1, Q, H), ytr[va], q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd = v, cand
            wd = best_wd
            selection = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "chosen_wd": float(best_wd)}

        # final: scaler + probe refit on FULL train with the selected wd (same as quantile_probe)
        sc = StandardScaler().fit(train_feats[i])
        Xtr = torch.as_tensor(sc.transform(train_feats[i]), dtype=torch.float32, device=device)
        m = _fit_quantile_linear(Xtr, ytr, q, wd, epochs, lr, device)   # returned in eval() mode
        fitted[i] = {"scaler": sc, "linear": m, "wd": float(wd), "selection": selection,
                     "in_features": int(m.in_features), "out_features": int(m.out_features),
                     "device": str(device)}
        print(f"    [fit] L{i:>2}  wd={wd:g}  out_dim={m.out_features}")
    return fitted


def fit_quantile_probe_explicit_val(train_feats, train_labels, val_feats, val_labels,
                                    quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                                    weight_decay=1e-3, wd_grid=None, device=None,
                                    init_seed=SEED):
    """Like fit_quantile_probe, but weight-decay selection uses an EXPLICIT, temporally held-out
    validation set (val_feats/val_labels) instead of the seed-based 80/20 carve of train.

    Used by the rolling-origin sets (id_data.ROLLING_SETS): the val split is a dedicated LATER
    forecast origin per series, so wd — and the downstream source-selected layer — are chosen on
    genuine out-of-time data. Per layer: the StandardScaler AND the Linear are fit on the FULL
    train split (validation NEVER touches the scaler or the weights); each wd candidate is scored
    on val; the chosen-wd full-train model is kept (no refit needed — it is already trained on all
    of train). The returned dict shape and the recorded selection.{val_loss_by_wd, chosen_wd}
    match fit_quantile_probe exactly, so save_checkpoints / source_selected_layer /
    predict_quantile_probe are unchanged."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    Yva = np.asarray(val_labels, dtype=np.float32)
    if Ytr.ndim != 2 or Yva.ndim != 2:
        raise ValueError("fit_quantile_probe_explicit_val needs (n, H) trajectory labels for BOTH "
                         f"train and val -- got {Ytr.shape} / {Yva.shape}")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    Q, H = len(quantiles), Ytr.shape[1]
    ytr = torch.as_tensor(Ytr, device=device)
    yva = torch.as_tensor(Yva, device=device)

    fitted = {}
    for i in range(NUM_LAYERS):
        sc = StandardScaler().fit(train_feats[i])                       # scaler on FULL train only
        Xtr = torch.as_tensor(sc.transform(train_feats[i]), dtype=torch.float32, device=device)
        Xva = torch.as_tensor(sc.transform(val_feats[i]), dtype=torch.float32, device=device)
        if wd_grid is None:
            wd, selection = weight_decay, None
            m = _fit_quantile_linear(Xtr, ytr, q, wd, epochs, lr, device, init_seed=init_seed)
        else:
            best_wd, best_val, best_m, sel = wd_grid[0], float("inf"), None, {}
            for cand in wd_grid:
                cm = _fit_quantile_linear(Xtr, ytr, q, cand, epochs, lr, device,   # FULL train
                                          init_seed=init_seed)
                with torch.no_grad():
                    v = chronos2_quantile_loss(cm(Xva).view(-1, Q, H), yva, q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd, best_m = v, cand, cm
            wd, m = best_wd, best_m                              # keep the chosen-wd full-train model
            selection = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "chosen_wd": float(best_wd)}
        m.eval()
        fitted[i] = {"scaler": sc, "linear": m, "wd": float(wd), "selection": selection,
                     "in_features": int(m.in_features), "out_features": int(m.out_features),
                     "device": str(device)}
        print(f"    [fit-explicit-val] L{i:>2}  wd={wd:g}  out_dim={m.out_features}")
    return fitted


def predict_quantile_probe(fitted, feats, labels, quantiles=CHRONOS2_QUANTILES, device=None,
                           collect_test_median=False, collect_test_window_loss=False):
    """Apply a FROZEN fitted probe (from fit_quantile_probe) to `feats`/`labels`; NEVER trains.

    feats  : {layer: (n, d)} hidden states from ANY dataset (a target under transfer).
    labels : (n, H) arcsinh trajectory labels for those windows (the target futures).
    Returns {layer: loss} (Chronos-2 quantile loss, lower=better). When any collect_* flag is
    set returns (out, diag) with the SAME diag keys quantile_probe uses — test_mean_pinball
    always, test_median / test_window_loss on request — so the driver's MASE + cluster-bootstrap
    machinery works on a transferred probe unchanged. Deterministic and side-effect-free: the
    fitted weights are not mutated, so one frozen probe can be reused across many targets."""
    quantiles = validate_quantiles(quantiles)
    Yte = np.asarray(labels, dtype=np.float32)
    if Yte.ndim != 2:
        raise ValueError(f"predict_quantile_probe needs (n, H) trajectory labels, got {Yte.shape}")
    Q, H = len(quantiles), Yte.shape[1]
    q_mid = median_index(quantiles)
    if collect_test_median and q_mid is None:
        raise ValueError(
            f"collect_test_median needs the 0.5 level in the quantile set, got {quantiles.tolist()} "
            "— median/MASE metrics are unavailable for this set; skip them, don't substitute")
    out = {}
    diag = ({"test_mean_pinball": {}, "test_median": {}, "test_window_loss": {}}
            if (collect_test_median or collect_test_window_loss) else None)
    for i in range(NUM_LAYERS):
        dev = device or fitted[i]["device"]
        m = fitted[i]["linear"].to(dev)
        sc = fitted[i]["scaler"]
        q_t = torch.as_tensor(quantiles, dtype=torch.float32, device=dev)
        yte = torch.as_tensor(Yte, device=dev)
        Xte = torch.as_tensor(sc.transform(feats[i]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = m(Xte).view(-1, Q, H)
            out[i] = float(chronos2_quantile_loss(pred, yte, q_t).item())
            if diag is not None:
                diag["test_mean_pinball"][i] = float(mean_pinball_loss(pred, yte, q_t).item())
            if collect_test_median:
                diag["test_median"][i] = pred[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred, yte, q_t).cpu().numpy().astype(np.float64)
    return (out, diag) if diag is not None else out


# ---- Chronos-ALIGNED shared forecast-token probe -------------------------------------- #
# One shared Linear(768, Q*output_patch_size) reads EACH native forecast-slot hidden state
# (extract_kout_features -> (n, K, 768)); the K predicted output patches are laid end-to-end
# along the horizon and trimmed to H (K = ceil(H/P), the native slot count). Structurally
# mirrors Chronos-2's own head (shared weights across forecast tokens); strictly LINEAR with
# FRESHLY-INITIALIZED weights (native head is a nonlinear ResidualBlock — this is a
# lower-capacity analogue, NOT the pretrained head). K× fewer params than the pooled probe.

def _apply_shared_head(lin, X, Q, P, H):
    """Apply one shared Linear(d, Q*P) to every forecast slot of X (n, K, d), lay the K
    predicted patches end-to-end -> (n, Q, K*P), then trim to the requested horizon ->
    (n, Q, H). The patch layout mirrors the native 'b n (q p) -> b q (n p)' rearrange
    (concatenate, not add); the trim mirrors native inference for H not a multiple of P
    (the pipeline predicts K=ceil(H/P) whole patches and drops the tail; equivalently the
    native loss zero-pads + masks the target, model.py _compute_loss)."""
    n, K, _ = X.shape
    out = lin(X).view(n, K, Q, P)                            # (n, K, Q, P)
    out = out.permute(0, 2, 1, 3).reshape(n, Q, K * P)[:, :, :H]
    assert out.shape[-1] == H, (
        f"prediction horizon {out.shape[-1]} != requested H={H} — K*P={K*P} slots cover "
        f"less than H, so K was derived from a different horizon than the labels")
    return out


def _fit_slot_scaler(X):                                     # X: (n, K, d)
    n, K, d = X.shape
    return StandardScaler().fit(X.reshape(n * K, d))         # ONE scaler shared across all slots


def _slot_transform(sc, X):
    n, K, d = X.shape
    return sc.transform(X.reshape(n * K, d)).reshape(n, K, d)


def _fit_shared_forecast_linear(Xtr, ytr, q, P, weight_decay, epochs, lr, device,
                                Xval=None, yval=None, history=None, init_seed=SEED):
    """Fit the shared-slot linear head with Chronos-2's quantile loss. Same optimizer convention
    as _fit_quantile_linear (AdamW, decay on weight only; re-seeded each call). The head predicts
    K whole P-step patches; loss is computed on the first H = ytr.shape[1] steps (trimmed).
    `init_seed` selects the Linear init (the ONLY randomness in this deterministic full-batch fit) —
    threaded so the 3-independent-runs protocol can vary it; default SEED keeps every existing call
    byte-identical."""
    Q, H = len(q), ytr.shape[1]
    torch.manual_seed(init_seed)
    lin = torch.nn.Linear(Xtr.shape[-1], Q * P).to(device)
    opt = torch.optim.AdamW(
        [{"params": [lin.weight], "weight_decay": weight_decay},
         {"params": [lin.bias],   "weight_decay": 0.0}], lr=lr)
    lin.train()
    for _ in range(epochs):
        loss = chronos2_quantile_loss(_apply_shared_head(lin, Xtr, Q, P, H), ytr, q)
        if history is not None:
            history["train"].append(loss.item())
            if Xval is not None:
                with torch.no_grad():
                    history["val"].append(
                        chronos2_quantile_loss(_apply_shared_head(lin, Xval, Q, P, H), yval, q).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if history is not None:
        with torch.no_grad():
            history["train"].append(chronos2_quantile_loss(_apply_shared_head(lin, Xtr, Q, P, H), ytr, q).item())
            if Xval is not None:
                history["val"].append(chronos2_quantile_loss(_apply_shared_head(lin, Xval, Q, P, H), yval, q).item())
    lin.eval()
    return lin


def shared_forecast_token_probe(train_feats, train_labels, test_feats, test_labels,
                                quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                                weight_decay=1e-3, wd_grid=None, device=None,
                                collect_history=False, collect_test_median=False,
                                collect_test_window_loss=False,
                                output_patch_size=OUTPUT_PATCH_SIZE):
    """Chronos-aligned quantile probe. ONE shared linear head reads each forecast-slot hidden
    state (extract_kout_features -> (n, K, 768)) and emits that slot's output patch of Q quantiles
    x output_patch_size steps; the K patches concatenate along the horizon and are trimmed to H.

    train_feats/test_feats   : {layer: (n, K, 768)} forecast-slot states (3-D, NOT pooled),
                               with K = ceil(H / output_patch_size) (the native slot count).
    train_labels/test_labels : (n, H) arcsinh trajectories; H need NOT be a multiple of
                               output_patch_size — the head predicts K whole patches and the
                               prediction is trimmed to H (native inference does the same).
    output_patch_size        : the MODEL's output patch size (a Chronos-2 config fact), NOT
                               inferred from H — extract_kout_features asserts the loaded
                               model agrees with this constant.
    Returns {layer: test_loss}, LOWER = better — directly comparable to quantile_probe. Same
    collect_history / collect_test_median / collect_test_window_loss contract as quantile_probe
    (so the driver's MASE + bootstrap machinery works on this probe unchanged)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    Yte = np.asarray(test_labels, dtype=np.float32)
    if Ytr.ndim != 2:
        raise ValueError(f"shared_forecast_token_probe needs (n, H) trajectory labels, got {Ytr.shape}")
    f0 = train_feats[0]
    if np.ndim(f0) != 3:
        raise ValueError(f"needs (n, K, 768) forecast-slot features, got {np.shape(f0)} "
                         "— use extract_kout_features, not extract_window_features")
    K = f0.shape[1]
    Q, H = len(quantiles), Ytr.shape[1]
    P = int(output_patch_size)
    if K != math.ceil(H / P):
        raise ValueError(
            f"features carry K={K} forecast slots, but horizon H={H} with output_patch_size={P} "
            f"needs K=ceil(H/P)={math.ceil(H / P)} — re-extract with the matching horizon")

    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(Ytr, device=device)
    yte = torch.as_tensor(Yte, device=device)

    rng = np.random.default_rng(SEED)                       # same 80/20 carve as quantile_probe
    perm = rng.permutation(Ytr.shape[0])
    n_val = max(1, int(0.2 * Ytr.shape[0]))
    va_np, tr_np = perm[:n_val], perm[n_val:]
    va = torch.as_tensor(va_np, dtype=torch.long, device=device)
    tr = torch.as_tensor(tr_np, dtype=torch.long, device=device)

    out = {}
    diag = ({"wd": {}, "selection": {}, "history": {}, "test_median": {}, "test_window_loss": {},
             "test_mean_pinball": {}}
            if (collect_history or collect_test_median or collect_test_window_loss) else None)
    q_mid = median_index(quantiles)                       # exact 0.5 row, or None
    if collect_test_median and q_mid is None:
        raise ValueError(
            f"collect_test_median needs the 0.5 level in the quantile set, got "
            f"{quantiles.tolist()} — median/MASE metrics are unavailable for this set; "
            "skip them instead of substituting a neighboring quantile")
    for i in range(NUM_LAYERS):
        if wd_grid is None:
            wd = weight_decay
            if collect_history:
                diag["selection"][i] = None
        else:
            sc_sel = _fit_slot_scaler(train_feats[i][tr_np])
            Xtr_s = torch.as_tensor(_slot_transform(sc_sel, train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xva_s = torch.as_tensor(_slot_transform(sc_sel, train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            best_wd, best_val, sel = wd_grid[0], float("inf"), {}
            for cand in wd_grid:
                m = _fit_shared_forecast_linear(Xtr_s, ytr[tr], q, P, cand, epochs, lr, device)
                with torch.no_grad():
                    v = chronos2_quantile_loss(_apply_shared_head(m, Xva_s, Q, P, H), ytr[va], q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd = v, cand
            wd = best_wd
            if collect_history:
                diag["selection"][i] = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                                        "chosen_wd": float(best_wd)}

        if collect_history:
            sc_h = _fit_slot_scaler(train_feats[i][tr_np])
            Xh_tr = torch.as_tensor(_slot_transform(sc_h, train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xh_va = torch.as_tensor(_slot_transform(sc_h, train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            hist = {"train": [], "val": []}
            _fit_shared_forecast_linear(Xh_tr, ytr[tr], q, P, wd, epochs, lr, device,
                                        Xval=Xh_va, yval=ytr[va], history=hist)
            diag["wd"][i] = float(wd)
            diag["history"][i] = hist

        sc = _fit_slot_scaler(train_feats[i])
        Xtr = torch.as_tensor(_slot_transform(sc, train_feats[i]), dtype=torch.float32, device=device)
        Xte = torch.as_tensor(_slot_transform(sc, test_feats[i]), dtype=torch.float32, device=device)
        m = _fit_shared_forecast_linear(Xtr, ytr, q, P, wd, epochs, lr, device)
        with torch.no_grad():
            train_loss = chronos2_quantile_loss(_apply_shared_head(m, Xtr, Q, P, H), ytr, q).item()
            pred_te = _apply_shared_head(m, Xte, Q, P, H)
            out[i] = float(chronos2_quantile_loss(pred_te, yte, q).item())
            if diag is not None:
                diag["test_mean_pinball"][i] = float(mean_pinball_loss(pred_te, yte, q).item())
            if collect_test_median:
                diag["test_median"][i] = pred_te[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred_te, yte, q).cpu().numpy().astype(np.float64)
        print(f"    [fslot] L{i:>2}  wd={wd:g}  train={train_loss:.3f}  test={out[i]:.3f}")
    return (out, diag) if diag is not None else out


# ---- Frozen fit/predict split of the shared forecast-token probe (tunnel / PT-OOD) ---- #
# shared_forecast_token_probe() trains AND scores in one call, so a probe can't be re-applied to a
# DIFFERENT dataset (PT-OOD transfer) or fit with an explicit temporal-val split (rolling tunnel).
# These two functions split it the way fit_quantile_probe_explicit_val / predict_quantile_probe split
# the pooled linear probe — SAME wd-selection-on-explicit-val logic, SAME slot mechanics as
# shared_forecast_token_probe (_fit_slot_scaler shared across slots, _apply_shared_head, the
# K=ceil(H/P) contract). They are the LINEAR-shared-head twin of fit_forecast_slot_native_head /
# predict_forecast_slot_native_head (which do this for the NONLINEAR ResidualBlock head). The
# selection dict shape matches fit_quantile_probe_explicit_val so source_selected_layer /
# save_checkpoints are unchanged; predict never trains, so one frozen probe scores many targets.

def fit_shared_forecast_probe_explicit_val(train_feats, train_labels, val_feats, val_labels,
                                           quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                                           weight_decay=1e-3, wd_grid=None, device=None,
                                           init_seed=SEED, output_patch_size=OUTPUT_PATCH_SIZE):
    """Train the per-layer shared forecast-token probe with wd selected on an EXPLICIT temporal
    validation split, and RETURN the frozen fitted probe (not scores) — the shared-head twin of
    fit_quantile_probe_explicit_val.

    train_feats / val_feats  : {layer: (n, K, 768)} forecast-slot states (extract_kout_features),
                               3-D, K = ceil(H / output_patch_size). ONE StandardScaler shared
                               across all slots, fit on FULL train only.
    train_labels / val_labels: (n, H) arcsinh trajectory labels — raises on 1-D / on 2-D feats.
    Per layer: the slot-scaler AND the Linear are fit on FULL train (validation NEVER touches the
    scaler or the weights); each wd candidate is scored on val; the chosen-wd full-train model is
    kept (already trained on all of train — no refit). Returns
    {layer: {"scaler", "linear" (nn.Linear, eval mode), "wd", "selection": {val_loss_by_wd,
             chosen_wd} | None, "in_features", "out_features", "output_patch_size", "K",
             "family": "shared_forecast", "pooling_or_token_type": "forecast_slot", "device"}}.
    The selection dict shape matches fit_quantile_probe_explicit_val, so source_selected_layer /
    save_checkpoints / predict_shared_forecast_probe are unchanged."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    Yva = np.asarray(val_labels, dtype=np.float32)
    if Ytr.ndim != 2 or Yva.ndim != 2:
        raise ValueError("fit_shared_forecast_probe_explicit_val needs (n, H) trajectory labels for "
                         f"BOTH train and val -- got {Ytr.shape} / {Yva.shape}")
    f0 = train_feats[0]
    if np.ndim(f0) != 3:
        raise ValueError(f"needs (n, K, 768) forecast-slot features, got {np.shape(f0)} "
                         "— use extract_kout_features, not extract_window_features")
    K = f0.shape[1]
    Q, H = len(quantiles), Ytr.shape[1]
    P = int(output_patch_size)
    if K != math.ceil(H / P):
        raise ValueError(f"features carry K={K} slots, but H={H}, P={P} needs K=ceil(H/P)="
                         f"{math.ceil(H / P)} — re-extract with the matching horizon")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(Ytr, device=device)
    yva = torch.as_tensor(Yva, device=device)

    fitted = {}
    for i in sorted(train_feats):          # iterate the feature-dict keys, not range(NUM_LAYERS): the
        # fslot line appends a 14th point (post-final-LN slots) as an extra key beyond L12
        sc = _fit_slot_scaler(train_feats[i])                          # slot-scaler on FULL train only
        Xtr = torch.as_tensor(_slot_transform(sc, train_feats[i]), dtype=torch.float32, device=device)
        Xva = torch.as_tensor(_slot_transform(sc, val_feats[i]), dtype=torch.float32, device=device)
        if wd_grid is None:
            wd, selection = weight_decay, None
            m = _fit_shared_forecast_linear(Xtr, ytr, q, P, wd, epochs, lr, device, init_seed=init_seed)
        else:
            best_wd, best_val, best_m, sel = wd_grid[0], float("inf"), None, {}
            for cand in wd_grid:
                cm = _fit_shared_forecast_linear(Xtr, ytr, q, P, cand, epochs, lr, device,  # FULL train
                                                 init_seed=init_seed)
                with torch.no_grad():
                    v = chronos2_quantile_loss(_apply_shared_head(cm, Xva, Q, P, H), yva, q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd, best_m = v, cand, cm
            wd, m = best_wd, best_m                              # keep the chosen-wd full-train model
            selection = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "chosen_wd": float(best_wd)}
        m.eval()
        fitted[i] = {"scaler": sc, "linear": m, "wd": float(wd), "selection": selection,
                     "in_features": int(m.in_features), "out_features": int(m.out_features),
                     "output_patch_size": P, "K": int(K),
                     "family": "shared_forecast", "pooling_or_token_type": "forecast_slot",
                     "device": str(device)}
        print(f"    [fit-explicit-val fslot] L{i:>2}  wd={wd:g}  out_dim={m.out_features}")
    return fitted


def predict_shared_forecast_probe(fitted, feats, labels, quantiles=CHRONOS2_QUANTILES, device=None,
                                  collect_test_median=False, collect_test_window_loss=False,
                                  output_patch_size=OUTPUT_PATCH_SIZE):
    """Apply a FROZEN shared forecast-token probe (from fit_shared_forecast_probe_explicit_val) to
    `feats`/`labels`; NEVER trains — the shared-head twin of predict_quantile_probe.

    feats  : {layer: (n, K, 768)} forecast-slot states from ANY dataset (a PT-OOD target under
             transfer, or this dataset's own test split).
    labels : (n, H) arcsinh trajectory labels for those windows.
    Returns {layer: loss} (Chronos-2 quantile loss, lower=better). When any collect_* flag is set
    returns (out, diag) with the SAME diag keys predict_quantile_probe uses — test_mean_pinball
    always, test_median / test_window_loss on request — so the driver's MASE + cluster-bootstrap
    machinery works on a transferred shared-head probe unchanged. Deterministic and side-effect-free:
    the frozen weights are not mutated, so one probe can be reused across many targets."""
    quantiles = validate_quantiles(quantiles)
    Yte = np.asarray(labels, dtype=np.float32)
    if Yte.ndim != 2:
        raise ValueError(f"predict_shared_forecast_probe needs (n, H) trajectory labels, got {Yte.shape}")
    f0 = feats[0]
    if np.ndim(f0) != 3:
        raise ValueError(f"needs (n, K, 768) forecast-slot features, got {np.shape(f0)} "
                         "— use extract_kout_features, not extract_window_features")
    K = f0.shape[1]
    Q, H = len(quantiles), Yte.shape[1]
    P = int(output_patch_size)
    if K != math.ceil(H / P):
        raise ValueError(f"features carry K={K} slots, but H={H}, P={P} needs K=ceil(H/P)="
                         f"{math.ceil(H / P)}")
    q_mid = median_index(quantiles)
    if collect_test_median and q_mid is None:
        raise ValueError(
            f"collect_test_median needs the 0.5 level in the quantile set, got {quantiles.tolist()} "
            "— median/MASE metrics are unavailable for this set; skip them, don't substitute")
    out = {}
    diag = ({"test_mean_pinball": {}, "test_median": {}, "test_window_loss": {}}
            if (collect_test_median or collect_test_window_loss) else None)
    for i in sorted(feats):                # iterate feature-dict keys (14 for fslot: L0..L12 + post-LN)
        dev = device or fitted[i]["device"]
        m = fitted[i]["linear"].to(dev)
        m.eval()
        sc = fitted[i]["scaler"]
        q_t = torch.as_tensor(quantiles, dtype=torch.float32, device=dev)
        yte = torch.as_tensor(Yte, device=dev)
        Xte = torch.as_tensor(_slot_transform(sc, feats[i]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = _apply_shared_head(m, Xte, Q, P, H)
            out[i] = float(chronos2_quantile_loss(pred, yte, q_t).item())
            if diag is not None:
                diag["test_mean_pinball"][i] = float(mean_pinball_loss(pred, yte, q_t).item())
            if collect_test_median:
                diag["test_median"][i] = pred[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred, yte, q_t).cpu().numpy().astype(np.float64)
    return (out, diag) if diag is not None else out


# ========== Layerwise LINEAR classification probe (TASK-SHIFT Exp A) ============== #
# A strictly-linear Linear(d, n_classes) + cross-entropy probe, the classification twin of the
# fit/predict_shared_forecast_probe pair: wd selected on an EXPLICIT validation split, fit on FULL
# train, frozen predict on test. Two design points that matter here:
#   * it iterates ``sorted(feats)`` (the extracted feature-dict keys), NEVER range(NUM_LAYERS), so the
#     14th point (post-final-LN, L12+LN) is probed and never silently dropped — unlike the 13-point
#     ``linear_probe``/``fit_layerwise_probes`` reference used by the UEA baseline.
#   * ``init_seed`` selects the Linear init (the only randomness in the deterministic full-batch fit),
#     so N independent probe seeds give genuine SEED BANDS for Plot A (matches the fslot probe + the
#     classification FT head). wd is chosen by validation CROSS-ENTROPY (smooth), never by test.

def _fit_linear_cls(Xtr, ytr, n_classes, weight_decay, epochs, lr, device,
                    Xval=None, yval=None, history=None, init_seed=SEED):
    """Fit one strictly-linear map (d -> n_classes) with cross-entropy; return the trained module in
    eval() mode. Re-seeded each call (init_seed) so every layer / wd candidate starts from the same
    init and the 3 probe seeds are independent. AdamW decays the WEIGHT only (the bias is free), same
    convention as _fit_quantile_linear. If ``history`` (dict with 'train'/'val' lists) is passed,
    per-epoch train CE (and val CE when Xval/yval given) is appended for the training-curve diagnostic.
    """
    torch.manual_seed(init_seed)
    lin = torch.nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(
        [{"params": [lin.weight], "weight_decay": weight_decay},
         {"params": [lin.bias],   "weight_decay": 0.0}],
        lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    lin.train()
    for _ in range(epochs):
        loss = ce(lin(Xtr), ytr)
        if history is not None:
            history["train"].append(loss.item())
            if Xval is not None:
                with torch.no_grad():
                    history["val"].append(ce(lin(Xval), yval).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if history is not None:                       # final converged point (epoch = epochs)
        with torch.no_grad():
            history["train"].append(ce(lin(Xtr), ytr).item())
            if Xval is not None:
                history["val"].append(ce(lin(Xval), yval).item())
    lin.eval()
    return lin


def fit_linear_cls_probe_explicit_val(train_feats, train_labels, val_feats, val_labels,
                                      n_classes=2, epochs=300, lr=1e-2, weight_decay=1e-3,
                                      wd_grid=None, device=None, init_seed=SEED):
    """Train the per-layer LINEAR classification probe (Linear+CE) with wd selected on an EXPLICIT
    validation split, and RETURN the frozen fitted probes (not scores).

    train_feats / val_feats  : {layer: (n, d)} pooled features. Iterated via ``sorted(...)`` so a
                               14-key dict (L0..L12 + L12+LN) is fully probed.
    train_labels / val_labels: 1-D integer class labels; raises on non-1-D.
    Per layer: a StandardScaler is fit on FULL train (val/test never touch it); the Linear+CE head is
    fit on FULL train for each wd candidate and scored on val by CROSS-ENTROPY; the chosen-wd
    full-train model is kept (no refit). Returns
    {layer: {"scaler", "linear" (nn.Linear, eval), "wd", "selection": {val_ce_by_wd, chosen_wd}|None,
             "in_features", "n_classes", "family": "linear_cls", "device"}}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ytr_np = np.asarray(train_labels)
    yva_np = np.asarray(val_labels)
    if ytr_np.ndim != 1 or yva_np.ndim != 1:
        raise ValueError("fit_linear_cls_probe_explicit_val needs 1-D integer class labels for BOTH "
                         f"train and val -- got {ytr_np.shape} / {yva_np.shape}")
    ytr = torch.as_tensor(ytr_np.astype(np.int64), dtype=torch.long, device=device)
    yva = torch.as_tensor(yva_np.astype(np.int64), dtype=torch.long, device=device)
    ce = torch.nn.CrossEntropyLoss()

    fitted = {}
    for i in sorted(train_feats):          # feature-dict keys, NOT range(NUM_LAYERS): keep L12+LN (key 13)
        sc = StandardScaler().fit(train_feats[i])                     # scaler on FULL train only
        Xtr = torch.as_tensor(sc.transform(train_feats[i]), dtype=torch.float32, device=device)
        Xva = torch.as_tensor(sc.transform(val_feats[i]), dtype=torch.float32, device=device)
        if wd_grid is None:
            wd, selection = weight_decay, None
            m = _fit_linear_cls(Xtr, ytr, n_classes, wd, epochs, lr, device, init_seed=init_seed)
        else:
            best_wd, best_val, best_m, sel = wd_grid[0], float("inf"), None, {}
            for cand in wd_grid:
                cm = _fit_linear_cls(Xtr, ytr, n_classes, cand, epochs, lr, device,   # FULL train
                                     init_seed=init_seed)
                with torch.no_grad():
                    v = ce(cm(Xva), yva).item()                       # select wd by VAL cross-entropy
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd, best_m = v, cand, cm
            wd, m = best_wd, best_m                                   # keep the chosen-wd full-train model
            selection = {"val_ce_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "chosen_wd": float(best_wd)}
        m.eval()
        fitted[i] = {"scaler": sc, "linear": m, "wd": float(wd), "selection": selection,
                     "in_features": int(m.in_features), "n_classes": int(n_classes),
                     "family": "linear_cls", "device": str(device)}
    return fitted


def predict_linear_cls_probe(fitted, feats, labels, device=None,
                             collect_test_correct=False, collect_test_ce=False):
    """Apply a FROZEN linear classification probe (from fit_linear_cls_probe_explicit_val) to
    feats/labels; NEVER trains. Returns {layer: test_accuracy} (higher = better). With any collect_*
    flag returns (out, diag) where diag carries per-window correctness (for a test bootstrap) and/or
    per-layer test cross-entropy. Iterates ``sorted(feats)`` so the 14th point is scored. Frozen
    weights are not mutated (one probe can score many splits)."""
    yte_np = np.asarray(labels)
    if yte_np.ndim != 1:
        raise ValueError(f"predict_linear_cls_probe needs 1-D integer class labels, got {yte_np.shape}")
    ce = torch.nn.CrossEntropyLoss()
    out = {}
    diag = ({"test_correct": {}, "test_ce": {}}
            if (collect_test_correct or collect_test_ce) else None)
    for i in sorted(feats):
        dev = device or fitted[i]["device"]
        m = fitted[i]["linear"].to(dev)
        m.eval()
        sc = fitted[i]["scaler"]
        Xte = torch.as_tensor(sc.transform(feats[i]), dtype=torch.float32, device=dev)
        yte = torch.as_tensor(yte_np.astype(np.int64), dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = m(Xte)
            correct = (logits.argmax(dim=1) == yte).to(torch.float64).cpu().numpy()
            out[i] = float(correct.mean())
            if diag is not None:
                if collect_test_correct:
                    diag["test_correct"][i] = correct.astype(np.float64)
                if collect_test_ce:
                    diag["test_ce"][i] = float(ce(logits, yte).item())
    return (out, diag) if diag is not None else out


# ========== Higher-capacity forecasting probes (capacity controls) ================ #
# Nonlinear ResidualBlock heads (probing.heads) trained FROM SCRATCH — capacity controls
# for the linear quantile probe. They are NOT linear-accessibility measures: they quantify
# forecast decodability under a HIGHER-CAPACITY readout. Fit/predict mirror
# fit_quantile_probe / predict_quantile_probe exactly (per-layer StandardScaler,
# torch.manual_seed(SEED), AdamW with weight-decay on weights only, fixed epochs, seed-based
# 80/20 source-val carve for the wd grid, refit on FULL train), so the OOD-transfer driver's
# checkpoint / MASE / cluster-bootstrap machinery works unchanged. The frozen probe stores
# "head" (an nn.Module) instead of "linear", plus "family"/"hidden_dim"/"dropout" so a
# checkpoint can rebuild the exact module, and "source_val_loss" (the chosen-wd carve val loss)
# so the driver can pick a SOURCE-VALIDATED layer without ever touching the target.

# ---- content_mlp_head: nonlinear head on the (n, 768) mean-pooled content vector ---- #

def _fit_content_mlp(Xtr, ytr, q, H, hidden_dim, dropout, weight_decay, epochs, lr, device,
                     Xval=None, yval=None, history=None):
    """Fit one content-pooled ResidualBlock head (d -> Q*H) with Chronos-2's quantile loss.
    Same optimizer convention as _fit_quantile_linear (re-seeded each call; decay on weights
    only). Returns the head in eval() mode."""
    Q = len(q)
    torch.manual_seed(SEED)
    head = build_head(Xtr.shape[1], Q * H, hidden_dim=hidden_dim, dropout=dropout, device=device)
    opt = torch.optim.AdamW(wd_param_groups(head, weight_decay), lr=lr)
    head.train()
    for _ in range(epochs):
        loss = chronos2_quantile_loss(head(Xtr).view(-1, Q, H), ytr, q)
        if history is not None:
            history["train"].append(loss.item())
            if Xval is not None:
                head.eval()
                with torch.no_grad():
                    history["val"].append(chronos2_quantile_loss(head(Xval).view(-1, Q, H), yval, q).item())
                head.train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if history is not None:
        head.eval()
        with torch.no_grad():
            history["train"].append(chronos2_quantile_loss(head(Xtr).view(-1, Q, H), ytr, q).item())
            if Xval is not None:
                history["val"].append(chronos2_quantile_loss(head(Xval).view(-1, Q, H), yval, q).item())
    head.eval()
    return head


def fit_content_mlp_head(train_feats, train_labels, quantiles=CHRONOS2_QUANTILES,
                         epochs=300, lr=1e-2, weight_decay=1e-3, wd_grid=None, device=None,
                         hidden_dim=NATIVE_D_FF, dropout=0.0):
    """Train the per-layer content_mlp_head and RETURN the frozen fitted probe (not scores).

    train_feats  : {layer: (n, 768)} mean-pooled content features (extract_window_features).
    train_labels : (n, H) arcsinh trajectory labels (id_data.Y_train_traj) — raises on 1-D.
    Returns {layer: {"scaler", "head" (nn.Module, eval mode), "wd", "selection",
                     "source_val_loss", "in_features", "out_features", "hidden_dim", "dropout",
                     "family", "pooling_or_token_type", "param_count", "device"}}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    if Ytr.ndim != 2:
        raise ValueError(f"fit_content_mlp_head needs (n, H) trajectory labels, got shape {Ytr.shape} "
                         "-- pass Y_train_traj, not the scalar y")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    Q, H = len(quantiles), Ytr.shape[1]
    ytr = torch.as_tensor(Ytr, device=device)

    rng = np.random.default_rng(SEED)                     # SAME 80/20 carve as fit_quantile_probe
    perm = rng.permutation(Ytr.shape[0])
    n_val = max(1, int(0.2 * Ytr.shape[0]))
    va_np, tr_np = perm[:n_val], perm[n_val:]
    tr = torch.as_tensor(tr_np, dtype=torch.long, device=device)
    va = torch.as_tensor(va_np, dtype=torch.long, device=device)

    fitted = {}
    for i in range(NUM_LAYERS):
        if wd_grid is None:
            wd, selection, src_val = weight_decay, None, None
        else:
            sc_sel = StandardScaler().fit(train_feats[i][tr_np])
            Xtr_s = torch.as_tensor(sc_sel.transform(train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xva_s = torch.as_tensor(sc_sel.transform(train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            best_wd, best_val, sel = wd_grid[0], float("inf"), {}
            for cand in wd_grid:
                m = _fit_content_mlp(Xtr_s, ytr[tr], q, H, hidden_dim, dropout, cand, epochs, lr, device)
                with torch.no_grad():
                    v = chronos2_quantile_loss(m(Xva_s).view(-1, Q, H), ytr[va], q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd = v, cand
            wd, src_val = best_wd, best_val
            selection = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "chosen_wd": float(best_wd)}

        sc = StandardScaler().fit(train_feats[i])
        Xtr = torch.as_tensor(sc.transform(train_feats[i]), dtype=torch.float32, device=device)
        m = _fit_content_mlp(Xtr, ytr, q, H, hidden_dim, dropout, wd, epochs, lr, device)
        fitted[i] = {"scaler": sc, "head": m, "wd": float(wd), "selection": selection,
                     "source_val_loss": (None if src_val is None else float(src_val)),
                     "in_features": int(m.hidden_layer.in_features), "out_features": Q * H,
                     "hidden_dim": int(hidden_dim), "dropout": float(dropout),
                     "family": "content_mlp_head", "pooling_or_token_type": "content",
                     "param_count": head_param_count(m), "device": str(device)}
        print(f"    [fit content_mlp] L{i:>2}  wd={wd:g}  params={fitted[i]['param_count']:,}")
    return fitted


def fit_content_mlp_head_explicit_val(train_feats, train_labels, val_feats, val_labels,
                                      quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                                      weight_decay=1e-3, wd_grid=None, device=None,
                                      hidden_dim=NATIVE_D_FF, dropout=0.0):
    """Like fit_content_mlp_head, but weight-decay selection uses an EXPLICIT, temporally
    held-out validation split (val_feats/val_labels) instead of the seed-based 80/20 carve of
    train — the nonlinear analogue of fit_quantile_probe_explicit_val.

    Used by the rolling-origin sets (id_data.ROLLING_SETS): the val split is a dedicated LATER
    forecast origin per series, so wd — and the downstream source-selected layer — are chosen on
    genuine out-of-time data. Per layer: the StandardScaler AND the ResidualBlock head are fit on
    the FULL train split (validation NEVER touches the scaler or the weights); each wd candidate is
    scored on val; the chosen-wd full-train head is kept (already trained on all of train — no
    refit). The returned dict shape, selection.{val_loss_by_wd, chosen_wd} and source_val_loss match
    fit_content_mlp_head exactly, so save_checkpoints / _source_selected_layer /
    predict_content_mlp_head are unchanged. selection additionally carries train_loss_by_wd and the
    fitted dict carries source_train_loss (train-loss diagnostics; the checkpoint keeps them in
    selection). NO random 80/20 carve is ever constructed here."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    Yva = np.asarray(val_labels, dtype=np.float32)
    if Ytr.ndim != 2 or Yva.ndim != 2:
        raise ValueError("fit_content_mlp_head_explicit_val needs (n, H) trajectory labels for BOTH "
                         f"train and val -- got {Ytr.shape} / {Yva.shape}")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    Q, H = len(quantiles), Ytr.shape[1]
    ytr = torch.as_tensor(Ytr, device=device)
    yva = torch.as_tensor(Yva, device=device)

    fitted = {}
    for i in range(NUM_LAYERS):
        sc = StandardScaler().fit(train_feats[i])                       # scaler on FULL train only
        Xtr = torch.as_tensor(sc.transform(train_feats[i]), dtype=torch.float32, device=device)
        Xva = torch.as_tensor(sc.transform(val_feats[i]), dtype=torch.float32, device=device)
        if wd_grid is None:
            wd, selection, src_val = weight_decay, None, None
            m = _fit_content_mlp(Xtr, ytr, q, H, hidden_dim, dropout, wd, epochs, lr, device)
            with torch.no_grad():
                tr_loss = chronos2_quantile_loss(m(Xtr).view(-1, Q, H), ytr, q).item()
        else:
            best_wd, best_val, best_m, sel, trl = wd_grid[0], float("inf"), None, {}, {}
            for cand in wd_grid:
                cm = _fit_content_mlp(Xtr, ytr, q, H, hidden_dim, dropout, cand, epochs, lr, device)
                with torch.no_grad():                                   # FULL-train head, val-only score
                    v = chronos2_quantile_loss(cm(Xva).view(-1, Q, H), yva, q).item()
                    t = chronos2_quantile_loss(cm(Xtr).view(-1, Q, H), ytr, q).item()
                sel[cand], trl[cand] = v, t
                if v < best_val:
                    best_val, best_wd, best_m = v, cand, cm
            wd, src_val, m = best_wd, best_val, best_m                  # keep the chosen-wd full-train head
            tr_loss = trl[best_wd]
            selection = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "train_loss_by_wd": {float(k): float(v) for k, v in trl.items()},
                         "chosen_wd": float(best_wd)}
        m.eval()
        fitted[i] = {"scaler": sc, "head": m, "wd": float(wd), "selection": selection,
                     "source_val_loss": (None if src_val is None else float(src_val)),
                     "source_train_loss": float(tr_loss),
                     "in_features": int(m.hidden_layer.in_features), "out_features": Q * H,
                     "hidden_dim": int(hidden_dim), "dropout": float(dropout),
                     "family": "content_mlp_head", "pooling_or_token_type": "content",
                     "param_count": head_param_count(m), "device": str(device)}
        vtxt = "n/a" if src_val is None else f"{src_val:.3f}"
        print(f"    [fit-explicit-val content_mlp] L{i:>2}  wd={wd:g}  train={tr_loss:.3f}  "
              f"val={vtxt}  params={fitted[i]['param_count']:,}")
    return fitted


def predict_content_mlp_head(fitted, feats, labels, quantiles=CHRONOS2_QUANTILES, device=None,
                             collect_test_median=False, collect_test_window_loss=False):
    """Apply a FROZEN content_mlp_head (from fit_content_mlp_head) to feats/labels; NEVER trains.
    Same diag contract as predict_quantile_probe. Deterministic and side-effect-free."""
    quantiles = validate_quantiles(quantiles)
    Yte = np.asarray(labels, dtype=np.float32)
    if Yte.ndim != 2:
        raise ValueError(f"predict_content_mlp_head needs (n, H) trajectory labels, got {Yte.shape}")
    Q, H = len(quantiles), Yte.shape[1]
    q_mid = median_index(quantiles)
    if collect_test_median and q_mid is None:
        raise ValueError(f"collect_test_median needs the 0.5 level, got {quantiles.tolist()}")
    out = {}
    diag = ({"test_mean_pinball": {}, "test_median": {}, "test_window_loss": {}}
            if (collect_test_median or collect_test_window_loss) else None)
    for i in range(NUM_LAYERS):
        dev = device or fitted[i]["device"]
        m = fitted[i]["head"].to(dev)
        m.eval()
        sc = fitted[i]["scaler"]
        q_t = torch.as_tensor(quantiles, dtype=torch.float32, device=dev)
        yte = torch.as_tensor(Yte, device=dev)
        Xte = torch.as_tensor(sc.transform(feats[i]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = m(Xte).view(-1, Q, H)
            out[i] = float(chronos2_quantile_loss(pred, yte, q_t).item())
            if diag is not None:
                diag["test_mean_pinball"][i] = float(mean_pinball_loss(pred, yte, q_t).item())
            if collect_test_median:
                diag["test_median"][i] = pred[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred, yte, q_t).cpu().numpy().astype(np.float64)
    return (out, diag) if diag is not None else out


# ---- forecast_slot_native_head: ONE shared nonlinear head over the K native slots ---- #
# Native-style: the SAME ResidualBlock(768 -> hidden -> Q*P) is applied to each of the K=ceil(H/P)
# forecast-slot states (extract_kout_features -> (n, K, 768)); the K output patches concatenate
# to (n, Q, K*P) and trim to H (reusing _apply_shared_head — a ResidualBlock broadcasts over the
# slot axis exactly as the Linear does). ONE StandardScaler shared across all slots. This is the
# nonlinear analogue of shared_forecast_token_probe and the closest capacity control to the real
# native head (which is this block with pretrained weights on the L12-through-final-norm slots).

def _fit_forecast_slot_head(Xtr, ytr, q, P, hidden_dim, dropout, weight_decay, epochs, lr, device,
                            Xval=None, yval=None, history=None, init_seed=SEED):
    """Fit one shared ResidualBlock over the K forecast slots with Chronos-2's quantile loss.
    Same optimizer convention as _fit_shared_forecast_linear; returns the head in eval() mode.
    `init_seed` selects the head init AND (with dropout > 0) the dropout stream — the only
    randomness in this deterministic full-batch fit; threaded so the 3-independent-runs protocol can
    vary it. Default SEED keeps every existing fit_forecast_slot_native_head call byte-identical."""
    Q, H = len(q), ytr.shape[1]
    torch.manual_seed(init_seed)
    head = build_head(Xtr.shape[-1], Q * P, hidden_dim=hidden_dim, dropout=dropout, device=device)
    opt = torch.optim.AdamW(wd_param_groups(head, weight_decay), lr=lr)
    head.train()
    for _ in range(epochs):
        loss = chronos2_quantile_loss(_apply_shared_head(head, Xtr, Q, P, H), ytr, q)
        if history is not None:
            history["train"].append(loss.item())
            if Xval is not None:
                head.eval()
                with torch.no_grad():
                    history["val"].append(
                        chronos2_quantile_loss(_apply_shared_head(head, Xval, Q, P, H), yval, q).item())
                head.train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if history is not None:
        head.eval()
        with torch.no_grad():
            history["train"].append(chronos2_quantile_loss(_apply_shared_head(head, Xtr, Q, P, H), ytr, q).item())
            if Xval is not None:
                history["val"].append(chronos2_quantile_loss(_apply_shared_head(head, Xval, Q, P, H), yval, q).item())
    head.eval()
    return head


def fit_forecast_slot_native_head(train_feats, train_labels, quantiles=CHRONOS2_QUANTILES,
                                  epochs=300, lr=1e-2, weight_decay=1e-3, wd_grid=None, device=None,
                                  hidden_dim=NATIVE_D_FF, dropout=0.0,
                                  output_patch_size=OUTPUT_PATCH_SIZE):
    """Train the per-layer forecast_slot_native_head and RETURN the frozen fitted probe.

    train_feats  : {layer: (n, K, 768)} forecast-slot states (extract_kout_features), 3-D,
                   with K = ceil(H / output_patch_size). One StandardScaler shared across slots.
    train_labels : (n, H) arcsinh trajectory labels. Returns the same frozen-probe dict shape
                   as fit_content_mlp_head (family='forecast_slot_native_head')."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    if Ytr.ndim != 2:
        raise ValueError(f"fit_forecast_slot_native_head needs (n, H) labels, got {Ytr.shape}")
    f0 = train_feats[0]
    if np.ndim(f0) != 3:
        raise ValueError(f"needs (n, K, 768) forecast-slot features, got {np.shape(f0)} "
                         "— use extract_kout_features, not extract_window_features")
    K = f0.shape[1]
    Q, H = len(quantiles), Ytr.shape[1]
    P = int(output_patch_size)
    if K != math.ceil(H / P):
        raise ValueError(f"features carry K={K} slots, but H={H}, P={P} needs K=ceil(H/P)="
                         f"{math.ceil(H / P)} — re-extract with the matching horizon")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(Ytr, device=device)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(Ytr.shape[0])
    n_val = max(1, int(0.2 * Ytr.shape[0]))
    va_np, tr_np = perm[:n_val], perm[n_val:]
    tr = torch.as_tensor(tr_np, dtype=torch.long, device=device)
    va = torch.as_tensor(va_np, dtype=torch.long, device=device)

    fitted = {}
    for i in range(NUM_LAYERS):
        if wd_grid is None:
            wd, selection, src_val = weight_decay, None, None
        else:
            sc_sel = _fit_slot_scaler(train_feats[i][tr_np])
            Xtr_s = torch.as_tensor(_slot_transform(sc_sel, train_feats[i][tr_np]),
                                    dtype=torch.float32, device=device)
            Xva_s = torch.as_tensor(_slot_transform(sc_sel, train_feats[i][va_np]),
                                    dtype=torch.float32, device=device)
            best_wd, best_val, sel = wd_grid[0], float("inf"), {}
            for cand in wd_grid:
                m = _fit_forecast_slot_head(Xtr_s, ytr[tr], q, P, hidden_dim, dropout, cand,
                                            epochs, lr, device)
                with torch.no_grad():
                    v = chronos2_quantile_loss(_apply_shared_head(m, Xva_s, Q, P, H), ytr[va], q).item()
                sel[cand] = v
                if v < best_val:
                    best_val, best_wd = v, cand
            wd, src_val = best_wd, best_val
            selection = {"val_loss_by_wd": {float(k): float(v) for k, v in sel.items()},
                         "chosen_wd": float(best_wd)}

        sc = _fit_slot_scaler(train_feats[i])
        Xtr = torch.as_tensor(_slot_transform(sc, train_feats[i]), dtype=torch.float32, device=device)
        m = _fit_forecast_slot_head(Xtr, ytr, q, P, hidden_dim, dropout, wd, epochs, lr, device)
        fitted[i] = {"scaler": sc, "head": m, "wd": float(wd), "selection": selection,
                     "source_val_loss": (None if src_val is None else float(src_val)),
                     "in_features": int(m.hidden_layer.in_features), "out_features": Q * P,
                     "hidden_dim": int(hidden_dim), "dropout": float(dropout),
                     "family": "forecast_slot_native_head", "pooling_or_token_type": "forecast_slot",
                     "output_patch_size": P, "param_count": head_param_count(m), "device": str(device)}
        print(f"    [fit fslot_native] L{i:>2}  wd={wd:g}  params={fitted[i]['param_count']:,}")
    return fitted


def _fit_converged(val_hist, tail_frac=0.1, rel_tol=1e-3):
    """Diagnostic convergence flag from a fixed-budget per-epoch loss history (val preferred — it is
    eval-mode, so dropout-free and smooth). True iff the loss improved by < rel_tol (relative to the
    final loss) over the last tail_frac of epochs. Training is fixed-epoch (NO early stopping); this
    only flags layers still visibly improving at the budget. Returns None on a too-short history."""
    h = [float(x) for x in val_hist if np.isfinite(x)]
    if len(h) < 3:
        return None
    k = max(1, int(len(h) * tail_frac))
    return bool((h[-1 - k] - h[-1]) / (abs(h[-1]) + 1e-8) < rel_tol)


def fit_forecast_slot_native_head_explicit_val(train_feats, train_labels, val_feats, val_labels,
                                               quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                                               weight_decay=1e-3, wd_grid=None, device=None,
                                               hidden_dim=NATIVE_D_FF, dropout=0.0, init_seed=SEED,
                                               output_patch_size=OUTPUT_PATCH_SIZE):
    """Nonlinear (native-structure ResidualBlock) twin of fit_shared_forecast_probe_explicit_val:
    ONE shared head over the K forecast slots, weight-decay chosen on an EXPLICIT temporal validation
    split, returning the frozen fitted probe. Mirrors fit_content_mlp_head_explicit_val (full-train
    scaler + head, val-only wd score, chosen-wd full-train head kept) but with the slot mechanics
    (_fit_slot_scaler / _slot_transform / _apply_shared_head, K = ceil(H/P)) and the 14-key fslot
    readout. The heads are freshly initialised (build_head + init_seed) — they NEVER load Chronos-2's
    pretrained native-head weights.

    train_feats / val_feats  : {layer: (n, K, 768)} forecast-slot states (extract_kout_features), 3-D.
                               Iterates sorted(train_feats), so the v4 post-final-LN key (NUM_LAYERS)
                               is fit alongside L0..L12; a legacy 13-key dict is handled unchanged.
    train_labels / val_labels: (n, H) arcsinh trajectory labels — raises on 1-D / on 2-D feats.
    init_seed                : head init + dropout stream (the only randomness); default SEED.
    Per layer fitted[i] carries the frozen head plus training diagnostics (§3): selection.{
    val_loss_by_wd, train_loss_by_wd, chosen_wd}; source_val_loss / source_train_loss; a per-epoch
    "history" {train, val} for the chosen wd; final_train_loss / final_val_loss; epochs /
    selected_epoch (= epochs; fixed budget, no early stopping) / lr / dropout / hidden_dim /
    param_count / init_seed / converged. Shape is predict_forecast_slot_native_head-compatible and
    the selection dict matches the linear twin, so save/load + source_selected_layer are unchanged."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    Ytr = np.asarray(train_labels, dtype=np.float32)
    Yva = np.asarray(val_labels, dtype=np.float32)
    if Ytr.ndim != 2 or Yva.ndim != 2:
        raise ValueError("fit_forecast_slot_native_head_explicit_val needs (n, H) trajectory labels "
                         f"for BOTH train and val -- got {Ytr.shape} / {Yva.shape}")
    f0 = train_feats[0]
    if np.ndim(f0) != 3:
        raise ValueError(f"needs (n, K, 768) forecast-slot features, got {np.shape(f0)} "
                         "— use extract_kout_features, not extract_window_features")
    K = f0.shape[1]
    Q, H = len(quantiles), Ytr.shape[1]
    P = int(output_patch_size)
    if K != math.ceil(H / P):
        raise ValueError(f"features carry K={K} slots, but H={H}, P={P} needs K=ceil(H/P)="
                         f"{math.ceil(H / P)} — re-extract with the matching horizon")
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(Ytr, device=device)
    yva = torch.as_tensor(Yva, device=device)

    fitted = {}
    for i in sorted(train_feats):          # 14 keys for fslot v4 (L0..L12 + post-LN); range(13) legacy
        sc = _fit_slot_scaler(train_feats[i])                          # slot-scaler on FULL train only
        Xtr = torch.as_tensor(_slot_transform(sc, train_feats[i]), dtype=torch.float32, device=device)
        Xva = torch.as_tensor(_slot_transform(sc, val_feats[i]), dtype=torch.float32, device=device)
        if wd_grid is None:
            wd, src_val, sel = weight_decay, None, None
        else:
            best_wd, best_val, sel_v, sel_t = wd_grid[0], float("inf"), {}, {}
            for cand in wd_grid:
                cm = _fit_forecast_slot_head(Xtr, ytr, q, P, hidden_dim, dropout, cand, epochs, lr,
                                             device, init_seed=init_seed)              # FULL train
                with torch.no_grad():
                    v = chronos2_quantile_loss(_apply_shared_head(cm, Xva, Q, P, H), yva, q).item()
                    t = chronos2_quantile_loss(_apply_shared_head(cm, Xtr, Q, P, H), ytr, q).item()
                sel_v[cand], sel_t[cand] = v, t
                if v < best_val:
                    best_val, best_wd = v, cand
            wd, src_val = best_wd, best_val
            sel = {"val_loss_by_wd": {float(k): float(x) for k, x in sel_v.items()},
                   "train_loss_by_wd": {float(k): float(x) for k, x in sel_t.items()},
                   "chosen_wd": float(best_wd)}
        # refit the chosen wd once WITH per-epoch history (val-mode -> dropout-free); same init_seed
        # makes this head byte-identical to the grid's chosen candidate, so the stored head and the
        # diagnostic curve's endpoint agree exactly.
        hist = {"train": [], "val": []}
        m = _fit_forecast_slot_head(Xtr, ytr, q, P, hidden_dim, dropout, wd, epochs, lr, device,
                                    Xval=Xva, yval=yva, history=hist, init_seed=init_seed)
        m.eval()
        with torch.no_grad():
            tr_loss = chronos2_quantile_loss(_apply_shared_head(m, Xtr, Q, P, H), ytr, q).item()
            va_loss = chronos2_quantile_loss(_apply_shared_head(m, Xva, Q, P, H), yva, q).item()
        fitted[i] = {"scaler": sc, "head": m, "wd": float(wd), "selection": sel,
                     "source_val_loss": (None if src_val is None else float(src_val)),
                     "source_train_loss": float(tr_loss),
                     "history": {"train": [float(x) for x in hist["train"]],
                                 "val": [float(x) for x in hist["val"]]},
                     "final_train_loss": float(tr_loss), "final_val_loss": float(va_loss),
                     "epochs": int(epochs), "selected_epoch": int(epochs), "lr": float(lr),
                     "converged": _fit_converged(hist["val"] or hist["train"]),
                     "in_features": int(m.hidden_layer.in_features), "out_features": Q * P,
                     "hidden_dim": int(hidden_dim), "dropout": float(dropout), "K": int(K),
                     "output_patch_size": P, "family": "forecast_slot_native_head",
                     "pooling_or_token_type": "forecast_slot", "param_count": head_param_count(m),
                     "init_seed": int(init_seed), "device": str(device)}
        vtxt = "n/a" if src_val is None else f"{src_val:.3f}"
        print(f"    [fit-explicit-val fslot_native] L{i:>2}  wd={wd:g}  train={tr_loss:.3f}  "
              f"val={vtxt}  params={fitted[i]['param_count']:,}  conv={fitted[i]['converged']}")
    return fitted


def predict_forecast_slot_native_head(fitted, feats, labels, quantiles=CHRONOS2_QUANTILES,
                                      device=None, collect_test_median=False,
                                      collect_test_window_loss=False,
                                      output_patch_size=OUTPUT_PATCH_SIZE):
    """Apply a FROZEN forecast_slot_native_head to feats/labels; NEVER trains. Same diag
    contract as predict_quantile_probe. Deterministic and side-effect-free."""
    quantiles = validate_quantiles(quantiles)
    Yte = np.asarray(labels, dtype=np.float32)
    if Yte.ndim != 2:
        raise ValueError(f"predict_forecast_slot_native_head needs (n, H) labels, got {Yte.shape}")
    f0 = feats[0]
    if np.ndim(f0) != 3:
        raise ValueError(f"needs (n, K, 768) forecast-slot features, got {np.shape(f0)}")
    K = f0.shape[1]
    Q, H = len(quantiles), Yte.shape[1]
    P = int(output_patch_size)
    if K != math.ceil(H / P):
        raise ValueError(f"features carry K={K} slots, but H={H}, P={P} needs K=ceil(H/P)="
                         f"{math.ceil(H / P)}")
    q_mid = median_index(quantiles)
    if collect_test_median and q_mid is None:
        raise ValueError(f"collect_test_median needs the 0.5 level, got {quantiles.tolist()}")
    out = {}
    diag = ({"test_mean_pinball": {}, "test_median": {}, "test_window_loss": {}}
            if (collect_test_median or collect_test_window_loss) else None)
    for i in sorted(feats):                # sorted keys: 14 for fslot v4 (L0..L12 + post-LN), 13 legacy
        dev = device or fitted[i]["device"]
        m = fitted[i]["head"].to(dev)
        m.eval()
        sc = fitted[i]["scaler"]
        q_t = torch.as_tensor(quantiles, dtype=torch.float32, device=dev)
        yte = torch.as_tensor(Yte, device=dev)
        Xte = torch.as_tensor(_slot_transform(sc, feats[i]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = _apply_shared_head(m, Xte, Q, P, H)
            out[i] = float(chronos2_quantile_loss(pred, yte, q_t).item())
            if diag is not None:
                diag["test_mean_pinball"][i] = float(mean_pinball_loss(pred, yte, q_t).item())
            if collect_test_median:
                diag["test_median"][i] = pred[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred, yte, q_t).cpu().numpy().astype(np.float64)
    return (out, diag) if diag is not None else out


# ======================= NEW PROBES GO HERE (collaborators) ======================= #
# Implement each with the standard 4-arg signature and register it in PROBES below.
# Raise NotImplementedError until filled in so the registry stays importable.

def effective_rank(train_feats, train_labels, test_feats, test_labels):
    """TODO: effective rank (e.g. entropy of normalized singular values) of each
    layer's TEST feature matrix. Unsupervised — ignore the labels."""
    raise NotImplementedError("effective_rank probe not implemented yet")


def entropy(train_feats, train_labels, test_feats, test_labels):
    """TODO: representation entropy per layer (unsupervised)."""
    raise NotImplementedError("entropy probe not implemented yet")


def epiplexity(train_feats, train_labels, test_feats, test_labels):
    """TODO: epiplexity per layer."""
    raise NotImplementedError("epiplexity probe not implemented yet")


# ================================ probe registry ================================= #
# name -> callable(train_feats, train_labels, test_feats, test_labels) -> {layer: score}
PROBES = {
    "linear": linear_probe,                       # UEA classification (reference)
    "ridge_regression": ridge_regression_probe,   # ID forecasting, R^2
    "binned_future": binned_future_probe,         # ID forecasting, accuracy (primary readout)
    "quantile": quantile_probe,                   # ID forecasting, Chronos-2 quantile loss (lower=better)
    "shared_forecast": shared_forecast_token_probe,  # Chronos-aligned shared forecast-token readout
    # "effective_rank": effective_rank,           # uncomment once implemented
    # "entropy": entropy,
    # "epiplexity": epiplexity,
}
