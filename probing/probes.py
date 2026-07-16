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


def chronos2_quantile_loss(pred, target, q):
    """Chronos-2's quantile (pinball) loss, formula + reduction verbatim from
    chronos/chronos2/model.py:551,564.

        pred   : (B, Q, H) predicted quantiles
        target : (B, H)    true trajectory   (broadcast to (B, 1, H))
        q      : (Q,)      quantile levels    (broadcast to (1, Q, 1))

    loss = 2*|(y - q̂)(1[y<=q̂] - τ)|, reduced mean(horizon) -> sum(quantiles) -> mean(batch).
    """
    target = target.unsqueeze(1)                                       # (B, 1, H)
    qv = q.view(1, -1, 1)                                              # (1, Q, 1)
    ql = 2.0 * torch.abs((target - pred) * ((target <= pred).to(pred.dtype) - qv))
    return ql.mean(dim=-1).sum(dim=-1).mean()


def chronos2_quantile_loss_per_window(pred, target, q):
    """Per-window Chronos-2 quantile loss: identical formula and mean(horizon) -> sum(quantiles)
    reduction as ``chronos2_quantile_loss``, but WITHOUT the final batch mean. Returns (B,);
    its ``.mean()`` is the reported scalar loss (same op chain), which is what lets the
    series-level cluster bootstrap resample test windows post hoc without refitting anything."""
    target = target.unsqueeze(1)                                       # (B, 1, H)
    qv = q.view(1, -1, 1)                                              # (1, Q, 1)
    ql = 2.0 * torch.abs((target - pred) * ((target <= pred).to(pred.dtype) - qv))
    return ql.mean(dim=-1).sum(dim=-1)


def _fit_quantile_linear(Xtr, ytr, q, weight_decay, epochs, lr, device,
                         Xval=None, yval=None, history=None):
    """Fit one strictly-linear map (d -> Q*H) with Chronos-2 loss; return the trained module.
    Re-seeded each call so every layer / weight_decay candidate starts from the same init.

    AdamW with decay on the WEIGHT only: the pinball-optimal bias is the target's quantile
    vector itself, so decaying the bias would shrink every predicted quantile toward 0 (and
    plain Adam's weight_decay is warped by the adaptive scaling, making grid values
    incomparable across layers).

    If `history` (a dict with "train"/"val" lists) is passed, per-epoch train loss (and val
    loss when Xval/yval are given) is appended BEFORE each update, plus once more after the
    loop -> length epochs+1 (init ... converged), for the training-curve diagnostic.
    history=None leaves the original loop exactly unchanged (behavior-preserving)."""
    Q, H = len(q), ytr.shape[1]
    torch.manual_seed(SEED)
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
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    diag = ({"wd": {}, "selection": {}, "history": {}, "test_median": {}, "test_window_loss": {}}
            if (collect_history or collect_test_median or collect_test_window_loss) else None)
    q_mid = int(np.argmin(np.abs(np.asarray(quantiles, dtype=np.float64) - 0.5)))  # 0.5 row
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
            if collect_test_median:
                diag["test_median"][i] = pred_te[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred_te, yte, q).cpu().numpy().astype(np.float64)
        print(f"    [quantile] L{i:>2}  wd={wd:g}  train={train_loss:.3f}  test={out[i]:.3f}")
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
                                Xval=None, yval=None, history=None):
    """Fit the shared-slot linear head with Chronos-2's quantile loss. Same optimizer convention
    as _fit_quantile_linear (AdamW, decay on weight only; re-seeded each call). The head predicts
    K whole P-step patches; loss is computed on the first H = ytr.shape[1] steps (trimmed)."""
    Q, H = len(q), ytr.shape[1]
    torch.manual_seed(SEED)
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
    diag = ({"wd": {}, "selection": {}, "history": {}, "test_median": {}, "test_window_loss": {}}
            if (collect_history or collect_test_median or collect_test_window_loss) else None)
    q_mid = int(np.argmin(np.abs(np.asarray(quantiles, dtype=np.float64) - 0.5)))
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
            if collect_test_median:
                diag["test_median"][i] = pred_te[:, q_mid, :].cpu().numpy().astype(np.float32)
            if collect_test_window_loss:
                diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                    pred_te, yte, q).cpu().numpy().astype(np.float64)
        print(f"    [fslot] L{i:>2}  wd={wd:g}  train={train_loss:.3f}  test={out[i]:.3f}")
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
