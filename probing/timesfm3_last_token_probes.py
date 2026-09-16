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
``WD_GRID_V2``, ``SEED``, the 80/20 carve of TRAIN WINDOWS, StandardScaler on the carve for
selection then refit on full train, AdamW with weight decay on the WEIGHT only, full-batch
300 epochs.

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
from probing.probes import WD_GRID_V2, validate_quantiles
from probing.timesfm3_last_token import (LAYER_NAMES, MODEL_DIMS, NATIVE_MEDIAN_IDX,
                                         NATIVE_QUANTILES, NUM_LAYERS, NUM_QUANTILES,
                                         safe_sigma)
from probing.tunnel import tunnel_start

__all__ = ["NATIVE_QUANTILES", "NUM_QUANTILES", "make_probe", "reshape_prediction",
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
                         quantiles=NATIVE_QUANTILES, epochs: int = 300, lr: float = 1e-2,
                         wd_grid=WD_GRID_V2, weight_decay: float = 1e-3, device=None,
                         batch_size: int = 0, layers=None, collect_history: bool = False,
                         verbose: bool = True, model_for_grad_check=None):
    """One independent probe per representation point. Returns ({layer: test Q9 loss}, diag).

    Per layer:
      1. SELECT weight decay on the 80/20 carve -- scaler AND probe fit on the 80% only, scored
         on the held-out 20% with the Q=9 objective. ``diag["val_loss"]`` is that number; it is
         the curve the 5% tunnel entrance is computed from, and it never saw the test split.
      2. REFIT scaler + probe on ALL valid train windows with the selected weight decay.
      3. SCORE on test: Q=9 loss (+ per-window for the bootstrap), median-only loss, the median
         prediction (for MASE, un-transformed by the driver), per-quantile losses.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    q_np = validate_quantiles(quantiles)
    Q = len(q_np)
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

    va_w, tr_w = _carve(len(train_targets))
    sel_tr = tr_w[train_valid[tr_w]]
    sel_va = va_w[train_valid[va_w]]
    all_tr = np.flatnonzero(train_valid)
    te_rows = np.flatnonzero(test_valid)
    for name, rows in (("carve-train", sel_tr), ("carve-val", sel_va), ("test", te_rows)):
        if rows.size == 0:
            raise RuntimeError(f"no valid window left in the {name} split")

    yte = torch.as_tensor(test_targets[te_rows], device=device)
    ytr_all = torch.as_tensor(train_targets[all_tr], device=device)
    ytr_sel = torch.as_tensor(train_targets[sel_tr], device=device)
    yva_sel = torch.as_tensor(train_targets[sel_va], device=device)

    out: dict[int, float] = {}
    diag: dict = {"wd": {}, "selection": {}, "val_loss": {}, "wd_at_grid_edge": {},
                  "history": {}, "train_loss": {}, "test_q9": {}, "test_q9_window": {},
                  "test_median_loss": {}, "test_median_window": {}, "test_median_pred": {},
                  "test_per_quantile": {}, "n_train_rows": {}, "n_val_rows": int(sel_va.size),
                  "n_test_rows": int(te_rows.size), "test_rows": te_rows,
                  "carve_train_windows": sel_tr, "carve_val_windows": sel_va,
                  "layers": layers, "layer_names": [LAYER_NAMES[i] for i in layers],
                  "quantiles": q_np.tolist(), "num_quantiles": Q}

    for i in layers:
        F_tr, F_te = train_feats[i], test_feats[i]
        for nm, F in (("train", F_tr), ("test", F_te)):
            F = np.asarray(F)
            if F.ndim != 2 or F.shape[1] != MODEL_DIMS:
                raise RuntimeError(f"layer {i} {nm} features are {F.shape}; this experiment "
                                   f"needs (n, {MODEL_DIMS}) last-token states")

        # ---- 1. weight decay on the carve (scaler fit on the 80% only) ----
        Xs = np.asarray(F_tr[sel_tr], np.float32)
        sc_sel = StandardScaler().fit(Xs)
        Xtr_s = torch.as_tensor(sc_sel.transform(Xs), dtype=torch.float32, device=device)
        Xva_s = torch.as_tensor(sc_sel.transform(np.asarray(F_tr[sel_va], np.float32)),
                                dtype=torch.float32, device=device)
        if wd_grid is None:
            wd, sel = weight_decay, None
            m = fit_last_token_probe(Xtr_s, ytr_sel, q, H, wd, epochs, lr, device,
                                     batch_size=batch_size)
            with torch.no_grad():
                val = float(pinball_loss(reshape_prediction(m(Xva_s), H, Q), yva_sel, q).item())
        else:
            best, wd, sel = float("inf"), wd_grid[0], {}
            for cand in wd_grid:
                m = fit_last_token_probe(Xtr_s, ytr_sel, q, H, cand, epochs, lr, device,
                                         batch_size=batch_size)
                with torch.no_grad():
                    v = float(pinball_loss(reshape_prediction(m(Xva_s), H, Q),
                                           yva_sel, q).item())
                sel[float(cand)] = v
                if v < best:
                    best, wd = v, cand
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
        del Xtr_s, Xva_s

        # ---- 2. refit on ALL valid train windows ----
        Xa = np.asarray(F_tr[all_tr], np.float32)
        sc = StandardScaler().fit(Xa)
        Xtr = torch.as_tensor(sc.transform(Xa), dtype=torch.float32, device=device)
        diag["n_train_rows"][i] = int(Xtr.shape[0])
        lin = fit_last_token_probe(Xtr, ytr_all, q, H, wd, epochs, lr, device,
                                   batch_size=batch_size,
                                   model_for_grad_check=model_for_grad_check)
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
            mpw = median_pinball_per_window(pred, yte, NATIVE_MEDIAN_IDX)
            diag["test_median_loss"][i] = float(mpw.mean().item())
            diag["test_median_window"][i] = mpw.cpu().numpy().astype(np.float64)
            diag["test_median_pred"][i] = pred[:, NATIVE_MEDIAN_IDX, :].cpu().numpy(
                ).astype(np.float32)
            diag["test_per_quantile"][i] = per_quantile_loss(pred, yte, q)
            del Xte, pred
        if verbose:
            print(f"    [{LAYER_NAMES[i]:>3}] wd={wd:<6g} rows={diag['n_train_rows'][i]:>5}  "
                  f"train={diag['train_loss'][i]:.5f}  val={val:.5f}  "
                  f"test(Q9)={out[i]:.5f}  test(median)={diag['test_median_loss'][i]:.5f}",
                  flush=True)
    return out, diag


# --------------------------------------------------------------------------- #
# native baseline, scored on the probes' own axis
# --------------------------------------------------------------------------- #

def native_reference(native_raw, mu, sd, trend, targets, valid, quantiles=NATIVE_QUANTILES,
                     device=None) -> dict:
    """The native TimesFM-3 forecast, normalized into the probe target space and scored there.

    decode() returns RAW units; dividing out the SAME (trend, mu, sigma) the probe targets use
    puts all 9 native quantiles on the probes' axis, so the native Q=9 loss is directly
    comparable to every layer's. Returns the losses, the per-window vectors (for the cluster
    bootstrap) and the raw median forecast (for MASE).
    """
    device = device or "cpu"
    q_np = validate_quantiles(quantiles)
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
        mpw = median_pinball_per_window(pred, y, NATIVE_MEDIAN_IDX).cpu().numpy().astype(
            np.float64)
        per_q = per_quantile_loss(pred, y, q)
    return {"q9_loss": float(pw.mean()), "q9_window": pw,
            "median_loss": float(mpw.mean()), "median_window": mpw,
            "per_quantile": per_q, "median_raw": nat[:, :, NATIVE_MEDIAN_IDX],
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
