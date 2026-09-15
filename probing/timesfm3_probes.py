"""Shared linear probe across TimesFM-3's independent causal forecast origins.  Q=1, tau=0.5.

One ``Linear(1280, 64)`` per representation point, applied with the SAME weight and bias to
each of the J=16 origins:

    (B, 16, 1280) -> (B*16, 1280) -> Linear(1280, 64) -> (B*16, 64) -> (B, 16, 64)

No cross-origin mixing anywhere, and the 16 outputs are NEVER concatenated into one forecast:
they are 16 different forecasting problems with 16 different origins.

    probe TRAINING     : every valid origin 1..16
    headline VAL/TEST  : origin 16 only (C=512 -> H=64, the real task)

Reused unchanged from the Chronos-2 pipeline so the two model lines stay comparable:
``chronos2_quantile_loss`` (objective), ``chronos2_quantile_loss_per_window`` (bootstrap
inputs), ``mean_pinball_loss`` (reporting), ``WD_GRID_V2``, ``SEED``, the 80/20 carve, and
AdamW with weight decay on the weight only.

At Q=1 / tau=0.5 the Chronos-2 objective reduces to MAE:
    2*|(y-yhat) * (1[y<=yhat] - 0.5)| = |y - yhat|
and ``mean_pinball_loss`` = 0.5*MAE, which is exactly the reported 1/(J*H) sum rho_0.5.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from probing.config import SEED
from probing.probes import (WD_GRID_V2, chronos2_quantile_loss,
                            chronos2_quantile_loss_per_window, mean_pinball_loss)
from probing.timesfm3 import LAYER_NAMES, MODEL_DIMS, NUM_LAYERS

MEDIAN_QUANTILE = np.array([0.5], dtype=np.float32)      # Q = 1, tau = 0.5
Q = 1

__all__ = ["MEDIAN_QUANTILE", "Q", "make_probe", "apply_probe_origins", "flatten_valid",
           "fit_origin_scaler", "origin_transform", "fit_shared_origin_probe",
           "shared_origin_layerwise", "native_median_reference", "median_pinball"]


def make_probe(H: int, device="cpu", seed: int = SEED) -> torch.nn.Linear:
    """THE probe: exactly one Linear(1280, H) module, shared across all origins."""
    torch.manual_seed(seed)
    lin = torch.nn.Linear(MODEL_DIMS, H).to(device)
    assert lin.in_features == MODEL_DIMS and lin.out_features == H, (
        f"probe must be Linear({MODEL_DIMS}, {H}), got "
        f"Linear({lin.in_features}, {lin.out_features})")
    return lin


def apply_probe_origins(lin, X_bjd):
    """(B, J, 1280) -> (B, J, H) with ONE shared weight; no origin ever sees another."""
    X = torch.as_tensor(X_bjd)
    if X.ndim != 3 or X.shape[-1] != MODEL_DIMS:
        raise ValueError(f"expected (B, J, {MODEL_DIMS}) origin states, got {tuple(X.shape)}")
    B, J, d = X.shape
    out = lin(X.reshape(B * J, d)).view(B, J, lin.out_features)
    assert out.shape == (B, J, lin.out_features), (
        f"prediction shape {tuple(out.shape)} != (B, J, H) = {(B, J, lin.out_features)}")
    return out


def median_pinball(pred_flat, target_flat, q=None):
    """Chronos-2's objective at Q=1. pred (rows, H) -> (rows, 1, H) for the shared contract."""
    q = torch.as_tensor(MEDIAN_QUANTILE if q is None else q,
                        dtype=torch.float32, device=pred_flat.device)
    assert q.numel() == 1 and float(q.item()) == 0.5, f"Q must be 1 at tau=0.5, got {q.tolist()}"
    return chronos2_quantile_loss(pred_flat.unsqueeze(1), target_flat, q)


def flatten_valid(F_bjd, T_bjh, V_bj):
    """Valid (window, origin) rows only. Row i*J+j in the flattened order; returns the index."""
    F = np.asarray(F_bjd)
    n, J, d = F.shape
    idx = np.flatnonzero(np.asarray(V_bj).reshape(n * J))
    X = F.astype(np.float32).reshape(n * J, d)[idx]
    Y = np.asarray(T_bjh, np.float32).reshape(n * J, T_bjh.shape[-1])[idx]
    return X, Y, idx


def fit_origin_scaler(X):
    """ONE StandardScaler shared across origins (twin of probes._fit_slot_scaler). float32:
    sklearn preserves dtype and a float64 upcast would cost ~0.5 GB per copy at 48k x 1280."""
    return StandardScaler().fit(np.asarray(X, np.float32))


def origin_transform(sc, X):
    return sc.transform(np.asarray(X, np.float32))


def fit_shared_origin_probe(Xtr, ytr, H, weight_decay, epochs, lr, device,
                            Xval=None, yval=None, history=None, init_seed=SEED,
                            batch_size=0, model_for_grad_check=None):
    """Fit the shared probe: ONE loss over all origin rows, ONE backward, ONE step per epoch.

    Same optimizer convention as probes._fit_shared_forecast_linear (AdamW, decay on the
    weight only, re-seeded per call). ``batch_size=0`` = full batch, the Chronos-2 protocol.
    """
    lin = make_probe(H, device, init_seed)
    opt = torch.optim.AdamW(
        [{"params": [lin.weight], "weight_decay": weight_decay},
         {"params": [lin.bias], "weight_decay": 0.0}], lr=lr)
    n = Xtr.shape[0]
    full = batch_size <= 0 or batch_size >= n
    gen = torch.Generator(device="cpu").manual_seed(init_seed)

    def ev(X, y):
        with torch.no_grad():
            return median_pinball(lin(X), y).item()

    lin.train()
    for _ in range(epochs):
        if full:
            loss = median_pinball(lin(Xtr), ytr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if model_for_grad_check is not None:
                from probing.timesfm3 import assert_no_backbone_grads
                assert_no_backbone_grads(model_for_grad_check)
            opt.step()
        else:
            perm = torch.randperm(n, generator=gen).to(Xtr.device)
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                loss = median_pinball(lin(Xtr[b]), ytr[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if history is not None:
            history["train"].append(ev(Xtr, ytr))
            if Xval is not None:
                history["val"].append(ev(Xval, yval))
    lin.eval()
    return lin


def shared_origin_layerwise(train_feats, train_targets, train_valid,
                            test_feats, test_targets, test_valid, *,
                            H, epochs=300, lr=1e-2, wd_grid=WD_GRID_V2, weight_decay=1e-3,
                            device=None, batch_size=0, layers=None, headline_origin_idx=None,
                            collect_history=False, verbose=True):
    """Per representation point: fit on ALL valid origins, report the HEADLINE origin only.

    Returns ({layer: headline test loss}, diag). The weight decay and L* are selected on the
    headline origin's VALIDATION loss (never test) -- the task actually being reported.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_tr, J, Hc = train_targets.shape
    if Hc != H:
        raise ValueError(f"targets have horizon {Hc}, expected {H}")
    hz = J - 1 if headline_origin_idx is None else headline_origin_idx
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(layers)

    rng = np.random.default_rng(SEED)                    # carve WINDOWS, not rows: a window's
    perm = rng.permutation(n_tr)                         # 16 origins share a series and overlap
    n_val = max(1, int(0.2 * n_tr))
    va_w, tr_w = np.sort(perm[:n_val]), np.sort(perm[n_val:])

    te_rows = np.flatnonzero(test_valid[:, hz])
    if len(te_rows) == 0:
        raise RuntimeError("no test window has a valid headline origin")
    yte = torch.as_tensor(test_targets[te_rows, hz, :], dtype=torch.float32, device=device)

    out, diag = {}, {"wd": {}, "selection": {}, "val_loss": {}, "history": {},
                     "test_median_pred": {}, "test_window_loss": {}, "test_mean_pinball": {},
                     "test_loss_all_origins": {}, "test_loss_by_origin": {},
                     "n_train_rows": {}, "headline_origin_idx": hz, "test_rows": te_rows,
                     "layers": layers, "layer_names": [LAYER_NAMES[i] for i in layers]}

    for i in layers:
        Ftr, Fte = train_feats[i], test_feats[i]
        # ---- weight-decay selection on the HEADLINE origin's validation loss ----
        Xs, Ys, _ = flatten_valid(np.asarray(Ftr[tr_w]), train_targets[tr_w], train_valid[tr_w])
        sc_sel = fit_origin_scaler(Xs)
        Xtr_s = torch.as_tensor(origin_transform(sc_sel, Xs), device=device)
        ytr_s = torch.as_tensor(Ys, device=device)
        ok = np.flatnonzero(train_valid[va_w, hz])
        Xv = torch.as_tensor(origin_transform(sc_sel, np.asarray(Ftr[va_w])[ok, hz, :]),
                             device=device)
        yv = torch.as_tensor(np.asarray(train_targets[va_w][ok, hz, :], np.float32),
                             device=device)
        if wd_grid is None:
            wd, sel = weight_decay, None
        else:
            best, wd, sel = float("inf"), wd_grid[0], {}
            for cand in wd_grid:
                m = fit_shared_origin_probe(Xtr_s, ytr_s, H, cand, epochs, lr, device,
                                            batch_size=batch_size)
                with torch.no_grad():
                    v = median_pinball(m(Xv), yv).item()
                sel[float(cand)] = v
                if v < best:
                    best, wd = v, cand
        diag["wd"][i] = float(wd)
        diag["selection"][i] = None if sel is None else {"val_loss_by_wd": sel,
                                                         "chosen_wd": float(wd)}
        diag["val_loss"][i] = None if sel is None else float(sel[float(wd)])
        if collect_history:
            hist = {"train": [], "val": []}
            fit_shared_origin_probe(Xtr_s, ytr_s, H, wd, epochs, lr, device, Xval=Xv, yval=yv,
                                    history=hist, batch_size=batch_size)
            diag["history"][i] = hist
        del Xtr_s, ytr_s, Xv, yv

        # ---- refit on ALL train windows / ALL valid origins ----
        Xa, Ya, _ = flatten_valid(np.asarray(Ftr), train_targets, train_valid)
        sc = fit_origin_scaler(Xa)
        Xtr = torch.as_tensor(origin_transform(sc, Xa), device=device)
        ytr = torch.as_tensor(Ya, device=device)
        diag["n_train_rows"][i] = int(Xtr.shape[0])
        lin = fit_shared_origin_probe(Xtr, ytr, H, wd, epochs, lr, device,
                                      batch_size=batch_size)
        with torch.no_grad():
            tr_loss = median_pinball(lin(Xtr), ytr).item()
        del Xtr, ytr

        with torch.no_grad():
            # diagnostic: all origins, and per-origin
            Xall, Yall, _ = flatten_valid(np.asarray(Fte), test_targets, test_valid)
            Xa_t = torch.as_tensor(origin_transform(sc, Xall), device=device)
            diag["test_loss_all_origins"][i] = float(
                median_pinball(lin(Xa_t), torch.as_tensor(Yall, device=device)).item())
            del Xa_t
            per = {}
            for jj in range(J):
                rows = np.flatnonzero(test_valid[:, jj])
                if len(rows) == 0:
                    continue
                Xj = torch.as_tensor(origin_transform(sc, np.asarray(Fte)[rows, jj, :]),
                                     device=device)
                yj = torch.as_tensor(test_targets[rows, jj, :], dtype=torch.float32,
                                     device=device)
                per[jj + 1] = float(median_pinball(lin(Xj), yj).item())   # 1-based key
                del Xj, yj
            diag["test_loss_by_origin"][i] = per

            # HEADLINE: origin hz only
            Xh = torch.as_tensor(origin_transform(sc, np.asarray(Fte)[te_rows, hz, :]),
                                 device=device)
            pred = lin(Xh)
            out[i] = float(median_pinball(pred, yte).item())
            diag["test_mean_pinball"][i] = float(
                mean_pinball_loss(pred.unsqueeze(1), yte,
                                  torch.as_tensor(MEDIAN_QUANTILE, device=device)).item())
            diag["test_median_pred"][i] = pred.cpu().numpy().astype(np.float32)
            diag["test_window_loss"][i] = chronos2_quantile_loss_per_window(
                pred.unsqueeze(1), yte,
                torch.as_tensor(MEDIAN_QUANTILE, device=device)).cpu().numpy().astype(np.float64)
            del Xh, pred
        if verbose:
            print(f"    [{LAYER_NAMES[i]:>3}] wd={wd:<6g} rows={diag['n_train_rows'][i]:>6}  "
                  f"train={tr_loss:.4f}  test(origin {hz+1})={out[i]:.4f}  "
                  f"test(all origins)={diag['test_loss_all_origins'][i]:.4f}", flush=True)
    return out, diag


def native_median_reference(native_raw, mu, sd, trend, targets, valid, hz, device=None):
    """Native TimesFM-3 tau=0.5 forecast, scored on the probes' axis.

    decode() returns RAW units; normalizing with the headline origin's (mu, sigma, trend) puts
    it in the probe target space so its loss is directly comparable to every layer's.
    Returns (loss, per_window_loss, mean_pinball, median_raw).
    """
    from probing.timesfm3 import NATIVE_MEDIAN_IDX, safe_sigma
    device = device or "cpu"
    rows = np.flatnonzero(valid[:, hz])
    med_raw = np.asarray(native_raw, np.float64)[rows, :, NATIVE_MEDIAN_IDX]      # (n, H)
    z = (med_raw - trend[rows, hz, :] - mu[rows, hz][:, None]) / safe_sigma(sd[rows, hz])[:, None]
    pred = torch.as_tensor(z, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets[rows, hz, :], dtype=torch.float32, device=device)
    q = torch.as_tensor(MEDIAN_QUANTILE, device=device)
    with torch.no_grad():
        loss = float(median_pinball(pred, y).item())
        pw = chronos2_quantile_loss_per_window(pred.unsqueeze(1), y, q).cpu().numpy().astype(
            np.float64)
        mp = float(mean_pinball_loss(pred.unsqueeze(1), y, q).item())
    return loss, pw, mp, med_raw
