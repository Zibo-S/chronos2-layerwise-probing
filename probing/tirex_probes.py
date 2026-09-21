"""TiRex layer-wise linear forecasting probe.  Q = 1, tau = 0.5 (median).  SHARED patch head.

ONE SHARED ``Linear(512, 32)`` per representation depth, applied IDENTICALLY to both native
forecast-producing states of the same window:

    h_1 = h_{l, token 63}  (last REAL context patch)    ->  yhat[T   : T+32]
    h_2 = h_{l, token 64}  (first MASKED future patch)  ->  yhat[T+32: T+64]
    yhat = concat(q_l(h_1), q_l(h_2))                   ->  (B, 64)

The sharing is STRUCTURAL, not a convention: the probe is a single ``nn.Linear`` applied to a
(B, K, 512) tensor, so there is exactly one weight tensor and it cannot differ between positions.
This mirrors TiRex's own interface -- ``output_patch_embedding`` is likewise one head applied to
every token -- which is why it, and not a Linear(512, 64) from one state or a pooled/concatenated
readout, is the right linear analogue of the native pathway.

What this probe deliberately is NOT:
    * not a separate head per position          (would double capacity and break the analogy)
    * not Linear(512, 64) from the last context state alone
    * not a mean-pooled or concatenated readout
    * not a 9-quantile head -- TiRex's headline is Q=1, tau=0.5

OBJECTIVE.  ``probes.mean_pinball_loss`` with q = [0.5], which is documented there to equal
0.5 * MAE.  Tau=0.5 pinball and MAE therefore differ by a constant factor of exactly 2, so every
argmin -- the weight-decay selection AND the tunnel entrance, which compares ratios against the
final depth -- is identical under either. ``tau_half_is_half_mae`` proves it numerically and the
test module asserts the tunnel invariance.

PROTOCOL, reused verbatim from the Chronos-2 / TimesFM-3 lines so probe capacity and training
stay comparable across the three models (``probes._fit_quantile_linear``):
    SEED, full-batch AdamW, lr 1e-2, 300 epochs, weight decay on the WEIGHT ONLY (the pinball-
    optimal bias IS the target's quantile, so decaying it would shrink the forecast toward 0),
    a weight decay selected on an EXPLICIT validation split with NO refit, and a
    ``StandardScaler`` fit on TRAIN ONLY. The grid is ``WD_GRID_TIREX`` = ``probes.WD_GRID_V2``
    extended by 10/30 -- see that constant for the measurement that forced the extension.

ONE SCALER, NOT TWO.  The scaler is fit on the STACKED (2N, 512) train rows, never per position.
A per-position scaler would place a different affine map in front of each position and the head
would no longer be shared in any meaningful sense -- ``fit_shared_patch_probe`` enforces this and
test 14/11 pin it.

The 5% tunnel entrance reuses ``probing.tunnel.tunnel_start`` -- the project's existing
first-crossing rule -- so the criterion is byte-identical to the other two models'.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from probing.config import SEED
from probing.probes import WD_GRID_V2, mean_pinball_loss, validate_quantiles
from probing.tirex_model import MODEL_DIMS, NUM_POINTS, REP_NAMES, REP_POINTS
from probing.tunnel import TUNNEL_TOL, tunnel_start

__all__ = ["TAU", "QUANTILES_Q1", "WD_GRID_TIREX", "WD_DECOUPLED_LIMIT", "assert_wd_grid",
           "EPOCHS", "LR", "make_shared_patch_probe",
           "probe_forward", "pinball_tau_half", "tau_half_is_half_mae", "fit_shared_patch_probe",
           "layerwise_probe", "native_median_reference", "tunnel_entrance", "tunnel_entrances",
           "per_patch_losses", "representation_norms", "constant_forecast_floor", "predict",
           "per_window_loss"]

TAU = 0.5
QUANTILES_Q1 = tuple(float(x) for x in validate_quantiles((0.5,)))  # Q=1: the median row
                                      # only. NOT TiRex's native 9. Validated at import by the
                                      # project-wide checker (strictly inside (0,1), increasing).
EPOCHS = 300
LR = 1e-2

# SELECTION GRID: probes.WD_GRID_V2 (the Chronos-2 grid) extended upward by 10 and 30 -- the SAME
# extension, for the same reason and with the same ceiling, that WD_GRID_LAST_TOKEN applies on the
# TimesFM-3 line. Defined as a superset so the lineage is explicit and WD_GRID_V2 stays untouched:
#     WD_GRID_V2     = 1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3
#     WD_GRID_TIREX  = ... + 10 30
#
# WHY IT HAD TO BE EXTENDED -- measured, not predicted. An earlier version of this file argued the
# unextended grid would suffice here because Linear(512, 32) = 16,416 parameters on 2*1394 = 2788
# train rows is a far better row/parameter ratio than TimesFM-3's 737k on 1394. THAT REASONING WAS
# WRONG. On the full Electricity run (1394/262/262, 300 epochs, lr 1e-2) 12 of the 14 depths
# selected wd = 3, the old grid MAXIMUM, with train loss far below validation at every depth
# (e.g. L12+RMS: train 0.113, val 0.129) -- a clipped grid, not a converged selection. The ratio
# argument ignored that these are 2788 rows of a bfloat16 recurrent state whose effective
# dimension is far below 512 (measured entropy effective rank ~7), so the fit is much closer to
# degenerate than the raw row count suggests.
#
# WHY IT STOPS AT 30. AdamW's decay is DECOUPLED: each step multiplies the weight by (1 - lr*wd),
# so at lr = 1e-2 the candidate wd = 100 gives lr*wd = 1 and zeroes the weight EVERY step (the fit
# collapses to a bias-only predictor, i.e. the marginal-quantile forecast), and wd = 300 gives
# |1 - lr*wd| > 1, which amplifies the weight with alternating sign. Selecting either would report
# an optimizer artifact as a probe, so `assert_wd_grid` REFUSES any candidate with lr*wd >= 1.
# The closed-form no-information reference is `constant_forecast_floor`, which needs no fit at all.
WD_GRID_TIREX = tuple(WD_GRID_V2) + (10.0, 30.0)
WD_DECOUPLED_LIMIT = 1.0              # refuse candidates with lr * wd >= this


def assert_wd_grid(wd_grid, lr: float = LR) -> tuple:
    """Refuse weight decays that leave AdamW's well-behaved decoupled-decay regime (lr*wd >= 1).
    Such a candidate does not regularize a fit, it destroys it -- see the constant above."""
    bad = [w for w in wd_grid if float(w) * lr >= WD_DECOUPLED_LIMIT]
    if bad:
        raise ValueError(
            f"weight-decay candidates {bad} give lr*wd >= {WD_DECOUPLED_LIMIT} at lr={lr:g}: "
            "AdamW's decoupled decay would zero (or blow up) the weight every step, so selecting "
            "one would report an optimizer artifact as a probe. Use constant_forecast_floor() for "
            "the no-information reference instead.")
    if not len(tuple(wd_grid)):
        raise ValueError("the weight-decay grid is empty")
    return tuple(float(w) for w in wd_grid)


# --------------------------------------------------------------------------- #
# the probe
# --------------------------------------------------------------------------- #
def make_shared_patch_probe(d: int = MODEL_DIMS, out_patch: int = 32, device="cpu",
                            seed: int = SEED, bias: bool = True) -> torch.nn.Linear:
    """ONE ``Linear(d, out_patch)``. Bias ON: the existing project convention is a biased linear
    probe with the bias excluded from weight decay (probes._fit_quantile_linear)."""
    torch.manual_seed(seed)
    return torch.nn.Linear(d, out_patch, bias=bias).to(device)


def probe_forward(probe: torch.nn.Linear, X: torch.Tensor) -> torch.Tensor:
    """(B, K, d) -> (B, 1, K*out_patch). ONE weight tensor, applied to every position.

    The (B, 1, H) layout is ``probes._check_pred_shape``'s (B, Q, H) contract at Q=1, so the
    shared project loss functions apply without a TiRex-specific variant."""
    if X.ndim != 3 or X.shape[-1] != probe.in_features:
        raise ValueError(f"probe input must be (B, K, {probe.in_features}), got {tuple(X.shape)}")
    per_patch = probe(X)                                     # (B, K, out_patch)
    return per_patch.reshape(X.shape[0], 1, -1)              # (B, 1, K*out_patch) = (B, 1, H)


def pinball_tau_half(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The training objective and the reported loss: mean pinball at tau=0.5.
    ``pred`` (B, 1, H), ``target`` (B, H). Delegates to the project-wide implementation."""
    q = torch.as_tensor([TAU], dtype=torch.float32, device=pred.device)
    return mean_pinball_loss(pred, target, q)


def tau_half_is_half_mae(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Contract 16: tau=0.5 pinball == 0.5 * MAE exactly, so MAE and the objective induce the
    SAME ordering over layers and over weight-decay candidates -- the tunnel entrance and every
    selection are therefore unaffected by the choice between them."""
    pin = float(pinball_tau_half(pred, target))
    mae = float((target.unsqueeze(1) - pred).abs().mean())
    return {"pinball_tau_half": pin, "mae": mae, "half_mae": 0.5 * mae,
            "abs_diff": abs(pin - 0.5 * mae), "identity_holds": abs(pin - 0.5 * mae) <= 1e-7}


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _stack(F: np.ndarray) -> np.ndarray:
    """(n, K, d) -> (n*K, d), row-major: row i*K + k is window i, readout position k. The SAME
    ordering the CKA / effective-rank matrix uses (probing.cka.stack_slots)."""
    A = np.asarray(F, dtype=np.float32)
    if A.ndim != 3:
        raise ValueError(f"expected (n, K, d) readout features, got shape {A.shape}")
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def _fit_one(Xtr, ytr, weight_decay, epochs, lr, device, out_patch, init_seed):
    """One full-batch AdamW fit. Mirrors probes._fit_quantile_linear: re-seeded init, decay on
    the WEIGHT only, deterministic, eval() on return."""
    probe = make_shared_patch_probe(Xtr.shape[-1], out_patch, device=device, seed=init_seed)
    opt = torch.optim.AdamW([{"params": [probe.weight], "weight_decay": weight_decay},
                             {"params": [probe.bias], "weight_decay": 0.0}], lr=lr)
    probe.train()
    for _ in range(epochs):
        loss = pinball_tau_half(probe_forward(probe, Xtr), ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    return probe


def fit_shared_patch_probe(Ftr, ytr, Fva, yva, *, out_patch=32, wd_grid=WD_GRID_TIREX,
                           epochs=EPOCHS, lr=LR, device="cpu", init_seed=SEED):
    """Fit ONE shared patch probe at one depth, selecting weight decay on an EXPLICIT val split.

    Ftr/Fva : (n, K, d) readout features.   ytr/yva : (n, H) targets in normalized space.

    Train-only feature standardization, fit on the STACKED train rows so a SINGLE affine
    transform precedes the shared head (contract 14). No refit after selection: every candidate
    is already trained on the FULL train split, so the chosen one is kept as-is -- the
    ``fit_quantile_probe_explicit_val`` contract.
    """
    wd_grid = assert_wd_grid(wd_grid, lr)
    Ftr, Fva = np.asarray(Ftr, np.float32), np.asarray(Fva, np.float32)
    ytr_a, yva_a = np.asarray(ytr, np.float32), np.asarray(yva, np.float32)
    K, d = Ftr.shape[1], Ftr.shape[2]
    if ytr_a.shape[1] != K * out_patch:
        raise ValueError(f"targets have H={ytr_a.shape[1]} but K*out_patch = {K}*{out_patch} = "
                         f"{K * out_patch}; the probe cannot cover the horizon")

    scaler = StandardScaler().fit(_stack(Ftr))           # TRAIN ONLY, both positions together
    if np.any(scaler.scale_ <= 0) or not np.all(np.isfinite(scaler.scale_)):
        raise RuntimeError("train feature scale is non-positive/non-finite at some dimension")

    def prep(F):
        return torch.as_tensor(scaler.transform(_stack(F)).reshape(F.shape[0], K, d),
                               dtype=torch.float32, device=device)

    Xtr, Xva = prep(Ftr), prep(Fva)
    ytr_t = torch.as_tensor(ytr_a, device=device)
    yva_t = torch.as_tensor(yva_a, device=device)

    best, sel = None, {}
    for cand in wd_grid:
        m = _fit_one(Xtr, ytr_t, cand, epochs, lr, device, out_patch, init_seed)
        with torch.no_grad():
            v = float(pinball_tau_half(probe_forward(m, Xva), yva_t))
        sel[float(cand)] = v
        if best is None or v < best[1]:
            best = (cand, v, m)
    wd, val_loss, probe = best
    with torch.no_grad():
        train_loss = float(pinball_tau_half(probe_forward(probe, Xtr), ytr_t))
    return {"probe": probe, "scaler": scaler, "wd": float(wd), "K": int(K),
            "out_patch": int(out_patch), "in_features": int(d),
            "n_params": int(sum(p.numel() for p in probe.parameters())),
            "train_loss": train_loss, "val_loss": float(val_loss),
            "selection": {"val_loss_by_wd": sel, "chosen_wd": float(wd),
                          "at_grid_max": float(wd) == float(max(wd_grid)),
                          "at_grid_min": float(wd) == float(min(wd_grid))}}


def predict(fit: dict, F, device="cpu") -> np.ndarray:
    """Apply a FROZEN fitted probe to (n, K, d) features -> (n, H) normalized forecast."""
    F = np.asarray(F, np.float32)
    K, d = fit["K"], fit["in_features"]
    X = torch.as_tensor(fit["scaler"].transform(_stack(F)).reshape(F.shape[0], K, d),
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        return probe_forward(fit["probe"], X).squeeze(1).cpu().numpy()


def per_window_loss(pred_norm, target_norm, sl: slice | None = None) -> np.ndarray:
    """Per-window tau=0.5 pinball, optionally over a horizon SLICE (the per-patch diagnostic)."""
    p = torch.as_tensor(np.asarray(pred_norm, np.float32)).unsqueeze(1)
    t = torch.as_tensor(np.asarray(target_norm, np.float32))
    if sl is not None:
        p, t = p[:, :, sl], t[:, sl]
    q = torch.as_tensor([TAU], dtype=torch.float32)
    e = t.unsqueeze(1) - p
    return torch.maximum(q.view(1, -1, 1) * e, (q.view(1, -1, 1) - 1.0) * e).mean(dim=(1, 2)).numpy()


def per_patch_losses(pred_norm, target_norm, geom) -> dict:
    """Spec section K: the SAME shared probe scored on the full horizon and on each native patch
    separately. Diagnostic only -- it answers "does the masked-future position behave radically
    differently from the last-context position?" without fitting anything extra."""
    out = {"full": float(per_window_loss(pred_norm, target_norm).mean())}
    for k in range(geom.n_forecast_patches):
        sl = geom.target_slice(k)
        lbl = "patch0_last_context" if k == 0 else f"patch{k}_masked_future"
        out[lbl] = float(per_window_loss(pred_norm, target_norm, sl).mean())
    return out


def constant_forecast_floor(ytr, yte, yva=None) -> dict:
    """The NO-INFORMATION reference: the best CONSTANT forecast under this objective.

    At tau=0.5 the loss-minimizing constant is the per-step MEDIAN of the training targets, so
    this is a bias-only predictor fit on train and scored on the held-out splits -- the exact
    analogue of the TimesFM-3 line's ``--null-wd`` null baseline, obtained in closed form instead
    of by an extreme-decay fit. A probe that does not beat it carries no linearly decodable
    forecast information at that depth, and the driver flags any depth where that happens.
    """
    ytr_a = np.asarray(ytr, np.float32)
    const = np.median(ytr_a, axis=0)[None, :]
    out = {"predictor": "per-step median of the TRAIN targets (tau=0.5 optimal constant)",
           "train_loss": float(per_window_loss(np.repeat(const, len(ytr_a), 0), ytr_a).mean()),
           "test_loss": float(per_window_loss(np.repeat(const, len(np.asarray(yte)), 0),
                                              np.asarray(yte, np.float32)).mean())}
    if yva is not None:
        yva_a = np.asarray(yva, np.float32)
        out["val_loss"] = float(per_window_loss(np.repeat(const, len(yva_a), 0), yva_a).mean())
    return out


def representation_norms(feats: dict) -> dict:
    """Mean L2 norm of each representation point, per readout position, plus two degeneracy
    flags. An obvious hook or indexing failure (a constant, a zeroed pad token, a duplicated
    position) is visible here -- and so is one REAL degeneracy that is not a bug:

    ``constant_across_windows[k]`` is True at (Emb, position 1). The first masked-future token's
    input is (values=0, mask=0) before any recurrence has touched it, so the patch embedding of
    it is a pure bias term -- bit-identical for every window. It is a true property of the
    representation, not an extraction fault, and it is reported rather than hidden because it
    changes how Emb's second-patch probe and Emb's geometry row block must be read. Every deeper
    depth is window-dependent at both positions (the sLSTM recurrence mixes the context in).
    """
    out = {}
    for name, F in feats.items():
        A = np.asarray(F, np.float64)
        n = np.linalg.norm(A, axis=-1)                       # (n, K)
        const = [bool(np.all(A[:, k] == A[0:1, k])) for k in range(A.shape[1])]
        out[name] = {"mean_l2_by_position": [float(x) for x in n.mean(axis=0)],
                     "std_l2_by_position": [float(x) for x in n.std(axis=0)],
                     "across_window_std_by_position":
                         [float(A[:, k].std(axis=0).mean()) for k in range(A.shape[1])],
                     "constant_across_windows": const,
                     "positions_identical": bool(np.allclose(A[:, 0], A[:, -1])) if A.shape[1] > 1
                                            else None}
    return out


# --------------------------------------------------------------------------- #
# the layer-wise sweep
# --------------------------------------------------------------------------- #
def layerwise_probe(train_feats, ytr, val_feats, yva, test_feats, yte, geom, *,
                    points=None, device="cpu", wd_grid=WD_GRID_TIREX, epochs=EPOCHS, lr=LR,
                    init_seed=SEED, verbose=True):
    """Fit + score one shared patch probe at EVERY representation depth.

    Returns {point name -> record} with train/val/test loss, the per-patch diagnostic, the wd
    selection, and the normalized test forecast (for MASE in raw units downstream).
    Train fits, val selects, test is touched ONCE and never influences anything.
    """
    names = list(points) if points is not None else list(REP_NAMES)
    out = {}
    for name in names:
        fit = fit_shared_patch_probe(train_feats[name], ytr, val_feats[name], yva,
                                     out_patch=geom.output_patch, wd_grid=wd_grid, epochs=epochs,
                                     lr=lr, device=device, init_seed=init_seed)
        pred_te = predict(fit, test_feats[name], device=device)
        pw = per_window_loss(pred_te, yte)
        rec = {
            "point": name,
            **{k: v for k, v in _point_meta(name).items()},
            "wd": fit["wd"], "selection": fit["selection"], "n_params": fit["n_params"],
            "train_loss": fit["train_loss"], "val_loss": fit["val_loss"],
            "test_loss": float(pw.mean()),
            "test_per_patch": per_patch_losses(pred_te, yte, geom),
            "val_per_patch": per_patch_losses(predict(fit, val_feats[name], device=device), yva, geom),
            "pred_test_norm": pred_te, "test_window_loss": pw,
        }
        out[name] = rec
        if verbose:
            print(f"    [probe] {name:<9s} wd={fit['wd']:<7g} "
                  f"train={fit['train_loss']:.5f} val={fit['val_loss']:.5f} "
                  f"test={rec['test_loss']:.5f}", flush=True)
    return out


def _point_meta(name: str) -> dict:
    p = [x for x in REP_POINTS if x.name == name]
    if not p:
        raise KeyError(f"unknown representation point {name!r}")
    return {k: v for k, v in p[0].as_dict().items() if k != "name"}


# --------------------------------------------------------------------------- #
# native baseline + tunnel
# --------------------------------------------------------------------------- #
def native_median_reference(native_quantiles, loc, scale, targets_norm, geom) -> dict:
    """TiRex's OWN median forecast on the same windows, scored with the SAME tau=0.5 objective in
    the SAME normalized space, so it sits directly on the probe curve as the model's reference.

    ``native_quantiles`` is (n, H, Q) in RAW units (what the native path returns); it is
    re-normalized PER OUTPUT PATCH with that patch's own (loc, scale) -- the exact inverse of
    ``denormalize`` -- so no metric is ever compared across two different spaces.
    """
    from probing.tirex_model import normalize_raw
    nq = np.asarray(native_quantiles, np.float32)
    if nq.ndim != 3 or nq.shape[1] != geom.H or nq.shape[2] != geom.num_quantiles:
        raise ValueError(f"native quantiles must be (n, {geom.H}, {geom.num_quantiles}), "
                         f"got {nq.shape}")
    med_raw = nq[:, :, geom.median_index]
    # PER-PATCH inverse: in two_pass mode each output patch has its own (loc, scale), so a single
    # global rescale would silently mix two normalized spaces.
    med_norm = normalize_raw(med_raw, loc, scale, geom).astype(np.float32)
    pw = per_window_loss(med_norm, targets_norm)
    return {"median_raw": med_raw, "median_norm": med_norm,
            "test_loss": float(pw.mean()), "test_window_loss": pw,
            "test_per_patch": per_patch_losses(med_norm, targets_norm, geom),
            "median_index": geom.median_index}


def tunnel_entrance(val_losses, points=None, tol: float = TUNNEL_TOL) -> dict:
    """The project's first-crossing rule, applied to the TiRex depth axis.

    ``probing.tunnel.tunnel_start`` is called UNCHANGED -- there is no second implementation of
    the criterion anywhere in this file. Reference depth = the LAST representation point
    (L12+RMS), the one the native head reads.

    Both the absolute and the normalized depth are stored; see ``RepPoint.relative_depth`` for
    why ``relative_depth`` (block-based) is the cross-model coordinate and ``relative_position``
    the monotone plotting one.
    """
    v = np.asarray(val_losses, dtype=np.float64)
    names = list(points) if points is not None else list(REP_NAMES)
    if v.size != len(names):
        raise ValueError(f"{v.size} validation losses for {len(names)} representation points")
    i = int(tunnel_start(v, tol))
    meta = _point_meta(names[i])
    return {"tolerance": float(tol), "definition": "first_crossing_95 (probing.tunnel.tunnel_start)",
            "criterion": "min { l : L_val(l) <= (1+tol) * L_val(last) }",
            "reference_point": names[-1], "index": i, "point": names[i],
            "block_index": meta["block_index"], "relative_depth": meta["relative_depth"],
            "position_index": meta["position_index"],
            "relative_position": meta["relative_position"], "kind": meta["kind"],
            "val_loss_at_entrance": float(v[i]), "val_loss_at_reference": float(v[-1]),
            "ratio_to_reference": float(v[i] / v[-1])}


def tunnel_entrances(val_losses, points=None, tols=(0.01, 0.02, 0.05, 0.10)) -> dict:
    """Every tolerance saved at once, so a 2% criterion never needs a re-run."""
    return {f"tol_{t:g}": tunnel_entrance(val_losses, points, t) for t in tols}
