"""Phase-2 alignment adapters: identity-anchored, hard-cut-nested, validation-selected.

ONE parameterization for every family and every model:

    a(x) = A x + b,   A = I + Delta,   Delta and b initialized to 0

so every adapter STARTS AT THE HARD CUT, and "do nothing" is always a selectable candidate:

    iterative fits (Chronos-2, TiRex)   epoch 0 of every fit IS the hard cut, and the validation
                                        criterion is recorded there before any step is taken;
    closed form (TimesFM-3, RA)         the hard cut (A = I, b = 0) is an explicit candidate beside
                                        every ridge strength.

Regularization shrinks toward the IDENTITY, never toward zero. The ext_v5 Chronos-2 adapter decayed
its weight toward ZERO under AdamW (0.97^300 ~ 1e-4 at wd=3), so the hard cut was not nested in its
family; that is the committed "adapter at L12 is worse than native on 6 of 7 datasets" pathology.
Here the decoupled decay acts on Delta alone (b undecayed), and the closed form penalizes
||A - I||_F^2.

OBJECTIVES (both families use the same pathway and the same selection discipline):
    NOA  label-free   mean || f(a(h_l)) - f(h_L) ||^2 over the pathway's NORMALIZED outputs
                      (the model's own forecast from the final block is the target; no future
                      value enters the fit or the selection -- it is the tuned-lens analogue)
    FL   supervised   the Phase-1 common Q=9 mean pinball of f(a(h_l)) against the normalized
                      target (Chronos-2 and TiRex only; for TimesFM-3 it IS the probe)
    RA   label-free   || a(h_l) - h_L ||^2 in the HIDDEN metric (diagnostic only)

Selection is on the VALIDATION split only: NOA and RA by their own label-free validation error, FL
by validation pinball. Test is never read here. Fits are deterministic (zero init, full batch), so
one seed; uncertainty is the Phase-1 cluster bootstrap on test.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probing.phase2 import (ADAPTER_EPOCHS, ADAPTER_EVAL_EVERY, ADAPTER_LR, ADAPTER_WD_GRID,
                            RIDGE_KAPPAS, array_sha256, assert_adapter_wd_grid)

__all__ = ["ResidualAdapter", "NestedResidualAdapter", "mse_loss", "pinball_mean",
           "fit_residual_adapter", "anchored_ridge_path", "affine_to_adapter", "save_adapter",
           "load_adapter", "adapter_arrays_sha256"]


class ResidualAdapter(nn.Module):
    """a(x) = x + x Delta^T + b, applied to the LAST dimension (broadcasts over slots / passes).

    With Delta = 0 and b = 0 the output is x + 0 exactly (``F.linear`` with an all-zero weight and
    bias produces exact zeros for finite x), so the adapter at initialization is BITWISE the hard
    cut -- the identity the whole design relies on, pinned by contract 3.
    """

    def __init__(self, d: int, dtype=torch.float32):
        super().__init__()
        self.d = int(d)
        self.delta = nn.Parameter(torch.zeros(self.d, self.d, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.d, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.linear(x, self.delta, self.bias)

    @torch.no_grad()
    def affine(self) -> tuple[np.ndarray, np.ndarray]:
        """(A, b) in float64, column convention a(x) = A x + b."""
        D = self.delta.detach().double().cpu().numpy()
        return np.eye(self.d) + D, self.bias.detach().double().cpu().numpy()

    @property
    def n_params(self) -> int:
        return int(self.delta.numel() + self.bias.numel())

    def decay_params(self) -> list:
        """Parameters under decoupled decay (toward the identity / toward the nested family)."""
        return [self.delta]

    def free_params(self) -> list:
        return [self.bias]


class NestedResidualAdapter(ResidualAdapter):
    """a(x) = x + x Delta^T + b + W2 gelu(W1 LN(x) + b1): the affine adapter PLUS a bottleneck
    nonlinear branch (width r). EXPLORATORY (Phase 2b smoke, notes/PLAN.md), not the H3 protocol.

    Nesting, by construction: W2 = 0 at initialization, so the branch contributes exact zeros and
    the module is BITWISE the affine adapter (and, with Delta = b = 0, the hard cut). W1 is random
    (seeded by the caller's torch.manual_seed) -- with W1 = 0 too, no gradient would ever reach
    the branch. The branch reads a parameter-free LayerNorm of x, so its pre-activations are
    O(1) whatever the scale of the residual stream (TimesFM-3 has no final norm).
    Decay: Delta (toward I), W1 and W2 (toward the affine family); b, b1 undecayed.
    """

    def __init__(self, d: int, r: int = 64, dtype=torch.float32):
        super().__init__(d, dtype=dtype)
        self.r = int(r)
        self.w1 = nn.Parameter(torch.randn(self.r, self.d, dtype=dtype) / np.sqrt(self.d))
        self.b1 = nn.Parameter(torch.zeros(self.r, dtype=dtype))
        self.w2 = nn.Parameter(torch.zeros(self.d, self.r, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = F.linear(F.gelu(F.linear(F.layer_norm(x, (self.d,)), self.w1, self.b1)), self.w2)
        return x + F.linear(x, self.delta, self.bias) + branch

    @property
    def n_params(self) -> int:
        return int(super().n_params + self.w1.numel() + self.b1.numel() + self.w2.numel())

    def decay_params(self) -> list:
        return [self.delta, self.w1, self.w2]

    def free_params(self) -> list:
        return [self.bias, self.b1]


def affine_to_adapter(A, b, dtype=torch.float32) -> ResidualAdapter:
    """A closed-form (A, b) as the SAME module the iterative fits produce (Delta = A - I)."""
    A = np.asarray(A, np.float64)
    ad = ResidualAdapter(A.shape[0], dtype=dtype)
    with torch.no_grad():
        ad.delta.copy_(torch.as_tensor(A - np.eye(A.shape[0]), dtype=dtype))
        ad.bias.copy_(torch.as_tensor(np.asarray(b, np.float64), dtype=dtype))
    return ad


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def pinball_mean(pred: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The Phase-1 common loss: mean of rho_tau over (batch, Q, H). pred (n, Q, H), target (n, H)."""
    u = target.unsqueeze(1) - pred
    qv = q.view(1, -1, 1)
    return torch.maximum(qv * u, (qv - 1.0) * u).mean()


# --------------------------------------------------------------------------- #
# iterative fit: full-batch AdamW, decay toward identity, checkpoint selection on validation
# --------------------------------------------------------------------------- #
def fit_residual_adapter(*, d: int, pathway_fn, train_x, train_y, val_x, val_y, train_loss,
                         val_criterion, wd_grid=ADAPTER_WD_GRID, epochs: int = ADAPTER_EPOCHS,
                         lr: float = ADAPTER_LR, eval_every: int = ADAPTER_EVAL_EVERY,
                         device=None, seed: int = 0, log=None, label: str = "",
                         make_adapter=None) -> dict:
    """Fit one residual adapter per decay value and keep the best (wd, epoch) on VALIDATION.

    ``pathway_fn(adapted_states)`` returns what the losses consume (NOA: the normalized head
    outputs; FL: the (n, 9, H) normalized quantiles). The frozen pathway stays in the graph, so
    gradients flow THROUGH it to Delta and b; no pathway parameter requires grad (asserted by the
    caller's pathway, contract 3).

    The validation criterion is evaluated at epoch 0 (== the hard cut, identical for every wd) and
    every ``eval_every`` epochs. The selected checkpoint is the global minimum over all (wd, epoch)
    pairs -- no refit after selection. Ties go to the smaller epoch, then the LARGER wd (closer to
    the identity). A non-finite training loss stops that candidate and is recorded, never selected.

    ``make_adapter(d)`` builds the module (default ``ResidualAdapter``; the exploratory nested
    adapter passes ``NestedResidualAdapter``). It is called right after ``torch.manual_seed(seed)``,
    so any random initialization is identical across the decay grid.
    """
    make_adapter = make_adapter or (lambda dd: ResidualAdapter(dd))
    grid = assert_adapter_wd_grid(wd_grid, lr)
    device = device or (train_x.device if torch.is_tensor(train_x) else "cpu")
    best, cands, hard_val = None, [], None

    def consider(v, epoch, wd, ad):
        nonlocal best
        key = (float(v), int(epoch), -float(wd))
        if np.isfinite(v) and (best is None or key < best["key"]):
            best = {"key": key, "val": float(v), "epoch": int(epoch),
                    "wd": None if epoch == 0 else float(wd),
                    "state": {k: t.detach().clone() for k, t in ad.state_dict().items()}}

    for wd in grid:
        torch.manual_seed(seed)
        ad = make_adapter(d).to(device)
        opt = torch.optim.AdamW([{"params": ad.decay_params(), "weight_decay": float(wd)},
                                 {"params": ad.free_params(), "weight_decay": 0.0}], lr=lr)

        def evaluate():
            with torch.no_grad():
                return float(val_criterion(pathway_fn(ad(val_x)), val_y))

        if hard_val is None:
            hard_val = evaluate()
        curve, train_curve = [(0, hard_val)], []
        consider(hard_val, 0, wd, ad)
        status, loss_v = "completed", float("nan")
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            loss = train_loss(pathway_fn(ad(train_x)), train_y)
            loss_v = float(loss.detach())
            if not np.isfinite(loss_v):
                status = f"non-finite training loss at epoch {ep}"
                break
            loss.backward()
            opt.step()
            if ep % eval_every == 0 or ep == epochs:
                v = evaluate()
                curve.append((ep, v))
                train_curve.append((ep, loss_v))
                if not np.isfinite(v):
                    status = f"non-finite validation criterion at epoch {ep}"
                    break
                consider(v, ep, wd, ad)
        cands.append({"wd": float(wd), "status": status, "val_curve": curve,
                      "train_curve": train_curve, "final_train_loss": loss_v,
                      "best_val": float(min(v for _, v in curve if np.isfinite(v)))})
        if log:
            log(f"      [{label}] wd={wd:<6g} best val {cands[-1]['best_val']:.6g} "
                f"(hard cut {hard_val:.6g}) {status}")

    torch.manual_seed(seed)
    out = make_adapter(d).to(device)
    out.load_state_dict(best["state"])
    out.eval()
    return {"adapter": out, "selected_wd": best["wd"], "selected_epoch": best["epoch"],
            "selected_val": best["val"], "hard_cut_val": float(hard_val),
            "selected_hard_cut": bool(best["epoch"] == 0),
            "candidates": cands, "grid": list(grid), "epochs": int(epochs), "lr": float(lr),
            "eval_every": int(eval_every), "optimizer": "AdamW full batch; decoupled decay on "
            "Delta only (toward the identity); bias undecayed", "seed": int(seed),
            "n_params": out.n_params, "adapter_class": type(out).__name__}


# --------------------------------------------------------------------------- #
# closed form: the identity-anchored ridge in an arbitrary output metric
# --------------------------------------------------------------------------- #
def anchored_ridge_path(X_tr, Y_tr, X_va, Y_va, *, metric=None, out_dim: int | None = None,
                        kappas=RIDGE_KAPPAS, include_hard_cut: bool = True) -> dict:
    """min_{A,b} sum_i ||A x_i + b - y_i||_M^2 + lambda ||A - I||_F^2   (b unpenalized)

    with ||v||_M^2 = v^T M v. ``metric=None`` is the hidden metric (M = I: the RA diagnostic);
    ``metric = W^T W`` for a linear head W makes it the head-weighted native-output alignment
    (TimesFM-3 NOA): ||W(A x + b) + c - (W y + c)||^2 = ||A x + b - y||_{W^T W}^2.

    With centred X, Y and B = A^T the optimality condition is X^T X B M + lambda B = X^T Y M +
    lambda I. In the eigenbases X^T X = V diag(g) V^T and M = Q diag(m) Q^T, B~ = V^T B Q has
    B~_ij = [V^T (X^T Y M + lambda I) Q]_ij / (g_i m_j + lambda), so ONE pair of eigen-
    decompositions serves the whole grid. Directions the metric ignores (m_j = 0) stay at the
    identity, so A is well-conditioned even when W is not (TimesFM-3: cond(W) ~ 1.3e5).

    lambda_k = kappa_k * g_max * m_max (scale-relative; see phase2.RIDGE_KAPPAS). The hard cut
    (A = I, b = 0) is an explicit candidate (``include_hard_cut=False`` removes it, which only the
    contracts and the final-depth identity check use, to examine the ridge solution itself).
    Selection = the smallest VALIDATION error in the same metric; ties go to the larger lambda
    (closer to the identity). float64 throughout.
    """
    X = np.asarray(X_tr, np.float64)
    Y = np.asarray(Y_tr, np.float64)
    Xv = np.asarray(X_va, np.float64)
    Yv = np.asarray(Y_va, np.float64)
    if X.shape != Y.shape or Xv.shape[1] != X.shape[1] or Yv.shape != Xv.shape:
        raise ValueError(f"shapes: X {X.shape}, Y {Y.shape}, Xv {Xv.shape}, Yv {Yv.shape}")
    n, d = X.shape
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    g, V = np.linalg.eigh(Xc.T @ Xc)
    g = np.clip(g, 0.0, None)
    if metric is None:
        M, m, Qm = None, np.ones(d), np.eye(d)
        p = d if out_dim is None else int(out_dim)
    else:
        M = np.asarray(metric, np.float64)
        M = 0.5 * (M + M.T)
        m, Qm = np.linalg.eigh(M)
        m = np.clip(m, 0.0, None)
        p = int(out_dim) if out_dim is not None else int((m > m.max() * 1e-12).sum())
    R = Xc.T @ Yc
    C0 = V.T @ (R if M is None else R @ M) @ Qm
    C1 = V.T @ Qm
    scale = float(g.max() * m.max()) if g.max() > 0 and m.max() > 0 else 1.0

    def crit(Xe, Ye, B, b):
        E = Xe @ B + b - Ye
        if M is None:
            return float((E * E).sum() / (E.shape[0] * p))
        return float(((E @ M) * E).sum() / (E.shape[0] * p))

    I = np.eye(d)
    hard = {"candidate": "hard_cut", "kappa": None, "lambda": None,
            "val": crit(Xv, Yv, I, np.zeros(d)), "train": crit(X, Y, I, np.zeros(d))}
    cands = [hard] if include_hard_cut else []
    sols = {"hard_cut": (I, np.zeros(d))}
    for k in kappas:
        lam = float(k) * scale
        Bt = (C0 + lam * C1) / (g[:, None] * m[None, :] + lam)
        B = V @ Bt @ Qm.T
        b = muY - muX @ B
        name = f"kappa={float(k):g}"
        cands.append({"candidate": name, "kappa": float(k), "lambda": lam,
                      "val": crit(Xv, Yv, B, b), "train": crit(X, Y, B, b),
                      "effective_dof": float((g / (g + lam)).sum())})
        sols[name] = (B.T, b)
    finite = [c for c in cands if np.isfinite(c["val"])]
    if not finite:
        raise RuntimeError("every closed-form candidate produced a non-finite validation error")
    # ties -> larger lambda (the hard cut counts as lambda = +inf, i.e. the identity)
    best = min(finite, key=lambda c: (c["val"], -(np.inf if c["lambda"] is None else c["lambda"])))
    A, b = sols[best["candidate"]]
    ks = [c["kappa"] for c in cands if c["kappa"] is not None]
    return {"A": A, "b": b, "selected": best["candidate"], "selected_kappa": best["kappa"],
            "selected_lambda": best["lambda"], "selected_val": best["val"],
            "selected_hard_cut": best["candidate"] == "hard_cut",
            "hard_cut_val": hard["val"],
            "at_grid_edge": best["kappa"] is not None and best["kappa"] in (min(ks), max(ks)),
            "candidates": cands, "metric": "hidden (I)" if M is None else "head-weighted W^T W",
            "out_dim": p, "n_train_rows": int(n), "d": int(d),
            "lambda_scale_g_max_times_m_max": scale,
            "spectrum": {"g_max": float(g.max()), "g_min": float(g.min()),
                         "m_max": float(m.max()), "m_min": float(m.min()),
                         "metric_rank": int((m > m.max() * 1e-12).sum()) if M is not None else d}}


# --------------------------------------------------------------------------- #
# persistence -- the physical H4 model must load EXACTLY the offline parameters
# --------------------------------------------------------------------------- #
_NESTED_KEYS = ("w1", "b1", "w2")


def _adapter_arrays(ad: ResidualAdapter) -> dict:
    """float32 numpy arrays, in the fixed order the checksum uses (delta, bias[, w1, b1, w2])."""
    keys = ("delta", "bias") + (_NESTED_KEYS if isinstance(ad, NestedResidualAdapter) else ())
    return {k: getattr(ad, k).detach().float().cpu().numpy() for k in keys}


def adapter_arrays_sha256(ad: ResidualAdapter) -> str:
    """Affine: sha256(delta, bias) exactly as before (committed adapters keep their checksums);
    nested: sha256(delta, bias, w1, b1, w2)."""
    return array_sha256(*_adapter_arrays(ad).values())


def save_adapter(path, ad: ResidualAdapter, meta: dict) -> dict:
    """float32 Delta and b (and the nested branch, if any), written atomically (temp +
    os.replace). The SAME float32 tensors the offline H3 evaluation applied, so the physical H4
    model reproduces them up to GEMM shape."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = _adapter_arrays(ad)
    sha = array_sha256(*arrs.values())
    extra = ({"adapter_class": "NestedResidualAdapter", "bottleneck": int(ad.r)}
             if isinstance(ad, NestedResidualAdapter) else {})
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    import json
    np.savez(tmp, **arrs, meta=json.dumps({**meta, **extra, "sha256": sha}, default=str))
    os.replace(tmp, path)
    return {"path": str(path), "sha256": sha, "d": int(arrs["delta"].shape[0]),
            "n_params": int(sum(a.size for a in arrs.values())), **extra}


def load_adapter(path, device=None) -> tuple[ResidualAdapter, dict]:
    """Rebuilds the affine or the nested adapter, whichever the file holds; checksum-verified."""
    import json
    with np.load(path, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    nested = all(k in arrs for k in _NESTED_KEYS)
    order = ("delta", "bias") + (_NESTED_KEYS if nested else ())
    sha = array_sha256(*(arrs[k] for k in order))
    if meta.get("sha256") != sha:
        raise RuntimeError(f"adapter {path} fails its own checksum ({sha} != {meta.get('sha256')})")
    d = arrs["delta"].shape[0]
    ad = NestedResidualAdapter(d, r=arrs["w1"].shape[0]) if nested else ResidualAdapter(d)
    with torch.no_grad():
        for k in order:
            getattr(ad, k).copy_(torch.as_tensor(arrs[k]))
    ad.eval()
    for p in ad.parameters():
        p.requires_grad_(False)
    return (ad.to(device) if device is not None else ad), meta
