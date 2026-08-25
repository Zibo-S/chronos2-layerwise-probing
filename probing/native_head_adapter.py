"""ext_v5 — a shared linear representation-ADAPTER into Chronos-2's own frozen native quantile head.

This is NOT the ext_v4 shared-forecast probe. ext_v4 trains a fresh readout
``forecast slots -> Linear(768, Q*P)`` from scratch on each probe dataset. Here we instead ask a
different question, using the model's OWN pretrained forecasting machinery:

    At each layer l, how compatible is the representation with Chronos-2's pretrained Quantile Head,
    and can a single shared linear 768->768 map make it compatible?

Three conditions, all flowing through the ACTUAL pretrained ``output_patch_embedding`` (never a
reimplementation):

    1. native baseline   :  final_slots -> native head                       (no new params)
    2. zero-shot head    :  h_l -> final RMSNorm -> native head              (no new params)
                            (L12+RMS is ALREADY post-RMSNorm -> straight to the head, no 2nd norm)
    3. linear adapter    :  h_l -> A_l -> final RMSNorm -> native head        (ONLY A_l trains)

``A_l = nn.Linear(768, 768)``, the SAME map applied independently to each of the K=4 forecast slots
(nn.Linear broadcasts over the (n, K) axes), IDENTITY-initialised (W=I, b=0). Consequences:
  * at init the adapter reproduces the zero-shot path EXACTLY (adapter == zero-shot every layer);
  * training can only move OFF zero-shot toward native.

Interpretation, stated precisely (see the review note): if ``A_l`` reaches native quality, we may
say only that *a linear transformation of the layer-l representation is SUFFICIENT to make it usable
by the frozen native head*. We may NOT say the adapter "recovers the l->L12 coordinate transform",
because ``A_l`` is supervised by the forecast target Y, not by the L12 representation. The clean
label-free follow-up ``min_A || RMS(A h_l) - h_{L12+RMS} ||^2`` is a PARKED future direction, not
built here.

Terminology: Chronos-2's ``encoder.final_layer_norm`` is a T5-style **RMSNorm** (no mean subtraction),
so we call the post-block-12 point **L12+RMS**, not "L12+LN".

Kept deliberately isolated from ext_v4: this module adds no probe to ``probing.probes`` and touches
no existing code path; every driver output lands under ``results/ext_v5_native_head_adapter/``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from probing.config import SEED, OUTPUT_PATCH_SIZE
from probing.probes import (CHRONOS2_QUANTILES, _apply_shared_head, chronos2_quantile_loss,
                            chronos2_quantile_loss_per_window, validate_quantiles)

NUM_NATIVE_QUANTILES = len(CHRONOS2_QUANTILES)   # 21 native levels (includes 0.5 and every q9 decile)


# --------------------------------------------------------------------------- #
# The two FROZEN pretrained modules every condition flows through.
# --------------------------------------------------------------------------- #
def native_head_modules(pipeline):
    """Return ``(head, final_rms)`` — the pretrained modules reused verbatim (NEVER reimplemented):

      head      = ``model.output_patch_embedding`` — a ResidualBlock(768 -> d_ff -> num_q*P), the
                  native Quantile Head (chronos2/model.py:265). Consumes the POST-final-RMSNorm
                  forecast slots (model.py:727-732).
      final_rms = ``model.encoder.final_layer_norm`` — T5-style RMSNorm, per-position (layers.py:129).

    Both are put in ``.eval()`` (so the ResidualBlock's dropout is OFF) and have ``requires_grad=False``
    on every parameter — yet they remain in the autograd graph, so gradients still flow THROUGH them to
    a trainable adapter. Only the adapter ever updates."""
    model = pipeline.model
    head = model.output_patch_embedding
    final_rms = model.encoder.final_layer_norm
    head.eval()
    final_rms.eval()
    for mod in (head, final_rms):
        for p in mod.parameters():
            p.requires_grad_(False)
    return head, final_rms


class LinearAdapter(nn.Module):
    """Shared linear 768->768 alignment map, applied INDEPENDENTLY to each forecast slot.

    ``nn.Linear`` applies to the last dim, broadcasting over the ``(n, K)`` slot axes, so the SAME
    weight matrix hits all K=4 slots (the shared-slot philosophy of the ext_v4 fslot readout). It is
    the ONLY trainable object in the whole pipeline.

    Identity-initialised (W=I, b=0): at init ``A(h) == h``, so the adapter path is byte-identical to
    the zero-shot path. Training moves it off identity to align the representation with the native
    head. (Full-batch training from this fixed init is deterministic, so there is ONE adapter per
    layer/wd — uncertainty comes from the test-set cluster bootstrap, not from init seeds.)"""

    def __init__(self, d: int = 768):
        super().__init__()
        self.linear = nn.Linear(d, d)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d))
            self.linear.bias.zero_()

    def forward(self, slots: torch.Tensor) -> torch.Tensor:   # slots: (n, K, d) -> (n, K, d)
        return self.linear(slots)


# --------------------------------------------------------------------------- #
# The shared forward path: forecast slots -> [adapter] -> [final RMSNorm] -> native head.
# --------------------------------------------------------------------------- #
def slots_to_normalized_quantiles(slots, adapter, apply_rms, final_rms, head, quantiles,
                                  output_patch_size=OUTPUT_PATCH_SIZE, horizon=None):
    """(n, Q, H) NORMALIZED (arcsinh-space) quantile predictions from forecast slots.

    slots      : (n, K, 768) forecast-slot hidden states (torch, on the head's device).
    adapter    : a ``LinearAdapter`` (condition 3) or ``None`` (conditions 1/2 = no new params).
    apply_rms  : True for PRE-RMSNorm layers (Emb..L12) -> apply the native final RMSNorm before the
                 head; False for the L12+RMS point, whose slots are ALREADY post-RMSNorm (never
                 double-normalise).
    quantiles  : the head's native quantile vector (len == head out_dim / output_patch_size).

    Reuses ``probes._apply_shared_head`` so the K predicted patches are laid out with Chronos-2's
    EXACT ``n k (q p) -> n q (k p)`` rearrange and trimmed to H — identical to the native head's own
    layout. Output is in the model's normalized (arcsinh) space; the caller inverts with mu + s*sinh.
    """
    Q = len(quantiles)
    P = int(output_patch_size)
    H = int(horizon) if horizon is not None else slots.shape[1] * P
    h = slots if adapter is None else adapter(slots)
    if apply_rms:
        h = final_rms(h)
    # head is callable (n, K, 768) -> (n, K, Q*P); _apply_shared_head reshapes to (n, Q, H).
    return _apply_shared_head(head, h, Q, P, H)


# --------------------------------------------------------------------------- #
# Fit one adapter (identity init, single deterministic full-batch fit) + wd-on-val selection.
# --------------------------------------------------------------------------- #
def _fit_one_adapter(Xtr, Ytr, final_rms, head, apply_rms, q, P, H, weight_decay, epochs, lr,
                     device, init_seed=SEED, history=None):
    """Fit a single ``LinearAdapter`` (identity init) with Chronos-2's quantile loss in normalized
    space. AdamW, weight-decay on the weight only (bias free) — same optimizer convention as the
    fslot probe's ``_fit_shared_forecast_linear``. Deterministic full-batch; ``init_seed`` only fixes
    the (identity) init RNG state for reproducibility. Only the adapter has grad; head/RMSNorm are
    frozen but pass gradients through."""
    Q = q.numel()
    torch.manual_seed(init_seed)
    adapter = LinearAdapter(Xtr.shape[-1]).to(device)          # identity init (W=I, b=0)
    opt = torch.optim.AdamW(
        [{"params": [adapter.linear.weight], "weight_decay": weight_decay},
         {"params": [adapter.linear.bias],   "weight_decay": 0.0}], lr=lr)
    adapter.train()
    for _ in range(epochs):
        pred = slots_to_normalized_quantiles(Xtr, adapter, apply_rms, final_rms, head, q, P, H)
        loss = chronos2_quantile_loss(pred, Ytr, q)
        if history is not None:
            history["train"].append(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if history is not None:
        with torch.no_grad():
            pred = slots_to_normalized_quantiles(Xtr, adapter, apply_rms, final_rms, head, q, P, H)
            history["train"].append(chronos2_quantile_loss(pred, Ytr, q).item())
    adapter.eval()
    return adapter


def fit_adapter_explicit_val(train_slots, train_labels, val_slots, val_labels, final_rms, head,
                             layers, quantiles=CHRONOS2_QUANTILES, epochs=300, lr=1e-2,
                             wd_grid=None, weight_decay=1e-3, device=None, init_seed=SEED,
                             output_patch_size=OUTPUT_PATCH_SIZE, post_rms_layers=()):
    """Per layer in ``layers``: fit a shared ``LinearAdapter`` with wd selected on an EXPLICIT
    validation split (never test), returning the frozen adapters. ONE deterministic fit per wd
    candidate (identity init + full batch => deterministic; no init-seed banding — uncertainty is the
    test cluster bootstrap). Mirrors ``fit_shared_forecast_probe_explicit_val``'s wd-on-val contract.

    train_slots/val_slots : {layer: (n, K, 768)} forecast-slot states (from ``_fslot_feats``).
    train_labels/val_labels: (n, H) arcsinh trajectory labels (== the model's normalized target).
    post_rms_layers        : layer keys whose slots are ALREADY post-RMSNorm (skip the RMSNorm).
                             The driver normally does NOT pass the L12+RMS key here — that point is
                             the native endpoint, not a trained adapter — but the arg keeps the path
                             correct for an optional "native + linear adaptation" control.

    Returns {layer: {"adapter" (eval), "wd", "selection": {val_loss_by_wd, chosen_wd},
                     "apply_rms", "param_count", "family": "native_head_adapter", "device"}}.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = validate_quantiles(quantiles)
    P = int(output_patch_size)
    Ytr = torch.as_tensor(np.asarray(train_labels, np.float32), device=device)
    Yva = torch.as_tensor(np.asarray(val_labels, np.float32), device=device)
    if Ytr.ndim != 2 or Yva.ndim != 2:
        raise ValueError(f"need (n, H) trajectory labels for train and val, got {tuple(Ytr.shape)} / "
                         f"{tuple(Yva.shape)}")
    Q, H = len(quantiles), Ytr.shape[1]
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=device)

    fitted = {}
    for L in layers:
        apply_rms = L not in set(post_rms_layers)
        Xtr = torch.as_tensor(np.asarray(train_slots[L], np.float32), device=device)     # (n, K, 768)
        Xva = torch.as_tensor(np.asarray(val_slots[L], np.float32), device=device)
        if np.ndim(train_slots[L]) != 3:
            raise ValueError(f"L{L}: need (n, K, 768) forecast slots, got {np.shape(train_slots[L])}")
        grid = list(wd_grid) if wd_grid is not None else [weight_decay]
        best_adapter, best_val, best_wd, sel = None, float("inf"), grid[0], {}
        for cand in grid:
            a = _fit_one_adapter(Xtr, Ytr, final_rms, head, apply_rms, q, P, H, cand, epochs, lr,
                                 device, init_seed=init_seed)
            with torch.no_grad():
                v = chronos2_quantile_loss(
                    slots_to_normalized_quantiles(Xva, a, apply_rms, final_rms, head, q, P, H),
                    Yva, q).item()
            sel[float(cand)] = v
            if v < best_val:
                best_val, best_wd, best_adapter = v, cand, a
        best_adapter.eval()
        fitted[L] = {"adapter": best_adapter, "wd": float(best_wd),
                     "selection": {"val_loss_by_wd": sel, "chosen_wd": float(best_wd)},
                     "apply_rms": bool(apply_rms),
                     "param_count": int(sum(p.numel() for p in best_adapter.parameters())),
                     "family": "native_head_adapter", "device": str(device)}
        print(f"    [adapter fit] L{L:>2}  wd={best_wd:g}  val_qloss={best_val:.4f}  "
              f"params={fitted[L]['param_count']}")
    return fitted


def predict_normalized_per_window(slots, adapter, apply_rms, final_rms, head, quantiles,
                                  labels, output_patch_size=OUTPUT_PATCH_SIZE, device=None):
    """Frozen forward -> (per-window normalized quantile loss, normalized quantile preds (n, Q, H)).
    Never trains. ``labels`` are the (n, H) arcsinh trajectories; the returned per-window loss is
    Chronos-2's quantile loss (its ``.mean()`` is the scalar). The (n, Q, H) preds let the driver
    invert to raw units (mu + s*sinh) for MASE/MAE/WQL."""
    quantiles = validate_quantiles(quantiles)
    P = int(output_patch_size)
    dev = device or "cpu"
    q = torch.as_tensor(quantiles, dtype=torch.float32, device=dev)
    X = torch.as_tensor(np.asarray(slots, np.float32), device=dev)
    y = torch.as_tensor(np.asarray(labels, np.float32), device=dev)
    H = y.shape[1]
    with torch.no_grad():
        pred = slots_to_normalized_quantiles(X, adapter, apply_rms, final_rms, head, q, P, H)
        pw = chronos2_quantile_loss_per_window(pred, y, q).cpu().numpy().astype(np.float64)
    return pw, pred.cpu().numpy().astype(np.float64)
