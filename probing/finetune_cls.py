"""Full fine-tuning of frozen Chronos-2 on a CLASSIFICATION task — TASK-SHIFT, Exp-A source.

Scientific frame (notes/PLAN.md, TASK-SHIFT): the DOMAIN-shift condition (BOOM forecasting FT) showed
no convincing late-layer specialization. This module produces the TASK-shift intervention: fine-tune
the SAME frozen Chronos-2 on a classification task and check whether changing the TASK creates stronger
late-layer specialization than changing only the forecasting domain. Three classification sources are
supported (probing.cls_data.CLS_SPECS): ``forda`` (easy/control, univariate), ``uwave`` and
``handwriting`` (non-saturated, 3-channel, with a pretrained intermediate-layer probe peak). It writes
two checkpoints of ONE run so run_task_shift can probe stage0 (pretrained) / stage1 (early) / stage2
(late) with the same layerwise-linear lens.

Head + forward (the ONE fixed pooling rule; MULTIVARIATE = per-channel encode then concat):
    for ch in range(c):
        enc = model.encode(context=ctx[:, ch, :], num_output_patches=1)[0]  # (b, ncp+1+1, 768) POST final-LN
        pooled_ch = enc[:, :ncp, :].mean(1)                                  # (b, 768) content-pool per channel
    pooled = concat(pooled_ch...)                                            # (b, c*768)
    logits = Linear(c*768, C)(pooled) ; loss = CrossEntropy(logits, y)
``model.encode`` is fully differentiable (extraction wraps it in no_grad EXTERNALLY), so gradients flow
into the whole ~119M backbone; the SAME backbone processes every channel (weights shared, gradients
accumulate over channels). K=1 (num_output_patches=1) is identical to the K used when run_task_shift
extracts the classification features — attention is non-causal, so K MUST match FT<->probe. The pooled
state read is the POST-final-LN state, i.e. exactly the L12+LN (index 13) probe point. For FordA (c=1)
this is the single-channel case and is numerically identical to the original univariate pipeline.

Trainable = full backbone + the fresh Linear head. The native forecast head (output_patch_embedding)
is NOT in the classification graph -> it receives no gradient; we FREEZE it for tidiness so the drift
diagnostic and the Exp-B fslot probe (which reads the ENCODER, which DOES change) stay clean.
Optimizer: AdamW, TWO param groups (backbone LR, head LR from the registry / CLI), linear decay,
warmup 5%, grad-clip 1.0, seed 0. Epochs/batch/lrs default per-source (CLS_SPECS) and are CLI-overridable.

CHECKPOINT RULE (user, 2026-08-18 — deterministic, NOT tuned on any probe curve):
    stage1_cls_early = the END-OF-EPOCH-1 checkpoint (deterministic).
    stage2_cls_late  = the BEST-VAL-ACCURACY checkpoint STRICTLY AFTER stage1, provided backbone
                       drift has INCREASED vs stage1. If no such later stage exists (best-val <= stage1,
                       or drift did not grow) -> the VALIDITY GATE FAILS and we REPORT it (do NOT force
                       a late stage). Deciding uses val-acc + backbone drift only, never a forecasting curve.

Runnable (the C1 job calls this):
    python -m probing.finetune_cls --cls-source uwave          # per-source defaults from CLS_SPECS
    python -m probing.finetune_cls --cls-source handwriting --epochs 80 --batch-size 16

CPU/synthetic contracts live in tests/test_task_shift.py (no model, no GPU, no download).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probing.cls_data import CLS_SPECS, load_cls, ncp_for_length
from probing.config import OUTPUT_PATCH_SIZE
from probing.finetune import (FT_DEFAULTS, MODEL_ID, _select_device, checkpoint_hash,
                              default_ckpt_root, default_out_root, ft_cache_prefix,
                              load_trainable_pipeline, param_drift, save_checkpoint,
                              snapshot_reference_state)

STAGE1 = "stage1_cls_early"
STAGE2 = "stage2_cls_late"


def cls_source_label(tag: str) -> str:
    """Cache/checkpoint source label for a classification source (``forda`` -> ``forda_cls``). Disjoint
    from the BOOM domain-shift source (``boom``) and disjoint across classification sources, so no
    checkpoint/cache can ever collide (forda_cls / uwave_cls / handwriting_cls)."""
    return f"{tag}_cls"


# --------------------------------------------------------------------------- #
# small, unit-testable pieces
# --------------------------------------------------------------------------- #
def build_cls_head(d_model: int, n_classes: int, channels: int = 1, device=None) -> nn.Linear:
    """The classification head is a SINGLE Linear(d_model*channels, n_classes) — strictly linear, no
    hidden layer, no activation. The input is the per-channel content-pooled vectors concatenated
    (c*768). (Tested: isinstance nn.Linear, in_features == d_model*channels, forward == W x + b.)"""
    head = nn.Linear(d_model * channels, n_classes)
    return head.to(device) if device is not None else head


def pool_content_cls(hs: torch.Tensor, ncp: int) -> torch.Tensor:
    """Mean-pool the first ``ncp`` content tokens of a (b, P, d) hidden state -> (b, d). The ONE fixed
    per-channel pooling rule shared by the FT forward and the probe-feature extractor."""
    return hs[:, :ncp, :].mean(dim=1)


def encode_pool_concat(model, ctx: torch.Tensor, ncp: int, channels: int) -> torch.Tensor:
    """Per-channel univariate encode -> content-pool -> concat, the ONE multivariate representation
    rule (matches the old UEA per-channel-concat convention and the Exp-A feature extractor).

    ctx : (b, c, L) raw contexts on-device. Returns (b, c*768). Each channel goes through the SAME
    backbone (``model.encode(context=ctx[:, ch, :], num_output_patches=1)[0]`` = POST final-LN), so
    gradients from every channel flow into the shared weights. Differentiable (no no_grad here)."""
    pooled = []
    for ch in range(channels):
        enc_out, *_ = model.encode(context=ctx[:, ch, :], num_output_patches=1)
        pooled.append(pool_content_cls(enc_out[0], ncp).float())
    return torch.cat(pooled, dim=1)


def build_optimizer_param_groups(model, head, backbone_lr: float, head_lr: float):
    """Two AdamW param groups: all TRAINABLE backbone params at backbone_lr, head params at head_lr.
    Frozen params (e.g. output_patch_embedding) are excluded by the requires_grad filter. Weight
    decay follows FT_DEFAULTS (0.0). Returned as the list AdamW consumes."""
    backbone = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": backbone, "lr": backbone_lr, "weight_decay": FT_DEFAULTS["weight_decay"]},
        {"params": list(head.parameters()), "lr": head_lr, "weight_decay": FT_DEFAULTS["weight_decay"]},
    ]


def backbone_drift_scalar(drift: dict) -> float:
    """Collapse the per-group param_drift dict into ONE backbone L2 magnitude (root-sum-square of the
    per-group L2s). The native head is frozen -> contributes 0. Used to test 'drift increased vs
    stage1' for the stage2 rule."""
    return math.sqrt(sum(g["l2"] ** 2 for g in drift.values()))


# --------------------------------------------------------------------------- #
# evaluation on the fixed validation split (dropout OFF)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_cls_val(model, head, X, y, ncp, channels, batch_size, device, autocast_ctx) -> tuple[float, float]:
    """(val_accuracy, val_cross_entropy) over the fixed validation split, in eval mode. X is (n, c, L)."""
    was_training = model.training
    model.eval(); head.eval()
    n_correct, ce_sum, count = 0, 0.0, 0
    try:
        for b0 in range(0, len(X), batch_size):
            ctx = torch.from_numpy(np.ascontiguousarray(X[b0:b0 + batch_size])).to(
                device=device, dtype=torch.float32)
            yb = torch.from_numpy(np.ascontiguousarray(y[b0:b0 + batch_size])).to(
                device=device, dtype=torch.long)
            with autocast_ctx():
                logits = head(encode_pool_concat(model, ctx, ncp, channels))
            n_correct += int((logits.argmax(dim=1) == yb).sum().item())
            ce_sum += float(F.cross_entropy(logits, yb, reduction="sum").item())
            count += len(yb)
    finally:
        model.train(was_training); head.train(was_training)
    return n_correct / max(count, 1), ce_sum / max(count, 1)


# --------------------------------------------------------------------------- #
# the classification fine-tuning run
# --------------------------------------------------------------------------- #
def finetune_cls(*, source: str = "forda", epochs: int | None = None, batch_size: int | None = None,
                 backbone_lr: float | None = None, head_lr: float | None = None, seed: int = 0,
                 warmup_ratio: float = 0.05, max_grad_norm: float = 1.0, ckpt_root=None, out_root=None,
                 device=None, model_id: str = MODEL_ID) -> dict:
    """Full classification fine-tuning of Chronos-2 on ``source``; write stage1_cls_early (end of epoch
    1) and, if the validity gate passes, stage2_cls_late (best-val after stage1 with increased backbone
    drift). epochs/batch_size/lrs default per-source (CLS_SPECS) when None. Returns the manifest dict.
    Checkpoints -> ckpt_root/<source>_cls/<stage>/; manifest + histories -> out_root/<source>_cls/."""
    spec = CLS_SPECS[source]
    tag = source
    src_label = cls_source_label(source)
    n_classes = spec["n_classes"]
    channels = spec["channels"]
    ncp = ncp_for_length(spec["length"])
    epochs = spec["epochs"] if epochs is None else epochs
    batch_size = spec["batch_size"] if batch_size is None else batch_size
    backbone_lr = spec["backbone_lr"] if backbone_lr is None else backbone_lr
    head_lr = spec["head_lr"] if head_lr is None else head_lr

    ckpt_root = default_ckpt_root() if ckpt_root is None else Path(ckpt_root)
    out_root = default_out_root() if out_root is None else Path(out_root)
    out_dir = out_root / src_label
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device(device)
    use_cuda = device.type == "cuda"
    has_sm80 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    if has_sm80:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def autocast_ctx():
        if has_sm80:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --- data (RAW (n, c, L) contexts; model normalizes internally) ---
    data = load_cls(source)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    n_train = len(Xtr)
    steps_per_epoch = math.ceil(n_train / batch_size)
    num_steps = steps_per_epoch * epochs
    num_warmup = int(round(warmup_ratio * num_steps))
    print(f"[ft-cls] {spec['aeon_name']}: {n_train} train / {len(Xva)} val | channels={channels} "
          f"ncp={ncp} n_classes={n_classes} batch={batch_size} epochs={epochs} "
          f"steps/epoch={steps_per_epoch} num_steps={num_steps} warmup={num_warmup} "
          f"backbone_lr={backbone_lr} head_lr={head_lr}")

    # --- trainable backbone + fresh head; freeze the native forecast head (no gradient anyway) ---
    pipeline = load_trainable_pipeline(model_id, device=device)
    model = pipeline.model
    assert model.chronos_config.output_patch_size == OUTPUT_PATCH_SIZE
    for name, p in model.named_parameters():
        if name.startswith("output_patch_embedding"):
            p.requires_grad_(False)                       # native head is outside the cls graph
    head = build_cls_head(model.config.d_model, n_classes, channels, device=device)
    reference_state = snapshot_reference_state(model)     # pretrained baseline for drift
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
        sum(p.numel() for p in head.parameters())

    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, head, backbone_lr, head_lr),
        betas=FT_DEFAULTS["adam_betas"], eps=FT_DEFAULTS["adam_eps"], fused=use_cuda)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup,
                                                num_training_steps=num_steps)

    def save_stage(stage_label):
        ckpt_dir = ckpt_root / src_label / stage_label
        h = save_checkpoint(pipeline, ckpt_dir)           # HF backbone (reloadable via from_pretrained)
        torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "in_features": model.config.d_model * channels, "n_classes": n_classes,
                    "channels": channels},
                   ckpt_dir / "cls_head.pt")               # the head at this checkpoint
        return str(ckpt_dir), h

    # --- manual epoch loop (FIXED finite classification set -> shuffled minibatches) ---
    step = 0
    train_hist: list[dict] = []
    epoch_hist: list[dict] = []
    v0_acc, v0_ce = eval_cls_val(model, head, Xva, yva, ncp, channels, batch_size, device, autocast_ctx)
    print(f"[ft-cls] epoch 0 (pretrained backbone + fresh head)  val_acc={v0_acc:.4f}  val_ce={v0_ce:.4f}")
    stage1_info = None
    best_late = None            # {epoch, step, val_acc, val_ce, hash, dir, drift_scalar, drift}
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_train)
        model.train(); head.train()
        for b0 in range(0, n_train, batch_size):
            bidx = perm[b0:b0 + batch_size]
            ctx = torch.from_numpy(np.ascontiguousarray(Xtr[bidx])).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(np.ascontiguousarray(ytr[bidx])).to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = head(encode_pool_concat(model, ctx, ncp, channels))
                loss = F.cross_entropy(logits, yb)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
                max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            train_hist.append({"step": step, "epoch": epoch,
                               "train_loss": float(loss.detach().to(torch.float64)),
                               "lr_backbone": float(scheduler.get_last_lr()[0]),
                               "lr_head": float(scheduler.get_last_lr()[1]),
                               "grad_norm": float(gnorm)})

        # --- end of epoch: eval + drift ---
        va_acc, va_ce = eval_cls_val(model, head, Xva, yva, ncp, channels, batch_size, device, autocast_ctx)
        drift = param_drift(model, reference_state)
        d_scalar = backbone_drift_scalar(drift)
        epoch_hist.append({"epoch": epoch, "step": step, "val_acc": va_acc, "val_ce": va_ce,
                           "backbone_drift_l2": d_scalar})
        print(f"[ft-cls] epoch {epoch:>2}/{epochs}  step {step}  val_acc={va_acc:.4f}  "
              f"val_ce={va_ce:.4f}  backbone_drift_l2={d_scalar:.4g}")

        if epoch == 1:                                    # stage1 = end of epoch 1 (deterministic)
            cdir, h = save_stage(STAGE1)
            stage1_info = {"step": step, "epoch": 1, "stage": STAGE1, "checkpoint_hash": h,
                           "checkpoint_dir": cdir, "cache_prefix": ft_cache_prefix(tag, src_label, STAGE1, h),
                           "val_acc": va_acc, "val_ce": va_ce, "backbone_drift_l2": d_scalar,
                           "param_drift": drift}
            print(f"[ft-cls] saved {STAGE1} @ epoch 1 step {step}: hash={h} val_acc={va_acc:.4f}")
        elif d_scalar > stage1_info["backbone_drift_l2"]:  # candidate late stage: drift increased vs stage1
            if best_late is None or va_acc > best_late["val_acc"]:
                cdir, h = save_stage(STAGE2)              # overwrite -> stage2 = best-val-after-stage1
                best_late = {"step": step, "epoch": epoch, "stage": STAGE2, "checkpoint_hash": h,
                             "checkpoint_dir": cdir,
                             "cache_prefix": ft_cache_prefix(tag, src_label, STAGE2, h),
                             "val_acc": va_acc, "val_ce": va_ce, "backbone_drift_l2": d_scalar,
                             "param_drift": drift}
                print(f"[ft-cls]   -> new {STAGE2} @ epoch {epoch}: hash={h} val_acc={va_acc:.4f} "
                      f"(drift {d_scalar:.4g} > stage1 {stage1_info['backbone_drift_l2']:.4g})")

    # --- validity gate (FT-side evidence only; user rule) ---
    late_found = best_late is not None
    val_acc_improved = late_found and best_late["val_acc"] > stage1_info["val_acc"]
    drift_increased = late_found and best_late["backbone_drift_l2"] > stage1_info["backbone_drift_l2"]
    if not late_found:
        verdict = ("FAIL_NO_LATE_STAGE: no epoch>1 had increased backbone drift over stage1 — "
                   "Chronos-2 may be robust to classification FT, or the budget is too short. "
                   "Do NOT force a late stage; report this and reconsider LR/epochs.")
    elif not val_acc_improved:
        verdict = (f"WEAK: a later stage exists (epoch {best_late['epoch']}) but its val_acc "
                   f"{best_late['val_acc']:.4f} did not rise above stage1 {stage1_info['val_acc']:.4f} — "
                   "specialization is marginal; report plainly, do not tune to force it.")
    else:
        verdict = "PASS: stage1 (epoch 1) + a best-val late stage with increased backbone drift and rising val-acc."
    print(f"[ft-cls] VALIDITY GATE -> {verdict}")

    checkpoints = {STAGE1: {k: stage1_info[k] for k in
                            ("step", "epoch", "stage", "checkpoint_hash", "checkpoint_dir",
                             "cache_prefix", "val_acc", "val_ce", "backbone_drift_l2", "param_drift")}}
    if late_found:
        checkpoints[STAGE2] = {k: best_late[k] for k in
                               ("step", "epoch", "stage", "checkpoint_hash", "checkpoint_dir",
                                "cache_prefix", "val_acc", "val_ce", "backbone_drift_l2", "param_drift")}
    else:
        # no late stage: remove any stale stage2 dir so downstream fails loud instead of reading a
        # mislabeled leftover
        stale = ckpt_root / src_label / STAGE2
        if stale.exists():
            shutil.rmtree(stale)

    manifest = {
        "experiment": "task_shift_classification_ft", "source": src_label, "cls_source": source,
        "tag": tag, "aeon_name": spec["aeon_name"], "model_id": model_id, "finetune_mode": "full",
        "trainable_params": int(n_trainable), "n_classes": n_classes, "channels": channels, "ncp": ncp,
        "head": "Linear(d_model*channels, n_classes)  # strictly linear, no hidden layer",
        "native_head_frozen": True,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "backbone_lr": backbone_lr,
            "head_lr": head_lr, "num_steps": num_steps, "steps_per_epoch": steps_per_epoch,
            "warmup_ratio": warmup_ratio, "warmup_steps": num_warmup,
            "lr_scheduler_type": "linear", "max_grad_norm": max_grad_norm,
            "optimizer": "adamw_torch_fused" if use_cuda else "adamw",
            "adam_betas": list(FT_DEFAULTS["adam_betas"]), "adam_eps": FT_DEFAULTS["adam_eps"],
            "weight_decay": FT_DEFAULTS["weight_decay"], "seed": seed,
            "bf16": bool(has_sm80), "tf32": bool(has_sm80),
        },
        "geometry": {"length": spec["length"], "channels": channels, "ncp": ncp,
                     "num_output_patches": 1, "P": OUTPUT_PATCH_SIZE,
                     "multivariate_convention": "per-channel encode -> concat -> c*768"},
        "stage_rule": ("stage1_cls_early=end-of-epoch-1 (deterministic); stage2_cls_late=best-val "
                       "STRICTLY after stage1 with increased backbone drift, else FAIL/report"),
        "validity": {"verdict": verdict, "late_stage_found": late_found,
                     "val_acc_improved": bool(val_acc_improved), "drift_increased": bool(drift_increased),
                     "stage0_val_acc": v0_acc, "stage0_val_ce": v0_ce,
                     "stage1_val_acc": stage1_info["val_acc"],
                     "stage2_val_acc": best_late["val_acc"] if late_found else None},
        "data": data["meta"],
        "checkpoints": checkpoints,
        "device": str(device), "wall_seconds": time.time() - t0,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "training_history.json").write_text(
        json.dumps({"train": train_hist, "epoch": epoch_hist,
                    "stage0": {"val_acc": v0_acc, "val_ce": v0_ce}}, indent=2))
    print(f"[ft-cls] DONE in {manifest['wall_seconds']:.0f}s  stages={list(checkpoints)}")
    print(f"[ft-cls] manifest -> {out_dir/'manifest.json'}")
    return manifest


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Classification full fine-tuning of Chronos-2 (TASK-SHIFT).")
    ap.add_argument("--cls-source", default="forda", choices=sorted(CLS_SPECS),
                    help="classification source (forda | uwave | handwriting)")
    ap.add_argument("--epochs", type=int, default=None, help="override CLS_SPECS default")
    ap.add_argument("--batch-size", type=int, default=None, help="override CLS_SPECS default")
    ap.add_argument("--backbone-lr", type=float, default=None, help="override CLS_SPECS default")
    ap.add_argument("--head-lr", type=float, default=None, help="override CLS_SPECS default")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    finetune_cls(source=a.cls_source, epochs=a.epochs, batch_size=a.batch_size,
                 backbone_lr=a.backbone_lr, head_lr=a.head_lr, warmup_ratio=a.warmup_ratio,
                 seed=a.seed, device=a.device)


if __name__ == "__main__":
    main()
