"""Full fine-tuning of frozen Chronos-2 — FT-specialization experiment, STAGE A only.

Scientific frame (notes/PLAN.md, FT-SPECIALIZATION): does domain specialization create a
tunnel-like FT-OOD degradation that broadly-pretrained Chronos-2 does NOT have? Stage A produces
the *intervention*: one full fine-tuning run on a single source (Electricity for the pilot), with
two checkpoints of the SAME run — ``stage1_ft_early`` (300 optimizer steps) and ``stage2_ft_late``
(1000 steps). Stage B (the 7-target frozen-probe transfer) is NOT built here.

Locked full-FT recipe — the OFFICIAL Chronos-2 defaults, verified from the installed 2.3.1
``Chronos2Pipeline.fit`` (see FT_DEFAULTS):

    finetune_mode = full  (entire ~119M model trainable: encoder + input_patch_embedding
                           + REG-embed (shared) + encoder.final_layer_norm + native head)
    learning_rate = 1e-6            lr_scheduler = linear decay to 0, warmup_steps = 0
    optimizer     = AdamW (betas 0.9/0.999, eps 1e-8, wd 0.0; fused on CUDA = adamw_torch_fused)
    max_grad_norm = 1.0             gradient_accumulation_steps = 1
    batch_size    = 64 (reduced from the official 256; see PLAN, and note below)   seed = 0
    bf16 + tf32 on sm80 (A100)      checkpoints @ 300 / 1000 optimizer steps

FT DATA (REDESIGN 2026-08-11). The two fixed-window pilots (batch 256, batch 64) both OVERFIT: the
FT corpus wrongly reused the 1073 cluster-balanced PROBE windows, so cycling them 60/200 epochs
memorized (train_loss down, ft_val up from the pretrained baseline). Fix: fine-tune from the
COMPLETE source histories using Chronos-2's OWN ``Chronos2Dataset`` TRAIN sampler — random cut
points across each full series (dataset.py:193-195) — fed our LEAKAGE-TRUNCATED per-series histories.
Per series we take rolling origins (_rolling_valid_starts): test=starts[-1], probe-val=starts[-2],
ft_val=starts[-3] (latest train origin). The training history is truncated to ``s[:starts[-3]+C]``
(strictly before the ft_val forecast target), so every sampled window's target ends <= that cutoff <
probe-val target < test target — the preserved probe-val/test windows are untouched. This yields
~millions of distinct training windows (vs 1073 fixed) -> ~zero repetition at 1000x64 presentations.
We still do NOT call ``pipeline.fit()`` (it samples across the full raw series and would hit the
preserved regions); we drive the official sampler over our truncated histories in a manual loop, and
select on a FIXED ``ft_val`` = each series' starts[-3] window. min_past = C = 512, so every training
window has a full 512-step context matching the probe/eval regime.

The model applies its OWN instance-norm + arcsinh internally, so fine-tuning receives RAW context and
RAW future — the sampler yields raw slices; ft_val slices raw context/target straight from the series.

FT SOURCE (2026-08-11). The active pilot source is a PT-OOD dataset (BOOM), NOT a PT-ID one: the
full-history Electricity pilot showed official-scale full-FT CANNOT reduce Chronos-2's loss on data it
was pretrained on (flat train/ft_val, ~0.4% loss-flat drift even at 3x LR). To get a MEANINGFUL
specialization the source must be something the model has not seen. build_ft_data routes PT-OOD tags
(OOD_TARGET_TAGS) through load_ood_target_series; PT-ID tags through load_seen_series.

Runnable (the pilot job calls this):

    python -m probing.finetune --source boom            # PT-OOD full-history FT, batch 64, seed 0

CPU/synthetic contracts live in tests/test_ft_specialization.py (no model, no GPU).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from probing.config import OUTPUT_PATCH_SIZE, REPO_ROOT
from probing.id_data import (load_seen_series, load_ood_target_series,
                             _rolling_valid_starts, OOD_TARGET_TAGS)

MODEL_ID = "amazon/chronos-2"

# PT-ID sources use the rolling-origin geometry of this set. build_ft_data reuses that split's origin
# rule (_rolling_valid_starts) directly, so it forces this set before loading a PT-ID series.
FT_DATASET_SET = "extended_v3_rolling"

# extended_v3_rolling geometry (matches build_windows defaults). build_ft_data reuses these directly
# via _rolling_valid_starts rather than building the heavy rolling windows.
FT_C = 512            # context length
FT_H = 64             # forecast horizon
SIGMA_EPS = 1e-6      # non-constant-context threshold (id_data default)

# Source label -> HF/OOD tag. The ACTIVE pilot source is a PT-OOD dataset (BOOM): full-FT on data the
# model has NOT seen has real room to specialize, unlike the PT-ID sources where official-scale full-FT
# cannot reduce loss (electricity pilot = flat train/ft_val, ~0.4% drift; see PLAN). PT-OOD tags route
# build_ft_data through load_ood_target_series (needs OOD_TARGET_ROOT staged); PT-ID tags through
# load_seen_series.
SOURCE_TAGS = {
    "boom": "boom_hourly",                        # PT-OOD (documented-unseen Datadog telemetry) — active
    "electricity": "monash_electricity_hourly",   # PT-ID robustness baseline (full-FT can't specialize it)
    # DEFERRED PT-ID (same in-pretraining wall as electricity): uber / m4 / windfarms
}

# Optimizer-step budget -> stage label. Both checkpoints are of the SAME run (PLAN §2).
DEFAULT_CHECKPOINT_STEPS = {300: "stage1_ft_early", 1000: "stage2_ft_late"}

# Official Chronos-2 full-FT defaults, verified from the installed 2.3.1 pipeline.fit()
# (transformers 5 -> warmup_steps=0). Recorded verbatim into every manifest.
FT_DEFAULTS = dict(
    finetune_mode="full",
    learning_rate=1e-6,
    num_steps=1000,
    batch_size=256,
    lr_scheduler_type="linear",
    warmup_steps=0,
    max_grad_norm=1.0,
    gradient_accumulation_steps=1,
    adam_betas=(0.9, 0.999),
    adam_eps=1e-8,
    weight_decay=0.0,
    seed=0,
)


# --------------------------------------------------------------------------- #
# paths (large checkpoints + FT feature caches -> $SCRATCH, gitignored/regenerable;
# small manifests + histories -> project results/ namespace)
# --------------------------------------------------------------------------- #
def default_ckpt_root() -> Path:
    """Where the ~478 MB safetensors checkpoints land. FT_CKPT_ROOT wins; else
    $SCRATCH/chronos2/ft_specialization (Narval); else a local fallback (gitignore it)."""
    env = os.environ.get("FT_CKPT_ROOT")
    if env:
        return Path(env)
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "chronos2" / "ft_specialization"
    return REPO_ROOT / "ft_checkpoints"


def default_out_root() -> Path:
    """Small, committable artifacts: manifest + train/ft_val histories + param-drift."""
    return REPO_ROOT / "results" / "ft_specialization"


def ft_cache_prefix(tag: str, source: str, stage: str, hash8: str) -> str:
    """Collision-proof FT feature-cache namespace. Cannot overlap the pretrained ``IDF_<tag>*``
    or ``IDF_<tag>__<set>`` prefixes, so a fine-tuned extraction never reads/writes the frozen
    model's caches. Carries source + stage + checkpoint hash per PLAN §7."""
    return f"IDF_{tag}__ft__{source}__{stage}__{hash8}"


# --------------------------------------------------------------------------- #
# FT data: full-history corpus (leakage-truncated) + fixed ft_val
# --------------------------------------------------------------------------- #
def build_ft_data(tag: str, min_past: int) -> dict:
    """Leakage-safe full-history FT data (REDESIGN 2026-08-11).

    The earlier fixed 1073-window corpus was a cluster-balanced PROBE subsample; cycling it 60/200
    epochs memorized (both pilots overfit). Here the corpus is the COMPLETE source histories, sampled
    with Chronos-2's OWN random-cut-point sampler (see finetune() -> Chronos2Dataset), with per-series
    LEAKAGE truncation:

      * PT-OOD source (tag in OOD_TARGET_TAGS, e.g. boom_hourly): series via load_ood_target_series
        (needs OOD_TARGET_ROOT). The model has NOT seen these, so full-FT can actually specialize.
        PT-ID source: series via load_seen_series on the extended_v3_rolling roster.
      * rolling origins per series (_rolling_valid_starts): test=starts[-1], probe-val=starts[-2],
        ft_val=starts[-3] (latest train origin). Series with < 3 origins are skipped.
      * cutoff = starts[-3]+C = the ft_val target start. The training history is ``s[:cutoff]``, so any
        sampled window's target ends <= cutoff < probe-val target < test target -> nothing preserved is
        touched. C=512, H=64 unchanged. (Sampled windows may contain NaN gaps for OOD sources; the
        model's InstanceNorm is nanmean/nanstd-robust and the loss masks NaN targets — the official
        Chronos2Dataset training path handles missing data.)
      * ft_val = the FIXED per-series window at context-start starts[-3] (one per eligible series):
        raw context s[starts[-3] : +C], raw future s[+C : +C+H].

    Returns {train_histories: [1-D float32 tensors], X_ft_val, y_ft_val, series_ft_val, meta}. meta
    reports the eligible-origin and unique-training-window counts (new vs the rejected 1073 fixed)."""
    from probing import config
    if tag in OOD_TARGET_TAGS:                           # PT-OOD source (unseen -> real room to specialize)
        loaded = load_ood_target_series(tag)             # reads OOD_TARGET_ROOT-staged arrow shards
        series = loaded["series"]
        source_kind = f"pt_ood:{loaded['cluster_unit']}"
    else:                                                # PT-ID source (extended_v3_rolling roster)
        if config.DATASET_SET != FT_DATASET_SET:         # _rolling_valid_starts geometry is set-defined
            print(f"[ft] activating dataset set {FT_DATASET_SET!r} for the rolling-origin geometry "
                  f"(was {config.DATASET_SET!r})")
            config.set_dataset_set(FT_DATASET_SET)
        series = load_seen_series(tag)
        source_kind = "pt_id:extended_v3_rolling"
    C, H = FT_C, FT_H

    train_hists, Xv, yv, val_ids = [], [], [], []
    n_eligible = 0
    n_eligible_train_origins = 0
    n_unique_train_windows = 0
    for i, s in enumerate(series):
        s = np.asarray(s, np.float64)
        starts = _rolling_valid_starts(s, C, H, SIGMA_EPS)
        if len(starts) < 3:                              # need test + probe-val + ft_val
            continue
        n_eligible += 1
        n_eligible_train_origins += len(starts) - 2      # H-spaced TRAIN origins (excl probe-val, test)
        ftv = starts[-3]                                 # latest train origin
        cutoff = ftv + C                                 # ft_val target start; truncate strictly before it
        assert cutoff <= starts[-2] + C and cutoff < starts[-1] + C, \
            f"{tag}: series {i} cutoff {cutoff} does not precede probe-val/test targets"
        hist = s[:cutoff]                                # leakage-truncated FT-train history
        if len(hist) >= min_past + H:                    # yields >= 1 samplable full-context window
            train_hists.append(torch.tensor(hist, dtype=torch.float32))
            n_unique_train_windows += len(hist) - min_past - H + 1
        Xv.append(s[ftv:ftv + C]); yv.append(s[ftv + C:ftv + C + H]); val_ids.append(i)

    if not train_hists:
        raise RuntimeError(f"{tag}: no leakage-safe FT-train history >= min_past+H ({min_past + H}) — "
                           f"lower min_past or check the data")
    if not Xv:
        raise RuntimeError(f"{tag}: empty ft_val — no eligible series (>= 3 rolling origins)")

    meta = {
        "tag": tag, "source_kind": source_kind, "C": C, "H": H, "min_past": min_past,
        "n_series_total": len(series), "n_eligible_series": n_eligible,
        "n_ft_train_series": len(train_hists), "n_ft_val": len(Xv),
        "n_eligible_train_origins": n_eligible_train_origins,   # H-spaced origins (old fixed regime)
        "n_unique_train_windows": int(n_unique_train_windows),  # random-cut-point corpus (new)
        "rejected_fixed_window_corpus": 1073,                   # the overfit probe subsample (record)
        "leakage_rule": "train history truncated at ft_val target start (starts[-3]+C); "
                        "probe-val (starts[-2]) and test (starts[-1]) windows untouched",
        "sampler": "chronos.chronos2.Chronos2Dataset TRAIN random cut points (dataset.py:193-195)",
    }
    return {"train_histories": train_hists,
            "X_ft_val": np.stack(Xv).astype(np.float32),
            "y_ft_val": np.stack(yv).astype(np.float32),
            "series_ft_val": np.asarray(val_ids, np.int64), "meta": meta}


# --------------------------------------------------------------------------- #
# trainable pipeline (SEPARATE from the frozen get_pipeline singleton)
# --------------------------------------------------------------------------- #
def _select_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trainable_pipeline(model_id: str = MODEL_ID, device=None):
    """Fresh, fully-trainable Chronos-2 pipeline, independent of the frozen extraction singleton
    (which sets requires_grad_(False)). EVERY parameter requires grad = full fine-tuning."""
    from chronos import Chronos2Pipeline

    dev = _select_device(device)
    print(f"[ft] loading {model_id} (float32, TRAINABLE) -> {dev}")
    pipeline = Chronos2Pipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    model = pipeline.model
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ft] {n_trainable:,} trainable params  d_model={model.config.d_model} "
          f"d_ff={model.config.d_ff} num_layers={model.config.num_layers} "
          f"output_patch_size={model.chronos_config.output_patch_size}")
    return pipeline


def snapshot_reference_state(model) -> dict:
    """CPU float64 clone of the PRETRAINED weights, taken before any optimizer step — the baseline
    for the parameter-drift diagnostic. Detached, so training never touches it."""
    return {name: p.detach().to("cpu", torch.float64).clone()
            for name, p in model.named_parameters()}


# --------------------------------------------------------------------------- #
# parameter-drift diagnostic (per-block + head, vs pretrained)
# --------------------------------------------------------------------------- #
def _param_group(name: str) -> str:
    """Bucket a parameter name into an FT drift group: input embedding / REG embedding /
    per-encoder-block / final LayerNorm / native head."""
    if name.startswith("input_patch_embedding"):
        return "input_patch_embedding"
    if name.startswith("shared"):
        return "reg_embedding"
    if name.startswith("encoder.final_layer_norm"):
        return "final_layer_norm"
    if name.startswith("output_patch_embedding"):
        return "native_head"
    if name.startswith("encoder.block."):
        return f"block_{int(name.split('.')[2]):02d}"
    return "other"


def param_drift(model, reference_state: dict) -> dict:
    """Per-group L2 drift ||w_ft - w_pt|| and relative drift ||w_ft - w_pt|| / ||w_pt|| vs the
    pretrained reference. ``changed`` flags any non-zero movement (the Stage-A acceptance check)."""
    sq: dict[str, float] = {}
    refsq: dict[str, float] = {}
    for name, p in model.named_parameters():
        g = _param_group(name)
        w = p.detach().to("cpu", torch.float64)
        r = reference_state[name]
        sq[g] = sq.get(g, 0.0) + float(((w - r) ** 2).sum())
        refsq[g] = refsq.get(g, 0.0) + float((r ** 2).sum())
    out = {}
    for g in sorted(sq):
        l2 = math.sqrt(sq[g])
        out[g] = {"l2": l2,
                  "relative": (l2 / math.sqrt(refsq[g])) if refsq[g] > 0 else float("nan"),
                  "changed": bool(l2 > 0.0)}
    return out


# --------------------------------------------------------------------------- #
# checkpoint save + identity hash
# --------------------------------------------------------------------------- #
def save_checkpoint(pipeline, ckpt_dir) -> str:
    """Save an HF-safetensors checkpoint reloadable via Chronos2Pipeline.from_pretrained, mirroring
    the official fit() save flow (persist chronos_config into model.config first). Returns the
    checkpoint hash."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = pipeline.model
    model.config.chronos_config = dict(model.chronos_config.__dict__)   # so reload sees it
    pipeline.save_pretrained(ckpt_dir)
    return checkpoint_hash(ckpt_dir)


def checkpoint_hash(ckpt_dir) -> str:
    """sha256(model.safetensors)[:8] — the checkpoint identity carried in cache keys and records."""
    p = Path(ckpt_dir) / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"no model.safetensors under {ckpt_dir} to hash")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


# --------------------------------------------------------------------------- #
# native forward loss (RAW context + RAW future -> Chronos-2 pinball loss)
# --------------------------------------------------------------------------- #
def _forward_loss(model, X, y, K, device, autocast_ctx):
    ctx = torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=torch.float32)
    fut = torch.from_numpy(np.ascontiguousarray(y)).to(device=device, dtype=torch.float32)
    with autocast_ctx():
        out = model(context=ctx, future_target=fut, num_output_patches=K)   # group_ids=None -> univariate
    return out.loss, ctx.shape[0]


@torch.no_grad()
def eval_ft_val(model, X, y, K, batch_size, device, autocast_ctx) -> float:
    """Dataset-mean native loss on ft_val, in eval mode (dropout OFF). The native loss is a
    batch-MEAN, so weight each batch by its size to recover the exact dataset mean."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for b0 in range(0, len(X), batch_size):
            loss, b = _forward_loss(model, X[b0:b0 + batch_size], y[b0:b0 + batch_size],
                                    K, device, autocast_ctx)
            total += float(loss.detach().to(torch.float64)) * b
            count += b
    finally:
        model.train(was_training)
    return total / max(count, 1)


# --------------------------------------------------------------------------- #
# the fine-tuning run
# --------------------------------------------------------------------------- #
def finetune(source: str, *, tag: str | None = None, num_steps: int = 1000,
             checkpoint_steps=None, batch_size: int = 64, learning_rate: float = 1e-6,
             seed: int = 0, max_grad_norm: float = 1.0, eval_every: int = 100,
             logging_steps: int = 100, min_past: int = FT_C, ckpt_root=None, out_root=None,
             device=None, model_id: str = MODEL_ID) -> dict:
    """Run ONE full fine-tuning of Chronos-2 on `source`, checkpointing at each of
    `checkpoint_steps` (default 300 -> stage1_ft_early, 1000 -> stage2_ft_late). Training windows
    are drawn by Chronos-2's own random-cut-point sampler over the leakage-truncated full histories
    (build_ft_data); selection is on the fixed ft_val. Writes each checkpoint under
    ckpt_root/<source>/<stage>/ and a manifest + histories under out_root/<source>/. Returns the
    manifest dict."""
    tag = SOURCE_TAGS[source] if tag is None else tag
    checkpoint_steps = dict(DEFAULT_CHECKPOINT_STEPS) if checkpoint_steps is None else dict(checkpoint_steps)
    ckpt_root = default_ckpt_root() if ckpt_root is None else Path(ckpt_root)
    out_root = default_out_root() if out_root is None else Path(out_root)
    out_dir = out_root / source
    out_dir.mkdir(parents=True, exist_ok=True)

    if max(checkpoint_steps) > num_steps:
        raise ValueError(f"checkpoint step {max(checkpoint_steps)} exceeds num_steps {num_steps}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = _select_device(device)
    use_cuda = device.type == "cuda"
    has_sm80 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    if has_sm80:
        torch.backends.cuda.matmul.allow_tf32 = True    # official fit: tf32 on sm80
        torch.backends.cudnn.allow_tf32 = True

    def autocast_ctx():
        if has_sm80:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)   # official fit: bf16 on sm80
        return contextlib.nullcontext()

    # --- data: full-history FT corpus (leakage-truncated) + fixed ft_val ---
    data = build_ft_data(tag, min_past)
    train_hists = data["train_histories"]
    X_va, y_va = data["X_ft_val"], data["y_ft_val"]
    H = data["meta"]["H"]
    K = math.ceil(H / OUTPUT_PATCH_SIZE)
    n_unique = data["meta"]["n_unique_train_windows"]
    coverage = num_steps * batch_size / max(n_unique, 1)   # << 1 -> ~no training-window repetition
    print(f"[ft] {source} ({tag}): {data['meta']['n_ft_train_series']} train series, "
          f"{n_unique:,} unique training windows (was 1073 fixed) ; ft_val={len(X_va)} windows | "
          f"H={H} K={K} batch={batch_size} min_past={min_past} coverage={coverage:.4f}")

    # --- trainable model + pretrained reference ---
    pipeline = load_trainable_pipeline(model_id, device=device)
    model = pipeline.model
    assert model.chronos_config.output_patch_size == OUTPUT_PATCH_SIZE, (
        f"model output_patch_size {model.chronos_config.output_patch_size} != config "
        f"OUTPUT_PATCH_SIZE {OUTPUT_PATCH_SIZE}")
    reference_state = snapshot_reference_state(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, betas=FT_DEFAULTS["adam_betas"],
                                  eps=FT_DEFAULTS["adam_eps"], weight_decay=FT_DEFAULTS["weight_decay"],
                                  fused=use_cuda)   # fused on CUDA == adamw_torch_fused
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=num_steps)   # linear decay, warmup 0

    # --- manual training loop over Chronos-2's OWN random-cut-point sampler ---
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    np.random.seed(seed)                    # Chronos2Dataset TRAIN draws from the global np.random stream
    ds = Chronos2Dataset(train_hists, context_length=FT_C, prediction_length=FT_H,
                         batch_size=batch_size, output_patch_size=OUTPUT_PATCH_SIZE,
                         min_past=min_past, mode=DatasetMode.TRAIN)
    batch_iter = iter(ds)                    # infinite in TRAIN mode

    train_hist: list[dict] = []
    val_hist: list[dict] = [{"step": 0, "ft_val_loss": eval_ft_val(model, X_va, y_va, K,
                                                                   batch_size, device, autocast_ctx)}]
    checkpoints: dict[str, dict] = {}
    t0 = time.time()
    model.train()
    print(f"[ft] step 0  ft_val_loss={val_hist[0]['ft_val_loss']:.5f}  (pretrained baseline)")
    for step in range(1, num_steps + 1):
        b = next(batch_iter)
        ctx = b["context"].to(device=device, dtype=torch.float32)          # (batch, C) raw
        fut = b["future_target"].to(device=device, dtype=torch.float32)    # (batch, H) raw
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            loss = model(context=ctx, future_target=fut,
                         num_output_patches=b["num_output_patches"]).loss   # group_ids default -> univariate
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        lr = float(scheduler.get_last_lr()[0])
        train_hist.append({"step": step, "train_loss": float(loss.detach().to(torch.float64)),
                           "lr": lr, "grad_norm": float(grad_norm)})

        is_ckpt = step in checkpoint_steps
        if step % eval_every == 0 or is_ckpt or step == num_steps:
            v = eval_ft_val(model, X_va, y_va, K, batch_size, device, autocast_ctx)
            val_hist.append({"step": step, "ft_val_loss": v})
            model.train()
        if step % logging_steps == 0 or is_ckpt:
            print(f"[ft] step {step:>4}/{num_steps}  "
                  f"train_loss={train_hist[-1]['train_loss']:.5f}  "
                  f"ft_val_loss={val_hist[-1]['ft_val_loss']:.5f}  "
                  f"lr={lr:.2e}  grad_norm={train_hist[-1]['grad_norm']:.3f}")
        if is_ckpt:
            stage = checkpoint_steps[step]
            ckpt_dir = ckpt_root / source / stage
            h = save_checkpoint(pipeline, ckpt_dir)
            drift = param_drift(model, reference_state)
            checkpoints[stage] = {
                "step": step, "stage": stage, "checkpoint_hash": h,
                "checkpoint_dir": str(ckpt_dir),
                "cache_prefix": ft_cache_prefix(tag, source, stage, h),
                "ft_val_loss": val_hist[-1]["ft_val_loss"],
                "coverage_fraction": step * batch_size / max(n_unique, 1),
                "param_drift": drift,
            }
            print(f"[ft] saved {stage} @ step {step}: hash={h}  ft_val={val_hist[-1]['ft_val_loss']:.5f}  "
                  f"-> {ckpt_dir}")

    best = min(val_hist, key=lambda r: r["ft_val_loss"])
    manifest = {
        "experiment": "ft_specialization_stageA",
        "source": source, "tag": tag,
        "model_id": model_id, "finetune_mode": "full", "trainable_params": int(n_trainable),
        "model_config": {
            "d_model": int(model.config.d_model), "d_ff": int(model.config.d_ff),
            "num_layers": int(model.config.num_layers),
            "num_quantiles": int(model.num_quantiles),
            "output_patch_size": int(model.chronos_config.output_patch_size),
            "dropout_rate": float(model.config.dropout_rate),
        },
        "hyperparameters": {
            "learning_rate": learning_rate, "num_steps": num_steps, "batch_size": batch_size,
            "min_past": min_past,
            "lr_scheduler_type": "linear", "warmup_steps": 0, "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw_torch_fused" if use_cuda else "adamw",
            "adam_betas": list(FT_DEFAULTS["adam_betas"]), "adam_eps": FT_DEFAULTS["adam_eps"],
            "weight_decay": FT_DEFAULTS["weight_decay"], "seed": seed,
            "bf16": bool(has_sm80), "tf32": bool(has_sm80),
            "matches_official_fit_defaults": (batch_size == FT_DEFAULTS["batch_size"]),
            "batch_size_note": ("official 256" if batch_size == FT_DEFAULTS["batch_size"]
                                else f"reduced from official 256 to {batch_size} (locked; see PLAN)"),
        },
        "geometry": {"C": data["meta"]["C"], "H": H, "P": OUTPUT_PATCH_SIZE, "K": K},
        "stage_labels": {str(s): l for s, l in checkpoint_steps.items()},
        "coverage_fraction": coverage,
        "data": data["meta"],
        "checkpoints": checkpoints,
        "best_ft_val_ckpt": {"step": best["step"], "ft_val_loss": best["ft_val_loss"],
                             "note": "DIAGNOSTIC only — both stage checkpoints are kept regardless"},
        "device": str(device), "wall_seconds": time.time() - t0,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "training_history.json").write_text(
        json.dumps({"train": train_hist, "ft_val": val_hist}, indent=2))
    print(f"[ft] DONE {source}: {num_steps} steps in {manifest['wall_seconds']:.0f}s  "
          f"stages={list(checkpoints)}  best_ft_val@step {best['step']}={best['ft_val_loss']:.5f}")
    print(f"[ft] manifest -> {out_dir/'manifest.json'}")
    return manifest


# --------------------------------------------------------------------------- #
# CLI (the pilot job calls this)
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Stage-A full fine-tuning of Chronos-2 (one source).")
    ap.add_argument("--source", default="boom", choices=sorted(SOURCE_TAGS),
                    help="fine-tuning source: 'boom' (PT-OOD, active) or 'electricity' (PT-ID baseline)")
    ap.add_argument("--num-steps", type=int, default=FT_DEFAULTS["num_steps"])
    ap.add_argument("--checkpoint-steps", type=int, nargs="+", default=[300, 1000],
                    help="optimizer steps at which to checkpoint (default 300 1000)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="locked-reduced from the official 256 (fixed-window pilots overfit; see PLAN)")
    ap.add_argument("--learning-rate", type=float, default=FT_DEFAULTS["learning_rate"])
    ap.add_argument("--seed", type=int, default=FT_DEFAULTS["seed"])
    ap.add_argument("--max-grad-norm", type=float, default=FT_DEFAULTS["max_grad_norm"])
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--min-past", type=int, default=FT_C,
                    help="min context steps per training window (512 = full-context, matches "
                         "probe/eval; 64 = native variable-context)")
    ap.add_argument("--device", default=None, help="override device (cuda/mps/cpu); default auto")
    ap.add_argument("--split-only", action="store_true",
                    help="build+report the FT data corpus (counts) then exit (no model load)")
    return ap.parse_args(argv)


def _stage_labels(steps) -> dict:
    steps = sorted(steps)
    return {s: DEFAULT_CHECKPOINT_STEPS.get(s, f"stage{i + 1}_ft_step{s}")
            for i, s in enumerate(steps)}


def main(argv=None):
    args = _parse_args(argv)
    tag = SOURCE_TAGS[args.source]
    if args.split_only:
        data = build_ft_data(tag, args.min_past)
        print(json.dumps(data["meta"], indent=2))
        return
    finetune(args.source, tag=tag, num_steps=args.num_steps,
             checkpoint_steps=_stage_labels(args.checkpoint_steps),
             batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
             max_grad_norm=args.max_grad_norm, eval_every=args.eval_every,
             min_past=args.min_past, device=args.device)


if __name__ == "__main__":
    main()
