"""Latency, throughput and memory of a depth-truncated Chronos-2 -- MEASURED, not derived.

The rest of this project never builds a truncated model: every layerwise number comes from running
the FULL 12-block encoder with forward hooks and applying the readout offline to the captured
layer-l states. That is mathematically identical for accuracy (the encoder is a pure feedforward
stack, so blocks l+1..12 cannot influence block l's output) but it means the truncated model has
never actually been instantiated -- and you cannot time a model that does not exist. This driver
builds it for real:

    model.encoder.block = ModuleList(block[:l])                  # blocks l+1..12 are DROPPED
    model.encoder.final_layer_norm = Sequential(adapter, rms)    # h_l -> A_l -> RMSNorm -> head

The second line is needed because Chronos2Encoder.forward applies final_layer_norm AFTER the last
block, so wrapping it reproduces the adapter path exactly. ``--verify`` proves the construction is
right: it runs the full model with a hook on block l and asserts RMSNorm(h_l) equals the truncated
model's encoder output element-wise. That is the assumption the entire paper rests on, checked.

Everything else here is a measurement, and measurements are easy to get wrong:
  * CUDA is asynchronous -- every timed region is bracketed by torch.cuda.synchronize(), or you
    time the Python loop and every depth looks identical;
  * the first calls pay kernel autotuning and allocator growth, so warm-up iterations are discarded;
  * peak memory is torch.cuda.max_memory_allocated() after reset_peak_memory_stats(), NOT nvidia-smi
    (which reports the caching allocator's reservation and will not shrink with depth);
  * the model is RELOADED for each depth so the dropped blocks are actually freed -- otherwise the
    memory column measures nothing;
  * two timings are reported because they answer different questions: ``predict_ms`` is the full
    serving path (pipeline.predict_quantiles on host arrays: preprocessing + host->device transfer
    included) and ``encode_ms`` is the encoder forward alone on a device-resident tensor (transfer
    excluded). The methodology section needs both.

Expect FLOPs and latency to disagree. At C=512/H=64 the encoder sees 37 tokens of width 768, so the
model is plausibly launch-overhead and bandwidth bound rather than arithmetic bound, and latency may
not scale with depth/12 at all -- especially at batch 1. That is a legitimate finding; report it.

GPU / compute node ONLY (loads amazon/chronos-2). Do NOT run on a login node.

    sbatch -J lat job_latency.sh --verify
    sbatch -J lat job_latency.sh --depths 3 6 12 --batch-sizes 1 256 --reps 200
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from probing import model_size
from probing.config import OUTPUT_PATCH_SIZE, REPO_ROOT, SEED

OUT_ROOT = REPO_ROOT / "results" / "ext_v5_native_head_adapter" / "latency"
NHA_ADAPTERS = REPO_ROOT / "results" / "ext_v5_native_head_adapter" / "adapters"
MODEL_ID = "amazon/chronos-2"
C, H = 512, 64
K = -(-H // OUTPUT_PATCH_SIZE)
DEFAULT_DEPTHS = (1, 3, 6, 8, 10, 11, 12)
DEFAULT_BATCHES = (1, 32, 256)


# --------------------------------------------------------------------------- #
# environment + model
# --------------------------------------------------------------------------- #
def _cpu_model():
    """Host CPU model string. The timing is GPU-bound, but the methodology section states it."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def environment(device):
    """Every field the methodology section has to state, read from the machine that ran it."""
    env = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "python": platform.python_version(), "platform": platform.platform(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
           "cpu_model": _cpu_model(), "cpu_count_visible": os.cpu_count(),
           "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
           "slurm_mem": os.environ.get("SLURM_MEM_PER_NODE"),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           # no torch.compile, no AMP, no manual fusion anywhere in this harness
           "compiled": False, "autocast": False,
           "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
           "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
           "device": str(device), "model_id": MODEL_ID,
           "context_length": C, "horizon": H, "forecast_slots": K,
           "encoder_tokens": model_size.num_encoder_tokens(C, H)[2]}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        env |= {"gpu_name": p.name, "gpu_total_mib": p.total_memory / 2 ** 20,
                "gpu_capability": f"{p.major}.{p.minor}", "gpu_count": torch.cuda.device_count()}
        try:
            env["driver_version"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        except Exception as e:
            env["driver_version"] = f"unavailable ({type(e).__name__})"
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                           capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        env["git_commit"] = None
    return env


def load_pipeline(device):
    """Fresh (uncached) pipeline. Mirrors probing.extraction.get_pipeline's model id and dtype
    (float32 -- the precision every committed accuracy number was produced at), but deliberately
    bypasses its module-level singleton so each depth gets a clean model and a clean allocator."""
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipeline.model.to(device).eval()
    return pipeline


def load_adapter(tag, depth, device):
    """The fitted adapter if --adapt saved one, else identity init. Timing depends only on SHAPE,
    so a missing checkpoint changes no measurement -- it only disables the accuracy interpretation.
    (adapters/ is gitignored: 91 x 2.4MB, regenerable via run_native_head_adapter --adapt.)"""
    from probing.native_head_adapter import LinearAdapter
    a = LinearAdapter(768).to(device).eval()
    p = NHA_ADAPTERS / f"native_head_adapter__{tag}__L{depth:02d}.pt"
    if p.exists():
        a.load_state_dict(torch.load(p, map_location=device)["state_dict"])
        return a, str(p.relative_to(REPO_ROOT))
    return a, "identity init (no fitted adapter on disk; shapes and therefore timings are unaffected)"


def truncate(pipeline, depth, adapter=None):
    """Drop blocks depth+1..12 and splice the adapter in front of the final RMSNorm."""
    model = pipeline.model
    if not 0 <= depth <= model_size.NUM_BLOCKS:
        raise ValueError(f"depth must be in 0..{model_size.NUM_BLOCKS}, got {depth}")
    kept = list(model.encoder.block)[:depth]
    model.encoder.block = nn.ModuleList(kept)          # old ModuleList drops its refs -> freeable
    if adapter is not None:
        model.encoder.final_layer_norm = nn.Sequential(adapter, model.encoder.final_layer_norm)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


# --------------------------------------------------------------------------- #
# verification: truncation == hooking
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_truncation(device, depth, n=8, atol=0.0, rtol=0.0):
    """Assert the truncated encoder output equals RMSNorm(block-`depth` output of the FULL model).

    This is the claim every layerwise number in the paper depends on: reading layer l off the full
    model with a hook is the same thing as running a model truncated at l.
    """
    torch.manual_seed(SEED)
    ctx = torch.randn(n, C, device=device, dtype=torch.float32).cumsum(-1)   # non-constant contexts

    full = load_pipeline(device)
    captured = {}
    hook = full.model.encoder.block[depth - 1].register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("h", (out[0] if isinstance(out, tuple) else
                                                       getattr(out, "hidden_states", out)).detach()))
    try:
        full.model.encode(context=ctx, num_output_patches=K)
    finally:
        hook.remove()
    reference = full.model.encoder.final_layer_norm(captured["h"]).cpu()
    del full, captured
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None

    trunc = truncate(load_pipeline(device), depth, adapter=None)
    enc_out, *_ = trunc.encode(context=ctx, num_output_patches=K)
    got = enc_out[0].detach().cpu()
    del trunc
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None

    if got.shape != reference.shape:
        raise AssertionError(f"depth {depth}: truncated encoder gives {tuple(got.shape)}, "
                             f"hooked full model gives {tuple(reference.shape)}")
    diff = float((got - reference).abs().max())
    ok = diff <= atol + rtol * float(reference.abs().max())
    print(f"  [verify] depth {depth:>2}: max|truncated - RMSNorm(hooked h_l)| = {diff:.3e}  "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(
            f"depth {depth}: a model truncated after block {depth} does NOT reproduce the hooked "
            f"layer-{depth} states (max abs diff {diff:.3e}). Every layerwise result in this project "
            "assumes it does -- stop and diagnose before reporting any latency number.")
    return diff


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_call(fn, device, warmup, reps):
    """Median / p95 / IQR wall-clock in ms, with warm-up discarded and CUDA synchronised."""
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)),
            "min_ms": samples[0], "iqr_ms": float(np.subtract(*np.percentile(samples, [75, 25]))),
            "reps": reps, "warmup": warmup}


def measure(device, depth, batch, adapter_tag, warmup, reps, quantile_levels):
    pipeline = load_pipeline(device)
    adapter, adapter_src = load_adapter(adapter_tag, depth, device)
    truncate(pipeline, depth, adapter)

    torch.manual_seed(SEED + batch)
    host = [x for x in np.cumsum(np.random.default_rng(SEED).standard_normal((batch, C)),
                                 axis=-1).astype(np.float32)]
    dev_ctx = torch.as_tensor(np.stack(host), device=device)

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weights_mib = torch.cuda.memory_allocated() / 2 ** 20      # after load+truncate, before any forward

    predict = time_call(lambda: pipeline.predict_quantiles(list(host), prediction_length=H,
                                                           quantile_levels=quantile_levels,
                                                           batch_size=batch),
                        device, warmup, reps)
    encode = time_call(lambda: pipeline.model.encode(context=dev_ctx, num_output_patches=K),
                       device, warmup, reps)

    row = {"depth": depth, "depth_label": (["Emb"] + [f"L{i}" for i in range(1, 13)])[depth],
           "batch_size": batch,
           "active_params": model_size.active_params(depth),
           "active_fraction": model_size.active_fraction(depth),
           "block_flops_fraction": model_size.block_flops_fraction(depth),
           "end_to_end_flops_fraction": model_size.end_to_end_flops_fraction(depth, C, H),
           "predict_median_ms": predict["median_ms"], "predict_p95_ms": predict["p95_ms"],
           "predict_min_ms": predict["min_ms"], "predict_iqr_ms": predict["iqr_ms"],
           "encode_median_ms": encode["median_ms"], "encode_p95_ms": encode["p95_ms"],
           "encode_min_ms": encode["min_ms"], "encode_iqr_ms": encode["iqr_ms"],
           "throughput_series_per_s": batch / (predict["median_ms"] / 1e3),
           "predicted_weight_mib": model_size.weight_bytes(depth, 4) / 2 ** 20,
           "reps": reps, "warmup": warmup, "dtype": "float32",
           "transfer_included_in_predict": True, "transfer_included_in_encode": False,
           "adapter_source": adapter_src}
    if device.type == "cuda":
        row |= {"weights_mib": weights_mib,
                "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20}

    del pipeline, adapter, dev_ctx
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--adapter-dataset", default="monash_electricity_hourly",
                    help="which fitted adapter to load (shapes are identical, so this changes no timing)")
    ap.add_argument("--verify", action="store_true",
                    help="prove truncation == hooking before timing (loads the model twice per depth)")
    ap.add_argument("--quantile-levels", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--tag", default=None, help="suffix for the output files (default: the GPU name)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] no CUDA device — CPU timings are NOT the paper's deployment setting and the "
              "memory columns will be absent. Run this under salloc/sbatch with --gres=gpu:1.")
    env = environment(device)
    print(json.dumps(env, indent=1))

    if args.verify:
        print("\n[verify] a truncated model must reproduce the hooked full-model states")
        env["verification"] = {f"L{d}": verify_truncation(device, d)
                               for d in args.depths if d >= 1}

    rows = []
    print(f"\n[timing] depths={args.depths} batches={args.batch_sizes} "
          f"warmup={args.warmup} reps={args.reps} dtype=float32")
    print(f"  {'depth':>6}{'batch':>7}{'FLOPs':>8}{'predict ms':>12}{'p95':>8}"
          f"{'encode ms':>11}{'series/s':>11}{'peak MiB':>10}")
    for batch in args.batch_sizes:
        for depth in args.depths:
            r = measure(device, depth, batch, args.adapter_dataset, args.warmup, args.reps,
                        args.quantile_levels)
            rows.append(r)
            print(f"  {r['depth_label']:>6}{batch:>7}{r['block_flops_fraction']:>7.2f}x"
                  f"{r['predict_median_ms']:>12.2f}{r['predict_p95_ms']:>8.2f}"
                  f"{r['encode_median_ms']:>11.2f}{r['throughput_series_per_s']:>11.1f}"
                  f"{r.get('peak_mib', float('nan')):>10.1f}")

    # how well does the FLOP proxy actually predict measured time? (the point of the exercise)
    summary = {}
    for batch in args.batch_sizes:
        sub = [r for r in rows if r["batch_size"] == batch]
        ref = next((r for r in sub if r["depth"] == model_size.NUM_BLOCKS), None)
        if ref is None or len(sub) < 2:
            continue
        summary[f"batch_{batch}"] = [
            {"depth": r["depth"], "flops_ratio": r["block_flops_fraction"],
             "predict_time_ratio": r["predict_median_ms"] / ref["predict_median_ms"],
             "encode_time_ratio": r["encode_median_ms"] / ref["encode_median_ms"],
             "memory_ratio": (r["peak_mib"] / ref["peak_mib"]) if "peak_mib" in r else None}
            for r in sub]

    tag = args.tag or (env.get("gpu_name", "cpu").replace(" ", "_"))
    for d in ("tables",):
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "tables" / f"latency__{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    json.dump({"environment": env, "rows": rows, "ratio_vs_full_model": summary,
               "notes": {
                   "predict_ms": "pipeline.predict_quantiles on host arrays — preprocessing and "
                                 "host->device transfer INCLUDED (the real serving path)",
                   "encode_ms": "model.encode on a device-resident tensor — transfer EXCLUDED",
                   "peak_mib": "torch.cuda.max_memory_allocated() after reset_peak_memory_stats()",
                   "weights_mib": "allocated bytes after load+truncate, before any forward",
                   "context": "synthetic random-walk contexts; timing has no data-dependent control "
                              "flow, so this is equivalent to real windows and needs no dataset"}},
              open(OUT_ROOT / "tables" / f"latency__{tag}.json", "w"), indent=1)
    print(f"\n[write] {OUT_ROOT / 'tables' / f'latency__{tag}.csv'}")
    for batch, entries in summary.items():
        print(f"\n  {batch}: FLOP ratio vs MEASURED time ratio (1.0 = the full 12-block model)")
        for e in entries:
            print(f"    L{e['depth']:<3} flops {e['flops_ratio']:.2f}x   "
                  f"predict {e['predict_time_ratio']:.2f}x   encode {e['encode_time_ratio']:.2f}x"
                  + (f"   memory {e['memory_ratio']:.2f}x" if e["memory_ratio"] else ""))


if __name__ == "__main__":
    main()
