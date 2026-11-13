"""Phase 2e latency -- MEASURED cost of every matched-budget mask (Chronos-2), not assumed.

Two masks with the same number of retained blocks have the same nominal FLOPs, but nothing is
assumed: every mask x {hard, adapter} is timed with the committed Phase-2 protocol, which this
module IMPORTS rather than re-implements (``experiments.run_phase2_latency``: ``load_fresh``,
``time_calls``, ``summarize``, ``synthetic_batches``):

    one fresh subprocess per configuration; 20 warmup / 100 timed calls; every call bracketed by
    torch.cuda.synchronize(); B in {1, 256}; levels api_e2e (predict_quantiles, host -> host,
    the HEADLINE) and device_forward; fp32, TF32 off, no AMP, no compile; a FULL A100 (MIG
    refused); the unmodified model timed at the start, middle and end of every repeat as the
    drift control; randomized order per repeat (seeded); raw int64 ns vectors saved.

Only the CONFIGURATION differs from Phase 2: the encoder's block list is replaced by the retained
blocks in their original order (the skipped blocks are released, as H4's ``truncate`` releases
removed blocks) and, for ``adapter``, a 768 x 768 residual adapter (random weights: timing depends
on shape only) is spliced before the final norm -- exactly where the fitted adapters run.

REPEATS. ``--repeats 3`` in-job repeats on ONE node. A repeat is atomic: built in
``latency/rep<r>.building-<job>/`` and renamed to ``latency/rep<r>/`` only after its index.json is
written, so a killed job redoes at most one repeat and a repeat never mixes two nodes. Speedups are
formed WITHIN a repeat (vs that repeat's native median) and then summarized across repeats.

A configuration runs only if its mask passed the model-level verification written by
``run_phase2e_noncontig`` (``verification/masks.json``); natives always run.

    python -m experiments.run_phase2e_latency --repeats 3          # compute node, full A100
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase2_latency import (PROTOCOL_REPS, PROTOCOL_WARMUP,         # noqa: E402
                                            load_fresh, summarize, synthetic_batches,
                                            time_calls)
from probing import phase2e_noncontig as P                                        # noqa: E402
from probing.phase2 import SEED                                                    # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2e_noncontig"
KINDS = ("hard", "adapter")
LEVELS = ("api_e2e", "device_forward")
D_MODEL = 768


def configure_mask(handle, cfg: dict) -> dict:
    """Apply one configuration to a FRESH pipeline, permanently (the process is discarded)."""
    import torch
    from probing import phase2_truncate as T
    from probing.phase2_align import ResidualAdapter
    core = T.core_of(handle, P.MODEL)
    kind = cfg["kind"]
    rec = {"kind": kind, "mask": cfg.get("mask"), "label": cfg.get("label")}
    if kind == "native":
        return rec
    blocks = list(T.blocks_of(core, P.MODEL))
    mask = P.validate_mask(cfg["mask"], len(blocks))
    T.set_blocks(core, P.MODEL, [blocks[b - 1] for b in mask])
    del blocks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rec.update({"blocks_kept": len(mask), "skipped": list(P.skipped_blocks_of(mask, P.N_BLOCKS)),
                "mask_string": P.mask_string(mask)})
    if kind == "adapter":
        dev = next(core.parameters()).device
        d = int(core.config.d_model)
        g = torch.Generator().manual_seed(SEED)
        ad = ResidualAdapter(d)
        with torch.no_grad():
            ad.delta.copy_(torch.randn(d, d, generator=g) * 1e-3)
        T.splice_adapter(core, P.MODEL, ad.to(dev).eval())
        rec["adapter_kind"] = "affine"
    return rec


def run_child(cfg: dict) -> dict:
    import torch
    from probing import phase2_truncate as T
    from probing.phase2_env import environment_record, set_precision_flags
    numerics = set_precision_flags(deterministic=False)           # timing: no forced determinism
    device = cfg["device"]
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
        if "MIG" in name and not cfg.get("allow_mig"):
            return {"status": "refused", "reason": f"MIG instance ({name}); request a full A100"}
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    t0 = time.time()
    handle = load_fresh(P.MODEL, {"kind": "native", "tiny": False}, device)
    conf = configure_mask(handle, cfg)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    out = {"config": cfg, "configured": conf, "params": T.param_breakdown(handle, P.MODEL),
           "weights_bytes": torch.cuda.memory_allocated() if cuda else None,
           "load_s": round(time.time() - t0, 2), "numerics": numerics, "levels": {},
           "environment": environment_record(), "status": "ok"}
    raw = {}
    for B in cfg["batch_sizes"]:
        batches, spec = synthetic_batches(B)
        out.setdefault("inputs", {})[f"B{B}"] = spec
        dev_batches = [torch.as_tensor(x, device=device) for x in batches]
        for level in cfg["levels"]:
            it = {"i": 0}

            def call():
                k = it["i"] % len(batches)
                it["i"] += 1
                if level == "api_e2e":
                    T.public_forecast(handle, P.MODEL, batches[k], batch_size=B)
                else:
                    T.device_forward(handle, P.MODEL, dev_batches[k])
            key = f"{level}__B{B}"
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                ts = time_calls(call, warmup=cfg["warmup"], reps=cfg["reps"], sync=sync)
                rec = summarize(ts, B)
                if cuda:
                    rec["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                rec["status"] = "ok"
                raw[key] = ts
            except torch.cuda.OutOfMemoryError as e:            # recorded, never substituted
                rec = {"status": "oom", "error": str(e)[:300]}
                if cuda:
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
            out["levels"][key] = rec
    out["raw_timings_ns"] = raw
    return out


def config_list(repeat: int, masks: list[dict], kinds, seed: int) -> list[dict]:
    """Natives at start / middle / end; the mask configurations in a seeded random order that
    differs between repeats."""
    rng = random.Random(seed * 1000 + int(repeat))
    body = [{"kind": k, "mask": list(e["mask"]), "label": e["label"], "strategy": e["strategy"],
             "budget": e["budget"]} for e in masks for k in kinds]
    rng.shuffle(body)
    nat = {"kind": "native", "mask": list(P.FULL_MASK), "label": "native", "strategy": "native",
           "budget": P.N_BLOCKS}
    mid = len(body) // 2
    return ([dict(nat, drift="start")] + body[:mid] + [dict(nat, drift="middle")] + body[mid:]
            + [dict(nat, drift="end")])


def eligibility(args) -> list[str]:
    why = []
    if (args.warmup, args.reps) != (PROTOCOL_WARMUP, PROTOCOL_REPS):
        why.append(f"warmup/reps {args.warmup}/{args.reps} != {PROTOCOL_WARMUP}/{PROTOCOL_REPS}")
    if not str(args.device).startswith("cuda"):
        why.append(f"device {args.device}")
    if args.allow_unverified:
        why.append("--allow-unverified")
    if args.allow_mig:
        why.append("--allow-mig")
    return why


def run_repeat(r: int, args, masks, passed: set, job: str) -> Path:
    base = Path(args.output_root) / "latency"
    final = base / f"rep{r}"
    for stale in base.glob(f"rep{r}.building-*"):
        shutil.rmtree(stale, ignore_errors=True)
        print(f"[latency] removed stale partial repeat {stale.name}")
    stage = base / f"rep{r}.building-{job}"
    stage.mkdir(parents=True)
    cfgs = config_list(r, masks, args.kinds, args.seed)
    why = eligibility(args)
    print(f"[latency] repeat {r}: {len(cfgs)} configurations -> {final}  (headline-eligible: "
          f"{'YES' if not why else 'NO -- ' + '; '.join(why)})")
    index, t_rep = [], time.time()
    for i, cfg in enumerate(cfgs):
        cid = f"{i:03d}_{cfg['kind']}_{cfg['label']}{'_' + cfg['drift'] if cfg.get('drift') else ''}"
        ms = P.mask_string(cfg["mask"])
        if cfg["kind"] != "native" and ms not in passed and not args.allow_unverified:
            print(f"  [refused] {cid}: mask {ms} has no PASSED model-level verification")
            index.append({"id": cid, **cfg, "status": "refused_unverified"})
            continue
        dest = stage / cid
        dest.mkdir(parents=True)
        full = {**cfg, "batch_sizes": list(args.batch_sizes), "levels": list(args.levels),
                "warmup": args.warmup, "reps": args.reps, "device": args.device,
                "allow_mig": args.allow_mig, "dest": str(dest), "repeat": r}
        (dest / "config.json").write_text(json.dumps(full, indent=1))
        t0 = time.time()
        res = subprocess.run([sys.executable, "-m", "experiments.run_phase2e_latency", "--child",
                              str(dest / "config.json")], cwd=REPO_ROOT,
                             timeout=args.config_timeout)
        ok = res.returncode == 0 and (dest / "summary.json").exists()
        s = json.loads((dest / "summary.json").read_text()) if ok else {}
        status = s.get("status", "failed") if ok else "failed"
        bad = sorted(k for k, v in s.get("levels", {}).items() if v.get("status") != "ok")
        if status == "ok" and bad:
            status = "partial"
        med = {k: v.get("median_ms") for k, v in s.get("levels", {}).items()}
        print(f"  [{status}] {cid} ({time.time() - t0:.0f}s) " +
              " ".join(f"{k}={v:.2f}ms" for k, v in med.items() if v is not None))
        index.append({"id": cid, **cfg, "status": status, "median_ms": med,
                      "params": s.get("params")})
        done = i + 1
        rate = (time.time() - t_rep) / done
        print(f"  [latency progress] repeat {r}: {done}/{len(cfgs)}  ETA this repeat "
              f"~{rate * (len(cfgs) - done) / 60:.0f} min")
    (stage / "index.json").write_text(json.dumps({
        "repeat": r, "job": job, "host": socket.gethostname(), "headline_eligible": not why,
        "ineligible_reasons": why, "finished_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "protocol": {"warmup": args.warmup, "reps": args.reps,
                     "batch_sizes": list(args.batch_sizes), "levels": list(args.levels),
                     "device": args.device, "seed": args.seed, "kinds": list(args.kinds),
                     "imported_from": "experiments.run_phase2_latency"},
        "configs": index}, indent=1, default=str))
    if final.exists():
        shutil.rmtree(final)
    os.replace(stage, final)
    return final


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.child:
        cfg = json.loads(Path(args.child).read_text())
        res = run_child(cfg)
        raw = res.pop("raw_timings_ns", {})
        dest = Path(cfg["dest"])
        np.savez(dest / "timings_raw.npz", **raw)
        (dest / "summary.json").write_text(json.dumps(res, indent=1, default=str))
        return 0
    from experiments.run_phase2e_noncontig import assert_output_root
    assert_output_root(args.output_root)
    masks = P.plan_masks()
    vpath = Path(args.output_root) / "verification" / "masks.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        passed = set(v.get("passed_masks", []))
    else:
        passed = set()
        print(f"[latency] WARNING: {vpath} missing -- only the native configuration will be timed "
              "(run experiments.run_phase2e_noncontig first; it writes the mask verification)")
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    for r in range(1, int(args.repeats) + 1):
        final = Path(args.output_root) / "latency" / f"rep{r}"
        if (final / "index.json").exists() and not args.force:
            print(f"[latency] repeat {r} COMPLETE ({final}); skipped")
            continue
        run_repeat(r, args, masks, passed, job)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 256])
    p.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    p.add_argument("--warmup", type=int, default=PROTOCOL_WARMUP)
    p.add_argument("--reps", type=int, default=PROTOCOL_REPS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--config-timeout", type=int, default=1800)
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--allow-mig", action="store_true")
    p.add_argument("--force", action="store_true", help="re-time repeats that are COMPLETE")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
