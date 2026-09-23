"""H4 latency / throughput / memory harness -- MEASURED, one fresh process per configuration.

    # smoke (all 3 models, 2 depths, 3 batch sizes, 5 reps; its own output root, and never
    # headline-eligible anyway) -- on a GPU compute node:
    python -m experiments.run_phase2_latency --depths 3 12 --reps 5 --warmup 2 --job-tag smoke \\
        --allow-unverified --output-root results/three_model_phase2_smoke

    # one full benchmark job (submit THREE of these: --job-tag job1 / job2 / job3):
    python -m experiments.run_phase2_latency --job-tag job1

WHAT IS TIMED (per model, per configuration, per batch size B in {1, 32, 256}):
    api_e2e         the package's PUBLIC forecasting API, host numpy -> host output (HEADLINE)
                    Chronos-2 predict_quantiles | TimesFM-3 predict_batch (per_core_batch_size=B)
                    | TiRex forecast (default two-pass)
    device_forward  device-resident raw context -> the model's own forward (in-model scaling
                    included) -> device quantiles

CONFIGURATIONS: native (unmodified; timed at the start, middle and end of the job as a drift
control), and for every depth l: ``hard`` (blocks[:l], no adapter), ``adapter`` (blocks[:l] + a
d x d residual adapter -- weights do not change the timing, only the shape does) and
``probe_head`` (blocks[:l] + a folded Linear head). Plus ``chronos2_small`` in the Chronos-2 panel.

PROTOCOL: a fresh subprocess per configuration (removed parameters, the allocator, cuBLAS
workspaces and the CUDA context are genuinely fresh); randomized order (seeded); warmup discarded;
every timed call bracketed by torch.cuda.synchronize(); the RAW per-call int64 ns vector is saved;
peak memory is reset before each level's timed loop; an OOM is RECORDED as status "oom" and the
batch size is NEVER changed silently; MIG instances are refused; fp32, TF32 off, no AMP, no
torch.compile. 100 in-run repetitions are not independent runs: submit 3 jobs.

Timing depends on (model, configuration, batch, level) -- not on the dataset (no inference path
has data-dependent control flow for finite full-length contexts; verified in the package source),
so each configuration is benchmarked once per job and every dataset's operating point READS its
entry. Speedups are formed WITHIN a job (same node), then aggregated across jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1                                                        # noqa: E402
from probing.phase2 import DEFAULT_OUT_ROOT, SEED                                  # noqa: E402

C, H = phase1.PHASE1_C, phase1.PHASE1_H
LEVELS = ("api_e2e", "device_forward")
CONFIG_KINDS = ("native", "hard", "adapter", "probe_head", "chronos2_small")
D_MODEL = {"chronos2": 768, "timesfm3": 1280, "tirex": 512}
HEAD_OUT = {"chronos2": 21 * 16, "timesfm3": 64 * 9, "tirex": 9 * 32}

# The HEADLINE timing protocol (HOW a configuration is timed). A job that deviates from it -- a
# smoke, tiny models, unverified configurations, CPU, a MIG slice -- still runs and is saved, but
# its index.json says headline_eligible = false and make_phase2_tables never folds it into the
# combined latency lookup. WHAT is timed (models, depths, kinds, batch sizes, levels) may vary.
PROTOCOL_WARMUP = 20
PROTOCOL_REPS = 100


def headline_eligibility(args) -> list[str]:
    """Why this job may NOT enter the combined latency tables (an empty list = eligible)."""
    why = []
    if args.tiny:
        why.append("tiny test models")
    if args.allow_unverified:
        why.append("--allow-unverified")
    if args.allow_mig:
        why.append("--allow-mig")
    if (args.warmup, args.reps) != (PROTOCOL_WARMUP, PROTOCOL_REPS):
        why.append(f"warmup/reps {args.warmup}/{args.reps} != {PROTOCOL_WARMUP}/{PROTOCOL_REPS}")
    if not str(args.device).startswith("cuda"):
        why.append(f"device {args.device}")
    return why


# --------------------------------------------------------------------------- #
# synthetic inputs -- valid for all three (no data-dependent control flow)
# --------------------------------------------------------------------------- #
def synthetic_batches(B: int, n_batches: int = 4, seed: int = SEED):
    """x_t = mu + sigma * cumsum(eps), mu ~ U(-50, 50), sigma ~ LogUniform(0.1, 10) per series.
    Four pre-generated batches are cycled so no two consecutive calls see identical input."""
    out, digests = [], []
    for k in range(n_batches):
        rng = np.random.default_rng([seed, B, k])
        mu = rng.uniform(-50, 50, size=(B, 1))
        sig = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=(B, 1)))
        x = (mu + sig * np.cumsum(rng.normal(size=(B, C)), axis=1)).astype(np.float32)
        out.append(x)
        digests.append("sha256:" + hashlib.sha256(x.tobytes()).hexdigest())
    spec = {"generator": "mu + sigma * cumsum(N(0,1)), mu~U(-50,50), sigma~LogUniform(0.1,10)",
            "rng": "numpy default_rng([SEED, B, k])", "seed": seed, "n_batches": n_batches,
            "C": C, "dtype": "float32", "sha256": digests}
    return out, spec


# --------------------------------------------------------------------------- #
# model loading (fresh, never a singleton) + configuration
# --------------------------------------------------------------------------- #
def load_fresh(model: str, cfg: dict, device: str):
    import torch
    if cfg.get("tiny"):
        from tests.test_phase2_h4 import BUILDERS
        return BUILDERS[model]()
    if model == "chronos2" or cfg["kind"] == "chronos2_small":
        from chronos import Chronos2Pipeline
        name = "autogluon/chronos-2-small" if cfg["kind"] == "chronos2_small" else "amazon/chronos-2"
        pipe = Chronos2Pipeline.from_pretrained(name, torch_dtype=torch.float32)
        pipe.model.to(device).eval()
        for p in pipe.model.parameters():
            p.requires_grad_(False)
        return pipe
    if model == "timesfm3":
        from timesfm3.torch import TimesFM3Forecaster
        fc = TimesFM3Forecaster.from_pretrained(cfg.get("checkpoint") or
                                                "google/timesfm-3.0-pytorch", device=device)
        fc.model.eval()
        for p in fc.model.parameters():
            p.requires_grad_(False)
        return fc
    from probing.tirex_model import get_model
    return get_model(cfg.get("checkpoint") or "NX-AI/TiRex", device, backend="torch")


def configure(handle, model: str, cfg: dict) -> dict:
    """Apply one configuration. Adapter / probe weights are random: timing depends on SHAPE only."""
    import torch
    from probing import phase2_truncate as T
    from probing.phase2_align import ResidualAdapter
    kind, depth = cfg["kind"], cfg.get("depth")
    core = T.core_of(handle, model)
    dev = next(core.parameters()).device
    d = 32 if cfg.get("tiny") else D_MODEL[model]      # tiny = the contract-test models
    rec = {"kind": kind, "depth": depth}
    if kind in ("native", "chronos2_small"):
        return rec
    rec.update({k: v for k, v in T.truncate(handle, model, depth).items()
                if k != "removed_weakrefs"})
    g = torch.Generator().manual_seed(SEED)
    if kind == "adapter":
        ad = ResidualAdapter(d)
        with torch.no_grad():
            ad.delta.copy_(torch.randn(d, d, generator=g) * 1e-3)
        T.splice_adapter(core, model, ad.to(dev).eval())
    elif kind == "probe_head":
        out = HEAD_OUT[model] if model != "chronos2" else 9 * 16
        rng = np.random.default_rng(SEED)
        probe = {"weight": rng.normal(size=(out, d)) * 1e-2, "bias": np.zeros(out),
                 "scaler_mean": np.zeros(d), "scaler_scale": np.ones(d)}
        nq = getattr(handle, "quantiles", None)
        rec["probe_head"] = T.install_probe_head(handle, model, probe, native_quantiles=nq)
    return rec


# --------------------------------------------------------------------------- #
# the child: one configuration, every batch size and level
# --------------------------------------------------------------------------- #
def time_calls(fn, *, warmup: int, reps: int, sync) -> np.ndarray:
    for _ in range(warmup):
        fn()
    sync()
    ts = np.empty(reps, np.int64)
    for i in range(reps):
        sync()
        t0 = time.perf_counter_ns()
        fn()
        sync()
        ts[i] = time.perf_counter_ns() - t0
    return ts


def summarize(ts: np.ndarray, B: int) -> dict:
    ms = ts.astype(np.float64) / 1e6
    q25, q75 = np.percentile(ms, [25, 75])
    med = float(np.median(ms))
    return {"median_ms": med, "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean()),
            "std_ms": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0, "min_ms": float(ms.min()),
            "iqr_ms": float(q75 - q25), "reps": int(len(ms)),
            "throughput_series_per_s": float(B / (med / 1e3)) if med > 0 else None}


def run_child(cfg: dict) -> dict:
    import torch
    from probing import phase2_truncate as T
    from probing.phase2_env import environment_record, set_precision_flags
    numerics = set_precision_flags(deterministic=False)       # timing: no forced determinism
    device = cfg["device"]
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
        if "MIG" in name and not cfg.get("allow_mig"):
            return {"status": "refused", "reason": f"MIG instance ({name}) -- timings on a GPU "
                    "slice are not comparable; request a full GPU"}
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    model = cfg["model"]
    t0 = time.time()
    handle = load_fresh(model, cfg, device)
    conf = configure(handle, model, cfg)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    weights_bytes = torch.cuda.memory_allocated() if cuda else None
    params = T.param_breakdown(handle, "chronos2" if cfg["kind"] == "chronos2_small" else model)
    out = {"config": cfg, "configured": conf, "params": params, "weights_bytes": weights_bytes,
           "load_s": round(time.time() - t0, 2), "numerics": numerics, "levels": {},
           "environment": environment_record(), "status": "ok"}
    raw = {}
    mdl = "chronos2" if cfg["kind"] == "chronos2_small" else model
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
                    T.public_forecast(handle, mdl, batches[k], batch_size=B)
                else:
                    T.device_forward(handle, mdl, dev_batches[k])
            key = f"{level}__B{B}"
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                ts = time_calls(call, warmup=cfg["warmup"], reps=cfg["reps"], sync=sync)
                rec = summarize(ts, B)
                if cuda:
                    rec["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                    rec["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                rec["status"] = "ok"
                raw[key] = ts
            except torch.cuda.OutOfMemoryError as e:           # recorded, never substituted
                rec = {"status": "oom", "error": str(e)[:300]}
                if cuda:
                    torch.cuda.empty_cache()
            except Exception as e:                  # one broken level must not erase the others
                import traceback
                traceback.print_exc()
                rec = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
            out["levels"][key] = rec
    out["raw_timings_ns"] = raw
    return out


# --------------------------------------------------------------------------- #
# the parent: config list, fresh subprocesses, collection
# --------------------------------------------------------------------------- #
def config_list(args) -> list[dict]:
    rng = random.Random(args.seed)
    cfgs = []
    for model in args.models:
        L = phase1.model_spec(model).reference_index
        depths = list(range(L + 1)) if args.depths is None else [int(d) for d in args.depths]
        body = []
        for d in depths:                                   # adapter at d == L = splice overhead
            for kind in args.kinds:
                if kind not in ("native", "chronos2_small") and 0 <= d <= L:
                    body.append({"model": model, "kind": kind, "depth": d})
        if model == "chronos2" and "chronos2_small" in args.kinds:
            body.append({"model": model, "kind": "chronos2_small", "depth": None})
        rng.shuffle(body)
        nat = {"model": model, "kind": "native", "depth": L}
        mid = len(body) // 2
        seq = [dict(nat, drift="start")] + body[:mid] + [dict(nat, drift="middle")] + \
            body[mid:] + [dict(nat, drift="end")]
        cfgs += seq
    for c in cfgs:
        c.update({"batch_sizes": list(args.batch_sizes), "levels": list(args.levels),
                  "warmup": args.warmup, "reps": args.reps, "device": args.device,
                  "tiny": bool(args.tiny), "checkpoint": None, "allow_mig": args.allow_mig})
    return cfgs


def verified(cfg: dict, args) -> tuple[bool, str]:
    if args.allow_unverified or cfg["kind"] in ("native", "chronos2_small"):
        return True, "not required" if cfg["kind"] in ("native", "chronos2_small") else "allowed"
    p = Path(args.verification_root) / f"{cfg['model']}.json"
    if not p.exists():
        return False, f"no verification record {p}"
    rec = json.loads(p.read_text())
    ok = rec.get("passed_depths", {}).get(cfg["kind"], [])
    return (cfg["depth"] in ok), f"{cfg['kind']} depth {cfg['depth']} in verified {ok}"


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.child:
        cfg = json.loads(Path(args.child).read_text())
        res = run_child(cfg)
        raw = res.pop("raw_timings_ns", {})
        dest = Path(cfg["dest"])
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / "timings_raw.npz", **raw)
        (dest / "summary.json").write_text(json.dumps(res, indent=1, default=str))
        return 0
    out = Path(args.output_root) / "latency" / args.job_tag
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"[latency] {out} already holds a job: every job tag is ONE independent "
                         "measurement. Use a new --job-tag (or delete that directory if the job "
                         "was killed before it wrote index.json).")
    out.mkdir(parents=True, exist_ok=True)
    cfgs = config_list(args)
    reasons = headline_eligibility(args)
    print(f"[latency] {len(cfgs)} configurations -> {out}")
    print("[latency] headline-eligible (enters the combined tables): " +
          ("YES" if not reasons else f"NO -- {'; '.join(reasons)}"))
    index = []
    for i, cfg in enumerate(cfgs):
        ok, why = verified(cfg, args)
        cid = (f"{i:03d}_{cfg['model']}_{cfg['kind']}_"
               f"{'na' if cfg['depth'] is None else 'L%02d' % cfg['depth']}"
               f"{'_' + cfg['drift'] if cfg.get('drift') else ''}")
        dest = out / cid
        if not ok:
            print(f"  [refused] {cid}: {why}")
            index.append({"id": cid, **cfg, "status": "refused_unverified", "reason": why})
            continue
        cfg = {**cfg, "dest": str(dest)}
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(json.dumps(cfg, indent=1))
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "experiments.run_phase2_latency",
                            "--child", str(dest / "config.json")], cwd=REPO_ROOT,
                           timeout=args.config_timeout)
        status = "ok" if r.returncode == 0 and (dest / "summary.json").exists() else "failed"
        s = json.loads((dest / "summary.json").read_text()) if status == "ok" else {}
        bad = sorted(k for k, v in s.get("levels", {}).items() if v.get("status") != "ok")
        if status == "ok" and bad:
            status = "partial"                      # visible in the log and in index.json
            print(f"  [partial] {cid}: levels not measured: {bad} (see the traceback above)")
        med = {k: v.get("median_ms") for k, v in s.get("levels", {}).items()}
        print(f"  [{status}] {cid} ({time.time() - t0:.0f}s) " +
              " ".join(f"{k}={v:.2f}ms" for k, v in med.items() if v is not None))
        index.append({"id": cid, **{k: cfg[k] for k in ("model", "kind", "depth")},
                      "drift": cfg.get("drift"), "status": status, "median_ms": med,
                      "params": s.get("params"), "weights_bytes": s.get("weights_bytes")})
    (out / "index.json").write_text(json.dumps({
        "job_tag": args.job_tag, "headline_eligible": not reasons, "ineligible_reasons": reasons,
        "protocol": {"warmup": args.warmup, "reps": args.reps,
                     "batch_sizes": list(args.batch_sizes), "levels": list(args.levels),
                     "device": args.device, "tiny": bool(args.tiny),
                     "allow_unverified": bool(args.allow_unverified),
                     "allow_mig": bool(args.allow_mig), "seed": args.seed},
        "configs": index}, indent=1, default=str))
    print(f"[latency] wrote {out / 'index.json'}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    p.add_argument("--models", nargs="+", default=list(phase1.MODELS),
                   choices=list(phase1.MODELS))
    p.add_argument("--depths", nargs="+", type=int, default=None,
                   help="block depths (default: every depth 0..L)")
    p.add_argument("--kinds", nargs="+", default=list(CONFIG_KINDS), choices=list(CONFIG_KINDS))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 256])
    p.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    p.add_argument("--warmup", type=int, default=PROTOCOL_WARMUP)
    p.add_argument("--reps", type=int, default=PROTOCOL_REPS)
    p.add_argument("--job-tag", required=False, default="job1")
    p.add_argument("--output-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--verification-root", default=None,
                   help="default: <output-root>/h4/verification (written by run_phase2_truncation "
                        "verify)")
    p.add_argument("--allow-unverified", action="store_true",
                   help="SMOKE ONLY: time configurations without a passed verification record")
    p.add_argument("--allow-mig", action="store_true")
    p.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--config-timeout", type=int, default=3600)
    a = p.parse_args(argv)
    if a.verification_root is None:
        a.verification_root = str(Path(a.output_root) / "h4" / "verification")
    return a


if __name__ == "__main__":
    raise SystemExit(main())
