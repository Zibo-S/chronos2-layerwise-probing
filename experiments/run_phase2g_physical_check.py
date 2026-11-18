"""Phase 2g physical check: zero-anchored TimesFM-3 adapters in physically truncated models.

    python -m experiments.run_phase2g_physical_check --device cpu [--datasets ...] [--depths ...]

The Phase-2g numbers (Appendix B.3.1) are OFFLINE: cached token-15 states -> adapter -> frozen
head -> inverse. This driver checks that a physically truncated TimesFM-3 (blocks l+1..20 removed,
the adapter spliced before the head, run through the public ``predict_batch`` API) reproduces
them. Per dataset and depth:

  1. refit the zero-anchored closed form exactly as the 2g driver does
     (``ReducedRidge.path("zero", ...)`` on the cached Phase-1 states) and require the SAME
     selected kappa as the committed 2g fit;
  2. offline (cached states, float64 adapter, as 2g): its per-window test MASE must equal the
     committed 2g ``zero__test_mase`` (gate 1e-6 relative on the mean);
  3. physical: ``probing.phase2_truncate.truncate`` + ``affine_to_adapter`` on a fresh copy of the
     public model; forecasts on the raw test contexts;
  4. compare physical with offline: relative difference of mean test MASE (the paper's V3 rule,
     1e-4) and per-value differences (reported, not gated);
  5. once per dataset: the unmodified model through the same API against the committed native
     test MASE (device / numerics check).

Read-only over every committed tree. Writes ONLY
``<output-root>/physical/<dataset>.json`` and ``<output-root>/physical/summary.json``.
Resumable: a dataset whose JSON exists is skipped. Adapters are rebuilt, not saved.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase2, registry                                               # noqa: E402
from probing.phase2 import (DEFAULT_PHASE1_ROOT, MEDIAN_INDEX, QUANTILES, Phase1Cell,  # noqa: E402
                            resolve_phase1_root)

MODEL = "timesfm3"
DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2g_timesfm3_zero_ridge"
DEFAULT_DEPTHS = (0, 1, 5, 10, 15, 18, 19)
V3_MEAN_RTOL = 1e-4          # the paper's physical-versus-offline rule (Appendix A.7)
OFFLINE_RTOL = 1e-6          # the refit must reproduce the committed 2g offline arrays


def _rel(a, b) -> float:
    a, b = float(np.mean(a)), float(np.mean(b))
    return abs(a - b) / max(abs(b), 1e-300)


def run_dataset(tag, args, pathway, base, mr, Qr, p, log=print) -> dict:
    import torch
    from experiments.run_phase2_h3 import windows_readonly
    from experiments.run_phase2_truncation import _forecast
    from experiments.run_phase2g_timesfm3_zero_ridge import RIDGE_KAPPAS, ReducedRidge, _flat
    from probing import phase2_truncate as T
    from probing.phase1_metrics import raw_window_metrics
    from probing.phase2_align import affine_to_adapter
    from probing.phase2_pathways import load_timesfm3

    g2 = Path(args.g2_root) / "zero_ridge" / MODEL / tag
    with np.load(g2 / "bootstrap_inputs.npz", allow_pickle=True) as z:
        com = {k: z[k] for k in z.files}
    fits = json.loads((g2 / "fits.json").read_text())
    root = resolve_phase1_root(tag, args.phase1_root, {})
    p1 = Phase1Cell(root, MODEL, tag)
    w = windows_readonly(tag, args)
    t0 = time.time()
    data = load_timesfm3(tag, w, p1.arrays(), p1.config(), cache_dir=args.cache_root)
    tr, va, te = (data.splits[s] for s in ("train", "val", "test"))
    if not np.array_equal(np.asarray(te.rows), np.asarray(com["rows_test"])):
        raise RuntimeError(f"{tag}: test rows differ from the committed 2g cell")
    L, labels = data.reference_index, list(data.depth_labels)
    log(f"  [{tag}] data loaded in {time.time() - t0:.0f}s; n_test={len(te.target)}")
    B = int(args.batch)

    def mase(raw):
        return np.asarray(raw_window_metrics(tag, te.X, te.y_raw, raw, QUANTILES, MEDIAN_INDEX)
                          ["mase_pw"], np.float64)

    # ---- the unmodified model through the public API (device / numerics check) ------------------
    t0 = time.time()
    raw_nat = _forecast(copy.deepcopy(base), MODEL, np.asarray(te.X, np.float32), B)
    m_nat = mase(raw_nat)
    out = {"dataset": tag, "n_test": int(len(te.target)), "device": args.device,
           "native": {"physical_mean_mase": float(m_nat.mean()),
                      "committed_mean_mase": float(com["native_recomputed__test_mase"].mean()),
                      "mean_rel_diff": _rel(m_nat, com["native_recomputed__test_mase"]),
                      "seconds": round(time.time() - t0, 1)},
           "depths": {}}
    log(f"  [{tag}] native: physical {m_nat.mean():.6f} vs committed "
        f"{com['native_recomputed__test_mase'].mean():.6f} "
        f"(rel {out['native']['mean_rel_diff']:.1e}, {out['native']['seconds']}s)")

    for l in args.depths:
        t0 = time.time()
        lab = labels[l]
        Xtr, Ytr = _flat(tr.feats[l]), _flat(tr.feats[L])
        Xva, Yva = _flat(va.feats[l]), _flat(va.feats[L])
        Xte = _flat(te.feats[l])
        red = ReducedRidge(Xtr, Ytr, mr, Qr, p)
        sel = red.path("zero", Xva, Yva, RIDGE_KAPPAS)
        k_com = fits[lab]["zero"]["selected_kappa"]
        rec = {"label": lab, "kappa": sel["selected_kappa"], "kappa_committed": k_com,
               "kappa_equal": sel["selected_kappa"] == k_com,
               "selected_endpoint": bool(sel["selected_endpoint"])}
        # offline, exactly as 2g scored it (float64 adapted states -> float32 -> head -> inverse)
        st = red.adapt(Xte, sel["Bt"]).astype(np.float32)[:, None, :]
        _, zt = pathway.run(st, None)
        raw_off = pathway.to_raw(zt, te)
        m_off = mase(raw_off)
        rec["offline_mean_mase"] = float(m_off.mean())
        rec["offline_vs_committed_rel"] = _rel(m_off, com["zero__test_mase"][l])
        # the same map as a module: a(x) = x M + c (rows) = A x + b with A = M^T, b = c
        Mrow = (np.zeros((red.d, red.d)) if sel["Bt"] is None
                else red.V @ sel["Bt"] @ red.Qr.T)
        c = red.muY - red.muX @ Mrow
        dev = next(T.core_of(base, MODEL).parameters()).device
        ad = affine_to_adapter(Mrow.T, c).to(dev)
        with torch.no_grad():
            a32 = ad(torch.as_tensor(Xte, dtype=torch.float32, device=dev)).double().cpu().numpy()
        a64 = red.adapt(Xte, sel["Bt"])
        rec["module_vs_closed_form_max_rel"] = float(np.abs(a32 - a64).max()
                                                     / max(np.abs(a64).mean(), 1e-300))
        # physical: blocks l+1..L removed, adapter spliced before the head, public API
        h = copy.deepcopy(base)
        trec = T.truncate(h, MODEL, l, adapter=copy.deepcopy(ad))
        rec["blocks_kept"] = trec["blocks_kept"]
        raw_phys = _forecast(h, MODEL, np.asarray(te.X, np.float32), B)
        del h
        m_phys = mase(raw_phys)
        rec["physical_mean_mase"] = float(m_phys.mean())
        rec["physical_vs_offline_rel"] = _rel(m_phys, m_off)
        rec["physical_vs_committed_rel"] = _rel(m_phys, com["zero__test_mase"][l])
        rec["per_window_mase_max_abs_diff"] = float(np.abs(m_phys - m_off).max())
        scale = float(np.abs(raw_off).mean())
        rec["value_max_abs_diff_over_mean_abs"] = float(np.abs(raw_phys - raw_off).max() / scale)
        rec["value_median_abs_diff_over_mean_abs"] = float(np.median(np.abs(raw_phys - raw_off))
                                                           / scale)
        rec["V3_mean_rule_passed"] = bool(rec["physical_vs_offline_rel"] <= V3_MEAN_RTOL)
        rec["offline_reproduces_committed"] = bool(rec["offline_vs_committed_rel"] <= OFFLINE_RTOL)
        rec["seconds"] = round(time.time() - t0, 1)
        out["depths"][lab] = rec
        log(f"  [{tag}] {lab:>4} kappa {rec['kappa']} (committed {k_com}) | offline {m_off.mean():.6f}"
            f" (vs 2g {rec['offline_vs_committed_rel']:.1e}) | physical {m_phys.mean():.6f} "
            f"(vs offline {rec['physical_vs_offline_rel']:.1e}; max value diff "
            f"{rec['value_max_abs_diff_over_mean_abs']:.1e} of mean|f|) | {rec['seconds']}s")
    d = list(out["depths"].values())
    out["passed"] = bool(all(r["kappa_equal"] and r["offline_reproduces_committed"]
                             and r["V3_mean_rule_passed"] for r in d))
    return out


def summarize(dest: Path) -> dict:
    recs = [json.loads(p.read_text()) for p in sorted(dest.glob("*.json")) if p.name != "summary.json"]
    rows = [r for x in recs for r in x["depths"].values()]
    s = {"schema": "phase2g_physical_check_summary/v1",
         "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
         "datasets": [x["dataset"] for x in recs], "n_configurations": len(rows),
         "all_passed": bool(recs) and all(x["passed"] for x in recs),
         "n_V3_mean_rule_passed": sum(r["V3_mean_rule_passed"] for r in rows),
         "max_physical_vs_offline_rel": max((r["physical_vs_offline_rel"] for r in rows), default=None),
         "max_physical_vs_committed_rel": max((r["physical_vs_committed_rel"] for r in rows),
                                              default=None),
         "max_offline_vs_committed_rel": max((r["offline_vs_committed_rel"] for r in rows),
                                             default=None),
         "n_kappa_equal": sum(r["kappa_equal"] for r in rows),
         "max_value_diff_over_mean_abs": max((r["value_max_abs_diff_over_mean_abs"] for r in rows),
                                             default=None),
         "native_max_rel": max((x["native"]["mean_rel_diff"] for x in recs), default=None),
         "rule": f"physical mean test MASE within {V3_MEAN_RTOL:g} relative of the offline pathway"}
    phase2.atomic_write_json(dest / "summary.json", s)
    return s


def main(argv=None) -> int:
    a = parse_args(argv)
    dest = Path(a.output_root) / "physical"
    dest.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        print(json.dumps(summarize(dest), indent=1))
        return 0
    import torch
    torch.set_num_threads(int(a.threads))
    from experiments.run_phase2_latency import load_fresh
    from experiments.run_phase2g_timesfm3_zero_ridge import metric_range
    from probing.phase2_env import set_precision_flags
    from probing.phase2_pathways import TimesFM3Pathway
    from probing.timesfm3_last_token import get_model
    set_precision_flags(deterministic=True)
    tags = [t for t in registry.roster("paper14") if not a.datasets or t in a.datasets]
    p1 = Phase1Cell(resolve_phase1_root(tags[0], a.phase1_root, {}), MODEL, tags[0])
    t0 = time.time()
    mdl, _ = get_model(p1.config()["checkpoint"], a.device)
    pathway = TimesFM3Pathway.from_model(mdl)
    base = load_fresh(MODEL, {"kind": "native", "tiny": False,
                              "checkpoint": os.environ.get("TIMESFM3_CHECKPOINT")}, a.device)
    W, _c = pathway.linear_head()
    mr, Qr = metric_range(W.T @ W, W.shape[0])
    print(f"[physical] models ready in {time.time() - t0:.0f}s on {a.device}; "
          f"{len(tags)} datasets; depths {a.depths}")
    fails = []
    for tag in tags:
        f = dest / f"{tag}.json"
        if f.exists() and not a.force:
            print(f"[skip] {tag}: {f.name} exists")
            continue
        try:
            rec = run_dataset(tag, a, pathway, base, mr, Qr, W.shape[0])
            rec["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            phase2.atomic_write_json(f, rec)
            print(f"[done] {tag}: passed={rec['passed']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append((tag, f"{type(e).__name__}: {e}"))
    s = summarize(dest)
    print(json.dumps(s, indent=1))
    for t, why in fails:
        print("[FAIL]", t, why)
    return 1 if fails or not s["all_passed"] else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS))
    p.add_argument("--output-root", default=str(DEFAULT_OUT))
    p.add_argument("--g2-root", default=str(DEFAULT_OUT), help="the committed Phase-2g tree")
    p.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    p.add_argument("--cache-root", default=os.environ.get(
        "PHASE1_CACHE", str(scratch / "chronos2" / "phase1_shared_cache")))
    p.add_argument("--window-cache", default=os.environ.get(
        "PHASE1_WINDOWS", str(scratch / "chronos2" / "phase1_shared_windows")))
    p.add_argument("--suite", default="paper14")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-window-rebuild", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--batch", type=int, default=64, help="the Phase-1 TimesFM-3 batch size")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
