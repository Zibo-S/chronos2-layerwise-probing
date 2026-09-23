"""H4 — physically truncate at the frozen Phase-1 tunnel entrance, verify, and score.

    If lightweight alignment can substitute for the remaining transformation performed by later
    blocks, can those blocks be physically removed for a meaningful reduction in inference cost
    while preserving forecasting performance?

Two stages (both on a COMPUTE NODE with a GPU):

    verify    model-level gates, every depth, before anything is timed:
              V1 truncated states == hooked full model (synthetic contexts; optionally the Phase-1
              cache on real test windows), V2 identity splice at full depth through the public API
              == native, V4 removed blocks never run and are freed, V5 identity adapter == hard cut,
              V6 TiRex passes, PH folded probe head == scaler + probe.
              -> results/three_model_phase2/h4/verification/<model>.json (read by the latency
                 harness, which REFUSES to time a configuration that did not pass)

    evaluate  per model x dataset, at the frozen Phase-1 entrance l_rec (read from the H3 cell):
              native, hard-cut truncation, label-free aligned truncation (NOA), supervised aligned
              truncation (FL; Chronos-2, TiRex), probe-head truncation, and Chronos-2-small (the
              Chronos-2 panel's separately pretrained smaller model -- NOT a depth-only control).
              V3: every physical forecast is compared with the OFFLINE H3 prediction on the same
              windows; test MASE / WQL / MAE, paired degradation vs native, measured active
              parameters.

    python -m experiments.run_phase2_truncation verify   --models chronos2
    python -m experiments.run_phase2_truncation evaluate --models chronos2 --datasets monash_electricity_hourly

The operating point is the frozen Phase-1 entrance, NOT a newly optimized compatibility tunnel
(that stays an appendix analysis). Latency comes from ``run_phase2_latency`` (a dataset-independent
lookup by model x configuration x depth), joined by the table builders.

THE H4 OUTCOME (``probing.phase2.H4_OUTCOME_RULE``, pre-registered): candidate depth = the frozen
Phase-1 entrance; successful truncation = the label-free aligned truncation's test-MASE degradation
<= 5%. A failure is written and reported as a failure -- ``evaluate`` has no depth option, and no
other depth is ever tried for H4.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing.phase2 import (DEFAULT_OUT_ROOT, DEFAULT_PHASE1_ROOT, MEDIAN_INDEX,    # noqa: E402
                            QUANTILES, Phase1Cell, Phase2CellStore, cluster_replicates,
                            parse_overrides, ratio_summary, resolve_phase1_root, standard_wql)

D_MODEL = {"chronos2": 768, "timesfm3": 1280, "tirex": 512}
#: The Phase-1 extraction batch sizes. TiRex's is a REPRODUCIBILITY parameter (bf16 recurrence is
#: batch-size dependent on accelerators), so the physical evaluation uses the same one.
EVAL_BATCH = {"chronos2": 128, "timesfm3": 64, "tirex": 256}
H4_REQUIRED = ("truncation.json", "physical_metrics.npz", "COMPLETE_INPUTS.json")


# --------------------------------------------------------------------------- #
# model handles: load ONCE, hand out fresh deep copies
# --------------------------------------------------------------------------- #
class Fresh:
    def __init__(self, model: str, args):
        from experiments.run_phase2_latency import load_fresh
        self.model = model
        self.base = load_fresh(model, {"kind": "native", "tiny": args.tiny,
                                       "checkpoint": args.checkpoint.get(model)}, args.device)

    def __call__(self):
        return copy.deepcopy(self.base)


def _forecast(handle, model, X, batch_size):
    from probing import phase2_truncate as T
    kw = {"sort_quantiles": False} if model == "timesfm3" else {}
    out = []
    for s in range(0, len(X), batch_size):
        out.append(T.public_forecast(handle, model, X[s:s + batch_size], batch_size=batch_size,
                                     **kw))
    return np.ascontiguousarray(np.concatenate(out).transpose(0, 2, 1))       # (n, 9, H)


# --------------------------------------------------------------------------- #
# stage 1: verify
# --------------------------------------------------------------------------- #
def run_verify(model: str, args) -> dict:
    import torch
    from probing import phase2_truncate as T
    from probing.phase2_env import environment_record, set_precision_flags
    numerics = set_precision_flags(deterministic=True)
    fresh = Fresh(model, args)
    L = len(T.blocks_of(T.core_of(fresh.base, model), model))
    d = 32 if args.tiny else D_MODEL[model]
    rng = np.random.default_rng(0)
    X = (np.cumsum(rng.normal(size=(args.n_verify, phase1.PHASE1_C)), axis=1)
         * rng.uniform(0.5, 3.0, size=(args.n_verify, 1))).astype(np.float32)
    depths = list(range(L + 1)) if not args.depths else [int(x) for x in args.depths]
    recs = {"V1": {}, "V3": {}, "V4": {}, "V5": {}, "PH": {}, "V6": {}}
    t0 = time.time()
    for l in depths:
        recs["V1"][l] = T.verify_state_equivalence(fresh, model, l, X, batch_size=args.n_verify)
        if l < L:
            recs["V4"][l] = T.verify_removed_blocks_never_run(fresh, model, l, X[:4])
        recs["V5"][l] = T.verify_identity_equals_hard_cut(fresh, model, l, X[:4],
                                                          batch_size=4, d=d)
        recs["PH"][l] = _verify_probe_head(fresh, model, l, X[:4], d)
        recs["V3"][l] = _verify_adapter(fresh, model, l, X[:4], d)
        if model == "tirex":        # depth 0 too: Emb is the frozen entrance for SZ Taxi and M5
            recs["V6"][l] = T.verify_tirex_passes(fresh, l, X[:3], d=d)
        print(f"  [{model}] depth {l}: " + " ".join(
            f"{k}={'ok' if v.get(l, {}).get('passed', True) else 'FAIL'}"
            for k, v in recs.items() if l in v))
    v2 = T.verify_identity_splice(fresh, model, X[:4], batch_size=4, d=d)
    cache_v1 = _verify_against_cache(model, args) if args.cache_dataset else None
    ok = lambda k, l: recs[k].get(l, {"passed": True})["passed"]           # noqa: E731
    base_ok = [l for l in depths if ok("V1", l) and ok("V4", l) and ok("V6", l)]
    passed = {"hard": base_ok, "adapter": [l for l in base_ok if ok("V5", l)
                                           and ok("V3", l)],
              "probe_head": [l for l in base_ok if ok("PH", l)]}
    if not v2["passed"]:
        passed = {k: [] for k in passed}
    out = {"model": model, "tiny": bool(args.tiny), "depths": depths, "V2": v2,
           "V1_vs_phase1_cache": cache_v1, "records": recs, "passed_depths": passed,
           "all_passed": bool(v2["passed"] and all(len(v) == len(depths) for v in passed.values())
                              and (cache_v1 is None or cache_v1.get("passed", False))),
           "numerics": numerics, "elapsed_s": round(time.time() - t0, 1),
           "environment": environment_record(),
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    dest = Path(args.output_root) / "h4" / "verification" / f"{model}.json"
    phase2.atomic_write_json(dest, out)
    print(f"[verify] {model}: all_passed={out['all_passed']} -> {dest}")
    del fresh
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _verify_probe_head(fresh, model, depth, X, d) -> dict:
    """PH: folded probe head through the public API == scaler + probe applied OFFLINE to the full
    model's block-l states (random probe: layout, folding and inverse are what is checked)."""
    from probing import phase2_truncate as T
    out = {"chronos2": 9 * 16, "timesfm3": 64 * 9, "tirex": 9 * 32}[model]
    rng = np.random.default_rng(depth)
    probe = {"weight": rng.normal(size=(out, d)) * 0.02, "bias": rng.normal(size=out) * 0.02,
             "scaler_mean": rng.normal(size=d) * 0.1, "scaler_scale": rng.uniform(0.5, 2, d)}
    return T.verify_offline_equals_physical(fresh, model, depth, X, probe=probe,
                                            batch_size=len(X))


def _verify_adapter(fresh, model, depth, X, d) -> dict:
    """V3-synthetic: a NON-trivial adapter, physical == offline, at this depth."""
    import torch
    from probing import phase2_truncate as T
    from probing.phase2_align import ResidualAdapter
    g = torch.Generator().manual_seed(depth)
    ad = ResidualAdapter(d)
    with torch.no_grad():
        ad.delta.copy_(torch.randn(d, d, generator=g) * (0.5 / d ** 0.5))
        ad.bias.copy_(torch.randn(d, generator=g) * 0.05)
    dev = next(T.core_of(fresh.base, model).parameters()).device
    return T.verify_offline_equals_physical(fresh, model, depth, X, adapter=ad.to(dev).eval(),
                                            batch_size=len(X))


def _verify_against_cache(model, args) -> dict:
    """V1 on REAL test windows: truncated readout states == the Phase-1 cached features, at the
    Phase-1 extraction batch size, every depth."""
    from experiments.run_phase2_h3 import load_cell_data, windows_readonly
    from probing import phase2_truncate as T
    tag = args.cache_dataset
    root = resolve_phase1_root(tag, args.phase1_root, parse_overrides(args.phase1_override))
    p1 = Phase1Cell(root, model, tag)
    w = windows_readonly(tag, args)
    handle = {"model": None, "geom": None}
    if model == "tirex":
        from probing.tirex_model import geometry_from_model
        fr = Fresh(model, args)
        handle = {"model": fr.base, "geom": geometry_from_model(fr.base, phase1.PHASE1_C,
                                                                phase1.PHASE1_H)}
    data = load_cell_data(model, tag, w, p1.arrays(), p1.config(), handle, args)
    te = data.splits["test"]
    n = min(args.n_cache_windows, len(te.target))
    X = te.X[:n]
    fresh = Fresh(model, args)
    recs = {}
    for l in range(data.reference_index + 1):
        h = fresh()
        T.truncate(h, model, l)
        got = T.readout_states(h, model, X, full_depth=None, batch_size=EVAL_BATCH[model])
        recs[l] = T.elementwise_gate(got, te.feats[l][:n])
    return {"dataset": tag, "n_windows": int(n), "batch_size": EVAL_BATCH[model],
            "by_depth": recs, "passed": all(r["passed"] for r in recs.values())}


# --------------------------------------------------------------------------- #
# stage 2: evaluate at the frozen Phase-1 entrance
# --------------------------------------------------------------------------- #
def run_evaluate(model: str, tag: str, args) -> dict:
    import torch
    from experiments.run_phase2_h3 import windows_readonly
    from probing import phase2_truncate as T
    from probing.phase1_metrics import raw_window_metrics
    from probing.phase2_align import load_adapter
    from probing.phase2_env import set_precision_flags
    set_precision_flags(deterministic=True)
    h3 = Path(args.output_root) / "h3" / model / tag
    if not (h3 / "COMPLETE").exists():
        raise phase2.Phase1DependencyError(f"no COMPLETE H3 cell at {h3}; run H3 first")
    ladder = json.loads((h3 / "ladder_at_tunnel.json").read_text())
    fits = json.loads((h3 / "fits.json").read_text())
    l_rec, lab = int(ladder["headline"]["depth_index"]), ladder["headline"]["label"]
    root = resolve_phase1_root(tag, args.phase1_root, parse_overrides(args.phase1_override))
    p1 = Phase1Cell(root, model, tag)
    p1_entrance = p1.entrance(phase2.HEADLINE_TOL)
    if l_rec != p1_entrance:
        raise phase2.Phase1DependencyError(
            f"{model}/{tag}: the H3 cell's headline depth {l_rec} is not the frozen Phase-1 "
            f"entrance {p1_entrance}; H4 evaluates the Phase-1 entrance and nothing else")
    # RESUME: an H4 cell is keyed on the exact H3 cell it was built from (its config hash), so a
    # re-run H3 cell makes the H4 cell stale instead of silently reusing it.
    h4_hash = phase1._digest({
        "h3_config_hash": json.loads((h3 / "COMPLETE").read_text()).get("config_hash"),
        "l_rec": l_rec, "v3_rtol": args.v3_rtol, "boot_b": args.boot_b,
        "h4_rule": phase2.H4_OUTCOME_RULE["version"], "chronos2_small": not args.no_small})
    store = Phase2CellStore(args.output_root, "h4", model, tag, required=H4_REQUIRED)
    status, reason = store.status(h4_hash)
    if status == "complete" and not args.force_recompute:
        print(f"[skip] h4 {model}/{tag}: COMPLETE ({reason})")
        return json.loads((store.final / "truncation.json").read_text())
    if status == "incompatible" and not args.force_recompute:
        raise RuntimeError(f"h4 {model}/{tag} exists under a different configuration ({reason}); "
                           "pass --force-recompute or a new --output-root")
    with np.load(h3 / "bootstrap_inputs.npz", allow_pickle=True) as z:
        bi = {k: z[k] for k in z.files}
    with np.load(h3 / "predictions_selected.npz", allow_pickle=True) as z:
        pred = {k: z[k] for k in z.files}
    rows = np.asarray(bi["rows_test"], np.int64)
    w = windows_readonly(tag, args)
    X = np.asarray(w["X_test"], np.float32)[rows]
    y_raw = np.asarray(pred["y_raw"], np.float64)
    cid = np.asarray(bi["cluster_ids_test"], np.int64)
    fresh = Fresh(model, args)
    L = len(T.blocks_of(T.core_of(fresh.base, model), model))
    native_params = T.param_breakdown(fresh.base, model)["active_params"]
    arms, per_window, v3 = {}, {}, {}
    B = EVAL_BATCH[model]

    def score(name, raw):
        m = raw_window_metrics(tag, X, y_raw, raw, QUANTILES, MEDIAN_INDEX)
        per_window[f"{name}__mase"] = m["mase_pw"]
        per_window[f"{name}__mae"] = m["mae_pw"]
        per_window[f"{name}__wql_num"] = m["wql_num_pw"]
        per_window["wql_den"] = m["wql_den_pw"]
        return m

    t0 = time.time()
    specs = [("native", None, None), ("hard", l_rec, None)]
    for fam in ("noa", "fl"):
        f = fits.get(fam, {}).get(lab)
        if f and f.get("fitted") and f.get("adapter_file"):
            specs.append((fam, l_rec, f["adapter_file"]["path"]))
        elif f is not None and not f.get("fitted"):
            specs.append((fam, l_rec, None))                    # reference depth: no adapter
    specs.append(("probe_head", l_rec, "probe"))
    for name, depth, src in specs:
        h = fresh()
        rec = {"arm": name, "depth_index": depth, "label": None if depth is None else lab,
               "blocks_removed": 0 if depth is None else L - depth}
        if name != "native":
            if src == "probe":
                slug = lab.replace("+", "_").replace(" ", "_")
                with np.load(p1.path / "probe_artifacts" / f"probe__{slug}.npz") as z:
                    probe = {k: z[k] for k in z.files}
                T.truncate(h, model, depth)
                rec["head"] = T.install_probe_head(h, model, probe,
                                                   native_quantiles=getattr(h, "quantiles", None))
            elif src is not None:
                ad, meta = load_adapter(src, device=next(T.core_of(h, model).parameters()).device)
                if meta.get("sha256") != fits[name][lab]["adapter_sha256"]:
                    raise RuntimeError(f"{name} adapter checksum differs from the H3 fit record")
                T.truncate(h, model, depth, adapter=ad)
                rec["adapter_params"] = ad.n_params
            else:
                T.truncate(h, model, depth)
        rec["active_params"] = T.param_breakdown(h, model)["active_params"]
        rec["fraction_params_removed"] = 1.0 - rec["active_params"] / native_params
        raw = _forecast(h, model, X, B)
        m = score(name, raw)
        rec["test_mase"] = float(m["mase_pw"].mean())
        rec["test_wql"] = standard_wql(m["wql_num_pw"], m["wql_den_pw"])
        rec["test_mae"] = float(m["mae_pw"].mean())
        # ---- V3: physical == offline H3 ----------------------------------------------
        key = "native__raw" if name == "native" else f"{name}__{lab}__raw"
        if name == "probe_head":
            ref_mase = np.asarray(p1.arrays()["test_mase_window"], np.float64)[
                phase1.model_spec(model).depth_indices[l_rec]]
            v3[name] = {"check": "V3 (metric level vs the Phase-1 probe)",
                        "mase_mean_rel_diff": float(abs(rec["test_mase"] - ref_mase.mean())
                                                    / ref_mase.mean())}
            v3[name]["passed"] = v3[name]["mase_mean_rel_diff"] <= args.v3_rtol
        elif key in pred:
            ref = np.asarray(pred[key], np.float64)
            scale = float(np.abs(ref).mean())
            g = T.elementwise_gate(raw, ref, atol=args.v3_rtol * scale, rtol=args.v3_rtol)
            src_key = "native_test_mase" if name == "native" else f"{name}__test_mase"
            ref_m = np.asarray(bi[src_key], np.float64)
            ref_m = ref_m if ref_m.ndim == 1 else ref_m[l_rec]
            g["mase_mean_rel_diff"] = float(abs(rec["test_mase"] - ref_m.mean()) / ref_m.mean())
            g["passed"] = bool(g["passed"] and g["mase_mean_rel_diff"] <= args.v3_rtol)
            v3[name] = {"check": "V3 (elementwise vs offline H3)", **g}
        rec["V3_passed"] = v3.get(name, {}).get("passed")
        arms[name] = rec
        print(f"  [{model}/{tag}] {name:<11} {lab if depth is not None else 'native':<6} "
              f"MASE {rec['test_mase']:.4f}  WQL {rec['test_wql']:.4f}  "
              f"params {rec['active_params']:,}  V3 {rec['V3_passed']}")
        del h
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model == "chronos2" and not args.no_small:
        from experiments.run_phase2_latency import load_fresh
        h = load_fresh("chronos2", {"kind": "chronos2_small", "tiny": args.tiny}, args.device)
        raw = _forecast(h, "chronos2", X, B)
        m = score("chronos2_small", raw)
        arms["chronos2_small"] = {
            "arm": "chronos2_small", "active_params": T.param_breakdown(h, "chronos2")[
                "active_params"], "test_mase": float(m["mase_pw"].mean()),
            "test_wql": standard_wql(m["wql_num_pw"], m["wql_den_pw"]),
            "test_mae": float(m["mae_pw"].mean()),
            "note": "separately pretrained smaller model (6 blocks, d=512, 13 quantiles): a "
                    "practical deployment baseline, NOT a depth-only control"}
        del h

    # ---- paired degradation vs native (the shared cluster bootstrap) ------------------------
    names = list(arms)
    W = np.stack([per_window[f"{n}__mase"] for n in names])
    rep = cluster_replicates(W, cid, args.boot_b, phase2.SEED)
    Nw = np.stack([per_window[f"{n}__wql_num"] for n in names])
    rn = cluster_replicates(Nw, cid, args.boot_b, phase2.SEED, reduce="sum")
    i0 = names.index("native")
    for i, n in enumerate(names):
        arms[n]["mase_vs_native"] = ratio_summary(W[i].mean(), W[i0].mean(), rep[:, i],
                                                  rep[:, i0])
        arms[n]["wql_vs_native"] = ratio_summary(Nw[i].sum(), Nw[i0].sum(), rn[:, i], rn[:, i0])
    low = (json.loads((h3 / "summary.json").read_text()).get("low_skill") or {}).get("flag")
    outcome = phase2.h4_outcome(arms, candidate_depth=l_rec, phase1_entrance=p1_entrance,
                                n_blocks=L, label=lab, low_skill=low)
    out = {"model": model, "dataset": tag, "operating_point": "frozen Phase-1 sustained 5% "
           "tunnel entrance", "label": lab, "depth_index": l_rec, "blocks_total": L,
           "h4_outcome": outcome,
           "native_active_params": native_params, "arms": arms, "V3": v3,
           "all_V3_passed": all(v.get("passed") for v in v3.values()),
           "n_test_windows": int(len(X)), "eval_batch_size": B, "elapsed_s":
               round(time.time() - t0, 1), "h3_cell": str(h3)}
    stage = store.begin()
    try:
        phase2.atomic_write_json(stage / "truncation.json", out)
        np.savez_compressed(stage / "physical_metrics.npz", cluster_ids_test=cid, **per_window)
        phase2.atomic_write_json(stage / "COMPLETE_INPUTS.json", {
            "h3_complete": json.loads((h3 / "COMPLETE").read_text()),
            "phase1": p1.dependency_record()})
        store.commit(stage, h4_hash)
    except Exception:
        store.abandon(stage)
        raise
    d, (lo, hi) = outcome["aligned_mase_degradation"], outcome["aligned_mase_degradation_ci"]
    print(f"[H4 OUTCOME] {model}/{tag} at {lab} ({outcome['blocks_removed']} of {L} blocks "
          f"removed): {outcome['status'].upper()} -- aligned dMASE {100 * d:+.1f}% "
          f"[{100 * lo:+.1f}, {100 * hi:+.1f}] vs budget {100 * phase2.BUDGET:.0f}%"
          + ("  (fragile: the CI contains the budget)" if outcome["fragile"] else "")
          + ("" if outcome["status"] != "failure" else
             "  -- reported as a failure; no other depth is evaluated"))
    if not out["all_V3_passed"]:
        print(f"[WARN] {model}/{tag}: a physical model does NOT reproduce its offline H3 "
              f"prediction: { {k: v.get('passed') for k, v in v3.items()} } -- do not report it")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.stage == "evaluate" and args.depths:
        raise SystemExit("evaluate has exactly one operating point per model x dataset -- the "
                         "frozen Phase-1 entrance (probing.phase2.H4_OUTCOME_RULE); --depths is "
                         "for verify only")
    fails = []
    if args.stage == "verify":
        for m in args.models:
            try:
                run_verify(m, args)
            except Exception as e:
                import traceback
                traceback.print_exc()
                fails.append((m, str(e)))
    else:
        tags = [t for t in registry.roster("paper14") if not args.datasets or t in args.datasets]
        for m in args.models:
            for t in tags:
                try:
                    run_evaluate(m, t, args)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    fails.append((m, t, str(e)))
    for f in fails:
        print("[FAIL]", f)
    return 1 if fails else 0


def parse_args(argv=None):
    import os
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["verify", "evaluate"])
    p.add_argument("--models", nargs="+", default=list(phase1.MODELS), choices=list(phase1.MODELS))
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--depths", nargs="+", type=int, default=None,
                   help="verify only; evaluate refuses it (one operating point: the Phase-1 "
                        "entrance)")
    p.add_argument("--output-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    p.add_argument("--phase1-override", action="append", default=[])
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    p.add_argument("--cache-root", default=os.environ.get(
        "PHASE1_CACHE", str(scratch / "chronos2" / "phase1_shared_cache")))
    p.add_argument("--window-cache", default=os.environ.get(
        "PHASE1_WINDOWS", str(scratch / "chronos2" / "phase1_shared_windows")))
    p.add_argument("--chronos-features-dir", default=None)
    p.add_argument("--allow-window-rebuild", action="store_true")
    p.add_argument("--suite", default="paper14")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dataset", default=None,
                   help="verify: also check V1 against the Phase-1 cache on this dataset's test "
                        "windows (default: skipped)")
    p.add_argument("--n-verify", type=int, default=8)
    p.add_argument("--n-cache-windows", type=int, default=256)
    p.add_argument("--v3-rtol", type=float, default=1e-4)
    p.add_argument("--boot-b", type=int, default=phase2.BOOT_B)
    p.add_argument("--no-small", action="store_true", help="skip Chronos-2-small")
    p.add_argument("--force-recompute", action="store_true",
                   help="evaluate: rebuild H4 cells that are already COMPLETE (default: skip them, "
                        "so resubmitting the same line resumes)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.checkpoint = {"timesfm3": os.environ.get("TIMESFM3_CHECKPOINT"),
                    "tirex": os.environ.get("TIREX_CHECKPOINT")}
    return a


if __name__ == "__main__":
    raise SystemExit(main())
