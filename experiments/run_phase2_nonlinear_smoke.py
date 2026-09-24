"""Phase 2b SMOKE (exploratory): does a nonlinear branch on the output-matched adapter help?

    python -m experiments.run_phase2_nonlinear_smoke --device cuda          # all 3 models
    python -m experiments.run_phase2_nonlinear_smoke --models timesfm3 --device cuda

Pre-registered in notes/PLAN.md ("TimesFM-3 aligned-pathway diagnostic", 2026-09-24) BEFORE any
result. Electricity only; representative depths at 25 / 50 / 75 % of each model's block depth
(Chronos-2 / TiRex L3 L6 L9, TimesFM-3 L5 L10 L15). Per depth, every arm uses the SAME label-free
output-matching objective (MSE to the full model's own normalized head outputs) and the SAME
validation-only selection:

    hard            f(h_l)                                   reference
    affine_closed   closed-form head-weighted ridge          TimesFM-3 only: the committed H3 arm
    affine_iter     residual affine, AdamW (the H3 protocol) the committed H3 arm for C2 / TiRex;
                                                             for TimesFM-3 it isolates the FITTING
                                                             METHOD from the nonlinearity
    nested          affine + W2 gelu(W1 LN(x) + b1), r=64, W2=0 at init (bitwise = affine at
                    epoch 0), same AdamW protocol        isolates the NONLINEARITY

Reported per arm: validation and test MASE ratio vs the full model, and the distillation R^2 (how
well the arm reproduces the full model's head outputs) on train / val / test -- which also answers
the "overfitting vs capacity" diagnostic. The decision rule is evaluated on VALIDATION only.

WHERE TO RUN: a GPU compute node (sbatch job_phase2_nonlinear_smoke.sh). Phase 1 and the committed
H3 tree are READ-ONLY; nothing is written outside --output-root, and no adapter is saved.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2                                                # noqa: E402

DEPTH_FRACTIONS = (0.25, 0.50, 0.75)
DEFAULT_TAG = "monash_electricity_hourly"
DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2_nonlinear_smoke"
COMMITTED_H3 = REPO_ROOT / "results" / "three_model_phase2" / "h3"

# The pre-registered adoption rule (notes/PLAN.md), evaluated on VALIDATION MASE ratios only.
RULE = {"version": "phase2b-nonlinear/v1",
        "text": "adopt the nested adapter beyond the smoke only if, for TimesFM-3, it improves "
                "validation MASE at all three representative depths relative to the committed "
                "affine arm, by >= 10 percentage points at >= 2 of them, while the median "
                "validation MASE of Chronos-2 and of TiRex is not worse than their affine arm by "
                "more than 1 percentage point",
        "min_gain_pp": 10.0, "min_depths_with_gain": 2, "max_other_loss_pp": 1.0}


def representative_depths(L: int) -> list[int]:
    return [int(round(f * L)) for f in DEPTH_FRACTIONS]


def r2(pred, ref) -> float:
    p = np.asarray(pred, np.float64).reshape(len(pred), -1)
    y = np.asarray(ref, np.float64).reshape(len(ref), -1)
    sst = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - ((p - y) ** 2).sum() / sst) if sst > 0 else float("nan")


def committed_noa(model: str, tag: str) -> dict:
    """The committed H3 output-matched arm's ratios, for the reproduction check."""
    import csv
    p = COMMITTED_H3 / model / tag / "depth_metrics.csv"
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        if r["family"] == "noa":
            out[int(r["depth_index"])] = {"val_mase_ratio": float(r["val_mase_ratio_vs_native"]),
                                          "test_mase_ratio": float(r["mase_ratio_vs_native"])}
    return out


def decide(rows: list[dict]) -> dict:
    """Apply RULE to the smoke rows (validation only)."""
    def val(model, arm):
        return {r["depth_index"]: r["val_mase_ratio"] for r in rows
                if r["model"] == model and r["arm"] == arm}
    out = {"rule": RULE}
    t_ref = val("timesfm3", "affine_closed")
    t_new = val("timesfm3", "nested")
    if t_ref and t_new:
        gains = {d: 100 * (t_ref[d] - t_new[d]) for d in t_ref if d in t_new}
        out["timesfm3_gain_pp"] = gains
        out["timesfm3_improves_all"] = bool(gains) and all(g > 0 for g in gains.values())
        out["timesfm3_n_meaningful"] = sum(g >= RULE["min_gain_pp"] for g in gains.values())
    for m in ("chronos2", "tirex"):
        a, n = val(m, "affine_iter"), val(m, "nested")
        if a and n:
            out[f"{m}_median_loss_pp"] = float(100 * (np.median([n[d] for d in n])
                                                      - np.median([a[d] for d in a])))
    ok = (out.get("timesfm3_improves_all") is True
          and out.get("timesfm3_n_meaningful", 0) >= RULE["min_depths_with_gain"]
          and all(out.get(f"{m}_median_loss_pp", 0.0) <= RULE["max_other_loss_pp"]
                  for m in ("chronos2", "tirex")))
    out["complete"] = all(k in out for k in ("timesfm3_gain_pp", "chronos2_median_loss_pp",
                                             "tirex_median_loss_pp"))
    out["adopt"] = bool(ok) if out["complete"] else None
    return out


def run_model(model, tag, args, h3args, bb, fit_cfg, log=print) -> list[dict]:
    import torch
    from experiments.run_phase2_h3 import load_cell_data, windows_readonly
    from probing.phase1_metrics import raw_window_metrics
    from probing.phase2 import MEDIAN_INDEX, QUANTILES, Phase1Cell, resolve_phase1_root
    from probing.phase2_align import (NestedResidualAdapter, affine_to_adapter,
                                      anchored_ridge_path, fit_residual_adapter, mse_loss)
    from probing.window_reference import window_digest

    root = resolve_phase1_root(tag, h3args.phase1_root, phase2.parse_overrides(
        h3args.phase1_override))
    w = windows_readonly(tag, h3args)
    p1 = Phase1Cell(root, model, tag)
    dep = p1.dependency_record()
    if dep["window_digest"] != window_digest(w):
        raise phase2.Phase1DependencyError(f"{model}/{tag}: window digest mismatch")
    handle = bb.pathway(model, p1.config())
    pw = handle["pathway"]
    pw.assert_frozen()
    data = load_cell_data(model, tag, w, p1.arrays(), p1.config(), handle, h3args)
    tr, va, te = (data.splits[s] for s in ("train", "val", "test"))
    L = data.reference_index
    labels = data.depth_labels
    device = h3args.device or pw.device
    q = QUANTILES

    nat = {s: pw.run(sd.feats[L]) for s, sd in (("train", tr), ("val", va), ("test", te))}

    def mase(z, sd):
        return float(np.mean(raw_window_metrics(tag, sd.X, sd.y_raw, pw.to_raw(z, sd), q,
                                                MEDIAN_INDEX)["mase_pw"]))
    nat_val, nat_test = mase(nat["val"][1], va), mase(nat["test"][1], te)
    log(f"  [{model}] native MASE val {nat_val:.4f} test {nat_test:.4f}; L = {labels[L]}")

    def rows_t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

    ref = committed_noa(model, tag)
    rows = []
    for l in representative_depths(L):
        lab = labels[l]
        arms = {"hard": (None, {})}
        if pw.linear_head() is not None:                       # TimesFM-3: the committed arm
            W, _c = pw.linear_head()
            r = anchored_ridge_path(tr.feats[l].reshape(-1, pw.d), tr.feats[L].reshape(-1, pw.d),
                                    va.feats[l].reshape(-1, pw.d), va.feats[L].reshape(-1, pw.d),
                                    metric=W.T @ W, out_dim=W.shape[0], kappas=fit_cfg["kappas"])
            arms["affine_closed"] = (affine_to_adapter(r["A"], r["b"]).to(device).eval(),
                                     {"selected_kappa": r["selected_kappa"],
                                      "n_params": pw.d * pw.d + pw.d})
        for arm, make in (("affine_iter", None),
                          ("nested", lambda d: NestedResidualAdapter(d, r=args.bottleneck))):
            t0 = time.time()
            res = fit_residual_adapter(
                d=pw.d, pathway_fn=pw.outputs, train_x=rows_t(tr.feats[l]),
                train_y=rows_t(nat["train"][0]), val_x=rows_t(va.feats[l]),
                val_y=rows_t(nat["val"][0]), train_loss=mse_loss, val_criterion=mse_loss,
                wd_grid=fit_cfg["wd_grid"], epochs=fit_cfg["epochs"], lr=fit_cfg["lr"],
                eval_every=fit_cfg["eval_every"], device=device, seed=fit_cfg["seed"], log=log,
                label=f"{arm} {lab}", make_adapter=make)
            arms[arm] = (res["adapter"], {"selected_wd": res["selected_wd"],
                                          "selected_epoch": res["selected_epoch"],
                                          "selected_hard_cut": res["selected_hard_cut"],
                                          "n_params": res["n_params"],
                                          "fit_seconds": round(time.time() - t0, 1)})
        for arm, (ad, info) in arms.items():
            out = {s: pw.run(sd.feats[l], ad) for s, sd in (("train", tr), ("val", va),
                                                            ("test", te))}
            row = {"model": model, "dataset": tag, "depth_index": l, "label": lab, "arm": arm,
                   "val_mase_ratio": mase(out["val"][1], va) / nat_val,
                   "test_mase_ratio": mase(out["test"][1], te) / nat_test,
                   **{f"distill_r2_{s}": r2(out[s][0], nat[s][0]) for s in out}, **info}
            if arm in ("affine_closed", "affine_iter") and l in ref and (
                    arm == "affine_closed" or pw.linear_head() is None):
                row["committed_test_mase_ratio"] = ref[l]["test_mase_ratio"]
            rows.append(row)
            log(f"    {lab:<4} {arm:<14} val x{row['val_mase_ratio']:.3f}  test "
                f"x{row['test_mase_ratio']:.3f}  R2 tr/va/te {row['distill_r2_train']:.3f} / "
                f"{row['distill_r2_val']:.3f} / {row['distill_r2_test']:.3f}"
                + (f"   (committed test x{row['committed_test_mase_ratio']:.3f})"
                   if "committed_test_mase_ratio" in row else ""))
    del data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--smoke-output", default=str(DEFAULT_OUT))
    args, rest = p.parse_known_args(argv)
    from experiments.run_phase2_h3 import Backbones, parse_args as h3_parse
    from probing.phase2_env import set_precision_flags
    from probing.phase2_h3 import DEFAULT_FIT_CFG
    h3args = h3_parse(rest)
    numerics = set_precision_flags(deterministic=True)
    tags = h3args.datasets or [DEFAULT_TAG]
    models = [m for m in phase1.MODELS if not h3args.models or m in h3args.models]
    fit_cfg = {**DEFAULT_FIT_CFG, "epochs": h3args.epochs, "lr": h3args.lr,
               "wd_grid": list(phase2.assert_adapter_wd_grid(h3args.wd_grid, h3args.lr)),
               "eval_every": h3args.eval_every, "seed": h3args.seed}
    out = Path(args.smoke_output)
    out.mkdir(parents=True, exist_ok=True)
    bb = Backbones(h3args)
    rows, failures = [], []
    for tag in tags:
        for model in models:
            try:
                rows += run_model(model, tag, args, h3args, bb, fit_cfg)
            except Exception as exc:                                  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures.append({"model": model, "dataset": tag, "error": f"{exc}"[:500]})
    summary = {"schema": "phase2b_nonlinear_smoke/v1", "exploratory": True,
               "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                   timespec="seconds"),
               "bottleneck": args.bottleneck, "depth_fractions": list(DEPTH_FRACTIONS),
               "fit_cfg": fit_cfg, "numerics": numerics, "rows": rows, "failures": failures,
               "decision": decide(rows)}
    phase2.atomic_write_json(out / "summary.json", summary)
    print(f"\nDECISION (validation only): {json.dumps(summary['decision'], indent=1)}")
    print(f"wrote {out / 'summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
