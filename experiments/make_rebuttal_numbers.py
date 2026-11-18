"""Every number quoted in the ICLR 2027 rebuttal, recomputed from committed result tables.

    python experiments/make_rebuttal_numbers.py [--root .] [--out results/rebuttal_numbers]

Read-only over the result trees; stdlib only (runs on the Mac base Python and on Narval). Writes
``<out>/rebuttal_numbers.json`` and ``<out>/rebuttal_numbers.txt`` (the printed report).

Sources (all relative to the repository root):
  results/three_model_phase2/combined/          H3/H4 (identity-anchored adapters, submission)
  results/three_model_phase1_q9_expanded/       Phase-1 readouts (LOOP Seattle: *_loop_lowlr)
  results/three_model_phase2_lodo/combined/     Phase 2c, leave-one-dataset-out transfer
  results/three_model_phase2f_source_scaling/   Phase 2f, number of source datasets
  results/three_model_phase2e_noncontig/tables/ Phase 2e, non-contiguous blocks
  results/three_model_phase2g_timesfm3_zero_ridge/combined/  Phase 2g, zero-anchored TimesFM-3

Definitions that matter for the rebuttal (Reviewer 96qN, Q1):
  * readout tunnel (paper, Eq. 2): R_j <= (1+eps) R_L for all j in [l, L], R = readout val loss.
  * native-referenced readout tunnel, DIRECT substitution: R_j <= (1+eps) V_native for all
    j in [l, L]; the final-block readout R_L is part of the suffix.
  * native-referenced H3 "compatibility" tunnel (probing.phase2.compatibility_tunnel): the curve
    is [R_0 .. R_{L-1}, V_native]; the final-block readout is REPLACED by the full model, so only
    readouts at truncation depths 0..L-1 are tested. For the hard cut and the aligned pathway the
    two definitions coincide (both are the full model at depth L).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

MODELS = ("chronos2", "timesfm3", "tirex")
NAME = {"chronos2": "Chronos-2", "timesfm3": "TimesFM-3", "tirex": "TiRex"}
L = {"chronos2": 12, "timesfm3": 20, "tirex": 12}
QUARTER_DEPTHS = {"chronos2": (3, 6, 9), "timesfm3": (5, 10, 15), "tirex": (3, 6, 9)}
DISPLAY = {"m4_hourly": "M4", "monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber TLC",
           "wind_farms_hourly": "Wind Farms", "boom_hourly": "BOOM", "sg_carpark": "SG Carpark",
           "coastal_ts": "Coastal T-S", "LOOP_SEATTLE_5T": "LOOP Seattle",
           "electricity_15min": "Electricity (15-min)", "SZ_TAXI_15T": "SZ Taxi",
           "monash_london_smart_meters": "London Smart Meters", "m5": "M5",
           "wiki_daily_100k": "Wiki Daily", "monash_traffic": "Monash Traffic"}
# the paper's Table tab:provenance (probing/registry.py is stale for Chronos-2)
HELD = {"chronos2": ("LOOP_SEATTLE_5T", "SZ_TAXI_15T", "m5", "monash_traffic"),
        "timesfm3": ("LOOP_SEATTLE_5T", "SZ_TAXI_15T", "m5"),
        "tirex": ("m5", "monash_traffic")}
NOT_LISTED = {"chronos2": ("boom_hourly", "sg_carpark", "coastal_ts"),
              "timesfm3": ("m4_hourly", "monash_electricity_hourly", "boom_hourly", "sg_carpark",
                           "coastal_ts", "electricity_15min"),
              "tirex": ("boom_hourly", "sg_carpark", "coastal_ts", "LOOP_SEATTLE_5T", "SZ_TAXI_15T")}

OUT: dict = {}
LINES: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def put(key, value):
    OUT[key] = value
    return value


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def med(v):
    v = [x for x in v if x == x]
    return st.median(v) if v else float("nan")


def quart(v):
    v = sorted(x for x in v if x == x)
    q = st.quantiles(v, n=4, method="inclusive")
    return q[0], q[2]


def pct(r, ratio=True):
    x = 100 * (r - 1) if ratio else 100 * r
    return f"{x:+.1f}%"


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def sustained_entrance(curve, ref, eps):
    """min l with curve[j] <= (1+eps)*ref for all j >= l; len(curve) if none (no tunnel)."""
    ent = len(curve)
    for j in range(len(curve) - 1, -1, -1):
        if curve[j] <= (1 + eps) * ref:
            ent = j
        else:
            break
    return ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/rebuttal_numbers")
    a = ap.parse_args()
    R = Path(a.root)
    P2 = R / "results/three_model_phase2/combined"

    # ------------------------------------------------------------------ H3 tables
    dm = rows(P2 / "h3_depth_metrics.csv")
    D = {(r["model"], r["dataset"], r["family"], int(r["depth_index"])): r for r in dm}
    DS = sorted({r["dataset"] for r in dm})
    assert len(DS) == 14, DS
    ENT = {(r["model"], r["dataset"]): int(r["depth_index"]) for r in dm
           if r["family"] == "hard" and r["is_phase1_entrance"] == "True"}
    T = {(r["model"], r["dataset"], r["family"], r["tolerance"]): int(r["compatibility_entrance"])
         for r in rows(P2 / "h3_tunnels.csv")}
    SPD = {}
    for r in rows(P2 / "h4_frontier_points.csv"):
        if r["arm"] == "noa" and r["speedup_e2e_b1"]:
            SPD[(r["model"], int(r["depth_index"]))] = (f(r["speedup_e2e_b1"]), f(r["speedup_e2e_b256"]))
    LOW = {(r["model"], r["dataset"]): r["low_skill"] == "True" for r in rows(P2 / "h4_outcomes.csv")}

    def ratio(m, ds, fam, d):
        return f(D[(m, ds, fam, d)]["mase_ratio_vs_native"])

    def vloss(m, ds, fam, d):
        return f(D[(m, ds, fam, d)]["val_loss"])

    # sanity: hard cut at the final block IS the full model
    for m in MODELS:
        for ds in DS:
            assert abs(ratio(m, ds, "hard", L[m]) - 1) < 1e-9, (m, ds)

    # ------------------------------------------------------------------ Phase-1 floors
    def p1_root(ds):
        return R / ("results/three_model_phase1_q9_loop_lowlr" if ds == "LOOP_SEATTLE_5T"
                    else "results/three_model_phase1_q9_expanded")

    FLOOR = {}   # (m, ds) -> (final-block readout val/floor, floor)
    P1LM = {}    # (m, ds) -> {point_index: (val_loss, test_loss)}
    for m in MODELS:
        for ds in DS:
            ws = rows(p1_root(ds) / m / ds / "wd_selection.csv")
            fin = [r for r in ws if r["include_in_main_depth_axis"] == "True"
                   and int(r["point_index"]) == L[m]]
            assert len(fin) == 1, (m, ds)
            FLOOR[(m, ds)] = (f(fin[0]["val_over_floor"]), f(fin[0]["constant_forecast_floor_val_loss"]))
            lm = rows(p1_root(ds) / m / ds / "layer_metrics.csv")
            P1LM[(m, ds)] = {int(r["point_index"]): (f(r["val_loss"]), f(r["test_loss"])) for r in lm
                             if r.get("include_in_main_depth_axis", "True") == "True"}

    # ------------------------------------------------------------------ Phase 2g
    G = R / "results/three_model_phase2g_timesfm3_zero_ridge/combined"
    gdm = rows(G / "depth_metrics.csv")
    GD = {(r["dataset"], r["arm"], int(r["depth_index"])): r for r in gdm}
    GENT = {(r["dataset"], r["arm"]): r for r in rows(G / "entrance_budget.csv")}
    GFIT = {(int(r["depth_index"]), r["family"]): r for r in rows(G / "fit_by_depth.csv")}

    def gz(ds, arm, d):
        return f(GD[(ds, arm, d)]["mase_ratio_vs_native"])

    # the 2g identity arm must reproduce the committed H3 aligned pathway (gate of the run)
    worst = max(abs(gz(ds, "identity", d) / ratio("timesfm3", ds, "noa", d) - 1)
                for ds in DS for d in range(20))
    put("g.identity_reproduces_h3_max_rel", worst)
    assert worst < 1e-4, worst

    # ================================================================== A. 96qN Q1
    say("=" * 100)
    say("A. 96qN Q1: tunnel referenced to the full model (validation Q9 loss)")
    tot = defaultdict(int)
    for m in MODELS:
        fb = [vloss(m, ds, "probe", L[m]) / vloss(m, ds, "hard", L[m]) for ds in DS]
        within = [ds for ds, x in zip(DS, fb) if x <= 1.05]
        put(f"A.{m}.final_readout_vs_native_val_median", med(fb) - 1)
        put(f"A.{m}.final_readout_within5", within)
        say(f"  {NAME[m]:9s} final-block readout val loss vs full model: median {pct(med(fb))} "
            f"[{pct(min(fb))}, {pct(max(fb))}]; within 5%: {len(within)}/14 {[DISPLAY[d] for d in within]}")
        tot["final_within5"] += len(within)
        for eps in (0.02, 0.05, 0.10):
            direct, h3 = [], []
            for ds in DS:
                curve = [vloss(m, ds, "probe", j) for j in range(L[m] + 1)]
                vn = vloss(m, ds, "hard", L[m])
                e_direct = sustained_entrance(curve, vn, eps)          # L+1 == no tunnel
                e_h3 = sustained_entrance(curve[:-1] + [vn], vn, eps)   # final readout replaced
                assert e_h3 == T[(m, ds, "probe", f"tol_{eps:g}")], (m, ds, eps, e_h3)
                direct.append(e_direct)
                h3.append(e_h3)
            n_direct_any = sum(e <= L[m] for e in direct)
            n_direct_pre = sum(e < L[m] for e in direct)
            n_h3_pre = sum(e < L[m] for e in h3)
            tot[f"direct_any_{eps}"] += n_direct_any
            tot[f"direct_pre_{eps}"] += n_direct_pre
            tot[f"h3_pre_{eps}"] += n_h3_pre
            pre_pairs = [DISPLAY[ds] for ds, e in zip(DS, direct) if e < L[m]]
            h3_only = [DISPLAY[ds] for ds, e, e2 in zip(DS, direct, h3) if e2 < L[m] and not e < L[m]]
            say(f"     eps {eps:.2f}: direct substitution: tunnel {n_direct_any}/14, pre-final "
                f"{n_direct_pre}/14 {pre_pairs}; H3 (final readout replaced): pre-final {n_h3_pre}/14"
                f" (only under H3: {h3_only})")
    for k in sorted(tot):
        put(f"A.total.{k}", tot[k])
    say(f"  TOTAL final-block readout within 5% of full model: {tot['final_within5']}/42")
    for eps in (0.02, 0.05, 0.10):
        say(f"  TOTAL eps {eps:.2f}: direct: any {tot[f'direct_any_{eps}']}/42, pre-final "
            f"{tot[f'direct_pre_{eps}']}/42 | H3 definition pre-final {tot[f'h3_pre_{eps}']}/42")

    say("  aligned pathway, native-referenced sustained tunnel (pre-final counts of 14):")
    for tol in ("tol_0.02", "tol_0.05", "tol_0.1"):
        parts, tot_id, tot_z = [], 0, 0
        for m in MODELS:
            n_id = sum(T[(m, ds, "noa", tol)] < L[m] for ds in DS)
            tot_id += n_id
            if m == "timesfm3":
                n_z = sum(int(GENT[(ds, "zero")][f"native_tunnel_{tol}"]) < 20 for ds in DS)
                n_gid = sum(int(GENT[(ds, "identity")][f"native_tunnel_{tol}"]) < 20 for ds in DS)
                assert n_gid == n_id, (n_gid, n_id)
                tot_z += n_z
                parts.append(f"{NAME[m]} identity {n_id} / zero {n_z}")
                put(f"A.aligned_native_tunnel.{tol}.timesfm3_zero", n_z)
            else:
                tot_z += n_id
                parts.append(f"{NAME[m]} {n_id}")
            put(f"A.aligned_native_tunnel.{tol}.{m}", n_id)
        put(f"A.aligned_native_tunnel.{tol}.total_identity", tot_id)
        put(f"A.aligned_native_tunnel.{tol}.total_with_tfm3_zero", tot_z)
        say(f"     {tol}: {'; '.join(parts)} | total {tot_id}/42 (with TimesFM-3 zero: {tot_z}/42)")
    for m in MODELS:
        e1 = [ENT[(m, ds)] for ds in DS]
        c = [T[(m, ds, "noa", "tol_0.05")] for ds in DS]
        rho = spearman(e1, c)
        near = sum(abs(x - y) <= 1 for x, y in zip(e1, c))
        put(f"A.spearman_p1_vs_aligned5.{m}", rho)
        put(f"A.within_one_block.{m}", near)
        say(f"     Spearman(Phase-1 entrance, aligned 5% native-referenced entrance) {NAME[m]}: "
            f"{rho:+.2f}; within one block {near}/14; median aligned entrance L{med(c)}")
    cz = [int(GENT[(ds, "zero")]["native_tunnel_tol_0.05"]) for ds in DS]
    e1 = [ENT[("timesfm3", ds)] for ds in DS]
    put("A.spearman_p1_vs_aligned5.timesfm3_zero", spearman(e1, cz))
    put("A.timesfm3_zero_median_aligned5_entrance", med(cz))
    say(f"     TimesFM-3 zero anchor: Spearman {spearman(e1, cz):+.2f}; within one block "
        f"{sum(abs(x - y) <= 1 for x, y in zip(e1, cz))}/14; median entrance L{med(cz)}")

    say("  the seven L0 entrances: final-block readout val loss / constant-forecast floor; full model low-skill flag")
    l0 = [(m, ds) for m in MODELS for ds in DS if ENT[(m, ds)] == 0]
    put("A.l0_pairs", [f"{m}/{ds}" for m, ds in l0])
    for m, ds in l0:
        say(f"     {NAME[m]:9s} {DISPLAY[ds]:22s} readout(final)/floor {FLOOR[(m, ds)][0]:.3f}; "
            f"full-model low-skill {LOW[(m, ds)]}")
        put(f"A.l0.{m}.{ds}.final_readout_over_floor", FLOOR[(m, ds)][0])

    # ================================================================== B. 2p99 Q2
    say("=" * 100)
    say("B. 2p99 Q2: cutting at the frozen Phase-1 entrance (aligned pathway, test MASE vs full model)")
    for m in MODELS:
        cut = [ds for ds in DS if ENT[(m, ds)] < L[m]]
        deg = [ratio(m, ds, "noa", ENT[(m, ds)]) for ds in cut]
        spd = [SPD[(m, ENT[(m, ds)])][0] for ds in cut]
        res = {"n": len(cut), "median": med(deg) - 1, "within5": sum(x <= 1.05 for x in deg),
               "worst": max(deg) - 1, "median_speedup_b1": med(spd)}
        put(f"B.entrance_rule.{m}", res)
        say(f"  {NAME[m]:9s} n={res['n']} median {pct(med(deg))}, within 5% {res['within5']}/{res['n']}, "
            f"worst {pct(max(deg))}, median speedup {med(spd):.2f}x")
        for d in ((3, 4, 9) if L[m] == 12 else (1, 5, 10)):
            dg = [ratio(m, ds, "noa", d) for ds in DS]
            put(f"B.fixed.{m}.L{d}", {"median": med(dg) - 1, "within5": sum(x <= 1.05 for x in dg),
                                       "worst": max(dg) - 1, "speedup_b1": SPD[(m, d)][0]})
            say(f"       fixed L{d}: median {pct(med(dg))}, within 5% {sum(x <= 1.05 for x in dg)}/14, "
                f"worst {pct(max(dg))} ({DISPLAY[max(DS, key=lambda s: ratio(m, s, 'noa', d))]}), "
                f"speedup {SPD[(m, d)][0]:.2f}x")
    zr = [f(GENT[(ds, "zero")]["mase_ratio_at_entrance"]) for ds in DS]
    put("B.entrance_rule.timesfm3_zero", {"within5": sum(x <= 1.05 for x in zr), "median": med(zr) - 1,
                                          "worst": max(zr) - 1})
    say(f"  TimesFM-3 zero anchor at the entrance: within 5% {sum(x <= 1.05 for x in zr)}/14, "
        f"median {pct(med(zr))}, worst {pct(max(zr))}")

    # ================================================================== C. 2p99 Q3
    say("=" * 100)
    say("C. 2p99 Q3: pairs documented as held out from pretraining")
    held = [(m, ds) for m in MODELS for ds in HELD[m]]
    pre = sum(ENT[p] < L[p[0]] for p in held)
    dtest = []
    for m, ds in held:
        lm = P1LM[(m, ds)]
        dtest.append(lm[ENT[(m, ds)]][1] / lm[L[m]][1] - 1)
    alltest = [P1LM[(m, ds)][ENT[(m, ds)]][1] / P1LM[(m, ds)][L[m]][1] - 1 for m in MODELS for ds in DS]
    put("C.n_held", len(held))
    put("C.held_prefinal", pre)
    put("C.held_dtest_median", med(dtest))
    put("C.held_dtest_range", [min(dtest), max(dtest)])
    put("C.all_dtest_median", med(alltest))
    say(f"  {len(held)} held-out pairs; tunnel before final block: {pre}; test loss change at entrance: "
        f"median {100 * med(dtest):+.1f}% [{100 * min(dtest):+.1f}%, {100 * max(dtest):+.1f}%] "
        f"(all 42: {100 * med(alltest):+.1f}%)")
    nd = [(m, ds) for m in MODELS for ds in HELD[m] + NOT_LISTED[m]]
    put("C.not_documented_pairs", len(nd))
    put("C.not_documented_prefinal", sum(ENT[p] < L[p[0]] for p in nd))
    say(f"  {len(nd)} pairs not documented as pretraining data: pre-final {sum(ENT[p] < L[p[0]] for p in nd)}")
    for m in MODELS:
        sub = HELD[m]
        line = f"  {NAME[m]:9s} (n={len(sub)})"
        for d in QUARTER_DEPTHS[m]:
            h = med([ratio(m, ds, 'hard', d) for ds in sub])
            al = med([ratio(m, ds, 'noa', d) for ds in sub])
            pr = med([ratio(m, ds, 'probe', d) for ds in sub])
            put(f"C.{m}.L{d}", {"hard": h - 1, "aligned_identity": al - 1, "probe": pr - 1})
            line += f" | L{d} hard {pct(h)} aligned {pct(al)} probe {pct(pr)}"
            if m == "timesfm3":
                z = med([gz(ds, 'zero', d) for ds in sub])
                put(f"C.{m}.L{d}.zero", z - 1)
                line += f" zero {pct(z)}"
        say(line)
        allq = [pct(med([ratio(m, ds, 'noa', d) for ds in DS])) for d in QUARTER_DEPTHS[m]]
        say(f"       all 14 aligned at 25/50/75%: {allq}")
    for ds in HELD["chronos2"]:
        say(f"     Chronos-2 L4 aligned {DISPLAY[ds]}: {pct(ratio('chronos2', ds, 'noa', 4))}; "
            f"full-model low-skill {LOW[('chronos2', ds)]}")
    # full-model validation loss / closed-form constant-forecast floor, as stored by H3 (both are
    # mean pinball in the model's target space; the low-skill flag is native/floor >= 0.9)
    say("  full-model validation loss / constant-forecast floor (held-out datasets):")
    for m in MODELS:
        for ds in HELD[m]:
            s = json.loads((R / "results/three_model_phase2/h3" / m / ds / "summary.json").read_text())
            v = s["low_skill"]["native_val_over_floor"]
            put(f"C.native_over_floor.{m}.{ds}", v)
            say(f"     {NAME[m]:9s} {DISPLAY[ds]:16s} {v:.2f}")

    # ================================================================== D. 96qN Q3 (Phase 2g)
    say("=" * 100)
    say("D. 96qN Q3: TimesFM-3 identity- vs zero-anchored closed form (Phase 2g)")
    say("  depth | hard | identity | zero | zero_out | probe | zero<=5% | zero<id (CI) | zero vs probe better/worse (CI)")
    for d in range(20):
        hz = med([gz(ds, 'hard', d) for ds in DS])
        idv = med([gz(ds, 'identity', d) for ds in DS])
        zv = med([gz(ds, 'zero', d) for ds in DS])
        zo = med([gz(ds, 'zero_out', d) for ds in DS])
        pr = med([gz(ds, 'probe', d) for ds in DS])
        w5 = sum(gz(ds, 'zero', d) <= 1.05 for ds in DS)
        w5i = sum(gz(ds, 'identity', d) <= 1.05 for ds in DS)
        zi = sum(f(GD[(ds, 'zero', d)]["mase_ratio_vs_identity_ci_hi"]) < 1 for ds in DS)
        zpb = sum(f(GD[(ds, 'zero', d)]["mase_ratio_vs_probe_ci_hi"]) < 1 for ds in DS)
        zpw = sum(f(GD[(ds, 'zero', d)]["mase_ratio_vs_probe_ci_lo"]) > 1 for ds in DS)
        lz, hz2 = quart([gz(ds, 'zero', d) for ds in DS])
        put(f"D.L{d}", {"hard_x": hz, "identity": idv - 1, "zero": zv - 1, "zero_out": zo - 1,
                        "probe": pr - 1, "zero_within5": w5, "identity_within5": w5i,
                        "zero_better_identity_ci": zi, "zero_better_probe_ci": zpb,
                        "zero_worse_probe_ci": zpw, "zero_iqr": [lz - 1, hz2 - 1],
                        "speedup": SPD.get(("timesfm3", d))})
        say(f"  L{d:<3} {hz:6.0f}x {pct(idv):>8} {pct(zv):>7} {pct(zo):>7} {pct(pr):>7} | {w5:>2} (id {w5i}) | "
            f"{zi:>2} | {zpb}/{zpw} | IQR [{pct(lz)}, {pct(hz2)}] | speedup {SPD.get(('timesfm3', d))}")
    # selection and fit diagnostics
    kid = [f(GD[(ds, 'identity', d)]["selected_kappa"]) for ds in DS for d in range(20)]
    kz = [f(GD[(ds, 'zero', d)]["selected_kappa"]) for ds in DS for d in range(20)]
    put("D.identity_kappa_le_1e-8", sum(k <= 1e-8 for k in kid))
    put("D.zero_kappa_1e-6_to_1e-4", sum(1e-6 * 0.999 <= k <= 1e-4 * 1.001 for k in kz))
    put("D.zero_kappa_ge_10", sum(k >= 10 * 0.999 for k in kz))
    zk_hi = sorted({DISPLAY[ds] for ds in DS for d in range(20) if f(GD[(ds, 'zero', d)]["selected_kappa"]) >= 9.99})
    vpref = sum(f(GD[(ds, 'zero', d)]["distill_val"]) < f(GD[(ds, 'identity', d)]["distill_val"])
                for ds in DS for d in range(20))
    put("D.val_prefers_zero", vpref)
    say(f"  identity kappa <= 1e-8: {OUT['D.identity_kappa_le_1e-8']}/280; zero kappa in [1e-6, 1e-4]: "
        f"{OUT['D.zero_kappa_1e-6_to_1e-4']}/280; zero kappa >= 10: {OUT['D.zero_kappa_ge_10']}/280 on {zk_hi}; "
        f"validation distillation error lower for zero: {vpref}/280")
    for fam in ("identity", "zero"):
        rr = [GFIT[(d, fam)] for d in range(1, 18)]
        vt = [f(r["val_over_train_median"]) for r in rr]
        dof = [f(r["dof_weighted_median"]) for r in rr]
        r2 = [f(r["r2_test_median"]) for r in rr]
        neg = [int(r["n_r2_test_negative"]) for r in rr]
        put(f"D.fit.{fam}.L1_L17", {"val_over_train": [min(vt), max(vt)], "dof_weighted": [min(dof), max(dof)],
                                    "r2_test": [min(r2), max(r2)], "n_r2_negative": [min(neg), max(neg)]})
        r0 = GFIT[(0, fam)]
        put(f"D.fit.{fam}.L0", {"val_over_train": f(r0["val_over_train_median"]),
                                "dof_weighted": f(r0["dof_weighted_median"]), "r2_test": f(r0["r2_test_median"])})
        say(f"  {fam:8s} L1-L17: val/train {min(vt):.1f}-{max(vt):.1f}; eff. params {min(dof):.0f}-{max(dof):.0f}; "
            f"test R2 {min(r2):+.2f} to {max(r2):+.2f}; negative on {min(neg)}-{max(neg)} | L0: val/train "
            f"{f(r0['val_over_train_median']):.1f}, eff. params {f(r0['dof_weighted_median']):.0f}, "
            f"R2 {f(r0['r2_test_median']):+.2f}")
    # L0 is the worst zero-anchored depth
    worse_all = [ds for ds in DS if all(gz(ds, 'zero', 0) > gz(ds, 'zero', d) for d in range(1, 19))]
    worse_19 = [ds for ds in DS if gz(ds, 'zero', 0) > gz(ds, 'zero', 19)]
    put("D.zero_L0_worse_than_every_L1_L18", len(worse_all))
    say(f"  zero: L0 worse than every L1..L18 on {len(worse_all)}/14 (exceptions "
        f"{[DISPLAY[ds] for ds in DS if ds not in worse_all]}); worse than L19 on {len(worse_19)}/14")
    for ds in ("boom_hourly", "wind_farms_hourly", "LOOP_SEATTLE_5T", "monash_electricity_hourly"):
        say(f"  {DISPLAY[ds]:12s} " + "  ".join(
            f"L{d}: zero {pct(gz(ds, 'zero', d))} id {pct(gz(ds, 'identity', d))} probe {pct(gz(ds, 'probe', d))}"
            for d in (1, 5, 10, 15, 19)))
    worse_probe = {d: [DISPLAY[ds] for ds in DS if f(GD[(ds, 'zero', d)]["mase_ratio_vs_probe_ci_lo"]) > 1]
                   for d in (1, 5, 10, 15, 19)}
    put("D.zero_worse_than_probe", worse_probe)
    say(f"  zero worse than probe (CI): {worse_probe}")
    # post hoc readings with the zero anchor
    for arm in ("identity", "zero"):
        rr = [GENT[(ds, arm)] for ds in DS]
        for eps in ("0.05", "0.2"):
            dep = [int(r[f"budget_eps_{eps}_depth"]) for r in rr]
            cut = [x for x in dep if x != 20]
            within = sum(r[f"budget_eps_{eps}_within"] == "True" and int(r[f"budget_eps_{eps}_depth"]) != 20
                         for r in rr)
            sp = [SPD.get(("timesfm3", x), (1.0, 1.0)) if x != 20 else (1.0, 1.0) for x in dep]
            put(f"D.budget.{arm}.{eps}", {"cut": len(cut), "within": within,
                                          "median_speedup_b1": med([s[0] for s in sp]),
                                          "median_speedup_b256": med([s[1] for s in sp])})
            say(f"  budget {arm:8s} eps {eps}: cut {len(cut)}/14, within on test {within}, median speedup "
                f"B=1 {med([s[0] for s in sp]):.2f}x B=256 {med([s[1] for s in sp]):.2f}x")
    # inside versus before the tunnel (zero anchor), the strata table
    strata = []
    ent = {ds: ENT[("timesfm3", ds)] for ds in DS}
    for d in range(20):
        ins = [gz(ds, 'zero', d) for ds in DS if d >= ent[ds]]
        bef = [gz(ds, 'zero', d) for ds in DS if d < ent[ds]]
        if len(ins) >= 2 and len(bef) >= 2:
            strata.append((d, len(ins), len(bef), med(ins) - 1, med(bef) - 1))
    put("D.strata_zero", strata)
    say(f"  strata (zero): cheaper inside at {sum(s[3] < s[4] for s in strata)}/{len(strata)} depths: " +
        "; ".join(f"L{d} {i}/{b} {100 * x:+.1f} vs {100 * y:+.1f}" for d, i, b, x, y in strata))

    # ================================================================== E. Phase 2c LODO
    say("=" * 100)
    say("E. Phase 2c: leave-one-dataset-out transfer (Chronos-2)")
    C = R / "results/three_model_phase2_lodo/combined"
    agg = rows(C / "lodo_aggregate.csv")
    for r in agg:
        if r["group"] not in ("all", "held_out"):
            continue
        key = f"E.{r['group']}.L{r['depth']}"
        put(key, {k: f(r[k]) for k in ("speedup_B1", "hard_median", "pooled_median", "specific_median",
                                       "pooled_within_5", "pooled_within_10", "specific_within_5",
                                       "pooled_improves_over_hard_ci", "pooled_minus_specific_pp_median",
                                       "pooled_gap_closure_median", "specific_gap_closure_median",
                                       "n_late_checkpoint", "n_grid_max_wd", "n", "hard_q1", "hard_q3",
                                       "pooled_q1", "pooled_q3", "specific_q1", "specific_q3")})
        say(f"  {r['group']:8s} L{r['depth']} ({f(r['speedup_B1']):.2f}x, n={r['n']}): hard {pct(f(r['hard_median']), False)} "
            f"pooled {pct(f(r['pooled_median']), False)} [{pct(f(r['pooled_q1']), False)}, {pct(f(r['pooled_q3']), False)}] "
            f"specific {pct(f(r['specific_median']), False)}; pooled<hard (CI) {r['pooled_improves_over_hard_ci']}; "
            f"within 5/10 {r['pooled_within_5']}/{r['pooled_within_10']}; pooled-specific {f(r['pooled_minus_specific_pp_median']):+.1f} pp; "
            f"gap closed {100 * f(r['pooled_gap_closure_median']):.0f}% (specific {100 * f(r['specific_gap_closure_median']):.0f}%); "
            f"late ckpt {r['n_late_checkpoint']}, grid max {r['n_grid_max_wd']}")
    pdx = [r for r in rows(C / "lodo_per_dataset.csv") if r["variant"] == "sibling_excluded"]
    for r in pdx:
        if r["dataset"] in ("LOOP_SEATTLE_5T", "m4_hourly") or r["held_out_from_backbone_pretraining"] == "True":
            say(f"     {r['display']:16s} L{r['depth']}: pooled {pct(f(r['pooled_dmase']), False)} hard "
                f"{pct(f(r['hard_dmase']), False)} specific {pct(f(r['specific_dmase']), False)}; "
                f"improves over hard (CI) {r['improves_over_hard_ci']}")
            put(f"E.per.{r['dataset']}.L{r['depth']}", {"pooled": f(r["pooled_dmase"]), "hard": f(r["hard_dmase"]),
                                                       "specific": f(r["specific_dmase"]),
                                                       "improves_ci": r["improves_over_hard_ci"] == "True"})
    not_improve = defaultdict(list)
    for r in pdx:
        if r["improves_over_hard_ci"] != "True":
            not_improve[r["depth"]].append(r["display"])
    put("E.not_improving_over_hard", dict(not_improve))
    say(f"  datasets where pooled does NOT beat the hard cut (CI): {dict(not_improve)}")
    sib = rows(C / "lodo_sibling_sensitivity.csv")
    if sib:
        say(f"  sibling sensitivity columns: {list(sib[0].keys())[:12]}")
        diffs = []
        for r in sib:
            for k, v in r.items():
                if "diff" in k and "pp" in k:
                    diffs.append(abs(f(v)))
        if diffs:
            put("E.sibling_max_abs_pp", max(diffs))
            say(f"  sibling included vs excluded: max |difference| {max(diffs):.2f} pp")

    # ================================================================== F. Phase 2f
    say("=" * 100)
    say("F. Phase 2f: number of source datasets (Chronos-2, L6)")
    S = R / "results/three_model_phase2f_source_scaling/combined"
    for r in rows(S / "source_scaling_aggregate.csv"):
        put(f"F.N{r['n_sources']}", {k: f(r[k]) for k in ("dmase_pct_mean", "dmase_pct_lo", "dmase_pct_hi",
                                                          "gap_to_specific_pp_mean", "n_negative_transfer_vs_hard_ci",
                                                          "n_within_5pct_of_full", "n_cells", "fit_minutes_mean",
                                                          "n_train_windows")})
        say(f"  N={r['n_sources']:>2}: mean {f(r['dmase_pct_mean']):+.1f}% [{f(r['dmase_pct_lo']):+.1f}, "
            f"{f(r['dmase_pct_hi']):+.1f}] (cells {r['n_cells']}); gap to specific {f(r['gap_to_specific_pp_mean']):+.1f} pp; "
            f"worse than hard (CI) {r['n_negative_transfer_vs_hard_ci']}; within 5% {r['n_within_5pct_of_full']}; "
            f"fit {f(r['fit_minutes_mean']):.2f} min on {r['n_train_windows']} windows")
    bt = rows(S / "source_scaling_by_target_n.csv")
    hard_mean = st.mean({r["target"]: f(r["hard_dmase_pct"]) for r in bt}.values())
    spec_mean = st.mean({r["target"]: f(r["specific_dmase_pct"]) for r in bt}.values())
    put("F.hard_mean", hard_mean)
    put("F.specific_mean", spec_mean)
    say(f"  hard cut mean over targets {hard_mean:+.1f}%; dataset-specific {spec_mean:+.1f}%")
    for r in bt:
        say(f"     {r['display']:20s} N={r['n_sources']:>2}: mean {f(r['dmase_pct_mean']):+.1f}% "
            f"[{f(r['dmase_pct_min']):+.1f}, {f(r['dmase_pct_max']):+.1f}] hard {f(r['hard_dmase_pct']):+.1f} "
            f"specific {f(r['specific_dmase_pct']):+.1f}")
        put(f"F.target.{r['target']}.N{r['n_sources']}", f(r["dmase_pct_mean"]))
    cells = rows(S / "source_scaling_cells.csv")
    n1 = [100 * f(r["pooled_dmase"]) for r in cells if r["n_sources"] == "1"]
    put("F.N1_range", [min(n1), max(n1)])
    n13 = [r for r in cells if r["n_sources"] == "13"]
    worse_spec = sum(f(r["pooled_minus_specific_pp_ci_lo"]) > 0 for r in n13)
    put("F.N13_worse_than_specific_ci", worse_spec)
    loop_worse_hard = [r for r in cells if r["dataset"] == "LOOP_SEATTLE_5T"]
    n_loop_worse = sum(f(r["pooled_over_hard_ci_lo"]) > 1 for r in loop_worse_hard)
    put("F.loop_worse_than_hard_cells", [n_loop_worse, len(loop_worse_hard)])
    say(f"  N=1 range over {len(n1)} cells: {min(n1):+.1f}% to {max(n1):+.1f}%; N=13 worse than specific (CI): "
        f"{worse_spec}/{len(n13)}; LOOP Seattle worse than hard cut (CI) in {n_loop_worse}/{len(loop_worse_hard)} cells")

    # ================================================================== G. Phase 2e
    say("=" * 100)
    say("G. Phase 2e: non-contiguous blocks (Chronos-2)")
    E = R / "results/three_model_phase2e_noncontig/tables"
    for r in rows(E / "summary_by_budget.csv"):
        k = f"G.k{r['budget']}.{r['strategy']}.{r['arm']}"
        put(k, {c: f(r[c]) for c in ("dmase_median", "dmase_q25", "dmase_q75", "n_within_5pct", "speedup_b1",
                                     "speedup_b256", "vs_prefix_median", "n_better_ci", "n_worse_ci",
                                     "n_invalid_v3", "dwql_median")})
        say(f"  k={r['budget']} {r['strategy']:10s} {r['arm']:4s} blocks {r['blocks']:>16s}: median "
            f"{f(r['dmase_median']):+.1f}% [{f(r['dmase_q25']):+.1f}, {f(r['dmase_q75']):+.1f}] within5 {r['n_within_5pct']}; "
            f"speedup B1 {f(r['speedup_b1']):.2f}x B256 {f(r['speedup_b256']):.2f}x; vs prefix median "
            f"{r['vs_prefix_median'] or '-'} better/worse CI {r['n_better_ci'] or '-'}/{r['n_worse_ci'] or '-'}; invalid_v3 {r['n_invalid_v3']}")
    for r in rows(E / "single_skip_summary.csv"):
        put(f"G.skip.{r['skipped_block']}", {"median": f(r["dmase_median"]), "worse_ci": int(r["n_worse_ci"]),
                                             "better_ci": int(r["n_better_ci"])})
        say(f"  skip block {r['skipped_block']:>2}: median {f(r['dmase_median']):+.2f}% worse/better (CI) "
            f"{r['n_worse_ci']}/{r['n_better_ci']}")

    # ================================================================== H. Chronos-2-small
    say("=" * 100)
    say("H. Chronos-2-small versus truncated Chronos-2")
    tr = rows(P2 / "h4_truncation.csv")
    sm = [r for r in tr if r["arm"] == "chronos2_small"]
    smr = [f(r["mase_ratio"]) for r in sm]
    q1, q3 = quart(smr)
    put("H.small", {"median": med(smr) - 1, "q1": q1 - 1, "q3": q3 - 1,
                    "within5": sum(x <= 1.05 for x in smr), "worst": max(smr) - 1,
                    "worst_dataset": DISPLAY[max(sm, key=lambda r: f(r['mase_ratio']))["dataset"]]})
    lat = rows(P2 / "latency_lookup.csv")
    sml = {(r["level"], r["batch"]): f(r["speedup_vs_native"]) for r in lat
           if r["model"] == "chronos2" and r["kind"] == "chronos2_small"}
    put("H.small_speedup", {f"{k[0]}_B{k[1]}": v for k, v in sml.items()})
    say(f"  Chronos-2-small: median {pct(med(smr))} IQR [{pct(q1)}, {pct(q3)}], within 5% {sum(x <= 1.05 for x in smr)}/14, "
        f"worst {pct(max(smr))} ({OUT['H.small']['worst_dataset']}); speedups {sml}")
    for d in (3, 4, 5, 6):
        dg = [ratio("chronos2", ds, "noa", d) for ds in DS]
        put(f"H.c2.L{d}", {"median": med(dg) - 1, "within5": sum(x <= 1.05 for x in dg),
                           "speedup_b1": SPD[("chronos2", d)][0], "speedup_b256": SPD[("chronos2", d)][1]})
        say(f"  Chronos-2 aligned L{d}: median {pct(med(dg))}, within 5% {sum(x <= 1.05 for x in dg)}/14, "
            f"speedup {SPD[('chronos2', d)][0]:.2f}x / {SPD[('chronos2', d)][1]:.2f}x")
    nat = {(r["level"], r["batch"]): f(r["median_ms"]) for r in lat if r["model"] == "chronos2" and r["kind"] == "native"}
    put("H.chronos2_native_ms", {f"{k[0]}_B{k[1]}": v for k, v in nat.items()})
    say(f"  Chronos-2 full model median ms: {nat}  -> one pass over 1,394 windows at B=256 ~ "
        f"{1394 / 256 * nat[('api_e2e', '256')] / 1000:.1f} s")

    # ================================================================== I. iterative adapters
    say("=" * 100)
    say("I. Chronos-2 / TiRex iterative adapters: validation / training distillation error at the "
        "selected checkpoint (the TimesFM-3 identity fit has 13-184 at L1-L17)")
    for m in ("chronos2", "tirex"):
        ratios, hard_sel = [], 0
        for ds in DS:
            fits = json.loads((R / "results/three_model_phase2/h3" / m / ds / "fits.json").read_text())["noa"]
            for lab, ft in fits.items():
                if "candidates" not in ft:          # the final block: nothing is fitted
                    continue
                if ft.get("selected_hard_cut"):
                    hard_sel += 1
                    continue
                cand = [c for c in ft["candidates"] if c["wd"] == ft["selected_wd"]][0]
                tr = dict((int(e), v) for e, v in cand["train_curve"])
                va = dict((int(e), v) for e, v in cand["val_curve"])
                ep = int(ft["selected_epoch"])
                if ep in tr and ep in va and tr[ep] > 0:
                    ratios.append(va[ep] / tr[ep])
        q1, q3 = quart(ratios)
        put(f"I.{m}.val_over_train", {"median": med(ratios), "q1": q1, "q3": q3, "max": max(ratios),
                                      "n": len(ratios), "hard_cut_selected": hard_sel})
        say(f"  {NAME[m]:9s} val/train median {med(ratios):.2f} IQR [{q1:.2f}, {q3:.2f}] max {max(ratios):.1f} "
            f"over {len(ratios)} fits; hard cut selected {hard_sel}")

    out = R / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "rebuttal_numbers.json").write_text(json.dumps(OUT, indent=1, default=str))
    (out / "rebuttal_numbers.txt").write_text("\n".join(LINES) + "\n")
    say(f"wrote {out}/rebuttal_numbers.json and .txt")


if __name__ == "__main__":
    main()
