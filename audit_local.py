import json, numpy as np

J = json.load(open("perdataset_summary.json"))
D = J["datasets"]
BAND = list(range(3,9)); LAST = 11
def band(a): return float(np.mean([a[i] for i in BAND]))
P=[]  # collect (level, item, msg)
def log(level,item,msg): P.append((level,item,msg)); print(f"[{level:4}] {item}: {msg}")

ns = [k for k,v in D.items() if v and not v["saturated"]]   # non-saturated
sat= [k for k,v in D.items() if v and v["saturated"]]
print(f"non-saturated: {ns}\nsaturated: {sat}\n"+"="*70)

# --- C1: late_drop significant count among non-saturated ---
sig=[k for k in ns if D[k]["id_late_drop_band"]["excludes_0"]]
log("PASS" if len(sig)==5 else "FAIL","C1 late_drop count",
    f"{len(sig)}/6 significant -> {sig}")
for k in ns:
    ld=D[k]["id_late_drop_band"]
    print(f"      {k:<22} {ld['point']:+.4f} [{ld['lo']:+.4f},{ld['hi']:+.4f}] excl0={ld['excludes_0']}")

# --- C2: LSST UNROUNDED lower bound ---
lo=D["LSST"]["id_late_drop_band"]["lo"]
log("FLAG" if lo<=0 else "PASS","C2 LSST lo",
    f"raw lower bound = {lo!r}  -> {'INCLUDES 0, soften claim' if lo<=0 else 'strictly >0, ok'}")

# --- C3: SCP2 near chance every layer ---
a=D["SelfRegulationSCP2"]["per_layer_accuracy"]["ID"]; ch=D["SelfRegulationSCP2"]["chance"]
near = max(abs(x-ch) for x in a)
log("PASS" if near<0.06 else "FLAG","C3 SCP2 null",
    f"max |acc-chance|={near:.3f} (chance={ch}); per-layer={[round(x,3) for x in a]}")

# --- C4: argmax in middle band for the 5 headroom datasets ---
for k in sig:
    am=D[k]["argmax_layer"]
    log("PASS" if 3<=am<=7 else "FLAG",f"C4 argmax {k}",f"L{am} {'(in band)' if 3<=am<=7 else '(OUT of band)'}")

# --- C5: amplification significant cells (18 non-saturated) ---
pos=[];neg=[]
for k in ns:
    for s in ["gauss","timewarp","drift"]:
        amp=D[k]["amplification"][s]
        if amp["excludes_0"]:
            (pos if amp["point"]>0 else neg).append(f"{k}/{s} ({amp['point']:+.3f})")
log("PASS" if len(pos)==3 else "FAIL","C5 amp positive",f"{len(pos)} -> {pos}")
log("PASS" if len(neg)==2 else "FAIL","C5 amp negative",f"{len(neg)} -> {neg}")
print(f"      (13 expected null; total significant = {len(pos)+len(neg)}/18)")

# --- C6: figure-prose claims vs raw arrays ---
def pa(k,s): return D[k]["per_layer_accuracy"][s]
# (a) UWave & SCP1 gauss early-layer crater
for k in ["UWaveGestureLibrary","SelfRegulationSCP1"]:
    g=pa(k,"gauss"); bm=band(g)
    early_drop = bm - min(g[0],g[1])
    log("PASS" if early_drop>0.10 else "FLAG",f"C6a gauss crater {k}",
        f"L0={g[0]:.3f} L1={g[1]:.3f} vs band={bm:.3f}  (early dip {early_drop:.3f})")
# (b) SCP1 gauss L11 collapse
g=pa("SelfRegulationSCP1","gauss"); bm=band(g)
log("PASS" if (bm-g[LAST])>0.08 else "FLAG","C6b SCP1 gauss L11",
    f"L11={g[LAST]:.3f} vs band={bm:.3f} (drop {bm-g[LAST]:.3f})")
# (c) Handwriting drift peels MORE at late layers
ID=pa("Handwriting","ID"); dr=pa("Handwriting","drift")
gap=[ID[i]-dr[i] for i in range(12)]
mid_gap=float(np.mean(gap[3:9])); late_gap=float(np.mean(gap[9:12]))
log("PASS" if late_gap>mid_gap else "FLAG","C6c Handwriting drift peel",
    f"mid-gap={mid_gap:.3f} late-gap={late_gap:.3f} -> {'peels late' if late_gap>mid_gap else 'no peel'}")
# (d) timewarp roughly parallel (low gap variance) on headroom datasets
for k in ["UWaveGestureLibrary","SelfRegulationSCP1","Handwriting","LSST"]:
    ID=pa(k,"ID"); tw=pa(k,"timewarp"); gap=[ID[i]-tw[i] for i in range(12)]
    m=float(np.mean(gap)); sd=float(np.std(gap))
    log("PASS" if (m>0 and sd<0.04) else "FLAG",f"C6d timewarp parallel {k}",
        f"mean gap={m:+.3f}, std={sd:.3f} -> {'parallel' if sd<0.04 else 'NOT parallel'}")

# --- D: cross-run consistency vs earlier forest/hardening point estimates ---
EARLIER={"UWaveGestureLibrary":0.085,"EthanolConcentration":0.070,"SelfRegulationSCP1":0.063,
         "Handwriting":0.050,"LSST":0.015,"SelfRegulationSCP2":-0.018}
for k,v in EARLIER.items():
    now=D[k]["id_late_drop_band"]["point"]; diff=abs(now-v)
    log("PASS" if diff<0.01 else "FLAG",f"D cross-run {k}",f"perdataset={now:+.3f} vs earlier={v:+.3f} (|diff|={diff:.3f})")

# --- E: multiple-comparison sanity ---
log("FLAG","E multiple-comp",
    f"18 amp cells, {len(pos)+len(neg)} significant ({len(pos)}+/{len(neg)}-); "
    f"~{18*0.05:.1f} expected by chance two-sided -> scattered = consistent with NULL")

print("\n"+"="*70)
fails=[x for x in P if x[0]=="FAIL"]; flags=[x for x in P if x[0]=="FLAG"]
print(f"VERDICT: {len(fails)} FAIL, {len(flags)} FLAG")
print("FAILs:", [f"{i}: {m}" for _,i,m in fails] or "none")
print("Key FLAGs to eyeball:", [i for _,i,_ in flags])