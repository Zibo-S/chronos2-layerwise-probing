# VERIFY_prefreeze.md — pre-freeze verification pass

Branch `zibo/repr-metrics` @ `7a940d9`. Read-only pass (no commits, no branch switches,
`ood-forecasting-pilot` untouched, no PR). Every check below shows the actual command
output followed by a one-line PASS/FAIL.

Working-tree note (pre-existing, not touched): `git status --porcelain` reports two
deleted files, `results/distance/ladder/fig_distance_vs_gain_v2.png` and
`fig_distance_vs_gain_v3.png` (committed earlier, since deleted from the working tree).
No check below depends on them.

---

## CHECK 1 — Masarczyk criterion logic (af4c21e)

### 1(a) threshold formula, crossing test, scan direction

```
$ grep -n "target = t \* acc_final\|hits = \|start_i\|acc_final = \|PROBE_AXIS = " probing/repr_metrics_masarczyk.py
59:PROBE_AXIS = [f"L{k + 1}" for k in range(12)]
121:    acc_final = float(acc[11])                      # acc(L12) = probe index 11
125:        target = t * acc_final
126:        hits = [i for i in range(12) if acc[i] >= target]
128:        start_i = hits[0]
129:        start_label = PROBE_AXIS[start_i]
130:        region = PROBE_AXIS[start_i:]               # [tunnel_start .. L12]
```

- threshold `target = t * acc_final`, with `acc_final = acc[11]` = acc(L12) — **line 125/121**
- crossing test is `>=` — **line 126**
- scan is low-to-high: `range(12)` ascending, then `hits[0]` takes the FIRST — **lines 126/128**

**PASS** — threshold is `t * acc(final)`, test is `>=`, first-crossing scan runs low→high.

### 1(b) layer mapping applied exactly once

```
$ python  # recompute tunnel_start from stored accuracy_by_layer, independent of the module
--- electricity  (results/repr_metrics/masarczyk_criterion/electricity/criterion.json)
    probe_axis           = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12']
    accuracy_final_L12   = 0.388  | acc[axis[-1]]=0.388  equal=True
    len(axis)=12  mapping applied ONCE? axis[0]=='L1' and axis[11]=='L12': True
    t=0.95: target=0.368600 (stored 0.368600, match=True) recomputed_start=L1  stored=L1  MATCH=True
    t=0.98: target=0.380240 (stored 0.380240, match=True) recomputed_start=L2  stored=L2  MATCH=True

--- m4  (results/repr_metrics/masarczyk_criterion/m4/criterion.json)
    probe_axis           = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12']
    accuracy_final_L12   = 0.7049418604651163  | acc[axis[-1]]=0.7049418604651163  equal=True
    len(axis)=12  mapping applied ONCE? axis[0]=='L1' and axis[11]=='L12': True
    t=0.95: target=0.669695 (stored 0.669695, match=True) recomputed_start=L2  stored=L2  MATCH=True
    t=0.98: target=0.690843 (stored 0.690843, match=True) recomputed_start=L2  stored=L2  MATCH=True
```

Cross-check against the raw probe source (proves the shift is applied once, not zero or twice):

```
### mapping applied EXACTLY ONCE — cross-check vs the raw probe source ###
  raw probe array (idx 0..11) from results/phase0_trio/id_probing_summary.json:
    [0.378, 0.39867, 0.37867, 0.37733, 0.39133, 0.41267, 0.41333, 0.40867, 0.36467, 0.37867, 0.39933, 0.388]
  criterion accuracy_by_layer L1..L12:
    [0.378, 0.39867, 0.37867, 0.37733, 0.39133, 0.41267, 0.41333, 0.40867, 0.36467, 0.37867, 0.39933, 0.388]
  probe idx k  ==  repr L(k+1) elementwise: True   (shift applied ONCE, no double-shift)
  acc(L12)==0.388: True
```

Recomputed tunnel_start: electricity **L1 (95%) / L2 (98%)**, m4 **L2 (95%) / L2 (98%)** —
matches the expected L1/L2 and L2/L2.

**PASS** — mapping `probe layer_k -> repr L(k+1)` applied exactly once; tunnel_start
reproduces L1/L2 (95%) and L2/L2 (98%) for both datasets.

### 1(c) electricity L9 saturation caveat from the stored arrays

```
  accuracy_by_layer['L9'] = 0.36466666666666664
  thresholds['0.95'].target_accuracy = 0.3686
  L9 < target : 0.36466666666666664 < 0.3686 -> True   (rounded: 0.36467 < 0.3686)
  raw source idx 8 (=L9)  = 0.36466666666666664  identical to stored: True
```

**PASS** — elec L9 = 0.36467 < target 0.3686, taken straight from the stored arrays and
identical to the raw probe source.

### CHECK 1 verdict: **PASS** (a, b, c all pass)

---

## CHECK 2 — Ladder join integrity (28 cells)

### 2(a) join keys: every (source,target) matched; n=28, no NaN, no drops/dupes

```
  [extended_v3_rolling]
    n rows                     = 28   (==28: True)
    unmatched (source,target)  = []   (empty: True)
    duplicated cells           = 0   (==0: True)
    NaN in distance/gain       = 0
    tier counts                = {'near': 16, 'far': 12}
    cell set == 4x4 U 4x3      = True  (missing=[], extra=[])

  [extended_v2]
    n rows                     = 28   (==28: True)
    unmatched (source,target)  = []   (empty: True)
    duplicated cells           = 0   (==0: True)
    NaN in distance/gain       = 0
    tier counts                = {'near': 16, 'far': 12}
    cell set == 4x4 U 4x3      = True  (missing=[], extra=[])
```

**PASS** — both versions: 28 rows (16 near + 12 far), every key matched a distance entry,
zero NaN, zero duplicates, cell set exactly the 4x4 U 4x3 grid.

### 2(b) distance matrix sanity

```
  datasets order = ['m4_hourly', 'monash_electricity_hourly', 'uber_tlc_hourly', 'wind_farms_hourly', 'sg_carpark', 'coastal_ts', 'boom_hourly']
  diagonal       = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  diagonal all == 0 exactly : True
  symmetric (allclose atol=0): True  -> SYMMETRIC, not directional
  max|M - M.T| = 0.000e+00
  d(m4_hourly, sg_carpark) = 0.512459  (2dp: 0.51)
  d(m4_hourly, monash_electricity_hourly) = 1.409768  (2dp: 1.41)
```

Note for the paper: the matrix is **symmetric** (energy distance is a metric-like symmetric
divergence), so "source -> target" in the join is a lookup into a symmetric matrix, not a
directional quantity.

**PASS** — diagonal exactly 0, exactly symmetric, d(m4,sg_carpark)=0.51 and
d(m4,electricity)=1.41 as expected.

### 2(c) Spearman recomputed from the joined tables

```
  [extended_v3_rolling]
    all cells      n=28  rho=-0.4106   expected ~-0.411  match2dp=True
    excl m4 source n=21  rho=-0.2297   expected ~-0.23  sign=-  |d|=0.0003

  [extended_v2]
    all cells      n=28  rho=-0.1026   expected ~-0.103  match2dp=True
    excl m4 source n=21  rho=+0.0911   expected ~+0.09  sign=+  |d|=0.0011
```

**PASS** — all four reproduce: v3 all -0.411, v3 excl-M4 -0.23, v2 all -0.103, and the v2
excl-M4 **sign flip to +0.09** is confirmed.

### 2(d) Wind Farms NaN rule

```
{
 "rule": "per series, take the LONGEST CONTIGUOUS FINITE segment; keep the series iff that segment has >= min_segment_points. Segments are NEVER concatenated across NaN gaps — splicing would corrupt the autocorrelation-family catch22 features.",
 "min_segment_points": 512,
 "applied_to": "all 7 datasets uniformly, before sampling and feature extraction",
 "supersedes": "the prototype's drop-series-if-any-NaN filter, which discarded 322/337 wind_farms_hourly series (5-7 usable) and caused a seed-stability failure (Spearman 0.9117 < 0.95)"
}

  per_dataset (all 7):
                     dataset  n_raw  n_kept  dropped   med_seg  min_seg
                   m4_hourly    414     414        0      1008      748
   monash_electricity_hourly    321     321        0     26304    26304
             uber_tlc_hourly    262     262        0      4344     4344
           wind_farms_hourly    337     337        0      5259     1715
                  sg_carpark    354     354        0      3623     3623
                  coastal_ts     48      48        0      5224     2733
                 boom_hourly    200     200        0      5230      732

  wind_farms: 337/337 kept  -> 337/337: True
  threshold >= 512: True  | min kept segment = 1715 (>=512: True)
  catch22-usable wind series: seed0=191, seed1=193
```

**PASS** — rule is longest-contiguous-finite-segment with threshold 512, applied to all 7
datasets uniformly, wind_farms 337/337 kept.

### CHECK 2 verdict: **PASS** (a, b, c, d all pass)

---

## CHECK 3 — Subsampling machinery (30667a2)

### 3(a) m=126, B=5000, WITHOUT replacement, paired/shared indices

```
$ grep -n "SUB_FRACTION\|DEFAULT_B\|_subsample_indices\|argsort(rng.random\|sub_idx = \|boot_idx = " probing/repr_metrics_bootstrap.py
26:PAIRED DRAWS: one index array per scheme per run, reused for every layer and metric, so
70:DEFAULT_B = 5000
71:SUB_FRACTION = 0.632                 # -> m = 126 of 200, matching the diagnostic
136:def _subsample_indices(n: int, m: int, B: int, seed: int) -> np.ndarray:
139:    return np.argsort(rng.random((B, n)), axis=1)[:, :m]
205:    m_sub = int(np.floor(SUB_FRACTION * n_series))
212:    boot_idx = np.random.default_rng(SEED).integers(0, n_series, size=(B, n_series))
213:    sub_idx = _subsample_indices(n_series, m_sub, B, SEED)
```

`argsort` of a random matrix then slice `[:, :m]` (line 139) = a random permutation
truncated to m -> sampling WITHOUT replacement by construction. Runtime asserts:

```
  SUB_FRACTION=0.632 -> m=floor(0.632*200)=126   (==126: True)
  DEFAULT_B=5000   (==5000: True)
  sub_idx.shape=(5000, 126)  (== (5000,126): True)
  WITHOUT replacement -> every row all-distinct: min_unique=126 max=126 all==m: True
  index sha256[:16] = d2cc2a218af8cab5

### paired/shared across layers: the SAME array object is reused (no per-layer redraw) ###
  lines inside the per-layer loop that draw NEW indices: []   (empty => shared)
  loop references sub_idx: True   boot_idx: True

### stored hashes identical across runs (same seed => same paired draws) ###
             electricity: sub m=126 sha=d2cc2a218af8cab5 | boot m=200 sha=c8086644b0c09f34
                      m4: sub m=126 sha=d2cc2a218af8cab5 | boot m=200 sha=c8086644b0c09f34
    electricity_randinit: sub m=126 sha=d2cc2a218af8cab5 | boot m=200 sha=c8086644b0c09f34
  runtime-regenerated sub sha == stored sha: True
```

**PASS** — m=126, B=5000, without replacement (all 5000 rows have 126 distinct indices),
index array drawn once per scheme and reused for every layer (no redraw inside the loop);
the runtime-regenerated hash matches the hash stored in all three run JSONs.

### 3(b) dip CI and argmax fractions re-derived from stored outputs

```
  [electricity] dip = ER(L11) - ER(L12_postln)   metric=dataset_effrank
    scheme      = subsampling m=126, no replacement (shared offset cancels)
    sub_mean    = +9.4681   point_full_n=+17.1802
    95% CI      = [+8.9030, +10.0364]   expected ~[+8.9,+10.0]  match1dp=True
    excludes_0  = True
    entropy argmax : mode=L6 frac_in_['L6', 'L7']=1.0000 mode_frac=0.9784  dist={'L6': 0.9784, 'L7': 0.0216}
    effrank argmax : mode=L11 frac_in_['L10', 'L11']=1.0000 mode_frac=1.0000  dist={'L11': 1.0}

  [m4] dip = ER(L11) - ER(L12_postln)   metric=dataset_effrank
    scheme      = subsampling m=126, no replacement (shared offset cancels)
    sub_mean    = +7.4099   point_full_n=+12.6114
    95% CI      = [+6.5369, +8.2608]   expected ~[+6.5,+8.3]  match1dp=True
    excludes_0  = True
    entropy argmax : mode=L6 frac_in_['L6', 'L7']=1.0000 mode_frac=1.0000  dist={'L6': 1.0}
    effrank argmax : mode=L11 frac_in_['L10', 'L11']=1.0000 mode_frac=1.0000  dist={'L11': 1.0}
```

**PASS** — dip CI [+8.90,+10.04] elec / [+6.54,+8.26] m4, both excluding 0.

> **Precision note for §4/§5 wording (not a failure).** "entropy argmax L6 ... at fraction
> 1.00" is exact for m4 (L6 = 1.0000) and for both effrank argmaxes (L11 = 1.0000), but for
> **electricity entropy the 1.00 is the BAND fraction** {L6,L7}; the single-layer mode
> fraction is **0.9784** (L6), with L7 taking 0.0216. If the paper states "argmax L6 in 100%
> of resamples" for electricity, it should instead say "argmax in {L6,L7} in 100%, L6 alone
> in 97.8%".

### 3(c) identity resample exact; m=199 within 0.4%

```
### stored m=199 near-full subsample gate ###
  [electricity] m=199  worst layer=L11  signed_dev=-0.3828%  |dev|<0.4%: True
  [m4]          m=199  worst layer=L11  signed_dev=-0.3515%  |dev|<0.4%: True
  [electricity_randinit] m=199 worst layer=L12 signed_dev=-0.0293%  |dev|<0.4%: True
```

```
### IDENTITY resample (idx=arange(n)) must reproduce point estimates EXACTLY ###
  [electricity]
          Embed: identity_ER=17.3942389505  point_ER=17.3942389505  stored=17.3942389505  |d_identity|=4.66e-12  |d_stored|=0.00e+00
             L6: identity_ER=62.5997113803  point_ER=62.5997113803  stored=62.5997113803  |d_identity|=1.71e-13  |d_stored|=0.00e+00
            L11: identity_ER=86.3004918328  point_ER=86.3004918328  stored=86.3004918328  |d_identity|=1.42e-13  |d_stored|=0.00e+00
            L12: identity_ER=23.0095925276  point_ER=23.0095925276  stored=23.0095925276  |d_identity|=0.00e+00  |d_stored|=0.00e+00
     L12_postln: identity_ER=69.1203385981  point_ER=69.1203385981  stored=69.1203385981  |d_identity|=1.28e-13  |d_stored|=0.00e+00
  [m4]
            L11: identity_ER=65.9373998006  point_ER=65.9373998006  stored=65.9373998006  |d_identity|=4.69e-13  |d_stored|=0.00e+00
     L12_postln: identity_ER=53.3260023221  point_ER=53.3260023221  stored=53.3260023221  |d_identity|=7.32e-13  |d_stored|=0.00e+00
```

**PASS** — identity resample reproduces the point estimates to <=5e-12 (float round-off) and
the stored point estimates match the recomputed ones exactly (0.00e+00); m=199 worst
deviation is -0.383% (elec) / -0.352% (m4) / -0.029% (randinit), all within 0.4%.

### CHECK 3 verdict: **PASS** (a, b, c all pass; one wording precision note in 3(b))

---

## CHECK 4 — the 44.2 / 35.5 / 0.010 number (randinit flatness)

### Location

```
$ grep -rn "effrank_range_across_layers" results/repr_metrics/
results/repr_metrics/bootstrap/m4/bootstrap.json:713:  "effrank_range_across_layers": {
results/repr_metrics/bootstrap/electricity_randinit/bootstrap.json:661:  "effrank_range_across_layers": {
results/repr_metrics/bootstrap/electricity/bootstrap.json:714:  "effrank_range_across_layers": {
```

Field: **`derived.effrank_range_across_layers.sub_mean`**. Stored blocks verbatim:

```
--- electricity            --- m4                     --- electricity_randinit
 "metric": "max-min dataset_effrank over Embed..L12"   (identical in all three)
 "scheme": "subsampling m=126, no replacement"         (identical in all three)
 point_full_n: 68.90625288229812   52.76761571501371   0.012018536974840899
 sub_mean    : 44.24894267921293   35.50245423013162   0.010079446335042585
 lo          : 43.2292424141763    33.45253492655043   0.00934510912269162
 hi          : 45.27077818805858   37.498630590663396  0.010740871687574314
```

### Exact definition (from the producing code)

```
$ grep -n "effrank_range_across_layers" -B14 probing/repr_metrics_bootstrap.py
312-    flat = er_mat.max(axis=1) - er_mat.min(axis=1)
313-    lo, hi = _pct(flat)
314:    derived["effrank_range_across_layers"] = {
315-        "metric": f"max-min dataset_effrank over {depth[0]}..{depth[-1]}",
316-        "scheme": f"subsampling m={m_sub}, no replacement",
317-        "point_full_n": float(max(point[ln]["dataset_effrank"] for ln in depth)
318-                              - min(point[ln]["dataset_effrank"] for ln in depth)),
319-        "sub_mean": float(flat.mean()), "lo": lo, "hi": hi}
```

`er_mat` is `(B=5000, 13)` of **subsampled** `dataset_effrank` draws; `flat` is the
**within-draw** range, and `sub_mean` is its mean over the 5000 draws.

- **Layers included:** `Embed .. L12` = **13 layers**. **Embed IS included.** **L12 is the
  PRE-final-norm block output.** **`L12_postln` is EXCLUDED** (confirmed at runtime:
  `L12_postln in depth = False`; its ER is 69.12 elec / 53.33 m4).
- **Statistic:** **subsampled** — mean over B=5000 draws of the within-draw max-min at
  **m=126 without replacement**. It is *not* the full-sample value.

### Recomputation from the per-layer arrays (to 1 dp)

```
  [electricity]  layers used = Embed..L12 (13), L12_postln in list: False
    metrics.json dataset_effrank: max=86.3005 @L11   min=17.3942 @Embed
    NAIVE full-sample max-min          = 68.9   (stored point_full_n=68.9)
    STORED sub_mean (the 44.2)         = 44.2   scheme='subsampling m=126, no replacement'
    ==> differ by 24.7  (-35.8% vs naive)

  [m4]           max=65.9374 @L11   min=13.1698 @Embed
    NAIVE full-sample max-min = 52.8   STORED sub_mean = 35.5   ==> differ by 17.3 (-32.7%)

  [electricity_randinit]  max=3.5278 @L12   min=3.5158 @Embed
    NAIVE full-sample max-min = 0.012  STORED sub_mean = 0.010  ==> differ by 0.002 (-16.1%)
```

Float-precision footnote (benign): recomputing `point_full_n` from `metrics.json` differs
from the bootstrap's own point estimates by ~3e-9..2e-7 because `metrics.json` accumulates
the per-series patch mean in float32 while the bootstrap path casts to float64 first. Identical
to 1 dp; not a definitional difference.

```
    [electricity] from metrics.json: 68.90625287912209 | stored: 68.90625288229812 | |d|=3.2e-09
    per-layer max @L11: metrics=86.30049172904893 bootstrap=86.30049183280967 |d|=1.0e-07
```

### **The definition DOES differ from a naive max-min — explicit statement for App C**

`44.2` (elec) / `35.5` (m4) / `0.010` (randinit) are **NOT** the naive max-min over the
`metrics.json` `dataset_effrank` column. The naive values are **68.9 / 52.8 / 0.012**.
The quoted numbers are the mean within-draw range under **m=126 subsampling without
replacement**, which is depressed relative to full-n by the documented distinct-unit effect
(-35.8% / -32.7% / -16.1%).

Both sides of the headline comparison (**44.2 vs 0.010**) use the **same** field, metric
string, layer axis and subsampling index (`sha256[:16]=d2cc2a218af8cab5`), so the comparison
is internally consistent. The risk is **mixing conventions across the paper**: if §4/§5 also
quotes `86.3 - 17.4 = 68.9` as the electricity rank span, that is the *naive full-n* number
while `44.2` is the *subsampled* one. App C needs the half-sentence, or switch the flatness
comparison to the `point_full_n` pair (**68.9 / 52.8 vs 0.012**).

### CHECK 4 verdict: **PASS (located and reproduced) — with a REQUIRED paper action**
Definition confirmed and recomputed; flagged that it differs from the naive max-min.

---

## CHECK 5 — Rank estimator spec (for the Egor reconciliation)

Written to **`RANK_ESTIMATOR_SPEC.md`** (10-line spec + code line references + runtime
shape confirmation). Source evidence:

```
$ grep -n "def effective_rank" -A14 probing/repr_metrics.py
114:def effective_rank(Z: np.ndarray) -> float:
120-    Z = np.asarray(Z, dtype=np.float64)
121-    s = np.linalg.svd(Z, compute_uv=False)
124-    s = s[s >= EIG_GUARD * s.max()]
125-    q = s / s.sum()
126-    return float(np.exp(-(q * np.log(q)).sum()))

$ grep -n "def compute_metrics" -A22 probing/repr_metrics.py
331-        s1 = np.array([matrix_entropy(Z) for Z in mats])
333-        er = np.array([effective_rank(Z) for Z in mats])
334-        # dataset-level: rows = per-series mean over content patches
335-        Zd = np.stack([Z.mean(axis=0) for Z in mats])         # (n_series, 768)
340-            "prompt_effrank_mean": float(er.mean()), ...
343-            "dataset_effrank": effective_rank(Zd),

$ grep -n "C = 512\|MAX_SERIES\|EIG_GUARD\|n_content = math.ceil\|content = full" probing/repr_metrics.py
74:C = 512
75:MAX_SERIES = 200
77:EIG_GUARD = 1e-12
261:    n_content = math.ceil(C / patch_size)
306:            content = full[:, :n_content, :]                  # REG + masked_future excluded
```

```
### runtime shape confirmation ###
  n_series=200  per-series matrix shape=(32, 768)  dtype=float32
  dataset-level matrix Zd shape=(200, 768)
  metrics.json n_series=200  D=768  n_patches[0]=32
  provenance: C=512 seed=0 n_content_patches=32 patch_size=16
```

Key facts captured in the spec: SVD is on the **raw matrix** (no centering, no
standardization, no covariance); normalized **singular values** (not eigenvalues);
`dataset_effrank` = `effective_rank` of the `(200 x 768)` stack of per-series
content-patch means (ceiling 200); `prompt_effrank` = mean over 200 series of
`effective_rank` of each `(32 x 768)` per-series matrix (ceiling 32).

### CHECK 5 verdict: **PASS** — spec written to `RANK_ESTIMATOR_SPEC.md`

---

## Summary

| check | verdict | note |
|---|---|---|
| 1 — Masarczyk criterion logic | **PASS** | threshold `t*acc(L12)`, `>=`, low->high; mapping applied exactly once; L1/L2 and L2/L2 reproduced; elec L9 caveat confirmed |
| 2 — Ladder join integrity | **PASS** | 28 cells both versions, no NaN/dupes/unmatched; matrix diag 0 + exactly symmetric; all 4 Spearman values reproduced incl. the v2 excl-M4 sign flip; wind 337/337 under seg>=512 |
| 3 — Subsampling machinery | **PASS** | m=126, B=5000, without replacement, paired shared indices (hash matches stored); dip CIs reproduce; identity resample exact; m=199 within 0.4% |
| 4 — 44.2 / 35.5 / 0.010 | **PASS (located + reproduced)** | **requires a paper action** — see below |
| 5 — Rank estimator spec | **PASS** | `RANK_ESTIMATOR_SPEC.md` written |

### Actions required before the freeze

1. **App C half-sentence (CHECK 4).** `44.2 / 35.5 / 0.010` is the **subsampled** (m=126,
   no replacement) mean within-draw max-min over `Embed..L12`, **not** the naive full-sample
   max-min, which is `68.9 / 52.8 / 0.012`. Both sides of the 44.2-vs-0.010 comparison use
   the same convention, so the comparison stands; the risk is quoting `86.3 - 17.4 = 68.9`
   elsewhere in the same section under the same name.
2. **Wording precision (CHECK 3b).** For electricity, entropy-argmax "fraction 1.00" is the
   **band** {L6,L7}; the single-layer mode fraction is **0.9784**. m4 entropy and both
   effrank argmaxes are exactly 1.0000.
3. **Optional (CHECK 2b).** The distance matrix is **symmetric**, so "source -> target
   distance" is a symmetric lookup, not a directional quantity — worth one clause if the
   text implies directionality.

No check FAILED. Nothing was fixed, committed, or branch-switched during this pass.
