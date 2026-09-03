# CKA_NATIVEHEAD_REPORT — CKA pre/post-LN + native-head readout brief

Branch `zibo/repr-metrics` @ `7a940d9`, tracked tree clean throughout (`git diff HEAD`
empty; no commits). Pre-run check: `results/repr_metrics/{cka,native_head}/` did **not**
exist — the killed previous attempt left no partial outputs; everything below is from
scratch. Prior-brief files (MORNING_REPORT.md etc.) ignored per preamble.

| task | verdict |
|---|---|
| Task 1 — CKA pre- vs post-LN | **DONE, all gates pass** |
| Task 2 — native-head readout | **STOP (interface + missing inputs — details below; no adapter improvised)** |
| Task 2b — OOD native-head | **NOT RUN** (conditional on Task 2's gate) |

---

## Task 1 — CKA pre- vs post-LN

**Estimator:** linear-kernel CKA with the unbiased (debiased) HSIC of Song et al. 2012,
float64 throughout; representations = dataset-level per-series mean content-patch
embeddings (n=200, seed 0 — the same matrices as `dataset_effrank`). Module:
`probing/repr_metrics_cka.py`; property tests `tests/test_repr_metrics_cka.py` (6: self=1,
symmetry, orthogonal + isotropic-scale invariance, independent→≈0, formula vs brute force).

**Files:** `results/repr_metrics/cka/{electricity,m4,electricity_randinit}/cka.json` +
`fig_cka_matrix.png` (left panel = 13-tap Amartya-comparable axis; right panel = 14 taps
with postln separated by a thin white line; randinit = single 13-tap panel).

### Gates (command output in transcript)

```
electricity          : CKA(X,X) ALL == 1.0 exactly (14 taps printed)   max asym 4.516e-13
m4                   : CKA(X,X) ALL == 1.0 exactly (14 taps printed)   max asym 2.386e-13
electricity_randinit : CKA(X,X) ALL == 1.0 exactly (13 taps printed)   max asym 1.161e-11
suite: 42 passed (36 existing + 6 new CKA tests)
```

All < 1e-10 asymmetry; every (i,j) computed independently, so symmetry is a genuine check.

### Seam numbers (the requested (i)/(ii)/(iii))

| run | (i) mean CKA(L12, L6..L11) | within-late ref (mean pairwise L6..L11) | (ii) mean CKA(postln, L6..L11) | CKA(postln, L12) |
|---|---:|---:|---:|---:|
| electricity | **0.5731** | 0.9035 | **0.8104** | 0.6708 |
| m4_hourly | **0.7810** | 0.9637 | **0.8301** | 0.8077 |
| electricity_randinit | 1.0000 | 1.0000 | — (no postln cache) | — |

**(iii) Does the postln row rejoin the late-layer block — numbers:**
electricity: L12's gap to the within-late reference is 0.9035 − 0.5731 = **0.3304**;
postln's gap is 0.9035 − 0.8104 = **0.0931** → postln closes **71.8%** of the seam.
m4: gaps 0.1827 → 0.1336 → postln closes **26.9%**. In both, postln sits closer to the
late block than to pre-norm L12 on electricity (0.8104 vs 0.6708), and about equidistant
on m4 (0.8301 vs 0.8077).

The 13-tap electricity panel visibly reproduces Amartya's isolated last-row seam (dark
L12 row/col against a bright L1..L11 block); the randinit matrix is uniformly 1.0000 —
the seam (and all inter-layer structure) is pretraining-induced, not architectural.

**Provenance recorded in every JSON:** Amartya's panels are electricity / **m4_daily** /
solar_1h vs our electricity / **m4_hourly** — electricity is the only directly comparable
panel; m4_daily was **not** extracted. Randinit has 13 taps (no postln cache exists).

---

## Task 2 — native-head readout: STOP (per the brief's own stop rule)

The brief's premise — "cached per-patch states → frozen LN → frozen head", "forward
passes through the HEAD only" — fails on three independently verified grounds. Per the
instruction ("If the head's input interface needs anything beyond (states -> LN -> head),
STOP and describe the interface — do NOT improvise an adapter"), here is the interface:

**1. The head does not consume content-patch states.** From the installed chronos2 source
(`Chronos2Model` forward path, printed in transcript):

```
hidden_states: torch.Tensor = encoder_outputs[0]
# slice the last num_output_patches hidden states to be input into the output_patch_embedding
forecast_embeds = hidden_states[:, -num_output_patches:]
quantile_preds = self.output_patch_embedding(forecast_embeds)
quantile_preds = rearrange(quantile_preds, "b n (q p) -> b q (n p)", ...)
```

The native quantile head (`output_patch_embedding`, 768 → num_quantiles×16) reads the
post-final-LN states at the **forecast-slot positions** (the masked_future tokens, K =
ceil(H/16) of them), then predictions live in **instance-normalized (arcsinh) space** and
are un-scaled via the same forward's `loc_scale` to produce real-unit forecasts/losses.
A faithful per-layer variant therefore needs, per layer k: the layer-k states **at the
forecast-slot positions**, the final LN, the head, **and** the matching `loc_scale` —
not content-patch states.

**2. The required inputs were never cached.** Every local cache stores content patches
only — `content = full[:, :n_content, :]` with "REG + masked_future excluded"
(`probing/repr_metrics.py:306`, same in the overnight extractor). No forecast-slot /
K-out per-layer cache exists anywhere locally (checked `features_cache/` and all
`results/**.npz`; the K-out extractions lived on the cluster, and the local bootstrap
npz hold per-window **losses**, not states).

**3. The cached windows have no held-out future.** Window rule is the LAST C=512 points
of each series/segment (`repr_metrics.py:173`, overnight `:50`) — there are no actual
future values after the window, so q9 loss / MASE are uncomputable on "our cached eval
windows" regardless of states.

**What a correct Task 2 requires:** new full-encoder passes on loss-bearing windows
(windows with H=64 held-out futures), run with K forecast slots, hooking each layer's
forecast-slot states plus `loc_scale`, then LN → head → un-scale → q9/MASE — i.e., a
small new extraction campaign (the existing `extract_kout_features` in
`probing/extraction.py` already implements exactly this interface for L12/final and could
be extended per-layer), not "head-only forwards" over existing caches. Stopped without
improvising; no `native_head/` outputs were produced. Time spent: ~15 min of source
verification, far under the 2 h box — CKA shipped as the guaranteed deliverable.

**Task 2b:** conditional on Task 2's self-consistency gate → not run. (Its OOD
extractions were not started; nothing to clean up.)

---

## Global gates

- `git diff --stat HEAD` — empty (no tracked file modified; no commits).
- New untracked from this brief only: `probing/repr_metrics_cka.py`,
  `tests/test_repr_metrics_cka.py`, `results/repr_metrics/cka/`, this report.
- All kernel math float64; suite **42 passed**.

## Decision needed

Task 2 as specced is blocked on inputs that don't exist locally. If you want it, the
follow-up brief should authorize: per-layer forecast-slot extraction on loss-bearing
windows (extend `extract_kout_features` to hook all 12 blocks + Embed), which also
unlocks Task 2b unchanged. Estimated: one extraction pass per dataset + head-only math.

---

# ADDENDUM — authorized follow-up: per-layer forecast-slot extraction + native-head readout

Authorization received for the extraction Task 2 was blocked on. Module:
`probing/repr_metrics_nativehead.py` (interface credit to
`probing.extraction.extract_kout_features` in provenance). Windows: ONE loss-bearing
window per series, ctx = seg[-(C+H):-H] (C=512), fut = seg[-H:] (H=64, K=4), longest-
finite-segment rule, <=200 series seed 0. ONE forward per batch captures all 14 fslot
taps + loc_scale + the model's own quantile_preds.

## Gate history (verbatim in transcript)

- **First run FAILED the self-consistent gate** (rel Δ ≈ 1.0) and STOPPED before any
  other layer, as specified. Diagnosis from source: `Chronos2Model.forward` UN-SCALES
  its returned predictions (`instance_norm.inverse(quantile_preds, loc_scale)` — the
  "# Unscale predictions" block) while our pathway stopped in normalized space. Fix =
  append the model's own final un-scale step to the pathway (verbatim, frozen,
  parameter-free — not an adapter); metrics recomputed in consistent spaces
  (q9 pinball over the 9 deciles in arcsinh space, both sides re-normalized with the
  same window loc_scale; MASE in real units, m=24 context denominator).
- **Gate then PASSED on all five datasets** before per-layer evaluation:

| run | n windows | rel Frobenius Δ | q9 rel Δ |
|---|---:|---:|---:|
| electricity | 200 | 1.505e-08 | 0.000e+00 |
| m4 | 200 | 1.760e-10 | 0.000e+00 |
| sg_carpark | 200 | 6.191e-09 | 0.000e+00 |
| coastal_ts | 48 | 0.000e+00 | 0.000e+00 |
| boom_hourly | 185 | 0.000e+00 | 0.000e+00 |

(BOOM restricted to the first 200 manifest entries by file order — the locally staged
subset and the distance-ladder convention; 185/200 windows survive the seg>=C+H rule.
The id_data all-356 loader hit a missing shard for entry #201+; loader switched to the
staged-subset rule, noted in provenance.)

## Results — shape + argmin, numbers only

q9 pinball loss (arcsinh space) per layer; native baseline = the model's own forward.

| run | shape | argmin | q9 range (min -> max) | max/min | native q9 | native MASE |
|---|---|---|---|---:|---:|---:|
| electricity | **STRUCTURED** | **L12** | 0.1011 -> 0.2922 | 2.89 | 0.1011 | 0.834 |
| m4 | **STRUCTURED** | **L12** | 0.0623 -> 0.2917 | 4.68 | 0.0623 | 0.962 |
| sg_carpark | **STRUCTURED** | **L12** | 0.0840 -> 0.2841 | 3.38 | 0.0840 | 0.714 |
| coastal_ts | **STRUCTURED** | **L12** | 0.1901 -> 0.3709 | 1.95 | 0.1901 | 1.191 |
| boom_hourly | **STRUCTURED** | **L12** | 0.1637 -> 0.2608 | 1.59 | 0.1637 | 2.127 |

Common trajectory (ID + sg_carpark): loss falls from Embed/L1 to a local minimum at
**L7-L8**, RISES through L9-L11, then drops to the global minimum at L12 (pre-norm L12
-> LN -> head = exactly the native output; L12_postln identical by construction).
Example (electricity): L8 = 0.1298, L11 = 0.1690, L12 = 0.1011. Coastal/BOOM show the
same fall-rise(-at-L11)-then-L12-minimum with smaller amplitude.

Files: `results/repr_metrics/native_head/{run}/native_head.json` +
`fig_native_head_by_layer.png` (native baseline dashed; existing probe accuracy in
faint grey on the ID panels) + `fig_ood_native_head_3panel.png`.

## Global gates (addendum)

- suite **42 passed**; `git diff --stat HEAD` empty; new untracked only
  `probing/repr_metrics_nativehead.py` + `results/repr_metrics/native_head/`.
- No fitting/calibration anywhere: LN, head, and un-scale are the frozen model's own
  modules invoked verbatim.
