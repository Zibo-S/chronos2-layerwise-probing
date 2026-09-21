# TimesFM-3: last-context-token probing (headline) — status 2026-09-15

## The question

> Does the TimesFM-3 last-context representation enter a forecasting tunnel **before** the final
> layer, when forecasting ability is measured with the model's full native Q=9 probabilistic
> objective?

Each model is probed where its own head reads: Chronos-2 at its K=4 forecast slots, TimesFM-3 at
its ONE last real context token (index 15 for C=512, H=64). No artificial shared scheme.

## Two TimesFM-3 lines (both kept)

| | HEADLINE (new) | ABLATION (kept intact) |
|---|---|---|
| entry point | `experiments/run_timesfm3_last_token_probing.py` | `experiments/run_timesfm3_probing.py` |
| readout | token 15 of ONE full-context pass | 16 independent causal prefixes, token j−1 |
| probe | independent `Linear(1280, 64*9)` per layer | one shared `Linear(1280, 64)` across origins |
| quantiles | all 9 native (Q=9) | median only (Q=1) |
| cache tag | `tfm3-last-token-q9-fp32-v1`, `(N,1280)` **float32** | `tfm3-prefix-v1`, `(N,16,1280)` |
| job | `job_timesfm3_last_token_q9.sh` | `job_timesfm3_probing.sh` |
| cost | 1 forward pass per batch | 16 forward passes per batch |

The two caches/results namespaces are disjoint and the loader **refuses** the other line's cache
(metadata + array-rank check, test 21). No Chronos-2 file is touched by either.

## Dataset suite (corrected 2026-09-16): the Chronos-2 paper SEVEN

`--suite paper7` (the default) probes the same seven datasets, on the same windows, as the
Chronos-2 run, taken from the Chronos-2 sources and never re-declared:

| paper name | internal key | kind | windows |
|---|---|---|---|
| M4 | `m4_hourly` | PT-ID | `build_windows` under `extended_v3_rolling` |
| Electricity | `monash_electricity_hourly` | PT-ID | same |
| Uber TLC | `uber_tlc_hourly` | PT-ID | same |
| Wind Farms | `wind_farms_hourly` | PT-ID | same |
| SG Carpark | `sg_carpark` | PT-OOD | `build_ood_rolling_windows(tag, C=512, H=64, seed=0)` |
| Coastal T-S | `coastal_ts` | PT-OOD | same |
| BOOM | `boom_hourly` | PT-OOD | same |

Roster from `probing/tunnel.py` (`PT_ID_TAGS` + `PT_OOD_TAGS`), display names from
`experiments/make_id_paper_figures.py:314`, window dispatch copied from
`run_native_head_adapter._windows()`. Expected counts (train/val/test): 1394/262/262 for the four
PT-ID, 1394/354/354 for SG Carpark and BOOM, 1394/48/48 (24 stations) for Coastal T-S.

Every run ABORTS unless the window counts **and** the per-window test series/cluster ids match
`results/ext_v5_native_head_adapter/{configs,bootstrap_inputs}` element-wise
(`--allow-window-mismatch` downgrades that to a report). `--audit-only` runs just that check with
no GPU and no model.

Validation is the datasets' own dedicated rolling split (full-train fit, wd chosen on val, **no
refit** — `probes.fit_quantile_probe_explicit_val`'s contract), not the 80/20 carve. KDD Cup 2018
and Pedestrian Counts are no longer in the headline default; they remain reachable via
`--suite extended_v1` for implementation checks.

## Files (all new)

- `probing/timesfm3_last_token.py` — geometry, full-context preprocessing/targets, single-pass
  extraction, cache, the L20 all-quantile native-head check, sorting discovery.
- `probing/timesfm3_last_token_probes.py` — the Q=9 probe, generic pinball loss, layerwise fit,
  native baseline, tunnel entrance.
- `experiments/run_timesfm3_last_token_probing.py` — driver (metrics, bootstrap, figures).
- `tests/test_timesfm3_last_token_probe.py` — 21 numbered contracts.
- `job_timesfm3_last_token_q9.sh` — SLURM.

## Run order (Narval)

1. login node: `python -m tests.test_timesfm3_last_token_probe` (~4 s, 2 threads).
1b. window audit (compute node, no GPU/model):
   `sbatch --gres=gpu:0 --time=0:40:00 -J tfm3lt-audit job_timesfm3_last_token_q9.sh --audit-only --device cpu`
2. GPU smoke: `sbatch --time=0:30:00 -J tfm3lt-smoke job_timesfm3_last_token_q9.sh
   --cache-dir $SCRATCH/timesfm3_last_token/smoke_cache
   --out-root $SCRATCH/timesfm3_last_token/smoke_results
   --datasets monash_electricity_hourly --layers 0 10 20`
   (also run `python -m tests.test_timesfm3_last_token_probe --with-model` inside an salloc).
3. full: `sbatch job_timesfm3_last_token_q9.sh` (4 datasets × 21 points).

## Verified so far (no GPU needed)

- All 10 model-free contract groups pass (3.9 s): geometry 512/16/15, probe `Linear(1280,576)`,
  horizon-major reshape, quantile order, pinball vs hand-computed + `mean_pinball_loss` +
  `chronos2_quantile_loss/(2Q)`, context-only preprocessing, target round-trip 3.9e-09,
  5%/2% tunnel rule, end-to-end fit, cache isolation.
- Full driver dry run with the backbone mocked: targets → 21 probes → MASE → native baseline →
  cluster bootstrap → tunnel → summary JSON → bootstrap npz → 5 figures, all green (now with
  rolling-shaped windows, a dedicated val split and the PT-ID/PT-OOD panels).
- Tests 22/23/24: the paper7 roster equals `PT_ID_TAGS + PT_OOD_TAGS` and its display names equal
  the paper figures'; the parity checker reads all 7 committed Chronos-2 artifacts and rejects a
  shuffled/short/miscounted window set; the explicit-val protocol fits on full train with no refit.
- Cache dtype is now **float32 by default** (tag `tfm3-last-token-q9-fp32-v1`, ~2 GB across the
  four datasets); float16 is opt-in via `--feature-dtype float16`. The bumped tag means an old
  float16 cache can never be silently reused. The 1e-2 dtype bar is UNCHANGED: float32 clears it
  bit-exactly (cache identity 0.0 of the layer std), and float16's measured ~1.2e-2 cost is
  exactly why it is no longer the default.

## Weight-decay grid + null baseline (2026-09-16)

**Selection grid** `WD_GRID_LAST_TOKEN` = `probes.WD_GRID_V2` extended by 10/30 → 10 candidates
`1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3 10 30` (the Chronos-2 grid itself is untouched). The probe is
737k parameters on 1394 train rows, so the optimum can sit above the old ceiling of 3; max
`lr*wd` = 0.3, safely inside AdamW's well-behaved decoupled-decay regime.

It **stops at 30 deliberately**. Measured at lr=1e-2 on synthetic features: `wd=100` (lr·wd = 1)
zeroes the weight every step → max|W| ≈ lr, a bias-only fit; `wd=300` (lr·wd = 3) → |1−lr·wd| > 1,
max|W| 3.8e7, loss 5.5e8. Selecting either would report an optimizer artifact as a probe, so the
driver **refuses** a `--wd-grid` containing them.

**Null baseline** `--null-wd 100` (default): one extra fit per layer at that extreme decay,
reported next to the probe as the no-information floor (≈ marginal-quantile forecast), with its
fitted max|W| recorded so the label is a measurement. `selected: False` always; `null_wd` inside
the grid raises. It is drawn as a grey curve on the Q=9 figure.

Robustness kept: non-finite candidates are counted (`nonfinite_wd_candidates`) and skipped, and
an all-non-finite grid raises. `wd=300` / `wd=1e9` survive only as test fixtures for that path.

## Open / to check on the first GPU run

- The L20 all-quantile reconstruction is gated **elementwise**: `|recon − decode| ≤ atol +
  rtol·|decode|` per element over the full (N,64,9) tensor (atol 1e-5, rtol 2e-6), failing only if
  the worst *scaled* error > 1. This replaces the old `max|d|/mean|ref|` ratio, which was unfair to
  heavy-tailed forecasts — BOOM's worst element (162.8, only 5e-7 relative, i.e. ~5 float32 ULP)
  inflated to ~1e-4 once divided by the 0.73 global mean. The worst-element error is a uniform
  2–7 float32 ULP across all seven datasets (mean|ref| spans 0.73→4706), so rtol 2e-6 (~16 ULP)
  clears them with 3–20× headroom; the run prints `max_scaled_error`, `max_abs_error` and the
  worst element (decode/recon/index) per dataset. Also whether the installed
  `timesfm3` sorts quantiles inside `decode()` (the run prints the discovered knobs). If it sorts
  with no bypass, re-run with `--allow-sorted-reference` — never silently.
- Whether any layer selects the weight-decay grid maximum (the driver warns); widen `--wd-grid`
  if so.
- Caveats to carry into the writeup: all four datasets are in-distribution for the backbone;
  targets are rebuilt through `arcsinh`/`sinh` (float32, ~1e-5 relative); the native head's
  `clamp(±value_clip)` and any quantile sorting are postprocessing outside the linear class.

---

# Representation geometry + frozen native-head transfer (added 2026-09-16)

Three analyses now sit side by side on the SAME paper7 windows, each answering a different
question about the same representation `h_{l,15}`:

| | question | entry point | fits anything? |
|---|---|---|---|
| learned probe | is the forecast linearly DECODABLE from `h_l`? | `run_timesfm3_last_token_probing.py` | yes (Linear(1280,576) per layer) |
| frozen native head | is `h_l` already in the pretrained readout's COORDINATE SYSTEM? | `run_timesfm3_native_head_transfer.py` | **no** |
| CKA / effective rank | how does the representation GEOMETRY evolve? | `run_timesfm3_representation_geometry.py` | **no** |

The forecasting tunnel stays defined by the trained probe's validation criterion. The other two
read the 5% entrance only as a figure overlay / table index; neither redefines it.

## Estimators — the Chronos-2 ones, imported, never reimplemented

- **CKA**: `probing.cka.cka_matrix(estimator="biased")` (the module default).
  `||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)`, centred across the N observations, float64.
  Verified against a hand-computed reference to 1 ULP.
- **Effective rank**: `probing.spectral_metrics.spectral_metrics`. **Squared** singular values:
  `s = svdvals(X - mean_0(X)); p = s**2/sum(s**2); exp(-sum p log p)`, natural log, eps 1e-12,
  float64. (Measured, not assumed: raw-σ would give 24.15 where the repo gives 18.41.)
  `normalized_effective_rank = r_eff/1280` and `r_eff/min(N-1,d)` are ADDED diagnostics; the raw
  metric is untouched and stays comparable to the committed Chronos-2 records.

Only the representation matrix is model-specific: Chronos-2 stacks `(n,K,768) -> (n*K,768)`;
TimesFM-3 has **K=1** at H=64, so its CKA input is `(N, 1280)` directly, no reshape.

## Splits — the two committed Chronos-2 analyses disagree, so parity is per-analysis

| | Chronos-2 default | TimesFM-3 default |
|---|---|---|
| CKA | `run_cka_analysis.py --fslot-split` = **test** | **test** headline + **train** robustness (`--cka-splits`) |
| effective rank | `run_spectral.py --split` = **train** | **train** (`--erank-split`), unchanged |

`--split {train,test}` forces everything onto one split for a single-split pass.

## CKA: biased for PARITY, unbiased for INTERPRETATION (decided 2026-09-16)

Both estimators run, on both splits, for all seven datasets — identical layer ordering and
plotting style throughout:

| namespace | role |
|---|---|
| `biased/test` | **HEADLINE.** Exact parity with the committed Chronos-2 CKA (`run_cka_analysis.py`: biased estimator, test split). |
| `unbiased/test` | **REQUIRED COMPANION.** The finite-sample-bias-corrected analysis; the informative estimator for *absolute* similarity. |
| `biased/train` | robustness, N=1394. |
| `unbiased/train` | robustness — the only reading with real power for **Coastal T-S** (48 test windows). |

**Why the companion is required.** The biased estimator carries an O(1/n) *upward* bias, and one
native readout token per window means n = the window count. On INDEPENDENT Gaussian
representations at d=1280, where the true CKA is 0:

| setting | biased | unbiased |
|---|---|---|
| TimesFM test N=262 | **0.83** | 0.001 |
| TimesFM PT-OOD N=354 | **0.78** | −0.001 |
| TimesFM Coastal T-S N=48 | **0.96** | 0.024 |
| TimesFM train N=1394 | 0.48 | −0.001 |
| Chronos-2 fslot n=1048, d=768 | 0.42 | −0.001 |

Chronos-2's committed CKA sat at ~0.42 because K=4 slot stacking gave it 1048 rows at d=768;
TimesFM gets 4× fewer rows at 1.7× the width. **A TimesFM test-split biased CKA of ~0.83 is the
floor, not a finding**, and Coastal T-S test (N=48, floor 0.96) is uninterpretable on it.

> **NEVER compare absolute biased CKA across Chronos-2 and TimesFM-3 as though they were on one
> scale** — the finite-sample baselines differ (~0.42 vs ~0.83). Compare patterns within a model,
> or compare the unbiased numbers. This caveat is stored in `summary.json.cross_model_caveat`.

The driver measures each estimator's own null floor per dataset (`--cka-null-floor-reps`, on by
default, memoized on (N, d, estimator)), annotates it on every heatmap, records it in the summary
alongside `*_above_null_floor` margins, and draws `figures/cka/cka_vs_null_floor.png`. Dropping
`biased` from `--cka-estimators` is REFUSED (it is the parity analysis).

`summary.json` carries this note verbatim:

> Biased CKA is retained for direct parity with Chronos-2, but its finite-sample baseline is
> elevated for TimesFM-3 because the representation dimension is large relative to the number of
> test observations. Unbiased CKA is therefore used as a robustness analysis for absolute
> interpretability.

Effective rank stays on **train** only, the committed Chronos-2 spectral protocol, unchanged.

## Frozen native-head transfer — IN MEMORY, no cache (revised 2026-09-16)

`model.output_head` (the checkpoint's own `Linear(1280, 576)`) applied to `h_{l,15}` at every
point, then decode()'s own inverse path: `revin(reverse, token-15 stats) -> clamp(±value_clip)
-> stitch_patches -> + context trend`. Scored by the SAME `native_reference` that scores
decode(), so the curves are directly comparable with the probe's and `A_l = L_l^{head} −
L_l^{probe}` is well defined.

**This experiment no longer uses the feature cache.** Its validity rests on one identity — at
L20 the path IS the native forecasting pathway — and that identity is only exact when the head
is applied to the states decode() just produced.

*Why the cached version failed (measured, Electricity, 2026-09-16):* reloading the token-15
state and re-applying the head gave `max|d| = 9.4e-2`, worst element `−5.18678` vs `−5.18622`
(~1,200 float32 ULP — far too large to be rounding). The cache itself is fine: extraction's own
`verify_native_head` recorded `max_abs = 0.0`, `per_quantile_max_abs = [0.0]×9`, and
`official_monotone_in_quantile_fraction = 0.9995` ruled out quantile sorting. The difference is
that the cached path applies the head to a **pre-sliced** `(b, 1280)` token while decode()
applies it to the full `(b, 1, 18, 1280)` sequence and slices afterwards — a different matmul
shape, hence a different accumulation order, which is not bit-identical under TF32 on an A100.
The old extraction-time check missed it only because it scaled by `mean|ref|` (597 for
Electricity), the same global-mean flaw `54fcb84` had already fixed elsewhere.

**Tolerances were NOT loosened. The cache was removed.** The driver now runs ONE `decode()` pass
per batch with layer hooks, applies the frozen head to all 21 points in memory, and never
serializes a representation. It therefore needs a **GPU** and runs on the **seven test splits
only** (no train/val representations are built).

The L20 identity is checked at **three stages** against decode()'s output from that same pass:

| stage | reference | isolates |
|---|---|---|
| 1 raw head output | none (records magnitude) + **slice-order control** | the TF32/shape effect above, measured directly |
| 2 pre-trend forecast | `decode() − trend` | head / RevIN / clamp / stitch |
| 3 final inverse-transformed | `decode()` | the trend add-back (stages 2+3 differ only by it) |

All three use one comparator (`_endpoint_identity`, elementwise `|d| ≤ atol + rtol·|ref|`,
atol 1e-5 / rtol 2e-6), with raising deferred so every stage is measured before any aborts. The
validated `verify_native_head` additionally runs verbatim on the same tensors. The scalar gate
(L20 Q=9 loss == the native baseline, <1e-5 relative), the head-checksum gate and window parity
are unchanged.

Entry points: `experiments/run_timesfm3_native_head_transfer.py`, `job_timesfm3_native_head.sh`
(**GPU**). `job_timesfm3_geometry.sh` stays CPU-only and now refuses `--head-only` with a pointer
to the GPU job rather than silently running the backbone on CPU. CKA, effective rank and the
learned Q=9 probes remain cache-based and untouched.

## Outputs — heavy on $SCRATCH, paper-ready in the repo

```text
$SCRATCH/timesfm3_geometry/
    matrices/<estimator>/<split>/cka__<tag>.{npy,csv}     # 2 x 2 x 7
    numerical_results/, temporary/
<repo>/results/timesfm3_representation_geometry/
    summary.json, geometry_summary_table.csv
    native_head_transfer_summary.json, native_head_transfer_table.csv
    figures/cka/<estimator>/<split>/{cka_<slug>.png, cka_all_datasets.png}
    figures/cka/cka_vs_null_floor.png
    figures/{effective_rank,native_head}/                  .png + .pdf throughout
```

## Run order (Narval)

1. login node: `python -m tests.test_timesfm3_representation_geometry` (~6 s, 2 threads).
2. `sbatch job_timesfm3_geometry.sh` — **no GPU**, both analyses, ~1.5 h wall (window building
   dominates; the maths is ~2 min). `--geometry-only` / `--head-only` to split them.
3. inside an `salloc` with a GPU: `python -m tests.test_timesfm3_representation_geometry
   --with-model` for contracts 20–25 and 30.

## Verified so far (no GPU, no model)

- All 10 model-free contract groups pass (~6 s): 21×(N,1280) points at token 15; rank-3
  shared-origin features rejected in memory AND on disk; biased CKA hand-checked, symmetric,
  unit-diagonal, scale/rotation invariant, observation-centred, float64; effective rank matches
  `spectral_metrics` exactly with rank-1 → 1.000 and isotropic-40d → 39.8; the null floor
  ordering N=1394 < N=262 < N=48; paper7 roster + PT-ID/PT-OOD labels; window identity and
  split/suite/checkpoint cross-load refusal; probe independence; the frozen-head contracts.
- Full driver dry runs with synthetic caches and the **real** Chronos-2 parity checker against
  the committed artifacts: geometry produced 7×21×21 CKA + rank curves and 20 figures
  (3.3 MB repo-local); native-head produced 7×21 points with both L20 identities holding
  (raw 1.3e-7, scalar 6.9e-9) and the head checksum unchanged.

---

# Linear representation alignment  h_l -> h_L20 -> frozen native head (added 2026-09-16)

The FOURTH analysis on the same paper7 windows. It separates "the information is there" from
"the information is in the right coordinate system".

| | question | entry point | fits anything? |
|---|---|---|---|
| learned probe | is the forecast linearly DECODABLE from `h_l`? | `run_timesfm3_last_token_probing.py` | yes, on FORECAST targets |
| frozen native head | is `h_l` already in the readout's COORDINATE SYSTEM? | `run_timesfm3_native_head_transfer.py` | **no** |
| CKA / effective rank | how does the GEOMETRY evolve? | `run_timesfm3_representation_geometry.py` | **no** |
| **alignment (new)** | can a LINEAR map put `h_l` into that coordinate system? | `run_timesfm3_representation_alignment.py` | yes, on the **REPRESENTATION** only |

## The objective is target-free, and that is the whole point

`W_native` is itself **linear** (`Linear(1280, 576)`). An adapter trained on forecast loss would
collapse to another linear forecasting map `W_native A_l` — the learned probe in disguise. So:

    min_{A,b} ||X_l A + 1 b^T - H_20||_F^2 + lambda ||A||_F^2      (bias NOT regularized)

lambda chosen on **validation representation MSE**, never on any forecast quantity. Enforced
structurally, not by convention: the fitting path loads through
`timesfm3_geometry.load_last_token_reps`, which verifies the cache's `mu`/`sd`/`native` and then
**discards** them, and `assert_no_forecast_targets` refuses any `(n,64)` trajectory or
`(n,64,9)` forecast reaching a fit. Targets are built only AFTER fitting, for scoring.

**NOT the Chronos-2 adapter.** `probing/native_head_adapter.py` trains `Linear(768,768)` on
FORECAST loss into a NONLINEAR ResidualBlock head; its own docstring parks
`min_A ||RMS(A h_l) - h_L12||^2` as "a PARKED future direction, not built here". This is that
parked direction, for TimesFM-3, where the head happens to be linear. Never conflated.

## Solver

N=1394 train rows against d=1280 (1.64M coefficients), so ridge is mandatory and conditioning
must be watched. ONE economy SVD per layer, float64:

    Xc = U S V^T     A_lambda = V diag(s/(s^2+lambda)) U^T Yc     b = muY - muX A

Every lambda is a re-weighting of that one decomposition — no refit, no iterative solver, no SGD.
Grid `1e-8 .. 1e6` (15 points): the spec's log grid extended upward because UNSTANDARDIZED
1280-d states can put `s_max^2` near ~1e6. Deliberately NOT `WD_GRID_LAST_TOKEN` (decoupled AdamW
decay, a different parameterization) and NOT `ridge_regression_probe`'s alphas (standardized
features, 1-D target). Per layer the run records the spectrum, the condition number and
`df(lambda) = sum s^2/(s^2+lambda)`, and WARNS on a grid edge.

## Two endpoints, both reported, NOT required to be bit-identical

| | what | role |
|---|---|---|
| `cached_L20_native_head` | frozen head on the CACHED L20 rep, the same `(N,1280)` path every aligned curve takes | the INTERNAL alignment endpoint; `A_20 = I, b_20 = 0` produces it by construction, and the run ABORTS unless the two agree EXACTLY |
| `official_native_decode` | decode()'s own forecast, cached at extraction | the TRUE model baseline |

Their difference (~1e-6 relative) is the `(N,1280)`-vs-`(b,1,18,1280)` accumulation effect the
native-head run measured with its own slice-order control (3.34e-6 on a raw head output of
max|h| 3.71). A synthesized `A h + b` has no token sequence to sit in, so the cached path is the
only possible one — and the endpoint is defined to travel it too. Numerical provenance, not an
adapter result. `--endpoint-report-rtol` only FLAGS a large difference; it never gates.

## Controls

L20 identity (exact, aborts), mean baseline (train mean of `h_L20`), permuted correspondence
(same X, same SVD, same grid — only the row pairing destroyed), the no-forecast-target shape
guard, the head checksum before/after, and train-fits / val-selects / test-once discipline.

## Files (all new; nothing existing modified)

- `probing/timesfm3_alignment.py` — ridge/Procrustes/identity solvers, representation metrics,
  controls, the frozen-head application, the leakage guard, the scoring-side cache reader.
- `experiments/run_timesfm3_representation_alignment.py` — driver (4 curves, bootstrap, figures).
- `tests/test_timesfm3_representation_alignment.py` — 26 numbered contracts.
- `job_timesfm3_alignment.sh` — SLURM, **CPU only, no GPU**.

## Run order (Narval)

1. login node: `python -m tests.test_timesfm3_representation_alignment` (~5 s, 2 threads).
2. `sbatch job_timesfm3_alignment.sh` — **no GPU**, ~2 h wall (window building dominates; the
   linear algebra is ~15 min). `--alignment ridge procrustes` adds the orthogonal control.
3. inside an `salloc` with a GPU:
   `python -m tests.test_timesfm3_representation_alignment --with-model` for contracts 13/14.

## Outputs

```text
$SCRATCH/timesfm3_alignment/
    adapters/<dataset>/<alignment>/L{00..19}.npz     # float32 A + b, ~6.6 MB each
    numerical_results/representation_alignment__<tag>.json
<repo>/results/timesfm3_representation_alignment/
    representation_alignment_summary.json, representation_alignment_table.csv
    figures/forecast/aligned_vs_direct_vs_probe_all_datasets.{png,pdf}
    figures/representation/representation_r2_by_layer.{png,pdf}
    figures/recovery/alignment_recovery_by_layer.{png,pdf}
```

## Open / to check on the first run

- Whether any layer selects the ridge grid maximum or minimum (the driver warns); widen
  `--ridge-grid` if so. `s_max^2` is printed per dataset so the grid can be judged, not guessed.
- Whether the aligned curve tracks the 5% tunnel entrance, the CKA change or the effective-rank
  collapse — the hypothesis is NOT hard-coded anywhere; the full Emb..L19 depth curve is fit so
  the answer can come out either way.
- Caveat to carry into the writeup: a high representation R^2 at layer l says a LINEAR map into
  the L20 basis EXISTS and was found from 1394 windows. It does not say the model performs that
  map, nor that the adapter generalizes off these windows.

---

# TiRex: Q=1 layer-wise probing + representation geometry (added 2026-09-20)

The THIRD model. Same paper7 windows, same tunnel rule, same CKA / effective-rank estimators.
No Chronos-2 or TimesFM-3 file is modified; the roster, window builders, MASE and tunnel are
imported from the modules that already own them.

## Geometry — DERIVED FROM THE CHECKPOINT, and three paper assumptions were wrong

Discovered via `tirex-ts==1.4.2` on `NX-AI/TiRex` (`--discover-only` prints and saves all of it):

| | value | matches the paper? |
|---|---|---|
| input / output patch | 32 / 32 | yes |
| embedding dim / blocks / heads | 512 / 12 / 4 | yes |
| input_ff_dim | 2048 | yes |
| quantiles | 0.1 … 0.9 (Q=9, median idx 4) | yes |
| final norm | `RMSNorm(512)` after block 12 | yes |
| output head | `ResidualBlock(512 → 2048 → 288)`, 288 = 9×32 | yes |
| **train_ctx_len** | **2048** | yes — but its CONSEQUENCE was missed |
| **readout tokens** | **63 and 64**, not 15 and 16 | **NO** |
| **native multi-patch path** | **two passes by default**, not one | **NO** |

**1. The readout tokens are 63 and 64.** `_forecast_single_step` calls
`_adjust_context_length(ctx, train_ctx_len, train_ctx_len)`, forcing EVERY context to exactly
2048 by NaN LEFT-padding. C=512 becomes `[1536 NaN = 48 masked patches][512 real = 16 patches]`
= 64 context tokens. "16 real context patches" is true; "token 15" is not — the real patches sit
at 48..63. Nothing in the code hardcodes 63/64: `TiRexGeometry` derives them and asserts
`readout_indices[0] == n_context_tokens - 1`.

**2. The package default is NOT the paper's path.** `max_accelerated_rollout_steps` defaults to
**1**, which runs TWO forward passes (second one on a context shifted by 32, with its own
loc/scale). Only `=2` gives the single pass with 65 tokens and two adjacent readouts — the
paper's "future patches as missing inputs", and the only mode where both states share one
recurrent scan and one normalization. **We use =2**; the gap to the default is measured per
dataset (`rollout_mode_gap`: first 32 steps identical, last 32 differ by ~7e-3 relative).

**3. `sLSTMCellTorch` runs the recurrence in bfloat16** (pointwise math promoted to fp32 each
step). The `cuda` backend is xLSTM's own kernel and will NOT agree bit-for-bit. Backend is
recorded in every cache (cross-backend reuse is REFUSED) and `--compare-backends` quantifies it.

## THE GATE — answered YES, bit-exactly

States at tokens [63, 64] → frozen `output_patch_embedding` → `unflatten(-1, (Q, P))`
(**quantile-major**, the opposite of TimesFM-3's horizon-major) → `transpose` →
`tokenizer.output_transform` → `swapaxes` **== decode()'s own H=64 forecast**, `max|d| = 0.0`,
`torch.equal` True, on synthetic AND on real Electricity windows. Three wrong-index controls
(one token early / both = first / reversed) each differ by 6.1–1.1e3. Re-proved every dataset.

## Normalization — read off `PatchedTokenizer`, not guessed

`loc = mean(x[T-C:T])`, `scale = population std(x[T-C:T])` (nanmean excludes the pad), with
TiRex's own degenerate guard `scale ← |loc| + 1e-5`. Verified numerically to 1e-4/1e-5. The
future is a separate argument to `build_targets` and reaches neither the model nor the scaler.

## Probe — ONE SHARED `Linear(512, 32)`, Q=1, tau=0.5

Applied to both readout states, concatenated to H=64. 16,416 params. Sharing is structural (one
`nn.Linear` on a `(B, K, 512)` tensor). ONE `StandardScaler` on the STACKED (2N, 512) TRAIN rows
— a per-position scaler would silently un-share the head.

**The wd grid had to be extended, and my first reasoning was wrong.** I argued `WD_GRID_V2`
would suffice because the row/param ratio is 6× better than TimesFM-3's. Measured on full
Electricity: **12 of 14 depths selected wd = 3, the grid maximum**, train ≪ val everywhere. The
ratio argument ignored that the states' effective rank is ~7–37, not 512. `WD_GRID_TIREX` =
`WD_GRID_V2 + (10, 30)`; after the change the max selected is 10 and no depth clips.
`assert_wd_grid` REFUSES any lr·wd ≥ 1 (wd=100 zeroes the weight every step). The
no-information reference is `constant_forecast_floor` — closed form, no fit.

## One real degeneracy, detected not hidden

**(Emb, position 1) is bit-identical across ALL windows.** The masked-future token reaches the
patch embedding as (values=0, mask=0) before any recurrence, so `input_patch_embedding` of it is
a pure bias. Every deeper depth varies at both positions. Consequences: Emb's second-patch
forecast is a constant, and Emb's headline (2N, d) geometry matrix is n varying rows plus n
copies of one point — which depresses its spectrum (test r_eff 4.1 headline vs 7.9 pos0). The
headline construction is kept as specified; a **`pos0` companion** (position 0 only) is computed
alongside it and is the like-for-like comparison. `degenerate_readouts` asserts the affected set
equals exactly `[{Emb, 1}]` and WARNS otherwise.

## Depth convention (needed for cross-model plots)

`relative_depth = block_index / 12` — Emb 0.0, Lk k/12, **L12+RMS 1.0 (ties with L12; the norm
adds no block, `kind` distinguishes them)**. `relative_position = position_index / 13` is
strictly monotone for plotting but NOT cross-model. Both are in every record.

## CKA / effective rank

Both estimators, both splits, on the SAME (2N, 512) matrix the probe's rows come from, RAW (a
z-scored matrix is refused). Measured null floors at the real sizes make the point:

| split | rows | biased floor | unbiased floor |
|---|---|---|---|
| train | 2788 | **+0.156** | +0.001 |
| test | 524 | **+0.492** | −0.004 |

So **absolute biased CKA on the test split is uninterpretable** and is never compared across
models (Chronos-2 ~0.42 at 1048×768, TimesFM-3 ~0.83 at 262×1280). Headline = **unbiased**.

## First real result (Electricity, full 1394/262/262)

Tunnel entrance **L11** (relative_depth 0.92) — decodability keeps improving nearly to the end.
Final-depth MASE 1.000 vs TiRex native 0.818 (no linear probe matches the native head, as on the
other two lines). Effective rank rises 7 → ~38 (L7–L10) then **collapses to 7.8 at L12+RMS**.
Second patch is consistently ~1.6× harder than the first at every depth — smoothly, not
catastrophically.

## Files (all new; nothing existing modified)

- `probing/tirex_model.py` — geometry dataclass, discovery, freezing, hooks, native path,
  `verify_native_head`, targets, cache, `compare_backends`.
- `probing/tirex_probes.py` — the Q=1 shared patch probe, wd grid + guard, tunnel, floors.
- `probing/tirex_geometry.py` — rep matrix, degeneracy, CKA/erank (imported, not reimplemented).
- `experiments/run_tirex_probing.py` — driver. `tests/test_tirex_probing.py` — 27 contracts.
- `job_tirex_probing.sh` — SLURM.

## Run order (Narval) — see the job script header for the full commands

1. login node: `pip install "tirex-ts==1.4.2"`, pre-cache the checkpoint, then
   `python -m tests.test_tirex_probing` (~5 s, 2 threads).
2. salloc: `--discover-only`, then `python -m tests.test_tirex_probing --with-model`.
3. `sbatch ... --limit 64` smoke, then one full dataset, then `sbatch job_tirex_probing.sh`.

## Rollout mode: BOTH native paths implemented and compared (2026-09-20)

`--rollout-mode {single_pass, two_pass}`; the mode is part of the CACHE PATH so the two can
never be mixed. The seven-dataset run is ON HOLD until the primary definition is chosen.

### What pass 1 consumes — derived from the package, not inferred

`_forecast_tensor`:

    context = torch.cat([context, torch.full_like(prediction[:, 0, :], fill_value=torch.nan)], -1)

`torch.full_like(X, fill_value=nan)` takes X's SHAPE/dtype/device and fills it with NaN, so
`prediction[:, 0, :]` (the tau=0.1 row) is a **shape template only**. **Pass 1 consumes MISSING
VALUES — never predicted ones, and never a reconstructed context.** TiRex's rollout is not
autoregressive; it re-asks the model with the horizon marked missing. `assert_rollout_appends_
missing` proves it against the running model every run: appended block all-NaN True, any finite
False, equals a forecast quantile row False, real prefix unchanged True (with the discarded
forecast's median magnitude printed, so "present and ignored" is visible).

Consequence: every pass's context — and hence its (loc, scale) — is constructible WITHOUT a
forward pass, which is what `rollout_contexts` / `build_targets` do.

### Per-pass layout (derived, asserted every run)

| pass | context | pad | real tokens | readout 63 is | -> horizon |
|---|---|---|---|---|---|
| 0 | x[T-512:T] | 1536 (48 patches) | 48..63 | the LAST REAL patch | y[T:T+32] |
| 1 | x[T-512:T] ++ 32 NaN | 1504 (47 patches) | 47..62 | an APPENDED MISSING patch | y[T+32:T+64] |

Both passes are 64 tokens and both read their own token 63. Geometry refuses any config with
C + (K-1)*P > train_ctx_len, which would left-truncate real values and silently change the
target space.

### Normalization: per-pass, carried faithfully

Each patch is normalized AND de-normalized with its own pass's (loc, scale); `build_targets`
returns `(n, K)` statistics in both modes and `denormalize` REFUSES flat `(n,)` stats. The two
passes' stats are the SAME mathematical quantity (mean/pop-std of the 512 real values) but not
bit-identical: **6e-8 relative**, pure float32 reduction order (the NaN slots sit differently).
My earlier claim that they were identical was wrong.

### THE GATE holds in both modes

| mode | readout | max scaled err | max abs | bitwise | controls |
|---|---|---|---|---|---|
| single_pass | tokens [63, 64] of one pass | 0.000 | **0.0** | **True** | 1.06e3 |
| two_pass | token 63 of each pass | 0.249 | 2.4e-4 | False | 1.05e3 |

two_pass is not bit-exact because the native path applies `output_patch_embedding` to the full
(n, 64, 512) sequence and slices after, while we apply it to the pre-sliced (n, 1, 512) readout —
a different matmul shape. A **slice-order control (1.9e-6)** attributes the residual to exactly
that, so it is measured, not excused. `assert_two_pass_matches_package` separately proves our
replicated loop equals the package's own default output BITWISE.

### RESULT — Electricity, full 1394/262/262, all 14 depths

**Causality first:** readout position 0 is **BIT-IDENTICAL across the two modes at all 14
depths** (an appended token cannot influence token 63 in a causal sLSTM). Only position 1
differs (L12 max|d| 1.05 vs layer std 15.8; Emb 0.0 — the shared constant).

| quantity | single_pass | two_pass | verdict |
|---|---|---|---|
| tunnel entrance @5% | **L11** (0.92) | **L11** (0.92) | identical |
| tunnel @1%/2%/10% | L12+RMS / L12+RMS / L11 | same | identical |
| test-loss argmin | L12+RMS | L12+RMS | identical |
| Spearman rho over depths (test / val) | — | **0.991 / 1.000** | ordering preserved |
| max d test loss | — | 0.0038 (**2.6%** of curve range) | small |
| max d MASE | — | 0.0261 (**3.1%** of range) | small |
| headline unbiased CKA (pos0) | — | **max d = 0.0, BIT-IDENTICAL** | identical |
| all_positions unbiased CKA | — | max d 0.004 (train) / 0.005 (test) | negligible |
| effective rank (mixed) | argmax L7 | argmax L7, max rel d 0.7%/1.4% | NOT material |
| final-depth MASE | 1.0004 | 1.0029 | — |
| native MASE | 0.8176 | 0.8126 | two_pass marginally better |
| native forecast gap between modes | — | max d 9.21 (3.9% rel), first patch identical | the real difference |

**Conclusion: the choice of rollout mode does NOT change any conclusion.** The tunnel entrance,
the depth ordering, the argmin and the headline CKA are identical; the curves shift by ~3% of
their own range. The headline CKA is bit-identical *because* it is built from position 0, which
causality makes mode-invariant. The modes differ where you would expect: the second readout
state and the native forecast's last 32 steps. Cost: two_pass extraction is 2x (472 s vs 279 s
per dataset on CPU).

**DECIDED 2026-09-20: `two_pass` is the paper's PRIMARY definition**, because it is the released
`tirex-ts` package's own default inference pathway. It is now the default of `--rollout-mode`,
and `single_pass` is retained as the documented robustness check (the comparison above is the
evidence that the choice costs no conclusion). Cost: K=2 forward passes per window, 472 s vs
279 s per dataset on CPU.

## Geometry row construction (FINAL, 2026-09-20)

Because (Emb, position 1) is constant before recurrence, and because an observation count that
changes with depth is not a comparable curve, **CKA and effective rank now use the SAME
construction, and it is constant-N**:

| variant | rows | depths | role |
|---|---|---|---|
| **`pos0`** | **N** (readout position 0 only) | **all 14, Emb..L12+RMS** | **HEADLINE, both estimators** |
| `all_positions_from_L1` | 2N (both forecast states) | 13, **L1..L12+RMS** | companion |

Why constant N is the headline, not a convenience: both `min(N-1, d)` (the attainable rank
ceiling) and the O(1/N) biased-CKA floor move with N, so an N that jumps from N at Emb to 2N at
L1 would be **indistinguishable from a representational change** at exactly the depth of
interest. The earlier `mixed` curve (Emb at N, L1+ at 2N) is therefore **not produced at all** —
`variant_spec` refuses the name, and test 28 asserts no such curve exists in the output.

It is also the only construction CKA admits: CKA is pairwise and requires MATCHED ROWS
(`probing.cka.require_matched_rows` refuses otherwise), so an N-row Emb could never be compared
with a 2N-row L1. Bonus: `pos0` is one row per window at d=512, matching TimesFM-3's
one-readout-row-per-window construction.

The companion starts at L1 precisely because Emb's second state is degenerate; it is
self-consistent (constant 2N across the depths it covers) and is where the second forecast
state's geometry is read. Measured on Electricity (train): a 2N-row Emb would read r_eff 6.9 vs
the headline's 6.9 — but on test, 4.1 vs 7.9, which is the distortion being avoided.

**The forecasting probe is UNCHANGED**: the same `Linear(512, 32)` shared across the two native
forecast-producing states of the two default inference passes. The geometry policy touches only
CKA and effective rank.

## First result under the FINAL configuration (two_pass, pos0 headline)

Electricity, full 1394/262/262, 14 depths: tunnel entrance **L11** (relative_depth 0.92), final-
depth MASE 1.0029 vs native 0.8126. Headline effective rank (pos0, train) 6.9 (Emb) -> 39.5 (L7)
-> 7.5 (L12+RMS); companion (2N, L1+) 9.6 (L1) -> 7.8. Biased CKA null floor +0.269 (train,
N=1394) / **+0.661** (test, N=262) vs unbiased +0.001 — the test-split biased estimator is
uninterpretable in absolute terms, so the headline estimator stays **unbiased**.

## Two numerical/device facts found on the first GPU attempt (2026-09-21)

### Device hygiene — was a PRODUCTION bug, not just a test bug

The first Narval GPU run crashed in the test at `model._forward_model(inp, ...)` with a CPU/CUDA
mismatch. Auditing the whole model-backed path turned up THREE faults, two of them in production:

| where | fault |
|---|---|
| `_forecast_quantiles` defaults `output_device="cpu"` | the official forecast came back on CPU while the hook-captured states stayed on the GPU -> contract 5 would have crashed next |
| `native_forward_two_pass` / `assert_rollout_appends_missing` call `_forward_model_tokenized` / `_forecast_single_step` directly, and those (unlike `_forecast_quantiles`) never move the context | **the driver in `two_pass`, the PRIMARY mode, would have crashed in the smoke job** |
| the test built `pad`/`mask`/`inp` on the CPU | the observed traceback |

Fixed with `model_device(model)` / `to_model_device(model, x)`: every entry point that RUNS the
backbone normalizes its context to the model's device and pins `output_device`, mirroring what
`_forecast_quantiles` already does for itself. Parameter-free paths (`scaler_state`,
`rollout_contexts`, the tokenizer) deliberately FOLLOW their input instead — they carry no
parameters, and forcing a device there would break the model-free tokenizer stub for no gain.
`model_device` falls back to CPU for a parameter-free stand-in. Nothing is hardcoded to `.cuda()`.

**Contract 32** is the regression guard: every entry point must accept a CPU/numpy context against
a model on any device and return ONE consistent device. Validated on **CPU and on MPS** (a real
non-CPU device on the Mac), plus a full driver smoke run on MPS in `two_pass`.

### Representations are mildly BATCH-SIZE dependent on accelerators

Found while validating the above on MPS. TiRex's sLSTM recurrence runs in **bfloat16**, and a
small-batch GEMM kernel switch perturbs block 1 by ~1e-3 of its std, which 64 timesteps x 12
blocks then amplify:

| batch pair (MPS) | Emb | L1 | L5 | **L6/L7** | L12 | L12+RMS | native forecast |
|---|---|---|---|---|---|---|---|
| 4 vs 8 | 0 | 0 | 0 | **0** | 0 | 0 | 0 (BITWISE identical) |
| 2 vs 8 | 0 | 1.1e-3 | 6.7e-3 | **6.1e-2 / 6.4e-2** | 1.0e-2 | 1.8e-2 | 8.6e-4 relative |

(as a fraction of each depth's own std). Emb is exactly 0 everywhere — no recurrence, pure Linear.
On CPU everything is bitwise identical. So it is a SMALL-batch effect amplified by bf16, not a
general instability.

Consequences, all implemented:
* **`batch_size` is now part of the feature-cache key** and a differing one REJECTS the cache.
  Numbers that depend on batch size must not be silently reused across batch sizes.
* `--batch-size` is documented as a reproducibility parameter, not a speed knob; keep it fixed
  for the whole paper and prefer a LARGE value (256).
* **Contract 19 split in two**: bitwise determinism at a FIXED batch size is asserted exactly on
  every device; across batch sizes it is asserted bitwise on CPU and otherwise MEASURED, PRINTED
  and held under a clearly-labelled catastrophe bar (0.2 of std) that is NOT a precision claim.
* `summary.json.extraction` records the batch size and the reason.

## Open / to check on the first Narval run

- Whether the `cuda` backend changes anything: run `--compare-backends` ONCE on the GPU and
  record it. Pick one backend for the whole paper.
- What the batch-size sensitivity measures on CUDA specifically (contract 19 prints it). MPS gave
  6e-2 of std at L6/L7 for batch 2; batch 4 and 8 were bitwise identical. The real runs use 256.
- Whether any depth still selects the wd grid maximum (30) on the PT-OOD datasets.
- Whether the effective-rank collapse at L12/L12+RMS reproduces on the other six datasets — it
  is the most striking Electricity result and the one most worth being skeptical about.
- Whether the mode-equivalence result holds on the PT-OOD datasets, where the horizon is harder
  and the second readout state may matter more (run `--compare-rollout-modes` on one of them).
- Whether the headline and companion effective-rank curves diverge on any dataset — they track
  each other closely on Electricity train but separate on test (7.9 vs 5.8 at L1).
- NOT DONE, deliberately: TiRex truncation / representation alignment.
