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

---

# Benchmark expansion to 14 datasets — APPROVED 2026-09-21 (yield screen reported, gate passed)

Answers the reviewer objection that layerwise "forecasting tunnels" might be an artifact of
pretraining exposure. Status: roster **FROZEN** on measured numbers (`experiments/
rolling_yield_screen14.py`, full run 2026-09-21, `$SCRATCH/yield_screen14.json`). One
casualty (`kdd_cup_2022_10T`), one promotion (`electricity_15min` as a labelled control).
The registry refactor is now UNBLOCKED — but four blockers must land before any new dataset
is extracted.

## The roster (14)

| # | dataset | freq | domain | m | source |
|---|---|---|---|---|---|
| 1-4 | m4_hourly, monash_electricity_hourly, uber_tlc_hourly, wind_farms_hourly | 1H | mixed/energy/transport | 24 | `autogluon/chronos_datasets` |
| 5-7 | boom_hourly, sg_carpark, coastal_ts | 1H | cloud/transport/nature | 24 | staged arrow shards |
| 8 | LOOP_SEATTLE_5T | 5min | road speed | 288 | `autogluon/fev_datasets` |
| 9 | electricity_15min | 15min | residential energy | 96 | `autogluon/chronos_datasets` |
| 10 | SZ_TAXI_15T | 15min | road speed | 96 | `autogluon/fev_datasets` |
| 11 | monash_london_smart_meters | 30min | residential energy | 48 | `autogluon/chronos_datasets` |
| 12 | m5 | 1D | retail | 7 | `autogluon/chronos_datasets` |
| 13 | wiki_daily_100k | 1D | web | 7 | `autogluon/chronos_datasets` |
| 14 | monash_traffic | 1H | road volume | 24 | `autogluon/chronos_datasets` |

**#9 `electricity_15min` is a CONTROL, not a domain.** It is the SAME 370 meters as
`monash_electricity_hourly` at 4x the sampling rate, so it holds domain AND series fixed and
varies only the rate: *does the tunnel entrance move with sampling frequency?* Every table and
figure must label it as a frequency control — it may never be counted toward domain diversity.

Designated alternates, screened but NOT in the roster: `rossmann_1D` (M5 fallback, target col
`Sales`) and `kdd_cup_2022_10T` (dropped at the gate — see below).

**FIVE** genuine frequency classes (5min / 15min / 30min / 1H / 1D) plus the 15min control
pair. The 10-min class is deliberately empty. Every `m_season` for a dataset shared with
fev-bench EQUALS fev-bench's published `seasonality` — the choice has an external citation,
not just our assertion.

## Decisions frozen 2026-09-21

- **Q=1 / tau=0.5 is the cross-model setting.** TimesFM-3's last-token line is currently
  hard-wired to Q=9 (`NUM_QUANTILES`); it gains `--quantile-set {q1,q9}` (the probe factory
  `timesfm3_last_token_probes.make_probe` already takes `Q`). Q=9 is retained as a
  TimesFM-3-specific native-distribution appendix. TiRex is already Q=1; Chronos-2 has both.
- **MASE never defines the tunnel.** The tunnel entrance stays on the common Q=1 validation
  loss. MASE is a post-hoc interpretable test metric only, so a dataset-specific seasonal
  denominator can never move the definition of "recoverable".
- **`seasonal_m` is centralized per dataset** (table above), with a test pinning it to 24 for
  the original seven so the committed numbers cannot move.
- **M5 is kept** — the only dataset explicitly held out by all three model authors. Automatic
  swap to `rossmann_1D` if the screen shows >30% of otherwise-eligible windows lost to
  zero/invalid seasonal denominators, or the task is degenerate.

## Provenance schema — status x evidence_scope (2-D, never collapsed)

```text
status:          pretraining_exposed | explicitly_held_out | not_listed | undocumented
evidence_scope:  exact_dataset | benchmark_exclusion | source_level | source_family
```

`status` alone would let `wiki_daily_100k x TimesFM-3` (source_level: the card names
"Wikipedia Pageviews, cutoff Nov 2023"; our series end 2022-12-31) read as identical evidence to
`wiki_daily_100k x Chronos-2` (exact_dataset: Table 6 "Wiki"). It is not. Counts in any paper
table must be reported per (status, evidence_scope) cell.

Paper wording this licenses:
> We track model-dataset pretraining provenance using explicit dataset inclusion, documented
> benchmark exclusions, and weaker source-level evidence; we do not equate absence from a
> corpus list with out-of-distribution evaluation.

## Primary-source citations (verbatim anchors)

- Chronos-2 (arXiv:2510.15821): Table 6 = "The full list" of real univariate pretraining data
  (treat as exhaustive). Sec 5.1 fev-bench "None of these datasets or tasks were seen by
  Chronos-2 during training."; Bench-II "None of these datasets were included in the training
  corpus of Chronos-2."; GIFT-Eval "did not overlap with the test portions ... Nonetheless, the
  corpus does include partial overlap with the training portions". NO cutoff stated.
- TimesFM-3 (model card `google/timesfm-3.0-pytorch`; blog 2026-08-31, no paper): corpus =
  "GiftEvalPretrain excluding the datasets that overlap with fev-bench" + "Wikipedia Pageviews,
  cutoff Nov 2023" + "Google Trends top queries, cutoff EoY 2022" + "Synthetic and augmented
  data". "augmented" is undefined — the weakest per-dataset documentation of the three.
- TiRex (arXiv:2505.23719 App C.2/C.3): Chronos-1 corpus (Table 5) + GiftEval subset (Table 6) +
  15M synthetic GP. "TiRex's pre-training data has no overlap with Chronos-ZS benchmark."
  16 of 97 GIFT-Eval settings excluded — **the 16 are never named**.

**CORRECTION to `data/chronos2_seen_manifest.md`:** it claims BOOM is "explicitly listed" in
Chronos-2's documented-unseen reservoir. The strings "BOOM" and "Datadog" do NOT appear in
arXiv:2510.15821. BOOM's held-out status is an INFERENCE via fev-bench (which contains BOOMLET,
a BOOM subset). Downgrade `boom_hourly x Chronos-2` to not_listed + a benchmark_exclusion note.

## Verified dataset facts (HF datasets-server, 2026-09-21)

| dataset | series | length (min/med/max) |
|---|---|---|
| LOOP_SEATTLE_5T | 323 | 105,120 uniform (std 0.0) |
| monash_traffic | 862 | 17,544 uniform |
| wiki_daily_100k | 100,000 | 2,741 uniform (2015-07-01 -> 2022-12-31) |
| monash_london_smart_meters | 5,560 | 288 / 30,864 / 39,648 — NOT uniform |
| m5 | 30,490 | 124 / 1,810 / 1,969 — NOT uniform |
| rossmann_1D | 1,115 | 942 uniform |
| SZ_TAXI_15T | 156 | **UNVERIFIED** — statistics endpoint 500s; canonical T-GCN = 156x2976 and the 2.63 MB parquet is consistent with 2,976, not with the 1,440 first-rows reported (cell truncation). THE SCREEN MUST SETTLE THIS. |

**Schema corrections:** `kdd_cup_2022_10T` has NO `target` column — the target is **`Patv`**
(fev's own task declares `target: Patv`); columns are id/timestamp/Wspd/Wdir/Etmp/Itmp/Ndir/
Pab1-3/Prtv/Patv. `rossmann_1D` target is **`Sales`**. fev's `SZ_TAXI_15T` task lists
solar/weather covariates that do not exist in the config (a copy-paste bug in fev's tasks.yaml) —
its real columns are only target/id/timestamp. fev target columns are NOT uniformly "target".

## YIELD SCREEN RESULT — the gate report (2026-09-21, compute node nc20108)

Full 16-dataset run (14 proposed + 2 alternates), `--json $SCRATCH/yield_screen14.json`.

### The `denomfail%` column was a FALSE ALARM — and this matters

The screen measures `id_data._seasonal_naive_scale(s[:te_st+C], m)`, the **canonical** per-series
denominator stored as `test_denominator`. **Nothing consumes it.** Grepped every consumer across
`probing/*.py` + `experiments/*.py`: zero, and `run_id_forecasting.py:250` says so in its own
docstring ("NOT the canonical train-series MASE (id_data.test_denominator, unused here)").

Every reported MASE — Chronos-2, TimesFM-3 AND TiRex — goes through
`run_timesfm3_probing.mase_denominator(X_test)` (`run_tirex_probing.py:197`,
`run_timesfm3_last_token_probing.py:358`, `run_timesfm3_native_head_transfer.py:356`,
`run_timesfm3_representation_alignment.py:294`): the **IN-CONTEXT** seasonal-naive scale over the
512-point context, floored at `MASE_DEN_FLOOR = 1e-8`. The context is finite by construction (the
window validity filter guarantees it), so that denominator **can never fail**.

**=> wind_farms_hourly's committed MASE is NOT broken.** 96.3% / 100% / 98.5% invalidate nothing
that has been published.

What the column DOES measure is "this series has >= 1 NaN anywhere in its history", because
`_seasonal_naive_scale` uses `.mean()`. London's arithmetic confirms the mechanism: median length
30,864 -> `(30864-576)//64+1` = 474 origins; one isolated NaN sits inside `C+H = 576` points and
therefore rejects 9 consecutive windows; 9/474 = **1.9%** vs the measured **2.15% rejection**.
That is *sparse isolated missing readings*, not a leading pad — harmless under the in-context
denominator.

Two consequences to carry into the writeup:
- our MASE is **not** the fev-bench / GIFT-Eval denominator, and a reviewer will notice. For
  wind_farms / kdd / london we now know we **could not** switch to the canonical one even if
  asked — unless `_seasonal_naive_scale` gains a `nanmean` over valid lagged pairs (a one-line
  change; it is currently fail-loud by design). NOT done, deliberately.
- `M_SEASON = 24` is a module constant with no per-dataset path (blocker 3).

### Per-dataset verdicts

| dataset | verdict | measured basis |
|---|---|---|
| m4, electricity, uber, sg_carpark, coastal | clean | committed; nothing moved |
| **m5** | **KEPT — swap trigger did NOT fire** | rej 0.62%, denom fail 0.000%, madMed **0.490**. The ">30% of windows lost to zero/invalid denominators, or degenerate" rule is not met, so `rossmann_1D` stays benched. |
| LOOP_SEATTLE, SZ_TAXI, wiki, traffic, electricity_15min | clean | rej <= 0.36%, denom fail 0 |
| **wind_farms_hourly** | kept (committed) + NEW CAVEAT | **21.95% of its val/test windows have a literally constant future** (`mad_const == 0`) — by far the highest in the roster. Those windows score ~0 at EVERY depth, so they are dead weight compressing the layer-to-layer differences. Plausible (curtailed turbines); belongs in the writeup. |
| **boom_hourly** | kept (committed) + caveat | madMed **0.121**, ~3x lower than any other dataset, plus 12.0% zero-MAD. Least dynamic range in the roster => small absolute deltas there mean less than they look. |
| **kdd_cup_2022_10T** | **DROPPED** | below |

### Why kdd_cup_2022_10T was dropped

**75.2% of candidate windows rejected** — not a footnote: the evaluation lands on a
**missingness-selected 25% of time**, and the sensor outages choose which 25%, not us.
Compounding it, val/test caps at **134 clusters** (51% of the 262 everything else gets), so it is
simultaneously the least trustworthy AND the least powered dataset in the roster. Its domain
(wind power) is already covered by `wind_farms_hourly`, so dropping it costs the 10-min frequency
class but **no unique domain**. Replaced by `electricity_15min`, promoted from alternate to
labelled frequency control.

### Screen-reporting bug (data fine, table under-reports)

`rolling_yield_screen14.screen` hardcodes `realized_valtest = min(BUDGET_VALTEST=262, n_clusters)`,
but `build_ood_rolling_windows` defaults `target_val = target_test = None` — EVERY eligible series
contributes. So the screen understates the three OOD rows:

| dataset | screen printed | builder actually gives |
|---|---|---|
| sg_carpark | 262 / 262 | **354 / 354** |
| boom_hourly | 262 / 262 | **354 / 354** |
| coastal_ts | 24 / 24 | **48 / 48** windows over 24 clusters |

These match the counts already recorded above, so nothing regressed — but do not read those rows
as a change. The cluster count is the statistically meaningful number either way.

### Still open from the screen

- `SZ_TAXI_15T` length is settled only in the JSON, not in the printed table (the realized column
  caps at 1394 and both 1,440 and 2,976 clear it). Read it off with
  `python -c "import json;print([(r['tag'],r['series_len_min_med_max']) for r in json.load(open('$SCRATCH/yield_screen14.json'))['rows']])"`
  (login node, seconds, one small JSON).

## Four blockers — ALL LANDED 2026-09-21 (see the implementation record below)

1. **`id_data._build_rolling_windows` RAISES when n_eligible_series > target_train (1394).**
   The 262 val/test series are drawn from the full eligible pool, but the cluster-balanced round
   robin reaches only 1394 distinct series, so the fail-loud
   `missing = sel_set - set(tr_sid)` trips. Hits m5, wiki_daily_100k, london_smart_meters (and
   the alternates). Fix = a DETERMINISTIC series-level cap applied before the protocol; it also
   removes the multi-GB raw-series memory spike.
   **CONFIRMED BY MEASUREMENT** — and for exactly three datasets: london (5555 eligible),
   m5 (28491), wiki_daily_100k (100000).
2. **`tunnel.domain_status()` raises on unknown tags** (`probing/tunnel.py:53`) and stamps a
   Chronos-2-relative label into every tunnel record. Any new tag breaks `tunnel_record` on the
   first call. STILL STANDS.
3. **NEW — the screen did not catch this one. `seasonal_m` is not centralized ANYWHERE yet.**
   `mase_denominator(X, m=M_SEASON)` defaults to the module constant **24**
   (`run_timesfm3_probing.py:40,49`) and all four drivers call it positionally with no override.
   Add LOOP_SEATTLE_5T (m=288) and the MASE denominator is built on a **2-hour** "season" of
   5-minute data: it will NOT crash, it will silently report a wrong number. Same for
   SZ_TAXI/electricity_15min (96), london (48), m5/wiki (7). The frozen decision below
   ("`seasonal_m` is centralized per dataset") is not implemented. Safe to land: all seven
   current datasets map to 24, so a per-dataset table is **bit-identical** on the committed
   numbers — which is exactly what the pinning test must assert.
4. **NEW — two duplicate rosters, not one.** `experiments/run_cka_analysis.py:71` keeps its own
   hardcoded `PT_ID_TAGS` **and** (line 72) its own `SHORT` display-name dict. Both must be
   reconciled with the registry, not just the tag set.

## Run order

1. Stage data on the LOGIN node (it has internet; compute nodes do not) — see the screen's header.
2. `salloc`/`sbatch` the yield screen (COMPUTE NODE: multi-GB raw series, sustained core,
   millions of origins). Report realized train/val/test, rejection fraction, eligible series,
   denominator failure rate, degeneracy MAD.
3. ~~**STOP. Roster approval gate.**~~ **PASSED 2026-09-21** — roster frozen at the 14 above.
4. ~~Land the FOUR blockers.~~ **DONE** — see the implementation record below.
5. ~~The common dataset registry + removal of the global PT-ID/PT-OOD logic.~~ **DONE.**
6. **NEXT: the window-only smoke** (compute node, no GPU, no model):
   `python -m experiments.window_smoke14 --json $SCRATCH/window_smoke14.json`
   then, deliberately, `--create-references` once the counts are approved.
7. Only after that: GPU extraction.

---

# Registry / 14-dataset refactor — IMPLEMENTED 2026-09-21

32 model-free contracts in `tests/test_dataset_registry.py`; the whole existing suite still
passes (the only red is `test_ood_targets`, which needs the staged OOD shards and fails
identically on the pre-refactor tree).

## New modules

| module | role |
|---|---|
| `probing/registry.py` | **THE** dataset facts: tag, display name, slug, source repo/config, target column, freq, domain, `seasonal_m`, cluster unit, builder, `max_series`, role, roster + the 2-D provenance table. Stdlib only; every lookup raises on an unknown tag. |
| `probing/mase.py` | THE reported MASE. `seasonal_denominator(X, m)` with `m` **required**; `denominator_for(tag, X)` resolves it from the registry. |
| `probing/windows.py` | THE window dispatch, on `registry.builder(tag)`. Torch-free. |
| `probing/window_reference.py` | The create/reference workflow (below). |
| `experiments/window_smoke14.py` | Window-only smoke: real builders, real data, no model. |

## What changed, per blocker

1. **Series cap** — `id_data.apply_series_cap(tag, ids, seed)`: seeded uniform sample without
   replacement over the **sorted** eligible ids (never Arrow order, never `[:n]`), applied after
   eligibility and before any split construction, recorded in `meta["series_cap"]` with the kept
   ids. It draws from a **dedicated RNG stream** (`default_rng([seed, 0x5E12E5])`), so when it
   does not fire it consumes no randomness and the windows are **byte-identical** to the
   pre-cap builder — verified directly against `git show HEAD:probing/id_data.py`.
   Fires for london (5555), m5 (28491), wiki (100000); inert for the original seven.
2. **`domain_status`** — no longer gates anything. `PT_ID_TAGS`/`PT_OOD_TAGS` are now DERIVED
   from `registry.PAPER7` (identical tuples, identical order); `domain_status` is kept for the
   committed seven and **raises with a pointer** for anything else, so a new dataset can never
   acquire a Chronos-2-relative `pt_ood` label. `tunnel_record` gained `model=` and always
   emits `pretraining_provenance` (2-D); it emits the legacy flat `domain_status` key ONLY for
   the seven. **The tunnel criterion itself is untouched.**
3. **`seasonal_m`** — the global `M_SEASON = 24` is GONE from both modules that had it. All 17
   denominator call sites now pass `seasonal_m(tag)`. Bit-identity for the seven is a test, not
   a claim (test 11 compares against a verbatim transcription of the old function).
4. **Duplicate rosters** — `run_cka_analysis`, `make_erank_stability_figure`,
   `run_compression_cost`, `run_native_head_adapter`, `run_spectral`, `run_ptood_probing`,
   `run_ptood_probing_ftok`, `run_ft_specialization`, `run_ood_transfer`,
   `make_id_paper_figures` and both TimesFM `SLUG` dicts now read the registry. Test 24 greps
   the tree and fails if any of them comes back. This also **reconciled drifted spellings**:
   "Uber"→"Uber TLC", "WindFarms"→"Wind Farms", "SG-Carpark"→"SG Carpark" (labels only).

## Three things found while implementing

- **`SZ_TAXI_15T` would have crashed.** It has 156 eligible series against a 262 val/test
  budget, and `_build_rolling_windows` *raised* below the budget. The budget is now a
  **ceiling**, matching what the OOD builder already did, and the reduction is printed and
  recorded in `meta["valtest_budget"]`. It is a real loss of bootstrap units → wider CIs for
  that dataset, and must be read that way.
- **`load_seen_series` hardcoded `autogluon/chronos_datasets`.** Two roster datasets
  (LOOP_SEATTLE_5T, SZ_TAXI_15T) live in `autogluon/fev_datasets`. It is registry-driven now,
  and cross-checks against the legacy `ID_DATASET_SPECS` entry so the two tables cannot drift.
- **`probing/__init__.py` imported torch eagerly**, so even `probing.registry` needed the whole
  DL stack. The heavy names are lazy (PEP 562) now — identical spellings, deferred import —
  which is what lets the window smoke run with no torch at all.

## The create/reference workflow (blocker 8)

A dataset with no committed Chronos-2 artifact used to report `parity_ok=None`, which reads
like a pass. Now `--reference-mode`:

| mode | behavior |
|---|---|
| `require` (default) | **ABORTS**, naming the command that would create the reference. No GPU time spent. |
| `create` | writes `results/window_references/window_reference__<tag>.{json,npz}` from this run's windows; **refuses to overwrite** without `--force-reference`. |
| `allow-missing` | the old permissive behavior, now opt-in, and it prints a warning that the run must not be used for a cross-model claim. |

A written reference pins counts, C/H, seasonal m, the cap audit, the per-window test ids
element-wise, **and a sha256 digest of the test contexts** — which catches a re-windowing that
preserves the ids, something the committed Chronos-2 artifacts cannot detect.

## Provenance: 9 of 42 cells are UNVERIFIED

`registry.unverified_provenance()` lists them. They are usable as working assumptions but must
NOT be cited until checked against the primary source:

- `LOOP_SEATTLE_5T` / `SZ_TAXI_15T` x {chronos2, timesfm3} — is each a *fev-bench task*, or
  merely a member of the `autogluon/fev_datasets` collection? The exclusion statement only
  covers the former.
- `m5` x {chronos2, timesfm3, tirex} — confirm Chronos Benchmark II / fev-bench / Chronos-ZS
  membership.
- `wiki_daily_100k` x timesfm3 — the card names the SOURCE ("Wikipedia Pageviews, cutoff Nov
  2023"), not this dataset.
- `monash_traffic` x chronos2 — our Table 6 transcription ends in "a.o." and does not settle
  whether "Traffic" is an entry. **Check the published Table 6.**

Also recorded deliberately: `uber_tlc_hourly` x chronos2 is `pretraining_exposed` at
**`source_family`** scope, not `exact_dataset` — Table 6 lists "Taxi" (NYC TLC), and Uber TLC is
the same source family, not a verbatim entry. That is a downgrade from the old flat `pt_id`
label and is exactly what the 2-D schema exists to express.

---

# PHASE 1 FROZEN — 14 datasets x 3 models = 42 cells (implemented 2026-09-21)

The only two Phase-1 questions:

> 1. At what depth does forecasting become RECOVERABLE?
> 2. How does that functional transition compare with representation GEOMETRY?

Four measurements and nothing else: layerwise **Q=9** forecasting probes, the **5% VALIDATION**
tunnel entrance, **unbiased linear CKA** (biased kept as a diagnostic), **entropy effective
rank**. NO alignment, NO adapters, NO truncation — those are later phases and nothing in the
Phase-1 code anticipates them.

## Files (new unless marked)

| file | role |
|---|---|
| `probing/phase1.py` | the contract: canonical quantiles + cross-model verification, common loss, per-model representation-point table, tunnel record, geometry block, cluster CIs, cell config hash, atomic `CellStore`, manifest/snapshot helpers |
| `probing/phase1_metrics.py` | raw-unit MASE / MAE / WQL, identical for all three models |
| `probing/phase1_cells.py` | the cell artifact contract (pure numpy in, files out — testable with no model) |
| `probing/phase1_{chronos2,timesfm3,tirex}.py` | the three adapters |
| `probing/window_parity.py` | the parity check, PROMOTED out of the TimesFM driver so all lines call one copy |
| `experiments/run_three_model_phase1.py` | the orchestrator |
| `experiments/make_phase1_tables.py` | combined tables, rebuilt from cell artifacts ONLY |
| `tests/test_three_model_phase1.py` | 49 model-free contracts + 3 delegated model-backed |
| `job_three_model_phase1.sh` | SLURM, 1 GPU, 12 h, resumable |
| MODIFIED `probing/tirex_probes.py` | generalized to Q>=1; Q=1 is BIT-identical (contract 26) |
| MODIFIED `probing/timesfm3_last_token_probes.py` | additive `collect_probe=` to capture the frozen probe + full predictions |
| MODIFIED `experiments/run_timesfm3_last_token_probing.py` | parity delegated to `probing/window_parity.py`; its own tests unchanged and passing |

## The protocol

C=512, H=64, **Q=9 = [0.1 .. 0.9]** — read from each model's OWN module and compared
element-wise at startup (`assert_canonical_quantiles`); the run ABORTS if they ever disagree.
Reported loss = **mean pinball over (batch, quantiles, horizon)** for all three, computed from
the one saved `(n, Q, H)` prediction tensor. Tunnel = `probing.tunnel.tunnel_start`, tol 0.05,
**validation only**; 0.01/0.02/0.05/0.10 all saved so no re-run is ever needed for another
tolerance.

| | readout | probe | DEPTH AXIS | also probed (not a depth) |
|---|---|---|---|---|
| Chronos-2 | K=4 native forecast slots | ONE shared `Linear(768, 9*16)` | 13: Emb, L1..**L12** | `L12+LN` |
| TimesFM-3 | last REAL context token (15) | `Linear(1280, 64*9)` per depth | 21: Emb, L1..**L20** | — |
| TiRex | `two_pass`, token 63 of each pass | ONE shared `Linear(512, 9*32)` | 13: Emb, L1..**L12** | `L12+RMS` |

**Correction made while implementing:** Chronos-2's final point was briefly labelled `L12+RMS`.
It is a **LayerNorm** (`encoder.final_layer_norm`), and `L12+LN` is the spelling the committed
Chronos-2 fslot line already uses. TiRex is the model with an RMSNorm.

## Three things a reader of the results must carry (in every `summary.json`)

1. **Absolute Q=9 loss is NOT cross-model comparable.** Each model is probed where its own head
   reads, in its OWN normalized target space (Chronos-2 arcsinh; TimesFM-3 detrend+RevIN; TiRex
   per-pass loc/scale). What IS comparable: the tunnel entrance (a within-model ratio against
   that model's own final depth) and MASE (raw units, one shared in-context denominator).
2. **The three lines keep their own TRAINING objectives** (Chronos-2 sums over quantiles;
   the others average). Measured, not assumed, to change nothing: the two differ by exactly
   2Q, so the wd argmin and the tunnel ratio are invariant (contracts 16, 49), and AdamW is
   invariant to a constant loss rescale — 18x moves the fitted weights by **1.4e-7 relative**
   (contract 48).
3. **`not_listed` is not OOD.** No Phase-1 artifact contains a global PT-ID/PT-OOD field;
   contract 13 proves it by AST over every Phase-1 module AND by scanning a real written cell
   plus all combined tables.

## Resumability

A cell is built in `<model>/<dataset>.building-<pid>/` and `os.replace`d into place only after
every required artifact validates and `COMPLETE` is written inside the staging dir. So a
preempted/timed-out/failed cell leaves NO partial cell — only a staging dir the next run
deletes. `sbatch job_three_model_phase1.sh` twice = resume (verified: 6/6 built, then 6/6
skipped). A cell whose config hash DIFFERS is **refused**, naming `--force-recompute` or a new
`--output-root`. Built windows are cached to `$SCRATCH` so a resume does not rebuild them.

## Bugs this work surfaced

- `--audit-only` set `args.models = []`, but the driver's filter reads an empty list as "no
  filter" — it would have loaded all three backbones under a flag that promises not to. Now an
  explicit `if args.audit_only: models = []`.
- Mixed `int`/`"final"` feature keys broke `fit_shared_forecast_probe_explicit_val`'s
  `sorted(train_feats)`. Chronos-2 now uses integer key 13 = `NUM_LAYERS`, which is also the
  committed `run_ptood_probing_ftok` convention.

## Open before/at the first GPU run

- **Wall time is NOT measured.** 12 h requested; the estimate must come from the STEP-3 smokes
  (read `cells.json`). Window building dominates a cold run.
- The seven NEW datasets have **no window reference** — STEP 2 of the job header creates them
  deliberately, and they must be inspected and committed before the full run.
- **9 of 42 provenance cells are UNVERIFIED** (`registry.unverified_provenance()`); usable as
  working assumptions, not citable.
- Watch for weight-decay grid-max clipping per (model, dataset, depth) — reported per cell as a
  decision to make, never silently accepted.
- Durable results ~2 GB (probe weights dominate; `--no-probe-artifacts` drops ~1 GB).


## Depth-axis correction — a final normalization is NOT a model depth (2026-09-21)

**The rule.** The main depth axis is **Emb, L1..L_N** — block outputs only. The final depth,
and therefore the tunnel's reference and the meaning of "final-depth probe loss", is the final
**BLOCK** output: **L12** (Chronos-2), **L20** (TimesFM-3), **L12** (TiRex). A final
normalization adds no block and no residual-stream step, so counting it as a depth would put a
15th point on a 14-point axis, let a norm define the final-depth loss, and shift every
normalized depth.

`L12+LN` (Chronos-2, `encoder.final_layer_norm`) and `L12+RMS` (TiRex) are still **probed,
scored, CKA'd, rank'd and saved** — they are what the native head literally reads and will be
wanted for the native-head/alignment section — but as **`point_type = head_input_diagnostic`**.
TimesFM-3 has none: its head reads L20 directly, so L20 is both.

**Enforced structurally, in four places:**

| | before | now |
|---|---|---|
| tunnel | scanned all points, reference = last point | `tunnel_record` scans `spec.depth_indices` only; reference `spec.reference_index` = final block; excluded points and their losses recorded beside the result |
| normalized depth | norm shared `relative_depth = 1.0` with the last block | `relative_depth` / `relative_position` are **`None`** off the depth axis; final block is exactly 1.0 |
| geometry | one CKA matrix over all points | `cka_<est>_<variant>_<split>.npy` is the **depth axis only**; the full matrix is a separate `__with_head_input.npy`. CKA is pairwise, so the main matrix IS the submatrix — computed once, cannot disagree. Effective rank keeps the point (cheap diagnostic) but never marks it headline |
| serialization | — | `point_type` + `include_in_main_depth_axis` on every point, row and record; `plot_data.csv` is depth-axis only and diagnostics go to `plot_data_head_input.csv`, so a main figure would have to open a differently-named file to get one |

`ModelSpec.last_index` (last probed point) is **gone**, replaced by `reference_index` (final
block) and `native_readout_index` (what the head reads). "The head reads this" and "this is a
model depth" are now two separate, separately-named claims — conflating them is what put a
normalization on the depth axis.

**Contracts 53-57** pin it: closed `point_type` vocabulary and per-model assignment; a
head-input loss scaled by 1e-6 or 1e6 moves neither the entrance nor the reference at **any**
tolerance; normalized depth is defined over block depths only; the main CKA matrix equals the
submatrix of the with-head-input one and contains no diagnostic row/column; and every written
artifact carries `point_type` with `plot_data.csv` provably free of head-input rows. Contracts
35/40/41/42/47/50 were updated to depth-axis coordinates. **54/54 model-free contracts pass**;
the full existing suite is unchanged.

**One fixture bug this surfaced:** contract 54 first asserted that a linear ramp enters at the
final depth. It does not for TimesFM-3 — with 21 points the second-to-last lands *exactly* on
the inclusive `(1+tol)*final` boundary. The fixture was wrong, not the criterion.

---

# PHASE 1 RERUN — expanded WD grid + SUSTAINED tunnel, in two runs (2026-09-22)

Two clean, separate configurations, both from ONE driver. The earlier run stays put:
`results/three_model_final/` is the audit trail and is never written to again (the driver's
default output root moved off it, so a bare invocation cannot touch it).

| | EXPERIMENT A (headline) | EXPERIMENT B (robustness) |
|---|---|---|
| quantiles | **Q=9** [0.1 .. 0.9] | **Q=1**, tau=0.5 |
| output tree | `results/three_model_phase1_q9_expanded/` | `results/three_model_phase1_q1/` |
| job | `job_three_model_phase1_q9_expanded.sh` | `job_three_model_phase1_q1.sh` |
| role | the canonical Phase-1 headline | probe-capacity / objective robustness |

Everything else is IDENTICAL: 14 datasets, the same windows/splits, C=512, H=64, the same
representation points and final-block references, the same probe architectures, optimizer,
300 epochs, lr 1e-2, seeds, bootstrap, MASE/MAE/WQL, CKA, effective rank, provenance,
checkpoints. **Q=1 is NOT a replacement for Q=9 and a Q1/Q9 difference is NOT by itself
evidence of overfitting.**

## Three changes, and nothing else

### 1. The weight-decay grid — ONE grid, all three models, both Q

    old chronos2   1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3                 (max 3)
    old timesfm3   ... + 10 30                                       (max 30)
    old tirex      ... + 10 30                                       (max 30)
    NEW all three  1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3 10 30 45 65 90   (max 90)

`probing.phase1.PHASE1_WD_GRID`. The three model lines' OWN grids (`probes.WD_GRID_V2`,
`WD_GRID_LAST_TOKEN`, `WD_GRID_TIREX`) are **untouched**, so no committed non-Phase-1 result
moves; Phase 1 asserts the new grid is a strict superset of each.

**Why it grew — measured on the committed Phase-1 cells, not predicted:**
* `timesfm3 x monash_electricity_hourly`: 16 of 21 depths at the old max 30, validation STILL
  falling there (L19: 0.12419 at wd=10 -> 0.11677 at 30).
* `chronos2 x m4_hourly`: 9 of 14 at ITS max 3.0. **The Chronos-2 grid was the NARROWEST of the
  three and clipped hardest** — the old comment in `phase1_chronos2.py` claiming "the committed
  runs do not clip" was falsified and has been corrected in place.
* `tirex x monash_electricity_hourly`: 0 clipped; its optimum is interior at wd=10.

**Why it stops at 90, not 100/300/1000** — the spec guessed "100, 300, 1000"; that is
mathematically unavailable. AdamW's decay is DECOUPLED: each step multiplies the weight by
(1 - lr*wd). At lr=1e-2, measured on probe-shaped synthetic features, 300 epochs:

| wd | lr*wd | max\|W\| | val | |
|---|---|---|---|---|
| 30 | 0.30 | 3.6e-2 | 0.3820 | regularized |
| 45 | 0.45 | 2.7e-2 | 0.3670 | regularized |
| 65 | 0.65 | 2.0e-2 | 0.3597 | regularized |
| 90 | 0.90 | 1.5e-2 | 0.3568 | **selection ceiling** |
| 100 | 1.00 | 1.4e-2 | 0.3564 | weight zeroed EVERY step -> bias-only: the NULL, not a regularizer |
| 300 | 3.00 | inf | nan | \|1 - lr*wd\| > 1 -> diverges |

And the direction is confirmed by the real data: on the committed TimesFM-3 Electricity cell the
wd=100 null beats wd=30 at only 3 of 21 depths, so **the optimum sits INSIDE (30, 100)** —
exactly what 45/65/90 covers.

**What a grid-max selection MEANS now.** The val curve is smooth into the null, so the new
maximum sits within ~1e-3 relative of the no-information floor. "At grid max" therefore has two
readings, and the code refuses to conflate them: every cell computes the closed-form
`constant_forecast_floor` (pinball-optimal constant = per-step train quantiles; no fit, so no
optimizer artifact can reach it) and reports `val_loss / floor`. Ratio ~1 = **this depth's
optimum IS the floor (a finding)**; ratio well below 1 = the search really was cut off.

### 2. The tunnel — SUSTAINED ENTRY, not first crossing

    E_l             = max_{j >= l} ( R_j / R_L - 1 )        # j over BLOCK DEPTHS only
    l_tunnel(delta) = min { l : E_l <= delta }

`probing.tunnel.sustained_tunnel_start`. `tunnel_start` (first crossing) is UNCHANGED and still
called — its value is saved as `first_crossing_<tol>` and is never called the tunnel.

Properties: an isolated early crossing no longer opens a tunnel; the excursion inside the tunnel
is bounded by tol BY CONSTRUCTION; and the entrance is monotone in the tolerance
(`depth(10%) <= depth(5%) <= depth(2%)`) as a **theorem**, asserted as a regression guard.
Boundary handling is byte-identical to first crossing (`v[j] <= (1+tol)*v[last]`, not
`v[j]/v[last]-1 <= tol`) so `sustained >= first_crossing` holds for EVERY curve — verified over
120k random curves.

### 3. Tolerances 2% / 5% / 10% (5% headline), 1% saved for free.

## The hash is now TWO halves

`probing.phase1.split_cell_config`: **FIT** (checkpoint, extraction params, wd grid, epochs, lr,
seed, probe architecture + width, windows, bootstrap B) and **POSTPROCESS** (tunnel definition,
headline tolerance, tolerance set). `cell_config_hash` = digest of both, so it keeps its old
meaning — Q1/Q9 can never collide and a narrow-grid q9 cell can never satisfy an expanded-grid
q9 run. But a changed TUNNEL DEFINITION now reports `stale_postprocess` instead of
`incompatible`, and `python -m experiments.rebuild_phase1_tunnels` re-derives it from artifacts
already on disk — **no GPU, no refit**. The driver refuses to proceed and names that tool rather
than silently refitting or silently keeping stale numbers.

## Compute reuse — the two runs SHARE one feature cache

Extraction and window building never see a quantile vector (verified by contract: no cache
metadata function takes Q). Both job scripts point at
`$SCRATCH/chronos2/phase1_shared_cache` and `$SCRATCH/chronos2/phase1_shared_windows`, so the
three backbones run ONCE for both experiments and the Q1 job is probe-fitting only. Geometry
(CKA / effective rank) is Q-independent and is cheaply RECOMPUTED from those same cached states
rather than copied; `make_phase1_q1_q9_comparison` then VERIFIES the two runs' matrices are
identical element-wise (0.0 expected) instead of assuming it.

## Files

| new | role |
|---|---|
| `experiments/rebuild_phase1_tunnels.py` | re-derive tunnels from saved artifacts, no GPU |
| `experiments/make_phase1_q1_q9_comparison.py` | the cross-run tables + the geometry-identity check |
| `tests/test_phase1_wd_and_sustained_tunnel.py` | 17 lettered contracts (A-O) |
| `job_three_model_phase1_q9_expanded.sh`, `job_three_model_phase1_q1.sh` | the two sbatch wrappers |
| `writing/phase1_q9_q1_methodology.md` | the paper wording (gitignored) |

| modified | what changed |
|---|---|
| `probing/tunnel.py` | ADDED `suffix_excursion` / `sustained_tunnel_start` / `assert_tolerance_monotone`; `tunnel_start` untouched |
| `probing/phase1.py` | `PHASE1_WD_GRID` + guard, `TUNNEL_DEFINITION_VERSION`, sustained `tunnel_record`, `constant_forecast_floor`, `wd_selection_rows`, the hash split, `CellStore.stale_postprocess` |
| `probing/phase1_{chronos2,timesfm3,tirex}.py` | `WD_GRID` -> the shared grid + superset assertion + guard + the floor |
| `probing/phase1_cells.py` | `wd_selection.csv` (now REQUIRED), ratio/excursion columns, richer clipping block |
| `experiments/run_three_model_phase1.py` | new default root, `--tunnel-tols` / `--wd-grid`, two-half hashes, the clipping warning that reads the floor |
| `experiments/make_phase1_tables.py` | tunnel matrices per tolerance, sensitivity, wd summary |
| `tests/test_three_model_phase1.py` | 33 and 35 updated — they encoded the old entrance; both statistics now pinned |

## Run order (Narval)

1. login node: `python -m tests.test_three_model_phase1` (57) and
   `python -m tests.test_phase1_wd_and_sustained_tunnel` (17).
2. GPU smoke — **`timesfm3 x m5` at Q=9 first**, at the REAL 300 epochs (wd selection depends on
   the step count, so a shortened smoke measures a different optimization problem). Then a cheap
   shape/device smoke for the other two models at both Q, where `--probe-epochs 30` IS fine.
   Exact commands are in the header of `job_three_model_phase1_q1.sh`.
3. `sbatch job_three_model_phase1_q9_expanded.sh` (fills the shared cache), then
   `sbatch job_three_model_phase1_q1.sh`. Resume either by resubmitting the identical line.
4. login node: `python -m experiments.make_phase1_q1_q9_comparison`.

## Open / to check on the first run

- **Is the expanded grid STILL clipped?** Read `combined/wd_selection_summary.csv`. If depths
  still sit at 90, read `at_grid_max_val_over_floor`: ~1.0 is a finding, well below 1.0 means
  report it before another rerun (the grid cannot be widened at this lr — 100 IS the floor).
- Whether any depth selects the grid MINIMUM (1e-5) widely — the grid would then need extending
  DOWNWARD, which has no stability wall and is cheap.
- How far the sustained entrance sits behind first crossing per cell
  (`sustained_minus_first_crossing` in `tunnel_sensitivity.csv`) — a large gap everywhere means
  the curves are non-monotone and the old headline was fragile.
- Whether the Q1 and Q9 entrances agree (exact / within 1 / within 2 blocks, per model) —
  descriptive only; do NOT label a difference "overfitting" without the train/val/test evidence.
- Wall time is STILL not measured end to end. The expanded grid adds 3 of 13 candidates = ~30%
  more probe fits per depth. Read `cells.json` after the smoke before trusting `--time=12:00:00`.
- **Pre-existing, not caused by this work:** `tests/test_q1q9_rerun.py::
  test_wide_wd_grid_value_and_sharing` fails under a FULL pytest run and passes alone —
  `tests/test_ft_specialization.py:301` sets `rfs.WD_GRID = (1e-3, 1e-2)` on the shared module
  and never restores it. Confirmed identical on a pristine `git archive HEAD` tree.

---

# PHASE 2 v3 (H3 functional replaceability + H4 truncate-to-accelerate) — CODE IMPLEMENTED 2026-09-23, NOT RUN, NOTHING SUBMITTED

Full design: `writing/phase2_design.md` (v3: flags F1–F23, compute, launch plan); LaTeX:
`writing/phase2_reproducibility.tex` (v3). **v2 is SUPERSEDED** (the three-hypothesis "H2 = CKA- vs
tunnel-guided depth selection"): the selector comparison is a deferred, secondary H4 analysis
(design §12) that must not block H3/H4.

**Story:** recoverable early (H1, Phase 1) → geometry still evolves (H2, Phase-1 geometry;
descriptive, NOT a pruning experiment) → alignment makes early states usable (H3) → remove later
blocks and accelerate (H4). H3 is the bridge between probing and compression — never collapsed into
the latency experiment.

| stage | what | entry point |
|---|---|---|
| H3 | per model x dataset x EVERY block depth, offline from the Phase-1 caches (READ-ONLY): native \| hard cut \| NOA (label-free, primary) \| FL (Chronos-2, TiRex; TimesFM-3 FL == probe by the rank theorem, contract 9) \| Phase-1 probe (reused, never refit) \| RA (hidden ridge, diagnostic) | `experiments/run_phase2_h3.py` |
| H4 | physical truncation at the FROZEN Phase-1 sustained-5% entrance: native, hard, NOA, FL (C2/TiRex), probe-head, Chronos-2-small; gates V1–V6 + PH + AP; then latency / throughput / memory | `run_phase2_truncation.py` (verify, evaluate), `run_phase2_latency.py` |

**Protocol** (`probing/phase2.py`, `phase2/h3-v1`):
- **Adapter:** residual `x + Δx + b`, initialized at Δ=b=0, which is the hard cut bitwise. Decay applies to Δ only, i.e. toward the IDENTITY.
- **Iterative fits:** full-batch AdamW, lr 1e-2, 300 epochs, validation every 10 epochs; epoch 0 = the hard-cut candidate. wd grid {0, 1e-3, 1e-2, 1e-1, 1, 10}.
- **Closed form:** TimesFM-3 NOA in the metric W^T W; RA for all models. λ = κ·g_max·m_max, κ ∈ 1e-12..1e2 (15 values), plus an explicit hard-cut candidate.
- **Selection** on validation only: NOA distillation MSE, RA hidden MSE, FL Q9 pinball. Test is read once.
- **Metrics:** test MASE and standard 1/Q WQL, as degradation vs native. Paired cluster bootstrap, B=5000, seed 0.
- **Flags:** 5% budget with a fragile flag; gap closure only where the hard-gap CI excludes 0; low-skill when native val ≥ 0.9 x floor.
- **Gates:** native-reproduction rtol 1e-4; window digest and cluster ids must match the Phase-1 cell.
- **H4 OUTCOME RULE** (pre-registered 2026-09-23; `phase2.H4_OUTCOME_RULE`, `h4-outcome/v1`):
  - candidate_depth = the frozen Phase-1 sustained 5% entrance (one depth per model x dataset).
  - successful_truncation = the NOA (label-free aligned) test-MASE degradation vs the full native model ≤ 5% (point estimate). `fragile` = the CI contains 5%.
  - **If false, the FAILURE is reported. No other depth is ever searched, evaluated or substituted.**
  - Statuses: `success` | `failure` | `no_truncation_possible` (entrance = final block: TiRex x M4 and Traffic) | `invalid_v3` (physical != offline: a pipeline error; both H4 tables refuse to compile).
  - `evaluate` refuses `--depths` and aborts if the H3 depth is not the Phase-1 entrance.
  - Every failure is a row in `h4_outcomes.csv`, a count in `h4_outcome_summary.csv` / `phase2_stats.json`, a column in Table H4-2 (`h4_outcome_table.tex`) and a `\phtwo<model>HfourFailure` macro.
  - The H3 compatibility tunnel is never an H4 operating point.
- **Aggregates exclude vacuous cells:** H3 Table H3-1 and H4 Table H4-1 aggregate only the datasets whose entrance lies BEFORE the final block. Cells with entrance = final block are counted separately (F24).

**Files (ALL NEW; no Phase-1 file modified.** `git status`: the only tracked changes are `.gitignore`, your own edit, and this file.)
- `probing/`: `phase2.py`, `phase2_align.py`, `phase2_pathways.py`, `phase2_h3.py`, `phase2_truncate.py`, `phase2_env.py`
- `experiments/`: `run_phase2_h3.py`, `run_phase2_truncation.py`, `run_phase2_latency.py`, `make_phase2_tables.py`, `make_phase2_paper_tables.py`
- `tests/`: `test_phase2_h3.py` (25 model-free contracts, incl. the H4 outcome rule), `test_phase2_h4.py` (28, on tiny real architectures)
- jobs: `job_phase2_h3.sh`, `job_phase2_h4.sh`, `job_phase2_latency.sh`

**Frozen Phase-1 entrances** (sustained 5%, read by `--plan` 2026-09-23; LOOP from the lr=1e-3 tree):

| dataset | Chronos-2 | TimesFM-3 | TiRex |
|---|---|---|---|
| m4_hourly | L8 | L16 | **L12 = final** |
| monash_electricity_hourly | L9 | L15 | L11 |
| uber_tlc_hourly | L4 | L1 | L10 |
| wind_farms_hourly | L3 | Emb | L11 |
| sg_carpark | L10 | L12 | L11 |
| coastal_ts | L1 | L5 | L1 |
| boom_hourly | L3 | Emb | L8 |
| LOOP_SEATTLE_5T | L9 | L14 | L9 |
| electricity_15min | L10 | L1 | L10 |
| SZ_TAXI_15T | L2 | L1 | Emb |
| monash_london_smart_meters | L3 | L1 | L8 |
| m5 | Emb | Emb | Emb |
| wiki_daily_100k | L2 | Emb | L1 |
| monash_traffic | L11 | L12 | **L12 = final** |

At TiRex x {M4, Traffic} the frozen entrance IS the final block, so H4 there is the native model
(speedup 1, degradation 0 by construction) — a legitimate outcome. Seven cells enter at Emb (H4
removes ALL blocks): the sharpest test of "recoverable != replaceable".

**Verified** (Mac scratch venv, CPU, NO real checkpoints):
- 25 + 28 contracts pass.
- The offline f(h_L) matches each package's own forecast: 3.5e-6 (Chronos-2), 2.4e-5 (TimesFM-3), 6.6e-6 (TiRex).
- V1, V2 and V5 are bitwise. V3 (physical == offline) is within 8.5e-6 / 3.5e-5 / 1.1e-5.
- V4: 0 calls, 0 alive. V6: 2 passes. PH ≤ 1.2e-5. AP is exact.
- The tiny verify driver passes for all three models.
- The tiny latency harness works end to end: fresh processes, refusal logic, eligibility.
- `--plan` on the real Phase-1 tree reports 42/42 COMPLETE.

**NOT verified:** anything on the real checkpoints or caches (needs Narval). The smokes below are the first real contact.

**Fixed during the final check (2026-09-23):**
- F21: V6 was skipped at depth 0, although Emb is the TiRex entrance for SZ Taxi and M5.
- F23: the latency tables would have averaged the smoke / unverified / killed jobs into the headline. Now only headline-protocol jobs count, the rest are reported, and job tags are one-shot.
- F24: cells whose entrance IS the final block counted as a vacuous "within 5%" in the H3/H4 aggregates. They are now excluded and counted separately.
- F1 CLOSED (user decision 2026-09-23): `writing/` does NOT have to be gitignored. Do not flag it again.
- F25: concurrent H3 jobs could delete each other's in-progress cells (an unscoped stale-staging sweep) and overwrite the shared `cells.json`. Cleanup is now scoped to the job's own cells, and each job writes to its own `h3/runs/<SLURM_JOB_ID>/`.
- F26: H4 `evaluate` now resumes (COMPLETE cells skipped; the cell key includes the H3 config hash).

## Run order (Narval) — the user submits; nothing has been submitted

**Queue policy (2026-09-23): every Phase-2 job requests ≤ 3 h.**
- On Narval, ≤ 3 h is the shortest scheduler time tier: the most nodes, plus backfill.
- The old 8 h / 12 h requests put jobs in a slower tier. Phase 1 actually needed 15–27 min of compute per model (measured from the Phase-1 `summary.json` timings).
- Every job resumes (H3 and H4-evaluate skip COMPLETE cells; latency runs one model per job), so a timeout only means resubmitting the same line.

0. **Login node** (seconds, `export OMP_NUM_THREADS=2`):
   - `python -m tests.test_phase2_h3`
   - `python -m tests.test_phase2_h4`
   - `python -m experiments.run_phase2_h3 --plan`
   - Chronos-2-small, download only: `HF_HOME=$SCRATCH/chronos2/hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('autogluon/chronos-2-small')"`
1. `sbatch --time=00:45:00 -J p2h3-smoke job_phase2_h3.sh --smoke`
   (Electricity x 3 models, 4 depths, real 300 epochs, B=500; output `results/three_model_phase2_smoke/`)
2. `sbatch --time=01:00:00 -J p2h4-verify job_phase2_h4.sh verify --cache-dataset monash_electricity_hourly`
3. `sbatch --time=01:00:00 -J p2lat-smoke job_phase2_latency.sh --depths 3 12 --reps 5 --warmup 2 --job-tag smoke --allow-unverified --output-root results/three_model_phase2_smoke`
4. **STOP and review.**
   - Native gates, identity checks, V1–V6 (and V1 vs the cache), timing sanity, and the TimesFM-3 API overhead (F22).
   - **Size the full runs:** `seff <jobid>`; full H3 minutes per model ≈ the smoke cell's `timings.cell_total_s`/60 x 50 (Chronos-2, TiRex) or x 70 (TimesFM-3).
5. Full H3, one job per model; they may run at the same time (F25 fixed):
   - `sbatch -J p2h3-chronos2 job_phase2_h3.sh --models chronos2`, and the same for `timesfm3` and `tirex`.
   - If a model is predicted > ~2.5 h, split it into two `--datasets` halves.
6. H4 evaluate, one job per model: `sbatch --time=01:00:00 -J p2h4-eval-chronos2 job_phase2_h4.sh evaluate --models chronos2`, and the same for `timesfm3` and `tirex`.
7. Latency, 9 short jobs: `for r in 1 2 3; do for m in chronos2 timesfm3 tirex; do sbatch --time=02:00:00 -J p2lat-$m-$r job_phase2_latency.sh --models $m --job-tag job${r}_$m; done; done`
8. **Login node:** `python -m experiments.make_phase2_tables && python -m experiments.make_phase2_paper_tables`

**Cost** (Phase-1-calibrated estimates; the smokes measure the rest):

| stage | estimate |
|---|---|
| smokes | ≈ 1 A100-h |
| H3, Chronos-2 | ≈ 1.3 A100-h (0.7–2.5) |
| H3, TiRex | ≈ 0.75 A100-h |
| H3, TimesFM-3 | ≈ 0.75 A100-h (the closed form was MEASURED at ~6 s per depth on CPU) |
| H4 verify | ≈ 0.3–0.5 A100-h |
| H4 evaluate | ≈ 0.5–1 A100-h |
| latency, per model per repeat | Chronos-2 ~0.4, TimesFM-3 ~0.8, TiRex ~0.8 A100-h |
| latency, 3 repeats | ≈ 6 A100-h |
| **total** | **≈ 9–11 A100-h** |
| LODO (later) | re-estimate after H3 |

**Open:**
- The native-gate rtol 1e-4 is not calibrated on real data. If it is raised for a model, write the measured value down.
- F22: TimesFM-3 `predict_batch` has a fixed host cost (~41 ms api vs ~4 ms device on the tiny CPU model).
- The Phase-1 caches on `$SCRATCH` (09-21..23) should be archived to `$PROJECT` before the ~11-20 purge.
- **Deferred; do NOT build before H3/H4 exist:** the geometry-selector analysis, the Wiliński baseline, LODO.

## SMOKE RESULTS (Narval, 2026-09-23, commit 14af351) — full run ON HOLD

**Passed.**
- **H3 gates, all three models (Electricity).**
  - Native reproduced to ~1e-9 relative (gate 1e-4).
  - hard@L == native; the identity adapter == the hard cut (bitwise); the closed-form identity holds to 2e-8.
- **H4 verify.** All pass, at every depth, for all 3 models: V1 (incl. vs the Phase-1 cache on real windows), V2, V3, V4, V5, V6, PH.
- **Timings match the estimates.**
  - Chronos-2: ~2.1 s per adapter fit (26 s per depth).
  - TiRex: ~0.7 s per fit.
  - TimesFM-3: ~4 s per depth (closed form).

**H3 ladder at the frozen entrance** (test MASE / native; provisional, B=500):

| model (entrance) | hard | NOA | FL | RA | probe |
|---|---|---|---|---|---|
| Chronos-2 (L9, 3 blocks removed) | 1.344 | 1.172 | 1.133 | 1.144 | 1.245 |
| TimesFM-3 (L15, 5 blocks removed) | 216.9 | 1.442 | (= probe) | 2.847 | 1.150 |
| TiRex (L11, 1 block removed) | 2.170 | 1.073 | 1.082 | 1.112 | 1.304 |

- A compatibility gap exists for all three. NOA closes 50% / 99.8% / 94% of it.
- Nothing is within 5% at the entrance on Electricity.
- **Interpretation point:** the entrance is defined relative to the final-depth PROBE, which is itself 15–30% worse than the native model here (the probe column). Reaching 5% of *native* at the probe's entrance is a stricter bar than recoverability. The rule is NOT changed.

**Two blockers found.**
1. **Latency harness (F27, FIXED).** `device_forward` ran `np.asarray` on a CUDA tensor, so 28/28 configurations failed. Reproduced and fixed on the Mac GPU (MPS) with the real Chronos-2. Contract DF was added. The latency smoke must be rerun.
2. **Adapters NOT converged (OPEN; decided on VALIDATION curves only).**
   - AdamW at lr 1e-2 for 300 epochs (copied from the Phase-1 probe) is too aggressive and too short for a residual adapter around the identity:
     - validation spikes ~2x above the hard cut at epoch 10;
     - NOA's validation distillation MSE is still falling at epoch 300 (TiRex L11: -24..-29% over epochs 250->300 for wd <= 0.1, -6.5% at the selected wd = 1; Chronos-2 L9: -9..-10%);
     - selections sit at epochs 240–300 and often at the grid-max wd = 10.
   - The H4 verdict is decided exactly by that NOA number (TiRex +7.3% vs the 5% budget), so an unconverged fit would bias it.
   - The TimesFM-3 NOA is a closed form (converged by construction).

**Next (all GPU; compute nodes).**
- Rerun the latency smoke.
- An optimizer check on Electricity, Chronos-2 + TiRex:
  - lr 1e-3 x 1500 epochs (eval every 25);
  - lr 3e-3 x 1000 epochs (eval every 20);
  - both with wd grid {0, 0.1, 1, 10, 30, 100}.
- **Pre-stated decision rule (validation only, never test):** pick the configuration whose selected validation criteria (NOA distillation MSE; FL validation pinball) at the entrance depth are lowest AMONG configurations that converged. Converged means the selected epoch is below 90% of the maximum and the selected wd is not at the grid maximum.
- Then freeze the choice in `probing/phase2.py`, bump `PHASE2_PROTOCOL_VERSION`, (probably) add patience-based early stopping, rerun the H3 smoke, and launch the full run.

## H4 = THE ACCURACY–COMPUTE FRONTIER — DECIDED + IMPLEMENTED 2026-09-23 (before any full Phase-2 result)

**The H4 question:** how much forecasting accuracy survives a given reduction in inference cost?

**The paper's H4 claim to test:** TSFMs admit a broad accuracy–compute frontier well before full depth; lightweight alignment improves it over naive truncation; and a validation accuracy budget turns ONE pretrained checkpoint into a family of faster forecasters without retraining the backbone.

**Arc:** recoverable early -> geometry keeps evolving -> native pathway misaligned -> lightweight alignment -> accuracy–compute frontier from physical truncation.

**Main result 1 — Figure H4-1 (`h4_frontier.pdf`).**
- Three panels, one per model.
- y = test ΔMASE = MASE_trunc / MASE_native − 1 (median + IQR over 14 datasets).
- x = measured speedup (end-to-end, B=1).
- Appendix variants: throughput (B=256) and parameters removed.
- Curves: hard | NOA + native head | FL + native head (C2, TiRex) | probe head.
- The full model is at (1x, 0); Chronos-2-small is a point.

**Main result 2 — Table H4-1 (`h4_budget_table.tex`).**
- For eps in {2, 5, 10, 20}%, per model x dataset x arm, pick on **VALIDATION MASE** the shallowest SUSTAINED depth with MASE_val(j) <= (1+eps) * MASE_val(native) for all truncation depths j in [l, L-1].
  - This is `phase2.BUDGET_RULE` / `budget_depth`. If none qualifies, the operating point is the full model at 1x.
- The choice is frozen in the H3 cell's `frontier.json`.
- `run_phase2_truncation evaluate --operating-points budget` physically instantiates exactly those cuts, reads test once, and V3-checks the per-window MASE. A failed point blocks the table.
- The table reports the median speedup over ALL datasets, the datasets cut, and how many of those stay within budget on test.
- Macros support the paper sentence (X% removed, Y× speedup, Z/n within budget).

**Secondary — the frozen 5% entrance test** (`H4_OUTCOME_RULE`), unchanged. Its reading: recoverability alone does not identify a lossless cut.

**NOT assumed:** that alignment beats the probe head. It falls where it falls, per model. (Smoke: alignment > probe for C2 and TiRex; probe >> NOA for TimesFM-3.)

**What changed in the code** (protocol `phase2/h3-v2`, so all earlier H3 cells are invalidated):
- H3 records validation MASE for every arm and depth + native.
- The probe's validation MASE is computed from Phase-1 `predictions_val.npz` through the pathway inverse. A probe inverse gate checks it against Phase 1's own probe test MASE (1e-4), and rows are checked element-wise.
- New `frontier.json` per cell.
- New tables: `h4_frontier_points / h4_frontier_summary / h4_budget_points / h4_budget_summary / h4_budget_physical`.
- New figures `h4_frontier(_appendix).pdf`, and budget macros.
- Contracts 26–27. **27 H3 + 31 H4 contracts pass.**

**Next (unchanged order):**
1. Optimizer check (running) -> freeze the converged adapter settings.
2. Rerun the H3 smoke (the new cell schema).
3. Full H3 run.
4. H4: `evaluate --operating-points budget` (main) and `evaluate` (entrance, secondary).
5. Latency, 9 jobs.
6. Tables.

**Contingency** (only if the frontier's knee is weak after converged training): a small nonlinear residual adapter as a declared variant. It cannot recover the cross-token mixing of the removed blocks.

## OPTIMIZER CHECK + LATENCY SMOKE — RESULTS (2026-09-23, commit bf960d9) -> FULL RUN READY

**Adapter optimizer FROZEN (protocol `phase2/h3-v3`, `probing/phase2.py`).** Decided by the pre-stated VALIDATION-only rule. Converged = best checkpoint before 90% of the epochs and decay not at the grid max, for all four entrance fits (C2 L9, TiRex L11; NOA + FL).

| setting | converged | C2 NOA val | TiRex NOA val | verdict |
|---|---|---|---|---|
| A lr 1e-2 x 300 | no (all four still improving) | 0.01579 | 0.01354 | excluded |
| **B lr 1e-3 x 1500, eval every 25, wd {0, .1, 1, 10, 30, 100}** | **yes** | 0.01399 | 0.01109 | **chosen** |
| C lr 3e-3 x 1000 | no (C2 NOA best at ep 920/1000) | 0.01397 | 0.01074 | excluded |

- **The robustness finding:** converged training moved test MASE at the entrance by <= 1.5% (C2 NOA 1.172 -> 1.158; TiRex NOA 1.073 -> 1.087). The entrance gaps are real, not an optimization artifact.
- **Probe inverse gate** passed on real data (exact match), and validation MASE is present for every arm.

**Latency harness works on the A100** (27/28 ok; Chronos-2-small failed, probably the checkpoint was not pre-downloaded). At depth 3, end to end:

| model | B=1 | B=256 | note |
|---|---|---|---|
| Chronos-2 | 2.9x | 3.0x | |
| TimesFM-3 | 1.35x | 3.6x | 2.0x device-only; a ~50 ms fixed API cost caps B=1 (F22 confirmed) |
| TiRex | 4.0x | 3.8x | ~ proportional to blocks; 0.5 s per full-depth call on the torch backend |

- Adapter cost: +1–4%.
- Native drift within a job: ~1%.
- **B=32 dropped from the defaults** (no figure or table reads it; ~30% of the latency GPU time).

**Measured cost -> full-run jobs (all <= 3 h tier).**

H3 (measured per dataset):

| model | per dataset | jobs |
|---|---|---|
| Chronos-2 | ~27 min (124 s per depth x 12 + FL at L) | 4 jobs of 3–4 datasets |
| TiRex | ~8.5 min | 2 jobs of 7 |
| TimesFM-3 | ~2 min (closed form) | 1 job |

- H3 total ~9 GPU-h.

Latency per repeat:

| model | estimate | jobs |
|---|---|---|
| Chronos-2 | ~20 min | 1 job |
| TimesFM-3 | ~1.4 h | 1 job |
| TiRex | ~2.3 h | 2 depth halves: 0–8 and 9–12 |

- Latency total over 3 repeats: ~12 GPU-h.
- H4 evaluate: ~1–2 GPU-h.
- **Grand total ~22 GPU-h.** The drivers are the 1500-epoch Chronos-2 fits and TiRex's slow torch recurrence.

**Order.**
1. H3 + latency now, in parallel; latency does not depend on H3.
2. H4 evaluate (budget = main; entrance = secondary) after H3.
3. Tables.

**Before the Chronos-2 latency jobs and H4 evaluate:** on the login node, `snapshot_download('autogluon/chronos-2-small')`.

## PHASE 2 RESULTS + PAPER DRAFT (2026-09-24, from commit 2651396)

**In the push:** 42/42 H3 cells, 12/12 headline-eligible latency jobs, and H4 verify (all gates pass at every depth).
**Not in the push:** the H4 `evaluate` cells (entrance and budget). They are still to run; see "Next" below.

**H3 at the frozen entrance** (median test ΔMASE over datasets; ≤5% count):

| | hard cut | label-free (NOA) | supervised (FL) | probe head |
|---|---|---|---|---|
| Chronos-2 (n=14) | +38.6% | **+5.5%**, 7/14, gap closed 85% | +6.7% | +16.9% |
| TimesFM-3 (n=14) | +3,702% | +35.5%, 0/14, gap closed 99% | = probe | **+15.8%** |
| TiRex (n=12; M4 and Traffic have entrance = final) | +98.6% | **+6.3%**, 5/12, gap closed 93% | +8.7% | +10.7% |

- Even at the final block, the probe is 15% / 14% / 10% worse than the native head, so its plateau is a readout deficit, not a depth effect.
- Compatibility lag (NOA, validation): +1 block (C2), +0.5 (TiRex), +17.5 (TimesFM-3). The hard cut is never compatible before the final block.

**H4 frontier** (fixed depth, median over 14 datasets, end-to-end speedup at B=1):
- **C2 NOA:** ≤3.1% from L11 to L4. L4 = 2.43× at +3.1%; L3 = 2.91× at +4.4%.
- **TiRex NOA:** +1.8% at 1.09×, +4.3% at 1.21×, 8.5–10.6% at 1.5–2×.
- **TimesFM-3:** no usable frontier. L19 costs +5.4%. The no-block floor is 69 of 98 ms at B=1, so the maximum B=1 speedup is 1.42× (6.0× at B=256).

**Budget table** (NOA; median speedup over all 14 datasets; within/cut):

| | 10% budget | 20% budget |
|---|---|---|
| C2 | 1.57×, 9/13 | 3.72×, 12/14 |
| TiRex | 1.52×, 10/13 | 1.94×, 13/14 (FL: 3.73×) |
| TimesFM-3 | ≤1.02× | ≤1.02× |

- Hard truncation stays at 1.00× at every budget.

**Frozen-entrance test** (offline; physical confirmation pending): C2 7/14, TiRex 5/12, TimesFM-3 0/14.

**Found while writing (reported in the draft):**
1. TiRex budget gains depend on the low-skill pairs SZ Taxi and M5. Without them, 1.52×→1.27× and 1.94×→1.50×. C2 is unchanged.
2. 74/168 TiRex NOA fits selected a checkpoint in the last 10% of epochs, so they were still improving slowly. Training was not extended after seeing results.
3. At B=1, TimesFM-3's API overhead dominates. The main figure keeps B=1; B=256, device-level and parameter axes are in the appendix.
4. WQL degrades faster than MASE at deep cuts. Example: C2 L4 is +5.5% WQL vs +3.1% MASE.

**Paper outputs:**
- `experiments/make_phase2_paper_tables.py` (rewritten) → `three_models_paper/figures/truncation/`: 8 figures + 9 tables + macros + `truncation_stats.json`.
- `writing/main_compatibility_truncation.tex`: main body H3 + H4, about 3.3 pages including 1 figure and 2 tables.
- `writing/appendix_truncation.tex`: protocol and additional results. It supersedes `phase2_reproducibility.tex`.
- Compiled cleanly with tectonic in a scratch test harness at 5.5in text width: no errors, no overfull boxes.
- 27 H3 contracts pass.

**Next:**
1. H4 evaluate on Narval (6 jobs), then rerun both table scripts. That fills the V3 counts, Chronos-2-small accuracy and the two MISSING appendix tables; lines are marked `% [EVAL]`.
2. Trim the main body to the page budget.
