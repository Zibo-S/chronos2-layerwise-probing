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

## Frozen native-head transfer

`model.output_head` (the checkpoint's own `Linear(1280, 576)`) applied to `h_{l,15}` at every
point, then decode()'s own inverse path: `revin(reverse, token-15 stats) -> clamp(±value_clip)
-> stitch_patches -> + context trend`. Scored by the SAME `native_reference` that scores
decode(), so the curves are directly comparable with the probe's, and `A_l = L_l^{head} −
L_l^{probe}` is well defined.

Gates (all abort): L20 must reproduce decode()'s nine quantiles **elementwise** — `|recon −
decode| ≤ atol + rtol·|decode|` per element over the full (N,64,9) tensor (atol 1e-5, rtol 2e-6),
failing iff the worst *scaled* error exceeds 1; this replaces the old `max|d|/mean|ref|` ratio,
which was unfair to heavy-tailed forecasts (BOOM's 162.8-magnitude element, 5e-7 relative, read
as ~1e-4 against the 0.73 mean). The worst-element error is a uniform 2–7 float32 ULP across all
seven datasets. The L20 Q=9 loss must equal the native baseline to **<1e-5 relative** (relative,
not absolute — the cache stores decode()'s forecast in float32 while this path is float64, so
they agree to ~1e-9 of the loss for the large-N datasets; the small-N Coastal T-S measures ~2e-6,
which is exactly why the bar is 1e-5 and not 1e-6). The head's parameter sha256 must be identical
before and after; window parity with the committed Chronos-2 artifacts must hold. A feature-cache
MISS aborts by default (`--allow-extraction` to override) because this job requests no GPU.

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
