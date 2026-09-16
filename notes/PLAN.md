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

- The L20 all-quantile reconstruction error (must be < 1e-4 relative) and whether the installed
  `timesfm3` sorts quantiles inside `decode()` (the run prints the discovered knobs). If it sorts
  with no bypass, re-run with `--allow-sorted-reference` — never silently.
- Whether any layer selects the weight-decay grid maximum (the driver warns); widen `--wd-grid`
  if so.
- Caveats to carry into the writeup: all four datasets are in-distribution for the backbone;
  targets are rebuilt through `arcsinh`/`sinh` (float32, ~1e-5 relative); the native head's
  `clamp(±value_clip)` and any quantile sorting are postprocessing outside the linear class.
