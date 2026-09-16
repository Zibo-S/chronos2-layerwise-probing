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
| cache tag | `tfm3-last-token-q9-v1`, `(N,1280)` | `tfm3-prefix-v1`, `(N,16,1280)` |
| job | `job_timesfm3_last_token_q9.sh` | `job_timesfm3_probing.sh` |
| cost | 1 forward pass per batch | 16 forward passes per batch |

The two caches/results namespaces are disjoint and the loader **refuses** the other line's cache
(metadata + array-rank check, test 21). No Chronos-2 file is touched by either.

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
  cluster bootstrap → tunnel → summary JSON → bootstrap npz → 5 figures, all green.

## Open / to check on the first GPU run

- The L20 all-quantile reconstruction error (must be < 1e-4 relative) and whether the installed
  `timesfm3` sorts quantiles inside `decode()` (the run prints the discovered knobs). If it sorts
  with no bypass, re-run with `--allow-sorted-reference` — never silently.
- Whether any layer selects the weight-decay grid maximum (the driver warns); widen `--wd-grid`
  if so.
- Caveats to carry into the writeup: all four datasets are in-distribution for the backbone;
  targets are rebuilt through `arcsinh`/`sinh` (float32, ~1e-5 relative); the native head's
  `clamp(±value_clip)` and any quantile sorting are postprocessing outside the linear class.
