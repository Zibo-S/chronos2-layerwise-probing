# Layer-wise Probing of Chronos-2

At which depth inside a **frozen [Chronos-2](https://huggingface.co/amazon/chronos-2)**
time-series foundation model is the forecast most *linearly decodable*? We hook all 12
encoder blocks, freeze everything, and train tiny linear probes on each layer's hidden
states to predict the future trajectory — trained and scored with **Chronos-2's own
quantile (pinball) loss**, and benchmarked against the native Chronos-2 forecast on the
same windows. The model is never trained or fine-tuned; only the probes are fit.

This is the "tunnel effect" question, asked in the model's own currency: if intermediate
layers beat the last layer *at the model's own task under a forecasting-native metric*,
the late layers are doing something other than accumulating linearly-readable forecast
information.

The project has two phases. **Phase 2 (current, headline): forecasting-native probing**,
described below. **Phase 1 (completed, archived): classification probing** on UEA
datasets, summarized [at the end](#phase-1--classification-probing-completed-archived).

## Headline findings

Setting: 4 long hourly datasets, context C=512 / horizon H=64, per-layer linear probes,
validation-selected probe layer **L\*** (chosen without touching test), compared against
the last layer **L11**. Uncertainty from a **series-level cluster bootstrap** (whole test
series resampled, B=5000, paired), so overlapping windows within a series never inflate
the confidence intervals.

**1. On pooled readouts, intermediate layers beat the last layer in 3 of 4 datasets.**
Δ = quantile_loss(L11) − quantile_loss(L\*), so positive = the earlier layer is better
(content pooling, 21-quantile probe; full table in
[`results/bootstrap/tables/bootstrap_table.csv`](results/bootstrap/tables/bootstrap_table.csv)):

| dataset | domain | L\* | Δ quantile loss [95% CI] | probe MASE L\* / L11 | native MASE |
|---|---|---|---|---|---|
| monash_electricity_hourly | energy | 2 | **+0.27** [+0.20, +0.33] | 1.42 / 1.55 | 0.85 |
| monash_kdd_cup_2018 | air quality | 11 | 0 (validation picks L11) | 1.12 / 1.12 | 0.83 |
| monash_pedestrian_counts | foot traffic | 4 | **+0.18** [+0.12, +0.25] | 1.35 / 1.41 | 0.57 |
| uber_tlc_hourly | ride demand | 9 | **+0.68** [+0.60, +0.77] | 1.05 / 1.19 | 0.66 |

The three positive CIs exclude zero; the same holds on MASE and for the REG-token
readout. KDD is an honest null: validation itself selects the last layer.

**2. The advantage survives a much smaller probe head.** Rerunning with the head
predicting only the median (q1: 49k params) or deciles (q9: 443k) instead of all 21
native quantiles (q21: 1.03M) reproduces the same pattern — the same three datasets stay
positive with CIs excluding zero, and KDD never favors an earlier layer (at q1 its
validation-selected L10 even tests slightly *worse* than L11). The mid-layer advantage is
not an artifact of probe capacity.

**3. The advantage is specific to *pooled* readouts.** A structurally Chronos-aligned
probe — one shared linear head reading the model's K=4 native forecast-slot states, the
linear analogue of the native output head — is uniformly the strongest probe (e.g. it
reaches MASE 0.86 on KDD vs native 0.83) and its best layer moves late (L8–L11) with
small or zero mid-vs-last deltas. Compressing the sequence into one pooled vector is what
makes intermediate layers look better; the forecast-slot states keep improving to the end.

**4. No linear probe matches the native head.** The native Chronos-2 forecast (a
nonlinear residual-block head, same windows, same MASE denominator) stays below every
probe. Expected — probes are deliberately low-capacity — but it bounds the claim:
intermediate layers are more *linearly* decodable, not better representations outright.

Caveats, stated up front: all four datasets are in Chronos-2's pretraining corpus (this
measures in-distribution decodability, not generalization); Chronos-2 is used
univariately (its group-attention / multivariate machinery is unprobed); pooled and
shared-slot probes differ in both representation *and* readout capacity, so finding 3 is
a statement about the readout structure, not a controlled representation comparison.

## Method

**Model.** `amazon/chronos-2`: encoder-only probabilistic forecaster — input patches of
16 steps → 768-d tokens, a REG token, 12 encoder blocks, 21-quantile output head,
arcsinh instance normalization. Frozen throughout; features extracted with forward hooks
on all 12 blocks (`probing/extraction.py`).

**Data** (`probing/id_data.py`). Four hourly datasets from
[`autogluon/chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets).
Windows of C=512 context / H=64 horizon, stride 64, within-series temporal split (test =
last 25% of each series; ≤3000 train / ≤1500 test windows per dataset). The probe target
is the H-step future trajectory in Chronos-2's own target space: context-standardized,
then arcsinh.

**Probes** (`probing/probes.py`, registry `PROBES`).

- `quantile` — per layer, `Linear(d, Q·H)` on a pooled 768-d vector (mean over content
  tokens, or the REG token), trained with Chronos-2's exact quantile loss
  `2·|(y−ŷ)·(𝟙[y≤ŷ]−τ)|`, AdamW, weight decay selected per layer on a validation carve
  of train (grid 1e-5…1e-1). Reported score = test quantile loss (lower = better).
- `shared_forecast` — one shared `Linear(768, Q·16)` applied to each of the K=⌈H/16⌉
  native forecast-slot states from a `num_output_patches=K` encoder pass; the K predicted
  patches are concatenated to the H-step forecast, mirroring the native head's layout.
  For a controlled comparison, pooled content/REG probes are re-run on the *same* K-slot
  pass (`content_K`, `reg_K`), since adding forecast slots changes all token states
  (encoder attention is non-causal).
- `binned_future` / `ridge_regression` — earlier scalar readouts (5-bin accuracy / R² on
  the normalized future mean), kept for continuity with Phase 1.

**Metrics.** Primary: Chronos-2 quantile loss per layer. Secondary: MASE of the probe's
median forecast, un-transformed back to raw units, against an in-context seasonal-naive
denominator (m=24) — the *same* windows and denominator used to score the native
`predict_quantiles` baseline. Cross-quantile-set comparisons use `mean_pinball_loss`
(= loss / 2Q; = 0.5·MAE for the median-only probe), since the raw loss sums over
quantiles.

**Uncertainty** (`experiments/run_bootstrap.py`, `probing/stats.py`). Test windows
overlap within a series, so the bootstrap resamples whole series (B=5000, seed 0), with
one shared resampling matrix per dataset so Δ-vs-L11 comparisons are paired. The
comparison layer L\* is selected on validation loss and frozen before any test CI is
computed.

**Probe-capacity ablation.** `--quantile-set {q1,q9,q21}` shrinks the head from 21
quantiles (1,033,536 params at H=64) to deciles (442,944) to median-only (49,216).
Feature caches are quantile-independent and shared; each set writes to its own output
paths so runs can never overwrite each other.

## How to run

### Narval (Compute Canada) — primary environment

One-time setup (login node, which has internet):

```bash
module load gcc python/3.11 arrow/24.0.0     # arrow supplies pyarrow; needs python/3.11 loaded
python -m venv .venv && source .venv/bin/activate
pip install --no-index -r requirements.txt   # cluster pre-built wheels
export HF_HOME=$SCRATCH/chronos2/hf_cache    # pre-download model+datasets here (compute nodes are offline)
```

GPU run (extraction + probe fitting; ~10–20 min on an A100 with warm feature cache):

```bash
sbatch job_id_forecasting.sh                     # default: q21 (all 21 native quantiles)
sbatch job_id_forecasting.sh --quantile-set q9   # extra args are forwarded to the script
```

The job script sets the module trio, `HF_HUB_OFFLINE=1`, and runs
`python -m experiments.run_id_forecasting`. Then, on the login node (CPU, seconds):

```bash
python -m experiments.run_bootstrap    # cluster-bootstrap CIs from the saved per-window inputs
```

### Laptop (Apple Silicon)

Runs on MPS/CPU — Apple Silicon only (no Intel macOS torch wheels), Python 3.11/3.12
(numpy 1.26.4 has no 3.13 wheel). First run downloads the model + datasets and extracts
features (slow); everything is cached afterwards.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_id_forecasting     # or: make forecasting
```

### Tests

```bash
python -m tests.test_quantile_sets    # no model, no GPU, no cache needed (pytest-compatible)
```

Covers the quantile-set registry, exact head param counts, the (B,Q,H) prediction
layout, the loss formula against an explicit reference, pinball identities, and both
probes end-to-end on synthetic features for all three sets.

## Outputs

| Path | Contents |
|---|---|
| `results/id_probing_summary.json` (`_q1`, `_q9`) | full per-dataset summary: per-layer scores for every probe/pooling, MASE, config |
| `results/quantile_loss/` (`_q1/`, `_q9/`) | per-layer loss figures (content/REG/pooling comparison), shared-vs-pooled figures, per-layer train/val curves, `mase_results.json`, `quantile_loss_results.json` |
| `results/bootstrap/inputs/` | per-window losses saved by the GPU run (the bootstrap's input contract) |
| `results/bootstrap/tables/bootstrap_table.{csv,json}` | every dataset × readout × metric × layer: point, 95% CI, Δ vs L11 + CI |
| `results/bootstrap/figures/primary/`, `.../controlled_k_slots/` | per-layer CI bands + Δ-vs-L11 plots, native baseline band, L\* marked |
| `results/quantile_loss_q{1,9}/probe_results_table__q{1,9}.csv` | flat per-row results for cross-set analysis |

Key figures: [`results/quantile_loss/content/quantile_loss_by_layer.png`](results/quantile_loss/content/quantile_loss_by_layer.png)
(the main per-layer curves), [`results/quantile_loss/shared_forecast_mase.png`](results/quantile_loss/shared_forecast_mase.png)
(shared-slot probe vs native MASE),
[`results/bootstrap/figures/primary/H64_K4/quantile_loss_delta_vs_L11__content.png`](results/bootstrap/figures/primary/H64_K4/quantile_loss_delta_vs_L11__content.png)
(the headline Δ with CIs).

## Repository layout

```
probing/                Reusable core
  config.py               constants + repo-anchored paths (SEED=0, NUM_LAYERS=12, OUTPUT_PATCH_SIZE=16, ...)
  extraction.py           get_pipeline (frozen model, CUDA/MPS/CPU), hooked feature extraction + caching
  id_data.py              dataset registry + windowing (C=512, H=64, within-series split)
  probes.py               PROBES registry, quantile/shared-forecast probes, Chronos-2 loss, QUANTILE_SETS
  stats.py                IID and series-level cluster bootstrap helpers
experiments/
  run_id_forecasting.py   GPU driver: extract → probe → summary JSON + figures + bootstrap inputs
  run_bootstrap.py        CPU post-hoc: cluster-bootstrap CIs → tables + figures
tests/                  test_quantile_sets.py (model-free)
job_id_forecasting.sh   SLURM batch script (A100, 1 h, forwards args to the driver)
notes/PLAN.md           rolling working notes (session-by-session design log)
paper/                  writeups + slides (Phase 1 classification material)
results/                summary JSONs, figures, bootstrap outputs (regenerable)
features_cache/         extracted hidden states (gitignored; regenerated on demand)
```

Feature extraction is the only expensive step; every `(dataset, split, pooling, K, H)`
combination is cached to `features_cache/*.npz`, so reruns and probe-only experiments
(e.g. the q1/q9 ablation) skip the model entirely.

## Phase 1 — classification probing (completed, archived)

The project began by probing frozen Chronos-2 with *classification* heads on UEA
multivariate datasets. Findings: task labels are more linearly decodable at intermediate
layers than the last layer in 5/6 datasets (bootstrap CIs exclude zero); a hypothesized
*amplification* of that gap under distribution shift was a null (18 dataset×shift cells,
small and inconsistently signed). That phase is preserved at the git tag
`classification-phase-final`, with the writeup in [`paper/`](paper/) and full numbers in
[`results/perdataset_summary.json`](results/perdataset_summary.json) — which the current
driver still reads as the UEA reference for the ID-vs-classification overlay figure.
Phase 2 exists because a classification metric is the wrong currency for a forecasting
model; the quantile-loss probes above re-ask the same question in the model's own terms.

## Reproducibility

Fixed `SEED = 0` across numpy / torch / sklearn; probe refits are deterministic, so a
rerun on cached features reproduces the committed numbers exactly. Paths are anchored to
the repo root (cwd-independent). Cache keys carry the slot count and horizon (`K{K}_H{H}`),
quantile sets write to disjoint output paths, layer selection for the primary comparison
uses validation loss only, and the bootstrap reproduces each reported point estimate from
the saved per-window inputs before resampling — mismatches fail loudly.
