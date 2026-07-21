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

The project has two lines. **Forecasting-native probing (current, headline)**, described
below. **UEA classification probing (completed, maintained as a baseline)**, summarized
in [its section](#uea-classification-baseline-maintained-baseline-not-under-active-development).

## Headline findings

Setting: 4 long hourly datasets, context C=512 / horizon H=64, per-layer linear probes,
validation-selected probe layer **L\*** (chosen without touching test), compared against
the last layer **L11**. Uncertainty from a **series-level cluster bootstrap** (whole test
series resampled, B=5000, paired), so overlapping windows within a series never inflate
the confidence intervals.

**1. On pooled readouts, intermediate layers beat the last layer in 3 of 4 datasets.**
Δ = quantile_loss(L11) − quantile_loss(L\*), so positive = the earlier layer is better
(content pooling, 21-quantile probe; full table in
[`results/extended_v1/bootstrap/tables/bootstrap_table.csv`](results/extended_v1/bootstrap/tables/bootstrap_table.csv)):

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
  the normalized future mean), kept for continuity with the classification phase.

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

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or:  make setup
```

Python 3.11 recommended (3.11 or 3.12; the pinned `numpy==1.26.4` has no 3.13 wheel).
Runs on CUDA, Apple Silicon (MPS), or CPU; the model auto-selects the device
(`probing/extraction.py:get_pipeline`). First run downloads `amazon/chronos-2` from the
Hugging Face Hub and the probing datasets from `autogluon/chronos_datasets` — both are
public, no HF login/token needed. `numpy` is pinned `<2` for Compute Canada wheel-stack
compatibility, and `aeon`/`numba`/`llvmlite` are pinned together for the same reason (the
UEA classification baseline loads its datasets through `aeon`).

## How to run

Three runnable pipelines ship in `experiments/` (repo root, venv active). Both ID
forecasting drivers take `--dataset-set` (which datasets + output namespace) and the main
driver also takes `--quantile-set` (probe-head capacity) — the two flags compose:

```bash
# 1) ID forecasting probes (extracts features on first run; MPS/CUDA/CPU)
python -m experiments.run_id_forecasting                     # defaults: extended_v1, q21
python -m experiments.run_id_forecasting --dataset-set phase0_trio --quantile-set q1
ID_DATASET_SET=phase0_trio python -m experiments.run_id_forecasting   # env form

# 2) series-level cluster bootstrap (CPU-only, post-hoc; reads step 1's per-window outputs)
python -m experiments.run_bootstrap                          # same --dataset-set / env rules

# 3) UEA classification baseline (maintained baseline, not under active development)
python -m experiments.run_perdataset                         # -> results/uea/
```

`make forecasting` / `make bootstrap` / `make uea` are aliases. `make help` lists all targets.

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

### Dataset sets

The named sets live in `probing/id_data.py` (`ID_DATASET_SPECS`); the *active* set is
`probing.config.DATASET_SET` — precedence **CLI `--dataset-set` > env `ID_DATASET_SET` >
default `extended_v1`** (the CLI writes through `config.set_dataset_set`, which re-derives
the output dirs). The same value selects the dataset roster **and** the output directory,
so a run can never write one set's numbers into the other set's folder. Both
`run_id_forecasting` and `run_bootstrap` accept `--dataset-set`.

| Set | Datasets | Split | Notes |
| --- | --- | --- | --- |
| `phase0_trio` | m4_hourly, monash_electricity_hourly, solar_1h | m4 falls back to cross-series | original Phase 0 run; solar_1h has a documented label pathology (`paper/phase0_fixes.md`) |
| `extended_v1` (default) | monash_electricity_hourly, monash_kdd_cup_2018, monash_pedestrian_counts, uber_tlc_hourly | all within-series | current set; fixes the solar/m4 caveats |

### Quantile sets (probe-capacity ablation)

`--quantile-set` selects the quantile *vector* the probe head predicts — it is orthogonal
to `--dataset-set` and does **not** change the probe registry. The vectors live in
`probing.probes.QUANTILE_SETS`: `q1` = median only, `q9` = deciles, `q21` = Chronos-2's
21 native levels (default; reproduces the committed numbers exactly). q21 writes to the
standard per-namespace paths; q1/q9 write to `_q1`/`_q9`-suffixed paths under the same
`results/<set>/` namespace. Cross-set comparisons use `mean_pinball_loss` (= loss / 2Q);
the raw Chronos-2 loss sums over quantiles and is only comparable within one set.

### Tests

```bash
python -m tests.test_quantile_sets    # no model, no GPU, no cache needed (pytest-compatible)
```

Covers the quantile-set registry, exact head param counts, the (B,Q,H) prediction
layout, the loss formula against an explicit reference, pinball identities, and both
probes end-to-end on synthetic features for all three sets.

### Where outputs land

`results/` is fully namespaced — one directory per experiment line:

```text
results/uea/                                    UEA classification baseline (step 3)
  perdataset_summary.json                         per-dataset per-layer accuracies — the
                                                  reference run_id_forecasting overlays against
  fig_*.png, probe_harden_artifacts.json          committed baseline figures/artifacts
results/<set>/                                  ID forecasting (steps 1-2), <set> per run:
  id_probing_summary.json  (_q1, _q9)             headline per-layer numbers (per quantile set)
  id_vs_classification_*.png                      overlays vs the UEA reference
  quantile_loss/  (quantile_loss_q{1,9}/)         per-layer loss + MASE figures, train/val
                                                  curves, focused JSONs, probe_results_table CSV
  bootstrap/inputs/                               per-window test metrics (written by step 1;
                                                  __q1/__q9 suffixed for the ablation sets)
  bootstrap/tables/bootstrap_table.{csv,json}     dataset × readout × metric × layer: point,
                                                  95% CI, Δ vs L11 + CI (quantile_set column)
  bootstrap/figures/{primary,controlled_k_slots}/ per-layer CI bands + Δ-vs-L11 plots
                                                  (H64_K4, H64_K4_q1, H64_K4_q9)
```

Key figures:
[`results/extended_v1/quantile_loss/content/quantile_loss_by_layer.png`](results/extended_v1/quantile_loss/content/quantile_loss_by_layer.png)
(the main per-layer curves),
[`results/extended_v1/quantile_loss/shared_forecast_mase.png`](results/extended_v1/quantile_loss/shared_forecast_mase.png)
(shared-slot probe vs native MASE),
[`results/extended_v1/bootstrap/figures/primary/H64_K4/quantile_loss_delta_vs_L11__content.png`](results/extended_v1/bootstrap/figures/primary/H64_K4/quantile_loss_delta_vs_L11__content.png)
(the headline Δ with CIs).

`results/phase0_trio/` holds the frozen original Phase 0 run; `results/extended_v1/`
holds the current set's runs. `run_id_forecasting` needs
`results/uea/perdataset_summary.json` to exist — it is committed, and regenerable with
`python -m experiments.run_perdataset`.

### UEA classification baseline (maintained baseline, not under active development)

The project began by probing frozen Chronos-2 with *classification* heads on UEA
multivariate datasets. Findings: task labels are more linearly decodable at intermediate
layers than the last layer in 5/6 datasets (bootstrap CIs exclude zero); a hypothesized
*amplification* of that gap under distribution shift was a null (18 dataset×shift cells,
small and inconsistently signed). The forecasting phase exists because a classification
metric is the wrong currency for a forecasting model; the quantile-loss probes above
re-ask the same question in the model's own terms.

The classification line is kept as a **maintained baseline** in the same tree:
`experiments/run_perdataset.py` (driver) on top of `extract_features()` +
`fit_layerwise_probes()` in `probing/extraction.py`, `linear_probe` /
`score_layerwise_correctness` in `probing/probes.py`, and the shift-type taxonomy in
`probing/shifts.py`. It writes everything under `results/uea/`; the full writeup lives in
[`paper/`](paper/). Its dataset loading uses `aeon` (pinned in `requirements.txt`
together with `numba`/`llvmlite`). The auxiliary one-off scripts of the original
classification phase (`run_pipeline`, `run_harden`, `run_improve`, `tools/`) are retired
and not part of the maintained surface.

## Repository layout

```text
probing/            Reusable core (import this)
  config.py           constants + paths; ID_DATASET_SET selector + results/<set>/ namespacing
  extraction.py       get_pipeline; extract_features (UEA); extract_window_features and
                      extract_kout_features (ID forecasting, K native forecast slots);
                      fit_layerwise_probes
  id_data.py          ID_DATASET_SPECS (phase0_trio / extended_v1) + windowing, labels,
                      trajectories, MASE denominators
  probes.py           PROBES registry + QUANTILE_SETS  ◄── ADD NEW PROBES HERE
  shifts.py           shift-type taxonomy (gauss / timewarp / drift tagging)
  stats.py            bootstrap_ci, paired_diff_ci, cluster bootstrap helpers
experiments/        Runnable drivers: run_id_forecasting.py, run_bootstrap.py,
                    run_perdataset.py (UEA baseline)
tests/              test_quantile_sets.py — quantile-set/loss contracts, no GPU needed
notebooks/          chronos2_probing.ipynb (consolidated, executed notebook)
paper/              writeups (phase0_fixes.md, perdataset_writeup.md, ...)
data/               uea_domain_audit.md, chronos2_seen_manifest.md
results/            results/{uea, phase0_trio, extended_v1}/ (see "Where outputs land")
features_cache/     extracted features (~13 GB, gitignored — see below)
```

## Feature cache

`extract_features()` caches every `(dataset, split, corruption, pooling)` to
`features_cache/*.npz`. Extraction is the only expensive step; caching makes reruns instant.

The cache is **~13 GB and gitignored** — it is not in the repo. Options:

- **Transfer it** out-of-band (it's the fastest path; the experiments then need no GPU).
- **Let it regenerate on demand:** every experiment extracts any missing entry when it runs,
  so a fresh clone just runs slower the first time. Cache key layouts:
  `<dataset>__<split>__<corruption>__<pooling>.npz` (UEA),
  `IDF_<tag>__<split>__clean__<pooling>.npz` (ID windows), and
  `IDF_<tag>__<split>__clean__K<K>_H<horizon>.npz` (K-forecast-slot states). The ID caches
  carry the window labels and **fail loudly** if the windowing changed since they were
  written — delete the named `features_cache/IDF_<tag>__*` files and re-run. Feature
  caches are quantile-set-independent: q1/q9/q21 runs share them.

## Probe registry (what's in `PROBES` and what each is for)

| Name | Task | Labels it needs | Score |
| --- | --- | --- | --- |
| `linear` | UEA classification (reference) | 1-D class labels | accuracy (higher = better) |
| `ridge_regression` | ID forecasting | 1-D scalar `y` (normalized future mean) | R² |
| `binned_future` | ID forecasting — scalar tunnel readout | 1-D scalar `y`, binned into 5 quantile bins | accuracy |
| `quantile` | ID forecasting, Chronos-2-native | **2-D** `(n, H)` arcsinh trajectories (`Y_*_traj`) | Chronos-2 quantile loss (**lower** = better) |
| `shared_forecast` | ID forecasting, Chronos-aligned shared head | `(n, H)` trajectories **and** `(n, K, 768)` forecast-slot features from `extract_kout_features` | Chronos-2 quantile loss (**lower** = better) |

`q1` / `q9` / `q21` are **not** separate probes: they are quantile vectors
(`probing.probes.QUANTILE_SETS`) that parameterize `quantile` and `shared_forecast`
through the driver's `--quantile-set` flag (see "Quantile sets" above). The registry
stays at these five entries.

Contract exceptions: `quantile` raises on 1-D labels (pass the trajectories, not `y`);
`shared_forecast` additionally needs the 3-D K-slot features — a driver looping `PROBES`
generically must special-case which label/feature arrays it hands those two.

## Adding a new probe (for collaborators)

The probe is the one pluggable piece. **Extraction and the experiment loops never change.**

1. Open [`probing/probes.py`](probing/probes.py) and implement a function with the standard
   signature:

   ```python
   def my_probe(train_feats, train_labels, test_feats, test_labels) -> dict[int, float]:
       # train_feats / test_feats : {layer_idx: np.ndarray (n_samples, d)}   d = n_channels * 768
       # *_labels                 : 1-D arrays aligned with the feature rows
       # return                   : {layer_idx: score}  (higher = more information)
       ...
   ```

   - Supervised probes (like `linear_probe`) fit on train, evaluate on test.
   - Unsupervised measures (**effective-rank, entropy, epiplexity** — stubs already present)
     may ignore the labels / train split and compute directly on `test_feats`. They still
     take all four arguments so every probe has one uniform signature.

2. Register it in the `PROBES` dict at the bottom of the file:

   ```python
   PROBES = {
       "linear": linear_probe,
       "effective_rank": effective_rank,   # your new probe
   }
   ```

3. Use it in a driver by pulling features once and calling any probe:

   ```python
   from probing.extraction import extract_features
   from probing.probes import PROBES

   f_tr, y_tr = extract_features("SelfRegulationSCP1", "train")
   f_te, y_te = extract_features("SelfRegulationSCP1", "test")
   scores = PROBES["effective_rank"](f_tr, y_tr, f_te, y_te)   # {layer: score}
   ```

Keep new probes side-effect-free and deterministic (respect `probing.config.SEED`).

## Reproducibility

Fixed seed (`SEED = 0`) across numpy / torch / sklearn; the bootstrap uses one shared
resampling matrix per dataset (B=5000, seed 0) so paired comparisons stay paired. Paths
are anchored to the repo root, so cache keys and results are cwd-independent, and every
output is namespaced by dataset set (and quantile set) so no configuration can overwrite
another. `python -m tests.test_quantile_sets` verifies the probe/loss contracts without a
GPU, model download, or cache.
