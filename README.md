# Layer-wise Probing of Chronos-2

Where does task-relevant information live, layer by layer, inside a **frozen Chronos-2**
time-series foundation model? We extract per-layer hidden states across UEA multivariate
classification datasets and fit **linear probes** to measure how *linearly decodable* the
task label is at each of the 12 encoder layers — the "tunnel effect" question for a
time-series foundation model.

The model is never trained or fine-tuned. We only fit tiny classifiers *on top of* its
fixed representations.

## The hypothesis (two parts, kept separate)

- **Part 1 — middle > last.** For frozen Chronos-2, the *middle* encoder layers carry more
  transfer-relevant information (higher linear-probe accuracy) than the *last* layer, whose
  representation is specialized to the forecasting objective.
- **Part 2 — amplified under shift.** That middle-over-last advantage *widens under
  distribution shift* (OOD) relative to in-distribution (ID).

## Current findings

Layer-wise linear probing of frozen Chronos-2 across UEA multivariate
classification datasets, testing the "tunnel effect" hypothesis.

- **Part 1 (in-domain decodability):** task-relevant information is more
  linearly decodable at intermediate layers than at the final layers.
  Holds robustly in 5/6 datasets (bootstrap CIs exclude zero). Note this
  is *in-domain, same-task, linearly-decodable* — not a claim about
  transfer or feature generality (cf. "Deeper is Not Always Better",
  arXiv:2606.21906).
- **Part 2 (OOD amplification):** whether OOD inputs *amplify* the
  mid-vs-final gap is a strong null — effects are small and
  inconsistently signed across 18 dataset×shift cells, consistent with noise.

Scope: probes are trained and evaluated within each dataset (train/test
split), univariate use of Chronos-2 (group-attention/multivariate
functionality not used). See `paper/` for full writeup.

Full numbers: [`results/perdataset_summary.json`](results/perdataset_summary.json);
narrative: [`paper/perdataset_writeup.md`](paper/perdataset_writeup.md).

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

## For collaborators (macOS)

> The rest of this README still describes the earlier **classification** phase and is
> mid-rewrite. For the current **forecasting** pipeline, use the commands in this section.

The forecasting probes run on a MacBook, subject to three constraints:

- **Apple Silicon only** — PyTorch ships arm64-only macOS wheels, so `torch==2.12.1` will not
  install on an Intel Mac.
- **Python 3.11 or 3.12** (not 3.13) — the pinned `numpy==1.26.4` has no 3.13 wheel.
- **First run needs internet and is slow** — with an empty `features_cache/`, the first run
  downloads `amazon/chronos-2` plus the selected dataset set's series and extracts features
  on MPS/CPU (minutes; there is no GPU). Everything is cached afterwards, so reruns are fast.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_id_forecasting      # or:  make forecasting
```

Outputs land in `results/<dataset-set>/` (default `results/extended_v1/`) — see
"Dataset sets" below for selecting the set and the full output layout.

## How to run

Three runnable pipelines ship in `experiments/` (run from the repo root, venv active):

```bash
# 1) ID forecasting probes (extracts features on first run; MPS/CUDA/CPU)
python -m experiments.run_id_forecasting                     # default set: extended_v1
python -m experiments.run_id_forecasting --dataset-set phase0_trio
ID_DATASET_SET=phase0_trio python -m experiments.run_id_forecasting   # env form

# 2) series-level cluster bootstrap (CPU-only, post-hoc; reads step 1's per-window outputs)
python -m experiments.run_bootstrap                          # same ID_DATASET_SET rules
ID_DATASET_SET=phase0_trio python -m experiments.run_bootstrap

# 3) UEA classification baseline (maintained baseline, not under active development)
python -m experiments.run_perdataset                         # -> results/uea/
```

`make forecasting` / `make bootstrap` / `make uea` are aliases. `make help` lists all targets.

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

### Where outputs land

`results/` is fully namespaced — one directory per experiment line:

```text
results/uea/                                    UEA classification baseline (step 3)
  perdataset_summary.json                         per-dataset per-layer accuracies — the
                                                  reference run_id_forecasting overlays against
  fig_*.png, probe_harden_artifacts.json          committed baseline figures/artifacts
results/<set>/                                  ID forecasting (steps 1-2), <set> per run:
  id_probing_summary.json                         headline per-layer numbers
  id_vs_classification_*.png                      overlays vs the UEA reference
  quantile_loss/                                  quantile-probe figures + focused JSONs
  bootstrap/inputs/                               per-window test metrics (written by step 1)
  bootstrap/{raw,tables,figures}/                 bootstrap CIs (written by step 2)
```

`results/phase0_trio/` holds the frozen original Phase 0 run; `results/extended_v1/`
holds the current set's runs. `run_id_forecasting` needs
`results/uea/perdataset_summary.json` to exist — it is committed, and regenerable with
`python -m experiments.run_perdataset`.

### UEA classification baseline (maintained baseline, not under active development)

The UEA classification line is kept as a **maintained baseline** in the same tree:
`experiments/run_perdataset.py` (driver) on top of `extract_features()` +
`fit_layerwise_probes()` in `probing/extraction.py`, `linear_probe` /
`score_layerwise_correctness` in `probing/probes.py`, and the shift-type taxonomy in
`probing/shifts.py`. It writes everything under `results/uea/`. Its dataset loading uses
`aeon` (pinned in `requirements.txt` together with `numba`/`llvmlite`). The auxiliary
one-off scripts of the original classification phase (`run_pipeline`, `run_harden`,
`run_improve`, `tools/`, `tests/`) are retired and not part of the maintained surface.

## Repository layout

```text
probing/            Reusable core (import this)
  config.py           constants + paths; ID_DATASET_SET selector + results/<set>/ namespacing
  extraction.py       get_pipeline; extract_features (UEA); extract_window_features and
                      extract_kout_features (ID forecasting, K native forecast slots);
                      fit_layerwise_probes
  id_data.py          ID_DATASET_SPECS (phase0_trio / extended_v1) + windowing, labels,
                      trajectories, MASE denominators
  probes.py           PROBES registry  ◄── ADD NEW PROBES HERE
  shifts.py           shift-type taxonomy (gauss / timewarp / drift tagging)
  stats.py            bootstrap_ci, paired_diff_ci
experiments/        Runnable drivers: run_id_forecasting.py, run_bootstrap.py,
                    run_perdataset.py (UEA baseline)
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
  written — delete the named `features_cache/IDF_<tag>__*` files and re-run.

## Probe registry (what's in `PROBES` and what each is for)

| Name | Task | Labels it needs | Score |
| --- | --- | --- | --- |
| `linear` | UEA classification (reference) | 1-D class labels | accuracy (higher = better) |
| `ridge_regression` | ID forecasting | 1-D scalar `y` (normalized future mean) | R² |
| `binned_future` | ID forecasting — primary tunnel readout | 1-D scalar `y`, binned into 5 quantile bins | accuracy |
| `quantile` | ID forecasting, Chronos-2-native | **2-D** `(n, H)` arcsinh trajectories (`Y_*_traj`) | Chronos-2 quantile loss (**lower** = better) |
| `shared_forecast` | ID forecasting, Chronos-aligned shared head | `(n, H)` trajectories **and** `(n, K, 768)` forecast-slot features from `extract_kout_features` | Chronos-2 quantile loss (**lower** = better) |

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

The reference `linear_probe` is verified to reproduce the committed per-layer accuracies
exactly (`make smoke`). Keep new probes side-effect-free and deterministic (respect
`probing.config.SEED`).

## Reproducibility

Fixed seed (`SEED = 0`) across numpy / torch / sklearn. Paths are anchored to the repo root,
so cache keys and results are cwd-independent. `make smoke` re-derives the committed
per-layer accuracies from the cache and asserts they match to < 1e-9.
