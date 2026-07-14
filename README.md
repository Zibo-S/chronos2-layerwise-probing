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

Python 3.11 recommended. Runs on Apple Silicon (MPS) or CPU; the model auto-selects the
device (`probing/extraction.py:get_pipeline`). First run downloads `amazon/chronos-2` from
the Hugging Face Hub. `numpy` is pinned `<2` for numba/aeon compatibility — keep the
numpy/numba/llvmlite pins together.

## For collaborators (macOS)

> The rest of this README still describes the earlier **classification** phase and is
> mid-rewrite. For the current **forecasting** pipeline, use the commands in this section.

The forecasting probes run on a MacBook, subject to three constraints:

- **Apple Silicon only** — PyTorch ships arm64-only macOS wheels, so `torch==2.12.1` will not
  install on an Intel Mac.
- **Python 3.11 or 3.12** (not 3.13) — the pinned `numpy==1.26.4` has no 3.13 wheel.
- **First run needs internet and is slow** — with an empty `features_cache/`, the first run
  downloads `amazon/chronos-2` plus three datasets (~660 MB) and extracts features on MPS/CPU
  (minutes; there is no GPU). Everything is cached afterwards, so reruns are fast.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.run_id_forecasting      # or:  make forecasting
```

Outputs land in `results/` (`id_probing_summary.json` and `id_vs_classification_*.png`).

## How to run

Entry points via the `Makefile` (run from the repo root; `make help` lists all):

```bash
make smoke        # fast: confirm results reproduce the committed numbers (no model needed)
make perdataset   # MAIN experiment -> results/perdataset_summary.json + 5 grid figures
make pipeline     # Phase B-D demo (Epilepsy ID, synthetic shift, cross-domain transfer)
make improve      # SCP1-primary run with bootstrap confidence intervals
make harden       # five hardening tests -> results/probe_harden_artifacts.json
make verify       # read-only: check UEA dataset shapes via aeon (no model)
make audit        # read-only: re-derive the conclusions from the summary JSON
```

Equivalently, run the underlying command directly, e.g. `python -m experiments.run_perdataset`
(each `Makefile` target lists its command). On macOS, `make` needs a working Xcode Command
Line Tools install; if `make` errors, use the `python -m ...` / `python tools/...` commands.

### Data flow

```
aeon UEA dataset ──► extract_features() ──► features_cache/*.npz ──► fit_layerwise_probes()
  (load_classification)  [Chronos-2 frozen,      (per layer / pooling /   (StandardScaler + LogReg
                          12 forward hooks,        corruption)              per layer, train-only)
                          content pooling]                                       │
                                                            probes[name](...) ◄──┘
                                                                   │
                                       score → bootstrap CIs → late_drop / amplification
                                                                   │
                                              results/ (summary JSON + figures)
```

### Expected outputs (land in `results/`)

| Command | Writes |
|---|---|
| `make perdataset` | `perdataset_summary.json`, `fig_grid_id_tunnel.png`, `fig_grid_idood_{gauss,timewarp,drift,all}.png` |
| `make harden` | `probe_harden_artifacts.json`, `fig_{pooling_ablation,dataset_forest,multisplit_stability,shift_amplification,scp1_timewarp_idood}.png` |
| `make pipeline` | `fig_id_epilepsy.png`, `fig_ood_*_epilepsy.png`, `fig_transfer_profiles.png` |
| `make improve` | `fig_scp1_*.png`, `fig_transfer_raw.png`, `fig_handwriting_*.png` |

## Repository layout

```
probing/            Reusable core (import this)
  config.py           constants + repo-root-anchored paths (SEED, NUM_LAYERS, MIDDLE_BAND, ...)
  extraction.py       get_pipeline, extract_features (hooks + caching), fit_layerwise_probes
  probes.py           PROBES registry + linear_probe reference  ◄── ADD NEW PROBES HERE
  stats.py            bootstrap_ci, paired_diff_ci
experiments/        Runnable drivers (run_pipeline / run_improve / run_harden / run_perdataset)
tools/              Read-only validation (verify_dataset_facts.py, audit_local.py)
tests/              test_smoke.py — behavior-preservation check
notebooks/          chronos2_probing.ipynb (consolidated, executed notebook)
paper/              writeups, slides, draft.tex
archive/            one-off / historical scripts (extract_check.py, notebook build tooling)
results/            figures + summary JSON (regenerable)
features_cache/     extracted features (~13 GB, gitignored — see below)
```

## Feature cache

`extract_features()` caches every `(dataset, split, corruption, pooling)` to
`features_cache/*.npz`. Extraction is the only expensive step; caching makes reruns instant.

The cache is **~13 GB and gitignored** — it is not in the repo. Options:

- **Transfer it** out-of-band (it's the fastest path; the experiments then need no GPU).
- **Regenerate it:** `make regen-cache` rebuilds everything from scratch (slow; downloads
  `amazon/chronos-2` and runs the encoder over all datasets). The experiment scripts also
  extract any missing entry on demand, so you can just run an experiment and let it fill in.

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
