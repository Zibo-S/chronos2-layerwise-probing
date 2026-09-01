# `ext_v4_future_tokens` — Experimental-Framework Audit

**Purpose.** A verified factual reference for writing `\section{Experimental Framework}` of a 4-page
NeurIPS workshop paper. Every claim below is traced to a file/function/line or an on-disk artifact.
Anything not verifiable from the repository is explicitly marked **UNVERIFIED**.

**Audit basis.**
* Repo: `/Users/egorserebriakov/eserebri/chronos2-layerwise-probing/chronos2-layerwise-probing`
* Branch `tunnel-effect-probing`, HEAD `1bf1b56` ("upd last experiment"), working tree **clean**.
* Model source read from the installed-equivalent `chronos-forecasting` **v2.3.1** checkout at
  `/Users/egorserebriakov/eserebri/Research/repos/src/chronos` (`__about__.py: __version__ = "2.3.1"`),
  which matches the pin in `requirements.txt:16` (`chronos-forecasting==2.3.1`).
* `features_cache/` is **not present on this machine** (gitignored, lives on Narval), so nothing was
  recomputed from features; all result numbers quoted are read from committed artifacts.

> **Two naming warnings up front, because they are the most likely sources of a false statement:**
> 1. `extended_v3_rolling` is **not** the experiment name. See §3.1.
> 2. The point this codebase labels **`L12+LN`** is produced by **RMSNorm**, not LayerNorm. See §1.4.

---

## 1. Model and Layer Extraction

### 1.1 Model / checkpoint

| Fact | Value | Evidence |
|---|---|---|
| Checkpoint | `amazon/chronos-2` | `probing/extraction.py:49` (`Chronos2Pipeline.from_pretrained("amazon/chronos-2", torch_dtype=torch.float32)`); `experiments/run_native_head_adapter.py:51` (`MODEL_ID`); every `results/ext_v5_native_head_adapter/configs/*.json` (`"model_id": "amazon/chronos-2"`) |
| Library | `chronos-forecasting==2.3.1` | `requirements.txt:16` |
| Precision | float32 | `probing/extraction.py:49` |
| Device | CUDA if available, else MPS, else CPU (production runs: A100 CUDA) | `probing/extraction.py:42-47`; job scripts request `--gres=gpu:1` |
| Frozen | `inner.eval()`; `p.requires_grad_(False)` for **all** parameters | `probing/extraction.py:56-58` |
| Determinism during extraction | `model.eval()` per batch with prior train/eval mode restored; `torch.no_grad()` | `probing/extraction.py:500-512` |

### 1.2 Architecture (values verified from a real model load, not from class defaults)

`results/ft_specialization/boom/manifest.json` was written by `probing/finetune.py:490-495` reading
`model.config` / `model.chronos_config` of the actually-loaded `amazon/chronos-2`:

```json
"trainable_params": 119477664,
"model_config": {"d_model": 768, "d_ff": 3072, "num_layers": 12,
                 "num_quantiles": 21, "output_patch_size": 16, "dropout_rate": 0.1}
```

| Quantity | Value | Evidence |
|---|---|---|
| Hidden dimension `d_model` | **768** | manifest above; independently, every probe checkpoint has `in_features = 768` (`results/ext_v4_future_tokens/ptood_probing/ptid_checkpoints/.../L05.pt`) |
| Encoder blocks | **12** | manifest `num_layers: 12`; extraction hooks every element of `model.encoder.block` and the caches contain exactly layer keys `0..12` (`probing/extraction.py:503`, `:441-448`) |
| FFN width `d_ff` | **3072** | manifest above |
| Native quantiles | **21** | manifest `num_quantiles: 21`; `probing/probes.py:145-148` `CHRONOS2_QUANTILES` (0.01, 0.05, 0.1…0.95, 0.99) |
| Output/input patch size `P` | **16** (asserted equal input=output by the model itself) | manifest; `probing/config.py:33` `OUTPUT_PATCH_SIZE = 16`; assertion `probing/extraction.py:464-468`; model-side assert `chronos2/model.py:226-229` |
| Dropout rate | 0.1 (irrelevant — everything runs in `eval()`) | manifest |
| Total params | 119,477,664 | manifest `trainable_params` (full-FT run) |
| Attention heads | 12 | **UNVERIFIED** from any saved artifact. Claimed in `notes/PLAN.md:635-636`. Consistent with `d_model/d_kv = 768/64`, but `d_kv` is itself unverified. **Do not state a head count in the paper.** |
| `context_length` = 8192, `max_output_patches` = 64 | claimed | **UNVERIFIED** from repo artifacts; asserted in `notes/PLAN.md:26-27`. Not needed for the paper (C=512 ≪ any plausible limit, and H=64 = K·P exactly). |

Per-block structure (source, `chronos2/model.py:48-90`): each `Chronos2EncoderBlock` = `TimeSelfAttention`
→ `GroupSelfAttention` → `FeedForward`. **Group attention is inert in this study**: `model.encode` is
called with `group_ids=None`, which the model turns into `torch.arange(batch_size)`, i.e. every series is
its own group and no information mixes across the batch (`chronos2/model.py:626-628`;
`probing/extraction.py:507`).

### 1.3 Token sequence at the probed depth

`extract_kout_features` runs **one** forward pass with `num_output_patches = K`. The encoder token
sequence is
```
[ ncp context patches | 1 REG token | K forecast slots ]        length P_expected = ncp + 1 + K
```
* `ncp = ceil(C / patch_size) = ceil(512/16) = 32` — `probing/extraction.py:470`
* REG present: `num_special = int(model.chronos_config.use_reg_token) = 1` — `probing/extraction.py:469`
* `K = 4` (§2) → sequence length **37**, asserted at batch 0 (`probing/extraction.py:516-518`).
* Self-attention is **non-causal**, so the presence of the K forecast slots changes *all* token states.
  This is why `extract_kout_features` *computes and caches* content/REG from this same K-pass rather than
  reusing a K=1 pass — note, however, that no ext_v4 driver actually **fits a probe** on those
  same-pass pooled features; the committed pooled results come from a separate K=1 extraction (§4.5)
  (`probing/extraction.py:404-407`).

### 1.4 The 14 representation points — exact extraction sites

The **fslot** readout (the `ext_v4` headline) probes **14 points**. Labels come from
`experiments/run_ptood_probing_ftok.py:113-114`:

```python
POST_LN_LABEL = "L12+LN"
LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + [POST_LN_LABEL]   # 14 entries
```

| Index | Label | What it is | Extraction site |
|---:|---|---|---|
| 0 | `Emb` | Embedded token sequence **entering block 1** (context patch embeddings + REG + K future-slot embeddings), pre-attention | `register_forward_pre_hook` on `model.encoder.block[0]`, capturing `args[0]` — `probing/extraction.py:495-502` |
| 1–12 | `L1`…`L12` | Output of encoder block *i* (block *i* → key *i+1*) — **pre**-final-norm | `register_forward_hook` on each `model.encoder.block[i]` — `probing/extraction.py:489-493, 503` |
| 13 | `L12+LN` | `encoder.final_layer_norm(L12)` = **the tensor the native head actually consumes** | `enc_out[0]` from `model.encode(...)` — `probing/extraction.py:507, 514`; produced by `chronos2/model.py:190` |

So **yes**: 1 embedding point, 12 transformer-block outputs, and a separate post-final-normalization
point. **Both L12 (pre-norm) and L12+LN (post-norm) are evaluated as distinct probe points.**

**The norm is RMSNorm, not LayerNorm.** `encoder.final_layer_norm` is a
`Chronos2LayerNorm` (`chronos2/model.py:105`), defined at `chronos2/layers.py:129-146` as *"a layernorm
module in the T5 style. No bias and no subtraction of mean"*:
```
h ← w ⊙ h / sqrt(mean(h²) + ε),   ε = config.layer_norm_epsilon (default 1e-6)
```
The ext_v5 module renames this point accordingly: `LAYER_LABELS[13] = "L12+RMS"`
(`experiments/run_native_head_adapter.py:58`; `probing/native_head_adapter.py:30-31`).
**Recommendation for the paper: call it "post-final RMSNorm"; if you keep the code label `L12+LN`,
define it explicitly as RMSNorm.** The ext_v4 filenames/figures say `L12+LN`; ext_v5 says `L12+RMS`;
these are the *same* tensor.

There is a `self.dropout(hidden_states)` after `final_layer_norm` (`chronos2/model.py:191`); in `eval()`
it is the identity, so it does not affect anything here.

### 1.5 Tensor shapes per representation

`extract_kout_features` returns three pooled views of every one of the 14 points
(`probing/extraction.py:476-478`), all derived from the same `(b, 37, 768)` hidden state:

| View | Pooling | Shape | Line |
|---|---|---|---|
| `content` | mean over the `ncp = 32` context tokens | `(n, 768)` | `:476` |
| `reg` | the single REG token at index `ncp` | `(n, 768)` | `:477` |
| **`fslot`** | the **last K tokens** (the forecast slots) | **`(n, K, 768)` = `(n, 4, 768)`** | `:478` |

Axis meaning for the fslot tensor: **dim 0 = window (batch), dim 1 = forecast slot k ∈ {0..K−1},
dim 2 = hidden dimension 768.**

`ext_v4` uses **`fslot` only**. Assembly of the 14-key dict is
`experiments/run_ptood_probing_ftok.py:185-198` (`_fslot_feats`): keys `0..12` from `fk["fslot"]`, key
`13` from `final["fslot"]`, with a per-key assert `ndim == 3 and shape[1] == K`.

### 1.6 The pre-/post-final-norm history (the "mismatch" you asked about) — **RESOLVED**

* **Problem.** The block forward-hooks capture states **before** `encoder.final_layer_norm`. The native
  head (`output_patch_embedding`) consumes states **after** it (`chronos2/model.py:190 → 731-732`). So a
  13-point curve ending at pre-norm `L12` never touched the native head's actual input.
* **Fix (current, canonical).** A 14th point `L12+LN` = `final["fslot"]` was added and is the **final
  reference** for every downstream statistic. Documented at `experiments/run_ptood_probing_ftok.py:109-114`;
  `notes/PLAN.md` §"v4 POST-FINAL-LN READOUT POINT". The `tunnel.py` statistics are length-agnostic and
  key off `curve[-1]`, so `L12+LN` becomes the reference automatically.
* **Scope.** The extra point exists **only on the fslot line**. The pooled-`content` line
  (`results/extended_v3_rolling/`, `run_ptood_probing.py`) stays at 13 points ending at pre-norm `L12`.
  `probing/config.py:24-26` still declares `NUM_LAYERS = 13`, `LAST_LAYER = 12`; the 14th point is a
  data-driven extra key, never a constant.
* **Consequence you must state correctly:** on the fslot curves, `L12` is *not* the native-head input;
  `L12+LN` is. Effective-rank and CKA on the fslot line likewise carry 14 points; the extended_v3
  *content* CKA carries 13 (`experiments/run_cka_analysis.py:80-81`, `:218` — "to L12 (extended_v3 has no
  L12+LN)").
* **Verified consequence at L12 (ext_v5):** because index 13 *is* RMSNorm(index 12), pushing index 12
  through the native head (which applies the RMSNorm first) reproduces the native forecast **exactly**.
  On-disk proof: `results/ext_v5_native_head_adapter/tables/native_head_adapter__gap_recovery__all.csv`,
  Electricity row `layer=12`: `zero_shot_mase = 0.836189 == native_mase = 0.836189`,
  `gap_denominator = 0.0`, `valid_flag = "undefined:gap~0"`. Same on all 7 datasets
  (`experiments/run_native_head_gap_recovery.py:13-15`).

---

## 2. Forecasting Task

### 2.1 Geometry

| Quantity | Value | Evidence |
|---|---|---|
| Context length `C` | **512** | `experiments/run_ptood_probing_ftok.py:88` (`C, H = 512, 64`); `probing/id_data.py:355` default |
| Horizon `H` | **64** | same |
| Patch length `P` (input == output) | **16** | `probing/config.py:33`; asserted against the loaded model at `probing/extraction.py:464-468` |
| Forecast slots `K` | **4** | `K = ceil(H / P) = ceil(64/16) = 4` — `experiments/run_ptood_probing_ftok.py:89`; derived *inside* `extract_kout_features` at `probing/extraction.py:427` (Chronos-2's own rule, `pipeline.get_num_output_patches`) |
| Quantile sets used in ext_v4 | `q9` = {0.1,…,0.9} (9 deciles) and `q1` = {0.5} | `probing/probes.py:154-158`; run by `job_full_q1_q9_rerun.sh:37` (`QSETS=(q9 q1)`) |
| Quantile set used in ext_v5 | `q21` = the 21 native levels (forced; the frozen head is 21-quantile) | `experiments/run_native_head_adapter.py:486-487` (`choices=["q21"]`), `:512` |

`H = K·P` exactly (64 = 4·16), so the trim step below is a no-op for this configuration.

### 2.2 Univariate treatment / channels

Every probing example is a **single univariate** length-512 context window. `model.encode` is called
with `group_ids=None` → each batch row is its own group → no cross-series mixing
(`probing/extraction.py:507`; `chronos2/model.py:626-628`). Multivariate OOD sources are split into
univariate series before windowing and clustered by their parent (§3.4). **Chronos-2's multivariate /
covariate machinery is unprobed** — this is a stated limitation
(`data/ood_targets_manifest.md`, "Known limitations" #4).

### 2.3 Normalisation / transforms

**Inside the model (untouched by us).** `Chronos2Model._prepare_patched_context` applies
`self.instance_norm(context)` (`chronos2/model.py:408`), i.e. `InstanceNorm(use_arcsinh=True)`
from `chronos/chronos_bolt.py:95-122`:
```
loc   = nanmean(x, dim=-1)                              # chronos_bolt.py:111
scale = sqrt(nanmean((x-loc)², dim=-1)),  0 → eps=1e-5  # chronos_bolt.py:112-113
z     = arcsinh((x - loc)/scale)                        # chronos_bolt.py:117-120
```
Inverse (`chronos_bolt.py:124-134`): `x = sinh(z) · scale + loc`.
So **raw** (unnormalised) values are fed to the model; the model normalises internally.

**In the probe labels (our side).** `probing/id_data.py:132-160` (`_make_examples`) constructs, per
window, `mu = ctx.mean()`, `sd = ctx.std()` (clamped at `sigma_eps = 1e-6`), and
```python
lin_vec = (fut - mu) / s                      # (H,) context-standardised future
yvec    = np.arcsinh(lin_vec)                 # -> Y_*_traj, the probe target      (id_data.py:156-157)
```
This is deliberately the **same space** the model's own loss lives in (context-only statistics ⇒
leakage-free). `notes/PLAN.md:24-26` records that these `_ctx_stats` match Chronos-2's `InstanceNorm`
(`nanmean`/`nanstd`+`arcsinh`), so the probe's normalised target space **is** the head's normalised space
— which is what makes the ext_v5 native-head reuse valid.

**Inverse for reporting.** `experiments/run_id_forecasting.py:152-158` (`_ctx_stats`) and
`experiments/run_fslot_transfer.py:188-201` (`_fslot_mase`):
`y_raw = mu + s·sinh(z)` — applied identically to the ground-truth trajectory and to any prediction, so
target and prediction stay exactly consistent.

**Nothing else is normalised**: no per-split statistics, no de-trending, no fill. Windows whose context
or target contains a non-finite value, or whose context is near-constant, are **dropped**
(`probing/id_data.py:145-152`) — never imputed.

### 2.4 Prediction assembly (concatenate + trim)

`probing/probes.py:605-619` (`_apply_shared_head`):
```python
out = lin(X).view(n, K, Q, P)                                # (n, K, Q, P)
out = out.permute(0, 2, 1, 3).reshape(n, Q, K*P)[:, :, :H]   # (n, Q, H)
```
The K predicted patches are laid **end-to-end** (concatenated, never summed), mirroring Chronos-2's own
`rearrange("b n (q p) -> b q (n p)")` (`chronos2/model.py:733-739`), then trimmed to `H`. At `H = 64,
K·P = 64` the trim is a no-op. There is an assert that `K·P ≥ H` (`probes.py:616-618`).

**Prediction layout contract, asserted at every loss call** (`probing/probes.py:191-199`):
`pred.shape == (B, Q, H)` — quantiles on dim −2, horizon on dim −1.

For `ext_v4`: `Q = 9` (q9) or `1` (q1), `H = 64` ⇒ prediction shape `(n, 9, 64)` / `(n, 1, 64)`.
For `ext_v5`: `Q = 21` ⇒ `(n, 21, 64)`.

---

## 3. Datasets and ID/OOD Protocol

### 3.1 What `extended_v3_rolling` means — **do not conflate**

`extended_v3_rolling` is a **dataset-set identifier**, defined once at `probing/id_data.py:74-81` and
selected via `probing.config.DATASET_SET`. It simultaneously fixes **four** things:

1. **The dataset roster** — the 4 PT-ID tags (`ID_DATASET_SPECS["extended_v3_rolling"]`).
2. **The window/split construction** — membership in `ROLLING_SETS` (`id_data.py:100`) routes
   `build_windows` to `_build_rolling_windows` (`id_data.py:374-377`).
3. **The window budget** — `BUDGET_BY_SET["extended_v3_rolling"] = (1394, 262, 262)` = (train, val, test)
   (`id_data.py:87-98`).
4. **The feature-cache namespace** — `_idf_prefix` yields `IDF_<tag>__extended_v3_rolling`
   (`probing/extraction.py:82-93`).

It is **not** the experiment name and (uniquely here) **not** the output directory. `ext_v4` deliberately
overrides the output root while keeping the dataset set:

```python
PTID_SET = "extended_v3_rolling"   # roster + rolling windows + cache namespace
READOUT  = "fslot"                 # the readout that defines this experiment
OUT_ROOT = config.REPO_ROOT / "results" / "ext_v4_future_tokens"
```
(`experiments/run_ptood_probing_ftok.py:85-87`, documented as a deliberate exception at `:19-26`).

**The scientific experiment is defined by the READOUT** (shared forecast-token / fslot), not the dataset
set. The pooled-`content` sibling run on the *same* datasets and *same* windows lives in
`results/extended_v3_rolling/` (`experiments/run_ptood_probing.py`).

### 3.2 Roster and role

| Tag | HF config (repo `autogluon/chronos_datasets`) | Short | Pretraining status | Role in ext_v4 |
|---|---|---|---|---|
| `monash_electricity_hourly` | `monash_electricity_hourly` | Electricity | **PT-ID** | probe source **and** target |
| `uber_tlc_hourly` | `uber_tlc_hourly` | Uber | **PT-ID** | probe source **and** target |
| `m4_hourly` | `m4_hourly` | M4 | **PT-ID** | probe source **and** target |
| `wind_farms_hourly` | `wind_farms_hourly` | WindFarms | **PT-ID** | probe source **and** target |
| `sg_carpark` | TIME `Real-TSF/TIME : SG_Carpark/15T` (→ hourly) | SG Carpark | **PT-OOD** | target only |
| `coastal_ts` | TIME `Real-TSF/TIME : Coastal_T_S/H` | Coastal T-S | **PT-OOD** | target only |
| `boom_hourly` | `Datadog/BOOM` (native-hourly subset) | BOOM | **PT-OOD** | target only |

Roster constants: `probing/tunnel.py:42-43` (`PT_ID_TAGS`, `PT_OOD_TAGS`);
`probing/id_data.py:489` (`OOD_TARGET_TAGS`). All seven are **hourly**, so the seasonal period is
`m = 24` everywhere (`experiments/run_id_forecasting.py:141`, `M_SEASON = 24`).

Provenance: `data/chronos2_seen_manifest.md` (PT-ID basis = Table 6 of arXiv:2510.15821) and
`data/ood_targets_manifest.md` (PT-OOD basis + preprocessing + licences + limitations).
Honest limitation, already written down: *"No proof of zero overlap"* for SG Carpark / Coastal T-S —
they are absent from the manifest and post-date the model release, but their underlying feeds are public
(`data/ood_targets_manifest.md`, "Known limitations" #1).

### 3.3 Terminology (three orthogonal axes — the repo never uses a bare "ID"/"OOD")

Defined in `probing/tunnel.py:1-9` and `experiments/run_fslot_transfer.py:1-13, 80-91`:

* **PT-ID / PT-OOD** — was the **TARGET** in Chronos-2's pretraining corpus. *`pt_status` always
  describes the target, never the source.*
* **Probe-ID / Probe-OOD** — was the frozen probe fit on the same dataset it is scored on
  (`Probe-ID ⇔ source == target`).
* **FT-ID / FT-OOD** — reserved for the fine-tuning lines; **unused in ext_v4** (`adaptation: None` in
  every ext_v4 record; `tunnel.py:46-54`).

Four quadrants:

| Quadrant | Where it lives in ext_v4 |
|---|---|
| PT-ID / Probe-ID | 4×4 **diagonal** (`run_fslot_transfer --experiment transfer_4x4`) |
| PT-ID / Probe-OOD | 4×4 **off-diagonal** (12 cells) |
| PT-OOD / Probe-OOD | 4×3 unseen grid (`run_fslot_transfer --experiment pt_ood`) |
| PT-OOD / Probe-ID | the **fresh-target-probe diagnostic**, `run_ptood_probing_ftok` default mode — *not* a transfer experiment (`run_ptood_probing_ftok.py:1-8, 105-110`) |

### 3.4 Split construction

**PT-ID (rolling-origin within-series)** — `probing/id_data.py:226-352` (`_build_rolling_windows`):

1. Candidate context-starts step by **`H = 64`**, so forecast **targets never overlap**
   (`_rolling_valid_starts`, `id_data.py:179-193`). A start is valid iff context+target are finite and
   `ctx.std() ≥ 1e-6`.
2. A series is **eligible** iff `len ≥ C + 3H = 704` and it has ≥3 valid origins.
3. Per eligible series: **last** origin → **test**, **2nd-last** → **val**, all earlier → **train**.
   Within a series every train target precedes the val target, which precedes the test target.
4. The **same deterministic** `target_val (=262)` series carry **both** the val and the test window
   (one each), drawn by `rng.permutation` with `seed = SEED = 0` (`id_data.py:276-278`).
5. Train windows come from **every** eligible series, cluster-balanced round-robin down to 1394
   (`_cluster_balanced_order`, `id_data.py:626`). **Fail-loud** if any selected val/test series retains
   no train window (`id_data.py:307-311`).
6. `split_mode = "rolling_origin_within_series"`, `mase_canonical = True`; the per-test-window MASE
   denominator stored in `test_denominator` is the seasonal-naive scale of that series' history
   **strictly before** the test target (`id_data.py:294`).

**Realised counts, all four PT-ID datasets: n_train = 1394, n_val = 262, n_test = 262, with 262 test
series ⇒ exactly 1 test window per series.** Evidence: `BUDGET_BY_SET` (`id_data.py:98`) and every
transfer row: `n_test_windows = 262`, `n_test_clusters = 262`
(`results/ext_v4_future_tokens/q9/cross_dataset/tables/transfer_summary__4x4__q9.csv`), and every tunnel
record `"n_windows": 262, "n_clusters": 262`.

**PT-OOD (rolling-origin, cluster-aware)** — `probing/id_data.py:724-830` (`build_ood_rolling_windows`):
same per-series protocol, with three forced differences (`id_data.py:740-751`):
* the balancing / bootstrap unit is the **parent cluster** (carpark / station / metric-query), not the
  series;
* `target_val = target_test = None` ⇒ **every** eligible series contributes its one val + one test window
  (no 262-cap);
* `target_train = 1394` (matched to the PT-ID budget), cluster-balanced.

Realised counts, read from `results/ext_v4_future_tokens/ptood_probing/per_target/<tag>__q9__seed0.json`
(`meta`):

| Target | cluster unit | series total | eligible | train | val | test | test clusters | train before subsample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sg_carpark` | carpark | 354 | 354 | 1394 | 354 | **354** | **354** | 16 283 |
| `coastal_ts` | station | 48 | 48 | 1394 | 48 | **48** | **24** | 3 586 |
| `boom_hourly` | metric_query | 356 | 354 | 1394 | 354 | **354** | **354** | 20 151 (2 series too short) |

Preprocessing per PT-OOD target (`probing/id_data.py:509-610`, `data/ood_targets_manifest.md:43-84`):
* **SG Carpark**: target = available-lot **count**; 15-min → hourly = mean of the *available* samples,
  requiring **≥3 of 4** present (`MIN_SG_SAMPLES_PER_HOUR = 3`, `id_data.py:509`); no fill, no cross-hour
  interpolation. (Requiring all 4 yielded **zero** windows — a systematic daily single-sample gap.)
* **Coastal T-S**: multivariate length-3 reduced to `TEMP` + `PSAL` only (`PRES_REL` dropped) →
  24 stations × 2 = 48 univariate series, clustered by **station**
  (`COASTAL_VARIATES`, `id_data.py:492`). Only **24 clusters** ⇒ widest CIs; treat nulls as under-powered.
  Also partly **tidal (~12.4 h)** while `m = 24` is kept for consistency.
* **BOOM**: **one** quality-passing variate per hourly metric query, pinned in the committed manifest
  `data/boom_hourly_selection.json` (356 selected / 22 dropped, missing-fraction ≤ 0.20), clustered by
  **parent metric query** (`id_data.py:591-609`).

### 3.5 Source → target matrix actually run

**Experiment 1 — 4×4 cross-dataset transfer** (`run_fslot_transfer --experiment transfer_4x4`), all
16 cells present, both `q9` and `q1`
(`results/ext_v4_future_tokens/{q9,q1}/cross_dataset/tables/transfer_summary__4x4__*.csv`):

| source ↓ / target → | Electricity | Uber | M4 | WindFarms |
|---|:--:|:--:|:--:|:--:|
| **Electricity** | Probe-ID | Probe-OOD | Probe-OOD | Probe-OOD |
| **Uber** | Probe-OOD | Probe-ID | Probe-OOD | Probe-OOD |
| **M4** | Probe-OOD | Probe-OOD | Probe-ID | Probe-OOD |
| **WindFarms** | Probe-OOD | Probe-OOD | Probe-OOD | Probe-ID |

All targets PT-ID; every cell `n_test = 262`, `n_clusters = 262`.

**Experiment 2 — 4×3 unseen transfer** (`--experiment pt_ood`), all 12 cells, both `q9` and `q1`
(`results/ext_v4_future_tokens/{q9,q1}/unseen/tables/transfer_summary__pt_ood__*.csv`). All cells are
PT-OOD / Probe-OOD; there is **no diagonal**, hence **no transfer gap** (the `transfer_gap` column is
empty in these files by design — `run_fslot_transfer.py:359`, `gap=None`).

| source ↓ / target → | SG Carpark (354/354) | Coastal T-S (48/24) | BOOM (354/354) |
|---|:--:|:--:|:--:|
| Electricity | ✓ | ✓ | ✓ |
| Uber | ✓ | ✓ | ✓ |
| M4 | ✓ | ✓ | ✓ |
| WindFarms | ✓ | ✓ | ✓ |

*(cell = n_test windows / n_test clusters)*

**Diagnostic — PT-OOD / Probe-ID** (fresh probe fit on each PT-OOD target, `run_ptood_probing_ftok`
default mode): 3 targets × 3 seeds. ⚠ **Only exists at the legacy protocol** (see §13).

**ext_v5** (§6) covers **all 7** datasets, single run each.

---

## 4. Linear Probe

### 4.1 What it is

The **shared forecast-token probe** (`family = "shared_forecast"`, code name `fslot`). One probe per
representation point; **the same linear map is applied to every forecast slot** (weights shared across
slots), mirroring the native head's own weight-sharing.

**Mapping.** For layer ℓ, with slot states `h_{ℓ,k} ∈ R^768` (k = 1..K):
```
ĥ_{ℓ,k} = StandardScaler_ℓ(h_{ℓ,k})                     # one scaler shared across slots
p_{ℓ,k} = W_ℓ ĥ_{ℓ,k} + b_ℓ ∈ R^{Q·P}                   # nn.Linear(768, Q*P), shared over k
ŷ_ℓ     = trim_H( concat_k reshape(p_{ℓ,k}, (Q, P)) ) ∈ R^{Q×H}
```

| Property | Value | Evidence |
|---|---|---|
| Module | `torch.nn.Linear(768, Q*P)`, bias on | `probing/probes.py:640` |
| Input | `(n, K, 768)` = `(n, 4, 768)` | `probes.py:825-833` |
| Output | `(n, Q, H)` = `(n, 9, 64)` (q9) / `(n, 1, 64)` (q1) | `probes.py:605-619` |
| One probe per layer | yes — loop over `sorted(train_feats)`, 14 keys | `probes.py:837` |
| Shared across slots | **yes** — `nn.Linear` broadcasts over `(n, K)` | `probes.py:605-612`; unit-tested by permutation-equivariance (`tests/test_ood_capacity.py`) |
| Each slot processed independently | yes (no slot-index features, no cross-slot mixing) | same |
| Feature scaler | **one** `StandardScaler` fit on the train slots flattened to `(n·K, 768)` | `probes.py:621-628` |
| Trainable params / layer, **q9** | `768·144 + 144 = ` **110 736** | verified by loading `.../q9__v2__seed0/L05.pt` → `weight (144, 768)`, `bias (144,)` |
| Trainable params / layer, **q1** | `768·16 + 16 = ` **12 304** | verified from `.../q1__v2__seed0/L05.pt` |
| Frozen | the **entire** Chronos-2 model (`requires_grad_(False)`, `eval()`); features are read from disk caches, so the backbone is not even in the graph | `probing/extraction.py:56-58`; probes consume numpy arrays |

So the "`Linear(768, Q·P)`" description is **exactly right**, with `P = 16`, `Q ∈ {9, 1}`
⇒ out-dim 144 (q9) or 16 (q1). *(Note: it is `Q·P`, not `Q·H`; the pooled-`content` probe used elsewhere
in the repo is the `Q·H` one — `probes.py:294`. Do not mix them up — see §4.5.)*

#### Why "shared", and what "Chronos-aligned" does *not* mean

Four points to get right in the method section, because three of them are easy to state falsely.

1. **Why the weights are shared.** Chronos-2's native head is a *single* module applied to
   `hidden_states[:, -K:]`, broadcasting over the token axis, so the **same weights map every forecast
   slot** (`chronos2/model.py:731-732`). The probe reproduces exactly that wiring: one
   `nn.Linear(768, Q·P)` per layer, reused across all K slots, each slot emitting its own `P`-step
   patch; the K patches are then **concatenated** along the horizon and trimmed to `H`
   (`probes.py:605-619`, mirroring the native `rearrange("b n (q p) -> b q (n p)")`).

2. **The native head is NOT linear.** `output_patch_embedding` is a `ResidualBlock`:
   `ReLU(W₁x) → W₂` (with dropout) **plus a linear skip** `W₃x`, summed; `768 → d_ff=3072 → Q·P = 336`
   (`chronos2/model.py:265-271`; `chronos2/layers.py:438-447`). So the probe matches the native head's
   **layout and weight-sharing — deliberately not its capacity or function class.** "Structurally
   comparable" is a claim about the *wiring*, not about being the same kind of function. That gap is the
   point of a probe; state it rather than letting a reader infer equivalence.

3. **`Q = 21` is wrong for this probe.** In ext_v4 the head is `Linear(768, 144)` (q9) or
   `Linear(768, 16)` (q1). **21 is the *native head's* quantile count** and appears only in ext_v5, where
   the frozen head is reused and the flag is hard-restricted to `q21`
   (`run_native_head_adapter.py:489`, `choices=["q21"]`). Writing "Linear(768, 21×16)" for the ext_v4
   probe is a paper-breaking error.

4. **The probe does not read the native head's input, except at one point.** Layers 0–12 are captured
   **pre**-final-norm (block forward hooks); the native head consumes the **post**-final-RMSNorm states.
   Only readout point 13 (`L12+LN`) is literally "the same input the native model uses" (§1.4). So
   "decoded with the same kind of shared readout as the native model" is exactly true at `L12+LN` and
   approximately true elsewhere.

### 4.2 Training

`probing/probes.py:631-664` (`_fit_shared_forecast_linear`) and `:792-861`
(`fit_shared_forecast_probe_explicit_val`).

| Detail | Value | Evidence |
|---|---|---|
| Objective | **Chronos-2's own quantile (pinball) loss**, formula + reduction copied verbatim from `chronos2/model.py`: `2·|(y − ŷ)·(1[y ≤ ŷ] − τ)|`, reduced `mean(horizon) → sum(quantiles) → mean(batch)` | `probes.py:201-221` |
| Optimizer | `torch.optim.AdamW`, **two param groups**: weight decays, **bias has `weight_decay = 0.0`** | `probes.py:641-643` |
| Learning rate | **1e-2**, constant | `probes.py:792-793` (`lr=1e-2` default); driver never overrides |
| Batching | **full batch** (whole train tensor per step) | `probes.py:644-652` — one `loss.backward()` per epoch, no dataloader |
| Epochs | **300**, fixed | `QUANTILE_EPOCHS = 300`, `run_ptood_probing_ftok.py:90` |
| LR scheduler | **none** | `probes.py:641-652` |
| Early stopping | **none** | same; ext_v5 config records `"early_stopping": false` explicitly |
| Checkpoint selection *within* a fit | **none** — the final-epoch weights are kept | `probes.py:653-659` |
| Regularisation | weight decay only, selected per layer on validation | below |
| Weight-decay grid | **`WD_GRID_V2 = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0)`** — 8 values | `probes.py:170`; `run_ptood_probing_ftok.py:91`; and stamped into every v2 checkpoint's `selection.val_loss_by_wd` (verified: keys `[1e-05, 0.0001, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0]`) |
| wd selection rule | For each candidate, fit on **FULL train**, score on the **explicit temporal validation split**; keep the chosen-wd full-train model (**no refit**). Validation never touches the scaler or the weights. | `probes.py:836-856` |
| Initialisation | PyTorch default `nn.Linear` init, seeded by `torch.manual_seed(init_seed)` immediately before construction | `probes.py:639-640` |
| Seeds | **3 independent probe runs, `init_seed ∈ {0, 1, 2}`** | `RUN_SEEDS = (0, 1, 2)`, `run_ptood_probing_ftok.py:92` |
| What varies across seeds | **only the Linear init** — the fit is full-batch/deterministic and the windows + cached features are fixed at `SEED = 0` | `probes.py:634-638`; `run_ptood_probing_ftok.py:27-30` |
| Seed aggregation | curves averaged across the 3 seeds; tunnels defined from the **mean validation curve**, never by averaging per-seed layer indices | `run_ptood_probing_ftok.py:467-480`; `probing/tunnel.py:125-160` |
| Final `.eval()` | yes (irrelevant for a bare Linear, but done) | `probes.py:658` |

`WD_GRID_V2` widened the legacy `(1e-5 … 1e-1)` grid because many layers were selecting the old
**maximum** while validation loss was still improving (`probes.py:164-171`). Empirical confirmation on
disk: Electricity q9 L5 selects `wd = 1.0`, q1 L5 selects `wd = 0.3` — both **outside** the legacy grid.
This is precisely why v2 results supersede the legacy ones (§13).

### 4.3 Transfer / prediction

`probing/probes.py:865-931` (`predict_shared_forecast_probe`): applies the **frozen** source scaler +
weights in `eval()`/`no_grad`, never mutates the probe, iterates `sorted(feats)` (14 keys). Diagnostics
optionally returned: `test_mean_pinball` (always), `test_median` (needs an exact 0.5 level — raises
rather than substituting a neighbour, `:914-918`), `test_window_loss` (per-window, `(n,)`).

### 4.4 MLP option — exists, but was **NOT** used in the final ext_v4 run

* Code: `probing/heads.py` (`ResidualBlock`, native-structure clone, `use_layer_norm=False`,
  `hidden = NATIVE_D_FF = 3072`), `probing/probes.py:1274-1536`
  (`fit/predict_forecast_slot_native_head*`), family plumbing at
  `experiments/run_ptood_probing_ftok.py:335-363` (`ProbeFamily`, `PROBE_FAMILIES["native_mlp"]`).
* Head size at q9/P=16: **2 915 616** params (26× the linear head) — arithmetic check:
  `768·3072+3072` + `3072·144+144` + `768·144+144` = 2 915 616. Dropout = native 0.1
  (`MLP_DROPOUT`, `run_ptood_probing_ftok.py:101`).
* Old MLP outputs exist under `results/ext_v4_future_tokens/fslot_mlp/` (dated **2026-08-18**, legacy
  protocol, no `probe_protocol_version` field).
* The **final** run explicitly excludes it: `job_full_q1_q9_rerun.sh:18-19` — *"NO MLP anywhere (every
  driver defaults `--probe-family shared_linear`; this script never passes `native_mlp`)"*; and
  `run_q1q9_compare.py:82` hard-sets `ftok.FAMILY = PROBE_FAMILIES["shared_linear"]`.

**→ Describe the probe as strictly linear. Mention the MLP only if you want to say a nonlinear
capacity control exists but is out of scope for these results.**

### 4.5 Pooled vs shared readout: three axes at once, and one confound

The two probes are often described as answering different questions — *"can the whole future be decoded
from one global vector?"* vs *"can each layer's forecast-slot states be decoded by the same kind of
shared readout the native model uses?"*. That framing is right in spirit, but the two probes differ on
**three axes simultaneously**, so a difference between their curves is not attributable to any one of
them.

| | pooled probe (`quantile`) | shared forecast-token probe (`shared_forecast`) |
|---|---|---|
| tokens read | the **32 context patches**, mean-pooled (`hs[:, :ncp, :].mean(1)`, `extraction.py:476`); the `reg` variant instead takes the single REG token (`:477`). REG and the forecast slots are excluded from content pooling. | the **K forecast slots**, `(n, K, 768)` (`extraction.py:478`) |
| weight sharing | one map for the whole horizon | one map reused per patch, K times |
| head | `nn.Linear(768, Q·H)` (`probes.py:268`) | `nn.Linear(768, Q·P)` (`probes.py:641`) |
| out-dim @ q9 / q1 | 576 / 64 | 144 / 16 |
| params @ q9 | **442,944** | **110,736** |
| params @ q1 | **49,216** | **12,304** |

The capacity ratio is exactly `H/P = K = 4` at this geometry — the pooled head has **K× more
parameters**, at every quantile set. Consequence to state plainly in the paper:

> **A lower fslot curve is not, by itself, evidence about the representation.**

The repo says this itself (`experiments/run_id_forecasting.py:718`: the shared probe *"has ~K× fewer
params + enforced patch-wise weight sharing, so a lower fslot_K curve is not"* conclusive), and the
README carries the same caveat (*"pooled and shared-slot probes differ in both representation **and**
readout capacity"*).

Also part of the probe, and easy to omit: a slot `StandardScaler` per layer, fit on the training slots
flattened to `(n·K, 768)` and **shared across slots**, is applied before the Linear
(`probes.py:621-628`, `:836`).

#### ⚠ The confound: the two lines come from different encoder passes

On `extended_v3_rolling` the pooled and shared numbers were **not** produced by the same forward pass:

* pooled → `experiments/run_ptood_probing.py:134-136, 262-264` calls `extract_window_features`;
* fslot → `experiments/run_ptood_probing_ftok.py:190` calls `extract_kout_features(..., horizon=64)`,
  i.e. a `num_output_patches = K = 4` pass.

Encoder self-attention is **non-causal**, so adding the K forecast slots changes *every* token state,
including the context tokens that the pooled probe averages. `probing/extraction.py:404-407` states this
outright:

> *"Because the encoder's self-attention is NON-causal, the context/REG states under K forecast tokens
> differ from the K=1 cache; we therefore re-derive content_K and reg_K from THIS pass so the
> pooled-vs-shared comparison is fully controlled."*

⇒ **The currently available pooled-vs-shared contrast on these datasets is confounded by the encoder
pass, not only by the readout.** Do not present the difference as a pure readout effect.

The controlled variants (`content_K` / `reg_K`, pooled from the *same* K=4 pass) exist in code and were
run **only for `extended_v1`** (`run_id_forecasting.py:360-362`). For `extended_v3_rolling` they have not
been run. A controlled comparison would very likely need **no GPU re-extraction**, on this evidence
chain: the writer stores all three views per cache file — `types = ("content", "reg", "fslot")` with
`save[f"{t}_L{i}"]` / `save[f"{t}_final"]` (`extraction.py:430`, `:533-534`) — and the required
`IDF_<tag>__extended_v3_rolling__<split>__clean__K4_H64.npz` files are asserted present by the final
job's preflight, which passed (`job_full_q1_q9_rerun.sh:57-67`). ⚠ Not directly confirmed here:
`features_cache/` is absent on the audit machine, so the *presence of the `content_L*` keys inside those
specific files* is inferred from the writer, not observed. Recorded as an available option;
**not run** as of this audit.

---

## 5. Native Chronos-2 Baseline

There are **two distinct** "native" references in this repo. Be precise about which you cite.

### 5.1 `ext_v5` native baseline — the **reconstructed** frozen native head (the rigorous one)

* **What it is.** The pretrained `model.output_patch_embedding` — a `ResidualBlock(768 → d_ff=3072 →
  21·16 = 336)` (`chronos2/model.py:265-271`; `chronos2/layers.py:414-447`) — applied to the extracted
  **post-final-RMSNorm forecast slots** (`final["fslot"]`, i.e. representation point 13).
* **Representation it consumes:** the **post-final-RMSNorm** states. Verified from source:
  `chronos2/model.py:190` (final norm) → `:731` (`forecast_embeds = hidden_states[:, -K:]`) →
  `:732` (`output_patch_embedding(forecast_embeds)`).
* **Code:** `probing/native_head_adapter.py:53-72` (`native_head_modules`) returns the *actual*
  pretrained `output_patch_embedding` and `encoder.final_layer_norm`, both `.eval()` with every
  parameter `requires_grad_(False)`. It is **never reimplemented**.
* **Frozen-parameter gate:** `experiments/run_native_head_adapter.py:180-191`
  asserts `sum(p.requires_grad for p in pipeline.model.parameters()) == 0`.
* **Reconstruction sanity check — YES, one was performed.**
  `experiments/run_native_head_adapter.py:216-231`:
  ```
  recon_max = max |native_qr[:, q=0.5, :] − pipeline.predict_quantiles(..., [0.5])|
  recon_rel = recon_max / (mean|pipe_med| + 1e-12)
  raise unless recon_rel < RECON_REL_TOL = 5e-3          # run_native_head_adapter.py:65
  ```
  The pipeline reference is `predict_quantiles(prediction_length=64, quantile_levels=[0.5])` on the very
  same windows, cached under an ext_v5-only key (`__<split>__nha_native_median_H64`,
  `run_native_head_adapter.py:147-169`).
  **The run completed for all 7 datasets**, so the gate passed by construction (a failure raises).
  ⚠ **The numerical value of `recon_rel` is NOT persisted anywhere on disk** (`logs/` is gitignored and
  empty here). ⇒ **UNVERIFIED numerically.** You may state *"reconstruction agreed with the pipeline
  forecast to within a 5×10⁻³ relative tolerance (enforced as a hard gate)"* but **not** a specific error.
* **Second gate — zero-shot at the reference point must equal native exactly**
  (`run_native_head_adapter.py:239-244`): `max|Δ MASE| == 0.0` asserted. Independently corroborated on
  disk at index **12** (§1.6): `zero_shot_mase(L12) == native_mase` to all 6 stored decimals on
  Electricity, and `gap_denominator = 0.0` for all 7 datasets.
* **Is it genuine zero-shot Chronos-2?** **Yes.** No parameter of the model is trained or fine-tuned;
  the head is the pretrained one; the check above ties it to `pipeline.predict_quantiles`. Single K=4
  pass (no autoregressive unrolling at H=64, since `K·P = H`).
* **Numbers on disk** (`results/ext_v5_native_head_adapter/tables/native_head_adapter__records__*.csv`,
  condition `native`, layer 13; 21 native quantiles; test split; original scale):

| Dataset | kind | native MASE | native WQL |
|---|---|---:|---:|
| Electricity | PT-ID | 0.836189 | 1.156007 |
| Uber | PT-ID | 0.806006 | 4.194076 |
| M4 | PT-ID | 0.928949 | 0.765341 |
| WindFarms | PT-ID | 4.354280 | 2.976095 |
| SG Carpark | PT-OOD | 0.597790 | 0.538593 |
| Coastal T-S | PT-OOD | 1.127465 | 0.154742 |
| BOOM | PT-OOD | 0.943638 | 5.457953 |

  *(WQL is scale-dependent — `Σ pinball / Σ|y|` — so it is comparable across methods **within** a
  dataset only.)*

### 5.2 The `run_id_forecasting` native median — a **different**, older reference

`experiments/run_id_forecasting.py:167-191` (`native_median_forecast`) calls
`pipeline.predict_quantiles(..., quantile_levels=[0.5])` directly and caches under
`IDF_<tag>__test__native_median_H64.npz`. This is what the older `extended_v1`/`extended_v2` MASE-vs-native
comparisons used. **It is not part of the `ext_v4` fslot outputs** — no `ext_v4` table contains a native
column. If the paper wants a native reference on the `ext_v4` windows, use the **ext_v5** numbers in §5.1
(same windows, same denominator, verified reconstruction).

---

## 6. Native-Head-Aligned Readout (`ext_v5`)

### 6.1 Identity

Directory `results/ext_v5_native_head_adapter/`; module `probing/native_head_adapter.py`; driver
`experiments/run_native_head_adapter.py`; job `job_native_head_adapter.sh`; tests
`tests/test_native_head_adapter.py` (10 CPU/synthetic tests). It is **deliberately isolated from
ext_v4**: it adds no probe to `probing.probes` and writes to a disjoint namespace
(`native_head_adapter.py:33-34`, asserted by `tests/test_native_head_adapter.py:174-181`).

**Status:** ran on GPU. Every config file records
`"git_commit": "b35bd646c541a8901063ca4e8a03fc41862bce77"` and
`"run_timestamp_utc": "2026-08-21T06:52…"`. ⚠ **That commit does not exist in this repository's
history** (`git cat-file` fails; HEAD is `1bf1b56`) — the run was made on a Narval-side working commit
that was never pushed. The *code* that produced the results is committed (in `1bf1b56`); the exact
run-time commit hash is **UNVERIFIED**.

### 6.2 The computation graph (from the implementation, not from a guess)

Three conditions, all flowing through the **actual pretrained** modules:

```
(1) native      :  h_13 (= post-final-RMSNorm slots)  ──────────────────────► HEAD ──► ŷ
(2) zero-shot   :  h_ℓ  ──────────────────────────────► RMSNorm_final ─────► HEAD ──► ŷ     (ℓ = 0..12)
                   h_13 ─────────────────────────────────────────────────► HEAD ──► ŷ      (ℓ = 13, no 2nd norm)
(3) adapter     :  h_ℓ  ──► A_ℓ ──► RMSNorm_final ─────────────────────────► HEAD ──► ŷ     (ℓ = 0..12)
```
where `HEAD = model.output_patch_embedding` (frozen) and `RMSNorm_final = model.encoder.final_layer_norm`
(frozen). Slot→prediction layout reuses `probes._apply_shared_head` verbatim, i.e. Chronos-2's own
`n k (q p) → n q (k p)` rearrange (`native_head_adapter.py:101-123`).
Then, for reporting: `ŷ_raw = mu + s·sinh(ŷ_norm)` (`run_native_head_adapter.py:136-145`).

Concretely, condition (3) written out:
`ŷ_ℓ = InvNorm( ApplyLayout( output_patch_embedding( RMSNorm_final( A_ℓ( h_ℓ ) ) ) ) )`.

### 6.3 The trainable module

| Property | Value | Evidence |
|---|---|---|
| Module | `nn.Linear(768, 768)` — **with bias** | `native_head_adapter.py:87-95` (`LinearAdapter`) |
| Initialisation | **Identity**: `W ← I`, `b ← 0` | `native_head_adapter.py:90-92` |
| Consequence of identity init | at step 0 the adapter path is **byte-identical** to zero-shot | `native_head_adapter.py:20-21`; `tests/test_native_head_adapter.py:59-68` |
| Trainable params | **590 592** (= 768·768 + 768) | `run_native_head_adapter.py:344`; every config `"adapter_param_count": 590592` |
| One adapter per layer | **yes**, ℓ = 0..12 (13 adapters) | `ADAPTER_LAYERS = list(range(NUM_LAYERS))`, `run_native_head_adapter.py:60`; `adapt_layers` excludes `REF_IDX` (`:246`) |
| Shared across slots | **yes** — `nn.Linear` applies to the last dim and broadcasts over `(n, K)` | `native_head_adapter.py:76-79`, `:94-95`; `tests/test_native_head_adapter.py:71-81` |
| Norm placement | RMSNorm is applied **after** the adapter, for ℓ = 0..12; **skipped** at ℓ = 13 (already post-RMS) | `native_head_adapter.py:119-123`, `apply_rms = (L != REF_IDX)`, `run_native_head_adapter.py:236`; no-double-norm test at `tests/test_native_head_adapter.py:84-92` |
| Native head frozen | **yes**, all params `requires_grad=False`, `.eval()` (dropout off) — but still in the autograd graph so gradients pass through to `A_ℓ` | `native_head_adapter.py:61-72` |
| Backbone frozen | **yes** — hard-asserted: 0 model params with `requires_grad=True` | `run_native_head_adapter.py:180-191`; `tests/test_native_head_adapter.py:107-118` |
| Anything else trained | **no** — `A_ℓ` only | same |
| The curve **terminates at native** | `pw[("linear_adapter", REF_IDX)] = native_pw` — index 13 is an alias for native, **not** a trained adapter | `run_native_head_adapter.py:259-260` |

**Why no adapter at L12+RMS:** training `A_13` would be dataset-specific adaptation of the native model
— a different experiment (`native_head_adapter.py:171-173`; `notes/PLAN.md` ext_v5 section).

### 6.4 Training details

| Detail | Value | Evidence |
|---|---|---|
| Loss | Chronos-2 quantile loss in **normalised (arcsinh) space** against `Y_*_traj`, with the head's **21 native quantiles** | `native_head_adapter.py:144-145`; `run_native_head_adapter.py:512` |
| Optimizer | `AdamW`, weight decays, **bias `weight_decay = 0.0`** (same convention as the fslot probe) | `native_head_adapter.py:139-141` |
| Learning rate | **1e-2** | `run_native_head_adapter.py:53` (`LR`) |
| Epochs | **300** (`--sanity`: 60) | `run_native_head_adapter.py:52`, `:67` |
| Batch | **full batch** | `native_head_adapter.py:143-150` |
| Scheduler / early stopping | none / none (`"early_stopping": false` in every config) | `native_head_adapter.py:143-150`; `run_native_head_adapter.py:343` |
| wd grid | `WD_GRID_V2` — the same 8 values (`--sanity`: 3) | `run_native_head_adapter.py:54`, `:68`; every config `"wd_grid"` |
| Selection rule | wd chosen on the **explicit validation split**; test never touches selection | `native_head_adapter.py:196-208`; signature test `tests/test_native_head_adapter.py:121-135` |
| Seeds | **ONE deterministic fit** per (dataset, layer, wd). Identity init + full batch ⇒ deterministic; there are no seed bands. Uncertainty comes from the **test cluster bootstrap**. | `native_head_adapter.py:83-85`, `:164-166`; configs `"seeds": "single deterministic fit (identity init)"`; determinism test `tests/test_native_head_adapter.py:138-148` |
| Fit data | each dataset's **own** train split (1394 windows) | `run_native_head_adapter.py:201-203, 249-252` |
| Val data | each dataset's own val split (262 PT-ID / all-eligible PT-OOD) | same |
| Eval data | that dataset's own test split | `run_native_head_adapter.py:254-258` |
| Datasets | **all 7** (4 PT-ID + 3 PT-OOD), each fit and evaluated **on itself** (no cross-dataset transfer here) | `ALL_TAGS`, `run_native_head_adapter.py:63`; 7 record files on disk, each 29 rows = 1 native + 14 zero-shot + 14 adapter |

### 6.5 What to call it

Ranked by literal implementation accuracy:

1. **"linear adapter into the frozen pretrained forecasting head"** / **"frozen-head linear adapter"** —
   most accurate. The whole model including the head is frozen; a single trainable `Linear(768,768)`
   sits in front of the frozen final-RMSNorm + head.
2. **"native-head-aligned readout"** — acceptable and descriptive, but "aligned" overstates: nothing
   aligns *representations to representations*; `A_ℓ` is supervised by the forecast target `Y`.
3. **"sandwich readout"** — not used anywhere in the code; avoid.

**The interpretation caveat is written into the code and stamped into every config file** — quote it
rather than paraphrase (`native_head_adapter.py:23-28`; `configs/*.json` `"note"`):

> *A_ℓ is supervised by the forecast target Y through the FROZEN native head; a low adapter loss shows a
> linear map of layer ℓ is **SUFFICIENT** to make it usable by the native head, **NOT** that it recovers
> the ℓ→L12 coordinate transform.*

The label-free alternative `min_A ‖RMS(A h_ℓ) − h_{L12+RMS}‖²` is explicitly **parked, not built**.

### 6.6 Derived diagnostic — gap-recovery ratio `R_ℓ`

`experiments/run_native_head_gap_recovery.py` (post-hoc, cache-only, no model load):
```
R_ℓ = ( MASE_zeroshot(ℓ) − MASE_adapter(ℓ) ) / ( MASE_zeroshot(ℓ) − MASE_native )
```
`R = 0` → adapter did nothing; `R = 1` → gap fully closed. **Not clipped**
(`run_native_head_gap_recovery.py:8-11`).
Two documented validity criteria (`:16-18, 51-56, 80-89`):
* point: `|gap_ℓ| ≥ DENOM_REL_TOL · max_k |gap_k|`, `DENOM_REL_TOL = 0.02`;
* bootstrap sign: `< DENOM_SIGN_FRAC = 0.01` of resamples may have `denominator ≤ 0`.
Failing points are written as `NaN` with a `valid_flag` (`undefined:gap~0` / `unstable:denom_sign`) and
omitted from the plotted curve — **never manufactured as 0 or 1**.
Plotted layers = 0..12 only (`PLOT_LAYERS`, `:55`); L12 is always `undefined:gap~0` by the §1.6
degeneracy. CI = percentile CI of the paired bootstrap ratio `(zb − ab)/(zb − nb)`; the central estimate
is the **plug-in ratio of means**, not the bootstrap mean (`:88-89`).

---

## 7. Effective Rank

**Where it belongs in the paper:** *Representation Diagnostics*.

### 7.1 Definition (project-wide, single source of truth)

`probing/spectral_metrics.py:1-19, 37-66` (`spectral_metrics`):
```
X   ∈ R^{N×d}                          representation matrix
Xc  = X − mean(X, axis=0)              column-centered (centering across examples)
s   = svdvals(Xc)                      singular values          (np.linalg.svd, compute_uv=False)
λ   = s²                               covariance-energy spectrum
p   = λ / Σλ                           normalized variance spectrum
H   = −Σ p_i log p_i                   spectral entropy (natural log)
effective_rank = exp(H)                                          (Roy & Vetterli, 2007)
```
* **Entropy-based effective rank.** Not participation ratio, not stable rank, not a hard numerical rank.
* **Centered: yes.** **Otherwise normalized: no** — no row/column scaling, no whitening. `exp(H)` is
  invariant to isotropic scaling anyway (`tests/test_spectral_metrics.py:44-50`).
* **SVD, not covariance eigendecomposition** (`spectral_metrics.py:45`).
* **Numerical stabilization:** `eps = 1e-12` — spectrum entries with `p < eps` contribute 0 to `H`
  (`:53-54`); a fully constant matrix returns rank 0, entropy 0 by convention (`:48-50`); all arithmetic
  in float64 (`:28-34`).
* Also recorded per layer (secondary): `spectral_entropy`, `pc1_fraction = p[0]`, `numerical_rank`
  (numpy `matrix_rank` default tolerance — **diagnostic only**).
* Note `rank(Xc) ≤ min(N−1, d)`, so every record carries `N` and `d` (`:13`).

### 7.2 What matrix is built (fslot readout)

`experiments/run_spectral.py:98-124` (`_load_features`), `--readout fslot`:
* rows = **every forecast slot of every window**: the `(N, K, 768)` cached slot tensor is
  reshaped to **`(N·K, 768)`** (`_stack`, `:113-118`). This is the locked geometry decision — *not*
  mean-over-K and *not* per-slot-separately (`run_spectral.py:10-12`, `notes/PLAN.md` v4 section).
* columns = the **768 hidden dimensions**.
* 14 layer keys: `fslot_L0..fslot_L12` + `fslot_final` (index 13) (`run_spectral.py:123-124`).
* Computed on the **cached probe input**, i.e. **before** the probe's internal `StandardScaler`
  (`representation_location = "probe_input"`, recorded in every record's `provenance.note`).

### 7.3 Sampling / aggregation

`experiments/run_spectral.py:137-190` (`analyze_dataset`):
* A **fixed row sample shared by every layer**: `--repr-sample-size` (default **4096**), drawn once with
  `np.random.default_rng(SEED=0)` **without replacement**, `np.sort`-ed; the *same* `ids` are used at every
  layer (`:141-149`). Sample ids are stored in the record.
* **Point estimate** = full-sample `spectral_metrics` on those 4096 rows.
* **Uncertainty** = `subsample_metrics` (`spectral_metrics.py:69-95`): **200** repeated subsamples
  **WITHOUT replacement**, each keeping `frac = 0.8` of rows (3277 of 4096), seed 0; reports
  `mean`, `std`, and a `[2.5%, 97.5%]` percentile interval over the subsample distribution.
  A with-replacement bootstrap is **deliberately not implemented** — duplicated rows deterministically
  deflate rank (`spectral_metrics.py:15-18`).
  ⚠ The docstring is explicit that this interval describes **subsample variability at the reduced size**,
  **not** a formal CI for the full-sample estimate (`:74-76`). Describe it as such.
* **Per layer:** yes (14 points). **Per dataset:** yes (one record per dataset). **Per seed:** **no** —
  the backbone is pretrained and deterministic, so `backbone_seed: null`, one curve per dataset
  (`run_spectral.py:172`). **Per channel:** N/A (univariate; slots are stacked as rows, not channels).
* **Split used: `train`** (`spectral__<tag>__fslot__probe_input__train.json`).

### 7.4 What is on disk

`results/ext_v4_future_tokens/spectral/spectral__<tag>__fslot__probe_input__train.json` for **all 7
datasets**, plus 3 figures in `spectral/figures/` (effective rank / spectral entropy / PC1 fraction by
layer, all datasets overlaid). Verified record (Electricity):
`sample_size = 4096` (of `n_available = 5576 = 1394·4`), `forecast_slot_count = 4`, 14 layers,
`subsample_protocol = {without_replacement, n_subsamples: 200, frac: 0.8, seed: 0}`.

| point | Emb | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L12+LN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eff. rank | 1.00 | 11.51 | 15.84 | 23.69 | 31.49 | 34.53 | 45.87 | 38.36 | 41.82 | 29.48 | 22.53 | 15.04 | 8.78 | 7.69 |

⚠ **Provenance caveat:** these records carry `"git_commit": "a5c099f…"` and `"computed": "2026-08-10"`,
i.e. they predate the v2 probe protocol. This is **harmless** — effective rank is computed from the
**feature caches**, which are protocol-independent (the caches are shared by q1/q9 and by every probe
family; `probing/extraction.py:398-431`). No probe parameter enters the calculation.

---

## 8. CKA

**Where it belongs in the paper:** *Representation Diagnostics*.

### 8.1 Formula and implementation

`probing/cka.py:9-23, 57-83`. **Linear CKA** (Kornblith et al. 2019), **feature-space (biased-HSIC)
form** — `O(d²)` memory, not `O(n²)`:

```
Xc = X − mean(X, axis=0),   Yc = Y − mean(Y, axis=0)          # column-centering only
CKA(X, Y) = ‖Xcᵀ Yc‖_F²  /  ( ‖Xcᵀ Xc‖_F · ‖Ycᵀ Yc‖_F )
```
* **Linear CKA, not kernel CKA.** No RBF, no kernel bandwidth.
* **Centering:** column (feature) centering of each matrix. No double-centering of a Gram matrix is
  needed — this is the equivalent feature-space form.
* **Normalization:** the denominator only. Invariant to orthogonal transforms and isotropic scaling; in
  `[0, 1]` up to float rounding. Unit-tested (`tests/test_cka.py:31-51`).
* **Degeneracy:** a constant/degenerate side returns `NaN`, never 0 (`cka.py:67-74`).
* All accumulation in **float64** (`cka.py:36-40`).

### 8.2 Input matrix and what counts as a sample

For the fslot line, layer representations are **folded**, not pooled or concatenated:
`cka.stack_slots` reshapes `(n, K, d) → (n·K, d)` row-major, so **row `w·K + k` is window `w`, slot `k`**
(`cka.py:148-159`). A 2-D input (content pooling) passes through unchanged.

⇒ **Each observation is one (window, forecast-slot) pair**; columns are the 768 hidden dims.

**Structural scientific rule, enforced in code:** CKA rows must be the **same examples**.
`require_matched_rows` raises on any row-count mismatch (`cka.py:43-54`), and
`cka_matrix(rows, cols)` re-checks row equality for the cross-set case (`cka.py:116-118`).
**There is no code path that pairs two different datasets' examples by position**
(`cka.py:17-20`; `experiments/run_cka_analysis.py:11-17`; test `tests/test_cka.py:120-131`).

### 8.3 Subsampling and seeds

`cka.subsample_indices(n, size, seed=0)` — deterministic, sorted, **without replacement**; returns
`arange(n)` when `size is None` or `size ≥ n` (`cka.py:162-169`). The **same** index set must be applied
to every layer/stage being compared (the caller's job; `run_cka_analysis.py:136-143`).
Driver default: `--max-rows` **`None` ⇒ no subsampling, all rows used**; `--seed 0`
(`run_cka_analysis.py:427-428`).
There is **no seed averaging** — CKA is computed on the frozen pretrained backbone's cached
representations, which are deterministic and probe-seed-independent
(`results/comparisons/cka/README.txt`: *"CKA is independent of the probe quantile set (backbone
representations are identical for q1/q9)"*).

### 8.4 Aggregation and what is on disk

**Within one dataset only.** Two summary forms:
* `cka_matrix(reps)` — symmetric **layer × layer** matrix; only the upper triangle computed, then
  mirrored (`cka.py:99-127`).
* `cka_to_reference(reps, ref_index=-1)` — CKA of every layer to the **last** point
  (`cka.py:138-142`).

Committed artifacts relevant to `ext_v4`:

| Path | Content |
|---|---|
| `results/cka/ext_v4_future_tokens_fslot/matrices/<tag>__fslot__layerxlayer.npy` | **14 × 14** fslot layer×layer CKA, for the **4 PT-ID** datasets |
| `results/cka/ext_v4_future_tokens_fslot/tables/<tag>__fslot__layerxlayer.csv` | same, labelled `Emb, L1..L12, L12+LN` |
| `results/cka/ext_v4_future_tokens_fslot/figures/id_cka_fslot_1x4.png` | 1×4 panel |
| `results/cka/extended_v3_rolling/…` | **13 × 13** *content-pooled* CKA (all 7 datasets) + `cka_to_final.csv` — a **different readout**, ends at pre-norm L12 |

Verified properties of `monash_electricity_hourly__fslot__layerxlayer.npy`: shape `(14, 14)`, float64,
diagonal exactly 1.0, symmetric, range `[0.1159, 1.0]`.

⚠ **`results/cka/ext_v4_future_tokens_fslot/` has NO producing script in the repository.**
`experiments/run_cka_analysis.py` writes `results/cka/{extended_v3_rolling, ft_specialization,
task_shift_classification, domain_vs_task}/` only — there is no `ext_v4_future_tokens_fslot` branch
(grep confirms zero hits outside the results dir). The files were added in commit `1bf1b56` by an
**ad-hoc script that was not committed**. The formula is certainly `probing.cka.cka_matrix` (labels,
shape, and value range all match), but the **exact split, subsample size, and seed used for these four
matrices are UNVERIFIED.** If you report them, either re-derive them with a committed script or state
the geometry (14 fslot points, `(n·K, 768)` rows) without claiming a specific n.

---

## 9. Metrics

Only metrics actually used by the `ext_v4` (and, where flagged, `ext_v5`) pipelines.

### 9.1 Chronos-2 quantile (pinball) loss — **the training + selection + primary test metric**

`probing/probes.py:201-221` (`chronos2_quantile_loss`), formula and reduction copied verbatim from
`chronos2/model.py`:
```
ℓ(ŷ, y) = 2·| (y − ŷ) · ( 1[y ≤ ŷ] − τ ) |                     elementwise over (B, Q, H)
L       = mean over horizon  →  SUM over quantiles  →  mean over batch
```
* **Lower is better.**
* Per-window variant `chronos2_quantile_loss_per_window` (`probes.py:223-233`) drops only the final batch
  mean, so `pw.mean()` reproduces the scalar exactly — this is what feeds the cluster bootstrap.
* ⚠ **Sums over quantiles**, so raw values are **not comparable across quantile sets** (q9 vs q1). Stated
  at `probes.py:207-211`.
* Used for: **training objective**, **validation model selection (wd)**, **test evaluation**, **plotting**
  (all `ext_v4` layerwise curves), and **all** tunnel/D/Δ statistics.
* Computed in the **normalised (arcsinh) target space**.

### 9.2 Mean pinball loss — cross-quantile-set comparability

`probing/probes.py:235-247`: `max(τ·e, (τ−1)·e)` with `e = y − ŷ`, mean over batch × horizon ×
quantiles ⇒ equals `chronos2_quantile_loss / (2Q)`. At `q = [0.5]` it equals `0.5·MAE`.
**Evaluation only** — never the training objective (rescaling would shift the AdamW weight-decay
balance). Returned as `diag["test_mean_pinball"]` whenever a `collect_*` flag is set
(`probes.py:923-925`). *Not tabulated in any committed ext_v4 table*; the q1-vs-q9 comparison instead uses
relative regret (§9.5).

### 9.3 MASE — the reported original-scale accuracy metric

Definition (`experiments/run_id_forecasting.py:152-164`, `experiments/run_fslot_transfer.py:188-201`):
```
mu, s      = per-window context mean, clamped std                          (_ctx_stats)
y_raw      = mu + s·sinh(Y_test_traj)
ŷ_raw(ℓ)   = mu + s·sinh( median prediction at layer ℓ )
d          = max( mean_t |x_t − x_{t−24}| over the context window, 1e-8 )  (_mase_denominator, m=24)
MASE(ℓ)    = mean over windows and horizon steps of  |y_raw − ŷ_raw| / d
```
* **Lower is better.** Averaged over `(n_windows × H)` — the per-window value is the mean over `H`, then
  the reported number is the mean over windows.
* Denominator = **in-context seasonal-naive**, `m = 24`, computed from the **context only** ⇒ leakage-free
  and identical for every layer/method scored on the same windows.
* Requires an exact 0.5 quantile level; `median_index` returns `None` otherwise and the code **raises**
  rather than substituting a neighbour (`probes.py:184-189, 914-918`). Both q9 and q1 contain 0.5.
* Used for: **test evaluation and plotting only** — never for training or selection.
* Where it appears: `mase`, `mase_at_selected`, `mase_at_reference` columns in the ext_v4 transfer tables;
  `test_mase` in every ext_v5 record (ext_v5's **primary, plotted** metric,
  `run_native_head_gap_recovery.py:52` `GAP_METRIC = "mase"`).
* ⚠ Note the split-level `test_denominator` array built by `id_data` (history-before-test seasonal naive)
  is a **different, unused** denominator for these tables; the ext_v4/ext_v5 MASE uses the **in-context**
  one. Both are leakage-free; do not conflate them.

### 9.4 WQL and median MAE — **ext_v5 only**

`experiments/run_native_head_adapter.py:100-111`:
* **median MAE**: `mean_t |y_raw − ŷ_median_raw|` per window, then mean over windows. Raw scale,
  lower better. Reported as `test_median_mae`. *(Purely diagnostic; scale-dependent.)*
* **WQL**: numerator and denominator kept **separate per window** so the bootstrap forms the ratio
  inside each replicate:
  ```
  num_w = 2 · Σ_{q,h} pinball(y, ŷ_q)          den_w = Σ_h |y_h|
  WQL   = Σ_w num_w / Σ_w den_w
  ```
  Lower better; scale-dependent ⇒ comparable **within** a dataset only.
* **Neither appears in any `ext_v4` table.** CRPS is **not computed anywhere in this repository**.

### 9.5 Derived / normalised quantities

| Name | Formula | Where | Interpretation |
|---|---|---|---|
| **Transfer gap** | `L_{s→t}(ℓ_s) / L_{t→t}(ℓ_t) − 1` | `run_fslot_transfer.py:283-287` | 0 on the diagonal **by construction**; both ℓ from validation. 4×4 only (no diagonal exists in the 4×3, so the column is empty there). |
| **Relative regret (supplementary)** | `(L(ℓ) − min_j L(j)) / min_j L(j)` | `run_fslot_transfer.py:203-208` | Explicitly flagged as unable to detect a uniformly-bad transfer (it is 0 at the per-target argmin). |
| **Relative regret shape (q1 vs q9)** | same, applied to the mean test curve | `run_q1q9_compare.py:66-72, 182-200` | The **only** scale-free way to compare q1 and q9 curves (raw losses live on different scales). |
| **D (tunnel-effect statistic)** | `D = (L_test(last) − L_test(ℓ_start)) / L_test(ℓ_start)` | `probing/tunnel.py:181-193` | `>0` ⇒ the final layer is **worse** than the tunnel entrance. `D_ID` on the source's own test set; `D_OOD` on a PT-OOD target at the *source's* `ℓ_start`. |
| **Δ (delta)** | `Δ(s,t) = D_OOD(s,t) − D_ID(s)` | `tunnel.py:209-216` | `>0` ⇒ late-layer degradation stronger PT-OOD. |
| **M (excursion)** | `M = max_{j ≥ ℓ_start} ( L(j)/L(last) − 1 )` | `tunnel.py:79-84` | Worst saturation violation inside the tunnel; **not** bounded by tol under the first-crossing rule. |
| **`R_ℓ` (gap recovery)** | `(MASE_zs(ℓ) − MASE_ad(ℓ)) / (MASE_zs(ℓ) − MASE_native)` | `run_native_head_gap_recovery.py:8` | ext_v5 only. Not clipped; NaN where the denominator collapses. |
| **Generalization gap** | `val − train` per layer | `run_q1q9_compare.py` (`generalization_gap.png`, `train_vs_val_table.csv`) | Overfitting diagnostic for the probe. |

---

## 10. Bootstrap and Statistical Aggregation

### 10.1 The resampler

`probing/stats.py:48-81`. This is a **series/cluster bootstrap**, not a window bootstrap, because test
windows within a series are correlated.

```python
def cluster_bootstrap_counts(n_series, B, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_series, np.full(n_series, 1/n_series), size=B)   # (B, S) counts

def cluster_bootstrap_apply(M, per_series_sum, per_series_count):
    return (M @ per_series_sum) / (M @ per_series_count)[:, None]             # (B, L)
```
* **Bootstrap unit = one whole cluster** (a series for PT-ID; a carpark / station / metric-query for
  PT-OOD — `probing/id_data.py:490`, `build_ood_rolling_windows` puts the cluster id into `series_*`).
* Sampling S clusters with replacement is *exactly* `Multinomial(S, uniform)`, so the window-mean under
  duplication is the closed-form matmul above — **exact, not approximate** (`stats.py:51-56`).
* Metrics are **first summed per cluster, then divided by the resampled cluster window count**, i.e.
  every window carries equal weight (a cluster with more windows contributes proportionally more) —
  matching the reported aggregate (`stats.py:66-76`; `tunnel.py:166-178`).
* **Paired:** yes. **One shared count matrix `M` is generated per dataset and reused across every layer /
  condition / metric**, so all layerwise differences are computed inside the *same* resample
  (`tunnel.py:167-169, 177`; `run_native_head_adapter.py:376`; `run_native_head_gap_recovery.py:64`).
* **CI:** 95% **percentile** interval, `[2.5, 97.5]` along the replicate axis (`stats.py:79-81`).

### 10.2 Where the ratio is formed

Always **inside** each replicate, never as `raw-CI / constant`:
* `D`: `db = (boot[:, last] − boot[:, l_start]) / boot[:, l_start]`, then percentile CI
  (`tunnel.py:187-192`).
* `M`: the max over `j ≥ ℓ_start` is taken inside each replicate, boundary held fixed
  (`tunnel.py:196-206`). Documented bias warning: the max of a noisy ratio is biased upward.
* `WQL`: `Σnum / Σden` per replicate (`run_native_head_adapter.py:130-134`).
* `R_ℓ`: `(zb − ab) / (zb − nb)` per replicate (`run_native_head_gap_recovery.py:89`).

### 10.3 `Δ` is **NOT** paired — and the code says so

`Δ(s,t) = D_OOD(s,t) − D_ID(s)` spans **two disjoint test sets**, so its CI is the percentile CI of
`(boot_ood − boot_id)` where the two replicate vectors are **independent**
(`tunnel.py:209-216`; also spelled out in every ext_v4 record's `delta_ci_note`:
*"independent-replicate bootstrap difference (disjoint test sets); point stats on 3-run mean curves"*).

### 10.4 Number of resamples — **two different values, verify which you cite**

| Pipeline | `B` | Evidence |
|---|---:|---|
| **All ext_v4 statistics** (`d_stat_boot`, `delta_stat`, tunnel `d_id_ci`) | **2000** | `probing/config.py:34` `BOOT_B = 2000`; call sites `run_ptood_probing_ftok.py:485, 627, 640` pass `B=config.BOOT_B` |
| **ext_v5** (`--figures` CIs, gap-recovery CIs) | **5000** | `experiments/run_native_head_adapter.py:55` `BOOT_B = 5000`; every ext_v5 config records `"bootstrap_B": 5000` |
| Legacy `extended_v1` bootstrap driver | 5000 | `experiments/run_bootstrap.py` (out of scope for this paper) |

⚠ `notes/PLAN.md` repeatedly says "B=5000". **For ext_v4 that is wrong: B = 2000.** Only ext_v5 uses 5000.

**Seed: `SEED = 0` everywhere** (`probing/config.py:21`; passed explicitly at every call site).

### 10.5 Seeds vs bootstrap — how they combine

* The **3 probe seeds** produce three full 14-point curves per condition.
* Point statistics (`D_ID`, `M_test`, tunnel entrance) are computed on the **mean** curve across seeds
  (`tunnel.py:125-160`).
* The bootstrap operates on **seed-averaged per-window losses**: `_seed_mean_windows`
  (`run_ptood_probing_ftok.py:455-462`) averages the `(14, n)` per-window matrices across seeds, after
  **asserting** that all runs share identical window shapes and identical series ids.
* Per-seed `D` values are also retained (`d_id_by_run`) for a seed-sensitivity read.
* ext_v5 has **no seeds** (single deterministic fit) — all uncertainty is the cluster bootstrap.

### 10.6 ⚠ What has **NO** confidence intervals

**The 4×4 and 4×3 transfer results have no bootstrap CIs.**
`run_fslot_transfer.eval_cell` writes per-window losses + series ids to
`results/ext_v4_future_tokens/{q9,q1}/{cross_dataset,unseen}/bootstrap_inputs/*.npz`
(48 and 36 files respectively per quantile set), but **no script in the repository reads them**
(verified by grep: `BOOT_IN_DIR` appears only at the write site). The committed transfer tables report
**seed-mean point estimates only**.

So, in `ext_v4`, the **only** quantities with CIs are:
`D_ID` (in the tunnel records), and `D_OOD` / `Δ` (in `tunnel_effect_stats__fslot__q9__runs0-1-2.csv`,
which is the **legacy-protocol** file — see §13).

**Do not write "95% CIs" next to a transfer-gap number.**

---

## 11. End-to-End Evaluation Protocol

### 11.1 Vocabulary — four distinct operations, deliberately separated

| Operation | Definition here | Data touched |
|---|---|---|
| **Training a probe** | fitting `W_ℓ, b_ℓ` + the slot `StandardScaler` for one (dataset, layer, seed, wd) | **source train** only |
| **Selecting a probe** | picking the weight decay per layer | **source validation** only |
| **Selecting a layer** | picking ℓ* = argmin over the mean **validation** curve | **source validation** only |
| **Evaluating** | scoring the frozen probe | **target test** only |

### 11.2 Stage A — PT-ID source probes (`run_ptood_probing_ftok --fit-ptid`)

For each of the 4 PT-ID datasets, each of 14 layers, each of 3 seeds (`fit_ptid`,
`run_ptood_probing_ftok.py:369-408`):
1. `build_windows(src)` → the rolling split (window seed fixed at `SEED = 0`).
2. Extract/load 14-point fslot features for **train / val / test**.
3. `fit_shared_forecast_probe_explicit_val(f_tr, Y_train_traj, f_va, Y_val_traj, …, init_seed=seed)`:
   scaler + weights fit on **FULL train**; wd chosen on **val**.
4. Freeze and checkpoint all 14 heads (`_save_ckpt`, `probes.py`-compatible dict).
5. `predict_shared_forecast_probe` on **test**, with `collect_test_window_loss=True`.
6. Persist: per-seed JSON (`val_loss_by_layer` = per-layer min over the wd grid; `test_loss_by_layer`;
   full protocol metadata) and per-seed NPZ (`window_loss (14, 262)`, `series_test (262,)`).

Idempotence + anti-contamination: `_run_compatible` (`:170-183`) skips a seed only if its recorded
`(quantile_set, probe_protocol_version, wd_grid)` **exactly matches**; a stale/foreign result **raises**
rather than silently satisfying the skip.

### 11.3 Stage B — tunnels (`--tunnels-only`)

`compute_ptid_tunnels` (`run_ptood_probing_ftok.py:467-495`) → `tunnel_record_multi`
(`probing/tunnel.py:125-160`):
* Tunnel entrance from the **MEAN validation curve** (never per-seed indices averaged):
  ```
  ℓ_start = min { ℓ : mean_val(ℓ) ≤ (1 + tol)·mean_val(last) },  tol = 0.05     (tunnel.py:64-76)
  ```
  This is the **FIRST-CROSSING** rule (`tunnel_definition = "first_crossing_95"`), forward scan from the
  embedding, one-sided (a layer that *beats* the last layer qualifies). `last` = index 13 = `L12+LN`.
* `D_ID`, `M_test`, and the test-generalization check are then evaluated on the **mean test curve** with
  the boundary **frozen**; `d_id_ci` from the paired cluster bootstrap on seed-averaged per-window losses
  (`run_ptood_probing_ftok.py:483-489`).

**On-disk v2 results (both quantile sets, all 4 sources; `n_windows = n_clusters = 262`):**

| dataset | q | ℓ_start | val-argmin (ℓ*) | D_ID | 95% CI | M_test | test criterion holds |
|---|---|---:|---:|---:|---|---:|---|
| Electricity | q9 | L12 | L12+LN | −0.0397 | [−0.0492, −0.0306] | +0.0414 | True |
| Electricity | q1 | L10 | L12+LN | −0.0767 | [−0.0928, −0.0619] | +0.0891 | False |
| Uber | q9 | L4 | L9 | −0.0509 | [−0.0602, −0.0413] | +0.0537 | False |
| Uber | q1 | L3 | L9 | −0.0638 | [−0.0745, −0.0534] | +0.0682 | False |
| M4 | q9 | L3 | L3 | +0.0156 | [−0.0166, +0.0510] | +0.0399 | True |
| M4 | q1 | L3 | L12 | −0.0217 | [−0.0530, +0.0114] | +0.0312 | True |
| WindFarms | q9 | L3 | L5 | +0.0091 | [−0.0469, +0.0657] | +0.0387 | True |
| WindFarms | q1 | L3 | L5 | +0.0237 | [−0.0413, +0.0877] | +0.0360 | True |

*(source: `results/ext_v4_future_tokens/{q9,q1}/tunnels/<tag>__fslot__<q>__v2__runs0-1-2.json`)*

⚠ **ℓ_start (tunnel entrance) and ℓ\* (the layer used in the transfer experiments) are different
quantities.** ℓ_start is the first-crossing boundary; ℓ\* is `argmin` of the mean validation curve
(`run_fslot_transfer.py:123-127`). Both come from validation only.

### 11.4 Stage C — transfer, predict-only (`run_fslot_transfer`)

* **Nothing is trained.** The driver loads the frozen source checkpoints (`FAMILY.load_ckpt`) and applies
  `predict_shared_forecast_probe` to each target's test split
  (`run_fslot_transfer.py:214-229`, `:265-276`).
* **No parameter is refit on the target**: no target scaler, no target wd, no target layer selection ever
  runs off-diagonal (`run_fslot_transfer.py:14-17`). Enforced by test
  `tests/test_fslot_transfer.py:93` (`test_offdiagonal_eval_is_predict_only` — patches the fit functions
  to raise).
* **Layer selection** ℓ_s comes from the **source's own validation** curve, one per source row, reused for
  its Probe-ID diagonal cell **and** for all its Probe-OOD cells (`run_fslot_transfer.py:257-260`).
* **Preflights, fail-loud before any predict**: all 14 heads + scalers with consistent dims per
  source×seed (`:132-153`); the feature cache path must keep its `K4_H64` key and must **not** embed a
  probe-family tag, proving caches are family-shared (`:155-168`); label/series/meta counts must agree
  and the 14 feature keys must be exactly `0..13` (`:170-186`).
* **Seed aggregation**: per-cell curves averaged over the 3 seeds
  (`mean_ql = np.mean(v["ql"], axis=0)`, `:279-280`); each seed's per-window losses are also saved
  (unused downstream — §10.6).
* Verified source-validation-selected layers (**identical in both quantile sets except M4**):
  q9 → Electricity `L12+LN`, Uber `L9`, M4 `L3`, WindFarms `L5`;
  q1 → Electricity `L12+LN`, Uber `L9`, M4 `L12`, WindFarms `L5`.

### 11.5 Stage D — PT-OOD / Probe-ID diagnostic (fresh target probe)

`eval_target` (`run_ptood_probing_ftok.py:536-584`): builds the OOD rolling split, extracts train/val/test
fslot features, fits a **fresh** probe on target-train with wd on **target-val**, scores target-test.
Explicitly labelled a **DIAGNOSTIC, not a transfer experiment** (`:1-8, 105-110`). Then `aggregate`
(`:617-681`) applies each PT-ID source's frozen tunnel boundary to the target's mean test curve and
computes `D_OOD` and `Δ` with the bootstrap described in §10.

### 11.6 ext_v5 protocol

Per dataset (all 7), everything on the **same** windows:
native (no params) / zero-shot (no params, 14 layers) / adapter (13 trained `A_ℓ`, ℓ = 0..12; the ℓ=13
point is the native alias). Adapter trained on that dataset's train, wd on its val, scored on its test.
**No cross-dataset transfer, no layer selection, no seeds** (§6.4).

### 11.7 Leakage summary

* Test is never used for training, wd selection, layer selection, or tunnel definition — at any stage.
* Normalisation statistics are **per-window, context-only** — never per-split.
* PT-ID: within a series, train targets < val target < test target, all H-spaced and non-overlapping.
* The MASE denominator is computed from the context window only.

---

## 12. Experimental Constants Table

Verified values only.

| Quantity | Value | Evidence |
|---|---|---|
| Model checkpoint | `amazon/chronos-2` | `probing/extraction.py:49` |
| Library version | `chronos-forecasting==2.3.1` | `requirements.txt:16` |
| Precision | float32, `eval()`, `requires_grad=False` | `probing/extraction.py:49, 56-58` |
| Hidden size `d_model` | 768 | `results/ft_specialization/boom/manifest.json` |
| FFN width `d_ff` | 3072 | same |
| Encoder blocks | 12 | same |
| Native quantiles | 21 | same; `probing/probes.py:145-148` |
| Total params | 119 477 664 | same manifest |
| Context length `C` | 512 | `experiments/run_ptood_probing_ftok.py:88` |
| Horizon `H` | 64 | same |
| Patch length `P` | 16 | `probing/config.py:33` (asserted vs model, `extraction.py:464`) |
| Forecast slots `K` | 4 = ceil(H/P) | `run_ptood_probing_ftok.py:89`; `extraction.py:427` |
| Context patches `ncp` | 32 = ceil(C/P) | `extraction.py:470` |
| Special tokens | 1 (REG) | `extraction.py:469` |
| Encoder sequence length | 37 = 32 + 1 + 4 | `extraction.py:472, 516-518` |
| Representation points probed (fslot) | **14** (`Emb`, `L1..L12`, `L12+LN`) | `run_ptood_probing_ftok.py:113-114`; `_fslot_feats:185-198` |
| Final reference point | index 13 = post-final-**RMSNorm** slots | `run_ptood_probing_ftok.py:109-114`; `chronos2/layers.py:129-146` |
| Probe type | shared linear `nn.Linear(768, Q·16)`, one per layer, shared over K slots | `probing/probes.py:640, 605-619` |
| Probe params/layer, q9 | 110 736 | verified from checkpoint `L05.pt` |
| Probe params/layer, q1 | 12 304 | verified from checkpoint `L05.pt` |
| Feature standardisation | 1 `StandardScaler` per layer, fit on train slots `(n·K, 768)` | `probes.py:621-628` |
| Quantile sets (ext_v4) | q9 = 9 deciles; q1 = {0.5} | `probes.py:154-158`; `job_full_q1_q9_rerun.sh:37` |
| Quantile set (ext_v5) | q21 = the 21 native levels | `run_native_head_adapter.py:486-487` |
| Loss | Chronos-2 quantile loss (`mean_H → sum_Q → mean_B`) | `probes.py:201-221` |
| Optimizer | AdamW, wd on weight only (bias wd = 0) | `probes.py:641-643` |
| Learning rate | 1e-2 | `probes.py:792` |
| Epochs | 300, full batch | `run_ptood_probing_ftok.py:90`; `probes.py:644` |
| Scheduler / early stopping | none / none | `probes.py:641-659` |
| Weight-decay grid (v2) | (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0) — 8 values | `probes.py:170`; verified in every v2 checkpoint |
| wd selected on | explicit temporal **validation** split | `probes.py:836-856` |
| Probe seeds | **3** (`init_seed` = 0, 1, 2) — Linear init only | `run_ptood_probing_ftok.py:92`; `probes.py:639` |
| Global seed | 0 (windows, subsampling, bootstrap) | `probing/config.py:21` |
| Dataset set | `extended_v3_rolling` (roster + rolling split + cache namespace) | `probing/id_data.py:74-81`; `run_ptood_probing_ftok.py:85` |
| PT-ID window budget | 1394 train / 262 val / 262 test per dataset | `probing/id_data.py:98`; every transfer row |
| PT-ID test clusters | 262 (1 window per series) | tunnel records `n_clusters: 262` |
| PT-OOD budget | 1394 train; val = test = all eligible | `id_data.py:724-751` |
| PT-OOD test windows / clusters | SG 354/354 · Coastal 48/24 · BOOM 354/354 | `per_target/*__q9__seed0.json` `meta` |
| Rolling origin spacing | H = 64 (non-overlapping targets) | `id_data.py:179-193` |
| Eligibility | `len ≥ C + 3H = 704` and ≥3 valid origins | `id_data.py:257, 265-270` |
| Seasonal period `m` | 24 (all datasets hourly) | `run_id_forecasting.py:141` |
| MASE denominator | in-context seasonal-naive, clamped at 1e-8 | `run_id_forecasting.py:161-164`; `run_fslot_transfer.py:196` |
| Tunnel tolerance | 0.05 (95% of last-layer performance) | `probing/tunnel.py:40` |
| Tunnel criterion | **first crossing**, on the **mean validation** curve | `tunnel.py:64-76`, `:125-160` |
| Bootstrap type | paired **series/cluster** bootstrap, multinomial counts, one shared matrix | `probing/stats.py:58-76` |
| Bootstrap B (**ext_v4**) | **2000** | `probing/config.py:34` |
| Bootstrap B (**ext_v5**) | **5000** | `run_native_head_adapter.py:55`; ext_v5 configs |
| Bootstrap seed | 0 | `probing/config.py:21` |
| CI | 95% percentile [2.5, 97.5] | `stats.py:79-81` |
| Effective rank | `exp(−Σ p log p)`, `p = s²/Σs²` on centered `(N·K, 768)` | `probing/spectral_metrics.py:37-66`; `run_spectral.py:113-124` |
| Effective-rank sample | 4096 rows, shared across layers, seed 0 | `run_spectral.py:141-149` |
| Effective-rank uncertainty | 200 subsamples, frac 0.8, **without replacement**, seed 0 | `spectral_metrics.py:69-95` |
| CKA | linear (feature-space biased HSIC), column-centered, float64 | `probing/cka.py:57-83` |
| CKA rows (fslot) | `(n·K, 768)` — one row per (window, slot) | `cka.py:148-159` |
| CKA subsample default | none (`--max-rows None`), seed 0 | `run_cka_analysis.py:427-428` |
| ext_v5 adapter | `nn.Linear(768, 768)` **with bias**, identity init | `native_head_adapter.py:87-95` |
| ext_v5 adapter params | 590 592 | ext_v5 configs |
| ext_v5 adapters trained | 13 (layers 0..12); curve terminates at native at index 13 | `run_native_head_adapter.py:60, 246, 259-260` |
| ext_v5 seeds | 1 deterministic fit (no bands) | ext_v5 configs |
| ext_v5 reconstruction tolerance | rel < 5e-3, enforced as a hard gate | `run_native_head_adapter.py:65, 216-231` |

---

## 13. Current vs Superseded Implementations

### 13.1 SAFE / CURRENT — describe **this**

| Component | Canonical artifact | Marker |
|---|---|---|
| ext_v4 ID probes + tunnels | `results/ext_v4_future_tokens/{q9,q1}/{id,tunnels}/` | filename contains `__v2__`; JSON has `"probe_protocol_version": "v2"` and the 8-value `wd_grid` |
| ext_v4 4×4 cross-dataset transfer | `results/ext_v4_future_tokens/{q9,q1}/cross_dataset/tables/` | CSV column `probe_protocol_version = v2` |
| ext_v4 4×3 unseen transfer | `results/ext_v4_future_tokens/{q9,q1}/unseen/tables/` | same |
| Frozen source probes | `.../ptood_probing/ptid_checkpoints/<tag>__fslot__C512_H64__{q9,q1}__v2__seed{0,1,2}/L00..L13.pt` | `__v2__` in the dir name |
| Train-vs-val diagnostics | `results/ext_v4_future_tokens/{q9,q1}/id/tables/train_vs_val__*__v2__seed*.json` | `__v2__` |
| q1-vs-q9 comparison | `results/comparisons/q1_vs_q9/` | produced by `run_q1q9_compare` |
| Effective rank | `results/ext_v4_future_tokens/spectral/spectral__<tag>__fslot__probe_input__train.json` | protocol-independent (cache-derived) — safe despite the older git stamp |
| CKA (fslot, 4 PT-ID) | `results/cka/ext_v4_future_tokens_fslot/` | protocol-independent; ⚠ no committed producer (§8.4) |
| ext_v5 native-head adapter | `results/ext_v5_native_head_adapter/` | whole directory |
| Probe protocol | linear only, `WD_GRID_V2`, `PROBE_PROTOCOL_VERSION = "v2"` | `probing/probes.py:170-171` |
| Orchestration of the final run | `job_full_q1_q9_rerun.sh` (Block A) | commits `1bf1b56` |

### 13.2 SUPERSEDED / DO NOT CITE

1. **Legacy narrow-WD-grid q9 outputs (dated 2026-08-18).** Anything under
   `results/ext_v4_future_tokens/` in the **flat** layout — `tunnels/`, `fslot_transfer/`,
   `fslot_pt_ood/`, `paper_figures/`, `ptood_probing/tunnel_effect_stats__*` — was produced with the old
   `(1e-5 … 1e-1)` grid. Marker: **no `probe_protocol_version` field / no `__v2__` in the filename**.
   Concrete difference: Electricity's tunnel entrance moves **L10 (legacy) → L12 (v2)** and `D_ID`
   changes −0.0492 → −0.0397. The v2 grid matters: Electricity q9 L5 selects `wd = 1.0`, outside the old
   grid's maximum.
2. **The `native_mlp` (ResidualBlock) readout.** Code exists and old outputs exist
   (`results/ext_v4_future_tokens/fslot_mlp/`, 2026-08-18), but the final run **explicitly excludes it**
   (`job_full_q1_q9_rerun.sh:18-19`). **Describe the probe as strictly linear.**
3. **The "sustained plateau" tunnel criterion.** `notes/PLAN.md` has a long 2026-08-10 section declaring
   sustained-plateau the *sole* criterion. **The code was reverted** in commit `d7aa328`:
   `git log -p -- probing/tunnel.py` shows `"tunnel_definition": "sustained_plateau"` →
   `"first_crossing_95"`. **Every committed record says `first_crossing_95`.** Describe first-crossing.
4. **The stale field name `l_start_sustained`.** Present in every transfer CSV
   (`run_fslot_transfer.py:352`) but it stores `rec["l_start"]`, the **first-crossing** boundary. Same for
   `in_sustained_tunnel` in the by-layer tables. **The name lies; the value is first-crossing.**
5. **`B = 5000` for ext_v4.** `notes/PLAN.md` says 5000 throughout; ext_v4 actually uses
   `config.BOOT_B = 2000`. Only ext_v5 uses 5000.
6. **PT-OOD / Probe-ID diagnostic (`per_target/*.json`, `tunnel_effect_stats__*.csv`).** These exist
   **only at the legacy protocol** — the final rerun's Block A never runs that mode
   (`job_full_q1_q9_rerun.sh:85-88`). Verified: the on-disk `chosen_wd_by_layer` never exceeds 0.1.
   If you use the `D_OOD`/`Δ` numbers, you must label them legacy-protocol.
7. **`results/extended_v3_rolling/`** — the pooled-**content** sibling run (13 points, ends at pre-norm
   L12). Same datasets, **different readout**. Not `ext_v4`.
8. **`results/extended_v1/`, `results/extended_v2/`, `results/phase0_trio/`, `results/uea/`** — earlier
   lines with different rosters, budgets, splits, and readouts.
9. **The README's "Headline findings"** — `CLAUDE.md` explicitly says the README is not verified, and the
   README itself carries a "pending regeneration" note. Its numbers are from `extended_v1`, pooled
   content, a different split and budget. **Do not reuse them.**
10. **`results/ext_v4_future_tokens/ptood_probing/figures/plot_linear_train_vs_val.py`** — an ad-hoc
    script committed inside `results/`; superseded by `run_q1q9_compare --train-recompute` (it calls
    `load_ptid_ckpt(tag, "q9", …)`, which now resolves to the v2 dir, so its docstring is misleading).

### 13.3 Additional caveats that could make you state something false

* **`L12+LN` is RMSNorm.** Every `ext_v4` filename, axis label and column says `LN`.
  (`chronos2/layers.py:129-146`; ext_v5 already renamed it `L12+RMS`.)
* **`ℓ_start` ≠ `ℓ*`.** Tunnel entrance (first crossing) vs validation-argmin (used by transfer). Both
  validation-derived, different definitions, different values.
* **The `Emb` point on the fslot readout is DEGENERATE — it carries no per-window information.**
  `extract_kout_features` calls `model.encode(context=ctx, num_output_patches=K)` with **no future
  covariates** (`extraction.py:507`). With `future_covariates=None`, `_prepare_patched_future` builds the
  future patches as **all-zero values + all-zero mask**, leaving only the future *time encoding*
  `[0..h-1]/context_length` — which is identical for every window (`chronos2/model.py`,
  `_prepare_patched_future`, else-branch). `loc_scale` is used **only** when covariates are supplied, so
  nothing window-specific enters. Therefore, before any attention, the K forecast-slot embeddings are the
  **same K vectors for every window**. Corroborated on disk: the Electricity spectral record gives
  `effective_rank = 1.0007`, `pc1_fraction = 0.9999` at layer 0 over 4096 stacked slot rows
  (`spectral__monash_electricity_hourly__fslot__probe_input__train.json`) — i.e. effectively rank 1.
  **Consequences:** a probe at `Emb` can only emit a constant (per-slot) forecast, so its loss/MASE is a
  constant-forecast baseline, not a representation measurement; and the sharp drop from `Emb` to `L1` in
  every fslot figure is the point at which attention first copies context information into the forecast
  slots — *not* evidence that layer 1 "learns a lot". Do not describe `Emb` as "the input representation
  of the series" on this readout. (The pooled-`content` readout has no such problem: it averages the
  *context* patch embeddings, which are window-specific.)
* **Transfer results have no CIs** (§10.6). The `bootstrap_inputs/` NPZs are written but never consumed.
* **`transfer_gap` is empty in the 4×3 tables** — by design, there is no diagonal.
* **q9 and q1 raw losses are not comparable** (the loss sums over quantiles). Compare via relative regret
  or `mean_pinball_loss = loss/(2Q)`.
* **ext_v5 point estimates differ slightly between two tables.** `native_head_adapter__records__<tag>.csv`
  reports the **plug-in** mean; `native_head_adapter__relative_regret__all.csv` reports the **bootstrap
  mean** (`run_native_head_adapter.py:387`, `b.mean()`). Verified example (Electricity, native):
  0.836189 vs 0.836712. Pick one table and say which.
* **ext_v5 adapter at L12 is slightly worse than native** (Electricity: 0.852675 vs 0.836189) even though
  zero-shot@L12 equals native exactly — an artifact of fitting `A_12` at all. Don't present the adapter
  curve as monotonically approaching native.
* **Figures with no committed producer.** `results/ext_v4_future_tokens/q1/id/figures/`
  {`loss_and_erank_2x4.png`, `test_loss_1x4_tunnel.png`, `train_vs_val_1x4_tunnel.png`},
  `results/cka/*/figures/id_cka_1x4.png`, `id_cka_fslot_1x4.png`,
  `BOOM__within_stages_1x3.png`, and the whole `results/cka/ext_v4_future_tokens_fslot/` tree were added
  in `1bf1b56` by scripts **not in the repository**. Their underlying quantities are defined by committed
  code, but their exact plotting/sampling parameters are **UNVERIFIED**.
* **ext_v5's recorded git commit `b35bd646…` is not in this repo's history.**
* **PT-OOD status is "documented absence", not proven disjointness** (SG Carpark, Coastal T-S);
  BOOM is the clean case (explicitly listed as unseen). `data/ood_targets_manifest.md`, limitations §1.
* **Coastal T-S has only 24 bootstrap clusters** → widest CIs; a Coastal null is under-powered.
  Also partly tidal (~12.4 h) while `m = 24`.
* **Pooled vs shared is not pass-controlled on these datasets** (§4.5). The pooled curves
  (`results/extended_v3_rolling/`) come from a `extract_window_features` pass; the fslot curves
  (`results/ext_v4_future_tokens/`) from a `num_output_patches=4` pass. Attention is non-causal, so all
  token states differ between the two. Do not attribute their difference to the readout alone.
* **The shared probe is a linear analogue of a NONLINEAR head** (§4.1). `output_patch_embedding` is a
  `ResidualBlock` (ReLU MLP + linear skip, 768→3072→336). "Chronos-aligned" refers to layout and
  weight-sharing, never to capacity or function class.
* **`Q = 21` never applies to the ext_v4 probe.** It is q9 or q1 there (out-dim 144 or 16). 21 is the
  native head's quantile count and belongs only to ext_v5.
* **No pooled vector was ever fed to the pretrained native head anywhere in this repo.** Both ext_v4
  probes train *fresh* heads; the pretrained head is used only in ext_v5, and there it is fed forecast
  slots. The "a pooled vector is shape-compatible but semantically mismatched with the native head"
  argument is a sound **motivation** for the shared readout — do not write it as a defect that was found
  and then corrected.

---

## 14. Missing / Unverified Details

| # | Item | Status |
|---|---|---|
| 1 | Attention-head count (12) and `d_kv` (64) | **UNVERIFIED** from any artifact. Claimed only in `notes/PLAN.md`. Omit from the paper. |
| 2 | `context_length = 8192`, `max_output_patches = 64` | **UNVERIFIED** from artifacts (PLAN.md only). Not needed. |
| 3 | Numerical value of the ext_v5 native-reconstruction error | **UNVERIFIED** — the gate is enforced (`rel < 5e-3`) and the run completed, but the value is printed to a gitignored log. State the tolerance, not a number. |
| 4 | Exact sampling parameters (split, n rows, seed) behind `results/cka/ext_v4_future_tokens_fslot/` | **UNVERIFIED** — no committed producer. Formula/geometry are certain; sample size is not. |
| 5 | Plotting parameters for the ad-hoc composite figures (§13.3) | **UNVERIFIED** |
| 6 | Exact SLURM job IDs / wall-clock for the final q1/q9 rerun | **UNVERIFIED** (`logs/` gitignored and empty here) |
| 7 | ext_v5 run-time git commit `b35bd646…` | **UNVERIFIED** (not in this repo's history) |
| 8 | Whether the PT-OOD/Probe-ID diagnostic was ever re-run under protocol v2 | **Verified NO** — only legacy-protocol artifacts exist (`chosen_wd_by_layer` max = 0.1; no `__v2__` files) |
| 9 | Bootstrap CIs for the 4×4 / 4×3 transfer cells | **Verified ABSENT** — inputs written, no consumer |
| 10 | Any CRPS computation | **Verified ABSENT** — not implemented anywhere |
| 11 | Native-Chronos-2 baseline on the ext_v4 tables | **Verified ABSENT** in `ext_v4`; available only via ext_v5 (§5) |
| 12 | Presence of `content_L*` / `reg_L*` keys inside the `extended_v3_rolling` K4_H64 caches (§4.5) | **UNVERIFIED by direct inspection** — `features_cache/` is absent locally. Inferred from the writer (`extraction.py:430, 533-534`) plus the passing preflight that asserts those cache files exist (`job_full_q1_q9_rerun.sh:57-67`) |
| 13 | Whether a pooled probe was ever fit on same-K-pass features for `extended_v3_rolling` | **Verified NO** — `content_K`/`reg_K` probe fits appear only in `run_id_forecasting.py:360-362` (the `extended_v1` line) |

---

## 15. Repository Evidence Index

### Core modules
| Path | Role | Key symbols / lines |
|---|---|---|
| `probing/config.py` | constants + path namespacing | `SEED=0` (:21), `NUM_LAYERS=13` (:24), `LAST_LAYER=12` (:26), `OUTPUT_PATCH_SIZE=16` (:33), `BOOT_B=2000` (:34) |
| `probing/extraction.py` | model load + layer hooks + caching | `get_pipeline` (:35), `_idf_prefix` (:82), `extract_kout_features` (:398), `K` derivation (:427), `ncp` (:470), poolers (:476-478), pre-hook L0 (:495-502), block hooks (:503), `enc_out` post-norm (:507, 514) |
| `probing/probes.py` | probes + losses + quantile registry | `CHRONOS2_QUANTILES` (:145), `QUANTILE_SETS` (:154), `WD_GRID_V2` (:170), `PROBE_PROTOCOL_VERSION` (:171), loss (:201), per-window loss (:223), `mean_pinball_loss` (:235), `_apply_shared_head` (:605), `_fit_slot_scaler` (:621), `_fit_shared_forecast_linear` (:631), `fit_shared_forecast_probe_explicit_val` (:792), `predict_shared_forecast_probe` (:865), MLP head (:1274-1536) |
| `probing/id_data.py` | rosters + windows + splits + OOD loaders | `ID_DATASET_SPECS` (:45), `BUDGET_BY_SET` (:87), `ROLLING_SETS` (:100), `_make_examples` (:132), `_rolling_valid_starts` (:179), `_seasonal_naive_scale` (:195), `_build_rolling_windows` (:226), `build_windows` (:353), `OOD_TARGET_TAGS` (:489), `MIN_SG_SAMPLES_PER_HOUR` (:509), OOD loaders (:537-609), `build_ood_rolling_windows` (:724) |
| `probing/tunnel.py` | tunnel criterion + D/Δ/M stats | `TUNNEL_TOL=0.05` (:40), `PT_ID_TAGS`/`PT_OOD_TAGS` (:42-43), `tunnel_start` (:64), `max_excursion` (:79), `tunnel_record_multi` (:125), `_layer_mean_boot` (:166), `d_stat_boot` (:181), `m_stat_boot` (:196), `delta_stat` (:209), `val_curve_from_selection` (:219) |
| `probing/stats.py` | cluster bootstrap | `cluster_bootstrap_counts` (:58), `cluster_bootstrap_apply` (:66), `ci_bounds` (:79) |
| `probing/spectral_metrics.py` | effective rank | `spectral_metrics` (:37), `subsample_metrics` (:69) |
| `probing/cka.py` | linear CKA | `require_matched_rows` (:43), `linear_cka` (:77), `cka_matrix` (:99), `cka_to_reference` (:138), `stack_slots` (:148), `subsample_indices` (:162) |
| `probing/native_head_adapter.py` | ext_v5 adapter + frozen head reuse | `native_head_modules` (:53), `LinearAdapter` (:75), `slots_to_normalized_quantiles` (:101), `_fit_one_adapter` (:129), `fit_adapter_explicit_val` (:159) |
| `probing/heads.py` | native-structure MLP head (**unused in the final run**) | `NATIVE_D_FF=3072` (:36), `ResidualBlock` (:41) |

### Drivers
| Path | Role |
|---|---|
| `experiments/run_ptood_probing_ftok.py` | **ext_v4 primary driver** — ID probes, tunnels, PT-OOD diagnostic. Constants :85-116, `_fslot_feats` :185, `fit_ptid` :369, `compute_ptid_tunnels` :467, `eval_target` :536, `aggregate` :617 |
| `experiments/run_fslot_transfer.py` | **ext_v4 transfer** — 4×4 and 4×3, predict-only. `_val_selected_layer` :123, preflights :132-186, `_fslot_mase` :188, `eval_cell` :214, `run_4x4` :252, `run_pt_ood` :297, `_write_records` :333 |
| `experiments/run_native_head_adapter.py` | **ext_v5** driver. Constants :50-68, metrics :96-134, gates :180-244, `process_dataset` :193, figures/regret :412-480 |
| `experiments/run_native_head_gap_recovery.py` | ext_v5 `R_ℓ` diagnostic. Tolerances :51-55, `gap_recovery_curve` :58 |
| `experiments/run_spectral.py` | effective rank (`--readout fslot`). `_load_features` :98, `analyze_dataset` :137 |
| `experiments/run_cka_analysis.py` | CKA driver (extended_v3 / FT lines; **no ext_v4 branch**). readers :112-134, `run_extended_v3` :199 |
| `experiments/run_q1q9_compare.py` | q1-vs-q9 comparison + train-vs-val recompute + CKA collection |
| `experiments/run_id_forecasting.py` | source of `M_SEASON` (:141), `_ctx_stats` (:152), `_mase_denominator` (:161), `native_median_forecast` (:167) |

### Job scripts
| Path | Role |
|---|---|
| `job_full_q1_q9_rerun.sh` | **the final v2 run** — Block A = ext_v4 (ID + 4×4 + 4×3) for q9 and q1; explicit no-MLP; preflight asserts every required cache/checkpoint |
| `job_v4_future_tokens.sh` | the earlier (legacy-protocol) ext_v4 run + `run_spectral --readout fslot` |
| `job_native_head_adapter.sh` | ext_v5 (`--sanity` → `--adapt` → login `--figures`) |

### Tests (all CPU/synthetic, no model)
`tests/test_shared_forecast_transfer.py` (8; incl. 14-key post-LN point), `tests/test_fslot_transfer.py`
(13; incl. off-diagonal predict-only, diagonal gap = 0, one tunnel per row),
`tests/test_tunnel.py` (13; first-crossing semantics, paired/unpaired bootstrap),
`tests/test_quantile_sets.py` (11; loss formula vs explicit reference, param counts, layout),
`tests/test_spectral_metrics.py` (10), `tests/test_cka.py` (11; invariances + row-mismatch guard),
`tests/test_native_head_adapter.py` (10; identity==zero-shot, shared-A, no-double-RMS, only-adapter-grad,
val-only + deterministic, namespace disjoint).

### Data provenance
`data/chronos2_seen_manifest.md` (PT-ID basis, Table 6 of arXiv:2510.15821);
`data/ood_targets_manifest.md` (PT-OOD provenance, licences, preprocessing, limitations);
`data/boom_hourly_selection.json` (356 pinned BOOM variates).

### Key result artifacts
```
results/ext_v4_future_tokens/{q9,q1}/tunnels/<tag>__fslot__<q>__v2__runs0-1-2.json
results/ext_v4_future_tokens/{q9,q1}/cross_dataset/tables/transfer_{summary,by_layer,curves}__4x4__<q>.*
results/ext_v4_future_tokens/{q9,q1}/unseen/tables/transfer_{summary,by_layer,curves}__pt_ood__<q>.*
results/ext_v4_future_tokens/{q9,q1}/id/{figures,tables}/
results/ext_v4_future_tokens/ptood_probing/ptid_{runs,checkpoints}/
results/ext_v4_future_tokens/spectral/spectral__<tag>__fslot__probe_input__train.json
results/cka/ext_v4_future_tokens_fslot/{matrices,tables,figures}/
results/ext_v5_native_head_adapter/{configs,tables,plots,bootstrap_inputs}/
results/comparisons/q1_vs_q9/
results/ft_specialization/boom/manifest.json          # the model_config / param-count source
```

---

## Checklist for paper writing

Only statements verified above. Each is safe to write as-is.

**Model & extraction**
- [ ] Frozen `amazon/chronos-2` (chronos-forecasting 2.3.1), float32, `eval()`, all parameters
      `requires_grad=False`; the model is never trained or fine-tuned in this study.
- [ ] Encoder: 12 blocks, `d_model = 768`, `d_ff = 3072`, 21 native quantiles, patch size 16,
      ≈119.5M parameters.
- [ ] Used **univariately**: `group_ids=None`, so each series is its own group and no information mixes
      across the batch; Chronos-2's multivariate/covariate machinery is unprobed.
- [ ] **14 representation points** per window: the embedded token sequence entering block 1 (`Emb`, via a
      forward-**pre**-hook on `encoder.block[0]`), the 12 block outputs (`L1..L12`, forward hooks, all
      **pre**-final-norm), and the **post-final-RMSNorm** states (`L12+LN` in our filenames) — the tensor
      the pretrained output head actually consumes.
- [ ] `encoder.final_layer_norm` is a **T5-style RMSNorm** (no mean subtraction, no bias). If you keep the
      label `L12+LN`, define it as RMSNorm.
- [ ] Both pre-norm `L12` and post-norm `L12+LN` are evaluated; `L12+LN` is the **final reference** for
      every layerwise statistic.
- [ ] The probed tensor is the **forecast-slot** state: `(n, K, 768)` — window × slot × hidden — taken
      from a single `num_output_patches = K` forward pass. Encoder attention is non-causal, so all token
      states are extracted from that same pass.

**Task**
- [ ] Context `C = 512`, horizon `H = 64`, patch `P = 16`, `K = ceil(H/P) = 4` forecast slots
      (Chronos-2's own rule); token sequence = 32 context patches + 1 REG + 4 forecast slots = 37.
- [ ] Raw values go into the model; Chronos-2 applies its own per-window instance norm followed by
      `arcsinh`. Probe targets are constructed in that **same** space:
      `y = arcsinh((future − mean(context)) / std(context))`. Context-only statistics ⇒ leakage-free.
- [ ] Predictions are inverted for reporting with `mu + s·sinh(·)`, applied identically to ground truth
      and prediction.
- [ ] The K predicted patches are **concatenated** along the horizon (Chronos-2's own
      `n k (q p) → n q (k p)` layout) and trimmed to H; at H = 64 the trim is a no-op.
- [ ] Prediction shape `(n, Q, H)`; `Q = 9` (deciles) or `Q = 1` (median) for the probes,
      `Q = 21` for the native-head experiment.

**Data**
- [ ] Four PT-ID hourly datasets (Electricity, Uber, M4, WindFarms; all in Chronos-2's documented
      pretraining corpus) and three PT-OOD targets (SG Carpark, Coastal T-S, BOOM).
- [ ] Uniform **rolling-origin within-series** split, targets spaced by H so they never overlap; per
      eligible series (length ≥ 704, ≥3 valid origins) the last origin is test, the second-to-last is
      validation, all earlier ones are train.
- [ ] PT-ID: **1394 train / 262 validation / 262 test** windows per dataset, with **262 test series**
      (exactly one test window per series). Train windows are cluster-balanced across all eligible series.
- [ ] PT-OOD test sets: SG Carpark 354 windows / 354 carparks; Coastal T-S 48 / 24 stations; BOOM 354 /
      354 metric queries. Coastal's 24 clusters give the widest intervals.
- [ ] Windows with non-finite or near-constant context/target are **dropped, never imputed**.
- [ ] `pt_status` describes the **target**; `probe_status` is Probe-ID iff source == target.
- [ ] PT-OOD status = documented absence from the Chronos-2 manifest (BOOM is explicitly listed unseen);
      not a proof of zero corpus overlap.

**Probe**
- [ ] One **strictly linear** probe per representation point: `Linear(768, Q·16)`, the **same** weights
      applied independently to each of the K = 4 forecast slots, preceded by one `StandardScaler` per
      layer fit on the training slots.
- [ ] 110,736 trainable parameters per layer at Q = 9; 12,304 at Q = 1. The backbone contributes zero
      trainable parameters.
- [ ] Trained with **Chronos-2's own quantile (pinball) loss** — `2|(y−ŷ)(1[y≤ŷ]−τ)|`, mean over the
      horizon, summed over quantiles, mean over the batch — using AdamW (weight decay on the weight only,
      bias undecayed), lr = 1e-2, **300 full-batch epochs, no scheduler, no early stopping**.
- [ ] Weight decay selected **per layer on the explicit temporal validation split** from an 8-value grid
      {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3}; scaler and weights are fit on the **full** training
      split and the chosen-wd model is kept without refitting.
- [ ] **Three probe seeds (0, 1, 2)** varying only the linear layer's initialisation — the sole source of
      randomness in an otherwise deterministic full-batch fit. Windows, features, and the bootstrap all
      use seed 0.
- [ ] No nonlinear probe is used in these results.
- [ ] The probe is **Chronos-aligned in layout and weight-sharing only**: like the native head, one map is
      reused across all K forecast slots and each slot emits its own `P`-step patch, the K patches being
      concatenated along the horizon. Unlike the native head — a nonlinear `ResidualBlock`
      (768→3072→`Q·P`, ReLU MLP plus a linear skip) — the probe is strictly linear. It matches the
      wiring, deliberately not the capacity.
- [ ] The probe head predicts `Q·16` values per slot with **Q = 9 or Q = 1**; Q = 21 is the *native*
      head's quantile count and applies only to the native-head experiment.
- [ ] The pooled alternative, `Linear(768, Q·64)`, has exactly **K = H/P = 4× more parameters**
      (442,944 vs 110,736 at Q = 9), so a lower shared-readout curve reflects readout structure **and**
      capacity, not the representation alone.
- [ ] Layers 0–12 are read pre-final-norm while the native head consumes post-RMSNorm states, so only the
      final readout point is literally the native head's own input.

**Native head / adapter (if you include it)**
- [ ] The reference forecaster is genuine zero-shot Chronos-2: the pretrained
      `output_patch_embedding` applied to the post-final-RMSNorm forecast slots, reused verbatim (never
      reimplemented), with **every** model parameter frozen (hard-asserted at 0 trainable parameters).
- [ ] The reconstruction was validated against `pipeline.predict_quantiles` on the identical windows under
      a hard **relative-tolerance gate of 5×10⁻³** (state the tolerance, not a number).
- [ ] The adapter experiment trains a single `Linear(768, 768)` **with bias, identity-initialised**
      (590,592 parameters), one per layer for layers 0–12, **shared across the K slots**, inserted
      **before** the frozen final RMSNorm and the frozen native head:
      `h_ℓ → A_ℓ → RMSNorm_final → output_patch_embedding → ŷ`.
      At initialisation the adapter path is byte-identical to the zero-shot path; the curve terminates at
      the native baseline at the post-RMSNorm point (no adapter is trained there).
- [ ] Same optimiser recipe as the probe (AdamW, lr 1e-2, 300 full-batch epochs, no scheduler, no early
      stopping, same 8-value wd grid on validation), 21 native quantiles, **one deterministic fit** per
      layer (no seed bands).
- [ ] Because the post-RMSNorm point *is* RMSNorm(L12), pushing L12 through the native head reproduces the
      native forecast exactly — verified per-window on all seven datasets.
- [ ] The precise claim: a linear map of layer ℓ is **sufficient** to make it usable by the frozen native
      head; it does **not** recover the ℓ→L12 transform, because the adapter is supervised by the forecast
      target.
- [ ] The most literally accurate name is **"linear adapter into the frozen pretrained forecasting head"**
      (equivalently "frozen-head adapter").

**Diagnostics**
- [ ] **Effective rank** = `exp(−Σ_i p_i log p_i)` with `p_i = σ_i² / Σ_j σ_j²`, the σ from an SVD of the
      **column-centered** representation matrix (entropy-based effective rank, Roy & Vetterli 2007).
- [ ] The matrix has one row per **(window, forecast-slot)** pair and 768 columns; the K slots are stacked,
      not pooled. Computed on the training split, on the cached probe input (before the probe's scaler).
- [ ] 4096 rows sampled once (seed 0) and reused at every layer; uncertainty from **200 subsamples at 80%
      without replacement** — describe this as subsample variability at the reduced size, not a formal CI.
- [ ] One curve per dataset (the pretrained backbone is deterministic; no seed dimension).
- [ ] **Linear CKA** in its feature-space form
      `‖XᶜᵀYᶜ‖_F² / (‖XᶜᵀXᶜ‖_F ‖YᶜᵀYᶜ‖_F)` with column-centered matrices, float64.
- [ ] CKA observations are the same **(window, slot)** rows; comparisons are strictly **within** a
      dataset (layer × layer), never across datasets — enforced by a row-count guard.

**Statistics**
- [ ] Uncertainty comes from a **paired series/cluster bootstrap**: whole clusters (series for PT-ID;
      carpark / station / metric-query for PT-OOD) are resampled with replacement, implemented exactly as a
      multinomial count matrix, with **one shared count matrix per dataset** so all layers and conditions
      are compared inside the same resample.
- [ ] Ratios (D, M, WQL, R) are formed **inside** each replicate; intervals are 95% percentile intervals.
- [ ] **B = 2000, seed 0** for the layerwise probing statistics; **B = 5000, seed 0** for the native-head
      adapter results.
- [ ] Δ = D_OOD − D_ID spans two disjoint test sets, so its interval is an **independent-replicate**
      difference, not a paired one.
- [ ] Per-window losses are averaged across the three probe seeds (window alignment asserted) before
      bootstrapping; point statistics are computed on the seed-mean curves.
- [ ] The cross-dataset transfer tables report seed-mean point estimates **without** confidence intervals.

**Protocol**
- [ ] Probe weights and the feature scaler are fit on **source train**; weight decay and the comparison
      layer are chosen on **source validation**; everything is then frozen and evaluated on **target test**.
- [ ] The tunnel entrance is `ℓ_start = min{ℓ : mean-validation-loss(ℓ) ≤ 1.05 × mean-validation-loss(last)}`
      — a **first-crossing** rule at a 5% tolerance, computed on the mean validation curve across seeds,
      with `last` = the post-RMSNorm point. It is frozen before any test loss is inspected.
- [ ] The comparison layer used in the transfer experiments is `ℓ* = argmin` of the mean **validation**
      curve — a different quantity from `ℓ_start`; both are validation-derived.
- [ ] Transfer is **predict-only**: off the diagonal, no scaler, no weight decay, and no layer selection is
      ever refit on the target (enforced by tests that make the fitting functions raise).
- [ ] The transfer gap `L_{s→t}(ℓ_s)/L_{t→t}(ℓ_t) − 1` is 0 on the diagonal by construction and is defined
      for the 4×4 grid only.
- [ ] The test split never influences training, weight decay, layer selection, or the tunnel definition at
      any stage.
