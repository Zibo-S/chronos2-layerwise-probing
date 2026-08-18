# Working plan — chronos2-layerwise-probing
_Rolling notes. Edit freely; run `/plan` to fold in recent conversation._
_Last updated: 2026-08-17_

## TASK-SHIFT (FordA CLASSIFICATION) EXPERIMENT — 2026-08-17  [DESIGN CAPTURED — code NOT started]
Self-contained handoff: a fresh session can begin from THIS section. Repo state at drafting: branch
`tunnel-effect-probing`; the BOOM domain-shift Stage B (run_ft_specialization.py) is DONE/RUNNING and must
NOT be modified. This new experiment COEXISTS with it in a separate namespace. Login-node discipline applies
(no model load / probe fits / extraction on login node — salloc/sbatch). [[submit-slurm-jobs-self]]

### 0. STATUS + TWO DECISIONS RESOLVED (2026-08-18) + REUSE RE-VERIFIED FROM SOURCE
- **STATUS: ALL 5 FILES + 1 additive probes.py edit BUILT & CPU-VERIFIED 2026-08-18 (NOT committed;
  user reviewing diff). NEXT = C1 classification FT on GPU + validity gate.**
  Built: `probing/cls_data.py` (FordA loader + C0 smoke), `probing/finetune_cls.py` (Linear(768,2)+CE
  full-FT, epoch-1/best-val-after-stage1 rule, validity gate), `probing/probes.py` +=
  `fit_linear_cls_probe_explicit_val`/`predict_linear_cls_probe` (14-pt, sorted-keys, seed bands, wd on
  val-CE — additive, existing probes byte-identical), `experiments/run_task_shift.py` (Exp A + Exp B via
  import of run_ft_specialization's `target_windows`/`_role_split`/`_load_fslot`/`_fslot_feats_stage`;
  own Stage duck-types run_ft_specialization.Stage; Plots A/B/C + CKA; namespace
  results/task_shift_classification/), `tests/test_task_shift.py` (15 tests), `job_task_shift.sh`.
  VERIFIED (login CPU, OMP=2): 15/15 task_shift tests PASS; regressions green (ft_specialization 18,
  shared_forecast 8, quantile_sets 11, fslot_transfer 13, tunnel 13); C0 smoke RAN on real FordA
  (2881 train / 720 val / 1320 test, ~51/49 balanced, ncp=32, split invariants hold); full import chain
  + synthetic cls-probe end-to-end OK. run_ft_specialization / finetune.py / tunnel.py UNTOUCHED.
  COMMIT PENDING (user reviews diff first; code + results SEPARATE commits, NO Co-Authored-By
  [[no-coauthor-trailer]]). Do NOT run C1–C4 on the login node.
- **DECISION A = (a) EDIT FILES + SHOW GIT DIFFS** (user, 2026-08-18). I write the ~5 files directly and
  show `git diff` before ANY SLURM submit or commit; nothing committed/submitted without the user.
  [[scoped-go-ahead-with-diff]].
- **DECISION B = (a) TORCH `Linear(768,2)` + CROSS-ENTROPY** (user, 2026-08-18). AdamW, wd-grid on VAL,
  `init_seed` varied over probe seeds → genuine SEED BANDS (Plot A uncertainty). NOT sklearn LogReg.
- **STAGE-CHECKPOINT RULE (user amendment, 2026-08-18, OVERRIDES §5's fuzzy version):**
  `stage1_cls_early` = **end-of-epoch-1 checkpoint (deterministic)**. `stage2_cls_late` = **best-val
  checkpoint STRICTLY AFTER stage1, provided backbone drift has INCREASED vs stage1**. If no such later
  stage exists (best-val ≤ stage1 step, or drift did not grow) → **FAIL/REPORT** it (do NOT force a late
  stage). This is an FT-side validity gate; decided on val-acc + drift only, never on a probe curve.
- **FordA split-overlap TEST is SOURCE-AWARE (user amendment):** UCR TRAIN and TEST have SEPARATE index
  spaces, so the test asserts train∩val=∅ WITHIN the TRAIN index space and treats TEST as a disjoint
  array (never compares raw indices across TRAIN/TEST). Store split source + indices in meta accordingly.
- **14-pt probe loop keys off `sorted(feats)` / extracted keys** (never `range(NUM_LAYERS)`) so L12+LN
  (index 13) is never dropped — enforced in `fit_linear_cls_probe_explicit_val` and the driver.
- **RE-VERIFIED FACTS (source, 2026-08-18) that pin the implementation:**
  - `extract_kout_features` returns `feats["content"]={0..12}` (13 block pts) + `final["content"]` (post-LN)
    → the 14-pt cls feature dict = `[feats["content"][i] for i in range(13)] + [final["content"]]`.
    Content pooling inside extraction = `hs[:, :ncp, :].mean(1)`, `ncp=ceil(500/16)=32`.
  - `model.encode(ctx, num_output_patches=1)[0]` = the POST-final-LN state. So the cls-FT head forward
    `pooled = enc[:, :ncp, :].mean(1)` trains on EXACTLY the L12+LN probe point (index 13) — consistent,
    K=1 identical FT↔extraction.
  - **GOTCHA:** stock `linear_probe`/`fit_layerwise_probes` loop `range(NUM_LAYERS)=13` and would DROP the
    14th point (L12+LN). The cls probe (Decision B torch head) MUST loop `sorted(feats)` (14 keys), like the
    fslot probes already do. → new small `fit_linear_cls_probe_explicit_val` in `probing/probes.py`.
  - Import-only (source-agnostic) for Exp B: `run_ft_specialization.target_windows`, `._fslot_feats_stage`;
    `probes.fit/predict_shared_forecast_probe`. Build my OWN `Stage` with source=`forda_cls` (NOT reuse
    run_ft_specialization.Stage, which hard-codes FT_SOURCE="boom"); cache prefix via
    `ft_cache_prefix(tag,"forda_cls",stage,hash8)`. finetune.py helpers reused verbatim (no edits).

### 1. SCIENTIFIC OBJECTIVE
Two specialization axes, SAME frozen-then-fine-tuned Chronos-2, SAME layerwise-linear-probe lens:
- **Condition 1 — DOMAIN shift (done):** forecasting-pretrained → BOOM *forecasting* FT → forecasting probes.
  Result so far: curves mostly flat after early layers, NO convincing U-shape / late collapse.
- **Condition 2 — TASK shift (this experiment):** forecasting-pretrained → FordA *classification* FT →
  BOTH (A) classification probes AND (B) forecasting probes, layer-by-layer.
Question: **does changing the TASK create stronger late-layer specialization than changing only the
forecasting DOMAIN?** Possible (NOT required) signature: classification accessibility rises toward late
layers while forecasting accessibility falls late (a tunnel/U for the OLD task). **Do NOT engineer a
U-shape** — flat / no-specialization is a valid answer (§9 stopping rule of the BOOM plan applies verbatim).
LINEAR PROBES ONLY (no MLP / nonlinear heads) — leave existing MLP code untouched, do not integrate it.

### 2. REPO INSPECTION / REUSE MAP (verified from source 2026-08-17, no model load)
The infra generalizes cleanly — reuse, don't re-implement:
- `Chronos2Model.encode(context, num_output_patches=1)` is **fully differentiable** (no internal
  no_grad/detach; extraction wraps it in no_grad EXTERNALLY) and returns post-final-LN hidden states
  `enc_out[0]` of shape `(b, ncp+1+K, 768)` (model.py:569-635, 727-732). → a pooled classification head
  attaches and back-props into the whole ~119M backbone.
- `extract_kout_features(tag, split, contexts, y, horizon, pipeline=, cache_prefix=)` (extraction.py:398)
  ALREADY returns content-pooled features per block **plus** a post-LN `final["content"]` = **14 points**
  (Emb, L1..L12, **L12+LN**) AND has the `pipeline=`/`cache_prefix=` injection Stage A added. → the SAME
  function extracts classification features off a FT checkpoint into a namespaced cache. NO new extractor.
  (`extract_features` (UEA/aeon) gives only 13 pts and no injection → NOT used here.)
- `probing/finetune.py` reusables (task-agnostic): `load_trainable_pipeline`, `snapshot_reference_state`,
  `param_drift`, `save_checkpoint`, `checkpoint_hash` (sha256[:8]), `ft_cache_prefix(tag,source,stage,hash8)`
  → `IDF_<tag>__ft__<source>__<stage>__<hash8>`, `default_ckpt_root` ($SCRATCH), `_select_device`.
- `experiments/run_ft_specialization.py` **source-agnostic** helpers to IMPORT for Exp B (they never read
  FT_SOURCE): `target_windows(tag)` (build_windows for PT-ID / build_ood_rolling_windows for PT-OOD),
  `_fslot_feats_stage(tag,split,X,y,pipeline,cache_prefix)` (14-pt fslot line).
- `probing/probes.py`: `fit_shared_forecast_probe_explicit_val` + `predict_shared_forecast_probe`
  (linear shared-head, wd-grid on val, `init_seed`, per-window loss) = the EXISTING fslot forecasting probe,
  reused UNCHANGED for Exp B. `linear_probe`/`fit_layerwise_probes` (StandardScaler+LogReg) = the existing
  linear classification probe (Decision B option b).
- `probing/tunnel.py`: `val_curve_from_selection`, `tunnel_record_multi`, `d_stat_boot` — reused as-is if a
  tunnel overlay is wanted (optional for Plot A).
- FordA is in aeon (`aeon.datasets.load_classification("FordA", split="TRAIN"|"TEST")`, tsc_datasets.py:67):
  univariate, length 500, 2 classes {-1,1}, train 3601 / test 1320. Downloads from
  timeseriesclassification.com (login node has internet; aeon needs `module load arrow/24.0.0`).

### 3. PROPOSED ARCHITECTURE (5 new files; BOOM experiment UNTOUCHED)
- NEW `probing/cls_data.py` — FordA loader: `load_forda()` returns raw (n,500) float32 + labels; map
  {-1,1}→{0,1}; deterministic STRATIFIED train→train/val carve (seed=0, VAL_FRAC≈0.2), **store the exact
  indices + seed** in the returned meta; NO test leakage (UCR TEST kept separate). Generic shape so
  Wafer/ECG5000/UWave drop in later via a tag table (do NOT add them now).
- NEW `probing/finetune_cls.py` — classification full-FT (Stage-A analogue), reuses finetune.py helpers
  (§2). Head = `nn.Linear(768, C=2)` (NO hidden layer). Forward: `enc=model.encode(ctx,num_output_patches=1)
  [0]`; `pooled=enc[:, :ncp, :].mean(1)` (mean over the ncp=ceil(500/16)=32 content tokens = the ONE fixed
  pooling rule); `logits=head(pooled)`; `CrossEntropyLoss`. Trains backbone+head. Saves `stage1_cls_early`,
  `stage2_cls_late` checkpoints (HF safetensors via save_checkpoint) + `cls_head.pt` + manifest (hashes,
  hypers, per-checkpoint step/epoch/val-acc/val-loss, seed, LRs, split indices ref) + train/val histories.
- NEW `experiments/run_task_shift.py` — driver, modes mirror BOOM (`--extract/--probe/--figures`,
  `--forecast-extract/--forecast-probe`, `--cka`). Its own `Stage` (source=`forda_cls`), output namespace
  `results/task_shift_classification/`. Exp A = classification probes (§7). Exp B = forecasting probes via
  imported `target_windows`+`_fslot_feats_stage` + `fit/predict_shared_forecast_probe` (§8). Plot C (§9).
- NEW `tests/test_task_shift.py` (§12) — CPU/synthetic, no GPU/model/download.
- NEW `job_task_shift.sh` — C1–C5 sbatch stages (§13), cloned from job_ft_stageB.sh (modules, HF offline,
  FT_CKPT_ROOT, OOD_TARGET_ROOT).
- **NO edits to `run_ft_specialization.py` / `finetune.py`** (Exp B only IMPORTS their source-agnostic fns).
- Cache source label = `forda_cls`; checkpoints → `$SCRATCH/.../ft_specialization/forda_cls/<stage>/`
  (default_ckpt_root). Everything namespaced disjoint from BOOM (`__ft__forda_cls__` ≠ `__ft__boom__`).

### 4. FordA DATA (preprocessing = minimal + explicit)
- Feed sequences RAW (length 500) as `context`; NO padding/truncation (500 < 8192; Chronos-2 handles any
  length). The model applies its OWN per-instance InstanceNorm+arcsinh internally → NO external
  normalization, NO label leakage (per-series, not per-split). Document this as the entire preprocessing.
- Split: UCR TRAIN(3601)→ stratified train/val (seed 0, ~0.2 val); UCR TEST(1320) held out. Store indices.
- Labels {-1,1}→{0,1}; assert 2 classes, near-balanced (report the ratio → decide if balanced-accuracy is
  worth adding; keep accuracy + CE as primary).

### 5. CLASSIFICATION FINE-TUNING (Stage-A analogue; validity-gated like the BOOM pilot)
- Trainable = backbone (encoder + input_patch_embedding + REG-embed + final_layer_norm) + `Linear(768,2)`.
  The native `output_patch_embedding` gets NO gradient (not in the classification graph) → stays unchanged
  (fine; optionally freeze it for tidiness). This is a FEATURE: Exp B's fslot probe reads the encoder, which
  DOES change, so representational change is isolated cleanly.
- Optimizer: AdamW, TWO param groups — backbone LR≈1e-5 (conservative full-FT), head LR≈1e-3 (a fresh head
  won't learn at 1e-6). Linear decay, warmup≈5%, grad-clip 1.0, batch 64, seed 0. (These are the ONE
  data-scale-sensitive knobs; conservative, do NOT overengineer a sweep — but see the validity gate.)
- Budget ≈ 10 epochs (~57 steps/epoch → ~570 steps). Save checkpoints densely early + ~every epoch; record
  step/epoch/train-loss/val-loss/val-acc for each. Track train loss, val loss, val acc, LRs, seed.
- **Early/late designation (documented, NOT arbitrary):** `stage1_cls_early` = earliest saved checkpoint
  with a MEANINGFUL adaptation (val-acc clearly above stage0 AND non-trivial backbone drift), ~1 epoch;
  `stage2_cls_late` = **best-val-accuracy** checkpoint (the pre-overfit peak — FordA is small, so late ≠
  necessarily final epoch). Record the exact steps chosen + the rule. Fallback if late catastrophically
  overfits: keep best-val.
- **VALIDITY GATE (FT-side evidence only, like BOOM):** proceed to probing ONLY if (i) backbone param drift
  GROWS early→late and (ii) val-acc rises then plateaus/peaks (real specialization, not memorization). If
  the backbone barely moves → report "Chronos-2 robust to classification FT" (a finding), don't crank LR
  blindly. If it overfits instantly → reduce LR / epochs. Do NOT look at ANY forecasting-probe curve while
  choosing the budget/checkpoints.

### 6. EXTRACTION (one pass per checkpoint×dataset×split; injected + namespaced)
- Exp A (classification) features: `extract_kout_features(tag="forda", split, contexts=FordA, y, horizon=16
  (→K=1), pipeline=<stage pipeline or None for stage0>, cache_prefix=<forda_cls stage prefix or None>)`.
  Take `feats["content"][0..12]` + `final["content"]` = the 14-pt classification feature dict. K=1 matches
  the FT forward's num_output_patches=1 (non-causal attention ⇒ K must be fixed & identical FT↔probe). The
  fslot arrays it also computes are ignored (tiny at K=1).
- Exp B (forecasting) features: the EXISTING 14-pt fslot line (`_fslot_feats_stage`, horizon=64→K=4) on the
  forecasting TARGETS (not FordA). stage0 REUSES the committed pretrained fslot caches (BOOM/M4/Coastal
  already on disk from BOOM Stage B B1 — same pretrained backbone, same default namespace); the two FordA-FT
  stages extract fresh into `IDF_<target>__ft__forda_cls__<stage>__<hash8>`.

### 7. EXP A — layerwise CLASSIFICATION probes  →  PLOT A
- For each of 3 stages (stage0_pretrained / stage1_cls_early / stage2_cls_late) × 14 layers × probe seeds:
  fit a FRESH LINEAR classification probe (Decision B) on FordA train features, wd/C selected on VAL only,
  score on TEST. Backbone FROZEN during probing. Identical split/budget/selection/seeds across all layers +
  stages. NEVER select on test.
- Metrics (machine-readable CSV+JSON): test accuracy (primary) + test cross-entropy; balanced accuracy only
  if the class ratio warrants. Per-stage per-layer point + uncertainty (seed bands if Decision B=a; else a
  test bootstrap). **stage0 is the mandatory control** (was FordA already linearly decodable pre-FT?).
- PLOT A: x=Emb,L1..L12,L12+LN; y=test accuracy (or error/CE); 3 curves (stage0/early/late) + bands.

### 8. EXP B — layerwise FORECASTING probes after classification FT  →  PLOT B
- Reuse the fslot linear forecasting probe VERBATIM (fit_shared_forecast_probe_explicit_val +
  predict_shared_forecast_probe, q9/WQL, 14-pt, 3 seeds, wd-grid on target-val, per-window loss).
- INITIAL target subset (keep compute small; code makes expansion to all Stage-B targets trivial): **BOOM
  (boom_hourly, PT-OOD), M4 (m4_hourly, PT-ID), + one OOD (coastal_ts)**. `target_windows` already builds
  all of these. Expand to the full 7 only if signal appears.
- PLOT B: per target, x=Emb..L12+LN, y = existing forecasting metric (prefer the Stage-B relative-regret /
  WQL presentation), 3 curves stage0/early/late.

### 9. PLOT C — DOMAIN shift vs TASK shift (normalized, apples-to-apples)
- LEFT = BOOM-FT → forecasting probe (read Stage B `results/ft_specialization/stageB/…`); RIGHT = FordA-FT →
  forecasting probe (this experiment), on the SAME forecasting targets + SAME fslot probe protocol.
- Use a NORMALIZED per-layer quantity only: **Δ(fslot quantile loss) vs stage0** (or relative regret) —
  absolute losses across the two FT conditions are NOT comparable, the Δ-vs-stage0 curves ARE. Point: does
  TASK FT alter the late-layer forecasting representation more than DOMAIN-only FT? If normalization makes
  them incomparable, do NOT fabricate the panel.

### 10. OPTIONAL — representation drift (CKA), only if it doesn't delay the core
- Linear CKA between stage0 and FT features at each layer, on the SAME fixed FordA examples, computed
  separately for early/late. Plot CKA vs layer. Answers WHERE FT changed the representation. **Must not block
  Exps A/B.** (Cheap: reuse the cached 14-pt content features; CKA on CPU.)

### 11. CACHING / RESUMABILITY / FAIL-LOUD
- One extraction per checkpoint×dataset×split; probes read caches. Cache keys carry source=`forda_cls` +
  stage + checkpoint hash ⇒ classification-FT and BOOM-FT reps can NEVER collide. Idempotent: skip completed
  probe runs (JSON-on-disk check, like run_ft_specialization). Fail LOUD if a requested FT-stage cache is
  missing (never silently extract off the pretrained singleton for an FT stage) and if a checkpoint hash
  disagrees with the manifest (reuse load_stages' hash check).

### 12. TESTS (`tests/test_task_shift.py`, CPU/synthetic — no GPU/model/download)
1 FordA load + deterministic stratified split (fixed indices). 2 no train/val/test overlap. 3 label
{-1,1}→{0,1} mapping. 4 classification representation shape (14 pts, (n,768)). 5 pooling = mean over ncp
content tokens (exact on a synthetic hidden tensor). 6 head is truly linear (`nn.Linear`, no activation).
7 backbone params trainable during cls-FT (requires_grad). 8 backbone frozen during probing (fit patched to
raise if it touches backbone grads). 9 extraction returns 14 ordered layers Emb..L12+LN. 10 stage→checkpoint
mapping (pretrained/early/late) + hash carried. 11 cls probe never uses test for selection (val-only wd/C).
12 Exp-B reuses the fslot fns (assert it calls fit_shared_forecast_probe_explicit_val, not a new head). 13
cache keys distinguish all 3 stages (forda_cls prefixes disjoint + vs BOOM). 14 resume (idempotent skip).
15 synthetic end-to-end smoke (tiny fake backbone → extract → cls probe → figure render).

### 13. COMPUTE STAGES (separate sbatch; user submits — [[submit-slurm-jobs-self]])
- **C0** data smoke: `python -m probing.cls_data --smoke` (login node OK after `module load arrow`; seconds).
- **C1** classification FT (GPU): `sbatch job_task_shift.sh --finetune` → 2 checkpoints + manifest. Resumable.
- **C2** extraction (GPU): `sbatch job_task_shift.sh --extract` (FordA 14-pt for 3 stages) `--forecast-extract`
  (fslot for the FT stages on BOOM/M4/Coastal; stage0 reuses committed caches).
- **C3** classification probes (GPU/CPU): `--probe` (14 layers × seeds).
- **C4** forecasting probes (GPU/CPU, warm caches): `--forecast-probe`.
- **C5** figures/stats (CPU/login): `--figures [--cka]` → Plots A/B/C + CSV/JSON.
Do NOT run C1–C4 (FT / extraction / probe fits) on the login node.

### 14. SCIENTIFIC CAVEATS / STOPPING RULE (honest interpretation)
- Native forecasting head is unchanged by cls-FT (no gradient) → Exp B measures encoder/representational
  change via the fslot PROBE, which is the intended lens. (An optional native-forecast eval would move only
  through the changed encoder feeding a frozen head.)
- FordA may already be linearly decodable at stage0 (in-distribution numeric) → the "task specialization"
  gain can be small; that's an HONEST outcome. Report stage0 vs FT plainly. Do NOT tune to force late-layer
  classification gains or late forecasting collapse. Flat = valid. (BOOM-plan §9 applies.)
- Comparability of Plot C requires the normalized same-target same-probe Δ (§9).

### 15. HANDOFF CHECKLIST (next session)
```
[ ] Resolve DECISION A (delivery) + DECISION B (probe estimator) at the top of this section
[ ] Build probing/cls_data.py → run C0 data smoke (module load arrow; login OK)
[ ] Build probing/finetune_cls.py (reuse finetune.py helpers; Linear(768,2)+CE; early/late rule §5)
[ ] Build experiments/run_task_shift.py (Exp A + Exp B-via-import + Plots A/B/C [+CKA])
[ ] Build tests/test_task_shift.py (15 pts) + run CPU suite (OMP_NUM_THREADS=2)
[ ] Build job_task_shift.sh; give the user exact sbatch commands C1..C5 (do NOT submit)
[ ] C1 validity gate (backbone drifts + val-acc rises) BEFORE trusting stages / running probes
[ ] Commit code + results separately, NO Co-Authored-By ([[no-coauthor-trailer]])
```

## FT-SPECIALIZATION EXPERIMENT — 2026-08-11 (status refreshed 2026-08-13)
**STATUS: STAGE B — B0/B1/B2 RAN + B3/B4/B5 BUILT & CPU-VERIFIED (2026-08-17). Stage A DONE (BOOM
specializes ft_val −5.3%; Electricity robust). Driver = experiments/run_ft_specialization.py (7-target
roster + pt/ft status; load_stages hash-verifies both BOOM checkpoints 18c93f86/f734bbc4; stage0=committed
caches, FT stages=collision-proof IDF_<tag>__ft__boom__<stage>__<hash8>). job_ft_stageB.sh added. Choices
locked: 3 probe seeds, ONE driver with modes, namespace results/ft_specialization/stageB/.
7th target (per code, not a guess) = Electricity (monash_electricity_hourly, PT-ID/FT-OOD).
- **B0/B1/B2 RAN on real caches** (verified on disk 2026-08-17): extract_report.json (full 3×7×3), 42
  ft__boom caches, 63 B2 probe records (3 stages × 7 targets × 3 seeds), 3 BOOM tunnels, 7 layerwise
  figures, tables/stageB_layerwise__q9.csv. B2 result = NO systematic late-layer U-shape/degradation
  even OOD (honest; do NOT engineer a U-shape).
- **B3/B4/B5 BUILT (2026-08-17), CPU/synthetic-verified, NOT yet run on real caches:**
  - **B3 `--native`** (PRIMARY forgetting): each stage's OWN native head on identical target-test windows
    (stage0=pretrained singleton; FT stages=load_ft_pipeline; predict_quantiles→(n,Q,H) reused from
    run_fslot_forecasting_comparison, cached per checkpoint hash). ORIGINAL-scale MASE (in-context
    seasonal m=24) + WQL + median MAE. → results/…/stageB/native/{native_metrics__q9.{csv,json},inputs/}.
  - **B4 `--transfer`** (SECONDARY, probe-OOD): frozen BOOM probe RE-FIT per (stage,seed) from BOOM
    train/val fslot caches (deterministic = B2's BOOM probe; B2 never persisted the object), then
    predict-only on the 6 non-BOOM targets; D at the stage's BOOM tunnel entrance. Idempotent/resumable.
    → stageB/transfer/{transfer_metrics__q9.{csv,json},inputs/}.
  - **B5 `--forgetting`**: paired series-cluster bootstrap ACROSS stages on identical windows (Δ=FT−pretr,
    POSITIVE=worse; BOOM=in-domain control expected NEGATIVE) → native_forgetting__q9.csv + heatmap
    (rows=targets, cols=early/late, ΔMASE, * if CI excl 0) + native_mase_bars + B2-vs-B4-at-entrance
    (fresh probe vs frozen-BOOM readout) → stageB/forgetting/.
  - Reuse (no parallel eval logic): _ctx_stats/_mase_denominator/M_SEASON (run_id_forecasting);
    _raw_future/_mase_pw/_mae_pw/_wql_pw_parts/_series_group/_boot_mean/_boot_ratio
    (run_fslot_forecasting_comparison); cluster_bootstrap_counts/ci_bounds/d_stat_boot; tunnel.py UNTOUCHED.
- **Tests: 18/18 green** (tests/test_ft_specialization.py; CPU/synthetic). 5 new: B3 native-cache
  namespacing, B3 native-cell metrics (pure over windows+qr), B4 predict-only-on-target (fit patched to
  raise; frozen weights unmutated), B5 paired-stats direction + fail-loud on window mismatch, B5 fail-loud
  on missing B3/B4 inputs. Regressions green: forecasting_comparison 8, shared_forecast 8, tunnel 13,
  fslot_transfer 13.
NEXT ACTION = **run B3 on GPU** (`sbatch job_ft_stageB.sh --native`; only cold native passes hit the GPU),
then B4 + B5 (CPU/warm-cache): `--transfer` → `--forgetting`. Then inspect stageB/{native,forgetting} +
commit code + results separately. Do NOT run B3-B5 probe fits / native passes on the login node.**

### PILOT RESULT + LOCKED REVISION (2026-08-11, FT-ID evidence only — no FT-OOD inspected)
Stage A implemented + GPU-verified: `probing/finetune.py` (manual full-FT loop, official defaults),
`extract_kout_features` pipeline/cache injection (byte-identical default), `tests/test_ft_specialization.py`
(11 green), `job_ft_pilot_electricity.sh`. Split validated: 1394 source-train → **1073 ft_train / 321 ft_val**
(all 321 series ≥2 origins → latest→ft_val). Mechanical acceptance ALL PASS: matches official fit defaults
(adamw_torch_fused, bf16+tf32, LR 1e-6, linear decay, warmup 0, clip 1.0); 119,477,664 trainable; early/mid/late
blocks + native head + final-LN + input-embed + REG all drift; drift GROWS early→late; stage hashes differ
(300=903125e9, 1000=cbc4e6a2); 1000 steps in 140s on A100.
- **OVERFIT (batch 256):** train_loss ↓ (mean first-50 2.70 → last-50 2.23) while ft_val (native pinball on the
  321 held-out FT-ID windows) rises MONOTONICALLY from the pretrained baseline: step0 2.711 → 300 2.766 (+2.0%)
  → 1000 2.858 (+5.4%). `best_ft_val`=step 0. Effective epochs 60 (step300) / 200 (step1000) over 1073 windows.
  Per §9 (FT-ID native worsens → setup INVALID) → do NOT run Stage B at batch 256.
- **LOCKED REVISION (user, 2026-08-11):** **batch_size = 64, LR 1e-6 (unchanged), checkpoints still 300/1000.**
  Rationale (§2 pre-authorized batch reduction for exactly this): batch 64 cuts effective epochs 200→~60 at
  step 1000 (steps/epoch=ceil(1073/64)=17 → ~15 ep @300, ~59 ep @1000) AND adds ~4× SGD-noise regularization
  vs the near-full-batch (256/1073≈¼) low-noise gradient. GOAL: an ft_val curve that does NOT rise from step 0
  (a specialize-without-overfitting window). Command: `python -m probing.finetune --source electricity --batch-size 64`.
- **BATCH-64 OUTCOME (ran 2026-08-11):** helped but did NOT fix it. ft_val now DIPS below baseline (min 2.6997
  @ step 100, −0.4%) then rises monotonically to 2.779 @ step 1000 (+2.5%); best=step 100. BUT at step 100
  train_loss is still ~2.78 (unmoved) — the dip is near-ZERO specialization; meaningful train adaptation only
  appears step ~400+, by which point ft_val is already overfit. checkpoints: early@300 +0.6%, late@1000 +2.5%.
  → NO checkpoint is both meaningfully specialized AND FT-ID-healthy; specialization↔overfitting are COUPLED.

### BOTH FIXED-WINDOW PILOTS REJECTED + FT-DATA REDESIGN (2026-08-11, user decision; FT-ID evidence only)
**REJECTED (do NOT proceed with these checkpoints):** batch 256 = immediate overfitting (ft_val ↑ from step 0,
+5.4% @1000); batch 64 = narrow improvement near step 100 then overfitting (+2.5% @1000). **DIAGNOSIS: the
fine-tuning corpus incorrectly reused the heavily-subsampled probe-training windows** (1073 cluster-balanced
rows from the 1394 probe budget) — cycling them 60/200 epochs memorizes. The 1073 windows are a PROBE-oriented
subsample, NOT the Electricity fine-tuning corpus.
**REDESIGN (locked): fine-tune from the COMPLETE source training histories via random-window sampling.**
- VERIFIED (2026-08-11, no model load): the installed native `Chronos2Dataset` TRAIN mode ALREADY samples random
  cut points from full input histories — `dataset.py:193-195` `slice_idx = np.random.randint(min_past,
  full_length - prediction_length + 1)`; random series in `_generate_train_batches` (:283); context =
  hist[slice_idx-C:slice_idx], future_target = hist[slice_idx:slice_idx+H]. PREFER this official sampler over
  cycling the 1073 cached probe windows.
- LEAKAGE DISCIPLINE (must hold): per series TRUNCATE the history strictly BEFORE its held-out ft_val forecast
  target (cutoff = ft_val target start = starts[-3]+C, the latest train origin just before probe-val starts[-2]);
  sample training windows ONLY from that earlier region; retain the FIXED ft_val (per-series window at
  context-start starts[-3]), probe-validation (starts[-2]) and test (starts[-1]) windows UNCHANGED; keep C=512,
  H=64. `pipeline.fit()` still BANNED (it would sample across the preserved regions) → feed the native sampler
  our leakage-truncated per-series histories, in a manual loop.
- REPORT (before/after redesign): total eligible forecast origins AND unique training windows available
  (old = 1073 fixed; new = sum over series of samplable cut points in the truncated region ≈ millions →
  ~0 repetition at 1000×64 presentations).
- PILOT SETTINGS unchanged: batch 64, LR 1e-6, checkpoints 300/1000 steps, seed 0. Do NOT inspect FT-OOD.
- STOP RULE: if full-history/random-window FT STILL cannot improve (or hold) FT-ID ft_val → STOP full
  fine-tuning; do NOT proceed with knowingly-overfit checkpoints.
- CHOICES (user, 2026-08-11): (a) REUSE `Chronos2Dataset` directly for TRAIN sampling (official sampler,
  minimal divergence); (b) min_past = C = 512 (every FT window full-context, matches the probe/eval regime).
- IMPLEMENTED (2026-08-11, probing/finetune.py): `build_ft_split`→`build_ft_data(tag, min_past)` (full histories,
  per-series truncate at starts[-3]+C, fixed ft_val = starts[-3] window, reports n_unique_train_windows vs the
  1073); the loop drives `Chronos2Dataset(train_histories, C=512, H=64, batch, P=16, min_past=512, TRAIN)` via
  `next(iter(ds))` (np.random.seed(seed) for determinism), model(context,future_target,num_output_patches).loss.
  Default batch=64, checkpoints 300/1000, everything else (AdamW fused, linear sched, warmup0, clip1.0, bf16/tf32)
  unchanged. `raw_future_from_traj` removed (ft_val now slices raw straight from the series). Tests rewritten
  (tests/test_ft_specialization.py, 10 green). extract_kout_features injection UNCHANGED (still byte-identical
  default). Re-pilot: `sbatch job_ft_pilot_electricity.sh` (or salloc → `python -m probing.finetune --source
  electricity`). Then paste manifest+history for the FT-ID ft_val verdict.

**STATUS(orig): PLANNED — code not started. Next action: implement and verify Stage A only.**
**SCOPE (2026-08-11, user): FIRST EXPERIMENT = ELECTRICITY fine-tuning ONLY (one source). Do NOT fine-tune
Uber / M4 / WindFarms yet — they are DEFERRED until the Electricity pilot + its 7-target eval validate.
The 4-source design below is retained as the parked extension (labels/counts still describe the full design).
First run = 1 Electricity FT run → 2 checkpoints (ft_early@300, ft_late@1000) → Electricity's 7 targets × 3 stages.**
Self-contained handoff: a fresh session can begin from THIS section without conversation history.
Repo state at drafting: `git status` CLEAN, branch `tunnel-effect-probing`, no FT files tracked/untracked,
no partial FT work anywhere. NOTE `notes/` IS GITIGNORED (`.gitignore:16`) → PLAN.md is untracked, so
`git diff -- notes/PLAN.md` is EMPTY BY DESIGN (not a bug). Do NOT modify/discard unrelated work.

### PT-OOD SOURCE PIVOT (2026-08-11, user decision; implemented + CPU-verified; no FT-OOD inspected)
LR-3e-6 escalation result (the ONE §2 revision, spent): weights moved ~3× more (block_06 rel 1.6e-3→4.0e-3;
early/late now distinguishable 0.31%/0.40%) but train_loss STILL FLAT (first50 2.660 → last50 2.653, Δ −0.007)
and ft_val hovered at baseline (best −0.57% @500 = noise trough, final −0.20%). Grad-norms real (~10, clipped) →
the model drifts through weight space WITHOUT reducing forecasting loss = **Chronos-2 is at its full-FT optimum
on in-pretraining Electricity**. §9 "robust". All 4 planned FT sources are PT-ID → same wall → the experiment
as designed = nulls by construction.
**DECISION (user): fine-tune on a PT-OOD SOURCE so the model has real room to specialize.** Source = **BOOM**
(Datadog telemetry, explicitly documented-unseen; 356 long queries ~5200; no missing-data pathology; cleanest
of the 3 staged PT-OOD sets). This FLIPS the framing to classic catastrophic forgetting: FT-ID = BOOM (the
model CAN specialize here); "FT-OOD" = the PT-ID datasets (Electricity/Uber/M4/WindFarms) it already knew +
the other PT-OOD sets. Cleaner/stronger than the (now-null) PT-ID-source design.
IMPLEMENTED (probing/finetune.py): `SOURCE_TAGS` += `boom`→`boom_hourly` (default `--source` now `boom`);
`build_ft_data` routes `tag in OOD_TARGET_TAGS` through `load_ood_target_series` (needs OOD_TARGET_ROOT), else
`load_seen_series`; identical rolling-truncation + Chronos2Dataset sampler + loop; meta carries `source_kind`.
VERIFIED (no model): InstanceNorm is nanmean/nanstd-robust (chronos_bolt.py:16-19) + loss masks NaN targets →
BOOM gaps are safe (official Chronos2Dataset path). 11/11 Stage-A tests (incl. PT-OOD path) + regressions green.
Job renamed `job_ft_pilot_electricity.sh`→`job_ft_pilot.sh` (default `--source boom`, sets OOD_TARGET_ROOT).
RE-EVAL RULE (FT-ID only, decisive): does BOOM `train_loss` now DROP (real specialization) with `ft_val`
improving/flat (not overfitting)? YES → valid intervention → distinguishable stages → Stage B (catastrophic-
forgetting framing). If BOOM train_loss ALSO stays flat → Chronos-2 is broadly robust even OOD (a strong
finding). If ft_val overfits → BOOM series too few/short per cluster; reconsider corpus.
NEXT: `python -m probing.finetune --source boom` (or `sbatch job_ft_pilot.sh`). Same batch 64 / LR 1e-6 / 300+1000.

BOOM PILOT RESULT (2026-08-11, GPU, 72s): **SUCCESS — real specialization.** 353 train series / 1.30M unique
windows (coverage 0.049 = ~no repetition) / 354 ft_val. ft_val 3.748(step0)→3.662→3.618→3.584(300)→3.556(400)
→3.551(700,min)→3.554(1000): **−5.3% monotonic, plateaus ~step400, no rebound.** train_loss mean first50 5.29
→ last50 5.15 (Δ−0.148, noisy; grad_norm 20–48 clipped). Drift GROWS early→late AND reduces loss: block_06 rel
1.6e-3@300→2.2e-3@1000; native_head 5.1e-4→7.5e-4. 3 distinct backbones: stage0 3.748 / stage1_ft_early@300
3.584 (−4.4%) / stage2_ft_late@1000 3.553 (−5.2%). Checkpoints at $SCRATCH/chronos2/ft_specialization/boom/
(hashes 18c93f86 / f734bbc4). CONTRAST with Electricity (PT-ID, flat) = the clean headline. Intervention VALID.
NOTE for Stage B: BOOM specializes FAST (plateau ~step400) → 300/1000 gives a COMPRESSED early/late gap
(early already captures ~83% of the gain). Consider moving stage1_ft_early earlier (~step100–150, ft_val 3.66,
−2.3%) for a wider specialization gradient — a Stage-B checkpoint-timing choice, decide on FT-ID evidence.

### STAGE B — DESIGN LOCKED (user, 2026-08-11). BUILD PLAN AWAITING GO-AHEAD.
Catastrophic-forgetting experiment on the 3 BOOM backbones. **FT source = BOOM (PT-OOD).**
CHECKPOINTS: stage0_pretrained / stage1_ft_early@300 / stage2_ft_late@1000 — KEEP 300/1000 (predeclared before
the BOOM curve; do NOT move early post-hoc). MAIN comparison = pretrained vs fine-tuned; early-vs-late SECONDARY.
7 EVAL TARGETS (all probe-ID for the primary; pt_status/ft_status/probe_status on every row):
  - BOOM               : PT-OOD / FT-ID  / probe-ID  (the source)
  - Electricity, Uber, M4, WindFarms : PT-ID / FT-OOD / probe-ID  (the "known" domains)
  - SG_Carpark, Coastal_TS           : PT-OOD / FT-OOD / probe-ID
METRIC HIERARCHY:
  1. **PRIMARY catastrophic forgetting = NATIVE forecasting.** Evaluate each stage's NATIVE head on IDENTICAL
     target-test windows; ORIGINAL-scale MASE + WQL. The cleanest forgetting measure (the model's own forecasts).
  2. **PRIMARY layerwise = FRESH target probes (probe-ID).** For each stage × target, train a shared-forecast-slot
     linear probe on the TARGET's own train (wd on target-val), score on target-test. Asks whether the FT backbone
     still holds linearly-accessible forecasting info for each domain. Fresh probe (NOT frozen BOOM) so
     representational forgetting is NOT confounded with a BOOM-trained readout/scaler failing to transfer.
  3. **SECONDARY transfer = FROZEN BOOM probe.** Train probe on BOOM per stage, freeze, apply to the 6 non-BOOM
     targets (all become probe-OOD). Measures READOUT transfer, not representation retention.
TUNNEL: each stage's tunnel defined ONLY from its BOOM FT-ID/probe-ID VALIDATION curve (sustained-plateau,
tunnel.py unchanged); that BOOM-defined layer region is then applied as the lens to the FT-OOD curves. NEVER
define a tunnel from FT-OOD data.
KEY INFRA REUSE: extract_kout_features(pipeline=, cache_prefix=) injection (Stage A) loads each FT checkpoint +
namespaces its fslot caches (IDF_<tag>__ft__boom__<stage>__<hash8>); stage0 reuses committed pretrained fslot
caches where present. fit/predict_shared_forecast_probe, tunnel.py, stats cluster-bootstrap, and the
run_fslot_forecasting_comparison original-scale inverse (mu+s*sinh) + MASE/WQL all reused. Targets' rolling
train/val/test: build_windows (4 PT-ID) / build_ood_rolling_windows (BOOM, SG, Coastal). LEAKAGE OK: FT training
truncated at each BOOM series' starts[-3]+C, strictly before probe-val(starts[-2]) / test(starts[-1]).
OPEN IMPLEMENTATION CHOICES (confirm before/at build): (i) probe seeds — 1 (fast) vs the v4 3-run protocol (0/1/2,
seed-averaged then bootstrapped); (ii) driver shape — ONE new experiments/ driver with modes (--extract/--probe/
--native/--transfer/--figures) vs several; (iii) output namespace (proposed results/ft_specialization/stageB/).
BUILD ORDER (incremental, verify each like Stage A): B0 verify 3-backbone extraction (GPU smoke) **[DONE
2026-08-12]** → B1 extract fslot for 3 stages × 7 targets × {train,val,test} (stage0 cached; stage1/2 fresh;
GPU-heavy) **[RAN — extract_report.json + 42 caches on disk]** → B2 fresh probes + per-stage BOOM tunnels +
layerwise curves **[RAN on real caches — 63 probe records, 3 tunnels, 7 figures, table]** → B3 native
MASE/WQL (3×7, identical windows) **[BUILT 2026-08-17, CPU-verified; not run on real caches]** → B4
frozen-BOOM transfer (3×6 probe-OOD) **[BUILT 2026-08-17]** → B5 paired cluster bootstrap +
catastrophic-forgetting figures/tables **[BUILT 2026-08-17]**. All three are new `--native/--transfer/
--forgetting` modes on the ONE driver; main() no longer raises. NEXT = GPU run of B3, then CPU B4/B5.

### FULL-HISTORY PILOT RESULT + §2 ESCALATION (2026-08-11, FT-ID evidence only — no FT-OOD inspected)
Ran `--source electricity` (batch 64, LR 1e-6, 1000 steps, full-history Chronos2Dataset sampler, min_past=512).
- **OVERFITTING ELIMINATED.** 8.18M unique training windows, coverage 0.0078 (each window seen <1×). ft_val no
  longer rises from step 0 — holds/slightly improves: baseline(step0)=10.337, min 10.285 @ step900 (−0.50%),
  final 10.303 (−0.33%); best_ft_val is now LATE (step 900). Confirms the memorization diagnosis. (NOTE ft_val
  absolute scale jumped ~2.7→~10.3 vs the fixed-window pilots because the ft_val SET changed — now the fixed
  starts[-3] window per series = strictest temporally-latest held-out point; only within-run trend is comparable.)
- **BUT INTERVENTION WEAK.** train_loss FLAT (mean first50 2.663, last50 2.666, no trend); param drift only ~0.1%
  rel (block_06 rel 1.6e-3 @1000); stage1_ft_early vs stage2_ft_late near-identical (drift 0.09% vs 0.11%, ft_val
  10.299 vs 10.303). Chronos-2 (already pretrained on Electricity) barely specializes at the official recipe
  (LR 1e-6 linearly decayed to 0 → avg ~5e-7). = §9 "intervention too weak / Chronos-2 robust". Three near-identical
  backbones cannot yield a meaningful FT-OOD signal.
- **§2 ESCALATION LOCKED (user, 2026-08-11): LR 3e-6, batch 64, 1000 steps — the EXACTLY-ONE pre-authorized
  revision.** 3× per-step movement (strongest single knob). Command: `python -m probing.finetune --source
  electricity --learning-rate 3e-6`. Fresh-window corpus (coverage 0.008) → no overfit risk from the higher LR.
  **DECISIVE: do NOT tune LR/steps again after this** (§2 permits only one revision; further tuning = fishing).
- RE-EVAL RULE (FT-ID only): (i) does train_loss now DROP and drift grow to a MEANINGFUL level (stages
  distinguishable)? (ii) does ft_val stay flat-or-better (no return of overfitting)? If YES to both → valid,
  meaningful intervention → pick checkpoints → Stage B. If train_loss still flat → Chronos-2 is robust to
  full-FT specialization on in-pretraining Electricity; REPORT that (§9) rather than escalating further. If
  ft_val overfits again → the LR is too high for this corpus; fall back to the batch-256 official recipe.

### 1. SCIENTIFIC OBJECTIVE
Test whether domain specialization creates tunnel-like FT-OOD degradation that is ABSENT in broadly
pretrained Chronos-2. Compare 3 backbone states — **stage0_pretrained / stage1_ft_early / stage2_ft_late**
— for 4 independently fine-tuned source models: **Electricity, Uber, M4, WindFarms**. Each source is
evaluated on 7 targets: itself + the other 3 PT-ID datasets + SG_Carpark + Coastal_TS + BOOM.
Terminology (RECORD ON EVERY ROW; never bare ID/OOD): **PT-ID/PT-OOD** = target in Chronos-2 pretraining;
**FT-ID/FT-OOD** = target is/ isn't THIS backbone's fine-tuning source; **probe-ID/probe-OOD** = probe
trained on the eval dataset or transferred. `pt_status` ALWAYS describes the TARGET. Per source & stage:
1 `PT-ID/FT-ID/probe-ID` + 3 `PT-ID/FT-OOD/probe-OOD` + 3 `PT-OOD/FT-OOD/probe-OOD` = 7 targets, 28 cells/stage.
**For now only the Electricity source is fine-tuned (7 cells/stage); the other 3 sources are deferred (see SCOPE).**

### 2. LOCKED FINE-TUNING DECISIONS
⛔ **OBSOLETE & REJECTED (do NOT use): the earlier `LR=1e-4`, `20/60 epochs`, cosine schedule, 5% warmup,
AdamW betas 0.9/0.95, wd 0.01, batch 32 proposal.** LR 1e-4 is 100× the official full-FT default and could
DESTRUCTIVELY alter the backbone → manufacture apparent FT-OOD forgetting (invalidates the whole comparison).

✅ **LOCKED pilot schedule (budget in OPTIMIZER STEPS, not epochs; record effective epochs as metadata only):**
```text
finetune_mode        = full
trainable parameters = ENTIRE ~119M model (encoder + input_patch_embedding + REG-embed + final LayerNorm + native head)
stage1_ft_early      = 300 optimizer steps        # checkpoint of the SAME run
stage2_ft_late       = 1000 optimizer steps       # official default budget
learning_rate        = 1e-6                        # official full-FT default
gradient clipping    = 1.0
fine-tuning seed     = 0 initially (add +1 seed ONLY if a tunnel/OOD effect appears — robustness)
```
**Use the official Chronos-2 full-FT convention as the starting point.** VERIFIED from installed 2.3.1
`Chronos2Pipeline.fit()` (`pipeline.py:111-288`): `finetune_mode="full"`, `learning_rate=1e-6`,
`num_steps=1000`, `lr_scheduler_type="linear"` (linear decay to 0), **warmup=0** (`warmup_steps=0`/
`warmup_ratio=0.0`), `optim="adamw_torch_fused"` (AdamW HF defaults: betas 0.9/0.999, eps 1e-8, **wd 0.0**),
`max_grad_norm=1.0` (HF default), `gradient_accumulation_steps=1`, `per_device_train_batch_size=256`,
bf16+tf32 on sm80 (A100), `logging_steps=100`. LoRA is a SEPARATE mode (`learning_rate≈1e-5` is a LoRA
recommendation, NOT full-FT) — **no LoRA/adapters here.** Ref:
`https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py`.
Match these optimizer/scheduler defaults; do NOT silently introduce beta2=0.95, cosine, or a large warmup.
- **IMPLEMENT AS A MANUAL LOOP, not `pipeline.fit()`.** `fit()` re-windows raw series via `Chronos2Dataset`
  (random sampling) and would bypass our fixed nested split → could sample into the PRESERVED probe-val/test
  regions (leakage). So replicate the official TrainingArguments HYPERPARAMETERS above in a manual loop over
  OUR fixed nested `ft_train` windows, evaluating OUR fixed `ft_val`. This is a justified deviation from the
  high-level API, NOT from its hyperparameters. Document exactly which settings are mirrored.
- **batch_size = 256 (official default) is the starting point, but it is the one data-scale-sensitive knob:**
  our `ft_train`≈1100 windows → 256/batch ⇒ ~4.3 steps/epoch ⇒ **300 steps ≈ 70 epochs, 1000 steps ≈ 233
  epochs** over the SAME small fixed set (record these effective-epoch numbers as metadata). If the pilot's
  FT-ID `ft_val` shows overfitting from this repetition, reducing batch is a PRE-AUTHORIZED "strong reason to
  deviate" — decided on FT-ID evidence ONLY and recorded here BEFORE any FT-OOD eval.
- **Pilot may use ONLY:** FT-ID training loss, nested FT-validation loss, FT-ID native forecasting MASE/WQL,
  encoder-weight drift, convergence diagnostics — to judge whether the intervention is strong enough.
  **Do NOT inspect FT-OOD curves while choosing the budget.**
- **If 1000 steps produce negligible adaptation, permit EXACTLY ONE locked revision before running the other
  sources: either `2000` steps @ `1e-6`, OR `1000` steps @ `3e-6`.** Justify from Electricity FT-ID evidence
  ONLY and RECORD it in this PLAN before any FT-OOD evaluation.

### 3. VERIFIED INFRASTRUCTURE FACTS (2026-08-11, from source; no model load unless noted)
- Installed `chronos-forecasting==2.3.1` at `.venv/.../chronos/chronos2/`. **No project FT implementation
  exists** (greenfield). Package is NOT vendored.
- `Chronos2Model.forward(context=raw_context, future_target=raw_future, num_output_patches=4)` returns
  `Chronos2Output.loss` = native training (pinball) loss (`_compute_loss`, model.py:518-567); `loss.backward()`.
- The model applies its INTERNAL instance-norm + arcsinh (`use_arcsinh=True`) to BOTH context and target, so
  fine-tuning receives **RAW** values. `group_ids=None` preserves the existing UNIVARIATE treatment.
- **RAW future is reconstructed EXACTLY** from the stored arcsinh trajectory via the canonical inverse already
  in the project: `y_raw = mu + s·sinh(Y_traj)` where mu,s = context mean/std (`_raw_future` in
  `experiments/run_fslot_forecasting_comparison.py`) → **no `probing/id_data.py` change needed.**
- Model ≈ **119,477,664 params** (encoder 113.3M = per block time-attn + group-attn + MLP; native head
  `output_patch_embedding` 3.65M; `input_patch_embedding` 2.55M; REG-embed 1,536). Config: d_model 768,
  d_ff 3072, 12 layers, 12 heads, dropout 0.1, 21 quantiles, output_patch_size 16 (⇒ K=ceil(H/P)=4 at H=64).
- **Full FT includes encoder, input embedding, REG embedding, final LayerNorm, and native forecasting head.**
- Checkpoints: `pipeline.save_pretrained(dir)`==`model.save_pretrained(dir)` (HF safetensors);
  reload `Chronos2Pipeline.from_pretrained(dir)`. **Cache keys MUST include source, stage, checkpoint hash**
  (= sha256(model.safetensors)[:8]).
- **Window counts (warm caches, all 4 sources): 1394 train / 262 val / 262 test.** Split = existing
  `extended_v3_rolling` rolling-origin within-series (`_build_rolling_windows`). `X_train`=RAW context;
  `Y_*_traj`=arcsinh(context-standardized future). `meta["origins"]["train"]` + `series_train` give per-window
  target-start index + series id (needed for the nested carve).
- **The nested temporal FT split is VIABLE for all four datasets (incl. M4).** Existing probe-validation and
  test origins remain UNTOUCHED. Per-series origin counts (analytic, C512/H64): Elec 321 series ×403; Uber
  262 ×59; Wind 337 (mostly ×129); **M4 414 series: 169 have 3 origins (→1 train window), 245 have 7 (→5),
  supply=1394 exactly (M4 is the floor, no subsample).** Estimated `ft_val` size ≈ #series with ≥2 train
  windows: Elec ~321, Uber ~262, Wind ~337, **M4 ~245**.

Split order (STRICT, per series):
```text
ft_train origins  <  ft_val origins  <  probe-validation origin  <  test origin
```
For each series, move the LATEST eligible source-training origin to `ft_val` ONLY when that series has ≥2
training origins. Single-training-window series remain entirely in `ft_train`. (All targets are H-spaced and
non-overlapping, so every ft_val target strictly precedes the preserved probe-val target = starts[-2].)

### 4. EXISTING STAGE-0 ARTIFACTS (IMMUTABLE — do not overwrite or move)
stage0 REUSES the v4 future-token artifacts already on disk: source probe checkpoints
(`results/ext_v4_future_tokens/ptood_probing/ptid_checkpoints/`, 3 seeds), source tunnels
(`results/ext_v4_future_tokens/tunnels/`), shared forecast-slot caches
(`features_cache/IDF_<tag>__extended_v3_rolling__{train,val,test}__clean__K4_H64.npz`,
`IDF_<ood>__ood__test_rolling__…`), the 4×4 PT-ID transfer results, and the Electricity→3 PT-OOD results.
There are **12 stage-0 PT-OOD source→target cells total; Electricity→3 already exists, so 9 remain** (compute
predict-only on CPU using the frozen pretrained probes + existing PT-OOD test caches). stage0 = no fine-tuning.

### 5. IMPLEMENTATION ORDER — TWO STAGES
**STAGE A — IMPLEMENT THIS ONLY next session, then stop for pilot review.** Files:
1. NEW `probing/finetune.py`.
2. Minimal dependency-injection / cache-namespacing EDIT to `probing/extraction.py`.
3. Focused Stage-A tests (`tests/test_ft_specialization.py`, Stage-A subset).
4. Small Electricity pilot job script (`job_ft_pilot_electricity.sh`).

Stage A must implement:
- fresh loading of a TRAINABLE Chronos-2 pipeline (separate from the frozen `get_pipeline` singleton);
- `requires_grad=True` for the entire model;
- nested temporal `ft_train/ft_val` construction (latest-origin rule above; fail-loud if `ft_val` empty);
- raw-future reconstruction via the canonical inverse (mu + s·sinh(Y_traj));
- native forward loss + backprop; official-scale FULL-FT optimizer settings (§2: LR 1e-6, linear decay,
  warmup 0, adamw_torch_fused-equiv, grad-clip 1.0, bf16 on A100, batch 256, seed 0);
- checkpoint saving at steps 300 (ft_early) and 1000 (ft_late) of ONE run;
- manifest JSON with ALL hyperparameters + checkpoint hashes (early & late), source, stage label,
  optimizer/scheduler/lr/wd/batch, step count + effective epochs, seed, best-ft_val-ckpt (diagnostic only);
- training + FT-validation loss histories;
- encoder/head parameter-drift diagnostics (per-block + head, vs the pretrained weights);
- injection of a supplied fine-tuned pipeline into `extract_kout_features` (new `pipeline=`/`cache_prefix=`
  kwargs, both default None → BYTE-IDENTICAL legacy behavior);
- collision-proof FT feature-cache prefixes `IDF_<tag>__ft__<source>__<stage>__<hash8>`;
- default extraction behavior UNCHANGED for pretrained models.

Stage-A acceptance criteria (verify in the pilot):
- at least one parameter in an EARLY, a MIDDLE, and a LATE encoder block changes vs pretrained;
- native head parameters change;
- stage-early and stage-late checkpoint hashes DIFFER;
- FT-validation loss is finite and does not catastrophically diverge;
- the ORIGINAL pretrained model remains unchanged (frozen singleton untouched; stage0 artifacts untouched);
- raw-context/raw-target preprocessing round-trips correctly;
- extracted fine-tuned representations contain all **14** points (Emb, L1..L12, L12+LN);
- `L12+LN` uses the FINE-TUNED checkpoint's final LayerNorm;
- pretrained and fine-tuned cache keys cannot collide;
- no source TEST or FT-OOD target data enter fine-tuning.
**STOP after Stage A + the short Electricity pilot. Do NOT build the transfer pipeline until pilot review.**

**STAGE B — ONLY after Stage-A approval** (planned components, do not build yet):
full source-checkpoint extraction (7 targets: source train/val/test, others test-only); 3-seed shared LINEAR
layerwise probing; current tunnel definition UNCHANGED; source-validation-defined tunnel frozen across each
source row; 7-target FROZEN transfer per source & stage; native forecasting evaluation; MASE + median MAE +
WQL; last-value + seasonal-naive baselines; per-window seed averaging then paired cluster bootstrap; figures
comparing stage0 / ft_early / ft_late. **FIRST RUN = ELECTRICITY ONLY** (source=Electricity, its 7 targets ×
3 stages); the Uber / M4 / WindFarms sweep is DEFERRED until Electricity validates (see SCOPE). **Do NOT
implement the native-shaped MLP in the first FT sweep** (linear only; add later only if linear reveals something important).

### 6. TUNNEL & LEAKAGE INVARIANTS (PROMINENT)
- Do NOT modify `probing/tunnel.py`. The current SUSTAINED 5% criterion remains authoritative.
- `L12+LN` remains the final reference (fslot curves = 14 points).
- Each source/stage tunnel is defined ONLY from its own `FT-ID/probe-ID` VALIDATION curve (mean over probe
  seeds). No FT-OOD target defines a tunnel or selects a layer.
- Source scaler, probes, weight decay, and tunnel remain FROZEN during transfer. No target-fitted scaling.
  No test-based hyperparameter selection.

### 7. OUTPUT NAMESPACE & RECORD FIELDS
```text
results/ft_specialization/
    electricity/   { stage1_ft_early/  stage2_ft_late/ }     # stage0 references the immutable v4 artifacts
    uber/          { stage1_ft_early/  stage2_ft_late/ }
    m4/            { stage1_ft_early/  stage2_ft_late/ }
    windfarms/     { stage1_ft_early/  stage2_ft_late/ }
```
Large checkpoints + FT feature caches → `$SCRATCH` (gitignored, regenerable). Manifests, metrics, and final
figures may live under the project `results/` namespace. **Every cache key AND every record must carry:**
fine-tuning source · target · stage · checkpoint hash · pt_status · ft_status · probe_status · model/config
version · C=512 · H=64 · P=16 · K=4. (Probe eval rows additionally: probe seed, layer + label, quantile loss,
MASE, median MAE, per-window losses, source tunnel entrance, final reference L12+LN, native metrics, naive
baselines.)

### 8. COMPUTE / STORAGE ESTIMATE (for planning; user submits all SLURM — [[submit-slurm-jobs-self]])
GPU full sweep ≈ 4–6h (splittable/resumable): FT itself is CHEAP (tiny 37-token sequences, ~1100 windows —
extraction passes dominate, not gradient steps); fresh fslot extraction of 8 FT backbones × ~9 splits ≈
1.5–2.5h (train split dominant); native eval ≈ 1–2h; linear probe fits ≈ 20–40 min (warm caches);
bootstrap+figures = CPU minutes. **Electricity pilot ≈ 1–1.5h.** Storage ($SCRATCH, gitignore): checkpoints
8×~478MB ≈ 3.8GB; fresh fslot caches ≈ 6.9GB (train 344MB, val/test 65MB, OOD test 12–88MB each).
**ELECTRICITY-ONLY first run (the current scope) is far smaller:** 2 checkpoints ≈ 956MB + ~1.7GB fresh
caches (2 FT backbones × 9 splits); GPU ≈ 1–1.5h total (1 FT run + 2×9 extractions + native eval on 7 targets).

### 9. STOPPING RULE (interpret honestly)
- Do NOT increase FT strength to manufacture a U-shape; do NOT select datasets or budgets on FT-OOD curves.
- FT-ID native performance WORSENS → the adaptation setup is invalid; FIX it before interpreting OOD.
- FT-ID improves but FT-OOD curves stay FLAT → report: specialization does NOT create the classic tunnel effect.
- FT-OOD degradation GROWS from early→late FT → report as specialization-induced forgetting.
- Native FT-OOD degrades but probes stay flat → distinguish native-head alignment from representational forgetting.
- Both native + probe FT-OOD degrade → broader representational forgetting. No change across stages → intervention
  too weak / Chronos-2 robust.

### 10. HANDOFF CHECKLIST (next session)
```text
[ ] Read this entire FT-specialization section
[ ] Inspect git status/diff and current installed Chronos version
[ ] Confirm official fit defaults in the installed 2.3.1 source
[ ] Implement Stage A only
[ ] Run CPU/synthetic tests
[ ] Provide the Electricity pilot SLURM command
[ ] Do not launch SLURM automatically
[ ] Review pilot before Stage B
```

## ⇒ WHEN GPU JOB 611670 FINISHES — login-node CPU steps (job submitted 2026-08-11)
Job `job_fslot_mlp_all.sh` (name fslot_mlp_all, id 611670) runs IN ORDER: linear 4×4 + PT-OOD transfer
(were never run) → MLP `--fit-ptid` (4 datasets × 3 seeds × 14 layers = the ONLY GPU-heavy stage) → MLP
`--tunnels-only` → MLP 4×4 + PT-OOD transfer → forecasting comparison. Feature caches ALL warm (no
re-extraction; the MLP fit never loads Chronos-2). Cheap stages run FIRST so a real-data bug dies in
minutes BEFORE the GPU stage. Resumable: on TIMEOUT just `sbatch job_fslot_mlp_all.sh` again (fit_ptid
skips finished seeds). Monitor: `squeue -u $USER`; `tail -f logs/fslot_mlp_all-611670.out`.

1. CONFIRM SUCCESS: `sacct -j 611670 --format=JobID,JobName%16,State,Elapsed,ExitCode,MaxRSS` (want
   State=COMPLETED, ExitCode 0:0) + `tail -25 logs/fslot_mlp_all-611670.out` (ends "=== DONE ==="").
   TIMEOUT → resubmit (resumes). FAILED → traceback is in the log.
2. CONVERGENCE / OVERFIT GATE (decides whether to trust the numbers — did 300 epochs converge? does
   WindFarms overfit = train↓ while val↑ at deep layers?). Histories at fslot_mlp/ptid_runs/*__history.json:
     python - <<'PY'
     import json, glob
     for f in sorted(glob.glob("results/ext_v4_future_tokens/fslot_mlp/ptid_runs/*__q9__seed0__history.json")):
         h=json.load(open(f)); bl=h["by_layer"]; vals={k:v["final_val_loss"] for k,v in bl.items()}
         best=min(vals,key=vals.get)
         print(f"\n{h['dataset']} (dropout={h['dropout']}, {h['epochs']}ep, {h['param_count_per_head']:,} p/head)")
         for k in sorted(bl,key=int):
             v=bl[k]; tag=" <-best" if k==best else (" **late-degrade" if vals[k]>vals[best]*1.05 else "")
             print(f"  {v['layer_label']:>7} wd={v['chosen_wd']:>6g} train={v['final_train_loss']:.3f} "
                   f"val={v['final_val_loss']:.3f} conv={v['converged']}{tag}")
     PY
   converged=False on most layers → bump QUANTILE_EPOCHS + resubmit. Deep layers **late-degrade with LOW
   train = the tunnel/overfit signal (a RESULT, not a bug — the WindFarms question). Repeat seed1/seed2.
3. HEADLINE:
   - `cat results/ext_v4_future_tokens/forecasting_comparison/summary__q9__runs0-1-2.md`  (§9.4 auto-findings)
   - `column -s, -t results/ext_v4_future_tokens/forecasting_comparison/tables/forecasting_comparison__q9__runs0-1-2.csv | less -S`
   - `column -s, -t results/ext_v4_future_tokens/fslot_mlp/transfer_4x4/tables/transfer_summary__4x4__q9.csv | less -S`  (MLP transfer gaps)
   - `find results/ext_v4_future_tokens/{fslot_transfer,fslot_pt_ood,fslot_mlp,forecasting_comparison} -name '*.png'`  (scp to view)
4. COMMIT (when happy). MLP ckpts ~12MB×168 ≈ 2GB + regenerable → gitignore them (`.gitignore` currently
   only excludes features_cache/, so results ARE tracked). Two commits, NO Co-Authored-By:
     echo 'results/ext_v4_future_tokens/fslot_mlp/ptid_checkpoints/' >> .gitignore
     git add probing/probes.py experiments/run_ptood_probing_ftok.py experiments/run_fslot_transfer.py \
             experiments/run_fslot_forecasting_comparison.py tests/test_slot_mlp_probe.py \
             tests/test_forecasting_comparison.py tests/test_fslot_transfer.py job_fslot_mlp_all.sh \
             notes/PLAN.md .gitignore
     git commit -m "Add native-structure MLP capacity-control readout to the v4 fslot pipeline"
     git add results/ext_v4_future_tokens/
     git commit -m "Run linear+MLP fslot transfers and original-scale forecasting comparison"
   NOTE native-Chronos-2 WQL is NOT computed by this job (needs a fresh multi-quantile model pass) — add
   `--native-wql` and rerun `python -m experiments.run_fslot_forecasting_comparison` later if wanted.

## NATIVE-MLP CAPACITY CONTROL — CODE DONE + CPU-TESTED 2026-08-11 (GPU job 611670 SUBMITTED 2026-08-11)
Added the nonlinear native-structure MLP readout to the v4 fslot pipeline as a **nonlinear-decoding
capacity control** (NOT a linear-accessibility measure). USER go-ahead given ("go ahead and implement");
USER chose **dropout=0.1 (native-faithful)**, stored in every checkpoint/record. Reuses the existing
`probing/heads.py::ResidualBlock` (fresh weights, NEVER Chronos-2's native head).
§1 VERIFIED FROM SOURCE (no model load): native `output_patch_embedding` (chronos2/model.py:265) = a
ResidualBlock(in=d_model 768, hid=d_ff 3072, out=Q*P, ReLU, residual skip, `use_layer_norm=False` —
the final_layer_norm is applied to its INPUT = the L12+LN readout point), dropout=config.dropout_rate
0.1. Head param count (q9, P16) = **2,915,616** (matches hand-calc + fit prints). ONE open item: confirm
`act_fn_name` in the real amazon/chronos-2 config.json (heads.py hardcodes ReLU) — needs GPU/HF cache.

KEY DESIGN — one `ProbeFamily` abstraction, shared cache, disjoint artifacts:
- feature_kind="fslot" (UNCHANGED cache key: `IDF_<tag>__<set>__<split>__clean__K4_H64.npz`, family-
  independent → NO re-extraction) vs probe_family="native_mlp" vs artifact_tag="fslot_mlp" (checkpoints/
  tunnels/figures/outputs ONLY). §4 preflight `preflight_feature_cache` asserts the artifact tag never
  leaks into the feature-cache path. Default family="shared_linear" is BYTE-IDENTICAL (its callables ARE
  the existing functions; linear ckpt/tunnel paths verified == the committed on-disk layout).
- native_mlp routes to a PARALLEL `results/ext_v4_future_tokens/fslot_mlp/{ptid_checkpoints,tunnels,
  ptid_runs,figures,transfer_4x4,ptood_transfer}/`. Nothing linear is moved/overwritten.

FILES (STAGES 1-5, all CPU-tested green; run_id_forecasting/tunnel.py/id_data.py UNTOUCHED):
- `probing/probes.py` (+115): threaded `init_seed` through `_fit_forecast_slot_head`
  (`manual_seed(init_seed)`); NEW `fit_forecast_slot_native_head_explicit_val` (nonlinear twin of
  `fit_shared_forecast_probe_explicit_val`: slot-scaler+head on FULL train, wd on EXPLICIT val,
  iterates `sorted(train_feats)` → 14-key, per-epoch train/val history + `converged` flag + full §3
  diagnostics); `predict_forecast_slot_native_head` loop `range(NUM_LAYERS)`→`sorted(feats)` (14-key,
  backward-compatible: sorted(range(13))==range(13) so legacy content-capacity path byte-identical).
  Low-level dropout default stays 0.0; the DRIVER passes 0.1.
- `experiments/run_ptood_probing_ftok.py` (+222): `--probe-family {shared_linear,native_mlp}` +
  `ProbeFamily` registry (ckpt_dir/tunnel_path/fit/predict/save_ckpt/load_ckpt per family) + MLP
  ckpt save/load (`_save_ckpt_mlp`/`load_mlp_ckpt`: head_state_dict+dims+dropout+scaler+wd+seed+epoch+
  param_count → rebuild via build_head, eval() so dropout OFF → round-trip exact) + `_save_fit_histories`
  (per-seed §3 history JSON, MLP only). native_mlp restricted to `--fit-ptid`/`--tunnels-only` (the
  fresh-target PT-OOD/Probe-ID MLP diagnostic is INTENTIONALLY NOT in this pass).
- `experiments/run_fslot_transfer.py` (+78): same `--probe-family`; family-aware preflight (MLP stores
  head_state_dict), predict, ckpt/tunnel load, output namespace; NEW `preflight_feature_cache` (§4).
  Removed 4 now-unused imports (hygiene).
- NEW `experiments/run_fslot_forecasting_comparison.py` (417): §9 ORIGINAL-scale 7-method comparison
  (last-value, seasonal-naive, shared-linear@entrance/@L12+LN, native-MLP@entrance/@L12+LN, native-
  Chronos-2) on the 4 PT-ID datasets. Inverse transform mu+s*sinh(z) (=run_id_forecasting); entrances =
  each family's VALIDATION tunnel l_start; probe seeds averaged PER-WINDOW (not ±1 std); ONE shared
  paired series-cluster bootstrap for MASE (primary)/median-MAE/WQL; WQL from each probe's own quantiles
  (via `_apply_shared_head`), point baselines flagged `point_as_quantiles`, native WQL opt-in
  `--native-wql` (multi-quantile GPU pass). Auto-summary (§9.4) + grouped MASE figure + tidy table.
  BUG FIXED (caught by test): `cluster_bootstrap_apply` needs per_series_sum (S,1)→(B,1); `_boot_mean`/
  `_boot_ratio` now reshape+squeeze (the ratio form gives WQL).
- TESTS: NEW `tests/test_slot_mlp_probe.py` (11) + `tests/test_forecasting_comparison.py` (8);
  `tests/test_fslot_transfer.py` (+5 → 13: MLP ckpt round-trip, MLP off-diag predict-only, MLP paths
  separate/features shared, diagonal gap=0, one tunnel per row). ALL GREEN.
- CPU SUITE (login node, OMP_NUM_THREADS=2): slot_mlp 11, fslot_transfer 13, forecasting_comparison 8,
  shared_forecast 8, tunnel 13, ood_capacity 10, ood_transfer 12, quantile_sets 10, rolling_split 11,
  spectral 10, ridge_r2 4 — ALL PASS. Only `test_ood_targets` fails = PRE-EXISTING pyarrow ModuleNotFound
  (needs `module load arrow/24.0.0`; id_data.py untouched) — environmental, not a regression.
- VERIFIED no-GPU: MLP ckpt round-trip exact (14 heads, param 2,915,616 at real D=768); linear namespace
  == committed; MLP → fslot_mlp/; eval_cell predict-only 14-pt; forecasting output path (table/fig/summary)
  renders on synthetic rows.

REMAINING = GPU runs (USER submits; nothing launched). RUNTIME EST: MLP head ≈26× linear params
(2.92M), full sweep = 4 datasets × 3 seeds × 14 layers × 5 wd × 300 ep → materially heavier than linear;
GPU only. Recipe (Narval; modules+venv, HF offline; [[submit-slurm-jobs-self]]):
1. CONVERGENCE PILOT first (inspect train/val before the full sweep): fit L5/L8/L12/L12+LN only,
   check source-val histories (esp. WindFarms: train improving while val worsens at deep layers =
   overfit). Histories land in fslot_mlp/ptid_runs/*__history.json.
2. `python -m experiments.run_ptood_probing_ftok --probe-family native_mlp --fit-ptid`  (GPU, all seeds)
   then `... --probe-family native_mlp --tunnels-only`  (CPU) → MLP checkpoints + tunnels.
3. `python -m experiments.run_fslot_transfer --probe-family native_mlp`                  (4×4, warm cache)
   `python -m experiments.run_fslot_transfer --probe-family native_mlp --experiment pt_ood` (COMPUTE node)
4. `python -m experiments.run_fslot_forecasting_comparison [--native-wql]`  (native median cached from the
   linear run; --native-wql adds a multi-quantile GPU pass). Needs BOTH families' tunnels on disk.
5. Commit code + results separately (no Co-Authored-By).
INTERPRETATION (per plan): linear worsens late but MLP flat/improves = info present but nonlinearly
encoded; both worsen = representation harder for both; MLP≈native = head capacity explains the gap;
native≫MLP = pretrained co-adaptation matters; MLP train↓ val↑ = overfit; MLP moves the tunnel entrance
= entrance depends on READOUT CAPACITY (not purely representation-intrinsic — state this).

## v4 TWO-AXIS TRANSFER — PLAN APPROVED w/ CORRECTIONS 2026-08-10 (code NOT started)
Separate PRETRAINING status (target vs Chronos-2 corpus) from PROBE-TRAINING status (was the
frozen probe fit on THIS dataset). Terminology EVERYWHERE (code/records/filenames/figures/captions):
PT-ID/Probe-ID (target pretrained, probe fit+tested same set); PT-ID/Probe-OOD (target pretrained,
probe fit elsewhere); PT-OOD/Probe-OOD (target NOT pretrained, probe fit elsewhere); PT-OOD/Probe-ID
(fresh target probe) = the EXISTING run_ptood_probing_ftok eval, demoted to DIAGNOSTIC. No bare ID/OOD.
`pt_status` describes the **TARGET** (correction 6). fslot readout only; content line untouched.

KEY REALIZATION: both new experiments are PREDICT-ONLY re-scorings of things already on disk —
`--fit-ptid` already trained+checkpointed the 4 source fslot probes (3 seeds, 14-pt, wd on source-val,
results/ext_v4_future_tokens/ptood_probing/ptid_checkpoints/), `--tunnels-only` already built the
sustained tunnels, and all 4 PT-ID + 3 PT-OOD target-test fslot caches exist. No GPU / no re-train.
`predict_shared_forecast_probe` (probes.py:852) is the clean frozen primitive: source scaler,
`.eval()`+no_grad, no mutation, iterates sorted(feats) (14-key safe). The ONLY target-refit in the
tree is run_ptood_probing_ftok::eval_target (fits scaler+Linear+wd on the target) — that IS the
fresh-probe diagnostic; it must NEVER be reused for transfer.

MAIN EXP 1 — 4×4 fslot cross-dataset transfer. Sources/targets = Electricity, Uber, M4, WindFarms.
Per source×seed: load frozen source probe (source-fit scaler+Linear+wd, source-val tunnel), predict
on every target-test (build_windows test split, _fslot_feats "test" cache HIT). 4 diagonal cells =
PT-ID/Probe-ID, 12 off-diagonal = PT-ID/Probe-OOD. 14-pt curves (Emb,L1..L12,L12+LN; L12+LN = ref).
MAIN EXP 2 — 1×3 fslot genuine PT-OOD transfer. Source = Electricity ONLY; targets = sg_carpark,
coastal_ts, boom_hourly (build_ood_rolling_windows test, "test_rolling" cache HIT). All = PT-OOD/Probe-OOD.

METRICS (correction 2): raw quantile loss + MASE (median via collect_test_median + canonical
seasonal-naive denom; reuse _ctx_stats/M_SEASON) on BOTH experiments. 4×4 HEADLINE summary =
transfer gap = L_{s→t}(ℓ_s)/L_{t→t}(ℓ_t) − 1, ℓ_s = source val-selected layer, ℓ_t = TARGET diagonal
probe's val-selected layer (BOTH from validation, never test); diagonal cells → 0 by construction;
gap defined for the 4×4 ONLY (exp 2 has no in-transfer diagonal). min-over-test relative regret =
SUPPLEMENTARY only (a broken probe still has a 0-regret layer). tunnel overlay = the row's source-val
sustained tunnel (existing l_start).

RECORD SCHEMA (every row): pt_status(TARGET) / probe_status(Probe-ID diag else Probe-OOD) /
source_dataset / target_dataset / probe_fitted_on(=source) / wd_selected_on(=source-val) /
tunnel_defined_on(=source-val) / l_start_sustained(= existing authoritative l_start) /
final_reference="L12+LN". NOTE: original spec's l_start_first_crossing is DROPPED — tunnel.py is
frozen (correction 1), no first-crossing helper; consume existing l_start only.

FILES (code NOT started; awaiting delivery-mode go-ahead):
- probing/tunnel.py — DO NOT TOUCH (correction 1).
- NEW experiments/run_fslot_transfer.py — one driver, two modes (--experiment transfer_4x4 | pt_ood).
  load_ptid_ckpt (reads _save_ckpt format: rebuild StandardScaler from mean/scale + nn.Linear from
  state_dict); preflight (correction 8: 14 heads+scalers/ckpt, window==label counts/cache); predict-only
  transfer; MASE; transfer-gap + supplementary regret; make_4x4_figure (rows=source cols=target,
  mean-over-seed + seed band, source-val tunnel shaded, diag/off-diag compound titles, SHARED Y PER
  COLUMN — correction 7) + make_pt_ood_figure (1×3, Electricity tunnel). Outputs → results/
  ext_v4_future_tokens/{fslot_transfer, fslot_pt_ood}/ (NEW dirs; nothing moved — correction 5).
- experiments/run_ptood_probing_ftok.py — RELABEL to "PT-OOD/Probe-ID (fresh target probe)" diagnostic
  (docstring/prints/titles/record pt_status=target,probe_status=Probe-ID) IN PLACE; keep same output
  paths (correction 5); keep fit_ptid + compute_ptid_tunnels (shared inputs); expose load_ptid_ckpt.
- NEW tests/test_fslot_transfer.py — 4-diag/12-offdiag; off-diag calls NO _fit_slot_scaler/fit_* on
  target (patch→raise); frozen-ckpt predict == in-memory predict; exp2 exactly 3 targets; every curve
  len==NUM_LAYERS+1 ref=L12+LN; tunnel from source-val only; no bare ID/OOD in records; content
  pipeline unchanged (test_ood_transfer stays green).
- UNTOUCHED: run_ood_transfer.py, run_ood_pretrain_transfer.py, probes.py quantile fns, id_data.py.
  run_spectral/make_paper_figures = follow-up terminology pass (out of scope).
DECISIONS answered by user: MASE yes; one driver two modes; content terminology later.
STATUS 2026-08-10: DRIVER DONE (user said "go ahead and edit"). run_ptood_probing_ftok.py +=
_ptid_ckpt_dir/_scaler_from_arrays/load_ptid_ckpt (additive loader, rebuilds StandardScaler from
mean/scale + nn.Linear from state_dict; relabel still pending). NEW experiments/run_fslot_transfer.py
= BOTH modes (transfer_4x4 + pt_ood) complete: preflight (14 heads+scalers, window==label counts),
predict-only eval_cell, 14-pt _fslot_mase (mase_context m=24; compute_mase is 13-only so a twin),
transfer gap (diag=0 verified), supplementary regret, two-axis record schema (pt_status=TARGET,
no bare ID/OOD), make_4x4_figure (shared y per column) + make_pt_ood_figure. device=cpu (predict-only,
warm cache). VERIFIED no-GPU: py_compile; import; real-checkpoint load (14 heads, scaler rebuild) +
synthetic eval_cell (per-window mean==scalar, ql finite) + aggregation (diag gap==0, records carry
metadata, no bare ID/OOD, both figures render). NOTE: synthetic-feature MASE overflows sinh (random
feats → real scaler → huge zhat) — artifact, finite on real features.
TESTS DONE: tests/test_fslot_transfer.py — 8 checks, all PASS (4diag/12offdiag; off-diag predict-only
via patched fit-fns→raise; frozen-ckpt predict==in-memory, self-built ckpt; exp2=3 targets; 14 depths
ref=L12+LN; selected layer + tunnel from source-VAL only; no bare ID/OOD in records; content fit→predict
==quantile_probe). No regressions: shared_forecast 8, ood_transfer 12, tunnel 13.
RELABEL DONE: run_ptood_probing_ftok relabeled IN PLACE (no path move) → PROBE_ID_DIAG = "PT-OOD /
Probe-ID (fresh target probe)"; docstring demoted to DIAGNOSTIC + cross-refs run_fslot_transfer; eval
print/figure titles/summary header carry the quadrant; per_target payload += pt_status(TARGET)/
probe_status="Probe-ID"/quadrant. fit_ptid/compute_ptid_tunnels unchanged (shared inputs).
REMAINING = USER RUNS (nothing submitted): (1) 4×4 — `python -m experiments.run_fslot_transfer`
(login/warm cache OK, predict-only; needs the committed ptid_checkpoints + tunnels); (2) pt_ood —
`python -m experiments.run_fslot_transfer --experiment pt_ood` on a COMPUTE NODE (SG/BOOM rebuild
rolling windows, heavy for login). Then commit code + results separately.

## TUNNEL CRITERION → SUSTAINED-PLATEAU ONLY (2026-08-10, code done, not re-run)
USER DECISION: a tunnel = region where performance has SATURATED AND STAYS saturated, so the
first-crossing definition is REMOVED ENTIRELY and the sustained plateau is the sole criterion:
`l_start = min{ l : val[j] <= (1+tol)*val[last] for EVERY j>=l }` (tol=0.05, VALIDATION only,
one-sided so better-than-last counts). REVERSES the 2026-08-08 advisor call (first-crossing primary);
user is aware. Applies to BOTH readouts (content + fslot) — the criterion is not readout-specific.
- `probing/tunnel.py`: `tunnel_start` now IS the sustained backward-scan; `tunnel_start_sustained`
  DELETED; records drop `l_start_sustained`/`tunnel_sustained`/`test_criterion_holds_sustained`,
  `tunnel_definition` = "sustained_plateau". `l_start`/`tunnel`/`D_ID` key off the sustained boundary.
  Excursion M: `max_excursion_val` is now <= tol BY CONSTRUCTION (val suffix forced flat); `M_test`
  stays informative (test not constrained → measures whether the val plateau holds OOS).
- Consumers updated (dropped the removed fields + relabeled "first-crossing"→"sustained plateau"):
  run_ptood_probing_ftok.py + run_ptood_probing.py (prints, _tunnel_figure, aggregate row, docstrings);
  make_paper_figures.py (dropped tunnel_start_sustained import; threshold_sensitivity now one `l_start`
  column; docstring). tests/test_tunnel.py reworked (13 still pass; U-shape test now asserts sustained
  refuses to open before the hump; record test asserts _sustained fields ABSENT + definition string).
- Effect: entrance moves later/equal vs first-crossing (sustained l_start >= first-crossing l_start),
  tunnels narrower/deeper, D_ID over a shorter span. L12+LN-as-reference unchanged (both used curve[-1]).
- VERIFIED no-GPU: compile all touched; test_tunnel 13, shared_forecast 8, quantile_sets 10, spectral 10,
  ood 12; 14pt ftok smoke (tunnels end L12+LN, sustained boundaries now vary L2/L10/L11 with the late
  hump, D_ID refs post-LN); threshold_sensitivity single-column CSV.
- ⚠ STALE ON-DISK: existing results/extended_v3_rolling/tunnels/*.json were computed under first-crossing
  and still carry the old fields → re-run `--tunnels-only` (both drivers) to refresh under the new
  definition before regenerating figures. Harmless to read (extra keys ignored; drivers no longer read
  the removed ones) but the l_start values are stale.

## v4 POST-FINAL-LN READOUT POINT — added 2026-08-10 (fslot only; code done, not run)
The shared-head probe mirrors Chronos-2's native head, but the native head reads the forecast slots
AFTER the encoder's final LayerNorm (chronos2/model.py:190 → output_patch_embedding at :730-732), while
our block hooks capture PRE-final-LN L12. So the fslot line now adds ONE extra readout point beyond L12:
index NUM_LAYERS (=13), label "L12+LN" = the POST-final-LN slots (extract_kout_features's final["fslot"],
the native-head input, already cached as the `fslot_final` key — NO extraction/model change). USER-LOCKED:
this post-LN point is the tunnel's **"last" reference** (D_ID/M/entrance all measure against curve[-1]);
fslot curves = 14 points, content stays 13.
- Design = DATA-DRIVEN length (len(feats)/curve.shape[0]), no new global constant. tunnel.py already
  length-agnostic (tunnel_record_multi/tunnel_start/max_excursion use curve[-1]) → 14-long curves make
  L12+LN the reference automatically; content (13) flows through unchanged (byte-identical).
- Files changed (fslot/v4 ONLY; content + config.NUM_LAYERS/LAST_LAYER untouched):
  probes.py fit_shared_forecast_probe_explicit_val/predict_shared_forecast_probe loop `sorted(feats)`
  not range(NUM_LAYERS); run_ptood_probing_ftok.py `_fslot_feats` appends final["fslot"] as key 13, all
  producers iterate dict keys, every d_stat_boot passes last=wl.shape[0]-1, figures use LAYER_LABELS +
  np.arange(len(curve)); run_spectral.py fslot `_load_features` adds feats[13]=stacked fslot_final,
  analyze_dataset loops sorted(feats), make_figures data-driven + `_point_labels`; make_paper_figures.py
  FULL_LABELS/_labels(n), painters derive n from record, _mark_tunnel(last_idx), d_/m_stat_boot(last=..).
- VERIFIED no-GPU: test_shared_forecast_transfer now 8 (new 14-key test: 13 shared points byte-identical
  to a 13-key run, key-13 finite); regressions green (quantile_sets 10, tunnel 13, spectral 10, ood 12);
  scratchpad smokes — ftok aggregate 14pt (tunnels end L12+LN, D_ID references post-LN, d_stat_boot(last=13)
  agrees, CSV/figs), spectral fslot_final (14 keys, 14-layer record+figs), paper painters (13 & 14, marker
  on last tick, content last=L12 / fslot last=L12+LN). Real GPU handshake (extract_kout_features →
  fslot_final → 14-pt curves) is part of step B (user submits).

## v4 FUTURE-TOKENS PIVOT — headline readout change (DECISION 2026-08-09; STEP A DONE)
**Decision (user):** the PRIMARY/headline probe readout is now the **shared-head future-token probe**
(`shared_forecast_token_probe`: one linear `Linear(768, Q·P)` shared across the K=ceil(H/P) native
forecast-slot states from the `num_output_patches=K` pass), replacing pooled content as the main method.
This is a **deliberate pivot** made knowing Finding #3 (the shared-forecast readout does NOT show the
pooled tunnel: best layer late L8–L11, ~0 mid-vs-last delta). Headline story shifts to the Chronos-native
readout; pooled content/reg are KEPT as the comparison, not dropped.
**Namespace rule:** every result NEW vs commit f1ab7f4 → `results/ext_v4_future_tokens/…` (NOT extended_v3_rolling).
**Spectral geometry:** SVD/effective-rank on the (n, K, 768) slot states = **stack slots → (n·K, 768)**
(every forecast slot a row; not mean-over-K, not per-slot).

**Already supports it:** `run_id_forecasting.py` runs this as `fslot_K` today (ID pipeline only).
**Gap for the ACTIVE line (tunnel/PT-OOD/spectral/paper figs — all hard-wired POOLING="content"):**
- (A) **Probe plumbing — DONE 2026-08-09.** Added `fit_shared_forecast_probe_explicit_val` +
  `predict_shared_forecast_probe` to `probes.py` (linear-shared-head twins of the `quantile`
  explicit-val/predict pair): explicit-temporal-val wd selection, slot-scaler on FULL train, keep
  chosen-wd full-train model, frozen predict on arbitrary targets; `selection` = {val_loss_by_wd,
  chosen_wd} (source_selected_layer-compatible); fitted dict carries family="shared_forecast",
  output_patch_size, K. Threaded `init_seed` through `_fit_shared_forecast_linear` (default SEED →
  existing calls byte-identical) for the 3-run protocol. NEW `tests/test_shared_forecast_transfer.py`
  (7 no-GPU tests: fit→predict==combined probe rtol 1e-5, frozen reuse unmutated, no target in fit sig,
  selection shape, 3-D contract, init_seed determinism, per-window mean==scalar). All green +
  test_quantile_sets 10/10 + test_ood_transfer 12/12 (no regressions). Did NOT add a plain
  `fit_shared_forecast_probe` (80/20 carve) — v4 tunnel/PT-OOD only needs explicit-val + predict; add
  later only if a square future-tokens transfer is wanted.
- (B) **Features — GPU re-extraction.** Consumes (n,K,768) from `extract_kout_features`, NOT the (n,768)
  `extract_window_features` content caches → re-extract for 4 PT-ID rolling + 3 PT-OOD targets on GPU.
  Cache keys already namespaced (`..._K<K>_H<horizon>`); rolling split names to avoid legacy collision.
- (C) **Driver edits.** PT-OOD/tunnel driver DONE 2026-08-09 (`experiments/run_ptood_probing_ftok.py`,
  SIBLING of run_ptood_probing.py — content driver untouched/byte-identical). Uses extract_kout_features
  → feats["fslot"] + fit/predict_shared_forecast_probe; ALL 3 PT-ID seeds fit fresh (no legacy seed-0
  reuse — no fslot 4×4 checkpoints exist); dataset_set stays extended_v3_rolling (roster+windows+cache
  namespace), OUTPUTS overridden to results/ext_v4_future_tokens/ via OUT_ROOT (documented exception to
  set=output coupling; separation is by READOUT). Stages: --fit-ptid (GPU, all seeds, idempotent) /
  --tunnels-only (CPU) / default GPU per-target eval / --figure-only (CPU aggregate). K=4, H=64, q9.
  Verified no-GPU: py_compile, import, fail-loud (--tunnels-only → "run --fit-ptid first"), and a
  scratchpad synthetic smoke of compute_ptid_tunnels+aggregate (4 tunnels, 12 cells, 8 figs, CSV — glue
  clean).
- (C-aggregators) DONE 2026-08-09. run_spectral + make_paper_figures PARAMETERIZED with `--readout
  {content,fslot}` (NOT siblings — read-only aggregators, no behavioral fork; content=default,
  byte-identical). run_spectral: fslot reads the K-slot cache (K4_H64 key), stacks fslot_L{i} (N,K,768)
  → (N·K,768), writes results/ext_v4_future_tokens/spectral/spectral__<tag>__fslot__probe_input__*.json.
  make_paper_figures: `--readout fslot` rebinds `ptood`→run_ptood_probing_ftok in main (identically-named
  helpers → zero call-site changes) + _results_root()=getattr(ptood,'OUT_ROOT',ID_OUT_DIR) routes
  paper_figures/spectral reads to v4. Verified no-GPU: py_compile, path routing (content=v3 unchanged,
  fslot=v4 + K4_H64/__ood/_rolling correct), fslot stacking loader + analyze_dataset on a synthetic
  K-slot cache (stacked (N·K,d), erank per layer, JSON tag=fslot). Regressions: spectral_metrics 10/10,
  tunnel 13/13.
- (D) **Spectral geometry:** DONE via the run_spectral `--readout fslot` stack-slots path above.
- NOT added (YAGNI): plain `fit_shared_forecast_probe` (80/20 carve) — no driver needs it; only a future
  SQUARE future-tokens transfer would. ~15-min add mirroring fit_quantile_probe if ever wanted.
**Build order:** A DONE → C-tunnel DONE → C-aggregators/D DONE → **B (GPU K-slot extract, PT-ID rolling
+ PT-OOD targets) = NEXT, user submits** → tunnels+ptood+spectral+paper figures run → commit code+results
separately. SHARED-HEAD IMPLEMENTATION COMPLETE (nothing run yet — user pausing to make other changes).
**Resolved:** K=4/H=64 kept; ext_v4 reuses the extended_v3_rolling rolling windows verbatim (deterministic
split, only the feature type + probe change); namespace = output-dir override (user), driver = sibling (user).

## PT-ID/PT-OOD tunnel reframing — CODE DONE + Stage 1 RUN 2026-08-08; PT-OOD GPU run PENDING
New framing (user + advisor): domain status relative to the BACKBONE (pt_id = 4 extended_v3_rolling
sources; pt_ood = sg_carpark/coastal_ts/boom_hourly; ft_id/ft_ood axis reserved for the future
fine-tuning block). PRIMARY analysis = per-dataset validation-defined TUNNEL RANGE, not a single
best layer: l_start = min{l : val(j) <= 1.05*val(last) ∀ j >= l} (one-sided suffix criterion),
frozen, then checked on test. PT-OOD = 2023 protocol: FRESH per-layer probes trained on each
PT-OOD dataset (rolling split, wd on target-VAL only), NOT the frozen-source transfer — the old
run_ood_pretrain_transfer 4×3 is retained as the legacy "zero-shot readout transfer" diagnostic.
Stats: D_ID(s) = (test(last)-test(l_s))/test(l_s) on s; D_OOD(s,t) same at s's l_s on t's fresh
curve; Delta(s,t)=D_OOD−D_ID, CI = independent-replicate bootstrap difference (disjoint test sets;
D CIs are paired within-dataset via the shared cluster-bootstrap count matrix).
- Files: NEW `probing/tunnel.py` (criterion, records, domain status, d_stat_boot/delta_stat);
  `probing/id_data.py::build_ood_rolling_windows` (rolling split on OOD targets, cluster-unit
  balancing + fail-loud train coverage, canonical MASE denom, val=test series, defaults: all
  eligible series, train cap 1394); NEW `experiments/run_ptood_probing.py` (--tunnels-only /
  GPU eval per target / --figure-only aggregate; outputs results/extended_v3_rolling/{tunnels,
  ptood_probing}/; feature caches use *_rolling split names to avoid the legacy 4×3 test cache);
  NEW `tests/test_tunnel.py` (11 tests). All green + no regressions (quantile_sets 10, ood_transfer
  12, rolling_split 9+2).
- TWO-DEFINITION update (2026-08-08, advisor): PRIMARY = first-crossing (2023-paper style)
  l_start = min{l : val(l) <= 1.05*val(last)}; the sustained suffix version demoted to ROBUSTNESS;
  new excursion stat M = max_{j>=l_start}(loss(j)/loss(last)−1) quantifies post-entrance
  non-monotonicity instead of forcing flatness. D stats key off the PRIMARY boundary. tol stays 5%.
- Stage 1 RESULT (q9 seed0, temporal val): PRIMARY tunnel = **[L1, L12] for ALL FOUR** (first
  crossing at L1 everywhere). Sustained: uber/wind [L1,12] (M_test≈0, holds on test), M4 [L11,12]
  (M_val +0.10, M_test +0.08, sustained check FAILS on test at L1-start), elec [L12,12] (M_val
  +0.21, M_test +0.12, fails). D_ID at L1: uber +0.020*, wind +0.041*, m4 +0.061*, elec −0.001
  (null). Narrative: Chronos-2 reaches final-layer quality by L1 on every PT-ID dataset, but a
  sustained plateau is strongly dataset-dependent (elec/M4 have a late-middle hump).
- 3-RUN update (2026-08-08, advisor): every condition = 3 independent probe runs (RUN_SEEDS
  0/1/2; backbone frozen → run seed = probe Linear-init seed, the ONLY randomness in the
  deterministic full-batch fit; `init_seed` threaded through `_fit_quantile_linear` /
  `fit_quantile_probe_explicit_val`, default SEED → seed-0 byte-identical). Tunnel defined from
  the MEAN validation curve (never per-seed indices averaged); all per-run curves retained
  (`tunnel_record_multi` schema: val/test_loss_by_run + mean/std + D_ID + M_test + run_type).
  Windows/features are run-seed-independent → seed-0 rolling checkpoints reused as run 1, no
  re-extraction; seeds 1/2 = warm-cache linear refits (`--fit-ptid-seeds`). PT-OOD stage loops
  the 3 seeds per target (features extracted once). D_ID/D_OOD/Delta = point stats on mean
  curves; CI = cluster bootstrap of seed-averaged per-window losses (identical windows across
  runs, asserted); per-run D's recorded for seed sensitivity. New outputs keyed `runs0-1-2`
  (single-seed files kept as legacy). Future backbone conditions recorded via run_type: ft_seed
  (3 FT seeds × 1 probe seed) and random_init (3 independent random backbones) — NOT implemented.
- SPECTRAL GEOMETRY (2026-08-08, code done, NOT yet run): NEW `probing/spectral_metrics.py`
  (project-wide definition: center → svdvals → p=s²/Σs² → H=-Σp log p → erank=exp(H); also
  pc1_fraction, numerical_rank, full spectrum; `subsample_metrics` = repeated subsampling
  WITHOUT replacement — no naive bootstrap) + `tests/test_spectral_metrics.py` (10 tests, PASS)
  + `experiments/run_spectral.py` (reads feature caches directly — no model/window rebuild;
  PRIMARY location=probe_input = cached content-pooled block outputs, L12 = PRE-final-LN,
  pre-probe-StandardScaler — recorded; post_final_ln fails loud, needs a GPU K-pass; pretrained
  backbone = ONE curve/dataset, backbone_seed null; --repr-sample-size 4096 --repr-subsamples
  200 --repr-subsample-frac 0.8; outputs results/extended_v3_rolling/spectral/). 200×13×4 SVDs
  too heavy for login → compute node.
- PAPER FIGURES (2026-08-08, code done): NEW `experiments/make_paper_figures.py` — reads-only
  pipeline: Fig1 PT-ID 4-panel test loss + mean-val tunnel shading; Fig2 same-tunnel effective
  rank; combined 2×4; Fig3 PT-OOD 3-panel fresh-probe curves + 4 tunnel-entrance markers; Fig4
  4×3 Delta heatmap (Δ^(b)=D_OOD^(b)−D_ID^(b), B=5000, • = CI excl 0) + cells JSON; PT-ID
  summary CSV (D_ID + M_test with new `tunnel.m_stat_boot` CIs, boundary fixed); threshold
  sensitivity (tol 0.02/0.05/0.08/0.10, both defs, primary stays 0.05); geometry Δr_eff prep
  CSV; supplementary val curves / per-run overlays / PT-OOD erank. Graceful skips naming the
  producing command. Synthetic smoke of all painters PASS; test_tunnel now 13.
- NEXT (user submits): (1) short salloc (CPU ok, GPU faster) → `python -m experiments.
  run_ptood_probing --fit-ptid-seeds` (PT-ID seeds 1/2, warm caches) → login `--tunnels-only`;
  (2) GPU salloc with OOD_TARGET_ROOT + HF offline → `python -m experiments.run_ptood_probing`
  (3 runs × 3 targets) → `--figure-only`; (3) compute-node CPU job: `python -m experiments.
  run_spectral` (PT-ID now, PT-OOD after step 2's caches exist); (4) login:
  `python -m experiments.make_paper_figures`; (5) commit code + results separately.

## What this is
Layer-wise linear probing of frozen **amazon/chronos-2** (12 encoder blocks, d=768).
Question: at which depth is the forecast most *linearly decodable*, measured in the model's
own currency (Chronos-2 quantile loss) and against the native head (MASE)? Phase 1
(classification on UEA) is archived at tag `classification-phase-final`.

## OOD 4×4 ROLLING-ORIGIN re-run (extended_v3_rolling) — CODE DONE 2026-07-31, seed-0 GPU smoke PENDING
**Why:** extended_v2's 4×4 has a split confound — elec/uber/wind use within_series but m4_hourly
falls back to cross_series (series < 2·(C+H)), so M4 trains/vals on DIFFERENT series and its rel-gains
aren't comparable to the other rows (the source of M4's negative transfers?). Fix = ONE uniform
rolling-origin within-series split for ALL FOUR, new namespace, extended_v2 KEPT as sensitivity comp.

**Locked budget (from the yield screen, seed 0): train=1394 / val=262 / test=262.** NOT the old 1500/650:
a clean non-overlapping-target rolling split yields exactly 1 test + 1 val window PER SERIES, so
test/val are capped by the smallest series count (Uber=262). Upside: with 1 test window/series the
cluster bootstrap reduces to resampling 262 series-level evaluation units (no within-series correlation
to model). Train=1394 = M4's full supply (the floor). All 4 datasets: 0 cross-split target overlap.

**Protocol (per series, C=512 H=64):** H-spaced origins t∈{512,576,...}, t≤L-H → targets never overlap;
valid = finite ctx+target AND non-constant ctx (same rule as `_make_examples`, so missing/const dropped
leakage-free); eligible ⇔ ≥3 valid origins (L≥C+3H=704). LAST origin→test, 2nd-last→val, earlier→train.
Same deterministic 262 series carry val+test; train from EVERY eligible series, cluster-balanced to 1394.

**Files changed (extended_v2 path byte-identical — dispatch guards on the set):**
- `probing/id_data.py`: `extended_v3_rolling` in ID_DATASET_SPECS; `BUDGET_BY_SET`=(1394,262,262) 3-tuple;
  `ROLLING_SETS`; `_rolling_valid_starts`; `_build_rolling_windows` (the split — returns X_val/y_val/
  Y_val_traj/series_val + meta.selected_series + meta.origins{train,val,test}; canonical `test_denominator`
  = seasonal-naive of history STRICTLY before the test target → `mase_canonical=True`; fail-loud RuntimeError
  if any selected val/test series keeps 0 train windows). Dispatch at top of `build_windows`.
- `probing/probes.py`: `fit_quantile_probe_explicit_val` — wd chosen on the EXPLICIT temporal val split,
  scaler+weights on FULL train (val never enters weights). NO 80/20 carve for this set. Same `selection`
  dict shape → save_checkpoints / source_selected_layer / predict_quantile_probe unchanged.
- `experiments/run_ood_transfer.py`: ORDER_BY_SET 4×4; imports ROLLING_SETS + the explicit-val fit;
  `get_source_probe` extracts a "val" feature split + uses the explicit-val fit for rolling sets only.
- `tests/test_rolling_split.py`: 9 no-GPU tests (targets non-overlap; ctx precedes origin; val≡test series;
  ≥1 train/selected series; deterministic seed0; MASE excludes test target; real budget resolves; all-four
  rolling; extended_v2 still cross_series/1500/650).
- UNCHANGED and confirmed OK for the new set: `run_ood_screen.window_supply` (reads the meta keys),
  `run_ood_baselines.main` (re-derives 4×4 order), `_idf_prefix` (auto-namespaces → `IDF_<tag>__extended_v3_rolling`).

**Verified (login CPU):** py_compile clean; rolling tests 9/9; test_quantile_sets 10/10 + test_ood_transfer
11/11 (no regressions); driver wiring (namespace → results/extended_v3_rolling/, order = the 4, dispatch live).

**Remaining (GPU — user submits):**
- [ ] Seed-0 SMOKE cell (1 cell, inspect before the matrix): `salloc --account=def-irina --gres=gpu:1
      --cpus-per-task=2 --mem=32G --time=1:00:00` → modules+venv → `export HF_HOME=$SCRATCH/chronos2/hf_cache
      HF_HUB_OFFLINE=1` → `python -m experiments.run_ood_transfer --dataset-set extended_v3_rolling
      --source-dataset monash_electricity_hourly --target-datasets uber_tlc_hourly`. Check: `[rolling]
      explicit temporal val: 262 windows/262 series`, `[fit-explicit-val] L0..L12`, per_source JSON n_test=262.
- [ ] Full 4×4: `sbatch job_ood_transfer.sh --dataset-set extended_v3_rolling --source-dataset <tag>` ×4
      (elec/uber/m4/wind) → `python -m experiments.run_ood_transfer --dataset-set extended_v3_rolling
      --figure-only` → `python -m experiments.run_ood_baselines --dataset-set extended_v3_rolling`.
- [ ] Old-vs-new comparison table (esp. M4): old/new split_mode, source-selected layer, rel-gains,
      ID/OOD mean+median. Core question: did M4's negative transfers come from cross_series, or persist?
- [ ] Seeds 1–4 ONLY after seed-0 validated (needs the seed threaded into the probe init per the
      single-seed caveat) — do not launch early.
- `experiments/rolling_yield_screen.py` (count-only pre-flight) committed to the tree; reusable.

## Current state — everything below is IMPLEMENTED and RUN
The full forecasting pipeline is built, run on Narval GPU, and analyzed. Code is committed;
the q1/q9 results + regenerated bootstrap are on disk but **not yet committed** (see Pending).

Pipeline (all in `probing/` + `experiments/`):
- **Quantile probe** — per layer `Linear(d, Q·H)`, trained AND scored with Chronos-2's exact
  quantile loss (arcsinh target space, lower=better), per-layer weight-decay picked on a
  validation carve. `probing/probes.py::quantile_probe`.
- **Shared forecast-token probe** — one shared `Linear(768, Q·16)` over the K=⌈H/16⌉ native
  forecast slots; linear analogue of Chronos-2's output head. Controlled content_K/reg_K from
  the same K-slot pass (encoder attention is non-causal, so K changes all token states).
- **MASE vs native** — probe median un-transformed to raw units vs native `predict_quantiles`,
  same windows, in-context seasonal-naive denominator (m=24, `mase_context`).
- **Cluster bootstrap CIs** — `experiments/run_bootstrap.py`, resamples WHOLE test series
  (B=5000, seed 0, paired), comparison layer L* frozen from validation. CPU/login-node.
- **Capacity ablation** — `--quantile-set {q1,q9,q21}` (49k / 443k / 1.03M head params).
  Cross-set comparison only via `mean_pinball_loss` (= loss/2Q). Disjoint output paths per set.

Data: 4 long hourly datasets (electricity, kdd_cup_2018, pedestrian_counts, uber_tlc), all in
`autogluon/chronos_datasets`, within-series split, C=512 / H=64. `probing/id_data.py`.

## Findings (from `results/bootstrap/bootstrap_summary.json`)
1. **Mid > last on pooled readouts, 3/4 datasets.** Δ = qloss(L11) − qloss(L*), positive =
   earlier layer better (content): electricity L2 +0.27 [0.20,0.33], pedestrian L4 +0.18
   [0.12,0.25], uber L9 +0.68 [0.60,0.77]; CIs exclude zero. KDD = honest null (val picks L11).
   Same on MASE and REG pooling.
2. **Survives smaller heads.** q1/q9 reproduce the pattern (same 3 positive, KDD null).
3. **Specific to POOLED readouts.** The shared forecast-slot probe (fslot_K) is uniformly the
   strongest probe and its best layer moves late (L8–L11) with ~0 mid-vs-last delta. Pooling
   into one vector is what makes intermediate layers look better.
4. **No linear probe matches native** (native MASE ≈ 0.6–0.85 < every probe) — the claim is
   about *linear decodability*, not representation quality.

Caveats to keep in any writeup: datasets are in-pretraining (decodability, not generalization);
univariate use only; pooled vs shared differ in representation AND readout capacity.

## Pending
- [ ] **L0 (input embedding) added — implemented in code, NOT yet run.** L0 = embedded token
      sequence entering block 1 (content + special tokens, pre-attention); L1..L12 = block
      outputs; native head consumes L12. `NUM_LAYERS=13`, `LAST_LAYER=12`, `MIDDLE_BAND` +1
      shifted (config.py AND run_perdataset.py local). Captured via a `register_forward_pre_hook`
      on `encoder.block[0]` in all 3 extractors; block i output -> key i+1. Deterministic
      `eval()` with train/eval-state restore added. Fail-loud guard: a stale 12-layer cache now
      errors instead of KeyError. All `L11` names -> `last`/`LAST_LAYER` repo-wide (bootstrap
      keys/labels/PNG filename `delta_vs_last`, run_id_forecasting JSON keys `last`/`last_mase`/
      `retention_last`/`excess_last`). No-GPU tests + import smoke test PASS. Structural docs
      updated; **README result numbers deferred until regeneration.**
  - [ ] **Invalidate caches (USER does this, after reviewing the diff):**
        `rm -f features_cache/IDF_*` (ID) and
        `rm -f features_cache/*__train__* features_cache/*__test__*` (UEA).
  - [ ] Pre-fetch UEA datasets on the login node (aeon; compute nodes are offline) — the 8 in
        run_perdataset.DATASETS.
  - [ ] Regenerate: `sbatch job_id_forecasting.sh` (+ q9/q1) → re-run run_perdataset (GPU) →
        `python -m experiments.run_bootstrap` → `python -m tests.test_quantile_sets`.
  - [ ] After regen: bump README headline L* labels +1, relabel native head L12, drop the
        "pending regeneration" note.
- [ ] **Commit the q1/q9 results + regenerated bootstrap + README** (currently uncommitted).
      Convention: implementation and results in separate commits. Suggested:
      `git add results/ && git commit -m "q1/q9 capacity-ablation results + bootstrap CIs"`,
      and README separately.
- [ ] Minor: `requirements.txt` dataset comment is stale (names m4/solar) — one-line fix.
- [ ] Delete stray `logs/*.out` from the tree if not wanted (untracked).

## OOD 4×4 extension (roster LOCKED 2026-07-31 — IMPLEMENTING)
Extend the cross-dataset linear-probe transfer from 3×3 to 4×4. Drop KDD from the MAIN matrix
(persistence-dominated: last-value ≤ seasonal-naive, the KDD warning); preserve the existing
KDD-inclusive run at `results/extended_v1/ood_transfer/` as a labeled legacy/pilot appendix (the
new set writes elsewhere, so it is untouched — add a README_LEGACY.md there).

**Roster = new dataset set `extended_v2`** (→ `results/extended_v2/…`), DATASET_ORDER =
electricity, uber_tlc, **m4_hourly**, **wind_farms_hourly**.
- **4th dataset = `wind_farms_hourly`** (accepted from the screen; renewable-generation domain,
  distinct from elec/uber). CPU screen (1500/650 budget, exact build_windows+_subsample): 337
  series, ~8784 h each (all within_series-capable), split_mode=**within_series**, supply 14 042
  train / 7 391 test → easily meets 1500/650, 321/261 distinct series contribute, no long-series
  dominance (top series 0.9%). last-value MASE 2.043 > seasonal-m24 1.322 (ratio 1.545 → healthy,
  NOT persistence-dominated). Cache 5.4 MB. **Caveat: 17% missing data** — handled leakage-free
  (`_make_examples` drops any non-finite 576-span; 0 denom issues; plenty of clean windows remain).
  **Still-open validation: native Chronos-2 MASE** on wind_farms + M4 (short GPU salloc) — recommended
  before final conclusions (see [[dataset-screening-rigor]]).
- **Wiki → dropped:** no hourly Wiki in `autogluon/chronos_datasets` (only daily `wiki_daily_100k`).
  WeatherBench 2m_temperature was tried as the substitute and **REJECTED** (last-value 1.223 <
  seasonal 1.300 → persistence warning, KDD-like). Alibaba/Azure/Borg cloud traces are NOT in this
  repo → shortlist rule collapsed to Wind Farms Hourly.
- **m4_hourly:** all 414 series 748–1008 < 2·(C+H)=1152 → FORCED onto `cross_series` (leakage-free;
  ~1555 train / ~667 test). Accepted + documented; in-context MASE stays defined. Other three use
  within_series. Screen: seasonal 1.55 ≪ last 17.9 (healthy).
- **Budget:** matched **TOTAL** = **1500 train / 650 test PER DATASET** (NOT a per-series cap), via
  the EXISTING uniform `_subsample`. M4 is the floor (~1555/667); elec/uber/wind supply far more and
  are capped down. Do NOT introduce a per-series cap or change the sampling protocol.
- **Season:** all four hourly → **M_SEASON=24 UNCHANGED**. m=168 weekly is an optional extra screen
  baseline. No per-dataset season threading needed.
- **Scope:** LINEAR quantile probe only (run_ood_transfer + run_ood_baselines). Capacity heads
  (run_ood_capacity) stay on the committed 3×3 — out of scope.

**Implementation — DONE in code (2026-07-31), GPU runs pending:**
1. [x] `probing/id_data.py`: `extended_v2` set (elec, uber, m4_hourly, wind_farms_hourly) +
   `BUDGET_BY_SET` (extended_v2→1500/650, others→3000/1500); `build_windows(target_train/test=None)`
   resolves the active-set budget (extended_v1/phase0 unchanged); budget recorded in meta.
   `run_id_forecasting.py`: ID_STYLE + DATASET_NOTES for wind_farms/extended_v2.
2. [x] `run_ood_transfer.py` N×N: `ORDER_BY_SET` (extended_v1 stays committed 3×3 elec/kdd/uber;
   extended_v2 = 4×4) + `_derive_datasets()` re-run after --dataset-set; all `3`→NDATA; figure
   filename `transfer_matrix_{N}x{N}`; --target-datasets default resolves post-override.
3. [x] `run_ood_baselines.py` N×N (re-derives DATASET_ORDER/SHORT from ood in main; figures N×N).
4. [x] **Cache-collision fix** (`probing/extraction.py` `_idf_prefix` + native cache in
   run_id_forecasting): ID feature/kout/native caches namespaced by set for new sets
   (`IDF_<tag>__extended_v2`), legacy sets keep `IDF_<tag>` → extended_v1's committed 13 GB caches
   still HIT; extended_v2 (different windows) can't collide/fail-loud. REQUIRED because the 1500/650
   budget changes the windows for shared elec/uber tags.
5. [x] `experiments/run_ood_screen.py` (reusable screen: diagnostics + last-value/seasonal-m24/m168
   /native) — RAN on extended_v2 (CPU): all four seasonal-stronger (elec 3.67, uber 1.27, m4 13.16,
   wind 1.55), artifact at `results/extended_v2/ood_transfer/screen/`. m4 supply 1534/688 ≥ budget.
6. [x] Tests: 3 added to test_ood_transfer (per-set order 4×4/3×3, budget+cache namespacing, gated
   m4/wind windows+split). `results/extended_v1/ood_transfer/README_LEGACY.md` (KDD legacy marker).
7. [x] **Normalized relative-gain (2026-07-31, additive, post-hoc from saved npz):**
   `relative_gain_pct = 100*(loss[L12]-loss[best])/loss[L12]`, q9. `_paired_delta_bootstrap` now also
   returns `boot` (B,NUM_LAYERS) so the CI reuses the SAME paired resamples (ratio-per-replicate
   percentile, NOT raw-CI/const). Best layer = full-sample test-argmin held fixed → CIs DESCRIPTIVE,
   not selection-adjusted. New in run_ood_transfer.aggregate: `relative_gain_summary_q9.csv/json`,
   `relative_gain_heatmap_q9.png`, `relative_gain_id_vs_ood_q9.png`, terminal summary. Raw delta_vs_last
   outputs untouched. RESULT (extended_v2): all 16 cells positive; 14/16 CIs exclude 0 (only M4→elec
   +0.2%, M4→wind +1.3% are nulls); ID mean +6.2%/median +6.3%, OOD mean +12.5%/median +14.9% — the
   early-layer advantage is ~2× LARGER under cross-dataset transfer (descriptive; cells not independent).
   Several OOD cells pick Embed (L0) as best.
8. [x] **Layer-selection-bias FIX (2026-07-31, CPU-only from saved npz — no retrain/extract).** The
   item-7 relative-gain used the TARGET-TEST argmin as "best" → the gain is ≥0 by construction
   (oracle) and its CI ignores the layer search. Corrected to a fair PRIMARY view: the layer is
   chosen on SOURCE VALIDATION only (per-layer min-over-wd val loss from each checkpoint's
   selection.val_loss_by_wd; argmin over depths, np.argmin=earliest-layer tie-break; ZERO target
   contact). One fixed layer per source, reused for its ID cell + all 3 transfer cells:
   **elec L3, uber L3, wind L3, m4 L1** (verified from checkpoints). Same paired series-cluster
   bootstrap (B=5000, seed 0), layer fixed BEFORE resampling; gains + CIs may be NEGATIVE.
   - `run_ood_transfer.py`: new `source_selected_layer`/`_selected_layers_by_source`; the 4 plot/
     table fns gained a `mode` arg (`oracle` default | `source_val`); `aggregate` emits BOTH views.
     PRIMARY → `source_val_{relative_gain_summary_q9.csv/json, relative_gain_heatmap_q9.png,
     relative_gain_id_vs_ood_q9.png, selected_layerwise_matrix_q9.png}` (full schema: sel layer,
     val loss used, sel/L12 test loss, raw gain+CI, rel gain+CI, both excl-zero flags, oracle layer
     +gain, split_mode, n_windows, n_clusters). DIAGNOSTIC → `oracle_*` renames of the old files.
     Old ambiguous PNGs (`relative_gain_heatmap`/`_id_vs_ood`/`transfer_matrix_4x4`) removed;
     legacy `relative_gain_summary_q9.*` KEPT as the oracle alias (run_ood_pretrain_transfer reads it).
   - `run_ood_baselines.py`: adds `probe source_val_selected` (48 rows) + new PRIMARY figure
     `source_val_baseline_comparison.png` (bars: val-sel / oracle-hatched / final / native / seasonal
     / last); legacy `baseline_comparison__q9.png` relabeled "probe best"→"probe oracle" (diagnostic).
   - **CORRECTED RESULT (q9):** positive 13/16 (was 16/16); rel-CI excludes 0 15/16 — but 3 are
     significantly NEGATIVE (M4→elec −6.1%, M4→uber −5.3%, M4→wind −7.0%: M4 val picks L1, great for
     M4 itself but transfers WORSE than L12). ID mean +6.2%/median +6.3% (unchanged — ID best already
     = val layer); OOD mean +7.0%/median +8.0% (oracle was +12.5%/+14.9%). One null Elec→Wind +0.6%.
     ~half the oracle OOD gain was selection bias; the mid-layer OOD advantage survives for elec/uber/
     wind sources. test_quantile_sets 10/10 + test_ood_transfer 11/11 green; no re-score drift warnings.

Verified on login node (no GPU): extended_v2 budget resolves 1500/650 all four; extended_v1
regression = 3000/1500 + committed caches HIT (no re-extract); test_quantile_sets + test_ood_transfer
green. Core pipeline (fit_quantile_probe, extraction, checkpoints, bootstrap) unchanged.

**Remaining = GPU only (user submits):**
- [x] Native-Chronos-2 gate PASSED (2026-07-31, GPU). native MASE: elec 0.82, uber 0.69, m4 0.97,
  wind 1.15 — all beat seasonal-naive (real skill), none trivially easy, none persistence-dominated.
  Roster FINAL. Caveat: wind_farms is the weakest (native only −13% vs seasonal, only native>1) +
  17% missing — a legitimately harder task, not pathological; state plainly in the writeup.
- The 4×4 matrix (extracts extended_v2 features fresh — new windows): `sbatch job_ood_transfer.sh
  --dataset-set extended_v2 --source-dataset <tag>` ×4 (elec/uber/m4_hourly/wind_farms_hourly) →
  `python -m experiments.run_ood_transfer --dataset-set extended_v2 --figure-only` →
  `python -m experiments.run_ood_baselines --dataset-set extended_v2`.
- Then commit (code + results separately; no Co-Authored-By trailer).

## Pretraining-OOD target transfer (4×3) — DESIGN LOCKED 2026-07-31, awaiting go-ahead
Next stage: score the 4 FROZEN extended_v2 source probes (electricity, uber_tlc, m4_hourly,
wind_farms_hourly — all pretraining-ID) on 3 **documented pretraining-OOD** targets. 12 source→
target cells, ALL off-diagonal (no diagonal). NO probe training on the new datasets. Question:
does the intermediate-vs-L12 advantage grow when the target is outside Chronos-2's documented
pretraining distribution?

**Targets + provenance (verified against data/chronos2_seen_manifest.md):**
- **BOOM** (`Datadog/BOOM`, Apache-2.0) — EXPLICITLY in the manifest's documented-unseen reservoir
  (Datadog-internal, arXiv:2505.14766). Cleanest OOD case. Layout: 2807 dirs `ds-<N>-<FREQ>`;
  **378 native-hourly `ds-*-H`** queries (no resampling). Each `ds-N-H` = ONE multivariate metric
  query, split name encodes variate count (`vars_K`); target shape (K, T≈5200). Cluster = parent query.
- **SG Carpark** (TIME `Real-TSF/TIME : SG_Carpark/15T`, CC BY-NC 4.0) — NOT in manifest; TIME
  benchmark (arXiv:2602.12147, Feb-2026 "fresh" zero-shot TSFM set). 15-min ONLY → aggregate to
  hourly. 354 carparks (item_ids RHM/TGM2/…), all 14495 steps, span 2025-01-01→~2025-06-01.
  target = **available-lot COUNT** (integer 0–2662, NOT a ratio). 1.12% missing. Cluster = carpark (354).
- **Coastal T-S** (TIME `Real-TSF/TIME : Coastal_T_S/H`, CC BY-NC 4.0) — NOT in manifest; official
  HOURLY config. 24 stations (IMOS/GBR moorings), variate_names=[TEMP,PSAL,PRES_REL], 0% missing,
  lengths 2733–8784. Cluster = station (24 — smallest cluster count → widest CIs; flag it).
  Seasonality is partly tidal (~12.4h) not diurnal; keep m=24 (note it).
- RIGOR: do NOT claim zero accidental overlap — the two TIME sources (data.gov.sg carpark;
  IMOS ocean moorings) are public feeds. They are absent from Table 6 and are not GIFT-Eval tasks;
  SG Carpark data postdates the Oct-2025 model release. State plainly. [[dataset-screening-rigor]]

**Decisions LOCKED (user, 2026-07-31):**
- Coastal T-S → **TEMP + PSAL only** (drop PRES_REL) = 48 univariate series, cluster by 24 stations.
- BOOM → **1 qualifying variate per query, many queries** (max independent clusters, deterministic
  = first quality-passing variate per hourly query).
- SG Carpark → **hourly = mean of the 4 fifteen-min samples, require ALL 4 finite else NaN** (no fill;
  existing _make_examples drops any window with a non-finite 576-span → leakage-free).

**Reuse audit (KEY — no source retrain):**
- 4 source probes trained+checkpointed at results/extended_v2/ood_transfer/checkpoints/<src>__content__
  C512_H64__q9__seed0/L00..L12.pt (+run_meta.json). ood.load_checkpoints reloads frozen.
- Each checkpoint stores selection.val_loss_by_wd + chosen_wd → **Part B (source-selected layer) is
  reconstructible with ZERO OOD contact.** Source-selected layers (argmin over layers of min-wd val
  loss): **electricity→L3, uber→L3, m4→L1, wind→L3**.
- predict_quantile_probe scores arbitrary feats, never trains, not mutated → one frozen probe → all
  targets. extract_window_features takes an arbitrary (n,C) array (not HF-bound). native_median_forecast/
  compute_mase/_ctx_stats/_mase_denominator/probing.stats reused verbatim once a build_windows-shaped
  test dict exists. ONLY HF-bound pieces = id_data.load_seen_series/build_windows + the square/diagonal
  matrix assumption in the OOD drivers.

**Protocol:** 650 deterministic target windows (build_windows target_test=650 + existing seed-0
_subsample), frozen before any layer look, reused for every source/layer/baseline. C=512/H=64, content
pooling, q9, arcsinh-context norm (leakage-free, dataset-agnostic). A. oracle = target-test argmin
(descriptive). B. source-selected layer (above) frozen, never uses OOD losses. relative_gain_pct =
100·(loss_L12−loss_sel)/loss_L12 with the CI computed as the ratio INSIDE each paired replicate — reuse
_relative_gain_cell / _paired_delta_bootstrap (shared boot matrix). Cluster unit = carpark / station /
parent-query. Deviations logged: 15min→hourly agg; multivariate→univariate; BOOM variate subset+manifest.

**Files to change (minimal, additive):**
1. probing/id_data.py — load_ood_target_series(tag) reading pre-downloaded arrow (pyarrow) from
   $SCRATCH/chronos2/ood_targets/<tag>/; build_ood_windows(tag, target_test=650) reusing _make_examples/
   _seasonal_naive_scale but taking an EXTERNAL cluster id for series_test. Existing sets untouched.
2. NEW experiments/run_ood_pretrain_transfer.py (sibling; leaves run_ood_transfer.py untouched) — non-
   square 4×3: 4 frozen sources × 3 OOD targets → 4×3 loss grid, 4×3 relative-gain heatmap, Part-B
   source-selected table/figure, the CSV (exact columns: source,target,ood_category,target_freq,selected
   _layer,selection_method,q9_loss,L12_loss,raw_gain,relative_gain,relative_ci,mase,cluster_count,window
   _count,preprocessing_notes), 3-group descriptive plot (extended_v2 ID + cross-dataset-ID cells overlaid
   with new OOD cells), baseline comparison (reuse run_ood_baselines helpers).
3. experiments/run_ood_screen.py — teach it the OOD-target loader (screen table + native gate).
4. NEW data/ood_targets_manifest.md (provenance README) + committed BOOM hourly-variate selection manifest.
5. Cache: new tags get IDF_<tag>__ood via _idf_prefix (no collision). probes/extraction/stats/
   run_id_forecasting core UNCHANGED.

**Resources:** storage SG 20MB + Coastal 1.5MB + BOOM hourly ~1.5–2GB (raw, $SCRATCH) + ~80MB/target
caches; RAM <8GB; screen+bootstrap+probe scoring CPU/login-node; ONE short A100 salloc (~15–30 min) for
target feature extraction (650×13-layer content) + native-Chronos-2 gate (nothing trained). USER submits
SLURM. [[submit-slurm-jobs-self]]

**Tests (scoped):** loader/shape+cluster-id per dataset (3); deterministic 650-window freeze; one
source→target end-to-end smoke on cached features. No broad refactor.

**Remaining:** (0) GET GO-AHEAD; (1) pre-download 3 datasets on login node; (2) implement files above;
(3) screen (+native GPU) — do NOT reject on strong persistence/inconvenient layerwise result, only on
broken preprocessing/insufficient data/invalid metric/leakage; (4) GPU extract+native; (5) CPU aggregate
+ figures + CSV + provenance README + BOOM manifest; (6) commit code + results separately (no Co-Authored-By).

### IMPLEMENTATION LOG (2026-07-31) — loader layer DONE + verified
- Locked user decisions applied: Coastal=TEMP+PSAL (station clusters); BOOM=1 variate/query;
  SG=hourly mean-of-4 all-4-required; missing cap 20%; **BOOM window sampling = cluster-balanced
  ROUND-ROBIN** (every query 1 window before any 2nd) — applied to all 3 targets via `_cluster_balanced_order`.
- Code: probing/id_data.py (load_ood_target_series + build_ood_windows + SG aggregator + BOOM lean
  per-variate reader `_boom_read_variate` + `_cluster_balanced_order`); probing/extraction.py
  (_idf_prefix → `IDF_<tag>__ood`); tests/test_ood_targets.py; experiments/select_boom_hourly.py.
- Data staged at $SCRATCH/chronos2/ood_targets/{sg_carpark,coastal_ts,boom_hourly}. BOOM: 378 hourly
  dirs downloaded; committed manifest data/boom_hourly_selection.json = **356 selected / 22 dropped**
  (20% cap), 20 863 candidate windows.
- VERIFIED (login CPU): SG 354 carparks→hourly 3623; Coastal 48 series/24 stations; Coastal
  build_ood_windows 650 windows 24/24 clusters wpc 27/27/28; BOOM lean reader == full-read ref (MV
  ds-1619 98-var + UV ds-1627); test_quantile_sets 10/10 + test_ood_transfer 11/11 green.
- Realized coverage: SG 354 clusters (max 2/cluster), Coastal 24 (27–28), BOOM 356 (max 2). Broad.
- ⚠ NOTE: full build_ood_windows for SG(354)/BOOM(356) is TOO HEAVY for the login-node arbiter
  (killed, sig 16) → the driver's window-build + extraction MUST run on a compute node (salloc/sbatch),
  as the pipeline already assumes. Coastal(24) is light enough to build on login node.
- NEXT: experiments/run_ood_pretrain_transfer.py (4×3 driver) → data/ood_targets_manifest.md → GPU runs.

### IMPLEMENTATION LOG (2026-07-31, cont.) — driver + screen + docs DONE, GPU runs pending
- experiments/run_ood_pretrain_transfer.py (4×3 driver): 4 frozen extended_v2 sources × 3 OOD
  targets, all-OOD (no diagonal). Outputs under results/extended_v2/ood_pretrain_transfer/. Emits:
  results CSV/JSON (both methods, exact columns), 4×3 q9-loss grid, 4×3 relative-gain heatmap,
  Part-B source-selected figure+table, per-target baseline comparison, 3-group descriptive plot.
  Reuses ood.load_checkpoints/_paired_delta_bootstrap/_relative_gain_cell + compute_mase/native.
  A=oracle (target-test argmin, descriptive); B=source-selected (from ckpt val record: elec/uber/
  wind=L3, m4=L1). `--figure-only` = CPU aggregate. Verified: source_selected_layer reads real
  ckpts (no retrain); config→ckpt resolves; figure-only graceful with no cells.
- experiments/run_ood_screen.py: `--ood-targets` path (data_diagnostics_ood + window_supply_ood +
  reuse naive_baselines) → results/extended_v2/ood_pretrain_transfer/screen/. Verified on Coastal
  (CPU): 48 series/24 stations, 650/24, missing 0. **Coastal shows a persistence WARNING (last 1.31
  < seasonal-m24 1.49) — EXPECTED (tidal ~12.4h ≠ m=24), NOT a reject; confirm via native gate.**
- data/ood_targets_manifest.md (provenance README) + job_ood_pretrain_transfer.sh (GPU, mem 32G).
- Regression: test_quantile_sets 10/10, test_ood_transfer 11/11 still green.

### PRIMARY = SOURCE-VAL reframing (2026-07-31, code-only — takes effect on the GPU aggregate)
Same layer-selection-bias fix as the 4×4, applied to run_ood_pretrain_transfer BEFORE the GPU run.
Key point: the OOD targets have NO validation split (never trained on), so the ONLY leakage-free
selection is the SOURCE's own ID val carve → source-selected (elec L3, uber L3, wind L3, m4 L1) is
the PRIMARY; oracle (target-test argmin) is now DIAGNOSTIC only. Edits (all figure/aggregate, no eval
change): `make_relative_gain_heatmap` + `make_three_group_plot` now emit BOTH `source_val_*` (primary)
and `oracle_*` (diagnostic) 4×3 figures (old single ambiguous names retired; nothing on disk yet since
eval pending); `make_loss_grid` title/marker leads with ◆ source-val (hollow ★ = oracle secondary),
reports source-val Δ+% (may be negative); 3-group PRIMARY reads the 4×4 `source_val_relative_gain_
summary_q9.json` for groups 1&2 (oracle variant reads `oracle_*`); `_print_summary` prints source_val
PRIMARY first; baseline_comparison caption marks oracle "light = diagnostic". Docstring A/B swapped
(A=source-selected primary). Verified: syntax OK, `--figure-only` no-ops cleanly with no cells, no
external refs to the retired figure names. Results CSV already carried both methods (`selection_method`
col) — unchanged. Run recipe below unchanged; the aggregate step now writes the source_val_* primaries.

### SG CARPARK 0-WINDOW FIX (2026-07-31, first GPU eval surfaced it)
First `run_ood_pretrain_transfer` GPU run crashed: sg_carpark built **0 windows** → extract_window_
features `need at least one array to concatenate`. ROOT CAUSE (diagnosed on login CPU, no heavy build):
SG has a SYSTEMATIC single missing 15-min sample at ONE clock hour EVERY day (≈150/163 NaN hours at the
same hour-of-day). Under the LOCKED "all-4-of-4-required" hourly rule that hour is NaN daily → longest
fully-finite hourly run = **23 h ≪ C+H=576** → 0 clean windows for ALL 354 carparks. Not persistence/
model/figure-related; the require-fully-finite-576 window rule can't tolerate a periodic gap.
- USER DECISION (AskUserQuestion): **relax to ≥3-of-4** (mean of the present samples, tolerate ≤1
  missing; 2-of-4 or fewer still NaN; NO fill, NO cross-hour interpolation). Recovers 16,992 candidate
  windows (was 0) → round-robin picks 650, ≤2/carpark. Verified: first 25 carparks 25/25 contribute
  (48 windows each).
- Code: `probing/id_data.py` `_aggregate_15min_to_hourly` (+ `MIN_SG_SAMPLES_PER_HOUR=3`) + SG loader
  notes string; `experiments/run_ood_pretrain_transfer.py` `eval_target` FAIL-LOUD guard on n_test==0
  (clear message + points at run_ood_screen) so a 0-window target can never crash cryptically again;
  `tests/test_ood_targets.py` aggregation test → ≥3 contract (1 missing→mean of 3; 2 missing→NaN);
  `data/ood_targets_manifest.md` SG preprocessing + a "why ≥3" deviation paragraph.
- No stale SG cache existed (crash was pre-write). Coastal verified 650; BOOM supply known-good from
  selection (20,863 candidate windows in the manifest) — so all 3 targets should now build on GPU.
- Tests: test_ood_targets (agg+SG/coastal/BOOM loaders+coastal build) + test_quantile_sets 10/10 +
  test_ood_transfer 11/11 all green. Recommend running `run_ood_screen --ood-targets ...` FIRST next
  time (it computes window_supply_ood and would have flagged SG=0 before the GPU salloc).

### GPU RUN RECIPE (USER submits; [[submit-slurm-jobs-self]])
salloc --account=def-irina --gres=gpu:1 --cpus-per-task=2 --mem=32G --time=2:00:00
  module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
  export HF_HOME=$SCRATCH/chronos2/hf_cache HF_HUB_OFFLINE=1 OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets
  python -m experiments.run_ood_screen --dataset-set extended_v2 --ood-targets sg_carpark coastal_ts boom_hourly --native
  python -m experiments.run_ood_pretrain_transfer          # extract+native+score+aggregate (4×3)
(or: sbatch -J ood_pt job_ood_pretrain_transfer.sh). Then commit code + results separately (no Co-Authored-By).

## Future directions (parked)
- Frozen native ResidualBlock head baseline: apply `final_layer_norm` to EACH layer's block
  output (not just L11), slice forecast slots, feed the pretrained head. `final` states are
  already extracted+cached for L11.
- Turn on multivariate / covariates (GroupSelfAttention) — the model's headline feature, unprobed.
- Longer context (up to 8192; currently 512 = ~6% of capacity).
- Representational-similarity view (CKA / effective rank) across layers alongside probe curves.

## Narval run recipe (validated — reuse verbatim)
- **Modules first, then venv** (order matters — arrow's PYTHONPATH wrapper needs python/3.11):
  `module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate`
- **HF cache offline:** `export HF_HOME=$SCRATCH/chronos2/hf_cache HF_HUB_OFFLINE=1`
  (holds amazon/chronos-2 + the 4 dataset configs; compute nodes have no internet).
  Pre-download on the login node, which has internet.
- **GPU run:** `sbatch job_id_forecasting.sh [--quantile-set q9]` (args forwarded), or interactive
  `salloc --account=def-irina --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=1:00:00` then
  `python -m experiments.run_id_forecasting`. ~10–20 min on A100 with warm feature cache.
- **Then bootstrap (login node, CPU, seconds):** `python -m experiments.run_bootstrap`.
- **Login-node CPU checks:** pin `export OMP_NUM_THREADS=2` (shared node throttles otherwise).
- **Model cannot load on a login node** — GPU required for any real extraction.

## Reference facts
- Cluster: Narval. Login `narval3`, branch `egor/phase0-forecasting`, only account `def-irina`.
- Storage: home 50 GB (code/venv) · scratch 20 TB (data/cache, purged 60 d) · project 1 TB
  (persistent, over quota — avoid). No internet on compute nodes.
- Determinism: SEED=0 everywhere; refits reproduce committed numbers exactly on cached features.
  Cache keys carry K and H; feature caches are quantile-independent (shared across sets).
- Verify without the model: `python -m tests.test_quantile_sets` (synthetic, login-node OK).

## Done (changelog)
- **Higher-capacity forecasting probes (capacity controls) — CODE IMPLEMENTED + synthetic tests
  PASS (2026-07-29), real runs pending.** Two nonlinear ResidualBlock heads trained from scratch,
  answering "poor probe forecast = weak representation OR limited linear readout?" Additive; the
  committed linear OOD pilot is untouched.
  - `probing/heads.py`: `ResidualBlock` (structural clone of Chronos-2's native head,
    Linear→ReLU→Dropout→Linear + Linear skip, NO LayerNorm; hidden=native d_ff=3072, dropout=0),
    `build_head` / `head_param_count` / `wd_param_groups` (decay weights only, biases free).
  - `probing/probes.py` (additive; existing probes UNCHANGED): `fit_/predict_content_mlp_head`
    (head on the (n,768) content vector → (n,9,64)) and `fit_/predict_forecast_slot_native_head`
    (ONE shared head over the K=4 native slots, reuses `_apply_shared_head`/`_fit_slot_scaler`).
    Frozen dict stores `head`+`family`+`hidden_dim`+`dropout`+`source_val_loss`. Same fit protocol
    as `fit_quantile_probe` (StandardScaler, seed, AdamW, fixed 300 epochs — NO early stopping — 80/20
    source-val wd grid). Verified param counts: content_mlp 4,575,360 (10.3× linear pooled);
    fslot_native 2,915,616 (26.3× linear shared; ≈ the real native head's 3.65M at q21).
  - `experiments/run_ood_capacity.py`: sibling driver (leaves `run_ood_transfer.py` untouched).
    `--probe-family {content_mlp_head,forecast_slot_native_head}` + `--source-dataset` /
    `--target-datasets` / `--quantile-set q9` / `--figure-only` / `--compare`. One source per job,
    fit once + score every target. Checkpoint id includes probe_family+patch_size, EXCLUDES target.
    Reuses `compute_mase`, `_mae_median_raw`, `_paired_delta_bootstrap`, `target_baselines`.
    Records `oracle_best_layer` (target-test argmin, diagnostic) AND `source_selected_layer`
    (source-val argmin, the fair OOD layer) — source-selected is also derived for the committed
    linear (from its checkpoints) so the 3-way comparison is consistent WITHOUT re-running linear.
    Outputs all under `results/<set>/ood_transfer/capacity/<family>/` (+ `comparison/`): tidy
    results, delta-vs-last, summary, param-count, selected-layers table, per-family 3×3 matrix, and
    (via `--compare`) family-comparison MASE bars + native-gap-closed heatmap.
  - `tests/test_ood_capacity.py`: 10 synthetic no-GPU checks (both families emit (B,9,64); fslot
    shares ONE weight set across slots via permutation-equivariance + param count; fit-once/reuse
    unmutated; no target in fit/selection signatures; checkpoint id has family, no target;
    source-selected uses source-val only; per-window==scalar; median = exact 0.5 row; determinism;
    outputs namespaced away from linear) + a RUN_OOD_CAP_SMOKE=1 real-cache smoke to a tempdir. ALL
    PASS; `test_quantile_sets` + `test_ood_transfer` still green (no regressions).
  - Decisions: fixed 300 epochs (no early stopping) for apples-to-apples with committed linear;
    dropout=0. All content + K4_H64 + native caches for elec/kdd/uber are ON DISK → CPU-runnable,
    but the nonlinear HEAD fit is heavier than linear → prefer a short GPU salloc.
  - PENDING: (1) electricity pilot — both families, source=electricity, all 3 targets, inspect
    train/val curves + layerwise loss + MASE + gap-closed BEFORE launching kdd/uber; (2) then kdd +
    uber sources per family; (3) `--figure-only` per family + `--compare`; (4) commit code + results
    separately (user commits; no push).
- **Cross-dataset OOD transfer pilot — IMPLEMENTED (2026-07-28), real runs pending.** Strict
  3×3 source→target transfer over electricity / kdd_cup_2018 / uber_tlc: train the linear
  quantile probe on ONE source, freeze it, score every target's TEST split (no target
  training/tuning/selection/refit). `is_ood = source != target`; diagonal = in-dataset.
  - `probing/probes.py` (additive; `quantile_probe` UNCHANGED): `fit_quantile_probe` returns the
    frozen per-layer probe (scaler+Linear+wd); `predict_quantile_probe` scores arbitrary feats.
    `predict(fit(tr), te)` reproduces `quantile_probe(tr, te)` on the same device (unit-tested).
  - `experiments/run_ood_transfer.py`: `--source-dataset` (one per job) `--target-datasets`
    `--quantile-set q9` (default, recorded in metadata) `--figure-only`. Per-layer checkpoints
    keyed by SOURCE only (target excluded) → electricity→kdd and electricity→uber reuse ONE
    checkpoint; resumable (fail-loud meta check). Reuses run_id_forecasting `compute_mase` +
    cached native. Writes tidy CSV/JSON (source,target,layer,seed,split,is_ood,pooling,
    quantile_config,…,quantile_loss,mean_pinball_loss,mase,mae_median_raw,n_windows,n_series),
    summary (`delta_vs_last`, `best_layer_shift`), 3×3 loss-by-layer grid (ID diagonal tinted,
    y shared within column, series cluster-bootstrap CI band) + delta heatmap.
  - **Paired Δ-vs-last cluster bootstrap added (2026-07-28).** `_paired_delta_bootstrap` (reuses
    `probing.stats`; ONE shared counts matrix → paired) → per-cell, per-layer
    `ood_transfer_delta_vs_last__q9.{csv,json}` with delta_vs_last / delta_ci_lo/hi /
    delta_above_zero. Summary JSON carries the best layer's Δ CI; heatmap prints best-layer + Δ +
    95% CI with a ★ when the CI excludes 0; matrix panels mark earlier layers whose Δ-vs-last CI>0.
    Post-hoc (runs in `aggregate`/`--figure-only`, no re-fit). Real run (epochs 300, q9): 8/9 cells'
    best-layer Δ CI excludes 0 (only Uber→KDD is a null, best = last layer L12). Smoke test hardened
    to write to a tempdir (never clobbers committed results).
  - `tests/test_ood_transfer.py`: no-GPU synthetic (fit→predict==quantile_probe; frozen reuse
    unmutated; no target args in the fit signature; source→target shape compat; checkpoint id
    excludes target; is_ood def) + a RUN_OOD_SMOKE=1 real-cache end-to-end smoke. All pass.
  - Normalization: per-window arcsinh context-standardize (context-only, leakage-free) is
    identical across datasets; the probe learns source-fit StandardScaler + Linear (source-only)
    → honest transfer, no target statistic fitted. q9 caches for all 3 datasets already on disk,
    so it is CPU/login-node (like the bootstrap) unless a cache is cold.
  - PENDING: (1) three real runs `python -m experiments.run_ood_transfer --source-dataset <tag>`
    (elec/kdd/uber) then `--figure-only`; (2) cross-check the diagonal against the committed q9
    `content` numbers (`id_probing_summary_q9.json`) — bit-exact needs the SAME device the
    committed run used (GPU), CPU agrees to tolerance; (3) commit code + results separately.
- Decoupled the ID-only forecasting figures from the UEA overlay (2026-07-21). New standalone
  `make_ridge_r2_plot` / `make_binned_accuracy_plot` in `run_id_forecasting.py` depend only on
  `id_results` (no UEA load/validate); q21-gated call site next to each other; Embed + L1..L12
  x-axis. Ridge R² and binned accuracy no longer block on regenerating the 12-layer UEA baseline.
  Regenerated both from the committed 13-layer `id_probing_summary.json` (no GPU):
  `results/extended_v1/id_{ridge_r2,binned_accuracy}_by_layer.png`. Smoke tests (UEA-absent +
  stale-12-layer, both plots) in `tests/test_ridge_r2_plot.py`. Legacy make_overlay/tsonly/dropoff
  untouched; they still gate on a 13-layer UEA reference (`and uea_curves`).
- Configurable q1/q9/q21 capacity ablation — implemented, run, analyzed (2026-07-14).
- Series-level cluster bootstrap CIs (loss + MASE) — implemented, run (2026-07-14).
- Horizon-aware K-slot shared forecast probe (ceil+trim, K/H cache key) — run (2026-07-14).
- Shared forecast-token probe (Chronos-aligned readout) — implemented, run (2026-07-14).
- MASE: per-layer probe vs native Chronos-2 head — run (2026-07-13).
- Swapped ID set to 4 long hourly datasets (killed solar/m4 pathologies) (2026-07-13).
- Quantile-loss figure reorg + per-epoch train/val curves + wd grid ON (2026-07-13).
- Chronos-2-native quantile probe (THE metric fix, replaced binned-accuracy) (2026-07-13).
- Forecasting-only cleanup: pruned classification code, kept the extraction engine (2026-07-09).
- First Narval GPU run end-to-end; solved the CC module/venv env (2026-07-09).
- Decided SALVAGE (keep engine, reset around it) over from-scratch (2026-07-09).
