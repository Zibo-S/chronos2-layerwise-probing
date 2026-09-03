# VERIFY_AXIS_FIGS_REPORT — axis verification + double-blind figure regeneration

Branch `zibo/repr-metrics` @ `7a940d9`. `git diff --stat HEAD` empty throughout; **no
commits made**; no `main.tex` edited (none exists in this repo — see Flags); **no
experiment recomputed** — Task 2 is replot-only from cached JSON/CSV. Older reports
ignored per preamble; everything below was re-read from files, not from memory.

**Preamble discrepancy (reported up front):** this repo has **no `figs/` directory** and
7 of the 8 named basenames do not exist anywhere in it (only `fig_cka_matrix.png` exists,
under `results/repr_metrics/cka/<run>/`). The paper assembly (figs/ + main.tex) evidently
lives outside this repo (Overleaf). I therefore mapped each commanded basename to its
counterpart figure in `results/` (mapping in the T2 table) and wrote the regenerated
versions to a new `paper_figs/` with the commanded basenames.

---

## Section T1 — axis verification (read-only; every number quoted from a file)

### 1a. ID binned probe accuracy per layer

**File:** `results/phase0_trio/id_probing_summary.json`
**Key:** `id_datasets["<tag>"].poolings.content.binned_accuracy`

**Indexing convention, with evidence:** a bare 12-element array — position `k` = probe on
encoder block output `k` (0-indexed), **no Embed entry**. Evidence: (i) length 12 with 12
encoder blocks and no 13th/14th tap; (ii) the companion file
`results/repr_metrics/masarczyk_criterion/<ds>/criterion.json` records
`provenance.layer_mapping = "probe layer_k -> repr L(k+1); Embed has no probe point;
L12_postln is rank-only"` and its `accuracy_by_layer["L(k+1)"]` equals `array[k]`
elementwise (verified `True` for all 12 entries, both datasets). So **probe idx k → paper
axis L(k+1)**.

**electricity** (`monash_electricity_hourly`), full array:
`[0.378, 0.39867, 0.37867, 0.37733, 0.39133, 0.41267, 0.41333, 0.40867, 0.36467, 0.37867, 0.39933, 0.388]`

| rank | probe idx (file) | paper axis | value |
|---|---|---|---|
| 1 | 6 | **L7** | 0.41333333333333333 |
| 2 | 5 | L6 | 0.4126666666666667 |
| 3 | 7 | L8 | 0.4086666666666667 |

**ARGMAX: probe idx 6 → paper L7.**

**m4** (`m4_hourly`), full array:
`[0.65552, 0.69913, 0.7064, 0.73256, 0.74855, 0.7311, 0.73547, 0.76163, 0.74855, 0.73983, 0.72093, 0.70494]`

| rank | probe idx (file) | paper axis | value |
|---|---|---|---|
| 1 | 7 | **L8** | 0.7616279069767442 |
| 2 | 8 | L9 | 0.748546511627907 |
| 2= | 4 | L5 | 0.748546511627907 (exact tie) |

**ARGMAX: probe idx 7 → paper L8.**

### 1b. Prompt-entropy (PE) peak per dataset

**Files:** `results/repr_metrics/electricity/metrics.json`,
`results/repr_metrics/m4/metrics.json`
**Key:** `per_layer[*].prompt_entropy_norm_mean`, layer identified by the string field
`per_layer[*].layer`.

**Indexing convention, with evidence:** entries carry explicit string labels
`["Embed","L1","L2","L3",…,"L12","L12_postln"]` — this **is already the 1-indexed paper
axis**; no translation needed. Peak taken over the depth axis Embed..L12 (the `L12_postln`
tap is the same block post-norm, not a deeper layer).

| dataset | top-3 (label = paper axis) | PEAK |
|---|---|---|
| electricity | L6 = 0.6745393070377361, L7 = 0.6738105509297244, L8 = 0.6543987247442974 | **L6** |
| m4 | L6 = 0.5575037654214999, L7 = 0.5509441369249121, L8 = 0.5386155772248914 | **L6** |

### 1c. VERDICT — PE peak vs ID probe argmax on the paper axis

| dataset | PE peak | ID probe argmax | equal? |
|---|---|---|---|
| electricity | L6 (0.67454) | L7 (0.41333) | **NO** (adjacent: L6 vs L7; probe L6 = 0.41267 is second by 0.00067) |
| m4 | L6 (0.55750) | L8 (0.76163) | **NO** (two layers apart) |

So any main.tex sentence claiming the PE peak *coincides* with the ID probe argmax is
wrong for both datasets on these files; "PE peaks in the same mid-network band (L6–L8) as
probe accuracy" would be supported.

### 1d. `late_drop_band` definition

**Producing code:** `experiments/run_perdataset.py:49` — `MIDDLE_BAND = list(range(3, 9))
# layers 3..8 inclusive` with `LAST_LAYER = 11`; lines 84–85 average the per-layer correct
indicators over exactly those indices; line 130 computes the paired diff vs
`correct_id[LAST_LAYER]`.
**Confirmed by the artifact itself:** `results/uea/perdataset_summary.json` →
`config.middle_band = [3, 4, 5, 6, 7, 8]`, `config.last_layer = 11` (recorded at run
time).

**File indexing:** probe layer indices 3..8 (0-indexed block outputs) averaged, minus
probe layer 11.
**Paper translation (k → L(k+1)):** band = **L4..L9**, last = **L12**, i.e. the quantity
is **mean(L4:L9) − L12** — exactly the expected definition. The forest x-label was set
accordingly.

⚠️ **Config-drift flag:** `probing/config.py` on disk *now* reads
`MIDDLE_BAND = list(range(4, 10))`, `NUM_LAYERS = 13`, `LAST_LAYER = 12` (a 1-indexed
re-indexing edit made after these results were produced — not by me and not committed).
It did **not** produce the committed forest data (the artifact's config block proves
[3..8]/11 did), but any future rerun importing `probing.config` would silently use a
shifted band. Worth reconciling before anything is recomputed.

**Which 6 transfer sets covered** (`datasets[*].id_late_drop_band` where
`saturated == false`): UWaveGestureLibrary, EthanolConcentration, SelfRegulationSCP1,
Handwriting, LSST, SelfRegulationSCP2.

| dataset | point | 95% CI | excludes 0 |
|---|---|---|---|
| UWaveGestureLibrary | +0.0854 | [+0.0510, +0.1224] | True |
| EthanolConcentration | +0.0703 | [+0.0133, +0.1223] | True |
| SelfRegulationSCP1 | +0.0631 | [+0.0279, +0.0984] | True |
| Handwriting | +0.0504 | [+0.0263, +0.0747] | True |
| LSST | +0.0151 | [+0.0005, +0.0294] | True |
| SelfRegulationSCP2 | −0.0176 | [−0.0907, +0.0584] | **False** |

**5/6 exclude zero; the null is SelfRegulationSCP2.**

---

## Section T2 — figure regeneration (`paper_figs/`, 300 dpi, replot-only)

New module: `probing/paper_figs.py` (`python -m probing.paper_figs`) — reads **only** the
cache files below; machine-readable source list in `paper_figs/_cache_sources.json`.

| commanded basename | old figure (this repo) | cache source(s) plotted | changes applied | gate |
|---|---|---|---|---|
| fig_masarczyk_overlay_elec.png | results/repr_metrics/masarczyk_criterion/electricity/fig_masarczyk_overlay.png | masarczyk_criterion/electricity/criterion.json + repr_metrics/electricity/metrics.json | title dropped; tick `L12_postln`→"L12 (post-LN)"; fonts ~2×; legend inside (upper headroom, covers no data); whitespace tightened | PASS |
| fig_masarczyk_overlay_m4.png | …/masarczyk_criterion/m4/fig_masarczyk_overlay.png | masarczyk_criterion/m4/criterion.json + repr_metrics/m4/metrics.json | same as above | PASS |
| fig_distance_vs_gain_combined.png | results/distance/ladder/fig_distance_vs_gain.png (+ v2/v3 single panels) | ladder/join_extended_v3_rolling.csv + join_extended_v2.csv | **panel order swapped: rolling-origin now LEFT**; headers "Primary split (rolling-origin), ρ = −0.41" / "Alternative split, ρ = −0.10"; corpus-seen = circles, corpus-unseen = triangles; display names Electricity/M4/Uber TLC/Wind Farms; legend moved to empty upper-right (it previously hid the cell at (3.41, −40.3)) | PASS |
| forest.png | results/uea/fig_dataset_forest.png | uea/perdataset_summary.json | x-label "mean(L4:L9) − L12 late-layer deficit (95% paired CI)"; no title; fonts ~2× | PASS |
| dropoff.png | results/phase0_trio/id_vs_classification_dropoff.png | phase0_trio/id_probing_summary.json | x-ticks **L1–L12** (paper axis; the axis fix); legend exactly as commanded: "M4 (cross-series)", "M4 (regression)", "Electricity", "Electricity (regression)", "Solar 1H (label pathology)", "UEA classification (n=6)"; legend lower-right (empty region); no title | PASS + ⚠️ see flag 2 |
| fig_common_pooling.png | (candidate: results/phase0_trio/quantile_loss/pooling_comparison/content_vs_reg.png — PNG only) | **NONE usable** | **NOT REGENERATED — STOP-flag** (flag 3) | STOP |
| fig_cka_matrix.png | results/repr_metrics/cka/electricity/fig_cka_matrix.png | cka/electricity/cka.json | both panel titles removed; ticks enlarged/legible; `L12_postln`→"L12 (post-LN)"; white seam line kept on 14-tap panel | PASS |
| fig_native_head_elec.png | results/repr_metrics/native_head/electricity/fig_native_head_by_layer.png | native_head/electricity/native_head.json (+ id_probing_summary.json for the pre-existing faint probe overlay) | title dropped; y-label "q9 quantile loss (arcsinh scale)"; baseline legend "native forecast baseline" (numeric removed); tick "L12 (post-LN)"; legend in added headroom | PASS |

All figures: 300 dpi PNG, no titles/suptitles (the two commanded panel *headers* on the
combined distance figure excepted, as specified), rcParams fonts ≈2× default (labels 18,
ticks 15), legends inside axes verified visually to cover no data points.

### Gates

1. **Cache source per figure** — table above + `paper_figs/_cache_sources.json`.
2. **Banned strings** — AST scan of every string literal reaching
   `set_title/set_xlabel/set_ylabel/set_*ticklabels/legend/text/annotate` plus all
   `label=` kwargs, DISP names, tick lists and the two headers (49 distinct rendered
   strings): **zero hits** for personal names, run names (`extended_v2`,
   `extended_v3_rolling`, round ids), `STRUCTURED`, bare `REG`, `late_drop_band`,
   `acc_layer`, `L12_postln`, `source_selected`. `grep -iE "amartya|egor|zibo|shang"`
   over the module: no matches. ("rolling-origin" appears only inside the commanded
   header text.)
3. **Value identity** — the new module reads the *same files and keys* the old figures
   were rendered from, plus explicit checks: Spearman ρ recomputed from the plotted CSVs
   = −0.4106 / −0.1026, matching the commanded headers −0.41 / −0.10 (28 cells each);
   masarczyk accuracy arrays == the probe JSON elementwise (12/12, both datasets);
   native q9 baseline 0.10108262300491333 == the per-layer L12_postln value; CKA 13-tap
   matrix == leading 13×13 submatrix of the 14-tap matrix; forest = 6 non-saturated
   rows with 5 excluding zero; dropoff includes all 6 UEA reference curves. (Pixel-level
   comparison to the old PNGs is impossible by design — cosmetics changed — so identity
   is established at the data layer.)
4. **Repo scope** — `git diff --stat HEAD` empty; HEAD still `7a940d9`; new files only:
   `probing/paper_figs.py`, `paper_figs/` (7 PNG + `_cache_sources.json`), this report.

---

## Flags for you to decide

1. **No `figs/` or `main.tex` here.** The regenerated files use the commanded basenames
   under `paper_figs/`; copying them into the Overleaf figs/ is on you (I could not
   verify what the current Overleaf versions plot).
2. **"(regression)" legend label is almost certainly a mislabel.** The two dashed curves
   are the **REG register-token pooling** variants (`poolings.reg.binned_accuracy` —
   same classification task, different pooling), not a regression objective. I rendered
   the commanded strings verbatim, but a reviewer reading "M4 (regression)" will assume
   a regression probe. Suggested honest label: "(register token)" or "(alt. pooling)".
   Say the word and I'll re-render.
3. **fig_common_pooling.png could not be regenerated from local cache.** The quantile-
   loss output dirs for the paper's dataset sets are empty locally
   (`results/extended_v2/quantile_loss/**` — empty; `results/phase0_trio/quantile_loss/`
   holds only a rendered PNG, `pooling_comparison/content_vs_reg.png`, with no backing
   JSON). The only JSON-backed pooling comparison is
   `results/extended_v1/quantile_loss/quantile_loss_results.json` — a superseded dataset
   set. Regenerating this figure would require either the missing JSON from the
   cluster/collaborator or a recompute, which the brief forbids.
4. **Provenance JSON (not figure text) contains a collaborator name:**
   `results/repr_metrics/cka/*/cka.json` field `provenance.amartya_axis_note`. Rendered
   figures are clean; if raw result JSONs ship as supplementary material, that field
   needs renaming.
5. **`probing/config.py` drift** (T1d): on-disk band/last-layer constants no longer
   match the committed artifacts' recorded config. Harmless for this replot-only task;
   dangerous for any future rerun.

---

## ADDENDUM (2026-09-01) — flags 2 and 5 resolved on instruction

**Flag 2 (dropoff relabel).** Confirmed from source what "REG" denotes: the probe's
"reg" pooling reads the hidden state of the model's **[REG] register token** — a special
token appended after the context patches — instead of mean-pooling the content patches.
Same classification task, different pooling; nothing to do with a regression objective.
Evidence: `probing/extraction.py:161` ("pool_reg : the REG token at index
num_context_patches"), `probing/extraction.py:418` (`pool_reg` returns
`hs[:, reg_idx, :]` with `reg_idx = ncp`, layout `[context(ncp) | REG | K forecast
slots]`), and the installed model source `chronos/chronos2/model.py:581-583` ("append
[REG] special token embedding", `use_reg_token`/`reg_token_id`). Legend entries
re-rendered as **"M4 (register-token pooling)"** and **"Electricity (register-token
pooling)"**; only `paper_figs/dropoff.png` changed (mtime diff over `paper_figs/*.png`
confirms the other six PNGs untouched).

**Flag 3 (fig_common_pooling) — resolved via authorized one-off recompute exception
(2026-09-01).** The backing JSON remained unavailable, so on instruction the comparison
was recomputed locally from the CACHED hidden states only
(`features_cache/IDF_monash_electricity_hourly__{train,test}__clean__{content,reg}.npz`;
existence checked before any computation, no model forwards anywhere). Driver:
`probing/common_pooling_recompute.py`, calling the exact pipeline functions of
`experiments/run_id_forecasting.py` (`build_windows`, `extract_window_features`,
`binned_future_probe`, `ridge_regression_probe`, `quantile_probe` with q21 /
epochs=300 / wd_grid, frozen `config.py`). **GATE: recomputed binned accuracy and ridge
R² match `id_probing_summary.json` at max|Δ| = 0.000e+00 (bit-identical) for BOTH
poolings** — checked before any quantile-probe training. Output JSON:
`results/phase0_trio/quantile_loss/pooling_comparison/common_pooling_recompute.json`
(four metrics per layer per pooling; native-MASE baseline omitted — its forecast cache
is absent and re-running the model was forbidden; the committed
`id_probing_summary.json` untouched). Figure rendered to
`paper_figs/fig_common_pooling.png` in the paper's Fig-6 form: FOUR-METRIC comparison on
the common content-pooling axis, one line per metric (Binned accuracy, Ridge R²,
Quantile loss, MASE), each normalized within-objective to [0,1] (1 = best layer,
0 = worst), ★ at each metric's best layer; no title, x "layer" with ticks L1–L12 (no
Embed — the pipeline probes only the 12 block outputs), doubled fonts, legend below the
axes. Best layers on the paper axis (content pooling): Binned accuracy **L7** (0.4133),
Ridge R² **L5** (0.4113), Quantile loss **L3** (4.7889), MASE **L3** (1.4231) — the L3
stars for Quantile loss and MASE coincide (one ★ overlays the other). Worst layers:
Binned accuracy **L9** (0.3647), Quantile loss **L9** (5.3322), MASE **L9** (1.6204),
but Ridge R² bottoms at **L2** (0.2387) — L9 is only a local dip for it.

**Flag 5 (config drift).** Correction to the original flag: the drift was not an
uncommitted disk edit — the 13-layer scheme is **committed**, introduced by `673e9f9`
("import from ood-forecasting-pilot (Egor)"; original values from `54c5389` shown by
`git log -L`). Only `config.py` was flipped; `extraction.py`, `run_perdataset.py`,
`run_bootstrap.py` and `run_id_forecasting.py` still assume the 0-indexed 12-layer
convention (`run_perdataset.py:50` even comments `# 11`), so the committed combination
would have crashed or mislabeled on any rerun. Restored `NUM_LAYERS = 12`,
`MIDDLE_BAND = list(range(3, 9))`, `LAST_LAYER = 11` plus the original docstring
paragraph, and added a FROZEN-FOR-FMTS comment block listing the result files these
values produced and forbidding changes before the submission freeze. Left as an
**uncommitted** working-tree modification per instruction (the only tracked-file diff);
suite re-run: **42 passed**.
