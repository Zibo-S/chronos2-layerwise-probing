# Phase 0 — responses to the three "must-fix" items

This note records how Phase 0 addresses the advisor's three problems with the original
layer-wise probing result. One section per problem. All work is additive; the UEA
classification pipeline, its cache, and its committed numbers are unchanged (`make smoke`
still reproduces them exactly).

---

## 1. A genuine in-distribution baseline

**Problem.** The original probing was entirely on UEA *classification* datasets. From
Chronos-2's perspective (a *forecasting* model) those are foreign/transfer tasks, so there
was no genuine in-distribution (ID) reference to compare against.

**What was added.** A forecasting-shaped probe on data Chronos-2 was actually **pretrained
on** (Table 6 of the Chronos-2 report, arXiv:2510.15821): M4-Hourly, Electricity (hourly),
Solar (hourly), from `autogluon/chronos_datasets`. Each probing example is a length-512
context window; the label is the normalized 64-step future mean (see `probing/id_data.py`).
Two **linear** probes were added to the registry (`probing/probes.py`):
`binned_future_probe` (accuracy over 5 quantile bins — the **primary**, accuracy-scale
readout, directly comparable to the classification curves) and `ridge_regression_probe`
(R², secondary). Probes stay linear so the tunnel-effect diagnostic is preserved.

**Where the evidence lives.**
- `results/id_vs_classification_overlay.png` — ID forecasting curves vs UEA classification,
  each normalized to its own max (primary: binned accuracy; secondary: ridge R²).
- `results/id_vs_classification_overlay_tsonly.png` — same, with non-TS UEA modalities greyed
  (see §2).
- `results/id_vs_classification_dropoff.png` — relative drop from each curve's own peak, which
  makes the post-peak loss legible (the normalize-to-max view compresses it away).
- `results/id_probing_summary.json` — raw per-layer scores (content + REG pooling), plus
  `late_layer_retention = score[L11] / score[peak]` per curve.

**Evidence (facts only, no interpretation).** Binned-accuracy peak layer and late-layer
retention (content pooling; binned chance = 0.20):

| ID dataset (seen) | split | peak layer / acc | L11 retention |
|---|---|---|---|
| m4_hourly | cross-series | L7 / 0.762 | 0.926 |
| monash_electricity_hourly | within-series | L6 / 0.413 | 0.939 |
| solar_1h | within-series | L4 / 0.289 | 0.839 |

## Interpretation — [Zibo to write after inspecting the overlay]

<!-- Intentionally left blank. Do not fill in automatically. -->

### Known limitation — the context-normalized label on strongly periodic series
`solar_1h` produces strongly **negative ridge R²** (the regression does worse than predicting
the mean). This is a **label pathology, not a representation finding**: solar power is strongly
diurnal (zeros at night), so the context-normalized future mean depends heavily on the window's
**phase** in the day/night cycle, and σ_context is deflated on night-heavy windows — making the
normalized target erratic. `solar_1h` is therefore **demoted** (thin / low-alpha, "label
pathology") in all overlays; its results are retained, not deleted. A seasonally-adjusted or
phase-aware label would be the fix; that is **future work** and is deliberately **not**
implemented now (changing the label mid-Phase-0 would break comparability).

### Note on M4-Hourly's split mode
M4-Hourly series (748–1008 steps) are shorter than `2·(C+H) = 1152`, so a within-series
train/test span pair is impossible at C=512/H=64. Rather than change C, H, or the label, M4-Hourly
uses a **cross-series** split (disjoint train/test series → still leakage-free), while Electricity
and Solar use the within-series temporal split. Consequently the M4 curve is comparable to the
others in **shape** (peak-layer location, late-layer behavior) but **not in absolute level**:
cross-series and within-series splits measure different kinds of generalization. The split mode
per dataset is recorded in `results/id_probing_summary.json`.

---

## 2. Domain restriction — TS domains only

**Problem.** Not all UEA "time series classification" datasets are genuine time-series domains;
several are other modalities re-encoded as sequences, so probing accuracy on them is a cross-modal
transfer result, not an in-domain one.

**What was done.** `data/uea_domain_audit.md` classifies each of the 6 non-saturated UEA datasets
by modality, from the official UEA archive descriptions (arXiv:1811.00075):

- **Genuine sensor time-series (retained, full color):** UWaveGestureLibrary (3-axis accelerometer).
- **Other modality re-encoded as sequence (excluded, greyed):** EthanolConcentration (spectroscopy —
  wavelength axis, not time), SelfRegulationSCP1 & SelfRegulationSCP2 (EEG bio-signals), Handwriting
  (handwriting / pen-trajectory motion), LSST (astronomical light curves).

Phase 0 conclusions are drawn **only** from the genuine-TS subset; the excluded curves are shown
greyed in `results/id_vs_classification_overlay_tsonly.png` for continuity only. A proper
within-time-series transfer leg (same probe task on documented-unseen TS data) is scheduled as
**Track B (feature-level shift)** and is not implemented here. See the audit doc for the (borderline)
UWave-vs-Handwriting call and full reasoning.

---

## 3. Shift-type reclassification — the Gaussian-noise result was the wrong shift type

**Problem.** The earlier "OOD" experiments perturbed the *inputs* (Gaussian noise, drift, time-warp)
and found **no** amplification of the middle-vs-last gap. That was read as a failed replication of the
tunnel effect's OOD prediction.

**Reframe.** Under the Surgical Fine-Tuning taxonomy (Lee et al., arXiv:2210.11466), shifts occur at
three levels — **input-level**, **feature-level**, **output-level** — each affecting a different part
of the network. The earlier Gaussian-noise / drift / time-warp perturbations are **input-level**
shifts: they corrupt the raw input signal. An input-level shift is predicted to disrupt the
**earliest** layers (closest to the input), **not** to amplify a middle-vs-last gap. So the earlier
null is **consistent with the taxonomy** — an input-level shift simply is not the condition under
which a late-layer/tunnel amplification is predicted — rather than a failed replication of the tunnel
effect.

This tagging is now explicit in code: `probing/shifts.py` registers `gauss` / `timewarp` / `drift`
under `input_level`, and provides documented stubs for `feature_level` (cross-domain transfer to a
documented-unseen TS domain — predicted to affect **middle** layers) and `output_level` (forecast-
horizon change — predicted to affect the **latest** layers). Those feature- and output-level shifts,
which are the conditions where a late-layer effect would actually be predicted, are **Track B** and
are not run in Phase 0.
