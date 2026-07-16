# UEA modality audit — which classification datasets are genuine time-series domains?

Closes the advisor item *"restrict to TS domains, not other modalities."* A time-series
foundation model like Chronos-2 is trained on **forecasting-style temporal signals**
(energy, weather, traffic, sales, …). Several UEA "time series classification" datasets are
really **other modalities re-encoded as sequences** — a spectrum (wavelength axis), a brain
signal, a handwriting trajectory, an astronomical light curve. Probing accuracy on those is a
cross-modal transfer result, not an in-domain time-series result, so Phase 0's tunnel-effect
conclusions should be read only on the genuine-TS subset.

Classifications below are based on the official UEA multivariate archive dataset descriptions
(Bagnall et al., *The UEA multivariate time series classification archive*, arXiv:1811.00075;
per-dataset descriptions at timeseriesclassification.com).

## The 6 non-saturated UEA datasets used in Phase 0

| Dataset | What each "series" actually is (UEA description) | Independent axis | Modality bucket | Genuine TS? |
|---|---|---|---|---|
| **UWaveGestureLibrary** | X/Y/Z **accelerometer** of 8 hand gestures (Wii-remote) — a motion **dynamics** signal evolving over time | time (temporal dynamics) | motion / inertial sensor | **yes (kept)** |
| EthanolConcentration | raw **spectra** of water-and-ethanol solutions in 44 whisky bottles; classify alcohol concentration | **wavelength** | spectroscopy | no |
| SelfRegulationSCP1 | **EEG** slow cortical potentials, healthy subject | time | bio-signal (EEG) | no |
| SelfRegulationSCP2 | **EEG** slow cortical potentials, ALS patient | time | bio-signal (EEG) | no |
| Handwriting | smart-watch accelerometer while writing the 26 letters — a **2-D spatial trajectory** (letter shape) unrolled into a sequence | spatial path over time | handwriting / pen-trajectory | no |
| LSST | simulated astronomical **light curves** (photometric flux, 6 passbands) | time | astronomical photometry | no |

**Retained as genuine time-series (full color in the TS-restricted overlay): UWaveGestureLibrary.**
**Excluded as other-modality-re-encoded-as-sequence (greyed): EthanolConcentration, SelfRegulationSCP1,
SelfRegulationSCP2, Handwriting, LSST.**

### Classification criterion: the generative process, not the sensor

The bucket is decided by the **generative process** the signal encodes — *"is this a temporal-
dynamics process (a quantity evolving over time, the kind a forecaster models)?"* — **not** by
what sensor recorded it and **not** merely by *"is the x-axis time?"*. Two datasets recorded with
the same sensor can fall in different buckets if one encodes temporal dynamics and the other
encodes a static form sampled over time.

- **EthanolConcentration is the hardest exclusion**: its independent axis is *wavelength*, so it
  is not a time series at all — it is a spectrum. Clear exclusion.
- **EEG (SCP1/SCP2)** and **LSST light curves** are sampled over time, but they are specialized
  bio-signal and astronomical-photometry modalities, not forecasting-domain signals; per the
  advisor's modality list they are excluded (EEG → bio-signal bucket).
- **UWaveGestureLibrary (kept) vs Handwriting (excluded)** — both happen to use an accelerometer,
  so the *sensor* is the same, but the **generative process differs**, which is what the criterion
  keys on:
  - **UWaveGestureLibrary** is a **temporal-dynamics** signal: a gesture is a motion *evolving over
    time*, and the discriminative information is in how the acceleration changes from step to step —
    the same character (autocorrelated dynamics with a forecastable next step) that sensor streams in
    forecasting corpora have. → genuine TS, **retained**.
  - **Handwriting** is a **2-D spatial trajectory** (the pen/hand traces the *shape* of a letter)
    that has merely been *unrolled into a sequence*. The task is **shape recognition**: the
    discriminative content is the spatial form of the character, and the temporal ordering is
    incidental to tracing that shape rather than a dynamics-to-forecast. → other modality,
    **excluded**. This matches the advisor's comment that **handwriting does not fall in the general
    forecasting domain**.

  So the split is principled (dynamics vs. spatial shape), not an arbitrary preference between two
  accelerometer datasets. Caveat retained for honesty: even UWave is gesture motion-capture rather
  than an energy/weather/traffic forecasting signal, so the genuine-TS UEA subset is small and only
  loosely in-domain for a forecasting TSFM — reinforcing that the true ID reference is the
  forecasting probe, not any UEA subset.

## How this constrains the Phase 0 conclusions

**Phase 0 conclusions are drawn ONLY from the genuine-TS subset.** The excluded-modality
classification curves are still plotted — greyed and labeled "excluded modality" in
`results/id_vs_classification_overlay_tsonly.png` — for visual continuity with the original
overlay (`results/id_vs_classification_overlay.png`, left unchanged), but they are not used to
support any claim about the tunnel effect in Chronos-2. The genuine in-distribution comparison
is the **forecasting** probe on Chronos-2-seen data (M4-Hourly / Electricity / Solar); the UEA
classification datasets are, from the model's perspective, out-of-modality transfer tasks of
varying distance.

A proper **within-time-series transfer** leg — the same forecasting probe task on
*documented-unseen* time-series data (fev-bench / Chronos Benchmark II / BOOM; see
`chronos2_seen_manifest.md`) — is scheduled as **Track B, the feature-level shift**
(`probing/shifts.py`). It is intentionally **not** implemented in Phase 0.
