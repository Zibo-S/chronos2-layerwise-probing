# Chronos-2 pretraining provenance — ID / OOD dataset manifest

This manifest records which datasets are **in-distribution (seen during Chronos-2
pretraining)** vs. **documented-unseen (reserved for a later OOD track)**, so that the
Phase 0 in-distribution condition uses only genuinely-seen data.

## In-distribution (SEEN) — used for the Phase 0 ID forecasting probe

Per **Table 6 of the Chronos-2 technical report (arXiv:2510.15821)**, the following
datasets are part of Chronos-2's real (non-synthetic) univariate pretraining corpus. We
use the **hourly** variants, available on HuggingFace at `autogluon/chronos_datasets`:

| Dataset (this study) | HF config | Target column | Role |
|---|---|---|---|
| M4 Hourly | `m4_hourly` | `target` | ID forecasting |
| Electricity (hourly) | `monash_electricity_hourly` | `target` | ID forecasting |
| Solar (hourly) | `solar_1h` | `power_mw` | ID forecasting |

Probing examples are length-512 context windows; the label is the normalized 64-step
future mean (see `probing/id_data.py`). Feature cache entries are prefixed `IDF_` and are
disjoint from the UEA classification cache.

### Note on the within-series split and M4 Hourly

M4-Hourly series are 748–1008 steps long — shorter than `2*(C+H) = 1152` — so **no single
series can hold a non-overlapping (train-span, test-span) pair** at C=512/H=64. Rather than
change C, H, or the label definition, M4-Hourly uses a **cross-series** split (disjoint
train/test series → still leakage-free), while Electricity and Solar use the primary
**within-series temporal** split. The split mode actually used is recorded per dataset in
`results/id_probing_summary.json`.

## GIFT-Eval overlap caveat

The Chronos-2 report states that the pretraining corpus **excludes the *test* portions of
all GIFT-Eval tasks but partially overlaps the *training* portions of some GIFT-Eval
datasets.** Therefore **GIFT-Eval domains are NOT clean OOD** for Chronos-2 and must not be
used as the unseen condition.

## Documented-unseen (reserved for the later OOD track — do NOT use here)

- **fev-bench** (100 tasks) — documented unseen during Chronos-2 training.
- **Chronos Benchmark II** (27 tasks) — documented unseen.
- **BOOM** (Benchmark of Observability Metrics) — Datadog-internal production metrics,
  documented unseen.

These are the OOD reservoir for **Track B (feature-level shift)**; they are intentionally
NOT touched in Phase 0.

---

<!-- CITATION BLOCK — pasted verbatim, do not paraphrase -->

## Sources

- **Chronos-2 training corpus (seen list):** Ansari et al., "Chronos-2: From
  Univariate to Universal Forecasting," arXiv:2510.15821. Table 6 (Appendix) lists
  the real univariate pretraining datasets, incl. M4 (Hourly/Daily/Weekly/Monthly),
  Electricity, Solar, Wind Farms, Weatherbench, Wiki, Taxi, USHCN, KDD Cup 2018,
  London Smart Meters, cloud-ops traces (Alibaba/Azure/Borg), a.o. Synthetic data
  via TSI / TCM / KernelSynth; multivariate structure via "multivariatizers."
- **GIFT-Eval overlap caveat:** the same report states the pretraining corpus
  excludes the *test* portions of all GIFT-Eval tasks but partially overlaps the
  *training* portions of some GIFT-Eval datasets → GIFT-Eval domains are NOT clean
  OOD for Chronos-2. (arXiv:2510.15821)
- **Documented-unseen (reserved for the OOD track):**
  - fev-bench (100 tasks): none of its datasets/tasks were seen during Chronos-2
    training. (arXiv:2510.15821; fev-bench: arXiv:2509.26468)
  - Chronos Benchmark II (27 tasks): none included in Chronos-2's training corpus.
    (arXiv:2510.15821; benchmark defined in Ansari et al. 2024, the Chronos paper)
  - BOOM (Benchmark of Observability Metrics): 350M observations / 2,807 production
    series sourced exclusively from Datadog-internal metrics, from an environment
    separated from public corpora — not part of the Chronos/GIFT-Eval pretraining
    sources. (BOOM: arXiv:2505.14766; Datadog Toto/BOOM release notes)
