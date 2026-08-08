# Documented pretraining-OOD targets — provenance & preprocessing

This file documents the three **documented pretraining-OOD** targets used by
`experiments/run_ood_pretrain_transfer.py` (4 frozen `extended_v2` ID source probes → 3 OOD
targets, evaluation-only). "Documented pretraining-OOD" means **absent from the exact Chronos-2
pretraining manifest used in this repo** (`data/chronos2_seen_manifest.md`, Table 6 of
arXiv:2510.15821). It does **not** claim that accidental overlap with the underlying public
sources is impossible — see *Known limitations*.

_Access date for all three: 2026-07-31 (Narval login node)._ Raw shards are staged off-repo at
`$SCRATCH/chronos2/ood_targets/` (override with `OOD_TARGET_ROOT`); only the small BOOM selection
manifest (`data/boom_hourly_selection.json`) is committed.

## Why each is pretraining-OOD (checked against the repo manifest)

The 4 **sources** (electricity, uber_tlc/"taxi", m4_hourly, wind_farms) are all named in the
manifest's *seen* list → pretraining-ID, the correct baseline distribution.

| Target | Manifest status | Basis |
|---|---|---|
| **BOOM** | **Explicitly listed** in the manifest's documented-unseen reservoir | Datadog-internal production observability metrics, isolated from public corpora (arXiv:2505.14766). Cleanest OOD case. |
| **SG Carpark** | Absent | From the TIME benchmark (arXiv:2602.12147, Feb 2026), "50 *fresh* datasets … for zero-shot TSFM evaluation"; the carpark window is Jan–Jun 2025, postdating the Oct-2025 Chronos-2 release. Not a GIFT-Eval task. |
| **Coastal T-S** | Absent | Same TIME "fresh" benchmark; Australian IMOS/Great-Barrier-Reef mooring data. Distinct from the seen "WeatherBench" / "KDD air-quality" entries. |

## Sources & licenses

- **TIME benchmark** (SG Carpark, Coastal T-S): data HF `Real-TSF/TIME` — **license CC BY-NC 4.0
  (non-commercial research)**; code `github.com/zqiao11/TIME` (Apache-2.0); paper arXiv:2602.12147.
  We load the committed HuggingFace-`datasets` arrow shards directly (pyarrow); the `timebench`
  package is not required.
- **BOOM**: HF `Datadog/BOOM` — **Apache-2.0**; paper arXiv:2505.14766. 2,807 series; we use only
  the **378 native-hourly** metric queries (dirs `ds-<N>-H`).

## Per-target schema, preprocessing, and screen facts

Common frame: frozen amazon/chronos-2, content-mean pooling, **C=512 / H=64**, q9, arcsinh
context-standardized target space, in-context seasonal-naive MASE (m=24). Evaluation-only: **650
deterministic windows** per target (`id_data.build_ood_windows`, seed 0), frozen before any layer
is inspected; **query-balanced round-robin** sampling (every cluster gets its 1st window before any
2nd) so no long series dominates. Incomplete (non-finite) 576-step spans are **dropped, never
filled**.

### SG Carpark  (`Real-TSF/TIME : SG_Carpark/15T`)
- Schema: GluonTS-style `item_id / start / freq / target`, **univariate** per carpark.
- 354 carparks; all 14,495 steps at 15-min (2025-01-01 → ~2025-06-01); ~1.12% missing (in-place NaN
  on a regular grid); 0 near-constant series.
- **Target = available-lot COUNT** (integer, range 0–2662) — verified from values; NOT occupancy,
  NOT a [0,1] ratio.
- **Preprocessing (deviation from the native-hourly protocol): 15min → hourly** = mean of the
  AVAILABLE 15-min samples per clock hour, **requiring ≥3 of the 4 present** (tolerate ≤1 missing;
  else that hour is NaN — no forward-fill, no cross-hour interpolation). Aligns to the first `:00`
  sample; ~3,623 hourly steps/carpark.
- **Why ≥3 and not all-4 (2026-07-31):** SG has a *systematic* single missing 15-min sample at ONE
  clock hour EVERY day (≈150 of ~163 NaN hours land at the same hour-of-day). Under "all-4-required"
  that hour is NaN daily → the longest fully-finite hourly run is **23 h ≪ C+H=576 h → 0 windows** for
  all 354 carparks. Requiring ≥3 (mean of the present samples) restores full coverage (16,992 candidate
  windows) while only averaging 3 vs 4 samples for ~1 hour/day — no fabricated values. 2-of-4 or fewer
  still yields NaN.
- Cluster unit = **carpark** (354 clusters). Realized: 650 windows, ≤2 windows/carpark.

### Coastal T-S  (`Real-TSF/TIME : Coastal_T_S/H`)
- Official **hourly** config. Schema adds `variate_names`; **multivariate length-3**:
  `TEMP` (°C, ~9.6–33), `PSAL` (practical salinity, ~31–41), `PRES_REL` (relative pressure, ~0.4–317).
- 24 stations (IMOS/GBR moorings, e.g. BMP120, GBRCCH, GBRHIS); per-station length 2,733–8,784; 0% missing.
- **Preprocessing (locked decision): use TEMP + PSAL only** (drop PRES_REL) → 24×2 = **48 univariate
  series**, expanded deterministically (fixed TEMP, PSAL order). Cluster unit = **station** (the 2
  variates of a station are correlated → not independent clusters). Realized: 650 windows over 24
  stations (27–28 each).
- Caveat: coastal T/S/pressure are partly **tidal (~12.4 h)**, not purely diurnal; we keep m=24
  (consistent with all other datasets). Only **24 clusters** → widest bootstrap CIs of the three
  targets; a Coastal null is likely under-powered rather than a true absence of effect.

### BOOM  (`Datadog/BOOM`, native-hourly subset)
- Each `ds-<N>-H` = one metric query; **multivariate** (target `(V, T)`) or occasionally univariate
  (stored flat `(T,)`); V from a handful up to ~100; T ≈ 5,000 hourly steps.
- **Selection (metadata/quality only, before any layerwise look; committed to
  `data/boom_hourly_selection.json`):** for each of the 378 hourly queries, take the **first variate**
  (ascending index) with **missing fraction ≤ 0.20** (matches the wind_farms tolerance) and **≥1
  fully-finite, non-constant 576-window** (exact `id_data._make_examples` contract). → **356 selected
  / 22 dropped**; 20,863 candidate windows. One variate per query → maximizes independent clusters.
- Cluster unit = **parent metric query** (356 clusters). Realized: 650 windows, ≤2 windows/query.
- The manifest records, per selected query, `query_dir`, `item_id`, `variate_index`, `length`,
  `missing_fraction`, `n_valid_windows` (+ the 22 dropped with reason).

## Reproduce

```bash
# 1) stage raw data on the login node (has internet)
#    SG Carpark + Coastal T-S: Real-TSF/TIME arrow shards -> $SCRATCH/chronos2/ood_targets/{sg_carpark,coastal_ts}/
#    BOOM native-hourly:
python - <<'PY'
from huggingface_hub import snapshot_download; import os
snapshot_download("Datadog/BOOM", repo_type="dataset", allow_patterns=["ds-*-H/*"],
                  local_dir=os.path.join(os.environ["SCRATCH"],"chronos2","ood_targets","boom_hourly"))
PY
# 2) build the committed BOOM selection manifest (metadata only; module load arrow)
python -m experiments.select_boom_hourly --missing-cap 0.20
# 3) screen (data-quality + native gate) and the 4x3 transfer run are GPU / compute-node steps
```

## Known limitations (state in any writeup)

1. **No proof of zero overlap.** SG Carpark's source (data.gov.sg / LTA) and Coastal T-S's source
   (IMOS moorings) are public feeds; we claim only *documented absence from the Chronos-2 manifest*
   and post-release recency, not that accidental corpus overlap is impossible.
2. **Coastal has only 24 clusters** → wide CIs; treat a Coastal null as under-powered.
3. **License**: TIME data is CC BY-NC 4.0 (non-commercial research use only).
4. **Univariate use**: Chronos-2's multivariate/covariate machinery is unprobed; Coastal & BOOM
   multivariate queries are split into univariate series (clustered by parent).
5. **Preprocessing deviations** logged per target above (SG hourly aggregation; Coastal variate
   subset; BOOM variate selection). Missingness is dropped, never filled; raw and discarded-window
   counts are recorded in `results/extended_v2/ood_pretrain_transfer/targets/<target>.json`.
6. **Pooled-readout + linear-probe + cross-dataset-transfer caveats** carry over from the ID study.
