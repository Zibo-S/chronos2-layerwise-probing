# Phase 2f: source-dataset scaling of zero-shot adapter transfer

## Question

Does the label-free adapter that aligns Chronos-2's truncated L6 states transfer better to an unseen dataset when it is fitted on more source datasets, and does it approach the dataset-specific H3 adapter? A testable hypothesis: the analysis below does not presume a monotone trend.

## Design

- Model and depth: Chronos-2 truncated at L6 (6 of 12 blocks kept); measured end-to-end speedup of the adapter pathway 1.78x at batch 1, 1.82x at batch 256.
- Adapter fit: the Phase 2c pooled fit, unchanged (the H3 label-free residual affine adapter: AdamW lr 1e-3, 1500 full-batch epochs, decay on Delta only over {0, 0.1, 1, 10, 30, 100}, validation every 25 epochs, epoch 0 = the hard cut, equal-weighted per-source objective and selection). The only variable is which source datasets the fit reads. The target is never used for fitting or selection; its test split is read once, after selection.
- Source pools: N in [1, 4, 8, 13]. For N < 13, three fixed orders per target (seeds 0, 1, 2), nested prefixes: a larger N adds datasets. N = 13 is the whole pool, one fit per target shared by the three orders.
- Order restrictions (fixed before any fit, blind to performance): the two Electricity datasets (shared meters) never sit in the same N <= 8 prefix; the N = 1 source differs across the three orders of a target.
- 40 adapter fits in total (4 targets x (3 orders x 3 sizes + 1)).

### Targets (selected from metadata before any result)

Applied to dataset metadata only (sampling frequency, domain family, Chronos-2 pretraining provenance, number of test series), never to any transfer or truncation result. R1: no Electricity dataset (series siblings), so every target's source pool is all 13 other datasets, as in the Phase 2c main protocol. R2: the test split reaches the full 262-series budget (excludes SZ Taxi, 156 series, and Coastal T-S, 24), so every target has comparable statistical power. R3: four distinct sampling frequencies, all three provenance groups (held out / not listed / listed), four distinct domain families. R4: ties go to more targets held out from Chronos-2 pretraining (the strictest zero-shot case: unseen by the backbone and by the adapter), then to roster order.

| Target | Frequency | Domain | Chronos-2 provenance | Signal | Why |
|---|---|---|---|---|---|
| Loop Seattle | 5min (m = 288) | transport | held out | freeway loop-detector speeds: smooth, strongly daily-periodic | the only 5-minute dataset; held out from Chronos-2 pretraining (fev-bench) |
| London Smart Meters | 30min (m = 48) | energy | listed | single-household electricity use: noisy and spiky | the only 30-minute dataset; documented pretraining data; same domain family as the two Electricity sources (a near-domain target) |
| BOOM | 1H (m = 24) | cloud | not listed | cloud observability metrics: the lowest dynamic range in the roster, many flat stretches (Phase-1 yield screen: 12% constant future windows) | the hourly slot must be 'not listed' (only hourly datasets are), and SG Carpark would repeat Loop Seattle's transport family |
| M5 | 1D (m = 7) | retail | held out | daily retail unit sales: intermittent low counts | daily slot; held out from Chronos-2 pretraining (Benchmark II, fev-bench); preferred to Wiki Daily (listed) by R4 |

### Source orders

| Target | Order | N = 1 | added for N = 4 | added for N = 8 | added for N = 13 |
|---|---|---|---|---|---|
| Loop Seattle | 0 | M5 | BOOM, Electricity 15min, SG Carpark | Wiki Daily, Wind Farms, Traffic, Uber TLC | Electricity, Coastal T-S, M4, SZ Taxi, London Smart Meters |
| Loop Seattle | 1 | Wind Farms | M5, Traffic, Uber TLC | Electricity 15min, Coastal T-S, M4, BOOM | SZ Taxi, London Smart Meters, SG Carpark, Electricity, Wiki Daily |
| Loop Seattle | 2 | BOOM | Coastal T-S, M4, Wiki Daily | SZ Taxi, M5, Wind Farms, SG Carpark | Uber TLC, Electricity 15min, Traffic, Electricity, London Smart Meters |
| London Smart Meters | 0 | Wind Farms | BOOM, Coastal T-S, SG Carpark | SZ Taxi, Wiki Daily, Traffic, Uber TLC | Loop Seattle, Electricity, M5, Electricity 15min, M4 |
| London Smart Meters | 1 | M4 | SG Carpark, Wiki Daily, Wind Farms | SZ Taxi, M5, Loop Seattle, Coastal T-S | Electricity 15min, Electricity, Uber TLC, BOOM, Traffic |
| London Smart Meters | 2 | Traffic | Wind Farms, Coastal T-S, M4 | Uber TLC, Loop Seattle, Electricity 15min, SZ Taxi | Wiki Daily, Electricity, BOOM, SG Carpark, M5 |
| BOOM | 0 | M4 | Wiki Daily, Uber TLC, Electricity | M5, SZ Taxi, SG Carpark, Wind Farms | London Smart Meters, Coastal T-S, Traffic, Loop Seattle, Electricity 15min |
| BOOM | 1 | Electricity 15min | Wiki Daily, M4, London Smart Meters | Uber TLC, Coastal T-S, SG Carpark, Wind Farms | SZ Taxi, Traffic, Electricity, Loop Seattle, M5 |
| BOOM | 2 | Wind Farms | M5, Loop Seattle, Electricity | SG Carpark, M4, SZ Taxi, London Smart Meters | Uber TLC, Electricity 15min, Traffic, Wiki Daily, Coastal T-S |
| M5 | 0 | BOOM | Electricity 15min, SG Carpark, Traffic | SZ Taxi, Wiki Daily, M4, London Smart Meters | Wind Farms, Loop Seattle, Coastal T-S, Electricity, Uber TLC |
| M5 | 1 | London Smart Meters | Uber TLC, Coastal T-S, Electricity | Loop Seattle, SZ Taxi, Wiki Daily, Traffic | M4, Electricity 15min, Wind Farms, BOOM, SG Carpark |
| M5 | 2 | Wiki Daily | BOOM, SG Carpark, Loop Seattle | Traffic, M4, Coastal T-S, London Smart Meters | Wind Farms, Electricity, Uber TLC, Electricity 15min, SZ Taxi |

## Coverage and gates

- Cells: 40 of 40 written.
- Preflight (Phase 2c reproduction gates per target before any fit): 4 of 4 targets passed.
- Fit reproduction (each N = 1 fit vs the committed H3 fit of its single source): 12 of 12 pass; bitwise-identical adapters: 12 of 12.
- Every cell re-ran the Phase 2c gates (full model vs Phase 1 and H3, hard cut and dataset-specific adapter vs H3, test rows, identity adapter); a failing cell is never tabulated.
- Phase 2c cross-check of the N = 13 cells: no Phase 2c L6 cell available when the tables were built (the N = 13 fits were made independently).

## Results per target

MASE change of the truncated model vs the full model on the target's test split (paired series-cluster bootstrap, B = 5000). 'Pooled - specific' is the difference of the two degradations in percentage points of the full model.

| Target | N | Mean over orders | Min | Max | Pooled - specific (mean) | Hard cut | Target-specific | Train windows |
|---|---|---|---|---|---|---|---|---|
| Loop Seattle | 1 | +25.8% | +10.6% | +44.8% | +29.4 pp | +10.4% | -3.6% | 1,394 |
| Loop Seattle | 4 | +18.5% | +9.6% | +23.3% | +22.0 pp | +10.4% | -3.6% | 5,576 |
| Loop Seattle | 8 | +26.8% | +20.7% | +31.8% | +30.4 pp | +10.4% | -3.6% | 11,152 |
| Loop Seattle | 13 | +26.2% | +26.2% | +26.2% | +29.8 pp | +10.4% | -3.6% | 18,122 |
| London Smart Meters | 1 | +13.3% | +12.5% | +13.8% | +8.7 pp | +16.6% | +4.6% | 1,394 |
| London Smart Meters | 4 | +11.8% | +10.9% | +12.4% | +7.1 pp | +16.6% | +4.6% | 5,576 |
| London Smart Meters | 8 | +10.7% | +10.0% | +12.0% | +6.1 pp | +16.6% | +4.6% | 11,152 |
| London Smart Meters | 13 | +9.6% | +9.6% | +9.6% | +5.0 pp | +16.6% | +4.6% | 18,122 |
| BOOM | 1 | +33.6% | +16.8% | +50.7% | +24.3 pp | +41.5% | +9.3% | 1,394 |
| BOOM | 4 | +16.4% | +12.0% | +21.9% | +7.0 pp | +41.5% | +9.3% | 5,576 |
| BOOM | 8 | +11.3% | +10.3% | +13.2% | +2.0 pp | +41.5% | +9.3% | 11,152 |
| BOOM | 13 | +11.2% | +11.2% | +11.2% | +1.9 pp | +41.5% | +9.3% | 18,122 |
| M5 | 1 | +6.0% | +5.0% | +6.8% | +3.7 pp | +10.8% | +2.3% | 1,394 |
| M5 | 4 | +4.0% | +3.6% | +4.4% | +1.8 pp | +10.8% | +2.3% | 5,576 |
| M5 | 8 | +3.8% | +3.2% | +4.3% | +1.5 pp | +10.8% | +2.3% | 11,152 |
| M5 | 13 | +4.3% | +4.3% | +4.3% | +2.0 pp | +10.8% | +2.3% | 18,122 |

Every cell:

| Target | N | Order | dMASE [95% CI] | Pooled - specific [95% CI] | Gap recovery | wd | epoch | fit (min) | sources |
|---|---|---|---|---|---|---|---|---|---|
| Loop Seattle | 1 | 0 | +10.6% [+3.3%, +18.7%] | +14.2 pp [+9.2 pp, +19.5 pp] | -2% | 1.0 | 450 | 1.0 | M5 |
| Loop Seattle | 1 | 1 | +44.8% [+36.6%, +54.1%] | +48.4 pp [+43.3 pp, +53.9 pp] | -331% | 30.0 | 300 | 1.0 | Wind Farms |
| Loop Seattle | 1 | 2 | +22.1% [+14.4%, +30.7%] | +25.7 pp [+20.9 pp, +30.8 pp] | -112% | 0.0 | 175 | 1.01 | BOOM |
| Loop Seattle | 4 | 0 | +9.6% [+3.8%, +16.3%] | +13.2 pp [+9.5 pp, +17.0 pp] | 8% | 0.1 | 625 | 3.75 | M5, BOOM, Electricity 15min, SG Carpark |
| Loop Seattle | 4 | 1 | +22.5% [+16.1%, +29.9%] | +26.1 pp [+22.1 pp, +30.4 pp] | -117% | 1.0 | 1050 | 3.75 | Wind Farms, M5, Traffic, Uber TLC |
| Loop Seattle | 4 | 2 | +23.3% [+16.7%, +30.6%] | +26.8 pp [+22.8 pp, +31.0 pp] | -124% | 0.0 | 400 | 3.75 | BOOM, Coastal T-S, M4, Wiki Daily |
| Loop Seattle | 8 | 0 | +20.7% [+14.1%, +28.2%] | +24.3 pp [+20.6 pp, +28.3 pp] | -99% | 0.1 | 675 | 7.37 | M5, BOOM, Electricity 15min, SG Carpark, Wiki Daily, Wind Farms, Traffic, Uber TLC |
| Loop Seattle | 8 | 1 | +27.9% [+20.2%, +36.1%] | +31.5 pp [+27.1 pp, +36.3 pp] | -169% | 1.0 | 1250 | 7.37 | Wind Farms, M5, Traffic, Uber TLC, Electricity 15min, Coastal T-S, M4, BOOM |
| Loop Seattle | 8 | 2 | +31.8% [+24.5%, +39.6%] | +35.4 pp [+31.2 pp, +39.8 pp] | -206% | 1.0 | 1000 | 7.37 | BOOM, Coastal T-S, M4, Wiki Daily, SZ Taxi, M5, Wind Farms, SG Carpark |
| Loop Seattle | 13 | all | +26.2% [+19.3%, +34.0%] | +29.8 pp [+25.9 pp, +34.0 pp] | -152% | 1.0 | 725 | 12.14 | M4, Electricity, Uber TLC, Wind Farms, SG Carpark, Coastal T-S, BOOM, Electricity 15min, SZ Taxi, London Smart Meters, M5, Wiki Daily, Traffic |
| London Smart Meters | 1 | 0 | +13.8% [+10.4%, +18.0%] | +9.2 pp [+6.6 pp, +12.4 pp] | 17% | 30.0 | 300 | 1.0 | Wind Farms |
| London Smart Meters | 1 | 1 | +13.6% [+10.1%, +18.0%] | +8.9 pp [+6.8 pp, +11.6 pp] | 18% | 1.0 | 750 | 1.0 | M4 |
| London Smart Meters | 1 | 2 | +12.5% [+9.4%, +16.1%] | +7.8 pp [+6.0 pp, +10.0 pp] | 25% | 100.0 | 1250 | 1.0 | Traffic |
| London Smart Meters | 4 | 0 | +12.0% [+8.9%, +15.6%] | +7.3 pp [+4.8 pp, +10.3 pp] | 28% | 1.0 | 350 | 3.75 | Wind Farms, BOOM, Coastal T-S, SG Carpark |
| London Smart Meters | 4 | 1 | +12.4% [+9.5%, +15.6%] | +7.8 pp [+5.6 pp, +10.2 pp] | 25% | 0.0 | 325 | 3.75 | M4, SG Carpark, Wiki Daily, Wind Farms |
| London Smart Meters | 4 | 2 | +10.9% [+8.2%, +14.3%] | +6.3 pp [+4.4 pp, +8.7 pp] | 34% | 10.0 | 225 | 3.75 | Traffic, Wind Farms, Coastal T-S, M4 |
| London Smart Meters | 8 | 0 | +10.2% [+7.5%, +13.3%] | +5.6 pp [+3.4 pp, +8.0 pp] | 39% | 0.0 | 425 | 7.37 | Wind Farms, BOOM, Coastal T-S, SG Carpark, SZ Taxi, Wiki Daily, Traffic, Uber TLC |
| London Smart Meters | 8 | 1 | +12.0% [+9.1%, +15.2%] | +7.3 pp [+5.2 pp, +9.9 pp] | 28% | 0.1 | 1225 | 7.38 | M4, SG Carpark, Wiki Daily, Wind Farms, SZ Taxi, M5, Loop Seattle, Coastal T-S |
| London Smart Meters | 8 | 2 | +10.0% [+7.3%, +13.2%] | +5.3 pp [+3.3 pp, +7.9 pp] | 40% | 1.0 | 1000 | 7.37 | Traffic, Wind Farms, Coastal T-S, M4, Uber TLC, Loop Seattle, Electricity 15min, SZ Taxi |
| London Smart Meters | 13 | all | +9.6% [+7.1%, +12.5%] | +5.0 pp [+3.0 pp, +7.3 pp] | 42% | 1.0 | 1300 | 12.15 | M4, Electricity, Uber TLC, Wind Farms, SG Carpark, Coastal T-S, BOOM, Loop Seattle, Electricity 15min, SZ Taxi, M5, Wiki Daily, Traffic |
| BOOM | 1 | 0 | +50.7% [+28.9%, +92.8%] | +41.4 pp [+23.4 pp, +75.9 pp] | -22% | 1.0 | 750 | 1.02 | M4 |
| BOOM | 1 | 1 | +33.4% [+18.7%, +61.0%] | +24.0 pp [+13.3 pp, +44.5 pp] | 20% | 0.0 | 250 | 1.01 | Electricity 15min |
| BOOM | 1 | 2 | +16.8% [+9.9%, +30.1%] | +7.5 pp [+4.3 pp, +13.5 pp] | 59% | 30.0 | 300 | 1.01 | Wind Farms |
| BOOM | 4 | 0 | +15.3% [+8.3%, +28.7%] | +5.9 pp [+2.5 pp, +12.4 pp] | 63% | 0.1 | 600 | 3.75 | M4, Wiki Daily, Uber TLC, Electricity |
| BOOM | 4 | 1 | +21.9% [+12.4%, +39.9%] | +12.6 pp [+6.6 pp, +23.7 pp] | 47% | 0.1 | 525 | 3.75 | Electricity 15min, Wiki Daily, M4, London Smart Meters |
| BOOM | 4 | 2 | +12.0% [+6.9%, +21.7%] | +2.6 pp [+0.3 pp, +6.1 pp] | 71% | 1.0 | 475 | 3.75 | Wind Farms, M5, Loop Seattle, Electricity |
| BOOM | 8 | 0 | +10.3% [+5.8%, +19.3%] | +1.0 pp [-1.3 pp, +4.0 pp] | 75% | 1.0 | 975 | 7.48 | M4, Wiki Daily, Uber TLC, Electricity, M5, SZ Taxi, SG Carpark, Wind Farms |
| BOOM | 8 | 1 | +10.5% [+5.7%, +19.5%] | +1.2 pp [-0.4 pp, +3.7 pp] | 75% | 0.0 | 950 | 7.37 | Electricity 15min, Wiki Daily, M4, London Smart Meters, Uber TLC, Coastal T-S, SG Carpark, Wind Farms |
| BOOM | 8 | 2 | +13.2% [+7.2%, +24.4%] | +3.8 pp [+1.5 pp, +7.9 pp] | 68% | 0.1 | 1100 | 7.37 | Wind Farms, M5, Loop Seattle, Electricity, SG Carpark, M4, SZ Taxi, London Smart Meters |
| BOOM | 13 | all | +11.2% [+5.9%, +21.1%] | +1.9 pp [+0.1 pp, +5.0 pp] | 73% | 0.1 | 1375 | 12.13 | M4, Electricity, Uber TLC, Wind Farms, SG Carpark, Coastal T-S, Loop Seattle, Electricity 15min, SZ Taxi, London Smart Meters, M5, Wiki Daily, Traffic |
| M5 | 1 | 0 | +5.0% [+3.7%, +6.3%] | +2.7 pp [+1.4 pp, +3.9 pp] | 54% | 0.0 | 175 | 1.01 | BOOM |
| M5 | 1 | 1 | +6.2% [+4.4%, +8.1%] | +3.9 pp [+1.6 pp, +6.0 pp] | 43% | 1.0 | 950 | 1.01 | London Smart Meters |
| M5 | 1 | 2 | +6.8% [+4.3%, +9.6%] | +4.5 pp [+1.3 pp, +7.5 pp] | 37% | 0.0 | 375 | 1.0 | Wiki Daily |
| M5 | 4 | 0 | +4.1% [+2.8%, +5.3%] | +1.9 pp [+0.5 pp, +3.0 pp] | 62% | 1.0 | 1100 | 3.75 | BOOM, Electricity 15min, SG Carpark, Traffic |
| M5 | 4 | 1 | +3.6% [+2.2%, +5.1%] | +1.3 pp [+0.5 pp, +2.2 pp] | 67% | 0.0 | 300 | 3.75 | London Smart Meters, Uber TLC, Coastal T-S, Electricity |
| M5 | 4 | 2 | +4.4% [+3.1%, +5.7%] | +2.1 pp [+0.8 pp, +3.5 pp] | 60% | 0.1 | 575 | 3.75 | Wiki Daily, BOOM, SG Carpark, Loop Seattle |
| M5 | 8 | 0 | +4.3% [+3.2%, +5.5%] | +2.0 pp [+0.5 pp, +3.5 pp] | 60% | 0.0 | 950 | 7.41 | BOOM, Electricity 15min, SG Carpark, Traffic, SZ Taxi, Wiki Daily, M4, London Smart Meters |
| M5 | 8 | 1 | +3.2% [+2.0%, +4.5%] | +0.9 pp [-0.5 pp, +2.1 pp] | 71% | 1.0 | 1400 | 7.36 | London Smart Meters, Uber TLC, Coastal T-S, Electricity, Loop Seattle, SZ Taxi, Wiki Daily, Traffic |
| M5 | 8 | 2 | +3.9% [+2.7%, +5.1%] | +1.6 pp [+0.1 pp, +2.9 pp] | 64% | 0.1 | 700 | 7.37 | Wiki Daily, BOOM, SG Carpark, Loop Seattle, Traffic, M4, Coastal T-S, London Smart Meters |
| M5 | 13 | all | +4.3% [+2.9%, +5.7%] | +2.0 pp [+0.9 pp, +3.3 pp] | 60% | 0.1 | 1375 | 12.14 | M4, Electricity, Uber TLC, Wind Farms, SG Carpark, Coastal T-S, BOOM, Loop Seattle, Electricity 15min, SZ Taxi, London Smart Meters, Wiki Daily, Traffic |

## Scaling analysis

Counts are over (target, order) pairs: 12 pairs with both N = 1 and N = 13 written. With four targets they describe these datasets; they are not a population test.

1. **Larger pools.** From N = 1 to N = 13 the degradation decreased with a paired 95% CI excluding zero in 8 pairs, increased in 2, and the CI contained zero in 2 (point estimate lower at N = 13 in 10). Mean change per target (pp): Loop Seattle +0.4 pp, London Smart Meters -3.7 pp, BOOM -22.4 pp and M5 -1.7 pp.
2. **Monotonicity.** 3 of 12 complete sequences (1, 4, 8, 13) never increase (point estimates). Consecutive steps: 13 improve, 5 worsen, 18 indistinguishable (paired CI), of 36.
3. **Dependence on which datasets.** Range over the three orders at a fixed N (pp): Loop Seattle N = 1: 34.3, N = 4: 13.7, N = 8: 11.1; London Smart Meters N = 1: 1.4, N = 4: 1.5, N = 8: 2.0; BOOM N = 1: 33.9, N = 4: 9.9, N = 8: 2.8; M5 N = 1: 1.8, N = 4: 0.8, N = 8: 1.1.
4. **Negative transfer.** Cells worse than the hard cut (CI of pooled / hard above 1): LOOP_SEATTLE_5T/N01_seed1, LOOP_SEATTLE_5T/N01_seed2, LOOP_SEATTLE_5T/N04_seed1, LOOP_SEATTLE_5T/N04_seed2, LOOP_SEATTLE_5T/N08_seed0, LOOP_SEATTLE_5T/N08_seed1, LOOP_SEATTLE_5T/N08_seed2, LOOP_SEATTLE_5T/N13_all, boom_hourly/N01_seed0. Steps that significantly increased the degradation: LOOP_SEATTLE_5T/seed0/4->8, LOOP_SEATTLE_5T/seed0/8->13, LOOP_SEATTLE_5T/seed1/4->8, LOOP_SEATTLE_5T/seed2/4->8, m5/seed1/8->13.
5. **N = 13 vs the dataset-specific adapter.** Loop Seattle: +29.8 pp [+25.9 pp, +34.0 pp], worse; London Smart Meters: +5.0 pp [+3.0 pp, +7.3 pp], worse; BOOM: +1.9 pp [+0.1 pp, +5.0 pp], worse; M5: +2.0 pp [+0.9 pp, +3.3 pp], worse.
6. **Diminishing returns.** Mean gain per added source dataset (pp of degradation removed per dataset, mean over orders): Loop Seattle 1->4: +2.46, 4->8: -2.08, 8->13: +0.11; London Smart Meters 1->4: +0.51, 4->8: +0.26, 8->13: +0.22; BOOM 1->4: +5.75, 4->8: +1.26, 8->13: +0.02; M5 1->4: +0.65, 4->8: +0.06, 8->13: -0.10. Strictly decreasing gains: London Smart Meters, BOOM, M5.

## Aggregate (descriptive)

Mean over targets of the mean over orders; the interval resamples targets, then orders within a target (four targets: descriptive, not a test).

| N | Targets | dMASE mean [interval] | Pooled - specific mean [interval] | Train windows | Steps per fit | Window passes | Fit (min) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | +19.7% [+9.3%, +32.5%] | +16.5 pp [+6.1 pp, +29.0 pp] | 1,394 | 9,000 | 12,546,000 | 1.0 |
| 4 | 4 | +12.7% [+6.9%, +18.3%] | +9.5 pp [+3.2 pp, +18.3 pp] | 5,576 | 9,000 | 50,184,000 | 3.8 |
| 8 | 4 | +13.2% [+5.7%, +22.6%] | +10.0 pp [+1.6 pp, +23.8 pp] | 11,152 | 9,000 | 100,368,000 | 7.4 |
| 13 | 4 | +12.8% [+6.0%, +22.1%] | +9.7 pp [+2.0 pp, +22.9 pp] | 18,122 | 9,000 | 163,098,000 | 12.1 |

## Fixed limitations

- Four targets. Counts are descriptive; no claim is made about datasets in general.
- Data volume grows with N: every source adds its full 1,394 training windows (the Phase 2c per-source budget), so N also scales the training data and the work per optimization step (table above). This measures the practical effect of a larger source pool, not diversity at a fixed amount of data.
- The two Electricity datasets share meters: N = 13 contains 12 dataset families, not 13 independent ones.
- No trend is fitted or extrapolated beyond N = 13. These results cannot show that more source datasets would make zero-shot alignment match dataset-specific alignment.
- The dataset-specific H3 adapter is calibrated on the target's unlabeled training inputs; it is a practical reference, not a guaranteed upper bound.
- One model (Chronos-2) and one depth (L6).
- Optional follow-up, not run: a fixed-total-window control (each of the N sources subsampled to 1,394 / N training windows, so every fit sees 1,394 windows) would separate the number of datasets from the amount of data; at about one minute per fit it would cost well under an hour of GPU time.

