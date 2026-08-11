# Forecasting comparison summary (q9, runs0-1-2)

## Electricity
- MLP vs linear @L12+LN:   ΔMASE = +0.251  (MLP worse)
- MLP vs linear @entrance: ΔMASE = +0.120  (MLP worse)
- entrance vs final: linear -0.018, MLP +0.113  (negative = entrance already as good/better)
- native vs MLP@ref: +0.457  (native better by this much MASE)
- native vs seasonal-naive: -0.429  (native beats seasonal)

## Uber
- MLP vs linear @L12+LN:   ΔMASE = +0.103  (MLP worse)
- MLP vs linear @entrance: ΔMASE = +0.167  (MLP worse)
- entrance vs final: linear -0.013, MLP -0.076  (negative = entrance already as good/better)
- native vs MLP@ref: +0.270  (native better by this much MASE)
- native vs seasonal-naive: -0.447  (native beats seasonal)

## M4
- MLP vs linear @L12+LN:   ΔMASE = +1.051  (MLP worse)
- MLP vs linear @entrance: ΔMASE = +0.303  (MLP worse)
- entrance vs final: linear +0.508, MLP +1.256  (negative = entrance already as good/better)
- native vs MLP@ref: +3.626  (native better by this much MASE)
- native vs seasonal-naive: -0.632  (native beats seasonal)

## WindFarms
- MLP vs linear @L12+LN:   ΔMASE = +0.169  (MLP worse)
- MLP vs linear @entrance: ΔMASE = +0.386  (MLP worse)
- entrance vs final: linear -0.041, MLP -0.258  (negative = entrance already as good/better)
- native vs MLP@ref: +0.623  (native better by this much MASE)
- native vs seasonal-naive: -0.188  (native beats seasonal)
- WindFarms seasonal dominance: no
- MLP removes linear's late-layer degradation? linear entrance→ref -0.041 vs MLP -0.258 — not clearly
