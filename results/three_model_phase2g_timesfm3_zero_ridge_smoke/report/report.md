# TimesFM-3: identity- vs zero-anchored closed-form alignment (1 of 14 datasets)

Median test MASE change vs the full model over datasets. `identity`, `hard` and `probe` are the committed H3 numbers; `zero` (primary) and `zero_out` are new.

| depth | hard | identity | zero | zero_out | best anchor | probe | zero < identity (CI) | zero vs probe better / worse (CI) |
|---|---|---|---|---|---|---|---|---|
| Emb | 69x | +30.6% | +24.3% | +23.6% | +24.3% | +34.7% | 1 / 1 | 1 / 0 |
| L5 | 74x | +62.7% | +3.5% | +3.6% | +3.5% | +22.9% | 1 / 1 | 1 / 0 |
| L19 | 9x | +3.6% | -0.1% | +0.1% | -0.1% | +12.4% | 1 / 1 | 1 / 0 |

Fit diagnostics (median over datasets; distillation error = MSE between the adapted and the full model's normalized head outputs).

| depth | family | log10 kappa | dof (variance-weighted) | train | val | test | val / train | R2 test | R2 test < 0 |
|---|---|---|---|---|---|---|---|---|---|
| Emb | identity | -8.0 | 211 | 0.1205 | 0.4348 | 0.209 | 3.6 | 0.413 | 0 |
| Emb | zero | -5.0 | 68 | 0.1419 | 0.3525 | 0.1746 | 2.5 | 0.510 | 0 |
| Emb | zero_out | -4.0 | 80 | 0.1381 | 0.3529 | 0.1723 | 2.6 | 0.516 | 0 |
| L5 | identity | -9.0 | 1271 | 0.002291 | 0.6046 | 0.383 | 263.9 | -0.075 | 1 |
| L5 | zero | -5.0 | 571 | 0.01343 | 0.1056 | 0.07535 | 7.9 | 0.788 | 0 |
| L5 | zero_out | -3.0 | 359 | 0.01778 | 0.0994 | 0.06394 | 5.6 | 0.820 | 0 |
| L19 | identity | -9.0 | 1246 | 0.0003078 | 0.03215 | 0.01878 | 104.4 | 0.947 | 0 |
| L19 | zero | -6.0 | 679 | 0.001576 | 0.007255 | 0.005278 | 4.6 | 0.985 | 0 |
| L19 | zero_out | -4.0 | 474 | 0.001018 | 0.006929 | 0.004775 | 6.8 | 0.987 | 0 |

Reproduction gates passed for 1 of 1 datasets.
