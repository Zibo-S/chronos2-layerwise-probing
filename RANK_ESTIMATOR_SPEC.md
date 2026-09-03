# Rank estimator spec — Chronos-2 repr-metrics (for the Egor / PT-ID reconciliation)

Spec only, no interpretation. Line references are to the files on branch `zibo/repr-metrics`
@ `7a940d9`. Both estimators use the SAME scalar functional (Roy & Vetterli 2007 effective
rank); they differ only in the matrix they are applied to.

## Shared: the effective-rank functional

```python
# probing/repr_metrics.py:114-126
def effective_rank(Z):                      # Z is (rows x D)
    Z = np.asarray(Z, dtype=np.float64)     # float64
    s = np.linalg.svd(Z, compute_uv=False)  # SINGULAR VALUES of Z itself (no Gram, no cov)
    s = s[s >= 1e-12 * s.max()]             # EIG_GUARD relative floor, repr_metrics.py:77,124
    q = s / s.sum()                          # normalized SINGULAR VALUES (not eigenvalues)
    return float(np.exp(-(q * np.log(q)).sum()))   # exp(Shannon entropy of q)
```

**No centering. No standardization. No covariance matrix.** SVD is taken on the raw matrix
as assembled below. (`matrix_entropy`, repr_metrics.py:92-111, is a *separate* functional
using eigenvalues `lambda = s^2`; do not conflate.)

## The 10-line spec

1. **Input windows** — one context window per series, the LAST `C = 512` points
   (`repr_metrics.py:74`, `:156`); `n_series = 200` max (`MAX_SERIES`, `:75`), sampled with
   `np.random.default_rng(seed=0)` (`SEED = 0`, `:73`, `:164`).
2. **Patching** — Chronos-2 `patch_size = 16`, so `n_content = ceil(512/16) = 32` content
   patches (`:261`); encoder positions are `[content..., REG, masked_future]`.
3. **Pooling / position selection** — **content patches ONLY**: `full[:, :n_content, :]`
   (`:306`). REG and masked_future are excluded. `Embed` is the `input_patch_embedding`
   output for the context patches; `L1..L12` are the 12 encoder-block outputs (pre final
   layer norm); `L12_postln` is the `encoder.final_layer_norm` output (a separate 14th entry,
   not part of the `Embed..L12` depth axis).
4. **Per-series matrix** — for each series and layer, a `(n_patches x D) = (32 x 768)`
   float32 array, stored ragged as an object array (never zero-padded).
5. **`prompt_effrank` (per-window estimator)** — `effective_rank` applied to EACH series'
   `(32 x 768)` matrix, i.e. rows = the 32 content patches of ONE series; the reported number
   is the **mean over the 200 series** of those per-series values
   (`prompt_effrank_mean`, `repr_metrics.py:333`, `:340`). Ceiling `min(32, 768) = 32`.
6. **`dataset_effrank` (dataset estimator)** — first mean-pool each series over its content
   patches -> a `(768,)` vector; stack the 200 of them into `Zd` of shape
   **`(n_series x D) = (200 x 768)`** (`repr_metrics.py:335`); apply `effective_rank(Zd)`
   once (`:343`). Ceiling `min(200, 768) = 200`.
7. **Order of operations (dataset)** — mean over patches FIRST, then SVD across series. No
   centering of `Zd`, no per-feature standardization, no whitening.
8. **Numerics** — cast to float64 before SVD; relative singular-value floor `1e-12 * s.max()`;
   `metrics.json` accumulates the patch mean in float32 (values agree with a float64 path to
   ~1e-7 relative).
9. **Resampling (only where CIs/bands are quoted)** — series-level; prompt-level means use
   bootstrap `m=200` WITH replacement; dataset-level spectral stats use subsampling
   `m=126` WITHOUT replacement, `B=5000`, one shared paired index array per run
   (`repr_metrics_bootstrap.py:136-139`, `:205`, `:212-213`).
10. **Constants** — `n_series = 200`, `D = 768`, `n_patches = 32`, `C = 512`,
    `patch_size = 16`, `seed = 0`, content-patch pooling, model `amazon/chronos-2` frozen
    (eval, `no_grad`, float32 forward).

## Runtime confirmation of the shapes

```
  n_series=200  per-series matrix shape=(32, 768)  dtype=float32
  dataset-level matrix Zd shape=(200, 768)
  metrics.json n_series=200  D=768  n_patches[0]=32
  provenance: C=512 seed=0 n_content_patches=32 patch_size=16
  layer_axis=['Embed','L1',...,'L12','L12_postln']
```

## The two numbers most likely to differ from a PT-ID plot

- `dataset_effrank` is bounded by **200** (n_series), `prompt_effrank` by **32** (n_patches).
  A plot whose rank ceiling is 768 or the window count is computing a third thing.
- The reported flatness figure `44.2 / 35.5 / 0.010` is the **subsampled** mean within-draw
  range at `m=126` over `Embed..L12`, NOT the naive full-sample max-min
  (which is `68.9 / 52.8 / 0.012`). See `VERIFY_prefreeze.md` CHECK 4.
