# ICLR 2027 rebuttal: answers to reviewers 2p99, 96qN and dHTD

Ready to paste into OpenReview. Each answer sits between two horizontal rules under the slot it
belongs to. Markdown tables and `$...$` math render on OpenReview.

- **Source of every number:** `python experiments/make_rebuttal_numbers.py` recomputes it from the
  committed result tables and writes `results/rebuttal_numbers/rebuttal_numbers.{txt,json}`.
- **Appendix and section numbers:** they refer to the revised `main.tex` and were checked against
  the compiled PDF.
- **TimesFM-3 corrected-adapter numbers:** they are offline (cached representations), and the
  answers say so. The physical-truncation check for the new adapters is reported in Appendix B.3.1.
- **Your own text:** the dHTD Q2 answer is unchanged; its numbers were re-checked against the 2d
  results.

---

## General comment to all reviewers (optional, post as an official comment)

---

**Summary of the revision.** We ran five additional experiments with the protocol of the paper (identical windows and splits, validation-only selection, the test split read once, paired series-cluster bootstrap) and added them to the appendix:

1. Leave-one-dataset-out adapter transfer for Chronos-2 at L4, L6 and L9 (Appendix B.4.2; Reviewers 2p99, 96qN, dHTD).
2. Transfer as a function of the number of source datasets, from 1 to 13, at L6 (Appendix B.4.3; 2p99, dHTD).
3. Non-contiguous block selection and single-block removal for Chronos-2 (Appendix B.4.4; dHTD).
4. Architecture-aware versus generic content-mean readouts for all three models (Appendix B.1.5; dHTD).
5. A corrected regularizer for the closed-form TimesFM-3 adapter (Appendix B.3.1; 96qN).

The fifth changes a result. Reviewer 96qN's question about TimesFM-3 at L0 led us to find that our closed-form TimesFM-3 adapter was regularized toward the hard cut, which for TimesFM-3 is 35 to 160 times worse than the full model in MASE. Validation therefore selected almost no regularization, and the adapter overfit its 1,394 training windows. Shrinking the adapter toward zero instead, with every other choice unchanged, reduces the median MASE increase of the aligned TimesFM-3 pathway at 25, 50 and 75% of depth from +69.8%, +67.3% and +47.9% to +10.6%, +11.3% and +8.5%. The Chronos-2 and TiRex adapters are fitted iteratively with early stopping and do not show this problem: at the selected checkpoint their validation error is a median 1.8 and 2.3 times their training error, against 13 to 184 times for the original TimesFM-3 fit. We made the change after seeing the original test results, so the revised paper reports both fits, uses the corrected one in Sections 4.3 to 5, and marks its numbers as computed offline.

We also report the tunnel referenced to the full model rather than to the final-block readout (Appendix B.1.6), the pre-specified test of cutting at the tunnel entrance (Section 4.4, Appendix B.4.1), and results on the model-dataset pairs documented as held out from pretraining (Appendix B.4.5).

---

## Reviewer 2p99

### Q1. Can the authors clarify whether an adapter trained on one dataset transfers to unseen datasets?

---

Yes, partially. We ran leave-one-dataset-out transfer for Chronos-2 at L4, L6 and L9, which run 2.43×, 1.78× and 1.28× faster than the full model at batch size 1 (Appendix B.4.2). For each target dataset, one label-free adapter is fitted on the remaining datasets (13, or 12 when the target is one of the two Electricity datasets, since they share meters) and evaluated once on the target, which is never used for fitting or selection. The objective, optimizer and selection rule are those of the paper.

Median test MASE increase over the full model, 14 datasets:

| | L4 (2.43×) | L6 (1.78×) | L9 (1.28×) |
|---|---|---|---|
| Hard cut | +37.0% | +20.1% | +17.1% |
| Transferred adapter (no target data) | +12.7% | +9.0% | +4.0% |
| Dataset-specific adapter (paper) | +3.1% | +2.3% | +2.2% |
| Transferred within 5% of the full model | 4/14 | 4/14 | 9/14 |

The transferred adapter is better than the hard cut on 13 of the 14 datasets at every depth (paired 95% CI) and recovers a median 70% to 80% of the hard-cut gap. It does not replace dataset-specific calibration: it is a median 7.1, 5.1 and 1.7 percentage points worse at L4, L6 and L9. The exception is LOOP Seattle, our only 5-minute dataset, where it is worse than the hard cut at L4 and L6 (+27.2% against +5.6% at L4).

We also varied the number of source datasets at L6 for four targets fixed in advance from metadata: LOOP Seattle, London Smart Meters, BOOM and M5 (Appendix B.4.3). An adapter fitted on a single other dataset is unreliable: across 12 target-source pairs, its MASE increase ranges from +5.0% to +50.7%, and on average it is no better than the hard cut (+19.7% against +19.8%). With 4, 8 and 13 sources, the mean is +12.7%, +13.2% and +12.8%, against +3.2% for dataset-specific adapters. In our collection, most of the benefit of pooling comes from the first four sources.

Alignment therefore transfers well enough to give a calibration-free operating point at mild truncation (+4.0% at 1.28×), while at 2.43× calibration on unlabeled target windows remains clearly better. These results are based on 14 datasets and do not show that a single adapter will transfer to every new dataset; LOOP Seattle is a counterexample. We have added both experiments and rewritten the corresponding limitation.

---

### Q2. Can the forecasting tunnel reliably select a truncation depth without separately evaluating multiple truncated models?

---

Not on its own, and the paper does not claim that it does (Section 4.4: "its entrance need not define the best truncation depth"). The tunnel is a recoverability diagnostic: it marks where later blocks stop improving what a linear readout can recover, not where a truncated model matches the full model.

We specified one test of using it as a cut rule before computing the truncation results: cut each model at its validation-selected 5% entrance and evaluate no other depth. With the aligned pathway, the truncated model stays within 5% of the full model on 7 of 14 datasets for Chronos-2 (median +5.5%, median speedup 2.67×) and on 5 of 12 for TiRex (+6.3%, 1.42×; the other two TiRex entrances are the final block). For TimesFM-3, it holds on 0 of 14 datasets with the submitted adapter (+35.5%) and on 2 of 14 with the corrected one (+14.2%; see our answer to Reviewer 96qN, Q3). This test is now reported in Section 4.4 and Appendix B.4.1.

For Chronos-2 and TiRex, the entrance is still informative in two ways:

- **It adapts the cut to the dataset.** At a similar speed, cutting at each dataset's entrance gives a somewhat higher median than a single fixed depth, but about half the worst case:
  - Chronos-2: worst dataset +15.8% at the entrance, against +32.9% at fixed L4 (2.43×) and +33.3% at fixed L3 (2.91×).
  - TiRex: +13.6% at the entrance, against +27.3% at fixed L9 (1.33×).
- **It predicts where the aligned pathway becomes compatible.** Applying the same sustained criterion to the aligned pathway against the full model's validation loss, the first depth from which it stays within 5% has a Spearman correlation of 0.81 (Chronos-2) and 0.80 (TiRex) with the tunnel entrance across datasets. The two depths differ by at most one block on 9 and 11 of 14 datasets. For TimesFM-3, the correlation is −0.02 with the corrected adapter: its aligned pathway becomes compatible only near the final block (median L18.5 of 20), regardless of where the readout tunnel begins.

Both comparisons are post hoc.

In practice, the tunnel narrows the search rather than replacing it. It requires one pass of the full model and linear readouts, without adapters or truncated models; adapters then need to be validated only around the entrance. The revised text describes the tunnel as an indicator of truncation tolerance, not as a depth-selection rule.

---

### Q3. How do the conclusions change when considering only model-dataset pairs documented as held out from pretraining?

---

Nine of the 42 pairs are documented as held out (Table 6): Chronos-2 on LOOP Seattle, SZ Taxi, M5 and Monash Traffic; TimesFM-3 on LOOP Seattle, SZ Taxi and M5; and TiRex on M5 and Monash Traffic. Another 14 pairs are not listed. This means only that the documentation does not identify the dataset as training data, so we do not count these pairs as held out.

Recoverability is unchanged on the held-out pairs. The tunnel begins before the final block in 8 of the 9 (the exception is TiRex on Monash Traffic). At the validation-selected entrance, the test loss differs from that of the final-block readout by a median +1.3% (range −8.3% to +6.9%), the same median as over all 42 pairs. Including the not-listed pairs, 22 of 23 pairs enter before the final block.

Median test MASE increase over the full model at 25, 50 and 75% of depth, held-out datasets only (all 14 datasets in parentheses):

| | Hard cut | Aligned pathway |
|---|---|---|
| Chronos-2 (4 datasets) | +9.3%, +10.6%, +6.3% | +0.9%, +1.1%, +1.1% (+4.4%, +2.3%, +2.2%) |
| TimesFM-3 (3 datasets) | 34×, 89×, 100× | +3.9%, +4.5%, +4.4% (+10.6%, +11.3%, +8.5%)* |
| TiRex (2 datasets) | +160%, +138%, +132% | +27.2%, +15.6%, +8.3% (+13.4%, +10.6%, +6.2%) |

\*Corrected adapter (see our answer to Reviewer 96qN, Q3).

The qualitative conclusions therefore hold on the held-out pairs: forecasts are recoverable early, the hard cut fails, and alignment recovers most of the hard-cut gap. Two caveats limit how far this subset can be pushed:

- It contains only 4, 3 and 2 datasets per model.
- On SZ Taxi and M5, the full models are close to the best constant forecast (validation loss 0.89 to 0.91 of the constant forecast's), so small degradations there are weak evidence. Monash Traffic, where the models have the most skill (0.35 for Chronos-2, 0.40 for TiRex), is also where truncation is least tolerated: +27.3% for aligned Chronos-2 at L4.

In the strictest setting, neither the backbone nor the adapter has seen the target. Using the leave-one-dataset-out adapter on Chronos-2's four held-out datasets, it improves on the hard cut on three of them at L4, L6 and L9, and fails on LOOP Seattle (Appendix B.4.5).

---

### Weaknesses not covered by the questions (post as a separate comment to 2p99, optional)

---

**Chronos-2-small and the practical value of truncation.** Where Chronos-2-small applies, it is the better choice at moderate speedups:

- At batch size 1, it runs 1.80× faster than Chronos-2 at a median +0.8% MASE (within 5% on 10 of 14 datasets). This is better than aligned truncation at a similar speed (L6: 1.78×, +2.3%).
- At batch size 256, it runs 3.28× faster. Aligned truncation only matches that speed when it keeps three blocks (L3: 3.10×, +4.4%).

The revised Section 4.4 states this explicitly. Truncation is useful when a separately pretrained smaller model is unavailable or not fast enough:

- At batch size 1, Chronos-2 truncated after L4 runs 2.43× faster at +3.1%, and after L3 2.91× faster at +4.4%, both beyond Chronos-2-small's speed.
- No smaller version of TimesFM-3 or TiRex has been released, so for these models truncation is the only training-free way to trade accuracy for speed. TiRex truncated after L7 runs 1.71× faster at +8.5%.
- For TimesFM-3, the corrected adapter (answer to Reviewer 96qN, Q3) changes the accuracy side substantially (+10.6% after L1, instead of +38.7%). Its batch-size-1 speedup is still capped at 1.43× by the per-call cost outside the backbone; at batch size 256, truncation after L1 runs 4.65× faster at +10.6% (offline accuracy).

The two approaches are complementary: a smaller pretrained model could itself be truncated, which we did not test.

**Novelty.** We agree that probing, alignment and truncation are established tools. For TSFMs, our contribution is:

- the sustained forecasting tunnel, measured at the states each model's own head reads;
- the separation between recoverability and native-pathway compatibility;
- physically truncated models with measured end-to-end speed.

Appendix B.1.5 shows that this TSFM-specific choice of states changes both the forecasting loss and the apparent tunnel location (our answer to Reviewer dHTD, Q2).

---

## Reviewer 96qN

### Q1. How do the tunnel entrances change if the 5% tolerance is measured relative to the native full-model performance instead of the final-block linear probe, and how many of the 42 pairs still exhibit a tunnel?

---

Measured against the full model, the readout criterion of Eq. (4) becomes $R_j \le 1.05\,R_{\mathrm{full}}$ for every depth $j$ from $\ell$ to the final block, where $R_{\mathrm{full}}$ is the full model's validation loss in the same target space. A tunnel then exists only where the final-block readout is itself within 5% of the full model. This holds for 6 of the 42 pairs: Chronos-2 and TiRex on SZ Taxi and Coastal T-S, TimesFM-3 on Coastal T-S, and TiRex on Wiki Daily. In all six, the tunnel begins before the final block. At a 10% tolerance, 15 of the 42 pairs have a tunnel, all beginning before the final block; at 2%, two pairs have one, and one of them begins before the final block.

The limiting factor is the ceiling of a linear readout, not depth: at the final block, the readout's validation loss is a median 17.7%, 20.3% and 12.7% higher than the full model's for Chronos-2, TimesFM-3 and TiRex. This is why we reference the tunnel to the final-block readout: it isolates what later blocks add for a fixed readout from what a linear readout cannot express at any depth. Section 3.1 states that the tunnel "does not indicate where a truncated model would match the full model". We agree that this should be more prominent, so we have added it to Section 4.1 and reported the native-referenced counts in Appendix B.1.6.

Near-native performance is a question for the aligned pathway, which, unlike a readout, reproduces the full model at the final block. Applying the same sustained criterion to the aligned pathway against the full model's validation loss, it stays within 5% from some depth before the final block in 33 of 42 pairs:

- Chronos-2: 13 of 14
- TiRex: 9 of 14
- TimesFM-3: 11 of 14 with the corrected adapter of Q3, 5 with the original

At 10%, this holds for 39 of 42 pairs. For Chronos-2 and TiRex, this depth is rank-correlated with the readout tunnel across datasets (Spearman 0.81 and 0.80). For TimesFM-3, it is not (−0.02), because its aligned pathway reaches the full model only near the final block.

On the seven L0 entrances, we agree with your reading for four of them. For M5 (all three models) and TiRex on SZ Taxi, the final-block readout is within 5% of the best constant forecast. An L0 entrance there means that later blocks add nothing a linear readout can use, not that L0 forecasts well. In the other three pairs (TimesFM-3 on BOOM, Wind Farms and Wiki Daily), the readout has skill (validation loss 0.72 to 0.88 of the constant forecast's), and L0 matches the final-block readout. We now mark the four low-skill pairs in Table 1.

---

### Q2. Does an adapter trained on 1,394 windows from one dataset or pooled data transfer to unseen datasets, and can you provide leave-one-dataset-out results at a few depths to show whether a single truncated checkpoint is deployable?

---

We ran leave-one-dataset-out transfer for Chronos-2 at L4, L6 and L9 (Appendix B.4.2). For each target, one label-free adapter is fitted on the other datasets (13, or 12 when the target is one of the two Electricity datasets, since they share meters) and evaluated once on the target, which is never used for fitting or selection.

Median test MASE increase over the full model, 14 datasets:

| | L4 (2.43×) | L6 (1.78×) | L9 (1.28×) |
|---|---|---|---|
| Hard cut | +37.0% | +20.1% | +17.1% |
| Transferred adapter | +12.7% | +9.0% | +4.0% |
| Dataset-specific adapter | +3.1% | +2.3% | +2.2% |
| Transferred within 5% / 10% (of 14) | 4 / 5 | 4 / 8 | 9 / 11 |

A single truncated checkpoint per depth is therefore deployable at mild truncation, but not at aggressive truncation:

- **At L9,** the transferred adapter is a median 1.7 percentage points worse than dataset-specific calibration and within 5% of the full model on 9 of 14 datasets.
- **At L4,** it is a median 7.1 points worse than calibration and within 5% on only 4 datasets.
- **Against the hard cut,** it is better on 13 of 14 datasets at every depth. The exception is LOOP Seattle, our only 5-minute dataset, where it is worse than the hard cut at L4 and L6.

Transfer from a single source dataset is unreliable. At L6, its MASE increase ranges from +5.0% to +50.7% over 12 target-source pairs, and its mean is no better than the hard cut (+19.7% against +19.8%). With four or more sources, the mean is about +13% (Appendix B.4.3).

On the cost of calibration: the adapter needs only the full model's own forecasts on unlabeled target windows. For Chronos-2, this means one forward pass over 1,394 windows (about 1 s at batch size 256 on an A100) and an adapter fit of about one minute on the same GPU, including the weight-decay sweep. We agree that calibration removes the zero-shot property at aggressive cuts, and the revised Section 4.4 and limitations now state both the cost and the transfer results.

---

### Q3. Why does aligned TimesFM-3 at L0 with +31.3% error outperform most intermediate depths from L2 to L17, and is the adapter essentially learning a linear forecaster directly from the input patch embedding?

---

Yes. TimesFM-3's head is a single linear map, so adapter plus head is an affine map from the last context token's state to the forecast. At L0, this state is computed from the last 32 context values alone (normalized with the context's mean and standard deviation), before any attention. The aligned pathway at L0 is therefore a linear forecaster on an embedding of the last input patch.

Your question led us to the reason why L0 outperformed L2 to L17, and it lies in our fit, not in the representations. The closed-form adapter was regularized toward the identity, $\lambda\lVert\Delta_\ell\rVert_F^2$ with $A_\ell(h)=h+\Delta_\ell h+b_\ell$, that is, toward the hard cut. For TimesFM-3, the hard cut is 35 to 160 times worse than the full model. Validation therefore selected almost no regularization ($\kappa\le10^{-8}$ in 231 of 280 fits), and the adapter nearly interpolated the 1,394 training windows: from L1 to L17, its median validation error was 13 to 184 times its training error. At L0, the same fit kept about 250 effective parameters per output direction, against 1,000 to 1,280 of 1,280 at later depths, and generalized (ratio 1.9).

We repeated the fit with the penalty on the whole linear map, $\lambda\lVert I+\Delta_\ell\rVert_F^2$, which shrinks the adapter toward zero, and changed nothing else: the states, splits, targets, $\kappa$ grid and validation-only selection are the same (Appendix B.3.1). Median test MASE increase over the full model, 14 datasets:

| | L0 | L1 | L5 | L10 | L15 | L18 | L19 |
|---|---|---|---|---|---|---|---|
| Aligned, identity anchor (submission) | +31.3% | +38.7% | +69.8% | +67.3% | +47.9% | +14.3% | +5.4% |
| Aligned, zero anchor | +24.3% | +10.6% | +10.6% | +11.3% | +8.5% | +3.9% | +1.3% |
| Supervised linear readout | +31.9% | +15.8% | +16.8% | +17.0% | +14.8% | +13.2% | +12.7% |

The zero-anchored adapter is better than the original on all 14 datasets at L1 to L8 and L10 to L15 (paired 95% CI). Its median validation error is 1.9 to 3.2 times its training error, and L0 becomes its worst depth. Validation prefers it in all 280 fits, but we made the change after seeing the test results of the original fit, so the paper reports both fits.

The corrected aligned pathway also has a lower test MASE than the supervised readout on 10 to 12 of the 14 datasets at every depth from L1 (paired CI). Both are affine maps of the same state, but they are trained differently: the readout is fitted to the observed future with the pinball loss, the adapter to the full model's forecasts. We do not attribute the difference to a single cause. The adapter remains poor on BOOM and Wind Farms (+181% and +38% at L5, against +47% and +10% for the readout), where validation selects a nearly constant forecast.

Two statements of the submission therefore change:

- The +48% to +70% degradation at 25 to 75% of depth was produced by our regularizer; the corrected values are +10.6%, +11.3% and +8.5%.
- The supervised readout is not closer to the full model than the aligned pathway.

What remains is that TimesFM-3 becomes compatible late: its aligned pathway stays about 10% above the full model up to L14 and is within 5% on most datasets only at L18 (9 of 14). We have revised Sections 4.3, 4.4 and 5, and kept the original fit in Appendix B.3.1. These numbers are computed offline. The adapter has the same shape as before, so latency is unchanged; Appendix B.3.1 reports the status of the physical-truncation check for the new adapters.

---

### Weaknesses not covered by the questions (post as a separate comment to 96qN, optional)

---

**The tunnel as a guide to cut points.** We agree that the tunnel does not locate the best cut, and the paper does not use it for that. Two parts of this point change with the new results:

- **TimesFM-3 inside the tunnel.** The 40% to 68% losses inside the tunnel were produced by the regularizer of our TimesFM-3 adapter (answer to Q3). With the corrected adapter, the median MASE increase at a matched depth is +8.9% to +16.5% inside the tunnel and +11.5% to +29.5% before it. Truncation is cheaper inside the tunnel at all 15 depths where both groups contain at least two datasets.
- **Cutting at the entrance.** The pre-specified test of cutting at the entrance is now reported in Section 4.4 and Appendix B.4.1 (answer to Reviewer 2p99, Q2).

On Table 14, we agree with your reading. The comparison is post hoc and between datasets, some groups contain only two datasets, and datasets whose tunnel begins early may be easier to truncate for other reasons. The text says so, the table reports the group sizes, and we describe the result as an association, as in our limitations.

**Adaptation cost and zero-shot use.** See our answer to Q2: the adapter needs no labels, it transfers across datasets at mild truncation, and calibration costs one forward pass and about one minute of fitting for Chronos-2.

---

## Reviewer dHTD

### W1. Dataset-specific calibration (replaces your earlier future-tense draft)

---

We agree that dataset-specific calibration has a cost and is in tension with zero-shot use. To test whether it can be avoided, we ran leave-one-dataset-out transfer for Chronos-2 at L4, L6 and L9. A single adapter is trained on the other datasets (13, or 12 when the target is one of the two Electricity datasets) and evaluated on the unseen target, with no target-specific fitting or selection (Appendix B.4.2).

The transferred adapter raises median test MASE by +12.7%, +9.0% and +4.0% at L4, L6 and L9. The hard cut raises it by +37.0%, +20.1% and +17.1%, and dataset-specific adapters by +3.1%, +2.3% and +2.2%. The transferred adapter improves on the hard cut on 13 of 14 datasets at every depth, but not on LOOP Seattle, our only 5-minute dataset. Adding source datasets helps only up to a point: at L6, the mean increase over four predefined targets is +19.7%, +12.7%, +13.2% and +12.8% with 1, 4, 8 and 13 sources (Appendix B.4.3).

Zero-shot truncation is therefore viable at mild truncation (+4.0% at 1.28×), while at aggressive cuts calibration on unlabeled target windows remains clearly better. Calibration needs no labels. For Chronos-2, it costs one forward pass of the full model over 1,394 windows (about 1 s at batch size 256 on an A100) and about one minute of adapter fitting, which is amortized when the truncated model forecasts repeatedly on the same target. Choosing the depth is a separate question: the transfer experiment uses fixed depths, while choosing a depth per dataset, for example at its tunnel entrance, still requires target data. We have added both experiments and rewritten the limitation on cross-dataset transfer.

---

### W2. Modality-agnostic method

Leave as "repeated in Questions For Authors" (answered in Q2).

### W3. Prefix-only truncation

Leave as "repeated in Questions For Authors" (answered in Q1).

### Q1. Have you tried non-contiguous layer selection or skipping specific intermediate layers, rather than always truncating from the end? If so, how does the performance compare?

---

Yes. We ran this experiment for Chronos-2 on all 14 datasets (Appendix B.4.4). At matched numbers of retained blocks (4, 6 or 9 of 12), we compared the prefix with three masks fixed in advance: uniformly spaced blocks, early plus late blocks, and a contiguous middle window as a control. Retained blocks keep their order and weights, and every configuration receives the same label-free adapter and selection rule. Measured speedups at a matched number of blocks agree within 6%.

Median test MASE increase over the full model, aligned (hard cut in parentheses):

| Blocks kept | Prefix | Uniform | Early + late | Middle window |
|---|---|---|---|---|
| 4 | +3.1% (+37.0%) | +8.9% (+42.2%) | +10.1% (+31.3%) | +7.2% (+63.3%) |
| 6 | +2.3% (+20.1%) | +4.2% (+17.5%) | +7.6% (+23.1%) | +7.6% (+52.0%) |
| 9 | +2.2% (+17.1%) | +0.9% (+1.1%) | +0.9% (+2.1%) | +3.4% (+20.2%) |

The best strategy depends on how many blocks are kept:

- **4 of 12 blocks (our 2.43× operating point):** the prefix is best. It is better than the three alternatives on 10, 11 and 12 of 14 datasets, respectively, and worse on none (paired 95% CI).
- **6 of 12:** the prefix and uniform spacing are comparable (each better on 4 datasets).
- **9 of 12 (mild compression):** you are right. Skipping interior blocks while keeping the last ones is better (uniform spacing: 7 datasets better, 1 worse) and needs almost no adapter (+1.1% without one, against +17.1% for the prefix hard cut).

Removing single blocks without an adapter shows why:

- final block: +18.5%, worse on all 14 datasets;
- block 1: +5.2%;
- block 2: +2.2%;
- block 11: +1.5%;
- each of blocks 3 to 10: at most +0.4%.

Layer importance in Chronos-2 is thus non-monotonic, as you suggest: the first two blocks and the last two matter most. This matches our compatibility finding: the final block produces the states the pretrained head expects, and prefix truncation removes it and relies on the adapter to stand in.

These are four predefined masks on one model, not a search over masks, and our adapter corrects only the final retained state. We have added the experiment to the appendix and qualified the prefix-only design in the main text.

---

### Q2. What aspects of the proposed method are specific to time series? (your text, unchanged)

---

We agree that the general framework of layerwise probing, alignment, and truncation can be considered modality-agnostic. Nonetheless, in our study, these methods are tailored to the forecasting mechanisms of time-series foundation models (TSFMs).

Layerwise probing. Unlike standard autoregressive LLMs, which predict next-token distributions, TSFMs use specialized forecasting heads to produce multi-step predictions, often across multiple quantiles. The hidden states consumed by these heads also differ across architectures. In our experiments, we probe the exact states used by each model's native forecasting head: forecast-slot tokens in Chronos-2, the last context token in TimesFM-3, and the final input token of each forecasting pass in TiRex. Thus, our choice of intermediate representations depends on how each TSFM produces its forecasts, rather than on a generic representation-extraction procedure.

To demonstrate why this distinction matters, we have added an experiment comparing our architecture-aware readouts with generic content-mean pooling in Appendix B.1.5. The latter averages the context-token representations without accounting for the model's forecasting architecture. Across 14 datasets, content-mean pooling increases median final-block test loss by 27.2% for Chronos-2, 15.9% for TimesFM-3, and 20.9% for TiRex, relative to our architecture-aware readouts. We additionally compare readouts with the same head structure and parameter count, changing only which hidden states are averaged to separate the effect of token selection from that of the readout architecture. Even under this comparison, content-mean pooling increases median test loss by 11.1%, 15.9%, and 15.6%, respectively.

Moreover, generic pooling generally produces earlier 5% tunnel entrances. The median relative entrance depth shifts from 29% to 8% for Chronos-2 and from 79% to 21% for TiRex, while remaining at 5% for TimesFM-3. However, these earlier entrances do not necessarily indicate earlier recovery of accurate forecasts. Since each tunnel is defined relative to its own final-block readout, the higher forecasting losses of generic pooling lead to earlier apparent saturation. In fact, the final-block content-mean readout performs worse than our architecture-aware readout already at the latter's tunnel entrance on 13/14, 14/14, and 13/14 datasets, respectively. Therefore, simply pooling intermediate representations can give a different, and potentially misleading, picture of where forecasting information becomes recoverable.

Alignment and truncation. The architecture of the pretrained forecasting head also matters when translating intermediate representations into computational savings. Unlike standard next-token prediction heads, TSFM forecasting heads generate multi-step outputs and vary considerably across architectures, from a linear projection in TimesFM-3 to residual MLPs in Chronos-2 and TiRex. We show that recovering an accurate forecast from an intermediate representation does not necessarily mean that the model's pretrained forecasting head can use that representation effectively. We therefore investigate whether intermediate representations can be aligned with the pretrained forecasting heads, allowing us to truncate the model while preserving its original forecasting pathway. This alignment must account for the forecasting architecture of each TSFM.

Overall, while the underlying methods can be considered general, their application to TSFMs requires accounting for how each model produces its forecasts. Our results show that the choice of hidden states matters for layerwise probing, and that recovering accurate forecasts from intermediate representations does not necessarily mean they can be used directly by the pretrained forecasting head. Both considerations are important when studying whether and where TSFMs can be truncated without substantially affecting forecasting accuracy.

---
