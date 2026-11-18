# Main-text changes for the rebuttal revision (exact OLD -> NEW)

Every block below is copied verbatim from `three_models_paper/main.tex` before the revision (commit `ea4896f`, the snapshot of the submitted working tree) and after it. Section and appendix numbers are those of the revised PDF. `git diff ea4896f -- three_models_paper/main.tex` shows the same changes as a diff.

## Section 1 (Introduction), contribution 4: Tunnel stratification is post hoc

**Why.** Reviewer 96qN pointed out that the inside-versus-before comparison is post hoc and uses small groups. The claim is kept, but no longer called consistent.

**OLD**

```latex
\item \textbf{From recoverability to acceleration.} We physically truncate the models and measure the resulting performance-compute frontier. At matched cut depths, tunnel membership consistently stratifies truncation tolerance, linking the recoverability diagnostic to actual depth reduction. Aligned Chronos-2 remains within a median 3.1\% of the full model while retaining only 4 of 12 blocks and running $2.43\times$ faster; TiRex reaches a $1.71\times$ speedup with an 8.5\% median MASE increase while retaining 7 of 12 blocks, while TimesFM-3 exhibits more modest gains.
```

**NEW**

```latex
\item \textbf{From recoverability to acceleration.} We physically truncate the models and measure the resulting performance-compute frontier. At matched cut depths, tunnel membership stratifies truncation tolerance in a post hoc comparison, linking the recoverability diagnostic to actual depth reduction. Aligned Chronos-2 remains within a median 3.1\% of the full model while retaining only 4 of 12 blocks and running $2.43\times$ faster; TiRex reaches a $1.71\times$ speedup with an 8.5\% median MASE increase while retaining 7 of 12 blocks, while TimesFM-3 exhibits more modest gains.
```

## Section 3.3 (Hard cut and native-pathway alignment): How the TimesFM-3 adapter is fitted; what calibration needs

**Why.** States that TimesFM-3 uses a closed form shrunk toward zero (the correction of Appendix B.3.1), that the adapter needs only unlabeled target windows, and points to the transfer experiment (Reviewers 2p99, 96qN, dHTD).

**OLD**

```latex
At depth $\ell<L$, the \emph{hard cut} feeds the intermediate representation directly into the pretrained forecasting pathway, $g(h^{(\ell)})$. To test whether a simple representation mismatch explains the resulting degradation, we insert \textit{a label-free affine adapter},
$
A_\ell(h)=h+\Delta_\ell h+b_\ell,
$
initialized at the identity and fitted with the backbone frozen to minimize the squared error between $g(A_\ell(h^{(\ell)}))$ and the full model's own forecast $g(h^{(L)})$. We additionally retain the linear readout $q_\ell$ as a supervised reference for how much forecast information is recoverable from the same representation. A nonlinear adapter variant is reported in Appendix~\ref{app:alignment}.
```

**NEW**

```latex
At depth $\ell<L$, the \emph{hard cut} feeds the intermediate representation directly into the pretrained forecasting pathway, $g(h^{(\ell)})$. To test whether a simple representation mismatch explains the resulting degradation, we insert \textit{a label-free affine adapter},
$
A_\ell(h)=h+\Delta_\ell h+b_\ell,
$
initialized at the identity and fitted with the backbone frozen to minimize the squared error between $g(A_\ell(h^{(\ell)}))$ and the full model's own forecast $g(h^{(L)})$. Because TimesFM-3's head is linear, its adapter is fitted in closed form, with a ridge penalty that shrinks $A_\ell$ toward zero (Appendix~\ref{app:alignment}). The adapter needs only unlabeled windows of the target dataset; Appendix~\ref{sec:app_lodo} tests adapters fitted on other datasets instead. We additionally retain the linear readout $q_\ell$ as a supervised reference for how much forecast information is recoverable from the same representation. A nonlinear adapter variant is reported in Appendix~\ref{app:alignment}.
```

## Section 4.1, Table 1 caption: Low-skill L0 entrances marked

**Why.** Reviewer 96qN: four of the seven L0 entrances are pairs where no depth beats the constant forecast. They are now marked in Table 1 and explained in Appendix B.1.6.

**OLD**

```latex
  \caption{\textbf{Sustained 5\% forecasting-tunnel entrances for the Q=9 readout.} L0 denotes the representation before the first model block, while L1, L2, and so on denote the outputs of successive blocks. Entrances are selected from validation loss only.}
```

**NEW**

```latex
  \caption{\textbf{Sustained 5\% forecasting-tunnel entrances for the Q=9 readout.} L0 denotes the representation before the first model block, while L1, L2, and so on denote the outputs of successive blocks. Entrances are selected from validation loss only. $^\dagger$The final-block readout is within 5\% of the best constant forecast on validation (Appendix~\ref{sec:app_native_reference}).}
```

## Section 4.1, Table 1 rows: Daggers on M5 (all models) and TiRex on SZ Taxi

**Why.** Same change as above; only these two rows change.

**OLD**

```latex
  Uber TLC & L4 & L1 & L10 & SZ Taxi & L2 & L1 & L0 \\
  SG Carpark & L10 & L12 & L11 & M5 & L0 & L0 & L0 \\
```

**NEW**

```latex
  Uber TLC & L4 & L1 & L10 & SZ Taxi & L2 & L1 & L0$^\dagger$ \\
  SG Carpark & L10 & L12 & L11 & M5 & L0$^\dagger$ & L0$^\dagger$ & L0$^\dagger$ \\
```

## Section 4.1, second paragraph: What the tunnel's reference implies

**Why.** Reviewer 96qN, Q1: the tunnel is referenced to the final-block readout, which is itself 13% to 20% worse than the full model; referenced to the full model, a readout tunnel exists in only 6 of 42 pairs. This is now stated in the main text, with the analysis in Appendix B.1.6.

**OLD**

```latex
The validation-selected tunnel entrances also remain close to final-block performance on held-out test data. Across all 42 model-dataset pairs, the median test-loss increase at the selected entrance is 1.3\% (interquartile range (IQR): $-0.7$\% to 4.3\%), with a maximum increase of 9.8\%. Full held-out results are reported in Appendix~\ref{sec:app_heldout_recoverability}. We additionally repeat the analysis with a substantially lower-capacity, median-only ($Q=1$) readout and obtain highly consistent tunnel entrances relative to the $Q=9$ analysis (Appendix~\ref{sec:app_q1_robustness}). 
We additionally compare our architecture-aware readouts with content-mean pooling and find that the choice of hidden states affects both forecasting loss and the location of the tunnel (Appendix~\ref{sec:app_content_mean}).
```

**NEW**

```latex
The validation-selected tunnel entrances also remain close to final-block performance on held-out test data. Across all 42 model-dataset pairs, the median test-loss increase at the selected entrance is 1.3\% (interquartile range (IQR): $-0.7$\% to 4.3\%), with a maximum increase of 9.8\%. Full held-out results are reported in Appendix~\ref{sec:app_heldout_recoverability}. We additionally repeat the analysis with a substantially lower-capacity, median-only ($Q=1$) readout and obtain highly consistent tunnel entrances relative to the $Q=9$ analysis (Appendix~\ref{sec:app_q1_robustness}). 
We additionally compare our architecture-aware readouts with content-mean pooling and find that the choice of hidden states affects both forecasting loss and the location of the tunnel (Appendix~\ref{sec:app_content_mean}).
Because the tunnel is referenced to the final-block readout, it does not measure closeness to the full model: the final-block readout's validation loss is a median 12.7\% to 20.3\% above the full model's, and a tunnel referenced to the full model exists in only 6 of the 42 pairs (Appendix~\ref{sec:app_native_reference}). Near-native performance is the subject of Sections~\ref{sec:results_alignment} and~\ref{sec:results_acceleration}.
```

## Section 4.3, first paragraph: TimesFM-3: aligned pathway now beats the readout; the original fit overfit

**Why.** Phase 2g: with the closed-form adapter shrunk toward zero instead of toward the hard cut, aligned TimesFM-3 outperforms the supervised readout at the depths of Table 2, so the statement that the readout stays closer to the full model is withdrawn. The original +48% to +70% is attributed to overfitting and kept in Appendix B.3.1.

**OLD**

```latex
Using the hard cut and label-free affine adapter introduced in Section~\ref{sec:adapter}, Figure~\ref{fig:h3_main} shows a clear separation between forecast recoverability and native-pathway compatibility. On Electricity, the hard cut never stays within 5\% of the full model before the final block, while the affine adapter closes a median 73\%, over 99\%, and 82\% of the hard-cut gap across intermediate depths for Chronos-2, TimesFM-3, and TiRex, respectively; the particularly large value for TimesFM-3 reflects its extremely poor hard-cut baseline. For TimesFM-3, the same supervised readout $q_\ell$ used in Section~\ref{sec:results_recoverability} remains substantially closer to full-model performance than the aligned pathway on Electricity, and this distinction persists across datasets; for Chronos-2 and TiRex, however, the aligned pathway outperforms both the hard cut and the supervised probe at every depth reported in Table~\ref{tab:h3_depth}. A nonlinear bottleneck adapter changes little relative to the affine adapter for Chronos-2 and TiRex, with a median difference of only $-0.2$ percentage points in test MASE, whereas it performs substantially worse for TimesFM-3 (Appendix~\ref{app:compatibility}).
```

**NEW**

```latex
Using the hard cut and label-free affine adapter introduced in Section~\ref{sec:adapter}, Figure~\ref{fig:h3_main} shows a clear separation between forecast recoverability and native-pathway compatibility. On Electricity, the hard cut never stays within 5\% of the full model before the final block, while the affine adapter closes a median 73\%, over 99\%, and 82\% of the hard-cut gap across intermediate depths for Chronos-2, TimesFM-3, and TiRex, respectively; the particularly large value for TimesFM-3 reflects its extremely poor hard-cut baseline. For all three models, the aligned pathway outperforms both the hard cut and the supervised readout $q_\ell$ used in Section~\ref{sec:results_recoverability} at every depth reported in Table~\ref{tab:h3_depth}. For TimesFM-3, this requires regularizing its closed-form adapter toward zero rather than toward the hard cut: anchored at the hard cut, which is 35 to 160 times worse than the full model from L0 to L17, the adapter overfits its training windows and degrades by $+48\%$ to $+70\%$ at 25\% to 75\% of depth (Appendix~\ref{sec:app_timesfm3_ridge}). A nonlinear bottleneck adapter changes little relative to the affine adapter for Chronos-2 and TiRex, with a median difference of only $-0.2$ percentage points in test MASE, whereas it performs substantially worse for TimesFM-3 (Appendix~\ref{app:compatibility}).
```

## Section 4.3, second paragraph: TimesFM-3: size of the remaining gap

**Why.** Replaces the qualitative 'substantial residual gap' with the measured gap of the corrected adapter.

**OLD**

```latex
We conclude that \textit{recoverability precedes native-pathway compatibility in TSFMs}, and that lightweight alignment substantially improves compatibility across all three architectures. The extent of recovery is architecture-dependent: for Chronos-2 and TiRex, a depth-specific affine map can bring intermediate states close to native performance several blocks before the final one, while for TimesFM-3 it removes most of the large hard-cut degradation but leaves a substantial residual gap to the full model. 
```

**NEW**

```latex
We conclude that \textit{recoverability precedes native-pathway compatibility in TSFMs}, and that lightweight alignment substantially improves compatibility across all three architectures. The extent of recovery is architecture-dependent: for Chronos-2 and TiRex, a depth-specific affine map can bring intermediate states close to native performance several blocks before the final one, while for TimesFM-3 the aligned pathway stays about 10\% above the full model from L1 to L14 and comes within 5\% of it on most datasets only at L18.
```

## Figure 3 (fig:h3_main): Regenerated TimesFM-3 panel and caption

**Why.** New figure file with the zero-anchored TimesFM-3 adapter (Chronos-2 and TiRex panels unchanged); the caption names the adapter. The original file is kept.

**OLD**

```latex
  \includegraphics[width=0.9\textwidth]{figures/h3_main_electricity.pdf}
  \caption{\textbf{Native-pathway compatibility across depth on Electricity.} Test MASE increase over the full model for the hard cut, affine-aligned pathway, and supervised probe reference. Shading marks the forecasting tunnel; bands show 95\% cluster-bootstrap intervals. The y-axis is linear to $+20\%$ and log-scaled above; triangles mark clipped values. Complete results are in Appendix~\ref{app:compatibility}.}
```

**NEW**

```latex
  \includegraphics[width=0.9\textwidth]{figures/h3_main_electricity_rev.pdf}
  \caption{\textbf{Native-pathway compatibility across depth on Electricity.} Test MASE increase over the full model for the hard cut, affine-aligned pathway, and supervised probe reference. Shading marks the forecasting tunnel; bands show 95\% cluster-bootstrap intervals. The y-axis is linear to $+20\%$ and log-scaled above; triangles mark clipped values. The TimesFM-3 adapter is regularized toward zero (Appendix~\ref{sec:app_timesfm3_ridge}). Complete results are in Appendix~\ref{app:compatibility}.}
```

## Table 2 (tab:h3_depth), caption: Which TimesFM-3 adapter; original values given

**Why.** Transparency about the post hoc change: the caption reports the identity-anchored values that the table previously showed.

**OLD**

```latex
\caption{\textbf{Native-pathway compatibility across depth.} Median test MASE increase over the full model across 14 datasets at 25\%, 50\%, and 75\% of network depth, corresponding to L3/L6/L9 for Chronos-2 and TiRex and L5/L10/L15 for TimesFM-3. $\times$ denotes ratios to the full-model MASE.}
```

**NEW**

```latex
\caption{\textbf{Native-pathway compatibility across depth.} Median test MASE increase over the full model across 14 datasets at 25\%, 50\%, and 75\% of network depth, corresponding to L3/L6/L9 for Chronos-2 and TiRex and L5/L10/L15 for TimesFM-3. $\times$ denotes ratios to the full-model MASE. TimesFM-3 uses the zero-anchored adapter; anchored at the hard cut it gives $+69.8\%$, $+67.3\%$, and $+47.9\%$ (Appendix~\ref{sec:app_timesfm3_ridge}).}
```

## Table 2 (tab:h3_depth), TimesFM-3 aligned row: TimesFM-3 aligned at 25 / 50 / 75% of depth

**Why.** Corrected values (Phase 2g, zero anchor). Hard cut and readout columns unchanged.

**OLD**

```latex
& $+69.8\%$ & $+67.3\%$ & $+47.9\%$
```

**NEW**

```latex
& $+10.6\%$ & $+11.3\%$ & $+8.5\%$
```

## Section 4.4, first paragraph: TimesFM-3 accuracy and Chronos-2-small

**Why.** Adds the corrected TimesFM-3 accuracy (offline), and states plainly that Chronos-2-small is better at moderate speedups, by a larger margin at batch size 256, and where truncation is still useful (Reviewer 2p99's weakness).

**OLD**

```latex
\textit{How much does it cost to remove blocks?} Figure~\ref{fig:h4_frontier} shows the resulting \textit{performance-compute frontier}, obtained by physically truncating each model, verifying agreement with the offline aligned pathway in mean test MASE to within $10^{-4}$ relative error (Appendix~\ref{app:latency}), and measuring end-to-end speedup over the full model. Aligned Chronos-2 remains within a median 3.1\% of the full model from L11 through L4, where it runs $2.43\times$ faster. TiRex exhibits a more gradual tradeoff, reaching $1.71\times$ speedup at an 8.5\% median MASE increase when truncated after L7. TimesFM-3 provides more modest gains at batch size 1: even removing all blocks reaches only $1.38\times$, indicating that a substantial fraction of its end-to-end runtime lies outside the Transformer blocks (Table~\ref{tab:h4_operating_points}).  Chronos-2-small \citep{ansari2025chronos2univariateuniversalforecasting} provides a strong separately pretrained baseline, running $1.80\times$ faster at only $+0.8\%$ median MASE; at a comparable speed, aligned truncation after L6 reaches $1.78\times$ with a $+2.3\%$ increase, while avoiding the need to train a separate smaller model and providing multiple speed-performance tradeoffs from a single pretrained Chronos-2 checkpoint. Complete layer-by-layer performance and speedup results are reported in Appendix~\ref{app:truncation_results}.
```

**NEW**

```latex
\textit{How much does it cost to remove blocks?} Figure~\ref{fig:h4_frontier} shows the resulting \textit{performance-compute frontier}, obtained by physically truncating each model, verifying agreement with the offline aligned pathway in mean test MASE to within $10^{-4}$ relative error (Appendix~\ref{app:latency}), and measuring end-to-end speedup over the full model. Aligned Chronos-2 remains within a median 3.1\% of the full model from L11 through L4, where it runs $2.43\times$ faster. TiRex exhibits a more gradual tradeoff, reaching $1.71\times$ speedup at an 8.5\% median MASE increase when truncated after L7. TimesFM-3 provides more modest gains at batch size 1: even removing all blocks reaches only $1.38\times$, indicating that a substantial fraction of its end-to-end runtime lies outside the Transformer blocks (Table~\ref{tab:h4_operating_points}); its aligned pathway stays about 10\% above the full model from L1 to L14, with accuracy computed offline (Appendix~\ref{sec:app_timesfm3_ridge}). Chronos-2-small \citep{ansari2025chronos2univariateuniversalforecasting} provides a strong separately pretrained baseline, running $1.80\times$ faster at only $+0.8\%$ median MASE; at a comparable speed, aligned truncation after L6 reaches $1.78\times$ with a $+2.3\%$ increase. At batch size 256 its advantage is larger ($3.28\times$, against $3.10\times$ at $+4.4\%$ for aligned truncation after L3). Truncation is therefore most useful beyond Chronos-2-small's speed at batch size 1, for models without a smaller pretrained version, such as TiRex and TimesFM-3, and for obtaining several speed-performance tradeoffs from a single pretrained checkpoint without training a separate model. Complete layer-by-layer performance and speedup results are reported in Appendix~\ref{app:truncation_results}.
```

## Section 4.4, new paragraph: Transfer without target data; cost of calibration

**Why.** Reviewers 2p99, 96qN and dHTD: an adapter fitted on other datasets (Appendix B.4.2) and the effect of the number of sources (Appendix B.4.3), with the cost of dataset-specific calibration.

**OLD**

```latex
(none: new paragraph)
```

**NEW**

```latex
\textit{Does one adapter serve unseen datasets?} The adapters above are fitted per dataset on unlabeled windows; for Chronos-2, this takes one forward pass of the full model over 1,394 windows and about one minute of fitting on one A100. An adapter fitted only on the other datasets, without any data from the target, raises the median MASE increase from $+3.1\%$ to $+12.7\%$ after L4 and from $+2.2\%$ to $+4.0\%$ after L9, compared with $+37.0\%$ and $+17.1\%$ for the hard cut, and improves on the hard cut on 13 of 14 datasets at every depth; the exception is LOOP Seattle, our only 5-minute dataset (Appendix~\ref{sec:app_lodo}). Pooling more than about four source datasets brings little further gain (Appendix~\ref{sec:app_source_diversity}). Truncation without target-specific calibration is therefore viable at mild truncation, whereas aggressive truncation still benefits from calibration on the target dataset.
```

## Section 4.4, tunnel paragraph: Tunnel stratification: post hoc wording, TimesFM-3, pre-specified entrance test

**Why.** Reviewer 96qN: the inside-versus-before comparison is now described as post hoc with small groups. TimesFM-3's inside-tunnel degradation is updated (+40% to +68% was the overfit adapter; it is now about 10%), the count becomes 15 of 15, and the pre-specified test of cutting at the entrance is reported (Reviewer 2p99, Q2; Appendix B.4.1). The conclusion that recoverability alone does not ensure compatibility is kept, with the new evidence.

**OLD**

```latex
\textit{Does the forecasting tunnel identify where truncation is well tolerated?}
Because inference speed depends only on the cut depth, we compare datasets at the same cut according to whether that depth lies inside or before their validation-selected tunnel. Despite being defined only from layerwise readouts, without evaluating a truncated model, the tunnel consistently stratifies truncation performance. After L4, aligned Chronos-2 incurs a median $+2.2\%$ MASE increase on datasets for which L4 lies inside the tunnel, compared with $+14.2\%$ when it lies before the tunnel; after L6, the corresponding TiRex increases are $+4.4\%$ and $+12.3\%$. Truncation is cheaper inside the tunnel at all depths where both groups are present for Chronos-2 and TiRex, and at 14 of 15 such depths for TimesFM-3 (Table~\ref{tab:h4_tunnel_strata}). For TimesFM-3, however, substantial degradation remains even inside the tunnel, indicating that early recoverability alone does not ensure native-pathway compatibility. Thus, the forecasting tunnel identifies datasets that better tolerate a given cut, even though its entrance need not define the best truncation depth.
```

**NEW**

```latex
\textit{Does the forecasting tunnel identify where truncation is well tolerated?}
Because inference speed depends only on the cut depth, we compare datasets at the same cut according to whether that depth lies inside or before their validation-selected tunnel. Despite being defined only from layerwise readouts, without evaluating a truncated model, the tunnel stratifies truncation performance in this post hoc comparison. After L4, aligned Chronos-2 incurs a median $+2.2\%$ MASE increase on datasets for which L4 lies inside the tunnel, compared with $+14.2\%$ when it lies before the tunnel; after L6, the corresponding TiRex increases are $+4.4\%$ and $+12.3\%$. Truncation is cheaper inside the tunnel at every depth where both groups contain at least two datasets, for all three models (Table~\ref{tab:h4_tunnel_strata}), although some groups contain only two datasets and datasets whose tunnel begins early may be easier to truncate for other reasons. For TimesFM-3, the aligned pathway remains about 10\% above the full model inside the tunnel from L1 to L14, even though its tunnel begins at a median 5\% of depth, so early recoverability alone does not ensure native-pathway compatibility. Cutting each dataset at its own tunnel entrance, a rule we fixed before computing the truncation results, keeps the aligned model within 5\% of the full model on 7 of 14 datasets for Chronos-2 and 5 of 12 for TiRex, and roughly halves the worst case of a single fixed cut at a similar speed (Appendix~\ref{sec:app_entrance_rule}). Thus, the forecasting tunnel identifies datasets that better tolerate a given cut, even though its entrance need not define the best truncation depth.
```

## Table 3 (tab:h4_operating_points), caption: Which TimesFM-3 adapter

**Why.** The TimesFM-3 operating points now use the zero-anchored adapter, with offline accuracy.

**OLD**

```latex
Median test MASE increase across 14 datasets and end-to-end speedup at batch size 1. For TimesFM-3 at L19, the adapter overhead slightly exceeds the saving from removing one block. Full layer-by-layer results are reported in Appendix~\ref{app:truncation_results}.}
```

**NEW**

```latex
Median test MASE increase across 14 datasets and end-to-end speedup at batch size 1. For TimesFM-3 at L19, the adapter overhead slightly exceeds the saving from removing one block. TimesFM-3 uses the zero-anchored adapter, whose accuracy is computed offline (Appendix~\ref{sec:app_timesfm3_ridge}). Full layer-by-layer results are reported in Appendix~\ref{app:truncation_results}.}
```

## Table 3 (tab:h4_operating_points), TimesFM-3 row: TimesFM-3 operating points

**Why.** Accuracy at L19 / L10 / L0 with the corrected adapter; speedups unchanged (same adapter shape).

**OLD**

```latex
& L19 & $+5.4\%$ & $0.99\times$
& L10 & $+67.3\%$ & $1.19\times$
& L0 & $+31.3\%$ & $1.38\times$ \\
```

**NEW**

```latex
& L19 & $+1.3\%$ & $0.99\times$
& L10 & $+11.3\%$ & $1.19\times$
& L0 & $+24.3\%$ & $1.38\times$ \\
```

## Figure 4 (fig:h4_frontier): Regenerated TimesFM-3 frontier and caption

**Why.** New figure file; only the TimesFM-3 aligned curve changes. The original file is kept.

**OLD**

```latex
  \includegraphics[width=0.9\textwidth]{figures/h4_frontier.pdf}
  \caption{\textbf{Performance-compute frontier of truncation.} Median test MASE increase across 14 datasets versus end-to-end speedup at batch size 1. Points denote aligned cut depths; the star is the full model and the diamond is Chronos-2-small; bands show interquartile ranges.}
```

**NEW**

```latex
  \includegraphics[width=0.9\textwidth]{figures/h4_frontier_rev.pdf}
  \caption{\textbf{Performance-compute frontier of truncation.} Median test MASE increase across 14 datasets versus end-to-end speedup at batch size 1. Points denote aligned cut depths; the star is the full model and the diamond is Chronos-2-small; bands show interquartile ranges. TimesFM-3 uses the zero-anchored adapter (Appendix~\ref{sec:app_timesfm3_ridge}).}
```

## Section 5, second paragraph: TimesFM-3 as the complementary case

**Why.** The conclusion that recoverability does not guarantee compatibility is kept, but is now supported by the corrected numbers (tunnel at a median 5% of depth, aligned pathway within 5% only at L18) instead of the overfit adapter's degradation.

**OLD**

```latex
Across three state-of-the-art TSFMs and 14 datasets spanning five sampling frequencies, the forecasting tunnel begins before the final block in 40 of 42 model-dataset pairs at our primary 5\% tolerance, and remains pre-final in 36 of 42 pairs even under the stricter 2\% tolerance. Representation geometry nevertheless continues to evolve inside the tunnel, showing that forecasting saturation need not coincide with geometric stabilization. Direct reuse of intermediate states through the pretrained forecasting pathway rarely succeeds, whereas lightweight affine alignment substantially improves compatibility across all three architectures. Physical truncation then turns this improved compatibility into measurable inference savings: Chronos-2 remains within a median 3.1\% of the full model after retaining only 4 of its 12 blocks while running $2.43\times$ faster, and TiRex reaches $1.71\times$ speedup with an 8.5\% median MASE increase while retaining 7 of 12 blocks. Importantly, at matched cut depths, datasets for which the cut lies inside the forecasting tunnel generally tolerate truncation better, directly linking early recoverability to practical depth reduction. TimesFM-3 provides the complementary case: substantial degradation can remain even inside the tunnel, showing that recoverability alone does not guarantee native-pathway compatibility. Its batch-size-1 speedups are also modest, with truncation before the first Transformer block reaching only $1.38\times$, consistent with a substantial fraction of end-to-end runtime lying outside the Transformer blocks. Chronos-2-small remains a strong separately pretrained baseline, while truncation provides multiple post-hoc operating points from a single pretrained checkpoint.
```

**NEW**

```latex
Across three state-of-the-art TSFMs and 14 datasets spanning five sampling frequencies, the forecasting tunnel begins before the final block in 40 of 42 model-dataset pairs at our primary 5\% tolerance, and remains pre-final in 36 of 42 pairs even under the stricter 2\% tolerance. Representation geometry nevertheless continues to evolve inside the tunnel, showing that forecasting saturation need not coincide with geometric stabilization. Direct reuse of intermediate states through the pretrained forecasting pathway rarely succeeds, whereas lightweight affine alignment substantially improves compatibility across all three architectures. Physical truncation then turns this improved compatibility into measurable inference savings: Chronos-2 remains within a median 3.1\% of the full model after retaining only 4 of its 12 blocks while running $2.43\times$ faster, and TiRex reaches $1.71\times$ speedup with an 8.5\% median MASE increase while retaining 7 of 12 blocks. Importantly, at matched cut depths, datasets for which the cut lies inside the forecasting tunnel generally tolerate truncation better, directly linking early recoverability to practical depth reduction. TimesFM-3 provides the complementary case: its tunnel begins at a median 5\% of depth, but its aligned pathway stays about 10\% above the full model up to L14 and comes within 5\% of it on most datasets only at L18, showing that recoverability alone does not guarantee native-pathway compatibility. Its batch-size-1 speedups are also modest, with truncation before the first Transformer block reaching only $1.38\times$, consistent with a substantial fraction of end-to-end runtime lying outside the Transformer blocks. Chronos-2-small remains a strong separately pretrained baseline, while truncation provides multiple post-hoc operating points from a single pretrained checkpoint.
```

## Section 5, limitations: TimesFM-3 gap, transfer, prefix-only truncation

**Why.** The open question on cross-dataset transfer is answered (Appendices B.4.2 and B.4.3) and replaced by its result; the TimesFM-3 limitation is updated to the corrected adapter (about 10% gap, failure on BOOM and Wind Farms, offline evaluation); the prefix-only design is qualified (Reviewer dHTD, Appendix B.4.4).

**OLD**

```latex
Our readouts and adapters are deliberately simple, making our estimates of recoverability and compatibility conservative with respect to more expressive mappings. For TimesFM-3, the affine adapter leaves a substantial gap to the full model, and richer alignment, for example over the full token sequence, may close more of it. Adapters are also fitted per dataset; whether alignment transfers across datasets, allowing a single truncated model at each depth, remains open. With 14 datasets, we can establish an association between forecast recoverability and truncation tolerance but not determine how reliably it generalizes or supports cut-depth selection. Larger benchmarks such as GIFT-Eval \citep{aksu2024giftevalbenchmarkgeneraltime} and fev-bench \citep{shchur2026fevbenchrealisticbenchmarktime} could test whether the forecasting tunnel can serve as a reliable indicator of truncation tolerance and guide cut-depth selection. Several datasets in our study are documented pretraining data for at least one model (Appendix~\ref{app:provenance}), so frozen-backbone experiments cannot fully separate generalizable structure from effects of pretraining exposure; extending the analysis to fev-bench, which is held out from the pretraining of Chronos-2 and TimesFM-3, would provide a cleaner test. Finally, we study univariate forecasting at a single context length and horizon, with speed measured on one GPU type; multivariate and covariate-aware forecasting, other horizons and context lengths, and additional hardware remain important directions for future work.
```

**NEW**

```latex
Our readouts and adapters are deliberately simple, making our estimates of recoverability and compatibility conservative with respect to more expressive mappings. For TimesFM-3, the affine adapter still leaves a gap of about 10\% to the full model at intermediate depths, remains far from it on BOOM and Wind Farms, where none of our affine fits reproduces the full model's forecasts on validation windows, and is evaluated offline (Appendix~\ref{sec:app_timesfm3_ridge}); richer alignment, for example over the full token sequence, may close more of the gap. Adapters are fitted per dataset on unlabeled windows. An adapter fitted on other datasets transfers well enough for mild truncation, but not for aggressive truncation, and fails on LOOP Seattle (Appendices~\ref{sec:app_lodo} and~\ref{sec:app_source_diversity}), so a single truncated model at each depth for all datasets remains limited to mild truncation. We also truncate contiguous prefixes only; for Chronos-2 at mild compression, skipping interior blocks while keeping the final ones is better (Appendix~\ref{sec:app_noncontig}). With 14 datasets, we can establish an association between forecast recoverability and truncation tolerance but not determine how reliably it generalizes or supports cut-depth selection. Larger benchmarks such as GIFT-Eval \citep{aksu2024giftevalbenchmarkgeneraltime} and fev-bench \citep{shchur2026fevbenchrealisticbenchmarktime} could test whether the forecasting tunnel can serve as a reliable indicator of truncation tolerance and guide cut-depth selection. Several datasets in our study are documented pretraining data for at least one model (Appendix~\ref{app:provenance}), so frozen-backbone experiments cannot fully separate generalizable structure from effects of pretraining exposure; extending the analysis to fev-bench, which is held out from the pretraining of Chronos-2 and TimesFM-3, would provide a cleaner test. Finally, we study univariate forecasting at a single context length and horizon, with speed measured on one GPU type; multivariate and covariate-aware forecasting, other horizons and context lengths, and additional hardware remain important directions for future work.
```

## AI use statement: AI assistance during the review period

**Why.** Reflects that generative AI tools assisted in analyzing the original TimesFM-3 fits, implementing and running the review-period experiments, and drafting the new appendices and main-text revisions. Please check that this matches your own account before submitting.

**OLD**

```latex
In this work, we used generative AI tools for writing assistance, such as improving word choice, correcting grammar, and improving the flow and transitions between sentences. Generative AI tools also assisted in drafting the Appendix sections describing the latency protocol and compute accounting; the underlying experimental procedures, measurements, and reported values were determined and verified by the authors, and all generated text was reviewed and revised by the authors. We also used generative AI tools for coding, with their behavior constrained by author-written instruction files. All AI-generated code was reviewed and debugged by the authors and validated through tests written both by the authors and with AI assistance. Additionally, generative AI tools were used to implement plotting and table-generation code for the paper, under the authors' review. Generative AI tools were also used to assist in writing scripts for collecting latency and compute information, such as device metadata, runtime records, and speedup measurements; all such code was reviewed by the authors.

We have not used generative AI tools to generate research ideas, conduct the literature review, design methods, select time-series foundation models or datasets, design experiments, or draft sections of the main paper. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.
```

**NEW**

```latex
In this work, we used generative AI tools for writing assistance, such as improving word choice, correcting grammar, and improving the flow and transitions between sentences. Generative AI tools also assisted in drafting the Appendix sections describing the latency protocol and compute accounting; the underlying experimental procedures, measurements, and reported values were determined and verified by the authors, and all generated text was reviewed and revised by the authors. We also used generative AI tools for coding, with their behavior constrained by author-written instruction files. All AI-generated code was reviewed and debugged by the authors and validated through tests written both by the authors and with AI assistance. Additionally, generative AI tools were used to implement plotting and table-generation code for the paper, under the authors' review. Generative AI tools were also used to assist in writing scripts for collecting latency and compute information, such as device metadata, runtime records, and speedup measurements; all such code was reviewed by the authors. During the review period, generative AI tools assisted in analyzing the fits of the original TimesFM-3 adapter, in implementing and running the additional experiments, and in drafting the appendices added in response to the reviews (Appendices~\ref{sec:app_native_reference}, \ref{sec:app_timesfm3_ridge}, and~\ref{sec:app_entrance_rule} to~\ref{sec:app_heldout_pairs}) and the corresponding revisions of the main text; all such text was reviewed and revised by the authors.

For the originally submitted version, we have not used generative AI tools to generate research ideas, conduct the literature review, design methods, select time-series foundation models or datasets, design experiments, or draft sections of the main paper. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.
```

## Reproducibility statement: Order of analysis disclosed

**Why.** States that the zero anchor was introduced after the identity-anchored test results, and which review-period analyses are post hoc.

**OLD**

```latex
The tunnel definition and all selection rules use validation data only and were fixed before the corresponding test results were computed. The comparison of truncation inside and before the tunnel (Section~\ref{sec:results_acceleration}) is a post hoc analysis. A pre-specified per-window equivalence check between the offline aligned pathway and the physically truncated implementation was not satisfied at every operating point; however, mean test MASE agrees to within $10^{-4}$ relative error. Appendix~\ref{app:latency} reports this check together with the latency protocol. Appendix~\ref{app:compute} lists the hardware, software, and compute.
```

**NEW**

```latex
The tunnel definition and all selection rules use validation data only and were fixed before the corresponding test results were computed. The comparison of truncation inside and before the tunnel (Section~\ref{sec:results_acceleration}) is a post hoc analysis. The zero-anchored TimesFM-3 adapter was introduced after the test results of the original, identity-anchored adapter had been computed; both are reported (Appendix~\ref{sec:app_timesfm3_ridge}). The analyses added during the review period (Appendices~\ref{sec:app_content_mean}, \ref{sec:app_native_reference}, and~\ref{sec:app_lodo} to~\ref{sec:app_heldout_pairs}) are post hoc, apart from the test of cutting at the tunnel entrance (Appendix~\ref{sec:app_entrance_rule}), which was specified before the truncation results; their own targets, block selections, and selection rules were fixed before their results were computed. A pre-specified per-window equivalence check between the offline aligned pathway and the physically truncated implementation was not satisfied at every operating point; however, mean test MASE agrees to within $10^{-4}$ relative error. Appendix~\ref{app:latency} reports this check together with the latency protocol. Appendix~\ref{app:compute} lists the hardware, software, and compute.
```

## Appendix A.7 (Alignment training), TimesFM-3 paragraph and Equation 7: The closed form now shrinks toward zero

**Why.** Method description of the adapter used in the revision; the submitted variant is named and pointed to Appendix B.3.1.

**OLD**

```latex
\paragraph{TimesFM-3.} Its head is a single linear map $W$, so the objective reduces to a ridge
problem in the metric $W^\top W$ with a closed-form solution:
\begin{equation}
\min_{\Delta_\ell,\,b_\ell}\;\sum_{i}\bigl\|(I+\Delta_\ell)h_i^{(\ell)}+b_\ell-h_i^{(L)}\bigr\|_{W^\top W}^2
+\lambda\,\|\Delta_\ell\|_F^2,
\label{eq:adapter_ridge}
\end{equation}
with $b_\ell$ unpenalized and $\lambda=\kappa\,g_{\max}m_{\max}$, where $g_{\max}$ and $m_{\max}$
are the largest eigenvalues of the centred training Gram matrix and of $W^\top W$, and
$\kappa\in\{10^{-12},10^{-11},\dots,10^{2}\}$. The hard cut is an explicit candidate; the candidate
with the lowest validation error in the same metric is kept, with ties going to the larger
$\lambda$.
```

**NEW**

```latex
\paragraph{TimesFM-3.} Its head is a single linear map $W$, so the objective reduces to a ridge
problem in the metric $W^\top W$ with a closed-form solution:
\begin{equation}
\min_{\Delta_\ell,\,b_\ell}\;\sum_{i}\bigl\|(I+\Delta_\ell)h_i^{(\ell)}+b_\ell-h_i^{(L)}\bigr\|_{W^\top W}^2
+\lambda\,\|I+\Delta_\ell\|_F^2,
\label{eq:adapter_ridge}
\end{equation}
with $b_\ell$ unpenalized and $\lambda=\kappa\,g_{\max}m_{\max}$, where $g_{\max}$ and $m_{\max}$
are the largest eigenvalues of the centred training Gram matrix and of $W^\top W$, and
$\kappa\in\{10^{-12},10^{-11},\dots,10^{2}\}$. The penalty shrinks the linear map
$A_\ell=I+\Delta_\ell$ toward zero; its limit, a constant forecast from the mean full-model state, is
an explicit candidate. The candidate with the lowest validation error in the same metric is kept,
with ties going to the larger $\lambda$. The submitted version of this paper penalized
$\lambda\|\Delta_\ell\|_F^2$ instead, shrinking toward the hard cut, with the hard cut as the
explicit candidate; for TimesFM-3 this overfits, and Appendix~\ref{sec:app_timesfm3_ridge} compares
the two.
```

## Appendix A.7, nonlinear variant: Training and validation R2 of both closed forms

**Why.** The R2 sentence described only the identity-anchored adapter; it now adds the zero-anchored fit's R2 on Electricity and states which adapter is used.

**OLD**

```latex
\paragraph{Nonlinear variant.} The nonlinear variant adds a bottleneck branch,
$A_\ell(h)=h+\Delta_\ell h+b_\ell+W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(h)+c)$, where $\mathrm{LN}$ is
a layer normalization without learned parameters, $W_1\in\mathbb{R}^{64\times d}$ is initialized
randomly with a fixed seed and $W_2\in\mathbb{R}^{d\times 64}$ at zero, so the branch contributes
nothing at initialization. It uses the same objective and iterative protocol for all three models,
with weight decay also on $W_1$ and $W_2$. We tested it on Electricity at 25, 50 and 75\% of depth
under an adoption rule fixed in advance and evaluated on validation data only; the rule was not met,
and the full run on all 14 datasets agrees (Appendix~\ref{app:compatibility}). For TimesFM-3, the
iterative fit, with or without the branch, does not fit even the training windows (on Electricity,
training $R^2$ between $-314$ and $-6$ against the full model's forecast), whereas the closed-form
affine adapter fits them almost exactly ($R^2\approx0.997$) but not the validation windows ($R^2$
between $-3.0$ and $-0.6$). We therefore use the affine adapter for all results.
```

**NEW**

```latex
\paragraph{Nonlinear variant.} The nonlinear variant adds a bottleneck branch,
$A_\ell(h)=h+\Delta_\ell h+b_\ell+W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(h)+c)$, where $\mathrm{LN}$ is
a layer normalization without learned parameters, $W_1\in\mathbb{R}^{64\times d}$ is initialized
randomly with a fixed seed and $W_2\in\mathbb{R}^{d\times 64}$ at zero, so the branch contributes
nothing at initialization. It uses the same objective and iterative protocol for all three models,
with weight decay also on $W_1$ and $W_2$. We tested it on Electricity at 25, 50 and 75\% of depth
under an adoption rule fixed in advance and evaluated on validation data only; the rule was not met,
and the full run on all 14 datasets agrees (Appendix~\ref{app:compatibility}). For TimesFM-3, the
iterative fit, with or without the branch, does not fit even the training windows (on Electricity,
training $R^2$ between $-314$ and $-6$ against the full model's forecast). The closed-form affine
adapter anchored at the hard cut fits them almost exactly ($R^2\approx0.997$) but not the validation
windows ($R^2$ between $-3.0$ and $-0.6$), whereas the zero-anchored closed form of
Equation~\ref{eq:adapter_ridge} reaches training $R^2$ between $0.96$ and $0.99$ and validation $R^2$
between $0.51$ and $0.82$. We therefore use the zero-anchored closed-form adapter for all TimesFM-3
results.
```

## Appendix A.9 (Latency protocol), physical versus offline: Scope of the physical check

**Why.** The 419 physically evaluated configurations used the submitted adapters; the corrected TimesFM-3 adapters are checked separately in Appendix B.3.1.

**OLD**

```latex
\paragraph{Physical versus offline truncation.}
Forecasting performance along the frontier (Figure~\ref{fig:h4_frontier} and
Table~\ref{tab:h4_frontier_full}) is computed by applying the aligned pathway to the cached
representations of each depth, while speed is measured on physically truncated models. We verify
that both describe the same model. At every depth of every model, the physically truncated network
reproduces the full model's state at that depth bitwise, on synthetic inputs and on 256 Electricity
windows against the cached features; the removed blocks are never executed; and the physically
truncated aligned model reproduces the offline aligned forecast within a relative tolerance of
$10^{-4}$. In addition, 419 truncated configurations, covering every model-dataset pair, were
evaluated physically on the complete test splits; their test MASE agrees with the offline pathway
to within $7.9\times10^{-5}$ relative, so no reported result depends on which pipeline is used.
Individual forecast values agree less tightly. For TiRex, even the unmodified model differs between
the two pipelines by up to 11\% of the mean forecast magnitude on single values, consistent with its
bfloat16 recurrence; for TimesFM-3, the aligned pathway differs by up to 5\% while the unmodified
model agrees. The element-wise check we had specified in advance for these test-split evaluations
(relative tolerance $10^{-4}$) therefore rejected 96 of the 419 configurations. We replaced it,
uniformly for all models, by the requirement that test MASE agree within $10^{-4}$ relative, which
every configuration satisfies.
```

**NEW**

```latex
\paragraph{Physical versus offline truncation.}
Forecasting performance along the frontier (Figure~\ref{fig:h4_frontier} and
Table~\ref{tab:h4_frontier_full}) is computed by applying the aligned pathway to the cached
representations of each depth, while speed is measured on physically truncated models. We verify
that both describe the same model. At every depth of every model, the physically truncated network
reproduces the full model's state at that depth bitwise, on synthetic inputs and on 256 Electricity
windows against the cached features; the removed blocks are never executed; and the physically
truncated aligned model reproduces the offline aligned forecast within a relative tolerance of
$10^{-4}$. In addition, 419 truncated configurations, covering every model-dataset pair, were
evaluated physically on the complete test splits; their test MASE agrees with the offline pathway
to within $7.9\times10^{-5}$ relative, so no reported result depends on which pipeline is used.
Individual forecast values agree less tightly. For TiRex, even the unmodified model differs between
the two pipelines by up to 11\% of the mean forecast magnitude on single values, consistent with its
bfloat16 recurrence; for TimesFM-3, the aligned pathway differs by up to 5\% while the unmodified
model agrees. The element-wise check we had specified in advance for these test-split evaluations
(relative tolerance $10^{-4}$) therefore rejected 96 of the 419 configurations. We replaced it,
uniformly for all models, by the requirement that test MASE agree within $10^{-4}$ relative, which
every configuration satisfies. These checks cover the adapters of the submitted version; the
zero-anchored TimesFM-3 adapters have the same shape and therefore the same latency, and their
physical check is reported in Appendix~\ref{sec:app_timesfm3_ridge}.
```

## Appendix A.10 (Compute): Compute of the review-period experiments

**Why.** Adds the 20.2 A100-hours of the review-period GPU jobs (Narval jobs 9229954, 9229962, 9229964, 9253740).

**OLD**

```latex
All experiments ran on a shared academic GPU cluster. Each GPU job used one NVIDIA A100 (40\,GB);
building windows and computing representation geometry need no GPU. The software stack was
Python~3.11, PyTorch~2.12 with CUDA~13.2, \texttt{chronos-forecasting}~2.3.1, \texttt{timesfm}~3.0.2
and \texttt{tirex-ts}~1.4.2. The pretrained backbones are never trained, so all computation is
inference plus the fitting of small readouts and adapters. Table~\ref{tab:compute} reports GPU time
by stage. Over half of it is latency measurement; the probing analyses themselves take under
3~GPU-hours. The nonlinear adapter, which we did not adopt, accounts for 18.4 of the 48.9
GPU-hours.
```

**NEW**

```latex
All experiments ran on a shared academic GPU cluster. Each GPU job used one NVIDIA A100 (40\,GB);
building windows and computing representation geometry need no GPU. The software stack was
Python~3.11, PyTorch~2.12 with CUDA~13.2, \texttt{chronos-forecasting}~2.3.1, \texttt{timesfm}~3.0.2
and \texttt{tirex-ts}~1.4.2. The pretrained backbones are never trained, so all computation is
inference plus the fitting of small readouts and adapters. Table~\ref{tab:compute} reports GPU time
by stage. Over half of it is latency measurement; the probing analyses themselves take under
3~GPU-hours. The nonlinear adapter, which we did not adopt, accounts for 18.4 of the 48.9
GPU-hours. The experiments added during the review period used a further 20.2 A100-hours for
Appendices~\ref{sec:app_content_mean}, \ref{sec:app_lodo}, \ref{sec:app_source_diversity},
and~\ref{sec:app_noncontig}, not counting duplicate runs on another cluster whose results we do not
use; the closed-form TimesFM-3 fits of Appendix~\ref{sec:app_timesfm3_ridge} and their physical check
ran on CPUs.
```

## Appendix B.1.5 (content-mean readouts): Moved to its own file, unchanged

**Why.** The whole subsubsection (text, Tables 11 and 12 of the pre-revision build, Figure 8, all labels) moved verbatim to appendices/content_mean_readouts.tex; diff-checked identical. The new Appendix B.1.6 follows it.

**OLD**

```latex
\subsubsection{Architecture-aware versus generic content-mean readouts}
\label{sec:app_content_mean}
... (162 lines, unchanged) ...
\label{fig:app_content_mean_curves}
\end{figure}
```

**NEW**

```latex
\input{appendices/content_mean_readouts}

\clearpage
\input{appendices/native_reference_tunnel}
```

## Appendix B.3, depth profiles: TimesFM-3 depth profile with the corrected adapter

**Why.** Replaces the identity-anchored profile with the corrected one and removes the speculative remark about the missing final normalization, which was offered as a possible explanation of a pattern that turned out to be produced by the fit.

**OLD**

```latex
\paragraph{Depth profiles.} Aligned Chronos-2 has a median test MASE increase between $+0.4\%$
(L11) and $+4.4\%$ (L3) at every depth from L3 onwards. For TimesFM-3, the aligned pathway stays
between roughly $+30\%$ and $+82\%$ up to L17 and comes within 15\% only at L18 ($+14.3\%$) and L19
($+5.4\%$). TimesFM-3's head is linear and reads the residual stream without a final
normalization, which is consistent with, but not shown to cause, this behaviour.
```

**NEW**

```latex
\paragraph{Depth profiles.} Aligned Chronos-2 has a median test MASE increase between $+0.4\%$
(L11) and $+4.4\%$ (L3) at every depth from L3 onwards. For TimesFM-3, the zero-anchored aligned
pathway stays between $+6\%$ and $+12\%$ from L1 to L17 and reaches $+3.9\%$ at L18 and $+1.3\%$ at
L19; the identity-anchored adapter of the submitted version stayed between roughly $+30\%$ and
$+82\%$ up to L17 (Appendix~\ref{sec:app_timesfm3_ridge}).
```

## Table tab:h3_depth_nonlinear, caption and TimesFM-3 rows: Affine column for TimesFM-3

**Why.** The affine TimesFM-3 column now uses the zero-anchored closed form; datasets within 5% in parentheses.

**OLD**

```latex
  5\% in parentheses. $^\dagger$Iterative fit; on Electricity it does not reproduce the full
  model's forecasts even on the training windows (training $R^2<0$).}
...
  TimesFM-3 & L5 (25\%)  & $+69.8\%$ (0) & $7.1\times^\dagger$ (0)  \\
            & L10 (50\%) & $+67.3\%$ (0) & $14.5\times^\dagger$ (0) \\
            & L15 (75\%) & $+47.9\%$ (0) & $20.5\times^\dagger$ (0) \\
```

**NEW**

```latex
  5\% in parentheses. TimesFM-3 affine: the zero-anchored closed form
  (Appendix~\ref{sec:app_timesfm3_ridge}). $^\dagger$Iterative fit; on Electricity it does not
  reproduce the full model's forecasts even on the training windows (training $R^2<0$).}
...
  TimesFM-3 & L5 (25\%)  & $+10.6\%$ (4) & $7.1\times^\dagger$ (0)  \\
            & L10 (50\%) & $+11.3\%$ (2) & $14.5\times^\dagger$ (0) \\
            & L15 (75\%) & $+8.5\%$ (3) & $20.5\times^\dagger$ (0) \\
```

## Figure fig:app_h3_depth_timesfm3: Regenerated, both anchors shown

**Why.** New figure file with the zero-anchored aligned pathway and the identity-anchored fit as a dotted line. The original file is kept.

**OLD**

```latex
  \includegraphics[width=\textwidth]{figures/h3_depth_timesfm3.pdf}
  \caption{\textbf{TimesFM-3: native-pathway compatibility across depth} on all 14 datasets, in
  the format of Figure~\ref{fig:app_h3_depth_chronos2}. A forecast-trained adapter is not shown:
  with TimesFM-3's linear head it spans exactly the function class of the probe head.}
```

**NEW**

```latex
  \includegraphics[width=\textwidth]{figures/h3_depth_timesfm3_rev.pdf}
  \caption{\textbf{TimesFM-3: native-pathway compatibility across depth} on all 14 datasets, in
  the format of Figure~\ref{fig:app_h3_depth_chronos2}. The aligned pathway uses the zero-anchored
  adapter; the dotted line is the identity-anchored adapter of the submitted version
  (Appendix~\ref{sec:app_timesfm3_ridge}). A forecast-trained adapter is not shown: with TimesFM-3's
  linear head it spans exactly the function class of the probe head.}
```

## Appendix B.3, new subsection: B.3.1 Regularization of the TimesFM-3 adapter

**Why.** Inserted after the TiRex depth figure. The \clearpage (also before B.1.6 and B.4.1) flushes queued floats so that each table and figure appears next to its own section; it changes page breaks only.

**OLD**

```latex
(none)
```

**NEW**

```latex
\clearpage
\input{appendices/timesfm3_zero_anchor}
```

## Appendix B.4, throughput paragraph: TimesFM-3 throughput operating points

**Why.** The sentence 'Its forecasting degradation is unchanged, so the larger throughput gains still come at the cost reported' referred to the overfit adapter; it now gives the corrected operating points at batch size 256.

**OLD**

```latex
\paragraph{Throughput at batch size 256.} For Chronos-2 and TiRex, speedups at batch size 256
closely track those at batch size 1; for example, Chronos-2 after L4 runs $2.43\times$ and
$2.51\times$ faster, and TiRex after L6 $1.99\times$ and $1.96\times$. TimesFM-3 differs: its
speedup at batch size 1 never exceeds $1.43\times$, whereas at batch size 256 it grows steadily
with the number of removed blocks, to $2.70\times$ after L5 and $5.94\times$ with no block retained.
This is consistent with a fixed per-call cost outside the backbone that dominates at batch size 1
(Appendix~\ref{app:latency}). Its forecasting degradation is unchanged, so the larger throughput
gains still come at the cost reported in Table~\ref{tab:h4_frontier_full}. Chronos-2-small also
benefits more from batching ($1.80\times$ at batch size 1, $3.28\times$ at 256).
```

**NEW**

```latex
\paragraph{Throughput at batch size 256.} For Chronos-2 and TiRex, speedups at batch size 256
closely track those at batch size 1; for example, Chronos-2 after L4 runs $2.43\times$ and
$2.51\times$ faster, and TiRex after L6 $1.99\times$ and $1.96\times$. TimesFM-3 differs: its
speedup at batch size 1 never exceeds $1.43\times$, whereas at batch size 256 it grows steadily
with the number of removed blocks, to $2.70\times$ after L5 and $5.94\times$ with no block retained.
This is consistent with a fixed per-call cost outside the backbone that dominates at batch size 1
(Appendix~\ref{app:latency}). Its forecasting degradation does not depend on the batch size, so
with the zero-anchored adapter truncation after L1 runs $4.65\times$ faster at batch size 256 for a
median $+10.6\%$, and after L5 $2.70\times$ faster for the same increase. Chronos-2-small also
benefits more from batching ($1.80\times$ at batch size 1, $3.28\times$ at 256).
```

## Table tab:h4_frontier_full, caption: Which TimesFM-3 adapter

**Why.** Caption notes the corrected adapter and offline accuracy.

**OLD**

```latex
\caption{\textbf{Performance-compute frontier of aligned truncation, all depths.} ``L$k$'' keeps
blocks $1,\dots,k$ and removes the rest; L0 keeps no block. MASE increase is the median over the
14 datasets of test MASE relative to the full model, with its interquartile range. Speedups are
end to end, from a context in host memory to forecasts returned to host memory, relative to the
full model at the same batch size: batch size 1 (B=1) measures single-forecast latency and is the
headline setting; batch size 256 (B=256) measures throughput. Chronos-2-small is a separately
pretrained model, not a truncation of Chronos-2.}
```

**NEW**

```latex
\caption{\textbf{Performance-compute frontier of aligned truncation, all depths.} ``L$k$'' keeps
blocks $1,\dots,k$ and removes the rest; L0 keeps no block. MASE increase is the median over the
14 datasets of test MASE relative to the full model, with its interquartile range. Speedups are
end to end, from a context in host memory to forecasts returned to host memory, relative to the
full model at the same batch size: batch size 1 (B=1) measures single-forecast latency and is the
headline setting; batch size 256 (B=256) measures throughput. Chronos-2-small is a separately
pretrained model, not a truncation of Chronos-2. TimesFM-3 uses the zero-anchored adapter, with
accuracy computed offline (Appendix~\ref{sec:app_timesfm3_ridge}).}
```

## Table tab:h4_frontier_full, TimesFM-3 rows L19 to L0: Corrected median and IQR; speedups unchanged

**Why.** All 20 TimesFM-3 rows; Chronos-2 and TiRex rows unchanged.

**OLD**

```latex
 & L19 & $+5.4\%$ & [$+2.9\%$, $+14.0\%$] & $0.99\times$ & $1.04\times$ \\
 & L18 & $+14.3\%$ & [$+6.8\%$, $+36.7\%$] & $1.03\times$ & $1.09\times$ \\
 & L17 & $+29.7\%$ & [$+13.4\%$, $+69.7\%$] & $1.07\times$ & $1.15\times$ \\
 & L16 & $+40.3\%$ & [$+15.9\%$, $+88.3\%$] & $1.05\times$ & $1.20\times$ \\
 & L15 & $+47.9\%$ & [$+30.1\%$, $+105.8\%$] & $1.08\times$ & $1.26\times$ \\
 & L14 & $+56.0\%$ & [$+36.9\%$, $+105.7\%$] & $1.11\times$ & $1.34\times$ \\
 & L13 & $+56.4\%$ & [$+37.4\%$, $+121.8\%$] & $1.11\times$ & $1.42\times$ \\
 & L12 & $+62.7\%$ & [$+42.3\%$, $+133.3\%$] & $1.15\times$ & $1.51\times$ \\
 & L11 & $+59.5\%$ & [$+45.5\%$, $+165.5\%$] & $1.12\times$ & $1.60\times$ \\
 & L10 & $+67.3\%$ & [$+41.0\%$, $+163.7\%$] & $1.19\times$ & $1.71\times$ \\
 & L9 & $+69.9\%$ & [$+39.1\%$, $+155.7\%$] & $1.18\times$ & $1.85\times$ \\
 & L8 & $+58.1\%$ & [$+43.2\%$, $+161.4\%$] & $1.25\times$ & $2.03\times$ \\
 & L7 & $+65.1\%$ & [$+40.6\%$, $+154.0\%$] & $1.22\times$ & $2.21\times$ \\
 & L6 & $+81.9\%$ & [$+42.9\%$, $+203.6\%$] & $1.24\times$ & $2.44\times$ \\
 & L5 & $+69.8\%$ & [$+45.8\%$, $+186.4\%$] & $1.24\times$ & $2.70\times$ \\
 & L4 & $+77.3\%$ & [$+45.2\%$, $+174.0\%$] & $1.29\times$ & $3.11\times$ \\
 & L3 & $+62.4\%$ & [$+37.4\%$, $+140.7\%$] & $1.34\times$ & $3.49\times$ \\
 & L2 & $+53.8\%$ & [$+40.0\%$, $+102.8\%$] & $1.43\times$ & $4.25\times$ \\
 & L1 & $+38.7\%$ & [$+25.9\%$, $+65.8\%$] & $1.37\times$ & $4.65\times$ \\
 & L0 & $+31.3\%$ & [$+19.3\%$, $+62.5\%$] & $1.38\times$ & $5.94\times$ \\
```

**NEW**

```latex
 & L19 & $+1.3\%$ & [$+0.6\%$, $+3.9\%$] & $0.99\times$ & $1.04\times$ \\
 & L18 & $+3.9\%$ & [$+1.6\%$, $+11.4\%$] & $1.03\times$ & $1.09\times$ \\
 & L17 & $+6.1\%$ & [$+2.8\%$, $+19.3\%$] & $1.07\times$ & $1.15\times$ \\
 & L16 & $+7.1\%$ & [$+4.1\%$, $+23.6\%$] & $1.05\times$ & $1.20\times$ \\
 & L15 & $+8.5\%$ & [$+6.0\%$, $+24.4\%$] & $1.08\times$ & $1.26\times$ \\
 & L14 & $+9.9\%$ & [$+5.4\%$, $+25.7\%$] & $1.11\times$ & $1.34\times$ \\
 & L13 & $+10.4\%$ & [$+5.5\%$, $+27.7\%$] & $1.11\times$ & $1.42\times$ \\
 & L12 & $+11.1\%$ & [$+6.1\%$, $+31.5\%$] & $1.15\times$ & $1.51\times$ \\
 & L11 & $+11.2\%$ & [$+6.4\%$, $+31.2\%$] & $1.12\times$ & $1.60\times$ \\
 & L10 & $+11.3\%$ & [$+6.6\%$, $+31.8\%$] & $1.19\times$ & $1.71\times$ \\
 & L9 & $+11.9\%$ & [$+6.4\%$, $+32.5\%$] & $1.18\times$ & $1.85\times$ \\
 & L8 & $+11.3\%$ & [$+6.5\%$, $+36.7\%$] & $1.25\times$ & $2.03\times$ \\
 & L7 & $+10.5\%$ & [$+6.4\%$, $+35.4\%$] & $1.22\times$ & $2.21\times$ \\
 & L6 & $+10.4\%$ & [$+7.0\%$, $+31.7\%$] & $1.24\times$ & $2.44\times$ \\
 & L5 & $+10.6\%$ & [$+4.5\%$, $+31.3\%$] & $1.24\times$ & $2.70\times$ \\
 & L4 & $+11.3\%$ & [$+4.9\%$, $+31.3\%$] & $1.29\times$ & $3.11\times$ \\
 & L3 & $+11.0\%$ & [$+4.8\%$, $+30.7\%$] & $1.34\times$ & $3.49\times$ \\
 & L2 & $+11.1\%$ & [$+5.1\%$, $+31.5\%$] & $1.43\times$ & $4.25\times$ \\
 & L1 & $+10.6\%$ & [$+7.2\%$, $+32.4\%$] & $1.37\times$ & $4.65\times$ \\
 & L0 & $+24.3\%$ & [$+15.7\%$, $+48.8\%$] & $1.38\times$ & $5.94\times$ \\
```

## Appendix B.4, truncation inside and before the tunnel: 15 of 15 depths for TimesFM-3

**Why.** Count with the corrected adapter; the identity-anchored count (14 of 15) is kept for reference.

**OLD**

```latex
\paragraph{Truncation inside and before the tunnel.} Table~\ref{tab:h4_tunnel_strata} gives the
depth-matched comparison of Section~\ref{sec:results_acceleration}. At each cut depth, the datasets
of a model are split according to whether that depth lies inside their validation-selected 5\%
tunnel ($\ell\geq\ell_{\mathrm{tunnel}}$) or before it, and the median aligned test MASE increase
is reported for each group. Because speedup depends only on the cut depth, both groups are compared
at the same speedup. Depths are listed where both groups contain at least two datasets. Truncation
is cheaper inside the tunnel at all 9 such depths for Chronos-2, all 12 for TiRex, and 14 of 15
for TimesFM-3; the exception is TimesFM-3 at L13, where only three datasets lie before the tunnel.
The comparison is observational: datasets whose tunnel begins early may also be easier to forecast
from intermediate states for other reasons.
```

**NEW**

```latex
\paragraph{Truncation inside and before the tunnel.} Table~\ref{tab:h4_tunnel_strata} gives the
depth-matched comparison of Section~\ref{sec:results_acceleration}. At each cut depth, the datasets
of a model are split according to whether that depth lies inside their validation-selected 5\%
tunnel ($\ell\geq\ell_{\mathrm{tunnel}}$) or before it, and the median aligned test MASE increase
is reported for each group. Because speedup depends only on the cut depth, both groups are compared
at the same speedup. Depths are listed where both groups contain at least two datasets. Truncation
is cheaper inside the tunnel at all 9 such depths for Chronos-2, all 12 for TiRex, and all 15 for
TimesFM-3 with the zero-anchored adapter (14 of 15 with the identity-anchored adapter of the
submitted version, the exception being L13, where only three datasets lie before the tunnel).
The comparison is observational: datasets whose tunnel begins early may also be easier to forecast
from intermediate states for other reasons.
```

## Table tab:h4_tunnel_strata, caption: Which TimesFM-3 adapter

**OLD**

```latex
\caption{\textbf{Aligned truncation inside versus before the forecasting tunnel.} Median test MASE
increase over the full model, at the same cut depth, for datasets where that depth lies inside
their validation-selected 5\% tunnel and for datasets where it lies before the tunnel entrance.
$n$ gives the number of datasets in each group.}
```

**NEW**

```latex
\caption{\textbf{Aligned truncation inside versus before the forecasting tunnel.} Median test MASE
increase over the full model, at the same cut depth, for datasets where that depth lies inside
their validation-selected 5\% tunnel and for datasets where it lies before the tunnel entrance.
$n$ gives the number of datasets in each group. TimesFM-3 uses the zero-anchored adapter
(Appendix~\ref{sec:app_timesfm3_ridge}).}
```

## Table tab:h4_tunnel_strata, TimesFM-3 rows: Inside versus before the tunnel, corrected adapter

**Why.** All 15 TimesFM-3 rows; group sizes unchanged.

**OLD**

```latex
TimesFM-3 & L0 & 4 / 10 & $+20.2\%$ & $+34.2\%$ \\
 & L1 & 8 / 6 & $+38.7\%$ & $+47.1\%$ \\
 & L2 & 8 / 6 & $+50.3\%$ & $+73.8\%$ \\
 & L3 & 8 / 6 & $+58.6\%$ & $+79.1\%$ \\
 & L4 & 8 / 6 & $+67.9\%$ & $+92.7\%$ \\
 & L5 & 9 / 5 & $+46.4\%$ & $+114.1\%$ \\
 & L6 & 9 / 5 & $+43.4\%$ & $+119.5\%$ \\
 & L7 & 9 / 5 & $+42.0\%$ & $+124.1\%$ \\
 & L8 & 9 / 5 & $+43.5\%$ & $+108.9\%$ \\
 & L9 & 9 / 5 & $+45.7\%$ & $+122.4\%$ \\
 & L10 & 9 / 5 & $+42.9\%$ & $+116.7\%$ \\
 & L11 & 9 / 5 & $+46.5\%$ & $+121.3\%$ \\
 & L12 & 11 / 3 & $+53.9\%$ & $+71.5\%$ \\
 & L13 & 11 / 3 & $+59.2\%$ & $+53.6\%$ \\
 & L14 & 12 / 2 & $+52.5\%$ & $+82.6\%$ \\
```

**NEW**

```latex
TimesFM-3 & L0 & 4 / 10 & $+16.5\%$ & $+29.5\%$ \\
 & L1 & 8 / 6 & $+10.6\%$ & $+15.0\%$ \\
 & L2 & 8 / 6 & $+11.1\%$ & $+12.1\%$ \\
 & L3 & 8 / 6 & $+11.0\%$ & $+11.5\%$ \\
 & L4 & 8 / 6 & $+11.3\%$ & $+12.6\%$ \\
 & L5 & 9 / 5 & $+10.3\%$ & $+19.5\%$ \\
 & L6 & 9 / 5 & $+10.3\%$ & $+18.7\%$ \\
 & L7 & 9 / 5 & $+10.3\%$ & $+19.1\%$ \\
 & L8 & 9 / 5 & $+10.5\%$ & $+20.9\%$ \\
 & L9 & 9 / 5 & $+10.3\%$ & $+21.1\%$ \\
 & L10 & 9 / 5 & $+10.6\%$ & $+22.7\%$ \\
 & L11 & 9 / 5 & $+10.3\%$ & $+22.6\%$ \\
 & L12 & 11 / 3 & $+10.9\%$ & $+13.8\%$ \\
 & L13 & 11 / 3 & $+10.4\%$ & $+13.0\%$ \\
 & L14 & 12 / 2 & $+8.9\%$ & $+20.6\%$ \\
```

## End of Appendix B.4: New subsections B.4.1 to B.4.5

**Why.** Inserted before \end{document}.

**OLD**

```latex
(none)
```

**NEW**

```latex
\clearpage
\input{appendices/entrance_cut_rule}

\input{appendices/lodo_transfer}

\input{appendices/source_diversity}

\input{appendices/noncontiguous_blocks}

\input{appendices/heldout_pairs}
```
