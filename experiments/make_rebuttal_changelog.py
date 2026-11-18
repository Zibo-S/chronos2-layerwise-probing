"""Build three_models_paper/rebuttal_main_changes.md: exact OLD -> NEW extracts of every main.tex change (python experiments/make_rebuttal_changelog.py <repo root>)."""
import re, subprocess, sys
REPO = sys.argv[1]
before = subprocess.run(["git", "-C", REPO, "show", "ea4896f:three_models_paper/main.tex"],
                        capture_output=True, text=True, check=True).stdout
after = open(f"{REPO}/three_models_paper/main.tex").read()


def para(t, marker):
    ps = re.split(r"\n\s*\n", t)
    hit = [p for p in ps if marker in p]
    assert len(hit) == 1, (marker, len(hit))
    return hit[0].strip("\n")


def line(t, marker):
    hit = [l for l in t.split("\n") if marker in l]
    assert len(hit) == 1, (marker, len(hit))
    return hit[0].rstrip()


def block(t, start, end):
    i = t.index(start)
    j = t.index(end, i)
    return t[i:j].rstrip()


ITEMS = []


def add(sec, title, why, old, new):
    ITEMS.append((sec, title, why, old, new))


# ---------------------------------------------------------------- Introduction
add("Section 1 (Introduction), contribution 4", "Tunnel stratification is post hoc",
    "Reviewer 96qN pointed out that the inside-versus-before comparison is post hoc and uses small groups. The claim is kept, but no longer called consistent.",
    line(before, "tunnel membership consistently stratifies"), line(after, "tunnel membership stratifies truncation tolerance in a post hoc comparison"))
# ---------------------------------------------------------------- Method
add("Section 3.3 (Hard cut and native-pathway alignment)", "How the TimesFM-3 adapter is fitted; what calibration needs",
    "States that TimesFM-3 uses a closed form shrunk toward zero (the correction of Appendix B.3.1), that the adapter needs only unlabeled target windows, and points to the transfer experiment (Reviewers 2p99, 96qN, dHTD).",
    para(before, "initialized at the identity and fitted with the backbone frozen"), para(after, "initialized at the identity and fitted with the backbone frozen"))
# ---------------------------------------------------------------- 4.1
add("Section 4.1, Table 1 caption", "Low-skill L0 entrances marked",
    "Reviewer 96qN: four of the seven L0 entrances are pairs where no depth beats the constant forecast. They are now marked in Table 1 and explained in Appendix B.1.6.",
    line(before, "Sustained 5\\% forecasting-tunnel entrances for the Q=9 readout"), line(after, "Sustained 5\\% forecasting-tunnel entrances for the Q=9 readout"))
add("Section 4.1, Table 1 rows", "Daggers on M5 (all models) and TiRex on SZ Taxi",
    "Same change as above; only these two rows change.",
    line(before, "Uber TLC & L4 & L1 & L10 & SZ Taxi") + "\n" + line(before, "SG Carpark & L10 & L12 & L11 & M5"),
    line(after, "Uber TLC & L4 & L1 & L10 & SZ Taxi") + "\n" + line(after, "SG Carpark & L10 & L12 & L11 & M5"))
add("Section 4.1, second paragraph", "What the tunnel's reference implies",
    "Reviewer 96qN, Q1: the tunnel is referenced to the final-block readout, which is itself 13% to 20% worse than the full model; referenced to the full model, a readout tunnel exists in only 6 of 42 pairs. This is now stated in the main text, with the analysis in Appendix B.1.6.",
    para(before, "The validation-selected tunnel entrances also remain close"), para(after, "The validation-selected tunnel entrances also remain close"))
# ---------------------------------------------------------------- 4.3
add("Section 4.3, first paragraph", "TimesFM-3: aligned pathway now beats the readout; the original fit overfit",
    "Phase 2g: with the closed-form adapter shrunk toward zero instead of toward the hard cut, aligned TimesFM-3 outperforms the supervised readout at the depths of Table 2, so the statement that the readout stays closer to the full model is withdrawn. The original +48% to +70% is attributed to overfitting and kept in Appendix B.3.1.",
    para(before, "Using the hard cut and label-free affine adapter introduced in Section"), para(after, "Using the hard cut and label-free affine adapter introduced in Section"))
add("Section 4.3, second paragraph", "TimesFM-3: size of the remaining gap",
    "Replaces the qualitative 'substantial residual gap' with the measured gap of the corrected adapter.",
    para(before, "We conclude that \\textit{recoverability precedes native-pathway compatibility in TSFMs}"),
    para(after, "We conclude that \\textit{recoverability precedes native-pathway compatibility in TSFMs}"))
add("Figure 3 (fig:h3_main)", "Regenerated TimesFM-3 panel and caption",
    "New figure file with the zero-anchored TimesFM-3 adapter (Chronos-2 and TiRex panels unchanged); the caption names the adapter. The original file is kept.",
    block(before, "  \\includegraphics[width=0.9\\textwidth]{figures/h3_main_electricity.pdf}", "  \\label{fig:h3_main}"),
    block(after, "  \\includegraphics[width=0.9\\textwidth]{figures/h3_main_electricity_rev.pdf}", "  \\label{fig:h3_main}"))
add("Table 2 (tab:h3_depth), caption", "Which TimesFM-3 adapter; original values given",
    "Transparency about the post hoc change: the caption reports the identity-anchored values that the table previously showed.",
    line(before, "\\caption{\\textbf{Native-pathway compatibility across depth.} Median test MASE"),
    line(after, "\\caption{\\textbf{Native-pathway compatibility across depth.} Median test MASE"))
add("Table 2 (tab:h3_depth), TimesFM-3 aligned row", "TimesFM-3 aligned at 25 / 50 / 75% of depth",
    "Corrected values (Phase 2g, zero anchor). Hard cut and readout columns unchanged.",
    line(before, "& $+69.8\\%$ & $+67.3\\%$ & $+47.9\\%$"), line(after, "& $+10.6\\%$ & $+11.3\\%$ & $+8.5\\%$"))
# ---------------------------------------------------------------- 4.4
add("Section 4.4, first paragraph", "TimesFM-3 accuracy and Chronos-2-small",
    "Adds the corrected TimesFM-3 accuracy (computed offline, physically confirmed at seven depths), and states plainly that Chronos-2-small is better at moderate speedups, by a larger margin at batch size 256, and where truncation is still useful (Reviewer 2p99's weakness).",
    para(before, "\\textit{How much does it cost to remove blocks?}"), para(after, "\\textit{How much does it cost to remove blocks?}"))
add("Section 4.4, new paragraph", "Transfer without target data; cost of calibration",
    "Reviewers 2p99, 96qN and dHTD: an adapter fitted on other datasets (Appendix B.4.2) and the effect of the number of sources (Appendix B.4.3), with the cost of dataset-specific calibration.",
    "(none: new paragraph)", para(after, "\\textit{Does one adapter serve unseen datasets?}"))
add("Section 4.4, tunnel paragraph", "Tunnel stratification: post hoc wording, TimesFM-3, pre-specified entrance test",
    "Reviewer 96qN: the inside-versus-before comparison is now described as post hoc with small groups. TimesFM-3's inside-tunnel degradation is updated (+40% to +68% was the overfit adapter; it is now about 10%), the count becomes 15 of 15, and the pre-specified test of cutting at the entrance is reported (Reviewer 2p99, Q2; Appendix B.4.1). The conclusion that recoverability alone does not ensure compatibility is kept, with the new evidence.",
    para(before, "\\textit{Does the forecasting tunnel identify where truncation is well tolerated?}"),
    para(after, "\\textit{Does the forecasting tunnel identify where truncation is well tolerated?}"))
add("Table 3 (tab:h4_operating_points), caption", "Which TimesFM-3 adapter",
    "The TimesFM-3 operating points now use the zero-anchored adapter; at these three depths its offline accuracy is confirmed on physically truncated models (Appendix B.3.1).",
    line(before, "Median test MASE increase across 14 datasets and end-to-end speedup at batch size 1. For TimesFM-3 at L19"),
    line(after, "Median test MASE increase across 14 datasets and end-to-end speedup at batch size 1. For TimesFM-3 at L19"))
add("Table 3 (tab:h4_operating_points), TimesFM-3 row", "TimesFM-3 operating points",
    "Accuracy at L19 / L10 / L0 with the corrected adapter; speedups unchanged (same adapter shape).",
    line(before, "& L19 & $+5.4\\%$ & $0.99\\times$") + "\n" + line(before, "& L10 & $+67.3\\%$ & $1.19\\times$") + "\n" + line(before, "& L0 & $+31.3\\%$ & $1.38\\times$ \\\\"),
    line(after, "& L19 & $+1.3\\%$ & $0.99\\times$") + "\n" + line(after, "& L10 & $+11.3\\%$ & $1.19\\times$") + "\n" + line(after, "& L0 & $+24.3\\%$ & $1.38\\times$ \\\\"))
add("Figure 4 (fig:h4_frontier)", "Regenerated TimesFM-3 frontier and caption",
    "New figure file; only the TimesFM-3 aligned curve changes. The original file is kept.",
    block(before, "  \\includegraphics[width=0.9\\textwidth]{figures/h4_frontier.pdf}", "  \\label{fig:h4_frontier}"),
    block(after, "  \\includegraphics[width=0.9\\textwidth]{figures/h4_frontier_rev.pdf}", "  \\label{fig:h4_frontier}"))
# ---------------------------------------------------------------- 5
add("Section 5, second paragraph", "TimesFM-3 as the complementary case",
    "The conclusion that recoverability does not guarantee compatibility is kept, but is now supported by the corrected numbers (tunnel at a median 5% of depth, aligned pathway within 5% only at L18) instead of the overfit adapter's degradation.",
    para(before, "Across three state-of-the-art TSFMs and 14 datasets spanning five sampling frequencies"),
    para(after, "Across three state-of-the-art TSFMs and 14 datasets spanning five sampling frequencies"))
add("Section 5, limitations", "TimesFM-3 gap, transfer, prefix-only truncation",
    "The open question on cross-dataset transfer is answered (Appendices B.4.2 and B.4.3) and replaced by its result; the TimesFM-3 limitation is updated to the corrected adapter (about 10% gap; far from the full model on BOOM and Wind Farms); the prefix-only design is qualified (Reviewer dHTD, Appendix B.4.4).",
    para(before, "Our readouts and adapters are deliberately simple"), para(after, "Our readouts and adapters are deliberately simple"))
add("AI use statement", "AI assistance during the review period",
    "Reflects that generative AI tools assisted in analyzing the original TimesFM-3 fits, implementing and running the review-period experiments, and drafting the new appendices and main-text revisions. Please check that this matches your own account before submitting.",
    para(before, "In this work, we used generative AI tools for writing assistance") + "\n\n" + para(before, "We have not used generative AI tools to generate research ideas"),
    para(after, "In this work, we used generative AI tools for writing assistance") + "\n\n" + para(after, "For the originally submitted version, we have not used generative AI tools"))
add("Reproducibility statement", "Order of analysis disclosed",
    "States that the zero anchor was introduced after the identity-anchored test results, and which review-period analyses are post hoc.",
    para(before, "The tunnel definition and all selection rules use validation data only"), para(after, "The tunnel definition and all selection rules use validation data only"))
# ---------------------------------------------------------------- Appendix A
add("Appendix A.7 (Alignment training), TimesFM-3 paragraph and Equation 7", "The closed form now shrinks toward zero",
    "Method description of the adapter used in the revision; the submitted variant is named and pointed to Appendix B.3.1.",
    para(before, "\\paragraph{TimesFM-3.} Its head is a single linear map"), para(after, "\\paragraph{TimesFM-3.} Its head is a single linear map"))
add("Appendix A.7, nonlinear variant", "Training and validation R2 of both closed forms",
    "The R2 sentence described only the identity-anchored adapter; it now adds the zero-anchored fit's R2 on Electricity and states which adapter is used.",
    para(before, "\\paragraph{Nonlinear variant.}"), para(after, "\\paragraph{Nonlinear variant.}"))
add("Appendix A.9 (Latency protocol), physical versus offline", "Scope of the physical check",
    "The 419 physically evaluated configurations used the submitted adapters; the corrected TimesFM-3 adapters are checked separately in Appendix B.3.1.",
    para(before, "\\paragraph{Physical versus offline truncation.}"), para(after, "\\paragraph{Physical versus offline truncation.}"))
add("Appendix A.10 (Compute)", "Compute of the review-period experiments",
    "Adds the 20.2 A100-hours of the review-period GPU jobs (Narval jobs 9229954, 9229962, 9229964, 9253740).",
    para(before, "All experiments ran on a shared academic GPU cluster."), para(after, "All experiments ran on a shared academic GPU cluster."))
# ---------------------------------------------------------------- Appendix B
add("Appendix B.1.5 (content-mean readouts)", "Moved to its own file, unchanged",
    "The whole subsubsection (text, Tables 11 and 12 of the pre-revision build, Figure 8, all labels) moved verbatim to appendices/content_mean_readouts.tex; diff-checked identical. The new Appendix B.1.6 follows it.",
    "\\subsubsection{Architecture-aware versus generic content-mean readouts}\n\\label{sec:app_content_mean}\n... (162 lines, unchanged) ...\n\\label{fig:app_content_mean_curves}\n\\end{figure}",
    "\\input{appendices/content_mean_readouts}\n\n\\clearpage\n\\input{appendices/native_reference_tunnel}")
add("Appendix B.3, depth profiles", "TimesFM-3 depth profile with the corrected adapter",
    "Replaces the identity-anchored profile with the corrected one and removes the speculative remark about the missing final normalization, which was offered as a possible explanation of a pattern that turned out to be produced by the fit.",
    para(before, "\\paragraph{Depth profiles.}"), para(after, "\\paragraph{Depth profiles.}"))
add("Table tab:h3_depth_nonlinear, caption and TimesFM-3 rows", "Affine column for TimesFM-3",
    "The affine TimesFM-3 column now uses the zero-anchored closed form; datasets within 5% in parentheses.",
    block(before, "  5\\% in parentheses. $^\\dagger$Iterative fit", "  \\vspace{4pt}") + "\n...\n" + block(before, "  TimesFM-3 & L5 (25\\%)", "  \\midrule\n  TiRex"),
    block(after, "  5\\% in parentheses. TimesFM-3 affine", "  \\vspace{4pt}") + "\n...\n" + block(after, "  TimesFM-3 & L5 (25\\%)", "  \\midrule\n  TiRex"))
add("Figure fig:app_h3_depth_timesfm3", "Regenerated, both anchors shown",
    "New figure file with the zero-anchored aligned pathway and the identity-anchored fit as a dotted line. The original file is kept.",
    block(before, "  \\includegraphics[width=\\textwidth]{figures/h3_depth_timesfm3.pdf}", "  \\label{fig:app_h3_depth_timesfm3}"),
    block(after, "  \\includegraphics[width=\\textwidth]{figures/h3_depth_timesfm3_rev.pdf}", "  \\label{fig:app_h3_depth_timesfm3}"))
add("Appendix B.3, new subsection", "B.3.1 Regularization of the TimesFM-3 adapter",
    "Inserted after the TiRex depth figure. The \\clearpage (also before B.1.6 and B.4.1) flushes queued floats so that each table and figure appears next to its own section; it changes page breaks only.", "(none)", "\\clearpage\n\\input{appendices/timesfm3_zero_anchor}")
add("Appendix B.4, throughput paragraph", "TimesFM-3 throughput operating points",
    "The sentence 'Its forecasting degradation is unchanged, so the larger throughput gains still come at the cost reported' referred to the overfit adapter; it now gives the corrected operating points at batch size 256.",
    para(before, "\\paragraph{Throughput at batch size 256.}"), para(after, "\\paragraph{Throughput at batch size 256.}"))
add("Table tab:h4_frontier_full, caption", "Which TimesFM-3 adapter",
    "Caption notes the corrected adapter and the seven depths at which its offline accuracy is physically confirmed.",
    block(before, "\\caption{\\textbf{Performance-compute frontier of aligned truncation, all depths.}", "\\vspace{4pt}"),
    block(after, "\\caption{\\textbf{Performance-compute frontier of aligned truncation, all depths.}", "\\vspace{4pt}"))
add("Table tab:h4_frontier_full, TimesFM-3 rows L19 to L0", "Corrected median and IQR; speedups unchanged",
    "All 20 TimesFM-3 rows; Chronos-2 and TiRex rows unchanged.",
    block(before, " & L19 & $+5.4\\%$", "\\midrule\nTiRex & L12 (full model)"),
    block(after, " & L19 & $+1.3\\%$", "\\midrule\nTiRex & L12 (full model)"))
add("Appendix B.4, truncation inside and before the tunnel", "15 of 15 depths for TimesFM-3",
    "Count with the corrected adapter; the identity-anchored count (14 of 15) is kept for reference.",
    para(before, "\\paragraph{Truncation inside and before the tunnel.}"), para(after, "\\paragraph{Truncation inside and before the tunnel.}"))
add("Table tab:h4_tunnel_strata, caption", "Which TimesFM-3 adapter", "",
    block(before, "\\caption{\\textbf{Aligned truncation inside versus before the forecasting tunnel.}", "\\vspace{4pt}"),
    block(after, "\\caption{\\textbf{Aligned truncation inside versus before the forecasting tunnel.}", "\\vspace{4pt}"))
add("Table tab:h4_tunnel_strata, TimesFM-3 rows", "Inside versus before the tunnel, corrected adapter",
    "All 15 TimesFM-3 rows; group sizes unchanged.",
    block(before, "TimesFM-3 & L0 & 4 / 10", "\\midrule\nTiRex & L0"), block(after, "TimesFM-3 & L0 & 4 / 10", "\\midrule\nTiRex & L0"))
add("End of Appendix B.4", "New subsections B.4.1 to B.4.5",
    "Inserted before \\end{document}.", "(none)",
    block(after, "\\clearpage\n\\input{appendices/entrance_cut_rule}", "\\end{document}"))

out = ["# Main-text changes for the rebuttal revision (exact OLD -> NEW)", "",
       "Every block below is copied verbatim from `three_models_paper/main.tex` before the revision "
       "(commit `ea4896f`, the snapshot of the submitted working tree) and after it. Section and "
       "appendix numbers are those of the revised PDF. `git diff ea4896f -- three_models_paper/main.tex` "
       "shows the same changes as a diff.", ""]
cur = None
for sec, title, why, old, new in ITEMS:
    out += [f"## {sec}: {title}", ""]
    if why:
        out += [f"**Why.** {why}", ""]
    out += ["**OLD**", "", "```latex", old, "```", "", "**NEW**", "", "```latex", new, "```", ""]
open(f"{REPO}/three_models_paper/rebuttal_main_changes.md", "w").write("\n".join(out))
print(len(ITEMS), "items written")
