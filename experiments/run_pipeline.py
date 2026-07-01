"""Layer-wise linear-probing demo over a frozen Chronos-2 encoder (Phases B-D).

Original end-to-end demonstration script. The reusable core (get_pipeline,
extract_features, fit_layerwise_probes) now lives in the ``probing`` package; this module
keeps only the demo-specific helpers and the Phase B-D driver. Behaviour and outputs are
unchanged — figures now land in ``results/`` (probing.config.OUT_DIR).

Run:  python -m experiments.run_pipeline
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing.config import NUM_LAYERS, SEED, OUT_DIR
from probing.extraction import extract_features, fit_layerwise_probes


def score_layerwise(probes, test_features, y_test):
    accs = np.zeros(NUM_LAYERS, dtype=np.float64)
    for i in range(NUM_LAYERS):
        Xs = probes[i]["scaler"].transform(test_features[i])
        accs[i] = probes[i]["clf"].score(Xs, y_test)
    return accs


def print_layer_table(label, accs, chance):
    print(f"\n  Per-layer test accuracy ({label}):   chance = {chance:.4f}")
    print(f"  {'layer':>6}  {'acc':>8}")
    for i, a in enumerate(accs):
        print(f"  {i:>6d}  {a:>8.4f}")
    print(f"  argmax = layer {int(np.argmax(accs))}  (acc={accs.max():.4f})")


# ----------------------------------------------------------------------- #
# Plotting helpers
# ----------------------------------------------------------------------- #

def plot_curve(path, curves, chance=None, title="", ylabel="test accuracy",
               annotate_argmax=False, ylim=None):
    """curves: list of (label, ndarray length 12, kwargs-dict-or-None)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = np.arange(NUM_LAYERS)
    for label, ys, kw in curves:
        kw = dict(kw or {})
        ax.plot(xs, ys, marker="o", label=label, **kw)
        if annotate_argmax:
            j = int(np.argmax(ys))
            ax.annotate(f"argmax L{j}", xy=(j, ys[j]),
                        xytext=(j, ys[j] + 0.03), ha="center", fontsize=8)
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", linewidth=1, label=f"chance ({chance:.3f})")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [saved] {path}")


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------ Phase B
    print("\n" + "=" * 72)
    print("PHASE B  --  ID layer-wise probe on Epilepsy")
    print("=" * 72)

    print("Extracting Epilepsy/train ...")
    f_tr, y_tr = extract_features("Epilepsy", split="train")
    print("Extracting Epilepsy/test  ...")
    f_te, y_te = extract_features("Epilepsy", split="test")

    n_classes_epi = len(np.unique(y_tr))
    chance_epi = 1.0 / n_classes_epi

    print("\nFitting per-layer probes (Epilepsy) ...")
    probes_epi = fit_layerwise_probes(f_tr, y_tr)
    accs_id_epi = score_layerwise(probes_epi, f_te, y_te)
    print_layer_table("Epilepsy ID", accs_id_epi, chance_epi)

    saturated_layers = int((accs_id_epi >= 0.95).sum())
    saturation_warning = saturated_layers >= 3
    if saturation_warning:
        print(f"\n  *** SATURATION WARNING ***")
        print(f"  {saturated_layers} layers reach acc >= 0.95 on Epilepsy.")
        print(f"  This dataset is saturating layer-wise probing; recommended swap: "
              f"Handwriting (150/850 cases, 26 classes).")
    else:
        print(f"\n  Saturation check: {saturated_layers} layers >= 0.95 (no warning).")

    plot_curve(
        OUT_DIR / "fig_id_epilepsy.png",
        curves=[("Epilepsy ID (clean test)", accs_id_epi, {"color": "C0"})],
        chance=chance_epi,
        title="Epilepsy: layer-wise linear probe (test accuracy)",
        annotate_argmax=True,
        ylim=(0.0, 1.05),
    )

    # ------------------------------------------------------------ Phase C
    print("\n" + "=" * 72)
    print("PHASE C  --  synthetic shift on Epilepsy (Gaussian noise on TEST only)")
    print("=" * 72)
    alphas = [0.1, 0.25, 0.5]
    accs_ood = {}
    for alpha in alphas:
        print(f"\nExtracting Epilepsy/test  corruption=gauss alpha={alpha} ...")
        f_te_n, y_te_n = extract_features(
            "Epilepsy", split="test",
            corruption={"kind": "gauss", "alpha": alpha, "seed": SEED},
        )
        assert np.array_equal(y_te_n, y_te), "test labels changed under corruption"
        accs_ood[alpha] = score_layerwise(probes_epi, f_te_n, y_te)
        print_layer_table(f"Epilepsy OOD alpha={alpha}", accs_ood[alpha], chance_epi)

    # Per-layer ID vs OOD(0.25) table + gap
    print(f"\n  Per-layer  ID  vs  OOD(0.25)  gap = ID - OOD")
    print(f"  {'layer':>6}  {'ID':>8}  {'OOD':>8}  {'gap':>8}")
    gap = accs_id_epi - accs_ood[0.25]
    for i in range(NUM_LAYERS):
        print(f"  {i:>6d}  {accs_id_epi[i]:>8.4f}  {accs_ood[0.25][i]:>8.4f}  {gap[i]:>+8.4f}")
    print(f"  smallest gap: layer {int(np.argmin(gap))} ({gap.min():+.4f})")
    print(f"  largest  gap: layer {int(np.argmax(gap))} ({gap.max():+.4f})")

    # Headline figure: ID vs OOD(0.25)
    plot_curve(
        OUT_DIR / "fig_ood_synthetic_epilepsy.png",
        curves=[
            ("Epilepsy ID (clean test)", accs_id_epi, {"color": "C0"}),
            ("OOD (Gaussian noise, alpha=0.25)", accs_ood[0.25], {"color": "C3"}),
        ],
        chance=chance_epi,
        title="Epilepsy: ID vs synthetic noise OOD (probes/scalers frozen from ID)",
        ylim=(0.0, 1.05),
    )

    # Sweep figure: ID + all three alphas
    sweep_curves = [("Epilepsy ID (clean test)", accs_id_epi, {"color": "C0", "linewidth": 2})]
    for j, alpha in enumerate(alphas):
        sweep_curves.append(
            (f"OOD alpha={alpha}", accs_ood[alpha], {"color": f"C{j+1}", "linestyle": "--"})
        )
    plot_curve(
        OUT_DIR / "fig_ood_sweep_epilepsy.png",
        curves=sweep_curves,
        chance=chance_epi,
        title="Epilepsy: ID vs Gaussian-noise OOD sweep",
        ylim=(0.0, 1.05),
    )

    # ------------------------------------------------------------ Phase D
    print("\n" + "=" * 72)
    print("PHASE D  --  cross-domain transfer (Epilepsy vs SelfRegulationSCP1)")
    print("=" * 72)
    print("Extracting SelfRegulationSCP1/train ...")
    f_tr_scp, y_tr_scp = extract_features("SelfRegulationSCP1", split="train")
    print("Extracting SelfRegulationSCP1/test  ...")
    f_te_scp, y_te_scp = extract_features("SelfRegulationSCP1", split="test")

    n_classes_scp = len(np.unique(y_tr_scp))
    chance_scp = 1.0 / n_classes_scp

    print("\nFitting per-layer probes (SCP1) ...")
    probes_scp = fit_layerwise_probes(f_tr_scp, y_tr_scp)
    accs_id_scp = score_layerwise(probes_scp, f_te_scp, y_te_scp)
    print_layer_table("SelfRegulationSCP1 ID", accs_id_scp, chance_scp)

    print(f"\n  Raw per-layer accuracy comparison:")
    print(f"  {'layer':>6}  {'Epilepsy':>10}  {'SCP1':>10}")
    for i in range(NUM_LAYERS):
        print(f"  {i:>6d}  {accs_id_epi[i]:>10.4f}  {accs_id_scp[i]:>10.4f}")
    print(f"  Epilepsy: chance={chance_epi:.4f}  argmax=layer {int(np.argmax(accs_id_epi))}")
    print(f"  SCP1:     chance={chance_scp:.4f}  argmax=layer {int(np.argmax(accs_id_scp))}")

    # Min-max normalize each dataset's curve to [0, 1] within its own range
    def minmax(a):
        lo, hi = a.min(), a.max()
        return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

    n_epi = minmax(accs_id_epi)
    n_scp = minmax(accs_id_scp)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xs = np.arange(NUM_LAYERS)
    ax.plot(xs, n_epi, marker="o", color="C0", label=f"Epilepsy (raw range {accs_id_epi.min():.2f}-{accs_id_epi.max():.2f})")
    ax.plot(xs, n_scp, marker="s", color="C3", label=f"SCP1 (raw range {accs_id_scp.min():.2f}-{accs_id_scp.max():.2f})")
    j_epi = int(np.argmax(accs_id_epi))
    j_scp = int(np.argmax(accs_id_scp))
    ax.annotate(f"Epilepsy argmax L{j_epi}", xy=(j_epi, n_epi[j_epi]),
                xytext=(j_epi, n_epi[j_epi] + 0.06), ha="center", fontsize=8, color="C0")
    ax.annotate(f"SCP1 argmax L{j_scp}", xy=(j_scp, n_scp[j_scp]),
                xytext=(j_scp, n_scp[j_scp] - 0.08), ha="center", fontsize=8, color="C3")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel("min-max normalized test accuracy")
    ax.set_xticks(xs)
    ax.set_title("Layer profile comparison across domains (not classifier transfer)")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_transfer_profiles.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] {OUT_DIR / 'fig_transfer_profiles.png'}")

    print("\n  NOTE: this figure compares layer PROFILES across two domains, "
          "not classifier transfer — each probe was trained on its own dataset.")

    # ------------------------------------------------------------ Final summary
    pngs = ["fig_id_epilepsy.png", "fig_ood_synthetic_epilepsy.png",
            "fig_ood_sweep_epilepsy.png", "fig_transfer_profiles.png"]
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    for p in pngs:
        path = OUT_DIR / p
        status = "OK" if path.exists() else "MISSING"
        print(f"  {status}  {p}  ({path.stat().st_size if path.exists() else 0} bytes)")
    print(f"  Saturation verdict (Epilepsy): "
          f"{'SATURATING - swap to Handwriting' if saturation_warning else 'not saturating'}  "
          f"({saturated_layers} layers >= 0.95)")
    print(f"  argmax layers:")
    print(f"    Epilepsy ID            : layer {int(np.argmax(accs_id_epi))} "
          f"(acc={accs_id_epi.max():.4f})")
    for alpha in alphas:
        print(f"    Epilepsy OOD alpha={alpha}: layer {int(np.argmax(accs_ood[alpha]))} "
              f"(acc={accs_ood[alpha].max():.4f})")
    print(f"    SelfRegulationSCP1 ID  : layer {int(np.argmax(accs_id_scp))} "
          f"(acc={accs_id_scp.max():.4f})")


if __name__ == "__main__":
    main()
