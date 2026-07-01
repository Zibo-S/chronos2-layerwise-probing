"""Append a 'Per-dataset ID vs OOD (latest)' section to chronos2_probing.ipynb.

The new cells ONLY read perdataset_summary.json + the 5 saved PNGs — they do not load
Chronos-2 or extract features. We execute just these cells in an isolated temp notebook
(tsexp kernel) to capture real outputs, then merge them in without re-running heavy cells.
"""
import nbformat as nbf
from nbclient import NotebookClient

NB = "chronos2_probing.ipynb"
MARKER = "Per-dataset ID vs OOD (latest)"

narrative = r"""
## Per-dataset ID vs OOD (latest)

*Redraw + re-check (read-only): this section reads `perdataset_summary.json` and the 5 saved
PNGs produced by `probe_perdataset.py`. It does **not** reload Chronos-2 or re-extract features.*

### Finding 1 — the tunnel shape holds, per dataset
The ID curves show the rise → plateau → late-drop shape in every headroom dataset, with the peak
(argmax) sitting in the **middle band (L3–L7), not at the last layer**. The last-layer deficit
(mean L3–8 − L11) is positive with its 95% CI excluding 0 in **5 of 6** non-saturated datasets.
**SCP2** is the lone null — its probe sits near chance at every layer, so it is *underpowered*,
not a counterexample. **LSST** is borderline (lower CI edge at 0.000): a real but very small
effect, resolvable only because its test set is large (2466). → Across EEG, handwriting, gesture,
spectra and light-curves, the middle layers carry more linearly-decodable class information than
the final layer.

### Finding 2 — OOD damage is shift-dependent and **not** last-layer-concentrated (amplification null)
- **Timewarp:** the OOD curve sits roughly **parallel** below ID — degrades all layers about
  equally → no amplification.
- **Gaussian noise:** where resolvable, damage concentrates at the **early** layers (UWave & SCP1
  crater at L0–L1 and recover by the middle) — the *opposite* of the tunnel prediction (hence
  negative cells like UWave/gauss −0.082). SCP1 is the one exception (its L11 also collapses → the
  lone positive gauss cell +0.082).
- **Drift:** mostly mild/overlapping, except Handwriting (late-layer peel-away, +0.040) and Ethanol
  (erratic, −0.086).

Tallying the 18 non-saturated amplification cells: **3 positive-significant** (SCP1/gauss,
UWave/timewarp, Handwriting/drift), **2 negative-significant** (UWave/gauss, Ethanol/drift), 13
≈0 — scattered, with as many wrong-direction as right-direction hits: the signature of no real
effect. → The last-layer-preferential degradation Part 2 predicted does not appear; the most
input-like shift (Gaussian) hits the **early** layers hardest, a different phenomenon.

### Bottom line
- **Part 1 (middle > last) — robust, per dataset.** 5/6 non-saturated datasets, middle argmax
  (L3–L7), across five distinct domains.
- **Part 2 (gap widens under shift) — null.** OOD damage is not last-layer-concentrated; its
  location is shift-dependent (early for Gaussian). Significant cells are scattered and balanced by
  equally-significant opposite-direction cells.
- **Saturated datasets (Epilepsy, Cricket)** are reference only (flat ID at ceiling); Cricket's
  large negative gauss amplification (−0.236) is a saturated probe collapsing unevenly, not
  interpretable.
"""

table_code = r"""
# Verdict table — read straight from perdataset_summary.json (no recompute)
import json
import pandas as pd

S = json.load(open("perdataset_summary.json"))["datasets"]

def _fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + ("*" if d["excludes_0"] else "")

rows = []
for ds, r in S.items():
    if r is None:
        rows.append({"dataset": ds, "n_test": "(failed)"}); continue
    rows.append({
        "dataset": ds,
        "n_test": r["n_test"],
        "K": r["n_classes"],
        "chance": round(r["chance"], 3),
        "saturated": r["saturated"],
        "argmax": f"L{r['argmax_layer']}",
        "late_drop_band [lo,hi] excl0": _fmt(r["id_late_drop_band"]),
        "amp_gauss [lo,hi] excl0":     _fmt(r["amplification"]["gauss"]),
        "amp_timewarp [lo,hi] excl0":  _fmt(r["amplification"]["timewarp"]),
        "amp_drift [lo,hi] excl0":     _fmt(r["amplification"]["drift"]),
    })

df = pd.DataFrame(rows).set_index("dataset")
print("(* = 95% paired-bootstrap CI excludes 0;  late_drop_band = mean(L3-8) - L11)")
df
"""

figs = [
    ("fig_grid_id_tunnel.png",
     "**ID layer-wise probe accuracy per dataset** — gold star = argmax layer. The rise→plateau→late-drop tunnel shape, with the peak in the middle band."),
    ("fig_grid_idood_timewarp.png",
     "**ID vs timewarp-OOD** — the OOD curve sits roughly parallel below ID (uniform degradation, no amplification)."),
    ("fig_grid_idood_gauss.png",
     "**ID vs Gaussian-noise OOD** — where resolvable, damage concentrates at the *early* layers (L0–L1), not the last."),
    ("fig_grid_idood_drift.png",
     "**ID vs drift-OOD** — mostly mild / overlapping, except Handwriting (late-layer peel-away) and Ethanol (erratic)."),
    ("fig_grid_idood_all.png",
     "**Overview** — ID + all three shifts overlaid per dataset (no CI bands, for readability)."),
]

# ---- assemble the new cells ----
new_cells = [nbf.v4.new_markdown_cell(narrative.strip("\n")),
             nbf.v4.new_markdown_cell("### Verdict table (from `perdataset_summary.json`)"),
             nbf.v4.new_code_cell(table_code.strip("\n"))]
for path, cap in figs:
    new_cells.append(nbf.v4.new_markdown_cell(cap))
    new_cells.append(nbf.v4.new_code_cell(
        f'from IPython.display import Image, display\ndisplay(Image(filename="{path}"))'))

# ---- execute ONLY the new cells in an isolated temp notebook (no heavy cells) ----
tmp = nbf.v4.new_notebook()
tmp.cells = [c for c in new_cells if c.cell_type == "code"]
tmp.metadata = {"kernelspec": {"name": "tsexp", "language": "python", "display_name": "tsexp"}}
NotebookClient(tmp, timeout=300, kernel_name="tsexp").execute()
print("isolated execution of new code cells: OK")

# copy executed outputs back onto the new code cells (in order)
exec_iter = iter(tmp.cells)
for c in new_cells:
    if c.cell_type == "code":
        ec = next(exec_iter)
        c.outputs = ec.outputs
        c.execution_count = ec.execution_count

# ---- merge: refresh if a previous run added the section, else append ----
nb = nbf.read(NB, as_version=4)
start = next((i for i, c in enumerate(nb.cells)
              if c.cell_type == "markdown" and MARKER in c.source), None)
if start is not None:
    print(f"refreshing existing section (was at cell {start})")
    nb.cells = nb.cells[:start] + new_cells
else:
    print("appending new section to end of notebook")
    nb.cells = nb.cells + new_cells

nbf.write(nb, NB)
print(f"wrote {NB}: {len(nb.cells)} cells total, +{len(new_cells)} in the new section")
"""end"""
