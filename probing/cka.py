"""Linear CKA (Centered Kernel Alignment) — reusable representation-similarity utility.

Canonical home for the linear-CKA measure used across the three probing lines
(extended_v3_rolling / ft_specialization / task_shift_classification). This is the exact
feature-space formula that experiments/run_task_shift.py::_linear_cka already used, promoted
here with a matched-row guard, degeneracy handling, and the matrix/diagonal helpers the
CKA analysis driver needs.

Linear CKA (Kornblith et al. 2019), feature-space (biased-HSIC) form — memory O(d^2), NOT
O(n^2), which is what we want for n up to a few thousand and d = 768:

    CKA(X, Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)

with Xc/Yc column-centered. Invariant to orthogonal transforms and isotropic scaling of the
features, in [0, 1] (up to float rounding). All accumulation is float64.

SCIENTIFIC RULE (enforced structurally): CKA rows must be the SAME examples. Every helper
here asserts matching row counts; there is no code path that pairs two different datasets'
examples by position. Within-dataset (layer x layer) and cross-checkpoint-on-the-same-dataset
(pretrained layer x fine-tuned layer) are the only valid comparisons — see the driver.

Pure numpy; matplotlib is imported lazily inside the plotting helpers only, so importing this
module (and the unit tests) never pulls torch or a plotting backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# core measure
# --------------------------------------------------------------------------- #
def _as_2d_f64(A) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"CKA expects a 2-D (n_examples, d) matrix, got shape {A.shape}")
    return A


def require_matched_rows(mats) -> int:
    """Assert every matrix in `mats` shares the same row count (the CKA validity precondition).
    Returns that row count. Raises ValueError on any mismatch — this is the guard that makes an
    accidental cross-dataset pairing (unequal n) fail loudly instead of silently aligning row i
    of one dataset with row i of another."""
    ns = [np.asarray(m).shape[0] for m in mats]
    if len(set(ns)) > 1:
        raise ValueError(f"CKA row-count mismatch across representations: {ns} — refusing to pair "
                         "examples that do not correspond (e.g. two different datasets)")
    if ns and ns[0] < 2:
        raise ValueError(f"CKA needs at least 2 rows, got {ns[0]}")
    return ns[0] if ns else 0


def _center(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def _self_fro(Xc: np.ndarray) -> float:
    """||Xc^T Xc||_F for a column-centered Xc (float64)."""
    G = Xc.T @ Xc                      # (d, d)
    return float(np.sqrt(np.sum(G * G)))


def _cka_from_centered(Xc: np.ndarray, fx: float, Yc: np.ndarray, fy: float) -> float:
    """Linear CKA from already-centered matrices and their self-Frobenius norms fx, fy."""
    denom = fx * fy
    if not np.isfinite(denom) or denom <= 0.0:      # a constant / degenerate representation
        return float("nan")
    cross = Xc.T @ Yc                                 # (dx, dy)
    hsic = float(np.sum(cross * cross))
    return hsic / denom


def linear_cka(X, Y) -> float:
    """Linear CKA between two (n, d) matrices with matched rows. 1 = identical up to an orthogonal
    transform + isotropic scale; 0 = unrelated. Returns NaN if either side is constant/degenerate."""
    Xc, Yc = _as_2d_f64(X), _as_2d_f64(Y)
    require_matched_rows([Xc, Yc])
    Xc, Yc = _center(Xc), _center(Yc)
    return _cka_from_centered(Xc, _self_fro(Xc), Yc, _self_fro(Yc))


# --------------------------------------------------------------------------- #
# matrices / diagonals over ordered layer representations
# --------------------------------------------------------------------------- #
def _prep_layers(reps):
    """Center every layer matrix once and cache its self-Frobenius norm. Asserts all layers share
    the same row count (same examples through every layer)."""
    mats = [_as_2d_f64(r) for r in reps]
    require_matched_rows(mats)
    cen = [_center(m) for m in mats]
    fro = [_self_fro(c) for c in cen]
    return cen, fro


def cka_matrix(rows, cols=None) -> np.ndarray:
    """Layer x layer linear-CKA matrix.

    - `cols is None`  -> symmetric within-set matrix M[i, j] = CKA(rows[i], rows[j]) (the layer x
      layer geometry of ONE representation set); only the upper triangle is computed then mirrored.
    - `cols` given    -> rectangular cross-set matrix M[i, j] = CKA(rows[i], cols[j]); NOT forced
      symmetric (this is the cross-checkpoint alignment: rows = pretrained layers, cols = FT layers).

    `rows` (and `cols`) is an ordered list of (n, d) matrices, one per layer. All row matrices must
    share n; if `cols` is given, its matrices must share the SAME n as `rows` (same examples through
    both checkpoints)."""
    rcen, rfro = _prep_layers(rows)
    symmetric = cols is None
    if symmetric:
        ccen, cfro = rcen, rfro
    else:
        ccen, cfro = _prep_layers(cols)
        if rcen[0].shape[0] != ccen[0].shape[0]:
            raise ValueError(f"cross-CKA row mismatch: rows have {rcen[0].shape[0]} examples, cols "
                             f"have {ccen[0].shape[0]} — cross-stage CKA needs the same examples")
    L, M = len(rcen), len(ccen)
    out = np.empty((L, M), dtype=np.float64)
    for i in range(L):
        j0 = i if symmetric else 0
        for j in range(j0, M):
            out[i, j] = _cka_from_centered(rcen[i], rfro[i], ccen[j], cfro[j])
            if symmetric:
                out[j, i] = out[i, j]
    return out


def same_layer_diagonal(reps_a, reps_b) -> np.ndarray:
    """CKA(a_l, b_l) per layer l — the same-layer drift curve between two checkpoints. Equals the
    diagonal of cka_matrix(reps_a, reps_b). Both lists must have equal length and matched rows."""
    if len(reps_a) != len(reps_b):
        raise ValueError(f"same_layer_diagonal length mismatch: {len(reps_a)} vs {len(reps_b)}")
    return np.array([linear_cka(a, b) for a, b in zip(reps_a, reps_b)], dtype=np.float64)


def cka_to_reference(reps, ref_index: int = -1) -> np.ndarray:
    """CKA(layer_l, layer_ref) per layer — how similar each layer is to a reference layer (default
    the last, ref_index=-1). Used for the 'CKA-to-final-representation' summary curve."""
    ref = reps[ref_index]
    return np.array([linear_cka(r, ref) for r in reps], dtype=np.float64)


# --------------------------------------------------------------------------- #
# representation adapters / deterministic subsampling
# --------------------------------------------------------------------------- #
def stack_slots(arr) -> np.ndarray:
    """Fold a forecast-slot tensor (n, K, d) into a CKA observation matrix (n*K, d) by row-major
    reshape: row w*K + k is window w, slot k. Deterministic and IDENTICAL across layers/stages as
    long as the same window set is used, so cross-stage rows correspond. A 2-D array passes through
    unchanged (content-pooling case)."""
    A = np.asarray(arr)
    if A.ndim == 2:
        return A
    if A.ndim == 3:
        n, K, d = A.shape
        return A.reshape(n * K, d)
    raise ValueError(f"stack_slots expects (n, K, d) or (n, d), got shape {A.shape}")


def subsample_indices(n: int, size: int | None, seed: int = 0) -> np.ndarray:
    """Deterministic sorted row indices for subsampling. Returns arange(n) if size is None or >= n.
    The SAME returned indices must be applied to every layer/stage being compared (the caller's job);
    store (size, seed) alongside the outputs for reproducibility."""
    if size is None or size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


# --------------------------------------------------------------------------- #
# cache / manifest helpers (generic, fail-loud, torch-free)
# --------------------------------------------------------------------------- #
def load_npz_reps(path, keys) -> list[np.ndarray]:
    """Load an ordered list of arrays from an .npz cache, in the given key order. Fails loudly if the
    file is missing (never silently returns nothing) or a requested key is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing feature cache {path} — extract it first (this analysis is "
                                "cache-only; it never loads a model)")
    z = np.load(path, allow_pickle=False)
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise KeyError(f"{path.name}: missing keys {missing}; present={list(z.files)[:6]}...")
    return [np.asarray(z[k]) for k in keys]


def stage_hash_from_manifest(manifest: dict, stage: str) -> str:
    """Read a fine-tuning checkpoint's 8-char hash from a loaded FT manifest dict, fail-loud if the
    stage is absent (e.g. a validity gate refused to emit it). Cheap — avoids re-hashing the large
    $SCRATCH safetensors; the on-disk cache filenames already embed this exact hash."""
    cks = manifest.get("checkpoints", {})
    if stage not in cks:
        raise RuntimeError(f"stage {stage!r} not in manifest checkpoints {list(cks)} — the FT validity "
                           "gate may have refused it, or the wrong manifest was passed")
    return cks[stage]["checkpoint_hash"]


# --------------------------------------------------------------------------- #
# publication-style figures (matplotlib imported lazily)
# --------------------------------------------------------------------------- #
def heatmap(M, xlabels, ylabels, path, *, title="", vmin=0.0, vmax=1.0,
            cbar_label="Linear CKA", xaxis_label="", yaxis_label="", annotate=False, dpi=200):
    """Square-cell CKA heatmap on a fixed [vmin, vmax] scale (default [0,1]) so panels are directly
    comparable. Layer labels on both axes; a colorbar labelled `cbar_label`; no per-cell numbers by
    default. Writes both `path` (.png) and its .pdf sibling. Returns the PNG Path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.asarray(M, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.45 * M.shape[1] + 2.2, 0.45 * M.shape[0] + 1.8))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal", origin="upper")
    ax.set_xticks(np.arange(M.shape[1])); ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(M.shape[0])); ax.set_yticklabels(ylabels, fontsize=7)
    if xaxis_label:
        ax.set_xlabel(xaxis_label, fontsize=9)
    if yaxis_label:
        ax.set_ylabel(yaxis_label, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if M[i, j] < 0.5 * (vmin + vmax) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def drift_curve(curves, xlabels, path, *, title="", ylabel="Linear CKA to pretrained (same layer)",
                xaxis_label="probed representation", ylim=(0.0, 1.0), dpi=200):
    """Line plot of one-or-more layerwise CKA curves. `curves` = list of (label, values[, color])
    tuples; each `values` is length len(xlabels). Writes `path` (.png) + .pdf. Returns the PNG Path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(xlabels))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(xlabels) + 3.0), 4.2))
    for item in curves:
        label, vals = item[0], np.asarray(item[1], dtype=float)
        color = item[2] if len(item) > 2 else None
        ax.plot(x, vals, "-o", ms=3, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(xaxis_label, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def save_matrix_csv(M, xlabels, ylabels, path) -> Path:
    """Save a labelled CKA matrix as CSV (first column = row label, header = column labels)."""
    import csv
    M = np.asarray(M, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(xlabels))
        for i, rl in enumerate(ylabels):
            w.writerow([rl] + [f"{v:.6f}" for v in M[i]])
    return path
