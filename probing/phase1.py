"""Phase-1 contract: ONE protocol, three models, fourteen datasets, two questions.

    Q1  At what depth does forecasting become RECOVERABLE?
    Q2  How does that functional transition compare with representation GEOMETRY?

Measured with exactly four quantities, and nothing else in Phase 1:
    * layer-wise Q=9 forecasting probes (the common quantile vector, the common reduction);
    * the 5% forecasting-tunnel entrance, on VALIDATION only;
    * unbiased linear CKA (biased retained as a diagnostic, never for cross-model absolutes);
    * entropy effective rank.

No alignment, no adapters, no truncation. Those are later phases and nothing here anticipates
them.

WHAT THIS MODULE OWNS
  the canonical quantile vector and the proof that all three model lines agree on it;
  the common loss reduction;
  the per-model representation-point table (labels, architectural index, normalized depth);
  the tunnel record (delegating the criterion to ``probing.tunnel.tunnel_start``, never
    re-implementing it);
  the geometry block (delegating both estimators to the existing implementations);
  the cell identity / config hash and the atomic, resumable cell store;
  the run manifest and the combined tables.

WHAT IT DELIBERATELY DOES NOT OWN
  window construction              -> probing.windows.build_for  (registry-dispatched)
  window parity                    -> probing.window_parity
  MASE                             -> probing.mase               (m from the registry)
  the tunnel criterion             -> probing.tunnel.tunnel_start
  linear CKA                       -> probing.cka                (via timesfm3_geometry wrappers)
  effective rank                   -> probing.spectral_metrics   (via the same wrappers)
  the cluster bootstrap            -> probing.stats
  dataset facts and provenance     -> probing.registry
  model-specific extraction/probes -> probing.phase1_{chronos2,timesfm3,tirex}

THREE THINGS A READER OF THE RESULTS MUST KNOW, recorded in every summary.json:

1. Each model is probed WHERE ITS OWN HEAD READS, in ITS OWN normalized target space
   (Chronos-2: context-standardized + arcsinh; TimesFM-3: detrend + RevIN at token 15; TiRex:
   per-patch loc/scale). The absolute Q=9 pinball loss is therefore NOT comparable across
   models. What IS comparable: the tunnel entrance (a within-model ratio against that model's
   own final depth) and MASE (raw units, one shared in-context seasonal-naive denominator).
   ``CROSS_MODEL_LOSS_CAVEAT`` says this verbatim in every artifact.

2. The three lines keep their own TRAINING objectives (Chronos-2 sums the pinball terms over
   quantiles; TimesFM-3 and TiRex average them) because those are validated, committed code
   paths. This changes nothing measurable and is not a loose end:
     * the two objectives differ by the exact constant 2Q (proved elementwise in
       ``probes.mean_pinball_loss``'s docstring and pinned by a Phase-1 contract test), so the
       weight-decay argmin and the tunnel's ratio test are invariant to the choice;
     * AdamW normalizes the gradient by its own second moment and its decay is decoupled from
       the loss, so a constant rescale of the objective leaves the fitted probe unchanged up to
       Adam's epsilon — also pinned by a contract test.
   Every REPORTED loss is the common mean pinball, computed by ``mean_pinball_per_window`` from
   the one saved prediction tensor.

3. ``not_listed`` in the provenance table means "absent from the documentation we have". It is
   not OOD, not unseen, not held out. No Phase-1 artifact contains a global PT-ID/PT-OOD field.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from probing import registry
from probing.cka import stack_slots
from probing.config import REPO_ROOT, SEED
from probing.probes import mean_pinball_loss, median_index, validate_quantiles
# Generic estimator wrappers. They were written for the TimesFM-3 geometry line but take the
# hidden dimension as a parameter and contain nothing model-specific; every one of them
# delegates to probing.cka / probing.spectral_metrics. Reused, never re-implemented — see the
# module docstring's "what this module does not own".
from probing.timesfm3_geometry import cka_layer_matrix, cka_null_floor, effective_rank_curve
from probing.tunnel import TUNNEL_TOL, tunnel_start

__all__ = [
    "PHASE1_PROTOCOL_VERSION", "PHASE1_C", "PHASE1_H", "PHASE1_QUANTILES",
    "PHASE1_QUANTILE_SETS", "PHASE1_TUNNEL_TOL", "PHASE1_TUNNEL_TOLS", "PHASE1_BOOT_B",
    "MODELS", "ROSTER", "FIRST_DATASET", "CROSS_MODEL_LOSS_CAVEAT", "GEOMETRY_CAVEAT",
    "POINT_TYPES", "BLOCK_DEPTH", "HEAD_INPUT_DIAGNOSTIC", "HEAD_INPUT_CAVEAT",
    "quantile_set", "assert_canonical_quantiles", "RepPoint", "ModelSpec", "MODEL_SPECS",
    "model_spec", "mean_pinball_per_window", "tunnel_record", "geometry_block",
    "cluster_ci", "cluster_ratio_ci", "CellStore", "cell_config_hash", "environment_record", "registry_snapshot",
    "registry_hash", "provenance_rows", "write_csv", "atomic_write_json", "git_state",
]

# --------------------------------------------------------------------------- #
# the frozen protocol
# --------------------------------------------------------------------------- #
#: Bumped BY HAND whenever the scientific semantics of a cell change (quantile vector, loss
#: reduction, representation points, geometry row construction, tunnel criterion). It is part of
#: the cell config hash, so a bump makes every existing cell refuse reuse instead of silently
#: mixing two protocols in one results tree. A docs-only commit must NOT bump it — that is why
#: the git commit is recorded but is deliberately not part of the hash.
PHASE1_PROTOCOL_VERSION = "phase1/v1"

PHASE1_C = 512
PHASE1_H = 64

#: THE cross-model quantile vector. Verified against all three model lines at import of
#: :func:`assert_canonical_quantiles`, never assumed.
PHASE1_QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)

#: q9 is THE Phase-1 setting. q1 stays reachable (the frozen 2026-09-21 cross-model decision,
#: kept for later robustness work) and is never mixed into a Phase-1 table.
PHASE1_QUANTILE_SETS = {
    "q9": PHASE1_QUANTILES,
    "q1": np.array([0.5], dtype=np.float64),
}

PHASE1_TUNNEL_TOL = TUNNEL_TOL                       # 0.05 — the project's existing 5% rule
PHASE1_TUNNEL_TOLS = (0.01, 0.02, 0.05, 0.10)        # all saved so a re-run is never needed
PHASE1_BOOT_B = 5000                                 # the frozen paper value (run_native_head_adapter)

MODELS = ("chronos2", "timesfm3", "tirex")
ROSTER = "paper14"

#: Run this dataset first: the first three completed cells then cover all three model
#: implementations on ONE dataset, so a systemic bug surfaces at cell 3, not cell 29.
FIRST_DATASET = "monash_electricity_hourly"

CROSS_MODEL_LOSS_CAVEAT = (
    "Each model is probed where its OWN head reads, in its OWN normalized target space "
    "(Chronos-2: context-standardized + arcsinh at the K=4 forecast slots; TimesFM-3: "
    "detrend + RevIN at the last real context token; TiRex: per-pass loc/scale at the two "
    "native forecast-producing states). The absolute Q=9 mean pinball loss is therefore NOT "
    "comparable ACROSS models. Cross-model comparison is valid for (a) the tunnel entrance, "
    "which is a within-model ratio against that model's own final depth, and (b) MASE, which "
    "is in raw units under one shared in-context seasonal-naive denominator.")

GEOMETRY_CAVEAT = (
    "Unbiased linear CKA is the Phase-1 headline. The biased estimator is saved as a "
    "diagnostic but its finite-sample floor differs across the three models because the number "
    "of readout rows per window and the representation dimension differ (Chronos-2: 4 rows per "
    "window at d=768; TimesFM-3: 1 at d=1280; TiRex headline pos0: 1 at d=512). Absolute "
    "biased CKA must NEVER be compared across models. The measured per-(n, d, estimator) null "
    "floor is saved beside every matrix and nothing is subtracted from the matrices.")


def quantile_set(name: str):
    """``(quantile vector float64, median index within the set)``. No default at a call site."""
    if name not in PHASE1_QUANTILE_SETS:
        raise ValueError(f"unknown Phase-1 quantile set {name!r}; "
                         f"known: {sorted(PHASE1_QUANTILE_SETS)}")
    raw = np.asarray(PHASE1_QUANTILE_SETS[name], np.float64)
    validate_quantiles(raw)          # the project-wide checker (in (0,1), strictly increasing)
    # Kept in float64. ``validate_quantiles`` returns a float32 COPY, and round-tripping through
    # it would turn 0.1 into 0.10000000149011612 in every manifest and CSV — a cosmetic lie
    # about the protocol. The probes cast to float32 at the tensor boundary anyway.
    q = raw
    mid = median_index(q)
    if mid is None:
        raise ValueError(f"quantile set {name!r} has no exact 0.5 level, so it admits no median "
                         "forecast and therefore no MASE")
    return q, mid


def assert_canonical_quantiles() -> dict:
    """Prove the three model lines already agree on the nine quantiles. Fails loudly otherwise.

    This is the check the spec asks for: "verify rather than assuming". Each model's native
    vector is read from ITS OWN module — Chronos-2's ``QUANTILE_SETS['q9']``, TimesFM-3's
    ``NATIVE_QUANTILES``, TiRex's ``EXPECT_QUANTILES`` (which ``tirex_model`` asserts against
    the live checkpoint) — and compared element-wise with :data:`PHASE1_QUANTILES`.
    """
    from probing.probes import QUANTILE_SETS as CHRONOS_SETS
    from probing.timesfm3 import NATIVE_QUANTILES as TFM3_Q
    from probing.tirex_model import EXPECT_QUANTILES as TIREX_Q

    sources = {"chronos2": np.asarray(CHRONOS_SETS["q9"], np.float64),
               "timesfm3": np.asarray(TFM3_Q, np.float64),
               "tirex": np.asarray(TIREX_Q, np.float64)}
    bad = {k: v.tolist() for k, v in sources.items()
           if v.shape != PHASE1_QUANTILES.shape or not np.allclose(v, PHASE1_QUANTILES, atol=1e-9)}
    if bad:
        raise RuntimeError(
            f"the three model lines do NOT share one nine-quantile vector. Phase-1 canonical "
            f"{PHASE1_QUANTILES.tolist()}, disagreeing sources {bad}. A cross-model Q=9 "
            "comparison is meaningless until this is reconciled — refusing to run.")
    return {"canonical": PHASE1_QUANTILES.tolist(),
            "median_index": int(median_index(PHASE1_QUANTILES)),
            "verified_against": {k: v.tolist() for k, v in sources.items()},
            "note": "each vector was read from that model line's own module, not re-typed here"}


# --------------------------------------------------------------------------- #
# representation points and depth coordinates
# --------------------------------------------------------------------------- #
#: The two kinds of representation point, and the whole reason the distinction exists.
#:
#: ``block_depth``            Emb and the block outputs L1..L_N. THESE AND ONLY THESE form the
#:                            model's depth axis. The final block output (L12 / L20 / L12) is
#:                            the final depth, and therefore the tunnel's reference.
#: ``head_input_diagnostic``  a FINAL NORMALIZATION applied after the last block — Chronos-2's
#:                            ``L12+LN`` (encoder.final_layer_norm) and TiRex's ``L12+RMS``.
#:                            It is what the native head literally reads, which makes it worth
#:                            probing and worth keeping for the later native-head / alignment
#:                            work, but IT IS NOT AN ADDITIONAL MODEL DEPTH: a norm adds no
#:                            block, no parameters of depth and no residual-stream step.
#:                            Counting it as one would (a) put a 15th point on a 14-point depth
#:                            axis, (b) let a normalization define "final-depth probe loss", and
#:                            (c) shift every normalized depth. So it is excluded from
#:                            ``tunnel_start``, from the normalized depth coordinates, and from
#:                            the main cross-model curves and heatmaps.
#:
#: TimesFM-3 has NO head_input_diagnostic point: its native head reads L20 directly, so L20 is
#: both the final block depth and the head input.
POINT_TYPES = ("block_depth", "head_input_diagnostic")
BLOCK_DEPTH = "block_depth"
HEAD_INPUT_DIAGNOSTIC = "head_input_diagnostic"

HEAD_INPUT_CAVEAT = (
    "A final normalization (Chronos-2 L12+LN, TiRex L12+RMS) is NOT an additional model depth. "
    "It is probed and saved as a head-input diagnostic for the later native-head/alignment "
    "work, but it is excluded from the tunnel criterion, from the normalized depth axis and "
    "from the main cross-model depth curves and heatmaps. The final depth — and therefore the "
    "final-depth probe loss and the tunnel's reference — is the final BLOCK output.")


@dataclass(frozen=True)
class RepPoint:
    """One probed representation point of one model.

    ``point_type``     ``block_depth`` (on the depth axis) or ``head_input_diagnostic``
                       (probed and saved, but never on the depth axis). See :data:`POINT_TYPES`.
    ``block_index``    architectural depth in BLOCKS: 0 for the embedding, k for Lk, and
                       ``num_blocks`` again for a final-norm point (a norm adds no block).
    ``position_index`` ordinal in the extracted sequence; always strictly increasing.
    ``kind``           "embedding" | "block" | "final_norm".
    """
    label: str
    slug: str
    kind: str
    block_index: int
    position_index: int
    num_blocks: int
    n_points: int
    n_depth_points: int
    point_type: str = BLOCK_DEPTH
    is_native_readout: bool = False

    def __post_init__(self):
        if self.point_type not in POINT_TYPES:
            raise ValueError(f"{self.label}: point_type {self.point_type!r} not in {POINT_TYPES}")

    @property
    def on_depth_axis(self) -> bool:
        return self.point_type == BLOCK_DEPTH

    @property
    def relative_depth(self):
        """block_index / num_blocks in [0, 1] — THE cross-model depth coordinate.

        It is the fraction of the residual stream traversed, the only quantity Chronos-2 (12
        blocks), TimesFM-3 (20) and TiRex (12) share.

        ``None`` for a head_input_diagnostic point, deliberately. Its block_index would be
        num_blocks (the norm sits after the last block), so it would collide with the final
        block at exactly 1.0 and silently double-plot the end of every depth curve. Returning
        None makes it drop out of a numeric axis instead of landing on top of L12/L20.
        """
        return None if not self.on_depth_axis else self.block_index / self.num_blocks

    @property
    def relative_position(self):
        """position_index / (n_depth_points - 1): strictly monotone over the DEPTH AXIS, so the
        final block output is exactly 1.0. ``None`` off the depth axis, for the same reason
        ``relative_depth`` is. NOT comparable across models with different point sets."""
        return None if not self.on_depth_axis else self.position_index / (self.n_depth_points - 1)

    def as_dict(self) -> dict:
        return {"label": self.label, "slug": self.slug, "kind": self.kind,
                "point_type": self.point_type,
                "include_in_main_depth_axis": self.on_depth_axis,
                "block_index": self.block_index, "position_index": self.position_index,
                "relative_depth": self.relative_depth,
                "relative_position": self.relative_position,
                "is_native_readout": self.is_native_readout}


def _points(labels, kinds, block_indices, num_blocks, point_types) -> tuple[RepPoint, ...]:
    """Build one model's point table.

    ``is_native_readout`` marks what the native head literally reads: the head-input diagnostic
    where one exists (Chronos-2, TiRex), and the final block otherwise (TimesFM-3's L20). It is
    ORTHOGONAL to ``point_type`` — "the head reads this" and "this is a model depth" are two
    different claims, and conflating them is exactly what put a normalization on the depth axis.
    """
    n = len(labels)
    n_depth = sum(1 for t in point_types if t == BLOCK_DEPTH)
    out = []
    for i, (lab, kind, bi, pt) in enumerate(zip(labels, kinds, block_indices, point_types)):
        slug = lab.replace("+", "_").replace(" ", "_")
        out.append(RepPoint(lab, slug, kind, bi, i, num_blocks, n, n_depth, point_type=pt,
                            is_native_readout=(i == n - 1)))
    return tuple(out)


def _chronos2_points():
    # "L12+LN", not "+RMS": Chronos-2's encoder ends in a LayerNorm (encoder.final_layer_norm),
    # and this is the spelling the committed Chronos-2 fslot line already uses
    # (run_ptood_probing_ftok.POST_LN_LABEL). TiRex is the model with an RMSNorm.
    # It is a HEAD-INPUT DIAGNOSTIC, not a 13th depth: the depth axis is Emb, L1..L12.
    labels = ["Emb"] + [f"L{k}" for k in range(1, 13)] + ["L12+LN"]
    kinds = ["embedding"] + ["block"] * 12 + ["final_norm"]
    blocks = [0] + list(range(1, 13)) + [12]
    types = [BLOCK_DEPTH] * 13 + [HEAD_INPUT_DIAGNOSTIC]
    return _points(labels, kinds, blocks, 12, types)


def _timesfm3_points():
    # No head-input diagnostic: the native quantile head reads L20 directly, so L20 is both the
    # final block depth and the head input. The depth axis is the whole point set.
    labels = ["Emb"] + [f"L{k}" for k in range(1, 21)]
    kinds = ["embedding"] + ["block"] * 20
    blocks = [0] + list(range(1, 21))
    return _points(labels, kinds, blocks, 20, [BLOCK_DEPTH] * 21)


def _tirex_points():
    # L12+RMS is the RMSNorm output that output_patch_embedding reads — a HEAD-INPUT
    # DIAGNOSTIC, not a 13th depth. The depth axis is Emb, L1..L12.
    labels = ["Emb"] + [f"L{k}" for k in range(1, 13)] + ["L12+RMS"]
    kinds = ["embedding"] + ["block"] * 12 + ["final_norm"]
    blocks = [0] + list(range(1, 13)) + [12]
    types = [BLOCK_DEPTH] * 13 + [HEAD_INPUT_DIAGNOSTIC]
    return _points(labels, kinds, blocks, 12, types)


@dataclass(frozen=True)
class ModelSpec:
    """Everything Phase 1 needs to know about one model that is not a runtime fact."""
    name: str
    display_name: str
    d: int
    num_blocks: int
    points: tuple[RepPoint, ...]
    probe: str
    probe_out_features: int
    readout: str
    geometry_rows: str
    rows_per_window: int
    geometry_variants: tuple[str, ...] = ("headline",)
    target_space: str = ""

    @property
    def labels(self) -> list[str]:
        """EVERY probed point, in order — including any head-input diagnostic."""
        return [p.label for p in self.points]

    @property
    def n_points(self) -> int:
        return len(self.points)

    # ---- the depth axis ------------------------------------------------- #
    @property
    def depth_indices(self) -> list[int]:
        """Positions of the ``block_depth`` points: THE depth axis (Emb, L1..L_N).

        A final normalization is not a model depth, so it is not here. Everything that defines
        a depth — the tunnel scan, the normalized coordinates, the main curves and heatmaps —
        indexes through this list and therefore cannot accidentally include one.
        """
        return [i for i, p in enumerate(self.points) if p.on_depth_axis]

    @property
    def depth_labels(self) -> list[str]:
        return [self.points[i].label for i in self.depth_indices]

    @property
    def n_depth_points(self) -> int:
        return len(self.depth_indices)

    @property
    def diagnostic_indices(self) -> list[int]:
        """Positions of the head-input diagnostic points (0 for TimesFM-3, 1 for the others)."""
        return [i for i, p in enumerate(self.points) if not p.on_depth_axis]

    @property
    def diagnostic_labels(self) -> list[str]:
        return [self.points[i].label for i in self.diagnostic_indices]

    @property
    def point_types(self) -> list[str]:
        return [p.point_type for p in self.points]

    @property
    def reference_index(self) -> int:
        """THE final depth: the last BLOCK output (L12 / L20 / L12).

        This is the tunnel's reference, the denominator of every ratio, the delta-vs-last
        baseline and what "final-depth probe loss" means. It is deliberately NOT the last
        element of ``points``: for Chronos-2 and TiRex that is a normalization, and letting a
        normalization define the final depth is precisely the error this property exists to
        prevent.
        """
        return self.depth_indices[-1]

    @property
    def reference_label(self) -> str:
        return self.points[self.reference_index].label

    @property
    def native_readout_index(self) -> int:
        """What the native head literally reads — the diagnostic where one exists, else the
        final block. Reported, never used as a depth."""
        return self.diagnostic_indices[-1] if self.diagnostic_indices else self.reference_index

    def as_dict(self) -> dict:
        return {"model": self.name, "display_name": self.display_name, "d": self.d,
                "num_blocks": self.num_blocks, "n_points": self.n_points,
                "n_depth_points": self.n_depth_points,
                "representation_points": [p.as_dict() for p in self.points],
                "labels": self.labels, "point_types": self.point_types,
                "depth_axis_labels": self.depth_labels,
                "depth_axis_indices": self.depth_indices,
                "head_input_diagnostic_labels": self.diagnostic_labels,
                "head_input_diagnostic_indices": self.diagnostic_indices,
                "probe": self.probe,
                "probe_out_features": self.probe_out_features, "readout": self.readout,
                "geometry_rows": self.geometry_rows, "rows_per_window": self.rows_per_window,
                "geometry_variants": list(self.geometry_variants),
                "target_space": self.target_space,
                "tunnel_reference_point": self.reference_label,
                "tunnel_reference_index": self.reference_index,
                "native_readout_point": self.points[self.native_readout_index].label,
                "head_input_caveat": HEAD_INPUT_CAVEAT}


MODEL_SPECS: dict[str, ModelSpec] = {
    "chronos2": ModelSpec(
        name="chronos2", display_name="Chronos-2", d=768, num_blocks=12,
        points=_chronos2_points(),
        probe="ONE shared Linear(768, Q*16) applied to each of the K=4 native forecast slots; "
              "the four predicted patches concatenate to (B, Q, 64)",
        probe_out_features=9 * 16,
        readout="K = ceil(H/16) = 4 native forecast-slot token states per window",
        geometry_rows="forecast slots stacked row-major, (n, 4, 768) -> (4n, 768) via "
                      "probing.cka.stack_slots — the construction the committed Chronos-2 CKA "
                      "analysis (run_cka_analysis --extv4-fslot) already uses",
        rows_per_window=4,
        target_space="context-standardized then arcsinh (Chronos-2's own InstanceNorm target "
                     "space; id_data._make_examples)"),
    "timesfm3": ModelSpec(
        name="timesfm3", display_name="TimesFM-3", d=1280, num_blocks=20,
        points=_timesfm3_points(),
        probe="Linear(1280, H*Q) per representation point, reshaped horizon-major to (B, Q, 64)",
        probe_out_features=64 * 9,
        readout="the LAST REAL context token (index 15 for C=512), one row per window",
        geometry_rows="the last real context-token state, one row per window: (N, 1280)",
        rows_per_window=1,
        target_space="linear detrend + RevIN statistics of the readout token "
                     "(timesfm3_last_token.build_last_token_targets)"),
    "tirex": ModelSpec(
        name="tirex", display_name="TiRex", d=512, num_blocks=12,
        points=_tirex_points(),
        probe="ONE shared Linear(512, Q*32) applied to BOTH native forecast-producing states "
              "(pass 0 token 63 -> y[0:32], pass 1 token 63 -> y[32:64]); the two predicted "
              "patches concatenate to (B, Q, 64)",
        probe_out_features=9 * 32,
        readout="token 63 of each of the two default (two_pass) inference passes",
        geometry_rows="HEADLINE `pos0`: readout position 0 only, (N, 512), at all 14 depths — "
                      "constant N, and it avoids the degenerate (Emb, position 1) state. "
                      "COMPANION `all_positions_from_L1`: both states, (2N, 512), L1..L12+RMS",
        rows_per_window=1,
        geometry_variants=("pos0", "all_positions_from_L1"),
        target_space="per-pass loc/scale of the 512 real context values "
                     "(tirex_model.build_targets, mode=two_pass)"),
}


def model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_SPECS)}") from None


# --------------------------------------------------------------------------- #
# the common loss
# --------------------------------------------------------------------------- #
def mean_pinball_per_window(pred, target, q) -> np.ndarray:
    """THE Phase-1 loss, per window: mean of rho_tau over the Q*H terms of one window. (B,).

    ``pred`` (B, Q, H) and ``target`` (B, H) in that model's OWN normalized target space; ``q``
    the quantile vector. Its ``.mean()`` is the reported scalar — the same op chain — which is
    what lets the series/cluster bootstrap resample test windows post hoc without refitting.

    One implementation for all three models. It reuses the generic per-window helper written for
    the TimesFM-3 line; ``mean_pinball_loss`` (probes.py) is the scalar twin and a contract test
    pins ``per_window(...).mean() == mean_pinball_loss(...)``.
    """
    import torch
    from probing.timesfm3_last_token_probes import pinball_loss_per_window
    p = torch.as_tensor(np.asarray(pred, np.float32))
    t = torch.as_tensor(np.asarray(target, np.float32))
    qt = torch.as_tensor(np.asarray(q, np.float32))
    with torch.no_grad():
        return pinball_loss_per_window(p, t, qt).cpu().numpy().astype(np.float64)


def per_quantile_mean_loss(pred, target, q) -> list[float]:
    """Mean pinball per quantile level — the calibration diagnostic, (Q,)."""
    import torch
    p = torch.as_tensor(np.asarray(pred, np.float32))
    t = torch.as_tensor(np.asarray(target, np.float32)).unsqueeze(1)
    qv = torch.as_tensor(np.asarray(q, np.float32)).view(1, -1, 1)
    with torch.no_grad():
        u = t - p
        return torch.maximum(qv * u, (qv - 1.0) * u).mean(dim=(0, 2)).cpu().numpy().tolist()


# --------------------------------------------------------------------------- #
# the tunnel — criterion delegated, never re-implemented
# --------------------------------------------------------------------------- #
def tunnel_record(spec: ModelSpec, val_losses, test_losses=None,
                  tol: float = PHASE1_TUNNEL_TOL, tols=PHASE1_TUNNEL_TOLS) -> dict:
    """The 5% forecasting-tunnel entrance for one model x dataset cell, from VALIDATION only.

    ``val_losses`` / ``test_losses`` are FULL curves, one entry per probed point in
    ``spec.points`` order — head-input diagnostics included, because they are probed and
    reported. The criterion, however, runs on the DEPTH AXIS ALONE
    (``spec.depth_indices``): a final normalization is not a model depth, so it can neither be
    selected as an entrance nor serve as the final-depth reference. The reference is
    ``spec.reference_index``, the final BLOCK output.

    The first-crossing scan is ``probing.tunnel.tunnel_start`` called verbatim on the
    depth-axis subcurve — there is no second implementation of the criterion anywhere in the
    Phase-1 code. Test losses, when given, enter only as a reported generalization check; they
    can never move the entrance.

    Every tolerance in ``tols`` is evaluated and saved, so a 1% / 2% / 10% criterion never
    requires re-running a foundation model. The excluded points and their losses are recorded
    beside the result, so the exclusion is visible rather than merely done.
    """
    v_all = np.asarray(val_losses, np.float64)
    if v_all.ndim != 1 or v_all.size != spec.n_points:
        raise ValueError(f"{spec.name}: validation curve has {v_all.shape} entries for "
                         f"{spec.n_points} representation points")
    if not np.all(np.isfinite(v_all)):
        raise ValueError(f"{spec.name}: the validation curve contains non-finite values "
                         f"{v_all.tolist()} — the tunnel entrance would be meaningless")
    dep = spec.depth_indices
    v = v_all[dep]                                  # DEPTH AXIS ONLY: Emb, L1..L_N

    def one(t):
        pos = int(tunnel_start(v, t))               # position within the depth axis
        i = dep[pos]                                # -> index into the full point list
        pnt = spec.points[i]
        return {"tolerance": float(t), "index": i, "depth_axis_index": pos, **pnt.as_dict(),
                "val_loss_at_entrance": float(v[pos]),
                "val_loss_at_reference": float(v[-1]),
                "ratio_to_reference": float(v[pos] / v[-1])}

    rec = {
        "definition": "first_crossing_95 (probing.tunnel.tunnel_start, called unchanged)",
        "criterion": "min { l in DEPTH AXIS : L_val(l) <= (1 + tol) * L_val(final block) }",
        "split_used": "validation",
        "test_never_used_for_selection": True,
        "depth_axis_labels": spec.depth_labels,
        "depth_axis_indices": dep,
        "excluded_points": spec.diagnostic_labels,
        "excluded_point_type": HEAD_INPUT_DIAGNOSTIC,
        "head_input_caveat": HEAD_INPUT_CAVEAT,
        "reference_point": spec.reference_label,
        "reference_index": spec.reference_index,
        "reference_is_final_block": True,
        "headline": one(tol),
        "by_tolerance": {f"tol_{t:g}": one(t) for t in tols},
        "val_loss_by_point": [float(x) for x in v_all],
        "val_loss_by_depth": [float(x) for x in v],
        "val_argmin_depth_index": int(np.argmin(v)),
        "val_argmin_point": spec.depth_labels[int(np.argmin(v))],
    }
    # The excluded points' own losses, reported so the diagnostic is available and the
    # exclusion is auditable -- never fed to the criterion.
    rec["head_input_diagnostic"] = {
        spec.points[i].label: {"val_loss": float(v_all[i]),
                               "val_ratio_to_final_block": float(v_all[i] / v[-1])}
        for i in spec.diagnostic_indices}
    if test_losses is not None:
        t_all = np.asarray(test_losses, np.float64)
        if t_all.size != spec.n_points:
            raise ValueError(f"{spec.name}: test curve has {t_all.shape} entries for "
                             f"{spec.n_points} representation points")
        t_ = t_all[dep]
        ls = rec["headline"]["depth_axis_index"]
        rec["test_loss_by_point"] = [float(x) for x in t_all]
        rec["test_loss_by_depth"] = [float(x) for x in t_]
        rec["test_margins"] = [float(x) for x in (t_ / t_[-1] - 1.0)]
        rec["test_criterion_holds"] = bool(np.all(t_[ls:] <= (1.0 + tol) * t_[-1]))
        rec["max_excursion_val"] = float((v[ls:] / v[-1] - 1.0).max())
        rec["max_excursion_test"] = float((t_[ls:] / t_[-1] - 1.0).max())
        rec["test_argmin_point"] = spec.depth_labels[int(np.argmin(t_))]
        for i in spec.diagnostic_indices:
            rec["head_input_diagnostic"][spec.points[i].label]["test_loss"] = float(t_all[i])
            rec["head_input_diagnostic"][spec.points[i].label][
                "test_ratio_to_final_block"] = float(t_all[i] / t_[-1])
    return rec


# --------------------------------------------------------------------------- #
# geometry — both estimators, both splits, raw representations
# --------------------------------------------------------------------------- #
def geometry_block(mats, labels, *, d: int, split: str, variant: str,
                   point_types=None, estimators=("unbiased", "biased"),
                   null_floor_reps: int = 3, seed: int = SEED,
                   row_construction: str = "") -> dict:
    """CKA (every estimator) + entropy effective rank on ONE ordered set of RAW representations.

    ``mats`` is a list of (N, d) float arrays, ONE PER POINT, with MATCHED ROWS: row i of every
    matrix must be the same observation. ``probing.cka.require_matched_rows`` (reached through
    ``cka_layer_matrix``) refuses anything else, which is what makes an accidental cross-point
    misalignment fail loudly instead of producing a plausible heatmap.

    The matrices must be RAW hidden states — never the probe's z-scored features. A
    dimension-wise standardization is not CKA-invariant (CKA is invariant to orthogonal maps and
    ISOTROPIC scaling, not to per-coordinate rescaling) and it changes the spectrum the effective
    rank is computed from. ``_refuse_standardized`` checks for it and raises.

    THE DEPTH AXIS vs THE HEAD INPUT. ``point_types`` (aligned with ``labels``, defaulting to
    all ``block_depth``) marks any final-normalization point. The returned ``cka`` matrices
    cover the DEPTH AXIS ONLY — that is the heatmap a main figure should draw, and it simply
    has no row or column for a normalization to be plotted as a depth. The full matrix,
    including the head-input point, is returned separately as ``cka_with_head_input`` for the
    later native-head/alignment work. CKA is pairwise, so the depth-axis matrix IS the
    submatrix of the full one — nothing is computed twice and the two can never disagree.

    The effective-rank curve covers EVERY point (the head-input rank is a useful diagnostic and
    costs one SVD), with ``point_types`` carried alongside so a main curve filters it out.
    """
    if len(mats) != len(labels):
        raise ValueError(f"{len(mats)} matrices for {len(labels)} labels")
    labels = list(labels)
    point_types = ([BLOCK_DEPTH] * len(labels) if point_types is None else list(point_types))
    if len(point_types) != len(labels):
        raise ValueError(f"{len(point_types)} point types for {len(labels)} labels")
    bad = sorted(set(point_types) - set(POINT_TYPES))
    if bad:
        raise ValueError(f"unknown point types {bad}; known: {POINT_TYPES}")
    mats = [np.asarray(m, np.float64) for m in mats]
    for lab, m in zip(labels, mats):
        if m.ndim != 2 or m.shape[1] != d:
            raise ValueError(f"{lab}: representation must be (N, {d}), got {m.shape}")
    _refuse_standardized(mats, labels)
    n = int(mats[0].shape[0])

    dep = [i for i, t in enumerate(point_types) if t == BLOCK_DEPTH]
    diag = [i for i, t in enumerate(point_types) if t != BLOCK_DEPTH]
    if not dep:
        raise ValueError(f"{variant}/{split}: no block_depth point — there is no depth axis")
    dep_labels = [labels[i] for i in dep]

    cka, cka_full, floors = {}, {}, {}
    for est in estimators:
        M = cka_layer_matrix(mats, estimator=est)          # over EVERY point
        cka[est] = M[np.ix_(dep, dep)]                     # depth axis (exact submatrix)
        if diag:
            cka_full[est] = M
        fl = cka_null_floor(n, d, reps=null_floor_reps, seed=seed, estimator=est)
        off = cka[est][~np.eye(len(dep), dtype=bool)]
        fl["min_offdiagonal_cka"] = float(off.min())
        fl["mean_offdiagonal_cka"] = float(off.mean())
        fl["mean_offdiagonal_above_null_floor"] = float(off.mean() - fl["mean"])
        floors[est] = fl

    er = effective_rank_curve(mats, labels, hidden_dim=d)
    er["point_types"] = point_types
    er["include_in_main_depth_axis"] = [t == BLOCK_DEPTH for t in point_types]

    out = {"variant": variant, "split": split,
           "labels": dep_labels,                           # what `cka` describes
           "all_labels": labels, "point_types": point_types,
           "depth_axis_indices": dep, "head_input_indices": diag,
           "head_input_labels": [labels[i] for i in diag],
           "n_rows": n, "feature_dim": int(d), "estimators": list(estimators),
           "row_construction": row_construction,
           "cka": cka, "cka_null_floor": floors, "effective_rank": er,
           "headline_estimator": "unbiased",
           "cka_axis": "block_depth only (a final normalization is not a model depth)",
           "geometry_caveat": GEOMETRY_CAVEAT,
           "head_input_caveat": HEAD_INPUT_CAVEAT}
    if diag:
        out["cka_with_head_input"] = cka_full
        out["cka_with_head_input_labels"] = labels
    return out


def _refuse_standardized(mats, labels, tol: float = 1e-3) -> None:
    """Refuse a matrix that looks dimension-wise z-scored (per-column mean ~0 AND std ~1).

    Feeding the probe's ``StandardScaler`` output to CKA / effective rank would silently answer a
    different question: per-coordinate rescaling is not a CKA invariance and it flattens the
    spectrum the entropy effective rank measures. The Phase-1 spec says RAW, so this is a
    structural guard rather than a convention.
    """
    for lab, M in zip(labels, mats):
        if M.shape[0] < 3:
            continue
        mu = np.abs(M.mean(axis=0)).max()
        sd = M.std(axis=0)
        if mu < tol and np.abs(sd - 1.0).max() < tol:
            raise ValueError(
                f"{lab}: this representation is dimension-wise standardized (max |column mean| "
                f"{mu:.2e}, max |column std - 1| {np.abs(sd - 1.0).max():.2e}). CKA and entropy "
                "effective rank must be computed on RAW hidden states — a per-coordinate "
                "rescale is not a CKA invariance and it changes the spectrum. Pass the "
                "extracted features, not the probe's scaled ones.")


def stack_rows(F) -> np.ndarray:
    """(n, K, d) -> (n*K, d) row-major, or (n, d) unchanged. ``probing.cka.stack_slots``.

    Row w*K + k is window w, readout position k — the SAME ordering the Chronos-2 forecast slots
    and the TiRex readout positions already use, so the two model lines' row conventions are
    provably identical.
    """
    return stack_slots(F)


# --------------------------------------------------------------------------- #
# uncertainty — the existing cluster bootstrap, one call site
# --------------------------------------------------------------------------- #
def cluster_ci(per_window, cluster_ids, B: int, seed: int, ref_idx: int) -> dict:
    """Per-depth point + 95% CI and the PAIRED delta vs the reference depth.

    ``per_window`` is (L, n) — one row per representation point. One shared multinomial count
    matrix (``probing.stats.cluster_bootstrap_counts``) is used for every depth, which is what
    keeps the delta-vs-last comparison paired. Whole clusters are resampled because the test
    windows are correlated within a series/cluster.
    """
    from probing.stats import ci_bounds, cluster_bootstrap_apply, cluster_bootstrap_counts
    W = np.asarray(per_window, np.float64)
    if W.ndim != 2:
        raise ValueError(f"per-window metrics must be (L, n), got {W.shape}")
    uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    S, L = len(uniq), W.shape[0]
    per_sum = np.zeros((S, L))
    np.add.at(per_sum, inv, W.T)
    per_cnt = np.bincount(inv, minlength=S).astype(np.float64)
    M = cluster_bootstrap_counts(S, B, seed)
    boot = cluster_bootstrap_apply(M, per_sum, per_cnt)
    lo, hi = ci_bounds(boot)
    dlo, dhi = ci_bounds(boot[:, [ref_idx]] - boot)
    point = W.mean(axis=1)
    return {"point": point.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
            "delta_vs_last": (point[ref_idx] - point).tolist(),
            "delta_ci_lo": dlo.tolist(), "delta_ci_hi": dhi.tolist(),
            "reference_index": int(ref_idx), "n_clusters": int(S),
            "n_windows": int(W.shape[1]), "B": int(B), "seed": int(seed)}


def cluster_ratio_ci(num, den, cluster_ids, B: int, seed: int, ref_idx: int) -> dict:
    """Cluster CI for a RATIO-OF-SUMS metric (WQL), formed INSIDE each bootstrap replicate.

    ``num`` is (L, n) per-window numerators and ``den`` is (n,) per-window denominators. WQL is
    ``sum_w num_w / sum_w den_w``, so averaging per-window ratios would answer a different
    question and a CI built that way would be wrong. The same shared count matrix is used for
    every depth, keeping the delta-vs-last comparison paired -- the convention
    ``run_native_head_adapter._boot_ratio`` already uses.
    """
    from probing.stats import ci_bounds, cluster_bootstrap_counts
    N = np.asarray(num, np.float64)
    D = np.asarray(den, np.float64).reshape(-1)
    if N.ndim != 2 or N.shape[1] != D.size:
        raise ValueError(f"numerators {N.shape} incompatible with denominators {D.shape}")
    uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    S, L = len(uniq), N.shape[0]
    nsum = np.zeros((S, L))
    np.add.at(nsum, inv, N.T)
    dsum = np.bincount(inv, weights=D, minlength=S).astype(np.float64)
    M = cluster_bootstrap_counts(S, B, seed)
    boot = (M @ nsum) / (M @ dsum)[:, None]
    lo, hi = ci_bounds(boot)
    dlo, dhi = ci_bounds(boot[:, [ref_idx]] - boot)
    point = N.sum(axis=1) / D.sum()
    return {"point": point.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
            "delta_vs_last": (point[ref_idx] - point).tolist(),
            "delta_ci_lo": dlo.tolist(), "delta_ci_hi": dhi.tolist(),
            "reference_index": int(ref_idx), "n_clusters": int(S),
            "n_windows": int(N.shape[1]), "B": int(B), "seed": int(seed),
            "statistic": "sum(numerator) / sum(denominator), formed inside each replicate"}


# --------------------------------------------------------------------------- #
# reproducibility records
# --------------------------------------------------------------------------- #
def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=REPO_ROOT, check=True, capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return "unknown"
    dirty = run("git", "status", "--porcelain")
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty) if dirty != "unknown" else None,
            "dirty_files": [l[3:] for l in dirty.splitlines()] if dirty not in ("", "unknown")
                           else []}


def _pkg_version(name: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(name)
    except Exception:
        return "not-installed"


def environment_record(device: str | None = None) -> dict:
    """Everything needed to say which software produced a number."""
    import torch
    gpu, cuda = None, None
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            cuda = torch.version.cuda
    except Exception:
        pass
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": np.__version__,
        "sklearn": _pkg_version("scikit-learn"),
        "timesfm": _pkg_version("timesfm"), "tirex-ts": _pkg_version("tirex-ts"),
        "chronos": _pkg_version("chronos-forecasting"),
        "datasets": _pkg_version("datasets"),
        "gpu": gpu, "cuda": cuda, "device_requested": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "hf_home": os.environ.get("HF_HOME"),
        "command_line": " ".join(sys.argv),
        "git": git_state(),
    }


def registry_snapshot(roster_name: str = ROSTER) -> dict:
    """The exact dataset facts this run used, copied into the results tree.

    A snapshot, not a pointer: if probing/registry.py later changes, the run's own numbers stay
    interpretable against the table that actually produced them.
    """
    return {"roster": roster_name, "tags": registry.roster(roster_name),
            "datasets": {t: registry.spec(t).as_dict() for t in registry.roster(roster_name)},
            "mase_definition": registry.MASE_DEFINITION,
            "statuses": list(registry.STATUSES),
            "evidence_scopes": list(registry.EVIDENCE_SCOPES),
            "models": list(registry.MODELS),
            "provenance": {f"{t}|{m}": registry.provenance(t, m).as_dict()
                           for t in registry.roster(roster_name) for m in MODELS},
            "unverified_provenance": [list(x) for x in
                                      registry.unverified_provenance(roster_name, MODELS)]}


def registry_hash(roster_name: str = ROSTER) -> str:
    """Stable digest of the dataset facts + provenance a run depends on."""
    snap = registry_snapshot(roster_name)
    blob = json.dumps(snap, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def provenance_rows(roster_name: str = ROSTER) -> list[dict]:
    """The full model x dataset provenance table, one row per cell (3 x 14 = 42).

    Two axes, never collapsed. ``not_listed`` is NOT converted to OOD or unseen anywhere.
    """
    rows = []
    for tag in registry.roster(roster_name):
        s = registry.spec(tag)
        for m in MODELS:
            p = registry.provenance(tag, m)
            rows.append({"model": m, "dataset": tag, "display_name": s.display_name,
                         "role": s.role, "domain": s.domain, "freq": s.freq,
                         "seasonal_m": s.seasonal_m,
                         "status": p.status, "evidence_scope": p.evidence_scope,
                         "verified": p.verified, "citation": p.citation})
    return rows


# --------------------------------------------------------------------------- #
# cell identity
# --------------------------------------------------------------------------- #
def cell_config_hash(cfg: dict) -> str:
    """Digest of everything that would make two cells scientifically different.

    Deliberately EXCLUDES the git commit and the wall-clock: a docs commit must not invalidate
    43 GPU-hours. What it includes is the protocol version (bumped by hand when semantics
    change), the geometry (C/H/Q, the quantile vector), the checkpoint, the extraction
    parameters that provably move numbers (batch size, backend, rollout mode, feature dtype),
    the probe protocol (wd grid / epochs / lr / seed), the bootstrap B, the dataset window
    digest, and the registry hash.
    """
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# atomic, resumable cell store
# --------------------------------------------------------------------------- #
#: Artifacts a cell MUST have produced before it may be marked COMPLETE. A missing one means the
#: cell failed partway; marking it complete anyway would make a later combined table quietly
#: report an incomplete run as a finished one.
REQUIRED_ARTIFACTS = ("cell_config.json", "summary.json", "layer_metrics.csv", "tunnel.json",
                      "probe_hparams.json", "effective_rank.csv", "spectral_metrics.npz",
                      "bootstrap_inputs.npz", "cka/cka_metadata.json")

STATUSES = ("pending", "running", "complete", "failed")


def atomic_write_json(path: Path, obj) -> Path:
    """Write JSON via a temp file + os.replace, so a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_csv(path: Path, rows, fieldnames=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = list(fieldnames) if fieldnames else (list(rows[0]) if rows else [])
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return path


@dataclass
class CellStore:
    """The on-disk contract for one ``model x dataset`` cell, and the resume policy.

    THE INVARIANT: a cell directory exists in its final location ONLY if it is complete.

    A cell is built entirely inside ``<model>/<dataset>.building-<pid>/``. When the work
    finishes, every required artifact is validated, ``COMPLETE`` is written INSIDE the staging
    directory, and only then is the directory moved into place with ``os.replace`` (atomic on
    POSIX). A preemption, a timeout or an exception therefore leaves behind a ``.building-*``
    directory that the next run deletes — never a partial cell that a combined table would read
    as finished.

    RESUME. On restart a cell whose ``COMPLETE`` carries the SAME config hash is skipped after
    its artifacts are re-validated. A cell whose hash DIFFERS is refused loudly: silently
    overwriting it would mix two protocols in one results tree, and silently keeping it would
    report the old protocol's numbers under the new run's manifest. Feature caches live
    separately (in $SCRATCH) and are keyed by their own metadata, so a recomputed cell still
    reuses valid extractions and never re-runs a backbone it does not have to.
    """
    root: Path
    model: str
    dataset: str

    @property
    def final(self) -> Path:
        return Path(self.root) / self.model / self.dataset

    @property
    def marker(self) -> Path:
        return self.final / "COMPLETE"

    def staging(self) -> Path:
        return Path(self.root) / self.model / f"{self.dataset}.building-{os.getpid()}"

    # -- state ------------------------------------------------------------- #
    def read_marker(self) -> dict | None:
        if not self.marker.exists():
            return None
        try:
            return json.loads(self.marker.read_text())
        except Exception:
            return {"config_hash": "unreadable"}

    def status(self, config_hash: str) -> tuple[str, str]:
        """``(status, reason)`` for this cell under ``config_hash``."""
        mk = self.read_marker()
        if mk is None:
            return "pending", "no COMPLETE marker"
        if mk.get("config_hash") != config_hash:
            return "incompatible", (f"COMPLETE marker carries config hash "
                                    f"{mk.get('config_hash')} but this run is {config_hash}")
        missing = [a for a in REQUIRED_ARTIFACTS if not (self.final / a).exists()]
        if missing:
            return "corrupt", f"COMPLETE but missing artifacts {missing}"
        return "complete", "validated"

    # -- writing ----------------------------------------------------------- #
    def begin(self) -> Path:
        """Fresh staging directory (any leftover from a killed run is removed first)."""
        stage = self.staging()
        if stage.exists():
            shutil.rmtree(stage)
        (stage / "cka").mkdir(parents=True, exist_ok=True)
        (stage / "probe_artifacts").mkdir(parents=True, exist_ok=True)
        return stage

    def validate(self, stage: Path) -> list[str]:
        return [a for a in REQUIRED_ARTIFACTS if not (Path(stage) / a).exists()]

    def commit(self, stage: Path, config_hash: str, extra: dict | None = None) -> Path:
        """Validate, write COMPLETE inside the staging dir, then move it into place atomically."""
        stage = Path(stage)
        missing = self.validate(stage)
        if missing:
            raise RuntimeError(
                f"{self.model}/{self.dataset}: refusing to mark COMPLETE — these required "
                f"artifacts were not produced: {missing}. The cell stays incomplete so a "
                "combined table cannot read a failed run as a finished one.")
        marker = {"config_hash": config_hash, "model": self.model, "dataset": self.dataset,
                  "protocol": PHASE1_PROTOCOL_VERSION,
                  "completed_utc": datetime.datetime.now(datetime.timezone.utc)
                                   .isoformat(timespec="seconds"),
                  "artifacts": sorted(str(p.relative_to(stage))
                                      for p in stage.rglob("*") if p.is_file()),
                  **(extra or {})}
        (stage / "COMPLETE").write_text(json.dumps(marker, indent=2, default=_json_default))
        final = self.final
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        os.replace(stage, final)
        return final

    def abandon(self, stage: Path) -> None:
        stage = Path(stage)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def clean_staging(root: Path) -> list[str]:
    """Remove leftover ``*.building-*`` directories from killed runs. Returns what it removed."""
    removed = []
    root = Path(root)
    for model in MODELS:
        d = root / model
        if not d.exists():
            continue
        for p in d.glob("*.building-*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p.relative_to(root)))
    return removed
