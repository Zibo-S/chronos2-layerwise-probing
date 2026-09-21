"""Frozen TiRex: geometry, native forecasting path, and layer-wise representation extraction.

The third model of the layer-wise forecasting paper, alongside Chronos-2 (encoder + K=4 forecast
slots) and TimesFM-3 (decoder + ONE last-context token).  Every Chronos-2 and TimesFM-3 file is
imported read-only and never modified.

    Chronos-2  : the native head reads K=4 forecast-SLOT states for H=64  -> probe those slots
    TimesFM-3  : the native head reads ONE last-context state for H=64    -> probe that token
    TiRex      : the native head reads K=2 PATCH states for H=64          -> probe those two

GEOMETRY -- RE-DERIVED FROM THE INSTALLED CHECKPOINT, NOT FROM THE PAPER
-----------------------------------------------------------------------
Measured on ``NX-AI/TiRex`` via ``tirex-ts`` (see ``discover_geometry``); every value below is
re-asserted against the loaded model on each run:

    input_patch_size = output_patch_size = 32     quantiles = 0.1 .. 0.9   (Q=9, median idx 4)
    embedding_dim    = 512     num_blocks = 12    num_heads = 4   input_ff_dim = 2048
    train_ctx_len    = 2048    <-- THE ONE THAT BREAKS THE OBVIOUS ASSUMPTION

``TiRexZero._forecast_single_step`` calls ``_adjust_context_length(context, train_ctx_len,
train_ctx_len)``, which forces EVERY context to exactly ``train_ctx_len`` by LEFT-PADDING with
NaN.  A C=512 context therefore does NOT become a 16-token sequence: it becomes

    [1536 NaN pad = 48 all-masked patches][512 real values = 16 real patches]   -> 64 tokens

so the LAST REAL CONTEXT PATCH is token **63**, not token 15.  The NaN pad patches are mapped to
(values=0, mask=0) by ``_forward_model_tokenized`` and are legitimate model input, not an
artifact we introduced -- this IS what TiRex does with a 512-point context.

TWO ROLLOUT MODES, BOTH IMPLEMENTED AND BOTH PROBEABLE
------------------------------------------------------
``prediction_length=64`` needs ceil(64/32) = 2 output patches.  TiRex produces them two ways, and
the package DEFAULT IS NOT THE ONE THE PAPER DESCRIBES.  ``ROLLOUT_MODES`` names both; a run picks
one with ``--rollout-mode`` and it is recorded in the cache key, so the two can never be mixed.

``single_pass`` -- ``max_accelerated_rollout_steps = K = 2``.  ONE forward pass.
    ``_forward_model_tokenized`` appends (K-1)=1 all-NaN token, giving 65 tokens, and reads the
    LAST K=2:
        token 63 = last REAL context patch        -> y[T   : T+32]
        token 64 = first MASKED future patch      -> y[T+32: T+64]
    Both states come from ONE recurrent scan under ONE normalization.

``two_pass``   -- ``max_accelerated_rollout_steps = 1``, the PACKAGE DEFAULT.  K separate passes,
    each 64 tokens, each reading its own token 63, each with its OWN (loc, scale):
        pass 0: context x[T-512:T]                 -> token 63 is the LAST REAL patch
        pass 1: context x[T-512:T] ++ 32 x NaN     -> token 63 is the appended MISSING patch
                (47 pad patches, real at 47..62, the NaN patch at 63)

WHAT PASS 1 CONSUMES -- DERIVED FROM THE PACKAGE, NOT INFERRED.  ``_forecast_tensor`` line:

    context = torch.cat([context, torch.full_like(prediction[:, 0, :], fill_value=torch.nan)], -1)

``torch.full_like(X, fill_value=nan)`` takes X's SHAPE/dtype/device and fills it entirely with
NaN, so ``prediction[:, 0, :]`` (the tau=0.1 row of the previous forecast) is a SHAPE TEMPLATE
ONLY and its values are discarded.  **Pass 1 therefore consumes MISSING VALUES, never predicted
ones** -- TiRex's rollout is not autoregressive in the usual sense, it re-asks the model with the
horizon marked missing.  ``assert_rollout_appends_missing`` proves this numerically (the appended
block contains no finite value and does not equal the previous forecast) rather than trusting the
reading.  A consequence worth stating: because the appended block is always NaN regardless of what
the model predicted, every pass's context -- and hence every pass's (loc, scale) -- can be
constructed WITHOUT running the model, which is what ``build_targets`` does.

The per-pass normalizations are mathematically the SAME quantity (mean/population-std of the 512
real values: the NaN pad is excluded by ``nanmean`` and no real value is ever truncated, which
``TiRexGeometry`` asserts via C + (K-1)*P <= train_ctx_len).  They are NOT bit-identical -- the
NaN slots sit in different places, so the float32 reduction order differs, measured at 6e-8
relative.  Each chunk is nevertheless normalized and de-normalized with ITS OWN pass's statistics,
because that is what the native path does and the cost of being faithful is zero.

The two modes agree EXACTLY on the first readout state (the sLSTM is causal, so an appended token
cannot influence token 63) and differ on the second; the forecasts differ on the last 32 steps
only.  ``rollout_mode_gap`` reports the forecast difference per dataset.

NORMALIZATION -- READ OFF ``PatchedTokenizer``, NOT GUESSED
-----------------------------------------------------------
``StandardScaler.get_loc_scale`` (patcher.py) uses ``nanmean`` over the WHOLE adjusted context
row, so the NaN pad contributes nothing and

    loc = mean(x[T-C:T])        scale = population std(x[T-C:T])        (verified to 1e-6/1e-5)
    scale <- |loc| + 1e-5  wherever scale == 0                          (degenerate guard)

The future is never passed to the model and never enters loc/scale (asserted structurally by
``build_targets``, which takes the context and the future as separate arrays, and tested).

NUMERICS
--------
``sLSTMCellTorch`` runs the recurrence with bfloat16 storage (pointwise math promoted to float32
each step).  Representations are float32 tensors carrying bf16-rounded values.  This is a
property of the ``torch`` backend; the ``cuda`` backend is xLSTM's own kernel and will NOT agree
bit-for-bit.  The backend is recorded in every cache and summary, and ``verify_native_head``
re-proves the readout identity on whichever backend is actually running.

Nothing here silently repairs an unexpected shape, token index, quantile order or cache: every
assumption is an explicit raise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---- one source of truth for every geometric constant -------------------------------------- #
DEFAULT_CHECKPOINT = "NX-AI/TiRex"
CACHE_VERSION = "tirex-q1-shared-patch-v1"

CONTEXT_LEN = 512          # C -- the project-wide context, shared with Chronos-2 / TimesFM-3
HORIZON = 64               # H -- the project-wide horizon
ROLLOUT_STEPS = 2          # max_accelerated_rollout_steps for single_pass: K adjacent readouts

# The two native ways to cover H > output_patch. See the module docstring; neither is "the" right
# one a priori, so both are implemented and the choice is an explicit, recorded run parameter.
ROLLOUT_SINGLE = "single_pass"     # max_accelerated_rollout_steps = K  (one 65-token pass)
ROLLOUT_TWO = "two_pass"           # max_accelerated_rollout_steps = 1  (K passes, PACKAGE DEFAULT)
ROLLOUT_MODES = (ROLLOUT_SINGLE, ROLLOUT_TWO)


def check_rollout_mode(mode: str) -> str:
    if mode not in ROLLOUT_MODES:
        raise ValueError(f"unknown rollout mode {mode!r}; choose one of {ROLLOUT_MODES}")
    return mode

# Expected checkpoint geometry. These are ASSERTIONS about NX-AI/TiRex, not configuration:
# a checkpoint that disagrees is a different model and must not be probed with this code.
EXPECT_INPUT_PATCH = 32
EXPECT_OUTPUT_PATCH = 32
EXPECT_EMBEDDING_DIM = 512
EXPECT_NUM_BLOCKS = 12
EXPECT_NUM_HEADS = 4
EXPECT_INPUT_FF_DIM = 2048
EXPECT_TRAIN_CTX_LEN = 2048
EXPECT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

MODEL_DIMS = EXPECT_EMBEDDING_DIM
NUM_BLOCKS = EXPECT_NUM_BLOCKS
SCALE_EPS = 1e-5           # PatchedTokenizer's own degenerate-scale epsilon

__all__ = [
    "DEFAULT_CHECKPOINT", "CACHE_VERSION", "CONTEXT_LEN", "HORIZON", "ROLLOUT_STEPS",
    "ROLLOUT_SINGLE", "ROLLOUT_TWO", "ROLLOUT_MODES", "check_rollout_mode",
    "native_forward_two_pass", "verify_native_head_two_pass", "assert_rollout_appends_missing",
    "scaler_states_for_mode", "readouts_from_reps",
    "MODEL_DIMS", "NUM_BLOCKS", "SCALE_EPS", "TiRexGeometry", "RepPoint", "REP_POINTS",
    "REP_NAMES", "REP_SLUGS", "NUM_POINTS", "rep_depth_table", "get_model", "tirex_version",
    "assert_frozen", "assert_no_grads", "discover_geometry", "assert_native_geometry",
    "register_rep_hooks", "native_forward", "apply_native_head", "verify_native_head",
    "rollout_mode_gap", "compare_backends", "build_targets", "assert_target_roundtrip",
    "denormalize", "normalize_raw", "rollout_contexts", "assert_two_pass_matches_package",
    "extract_window_features", "cached_features", "cache_metadata", "cache_root", "read_cache",
    "head_checksum",
]


# --------------------------------------------------------------------------- #
# representation points + the depth convention
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepPoint:
    """One probed representation depth.

    ``block_index``   architectural depth in BLOCKS: 0 for the patch embedding, k for Lk, and
                      ``num_blocks`` for the final RMSNorm (which normalizes block 12's output
                      and contributes no block of its own).
    ``position_index`` ordinal in the extracted sequence, 0 .. NUM_POINTS-1. Always monotone.
    ``kind``          "embedding" | "block" | "final_norm".
    """
    name: str
    slug: str
    kind: str
    block_index: int
    position_index: int
    is_native_readout: bool = False

    @property
    def relative_depth(self) -> float:
        """block_index / num_blocks in [0, 1]. THE cross-model depth coordinate: it is the
        fraction of the residual stream traversed, which is the only quantity Chronos-2 (12
        blocks), TimesFM-3 (20 blocks) and TiRex (12 blocks) share. L12 and L12+RMS both map to
        1.0 -- that is correct and deliberate (the RMSNorm adds no depth); ``kind`` distinguishes
        them, and ``relative_position`` gives a strictly monotone alternative for plotting."""
        return self.block_index / NUM_BLOCKS

    @property
    def relative_position(self) -> float:
        """position_index / (NUM_POINTS - 1) in [0, 1]. Strictly increasing, so it is the safe
        x-axis for a curve; NOT comparable across models with different point sets."""
        return self.position_index / (NUM_POINTS - 1)

    def as_dict(self) -> dict:
        return {"name": self.name, "slug": self.slug, "kind": self.kind,
                "block_index": self.block_index, "position_index": self.position_index,
                "relative_depth": self.relative_depth,
                "relative_position": self.relative_position,
                "is_native_readout": self.is_native_readout}


def _build_rep_points() -> tuple[RepPoint, ...]:
    pts = [RepPoint("Emb", "Emb", "embedding", 0, 0)]
    pts += [RepPoint(f"L{k}", f"L{k}", "block", k, k) for k in range(1, NUM_BLOCKS + 1)]
    pts.append(RepPoint(f"L{NUM_BLOCKS}+RMS", f"L{NUM_BLOCKS}_RMS", "final_norm",
                        NUM_BLOCKS, NUM_BLOCKS + 1, is_native_readout=True))
    return tuple(pts)


REP_POINTS = _build_rep_points()
NUM_POINTS = len(REP_POINTS)                       # 14 = Emb + L1..L12 + L12+RMS
REP_NAMES = tuple(p.name for p in REP_POINTS)
REP_SLUGS = tuple(p.slug for p in REP_POINTS)
NATIVE_READOUT_POINT = NUM_POINTS - 1              # index of L12+RMS, what output_patch_embedding reads


def rep_depth_table() -> list[dict]:
    """The full depth table, serialized into every summary so the figure/table code never has to
    re-derive a normalized depth (and cannot disagree with this file about it)."""
    return [p.as_dict() for p in REP_POINTS]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TiRexGeometry:
    """Derived, asserted token layout for one (C, H) configuration.

    Everything is COMPUTED from the checkpoint's own (input_patch_size, output_patch_size,
    train_ctx_len) -- no index is written down twice, and `15`/`16`/`63`/`64` appear nowhere
    else in the codebase.
    """
    C: int
    H: int
    input_patch: int
    output_patch: int
    train_ctx_len: int
    num_quantiles: int
    median_index: int

    # derived (filled by __post_init__ via object.__setattr__ since the dataclass is frozen)
    pad_len: int = field(init=False)
    n_pad_patches: int = field(init=False)
    n_real_context_patches: int = field(init=False)
    n_context_tokens: int = field(init=False)
    n_forecast_patches: int = field(init=False)
    n_tokens: int = field(init=False)
    readout_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        s = object.__setattr__
        if self.input_patch != self.output_patch:
            raise ValueError(f"TiRex requires input_patch == output_patch, got "
                             f"{self.input_patch} != {self.output_patch}")
        P = self.input_patch
        if self.C % P:
            raise ValueError(f"context length {self.C} is not a multiple of patch size {P}; "
                             "TiRex's tokenizer asserts divisibility and would raise")
        if self.H % self.output_patch:
            raise ValueError(
                f"horizon {self.H} is not a multiple of output_patch {self.output_patch}. TiRex "
                "would forecast ceil(H/P) patches and TRIM the tail; this experiment refuses to "
                "probe a silently truncated horizon -- choose H as a multiple of the patch size")
        if self.C > self.train_ctx_len:
            raise ValueError(f"context {self.C} exceeds train_ctx_len {self.train_ctx_len}; "
                             "_adjust_context_length would TRUNCATE it from the left")
        if (self.train_ctx_len - self.C) % P:
            raise ValueError(f"NaN pad length {self.train_ctx_len - self.C} is not a whole number "
                             f"of patches of size {P}")
        s(self, "pad_len", self.train_ctx_len - self.C)
        s(self, "n_pad_patches", (self.train_ctx_len - self.C) // P)
        s(self, "n_real_context_patches", self.C // P)
        s(self, "n_context_tokens", self.train_ctx_len // P)
        K = self.H // self.output_patch
        s(self, "n_forecast_patches", K)
        # _forward_model_tokenized appends (K-1) all-NaN rollout tokens and reads the last K.
        n_tokens = self.n_context_tokens + K - 1
        s(self, "n_tokens", n_tokens)
        s(self, "readout_indices", tuple(range(n_tokens - K, n_tokens)))
        if self.readout_indices[0] != self.n_context_tokens - 1:
            raise RuntimeError(
                f"first readout token {self.readout_indices[0]} is not the last real context "
                f"patch {self.n_context_tokens - 1} -- the rollout layout changed")
        # two_pass precondition: pass k prepends (K-1)*P NaN slots to the RIGHT of the context,
        # so the padded length only stays >= C if C + (K-1)*P fits inside train_ctx_len. If it did
        # not, _adjust_context_length would LEFT-TRUNCATE real values and the passes would then
        # normalize over DIFFERENT data -- a silent change of target space, not a rounding effect.
        if self.C + (K - 1) * P > self.train_ctx_len:
            raise ValueError(
                f"two_pass rollout would truncate real context: C + (K-1)*P = "
                f"{self.C + (K - 1) * P} exceeds train_ctx_len {self.train_ctx_len}, so later "
                "passes would see fewer real values and derive a different (loc, scale)")
        if self.median_index < 0 or self.median_index >= self.num_quantiles:
            raise ValueError(f"median index {self.median_index} outside 0..{self.num_quantiles-1}")

    @property
    def rollout_steps(self) -> int:
        """max_accelerated_rollout_steps that yields ONE forward pass covering the whole H."""
        return self.n_forecast_patches

    @property
    def first_readout(self) -> int:
        """Token index of the LAST REAL CONTEXT patch (predicts the first output patch)."""
        return self.readout_indices[0]

    @property
    def masked_readouts(self) -> tuple[int, ...]:
        """Token indices of the MASKED FUTURE patches (all readouts after the first)."""
        return self.readout_indices[1:]

    def target_slice(self, k: int) -> slice:
        """Horizon steps predicted by readout token k (0-based within readout_indices)."""
        return slice(k * self.output_patch, (k + 1) * self.output_patch)

    # ---- two_pass layout: K separate passes, each reading its OWN last token ---------------- #
    @property
    def two_pass_n_tokens(self) -> int:
        """Tokens per pass in two_pass mode. Every pass is padded to train_ctx_len and gets NO
        rollout pad (new_patch_count=1), so all K passes have the same token count."""
        return self.n_context_tokens

    @property
    def two_pass_readout_index(self) -> int:
        """``_forecast_single_step`` reads ``prediction[:, :, -1:, :]`` -- the LAST token -- in
        every pass. Derived, not written down."""
        return self.two_pass_n_tokens - 1

    def two_pass_layout(self, k: int) -> dict:
        """Token layout of pass k (0-based), derived from the package's own context construction:
        pass k is handed ``x[T-C:T] ++ (k*P) NaN`` and pads that to train_ctx_len on the LEFT."""
        if not 0 <= k < self.n_forecast_patches:
            raise ValueError(f"pass {k} outside 0..{self.n_forecast_patches - 1}")
        P = self.input_patch
        ctx_len = self.C + k * P
        pad = self.train_ctx_len - ctx_len
        n_pad_patches = pad // P
        real_first = n_pad_patches
        real_last = n_pad_patches + self.n_real_context_patches - 1
        readout = self.two_pass_readout_index
        return {"pass": k, "context_len": ctx_len, "pad_len": pad,
                "n_pad_patches": n_pad_patches,
                "real_token_range": [real_first, real_last],
                "n_appended_missing_patches": k,
                "n_tokens": self.two_pass_n_tokens, "readout_index": readout,
                "readout_is_last_real_context_patch": readout == real_last,
                "readout_is_appended_missing_patch": readout > real_last,
                "target_slice": [k * self.output_patch, (k + 1) * self.output_patch]}

    def readout_token_indices(self, mode: str) -> tuple[int, ...]:
        """The forecast-producing token index per output patch, for either rollout mode.
        single_pass: K tokens of ONE 65-token sequence. two_pass: the SAME index in each of K
        separate 64-token sequences."""
        if check_rollout_mode(mode) == ROLLOUT_SINGLE:
            return self.readout_indices
        return tuple(self.two_pass_readout_index for _ in range(self.n_forecast_patches))

    def n_tokens_for(self, mode: str) -> int:
        return (self.n_tokens if check_rollout_mode(mode) == ROLLOUT_SINGLE
                else self.two_pass_n_tokens)

    def as_dict(self) -> dict:
        return {"C": self.C, "H": self.H, "input_patch": self.input_patch,
                "output_patch": self.output_patch, "train_ctx_len": self.train_ctx_len,
                "pad_len": self.pad_len, "n_pad_patches": self.n_pad_patches,
                "n_real_context_patches": self.n_real_context_patches,
                "n_context_tokens": self.n_context_tokens,
                "n_forecast_patches": self.n_forecast_patches, "n_tokens": self.n_tokens,
                "readout_indices": list(self.readout_indices),
                "first_readout_is_last_real_context_patch": self.first_readout,
                "masked_future_readouts": list(self.masked_readouts),
                "rollout_steps": self.rollout_steps,
                "rollout_modes": list(ROLLOUT_MODES),
                "two_pass_n_tokens": self.two_pass_n_tokens,
                "two_pass_readout_index": self.two_pass_readout_index,
                "two_pass_layout": [self.two_pass_layout(k)
                                    for k in range(self.n_forecast_patches)],
                "num_quantiles": self.num_quantiles, "median_index": self.median_index,
                "model_dims": MODEL_DIMS, "num_blocks": NUM_BLOCKS,
                "representation_points": list(REP_NAMES)}


def geometry_from_model(model, C: int = CONTEXT_LEN, H: int = HORIZON) -> TiRexGeometry:
    """Build the geometry FROM THE LOADED CHECKPOINT. The only supported construction path --
    there is no constructor that takes patch sizes as free parameters."""
    c = model.config
    q = list(c.quantiles)
    med = [i for i, v in enumerate(q) if abs(v - 0.5) < 1e-9]
    if not med:
        raise RuntimeError(f"TiRex quantiles {q} contain no exact 0.5 level; the Q=1 median "
                           "probe and every MASE number depend on it")
    return TiRexGeometry(C=C, H=H, input_patch=c.input_patch_size,
                         output_patch=c.output_patch_size, train_ctx_len=c.train_ctx_len,
                         num_quantiles=len(q), median_index=med[0])


# --------------------------------------------------------------------------- #
# model loading / freezing
# --------------------------------------------------------------------------- #
def tirex_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("tirex-ts")
    except Exception:                                    # pragma: no cover - env dependent
        return "unknown"


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None,
              backend: str = "torch"):
    """Load TiRex, put it in eval mode and FREEZE it. Freezing is explicit because
    ``load_model`` returns a model whose parameters have ``requires_grad=True`` (measured), and
    an accidentally-trainable backbone would silently invalidate every probe."""
    from tirex import load_model
    device = resolve_device(device)
    if backend not in ("torch", "cuda"):
        raise ValueError(f"backend must be 'torch' or 'cuda', got {backend!r}")
    model = load_model(checkpoint, device=device, backend=backend)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert_frozen(model)
    return model


def assert_frozen(model) -> dict:
    """Contract 20: no backbone parameter may be trainable, and the model must be in eval mode."""
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"TiRex is NOT frozen: {len(live)} parameters require grad "
                           f"(e.g. {live[:4]}). Probing a trainable backbone is invalid.")
    if model.training:
        raise RuntimeError("TiRex is in train() mode; call .eval() before extracting features")
    return {"frozen": True, "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "training_mode": False}


def assert_no_grads(model) -> None:
    """Contract: probe training must never accumulate gradients in the backbone."""
    bad = [n for n, p in model.named_parameters() if p.grad is not None]
    if bad:
        raise RuntimeError(f"TiRex accumulated gradients in {len(bad)} backbone parameters "
                           f"(e.g. {bad[:4]}) -- the probe optimizer is touching the backbone")


def head_checksum(model) -> str:
    """Stable hash of the frozen native output pathway (output_patch_embedding + out_norm).
    Recorded before and after every run so a mutated head cannot go unnoticed."""
    h = hashlib.sha256()
    for mod_name in ("out_norm", "output_patch_embedding"):
        mod = getattr(model, mod_name)
        for pname, p in sorted(mod.named_parameters()):
            h.update(f"{mod_name}.{pname}".encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# geometry discovery (section B of the spec) + hard assertions
# --------------------------------------------------------------------------- #
def discover_geometry(model, geom: TiRexGeometry | None = None) -> dict:
    """Record EVERYTHING the spec asks to see about the loaded checkpoint. Pure introspection:
    no forward pass, no assertion. ``assert_native_geometry`` is the gate; this is the report."""
    c = model.config
    bk = dict(c.block_kwargs)
    geom = geom or geometry_from_model(model)
    blk = model.blocks[0]
    return {
        "package": "tirex-ts", "package_version": tirex_version(),
        "torch_version": torch.__version__,
        "checkpoint": DEFAULT_CHECKPOINT,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "num_blocks": len(model.blocks),
        "embedding_dim": bk.get("embedding_dim"),
        "num_heads": bk.get("num_heads"),
        "input_patch_size": c.input_patch_size,
        "output_patch_size": c.output_patch_size,
        "input_ff_dim": c.input_ff_dim,
        "train_ctx_len": c.train_ctx_len,
        "nan_mask_value": c.nan_mask_value,
        "quantiles": list(c.quantiles),
        "num_quantiles": len(c.quantiles),
        "median_index": geom.median_index,
        "tokenizer_class": type(model.tokenizer).__name__,
        "tokenizer_patch_size": model.tokenizer.patch_size,
        "input_embedding_class": f"{type(model.input_patch_embedding).__module__}."
                                 f"{type(model.input_patch_embedding).__name__}",
        "input_embedding_in_features": model.input_patch_embedding.hidden_layer.in_features,
        "final_norm_class": f"{type(model.out_norm).__module__}.{type(model.out_norm).__name__}",
        "final_norm_features": int(model.out_norm.weight.shape[0]),
        "output_head_class": f"{type(model.output_patch_embedding).__module__}."
                             f"{type(model.output_patch_embedding).__name__}",
        "output_head_out_features": model.output_patch_embedding.output_layer.out_features,
        "block_class": f"{type(blk).__module__}.{type(blk).__name__}",
        "block_children": [n for n, _ in blk.named_children()],
        "slstm_backend": getattr(getattr(blk.slstm_layer, "slstm_cell", None), "backend", None),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "geometry": geom.as_dict(),
        "representation_points": rep_depth_table(),
        "head_checksum": head_checksum(model),
    }


def assert_native_geometry(model, geom: TiRexGeometry) -> dict:
    """Fail LOUDLY if the loaded checkpoint is not the architecture this experiment is
    specified for. Every check names the offending value; nothing is repaired."""
    c = model.config
    bk = dict(c.block_kwargs)
    checks = [
        (c.input_patch_size == EXPECT_INPUT_PATCH,
         f"input_patch_size={c.input_patch_size}, expected {EXPECT_INPUT_PATCH}"),
        (c.output_patch_size == EXPECT_OUTPUT_PATCH,
         f"output_patch_size={c.output_patch_size}, expected {EXPECT_OUTPUT_PATCH}"),
        (c.input_patch_size == c.output_patch_size,
         "input_patch_size != output_patch_size (TiRexZero asserts they are equal)"),
        (bk.get("embedding_dim") == EXPECT_EMBEDDING_DIM,
         f"embedding_dim={bk.get('embedding_dim')}, expected {EXPECT_EMBEDDING_DIM}"),
        (len(model.blocks) == EXPECT_NUM_BLOCKS,
         f"num_blocks={len(model.blocks)}, expected {EXPECT_NUM_BLOCKS}"),
        (bk.get("num_heads") == EXPECT_NUM_HEADS,
         f"num_heads={bk.get('num_heads')}, expected {EXPECT_NUM_HEADS}"),
        (c.input_ff_dim == EXPECT_INPUT_FF_DIM,
         f"input_ff_dim={c.input_ff_dim}, expected {EXPECT_INPUT_FF_DIM}"),
        (c.train_ctx_len == EXPECT_TRAIN_CTX_LEN,
         f"train_ctx_len={c.train_ctx_len}, expected {EXPECT_TRAIN_CTX_LEN} -- the readout token "
         "indices are derived from it, so a different value changes the whole layout"),
        (tuple(float(x) for x in c.quantiles) == EXPECT_QUANTILES,
         f"quantiles={list(c.quantiles)}, expected {list(EXPECT_QUANTILES)}"),
        (int(model.out_norm.weight.shape[0]) == EXPECT_EMBEDDING_DIM,
         f"out_norm features={int(model.out_norm.weight.shape[0])}"),
        (model.output_patch_embedding.output_layer.out_features
         == len(c.quantiles) * c.output_patch_size,
         f"output head emits {model.output_patch_embedding.output_layer.out_features}, expected "
         f"{len(c.quantiles) * c.output_patch_size} = Q*P"),
        (model.tokenizer.patch_size == c.input_patch_size,
         f"tokenizer patch {model.tokenizer.patch_size} != config {c.input_patch_size}"),
        # geometry consistency
        (geom.input_patch == c.input_patch_size and geom.train_ctx_len == c.train_ctx_len,
         "TiRexGeometry was not derived from this checkpoint"),
        (geom.n_real_context_patches == geom.C // c.input_patch_size,
         f"n_real_context_patches={geom.n_real_context_patches}"),
        (geom.rollout_steps == ROLLOUT_STEPS,
         f"rollout_steps={geom.rollout_steps}, expected {ROLLOUT_STEPS} for H={geom.H}"),
        (len(geom.readout_indices) == geom.n_forecast_patches,
         f"{len(geom.readout_indices)} readout tokens for {geom.n_forecast_patches} patches"),
    ]
    bad = [m for ok, m in checks if not ok]
    if bad:
        raise RuntimeError("TiRex checkpoint geometry is INCOMPATIBLE with this experiment:\n  "
                           + "\n  ".join(bad))
    return {"geometry_asserted": True, **geom.as_dict()}


# --------------------------------------------------------------------------- #
# hooks -- one per representation point
# --------------------------------------------------------------------------- #
def register_rep_hooks(model, store: dict):
    """Forward hooks capturing exactly ``REP_POINTS``, in order.

        Emb      <- model.input_patch_embedding   (the ResidualBlock output that ENTERS block 1)
        Lk       <- model.blocks[k-1]             (the COMPLETE block: post-sLSTM AND post-FFN
                                                   residual; sLSTMBlock.forward returns
                                                   x + slstm(norm(x)) then x + ffn(norm(x)))
        L12+RMS  <- model.out_norm                (what output_patch_embedding consumes)

    A forward hook fires on the module's OUTPUT, so none of these can accidentally capture a
    pre-block tensor; ``tests`` additionally cross-checks L1..L12 against the model's own
    ``_forward_model(..., return_all_hidden=True)`` stack.

    Returns the handle list -- the caller MUST remove them (use ``native_forward``, which does).
    """
    def mk(name):
        def hook(_mod, _inp, out):
            store[name] = out.detach()
        return hook

    handles = [model.input_patch_embedding.register_forward_hook(mk("Emb"))]
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(mk(f"L{i + 1}")))
    handles.append(model.out_norm.register_forward_hook(mk(f"L{NUM_BLOCKS}+RMS")))
    return handles


# --------------------------------------------------------------------------- #
# the native forward pass
# --------------------------------------------------------------------------- #
def scaler_state(model, ctx: torch.Tensor, geom: TiRexGeometry):
    """TiRex's own (loc, scale) for a batch of C-length contexts, obtained by replaying the exact
    preprocessing ``_forecast_single_step`` performs: pad to train_ctx_len, then
    ``PatchedTokenizer.input_transform``. The FUTURE IS NOT AN ARGUMENT -- it cannot leak."""
    adj, pad_len = model._adjust_context_length(ctx, geom.train_ctx_len, geom.train_ctx_len)
    if pad_len != geom.pad_len:
        raise RuntimeError(f"pad_len {pad_len} != geometry {geom.pad_len}: context length is not "
                           f"{geom.C}")
    _, state = model.tokenizer.input_transform(adj)
    return state


@torch.inference_mode()
def native_forward(model, ctx: torch.Tensor, geom: TiRexGeometry, *,
                   rollout_steps: int | None = None, capture: bool = True):
    """ONE native accelerated-rollout pass. Returns (official_quantiles, reps, state).

        official_quantiles : (B, H, Q)  -- exactly what ``model.forecast`` would return
        reps               : {point name -> (B, n_tokens, d)} float32, or {} when capture=False
        state              : the StandardScalerState used by the pass

    ``rollout_steps`` defaults to ``geom.rollout_steps`` (= K), the ONE-PASS mode. Passing 1
    reproduces the package default (two passes) and is used only by ``rollout_mode_gap``.
    """
    steps = geom.rollout_steps if rollout_steps is None else int(rollout_steps)
    store: dict = {}
    handles = register_rep_hooks(model, store) if capture else []
    try:
        q, _ = model._forecast_quantiles(ctx, prediction_length=geom.H,
                                         max_accelerated_rollout_steps=steps)
    finally:
        for h in handles:
            h.remove()
    if capture:
        missing = [n for n in REP_NAMES if n not in store]
        if missing:
            raise RuntimeError(f"hooks did not fire for {missing}; the module layout changed")
        if steps == geom.rollout_steps:
            for n, v in store.items():
                if v.shape[1] != geom.n_tokens or v.shape[2] != MODEL_DIMS:
                    raise RuntimeError(
                        f"representation {n} has shape {tuple(v.shape)}, expected "
                        f"(B, {geom.n_tokens}, {MODEL_DIMS})")
    return q, store, scaler_state(model, ctx, geom)


def rollout_contexts(ctx: torch.Tensor, geom: TiRexGeometry) -> list[torch.Tensor]:
    """The context handed to each of the K native passes, built WITHOUT running the model.

    Pass k gets ``x[T-C:T] ++ (k*P) NaN``. This is exact, not an approximation, because
    ``_forecast_tensor`` appends ``full_like(prediction[:, 0, :], fill_value=nan)`` -- a shape
    template filled with NaN -- so what is appended never depends on what the model predicted.
    ``assert_rollout_appends_missing`` proves that against the running model.
    """
    ctx = torch.as_tensor(ctx).to(dtype=torch.float32)
    out, cur = [], ctx
    for k in range(geom.n_forecast_patches):
        out.append(cur)
        cur = torch.cat([cur, torch.full((ctx.shape[0], geom.output_patch), float("nan"),
                                         dtype=cur.dtype, device=cur.device)], dim=-1)
    return out


def scaler_states_for_mode(model, ctx: torch.Tensor, geom: TiRexGeometry, mode: str):
    """The (loc, scale) each output patch is de-normalized with, as (n, K) arrays.

    single_pass: ONE pass, so every patch shares one (loc, scale) -- broadcast to K columns.
    two_pass   : pass k's OWN statistics, from pass k's OWN context construction.

    Uniform (n, K) output means every downstream consumer (targets, denormalize, the native
    reference) is written once and is correct in both modes.
    """
    check_rollout_mode(mode)
    K = geom.n_forecast_patches
    if mode == ROLLOUT_SINGLE:
        st = scaler_state(model, torch.as_tensor(ctx), geom)
        lo = st.loc.flatten().cpu().numpy().astype(np.float32)
        sc = st.scale.flatten().cpu().numpy().astype(np.float32)
        return np.repeat(lo[:, None], K, axis=1), np.repeat(sc[:, None], K, axis=1)
    los, scs = [], []
    with torch.inference_mode():
        for c in rollout_contexts(ctx, geom):
            adj, _ = model._adjust_context_length(c, geom.train_ctx_len, geom.train_ctx_len)
            _, st = model.tokenizer.input_transform(adj)
            los.append(st.loc.flatten().cpu().numpy().astype(np.float32))
            scs.append(st.scale.flatten().cpu().numpy().astype(np.float32))
    return np.stack(los, axis=1), np.stack(scs, axis=1)


@torch.inference_mode()
def assert_rollout_appends_missing(model, ctx: torch.Tensor, geom: TiRexGeometry) -> dict:
    """Prove, against the RUNNING model, what the second native pass actually consumes.

    Re-runs ``_forecast_tensor``'s own loop body and inspects the context it builds for pass 1:
      * every appended slot is NaN and none is finite            -> MISSING values
      * the appended block does NOT equal any quantile row of the previous forecast
                                                                 -> not predicted values
      * the real prefix is untouched                             -> not a reconstructed context
    Returns the measurements; raises if the appended block is anything other than all-missing.
    """
    c = torch.as_tensor(ctx).to(dtype=torch.float32)
    pred = model._forecast_single_step(c, 1)                   # (n, Q, P), raw units
    nxt = torch.cat([c, torch.full_like(pred[:, 0, :], fill_value=torch.nan)], dim=-1)
    app = nxt[:, c.shape[-1]:]
    eq_any_q = [bool(torch.equal(torch.nan_to_num(app, nan=0.0), pred[:, q, :]))
                for q in range(geom.num_quantiles)]
    rec = {"appended_len": int(app.shape[-1]),
           "all_nan": bool(torch.isnan(app).all()),
           "any_finite": bool(torch.isfinite(app).any()),
           "equals_some_forecast_quantile_row": any(eq_any_q),
           "real_prefix_unchanged": bool(torch.equal(nxt[:, :c.shape[-1]], c)),
           "previous_forecast_median_abs_mean": float(pred[:, geom.median_index, :].abs().mean()),
           "verdict": "missing_values",
           "source": "_forecast_tensor: torch.cat([context, torch.full_like(prediction[:, 0, :], "
                     "fill_value=torch.nan)], dim=-1) -- the forecast is a SHAPE TEMPLATE only"}
    if not (rec["all_nan"] and not rec["any_finite"] and rec["real_prefix_unchanged"]):
        raise RuntimeError(f"the rollout does NOT append pure missing values: {rec}. The two_pass "
                           "target construction and its normalization assume it does.")
    return rec


def readouts_from_reps(reps, geom: TiRexGeometry, mode: str) -> dict:
    """{point -> (n, K, d)} forecast-producing states, for either mode.

    single_pass: ``reps`` is one {point: (n, n_tokens, d)} dict; take the K readout tokens.
    two_pass   : ``reps`` is a LIST of K such dicts, one per pass; take each pass's own readout.
    """
    if check_rollout_mode(mode) == ROLLOUT_SINGLE:
        return {n: select_readout(v, geom) for n, v in reps.items()}
    idx = geom.two_pass_readout_index
    names = list(reps[0])
    out = {}
    for n in names:
        cols = []
        for k, per_pass in enumerate(reps):
            v = per_pass[n]
            if v.shape[1] != geom.two_pass_n_tokens:
                raise RuntimeError(f"pass {k} point {n}: {v.shape[1]} tokens, expected "
                                   f"{geom.two_pass_n_tokens}")
            cols.append(v[:, idx, :])
        out[n] = torch.stack(cols, dim=1)
    return out


@torch.inference_mode()
def native_forward_two_pass(model, ctx: torch.Tensor, geom: TiRexGeometry, *, capture: bool = True):
    """The PACKAGE-DEFAULT rollout, replicated line-for-line so each pass can be hooked.

    Mirrors ``_forecast_tensor(prediction_length=H, new_patch_count=1)`` composed with
    ``_forecast_single_step`` and ``_forecast_quantiles``'s final ``swapaxes``. Every pass keeps
    its OWN context construction and its OWN tokenizer state, which is the whole point: the state
    responsible for output patch k is captured from the pass that actually produced patch k.

    ``assert_two_pass_matches_package`` proves this replication equals the package's own output.

    Returns (official (n, H, Q), reps: list of K {point: (n, n_tokens, d)}, states: list of K
    scaler states, layouts: list of K dicts).
    """
    context = torch.as_tensor(ctx).to(dtype=torch.float32)
    preds, states, per_pass, layouts = [], [], [], []
    for k in range(geom.n_forecast_patches):
        store: dict = {}
        handles = register_rep_hooks(model, store) if capture else []
        try:
            adj, pad_len = model._adjust_context_length(context, geom.train_ctx_len,
                                                        geom.train_ctx_len)
            input_token, st = model.tokenizer.input_transform(adj)
            prediction = model._forward_model_tokenized(input_token=input_token, new_patch_count=1)
            predicted_token = prediction[:, :, -1:, :].to(input_token)
            out = model.tokenizer.output_transform(predicted_token, st)
        finally:
            for h in handles:
                h.remove()
        exp = geom.two_pass_layout(k)
        if int(pad_len) != exp["pad_len"] or int(input_token.shape[1]) != exp["n_tokens"]:
            raise RuntimeError(f"two_pass pass {k}: pad_len {int(pad_len)} / tokens "
                               f"{int(input_token.shape[1])} disagree with the derived layout "
                               f"{exp}")
        if capture:
            missing = [n for n in REP_NAMES if n not in store]
            if missing:
                raise RuntimeError(f"hooks did not fire for {missing} in pass {k}")
            per_pass.append({n: store[n] for n in REP_NAMES})
        preds.append(out)
        states.append(st)
        layouts.append(exp)
        context = torch.cat([context, torch.full_like(out[:, 0, :], fill_value=torch.nan)], dim=-1)
    official = torch.cat(preds, dim=-1)[..., :geom.H].to(dtype=torch.float32).swapaxes(1, 2)
    return official, per_pass, states, layouts


@torch.inference_mode()
def assert_two_pass_matches_package(model, ctx: torch.Tensor, geom: TiRexGeometry) -> dict:
    """Our replicated two-pass loop must equal the package's own default-path output exactly."""
    ours, _, _, _ = native_forward_two_pass(model, ctx, geom, capture=False)
    theirs, _ = model._forecast_quantiles(ctx, prediction_length=geom.H,
                                          max_accelerated_rollout_steps=1)
    d = (ours - theirs).abs()
    rec = {"max_abs_error": float(d.max()), "bitwise_identical": bool(torch.equal(ours, theirs)),
           "n_windows": int(torch.as_tensor(ctx).shape[0])}
    if not rec["bitwise_identical"]:
        raise RuntimeError(f"the replicated two_pass loop does NOT match the package default "
                           f"path: {rec}. Fix the replication before probing it.")
    return rec


def verify_native_head_two_pass(model, per_pass, official, states, geom: TiRexGeometry, *,
                                atol: float = 1e-5, rtol: float = 2e-6) -> dict:
    """The decisive invariant for two_pass: each output patch reconstructed from the state of the
    pass that produced it, with that pass's OWN (loc, scale).

    Unlike single_pass this is not expected to be bit-exact: the native path applies
    ``output_patch_embedding`` to the FULL (n, 64, 512) sequence and slices afterwards, while we
    apply it to the pre-sliced (n, 1, 512) readout. That is a different matmul shape and hence a
    different accumulation order. A ``slice_order_control`` measures exactly that effect, so the
    residual is attributed rather than excused; the elementwise gate is unchanged.
    """
    K, P, Q = geom.n_forecast_patches, geom.output_patch, geom.num_quantiles
    final = f"L{NUM_BLOCKS}+RMS"
    with torch.inference_mode():
        chunks, ctrl = [], []
        for k in range(K):
            h = per_pass[k][final][:, geom.two_pass_readout_index:geom.two_pass_readout_index + 1, :]
            raw = model.output_patch_embedding(h)
            qp = torch.transpose(torch.unflatten(raw, -1, (Q, P)), 1, 2)
            chunks.append(model.tokenizer.output_transform(qp, states[k]))
            # slice-order control: head on the FULL sequence, sliced AFTER (the native order)
            raw_full = model.output_patch_embedding(per_pass[k][final])
            ctrl.append(raw_full[:, geom.two_pass_readout_index:geom.two_pass_readout_index + 1, :])
        recon = torch.cat(chunks, dim=-1)[..., :geom.H].swapaxes(1, 2)
        slice_gap = max(float((model.output_patch_embedding(
            per_pass[k][final][:, geom.two_pass_readout_index:geom.two_pass_readout_index + 1, :])
            - ctrl[k]).abs().max()) for k in range(K))
    d = (recon - official).abs()
    scaled = d / (atol + rtol * official.abs())
    flat = int(torch.argmax(d))
    idx = np.unravel_index(flat, tuple(d.shape))

    controls = {}
    # wrong PASS for a chunk: use pass 0's state (and stats) for every chunk
    with torch.inference_mode():
        h0 = per_pass[0][final][:, geom.two_pass_readout_index:geom.two_pass_readout_index + 1, :]
        raw = model.output_patch_embedding(h0)
        qp = torch.transpose(torch.unflatten(raw, -1, (Q, P)), 1, 2)
        one = model.tokenizer.output_transform(qp, states[0])
        alt = torch.cat([one] * K, dim=-1)[..., :geom.H].swapaxes(1, 2)
    controls["all_from_pass_0"] = float((alt - official).abs().max())
    if K > 1:
        with torch.inference_mode():
            sw = []
            for k in range(K):
                j = K - 1 - k
                h = per_pass[j][final][:, geom.two_pass_readout_index:geom.two_pass_readout_index + 1, :]
                raw = model.output_patch_embedding(h)
                qp = torch.transpose(torch.unflatten(raw, -1, (Q, P)), 1, 2)
                sw.append(model.tokenizer.output_transform(qp, states[j]))
            swapped = torch.cat(sw, dim=-1)[..., :geom.H].swapaxes(1, 2)
        controls["passes_swapped"] = float((swapped - official).abs().max())

    rec = {"mode": ROLLOUT_TWO, "readout_index_per_pass": list(geom.readout_token_indices(ROLLOUT_TWO)),
           "max_abs_error": float(d.max()), "max_scaled_error": float(scaled.max()),
           "exact_bitwise_match": bool(torch.equal(recon, official)),
           "mean_abs_official": float(official.abs().mean()),
           "worst_element": {"index": [int(i) for i in idx], "official": float(official[idx]),
                             "recon": float(recon[idx])},
           "slice_order_control_max_abs": slice_gap,
           "atol": atol, "rtol": rtol, "n_windows": int(official.shape[0]),
           "wrong_index_controls": controls,
           "controls_discriminate": bool(controls)
                                    and min(controls.values()) > 100 * max(float(d.max()), atol),
           "passed": bool(scaled.max() <= 1.0)}
    if not rec["passed"]:
        raise RuntimeError(
            f"TWO-PASS NATIVE-HEAD RECONSTRUCTION FAILED: the per-pass states do NOT reproduce "
            f"TiRex's own default-path H={geom.H} forecast (max scaled error "
            f"{rec['max_scaled_error']:.3f} > 1, max|d| {rec['max_abs_error']:.3e}). Fix the "
            "indexing or the per-pass normalization; do NOT loosen the tolerance.")
    if controls and not rec["controls_discriminate"]:
        raise RuntimeError(f"TWO-PASS CONTROL FAILED: wrong per-pass states reproduce the forecast "
                           f"almost as well ({controls}); the identity has no discriminating power.")
    return rec


@torch.inference_mode()
def apply_native_head(model, states: torch.Tensor, state, geom: TiRexGeometry) -> torch.Tensor:
    """TiRex's FROZEN native output pathway applied to chosen post-RMSNorm token states.

        (B, k, d) -> output_patch_embedding -> (B, k, Q*P)
                  -> unflatten(-1, (Q, P))  -> (B, k, Q, P)      [QUANTILE-MAJOR: idx = q*P + t]
                  -> transpose(1, 2)        -> (B, Q, k, P)
                  -> tokenizer.output_transform(state)           -> (B, Q, k*P)   [raw units]
                  -> swapaxes(1, 2)                              -> (B, k*P, Q)

    Verbatim the composition of ``_forward_model_tokenized`` (unflatten + transpose),
    ``_forecast_single_step`` (output_transform) and ``_forecast_quantiles`` (swapaxes). Note the
    layout is QUANTILE-major, the OPPOSITE of TimesFM-3's horizon-major head -- proven, not
    assumed, by ``verify_native_head``.
    """
    if states.ndim != 3 or states.shape[-1] != MODEL_DIMS:
        raise ValueError(f"expected (B, k, {MODEL_DIMS}) states, got {tuple(states.shape)}")
    raw = model.output_patch_embedding(states)
    qp = torch.unflatten(raw, -1, (geom.num_quantiles, geom.output_patch))
    qp = torch.transpose(qp, 1, 2)
    out = model.tokenizer.output_transform(qp, state)
    return out.swapaxes(1, 2)


def select_readout(reps: torch.Tensor, geom: TiRexGeometry) -> torch.Tensor:
    """(B, n_tokens, d) -> (B, K, d): only the native forecast-producing token states."""
    if reps.shape[1] != geom.n_tokens:
        raise ValueError(f"expected {geom.n_tokens} tokens, got {reps.shape[1]}")
    return reps[:, list(geom.readout_indices), :]


def verify_native_head(model, reps: dict, official: torch.Tensor, state,
                       geom: TiRexGeometry, *, atol: float = 1e-5, rtol: float = 2e-6) -> dict:
    """THE decisive invariant (spec section C).

    Take the final-RMSNorm states at the K native readout tokens, push them through the frozen
    native output pathway and the native de-normalization, and require the result to equal
    ``model``'s OWN H-step forecast from the same pass.

    Gated ELEMENTWISE: |recon - official| <= atol + rtol*|official|, failing only if the worst
    SCALED error exceeds 1. Three WRONG-INDEX CONTROLS are measured too -- a test that passes for
    the right indices but also for the wrong ones proves nothing.
    """
    final = reps[f"L{NUM_BLOCKS}+RMS"]
    recon = apply_native_head(model, select_readout(final, geom), state, geom)
    if recon.shape != official.shape:
        raise RuntimeError(f"reconstruction shape {tuple(recon.shape)} != official "
                           f"{tuple(official.shape)}")
    d = (recon - official).abs()
    scaled = d / (atol + rtol * official.abs())
    flat = int(torch.argmax(d))
    idx = np.unravel_index(flat, tuple(d.shape))
    K = geom.n_forecast_patches

    controls = {}
    n_tok = final.shape[1]
    cand = {"one_token_too_early": [i - 1 for i in geom.readout_indices],
            "all_first_readout": [geom.first_readout] * K,
            "reversed_order": list(geom.readout_indices)[::-1]}
    for lbl, sel in cand.items():
        if min(sel) < 0 or max(sel) >= n_tok or list(sel) == list(geom.readout_indices):
            continue
        alt = apply_native_head(model, final[:, list(sel), :], state, geom)
        controls[lbl] = float((alt - official).abs().max())

    rec = {
        "readout_indices": list(geom.readout_indices),
        "max_abs_error": float(d.max()), "max_scaled_error": float(scaled.max()),
        "exact_bitwise_match": bool(torch.equal(recon, official)),
        "mean_abs_official": float(official.abs().mean()),
        "worst_element": {"index": [int(i) for i in idx],
                          "official": float(official[idx]), "recon": float(recon[idx])},
        "atol": atol, "rtol": rtol, "n_windows": int(official.shape[0]),
        "wrong_index_controls": controls,
        "controls_discriminate": bool(controls) and min(controls.values()) > 100 * max(float(d.max()), atol),
        "passed": bool(scaled.max() <= 1.0),
    }
    if not rec["passed"]:
        raise RuntimeError(
            "NATIVE-HEAD RECONSTRUCTION FAILED: the states at tokens "
            f"{list(geom.readout_indices)} do NOT reproduce TiRex's own H={geom.H} forecast "
            f"(max scaled error {rec['max_scaled_error']:.3f} > 1, max|d| "
            f"{rec['max_abs_error']:.3e}). The readout indexing or the head pathway is wrong -- "
            "fix it; do NOT loosen the tolerance.")
    if controls and not rec["controls_discriminate"]:
        raise RuntimeError(
            f"NATIVE-HEAD CONTROL FAILED: wrong readout indices reproduce the forecast almost as "
            f"well ({controls}); the identity test has no discriminating power.")
    return rec


def compare_backends(ctx, geom: TiRexGeometry, *, checkpoint: str = DEFAULT_CHECKPOINT,
                     device: str = "cuda", points=("Emb", f"L{NUM_BLOCKS}", f"L{NUM_BLOCKS}+RMS"),
                     atol: float = 1e-5, rtol: float = 2e-6) -> dict:
    """Quantify the ``torch`` vs ``cuda`` sLSTM backends on the SAME windows and device.

    They are two different implementations of the same recurrence -- ``sLSTMCellTorch`` stores its
    state in bfloat16 and promotes each pointwise step to float32, while the ``cuda`` path is
    xLSTM's compiled kernel -- so they are NOT expected to agree bit-for-bit. This measures HOW
    FAR apart they are, on the official forecast and on each probed representation, instead of
    assuming either answer.

    Run it ONCE on the GPU before committing to a backend, record the numbers, and keep the same
    backend for the whole paper (the feature cache refuses a cross-backend reuse regardless).
    Returns the measurements; it does not raise on disagreement, because disagreement here is a
    fact to report, not a broken invariant -- the invariant that must hold on EACH backend is
    ``verify_native_head``, which this function also re-runs per backend.
    """
    ctx_t = torch.as_tensor(np.asarray(ctx, dtype=np.float32))
    out = {"device": device, "checkpoint": checkpoint, "atol": atol, "rtol": rtol,
           "n_windows": int(ctx_t.shape[0]), "points": list(points), "per_backend": {},
           "available": {}}
    runs = {}
    for backend in ("torch", "cuda"):
        try:
            model = get_model(checkpoint, device=device, backend=backend)
            q, reps, st = native_forward(model, ctx_t.to(device), geom)
            runs[backend] = (q.float().cpu(),
                             {p: select_readout(reps[p], geom).float().cpu() for p in points})
            out["per_backend"][backend] = verify_native_head(model, reps, q, st, geom,
                                                             atol=atol, rtol=rtol)
            out["available"][backend] = True
            del model
        except Exception as exc:                     # a missing xlstm / no GPU is a REPORT, not a crash
            out["available"][backend] = False
            out["per_backend"][backend] = {"error": f"{type(exc).__name__}: {exc}"}
    if len(runs) == 2:
        qa, qb = runs["torch"][0], runs["cuda"][0]
        d = (qa - qb).abs()
        out["forecast_agreement"] = {
            "max_abs_diff": float(d.max()),
            "relative_to_mean_abs": float(d.max() / qa.abs().mean().clamp_min(1e-12)),
            "max_scaled_error": float((d / (atol + rtol * qb.abs())).max()),
            "bitwise_identical": bool(torch.equal(qa, qb))}
        out["representation_agreement"] = {}
        for p in points:
            ra, rb = runs["torch"][1][p], runs["cuda"][1][p]
            out["representation_agreement"][p] = {
                "max_abs_diff": float((ra - rb).abs().max()),
                "relative_to_layer_std": float((ra - rb).abs().max()
                                               / (ra.std() + 1e-12))}
    return out


@torch.inference_mode()
def rollout_mode_gap(model, ctx: torch.Tensor, geom: TiRexGeometry) -> dict:
    """Measure the documented deviation: ONE-pass accelerated rollout (what we probe) vs the
    package DEFAULT ``max_accelerated_rollout_steps=1`` (two passes). Diagnostic only -- it never
    gates a run, but it is recorded per dataset so the choice of path is auditable."""
    q_fast, _, _ = native_forward(model, ctx, geom, capture=False)
    q_def, _, _ = native_forward(model, ctx, geom, rollout_steps=1, capture=False)
    d = (q_fast - q_def).abs()
    P = geom.output_patch
    return {"mode_used": f"max_accelerated_rollout_steps={geom.rollout_steps} (single pass)",
            "mode_compared": "max_accelerated_rollout_steps=1 (package default, two passes)",
            "max_abs_diff": float(d.max()),
            "relative_to_mean_abs": float(d.max() / q_def.abs().mean().clamp_min(1e-12)),
            "first_patch_identical": bool(torch.allclose(q_fast[:, :P], q_def[:, :P], atol=1e-6)),
            "later_patches_identical": bool(torch.allclose(q_fast[:, P:], q_def[:, P:], atol=1e-6)),
            "n_windows": int(ctx.shape[0])}


# --------------------------------------------------------------------------- #
# targets in TiRex's own normalized space
# --------------------------------------------------------------------------- #
def build_targets(ctx, future, model, geom: TiRexGeometry, *, mode: str = ROLLOUT_SINGLE):
    """Probe targets in the SAME normalized space the representations live in.

        loc, scale  <- TiRex's tokenizer, from the CONTEXT ONLY, PER OUTPUT PATCH   (n, K)
        target[:, patch k] = (future[:, patch k] - loc[:, k]) / scale[:, k]         (n, H)

    Per-patch statistics matter in ``two_pass``: each native pass derives its own (loc, scale)
    from its own context construction, and patch k is de-normalized with pass k's. In
    ``single_pass`` there is one pass, so the K columns are identical by construction -- the
    uniform (n, K) shape keeps ONE code path correct in both modes.

    ``ctx`` and ``future`` are separate arrays and ``future`` reaches neither the model nor the
    scaler -- the no-leakage property is structural here, not a convention. No forward pass is
    needed even in two_pass mode, because what the rollout appends is always NaN (see
    ``rollout_contexts``).
    """
    check_rollout_mode(mode)
    ctx_t = torch.as_tensor(np.asarray(ctx, dtype=np.float32))
    fut = np.asarray(future, dtype=np.float32)
    if ctx_t.ndim != 2 or ctx_t.shape[1] != geom.C:
        raise ValueError(f"context must be (n, {geom.C}), got {tuple(ctx_t.shape)}")
    if fut.ndim != 2 or fut.shape[1] != geom.H or fut.shape[0] != ctx_t.shape[0]:
        raise ValueError(f"future must be ({ctx_t.shape[0]}, {geom.H}), got {fut.shape}")
    loc, scale = scaler_states_for_mode(model, ctx_t, geom, mode)
    if not np.all(np.isfinite(loc)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("TiRex produced a non-finite or non-positive scale; refusing to build "
                           "targets (this would make the probe objective meaningless)")
    tgt = np.empty_like(fut)
    for k in range(geom.n_forecast_patches):
        sl = geom.target_slice(k)
        tgt[:, sl] = (fut[:, sl] - loc[:, k, None]) / scale[:, k, None]
    return tgt.astype(np.float32), loc, scale


def denormalize(pred_norm, loc, scale, geom: TiRexGeometry) -> np.ndarray:
    """Normalized forecast (n, H) -> RAW units, TiRex's own inverse (``StandardScaler.re_scale``),
    applied PER OUTPUT PATCH with that patch's own (loc, scale)."""
    p = np.asarray(pred_norm, dtype=np.float64)
    lo = np.asarray(loc, dtype=np.float64)
    sc = np.asarray(scale, dtype=np.float64)
    if lo.ndim != 2 or lo.shape[1] != geom.n_forecast_patches or lo.shape != sc.shape:
        raise ValueError(f"loc/scale must be (n, {geom.n_forecast_patches}), got {lo.shape} / "
                         f"{sc.shape} -- pass the per-patch statistics from build_targets")
    if p.shape[-1] != geom.H:
        raise ValueError(f"expected a horizon of {geom.H}, got {p.shape[-1]}")
    out = np.empty_like(p)
    for k in range(geom.n_forecast_patches):
        sl = geom.target_slice(k)
        out[..., sl] = p[..., sl] * sc[:, k, None] + lo[:, k, None]
    return out


def normalize_raw(raw, loc, scale, geom: TiRexGeometry) -> np.ndarray:
    """RAW (n, H) -> the per-patch normalized space. Exact inverse of ``denormalize``; used to put
    TiRex's own native forecast on the probe's scale without ever crossing two spaces."""
    r = np.asarray(raw, dtype=np.float64)
    lo = np.asarray(loc, dtype=np.float64)
    sc = np.asarray(scale, dtype=np.float64)
    out = np.empty_like(r)
    for k in range(geom.n_forecast_patches):
        sl = geom.target_slice(k)
        out[..., sl] = (r[..., sl] - lo[:, k, None]) / sc[:, k, None]
    return out


def assert_target_roundtrip(future, targets, loc, scale, geom: TiRexGeometry,
                            tol: float = 1e-4) -> dict:
    """denormalize(normalize(y)) == y. float32 storage of large raw values is the limiting
    precision, so the gate is RELATIVE to the series scale, not absolute."""
    y = np.asarray(future, dtype=np.float64)
    back = denormalize(targets, loc, scale, geom)
    err = np.abs(back - y)
    denom = np.maximum(np.abs(y).mean(), 1e-12)
    rel = float(err.max() / denom)
    if not np.isfinite(rel) or rel > tol:
        raise RuntimeError(f"target round-trip failed: max|d| {err.max():.3e}, relative {rel:.3e} "
                           f"> {tol}. The normalization convention is wrong.")
    return {"max_abs_error": float(err.max()), "relative_error": rel, "tol": tol}


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def extract_window_features(X, model, geom: TiRexGeometry, *, batch_size: int = 64,
                            points=None, device: str | None = None, verbose: bool = False,
                            mode: str = ROLLOUT_SINGLE):
    """Per-point readout states for a window matrix X (n, C), in either rollout mode.

    Returns ({point name -> (n, K, d) float32}, native (n, H, Q)). ONLY the K native
    forecast-producing token states are kept -- in ``two_pass`` each one comes from the pass that
    actually produced its output patch. The pad/context tokens are never stored (they are not what
    the head reads, and keeping them would multiply the cache by 32/64).
    """
    check_rollout_mode(mode)
    Xa = np.asarray(X, dtype=np.float32)
    if Xa.ndim != 2 or Xa.shape[1] != geom.C:
        raise ValueError(f"windows must be (n, {geom.C}), got {Xa.shape}")
    names = list(points) if points is not None else list(REP_NAMES)
    unknown = [n for n in names if n not in REP_NAMES]
    if unknown:
        raise ValueError(f"unknown representation points {unknown}; known: {list(REP_NAMES)}")
    dev = resolve_device(device)
    out = {n: np.empty((Xa.shape[0], geom.n_forecast_patches, MODEL_DIMS), dtype=np.float32)
           for n in names}
    native = np.empty((Xa.shape[0], geom.H, geom.num_quantiles), dtype=np.float32)
    for s in range(0, Xa.shape[0], batch_size):
        e = min(s + batch_size, Xa.shape[0])
        ctx = torch.as_tensor(Xa[s:e]).to(dev)
        if mode == ROLLOUT_SINGLE:
            q, reps, _ = native_forward(model, ctx, geom)
        else:
            q, reps, _, _ = native_forward_two_pass(model, ctx, geom)
        native[s:e] = q.float().cpu().numpy()
        ro = readouts_from_reps(reps, geom, mode)
        for n in names:
            out[n][s:e] = ro[n].float().cpu().numpy()
        if verbose and (s // batch_size) % 10 == 0:
            print(f"      extracted {e}/{Xa.shape[0]} windows", flush=True)
    return out, native


# --------------------------------------------------------------------------- #
# cache -- metadata-gated, fail-loud, never silently reused
# --------------------------------------------------------------------------- #
def window_identity_hash(X) -> str:
    """Hash of the window matrix itself. A cache whose windows differ is REJECTED, so a changed
    split / suite / seed can never be silently reused."""
    a = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def cache_metadata(tag, split, geom: TiRexGeometry, *, checkpoint, backend, points, seed, X,
                   mode: str = ROLLOUT_SINGLE, feature_dtype="float32") -> dict:
    return {"cache_version": CACHE_VERSION, "dataset": tag, "split": split,
            "checkpoint": checkpoint, "backend": backend,
            "rollout_mode": check_rollout_mode(mode),
            "tirex_version": tirex_version(), "torch_version": torch.__version__,
            "points": list(points), "seed": int(seed), "feature_dtype": str(feature_dtype),
            "n_windows": int(np.asarray(X).shape[0]),
            "window_hash": window_identity_hash(X), "geometry": geom.as_dict()}


def cache_root(cache_dir, tag, split, mode: str = ROLLOUT_SINGLE) -> Path:
    """The rollout mode is part of the PATH, not just the metadata, so the two modes' caches
    coexist and a mode switch can never overwrite or silently reuse the other's features."""
    return Path(cache_dir) / f"{CACHE_VERSION}__{check_rollout_mode(mode)}__{tag}__{split}"


def read_cache(root: Path, meta_expected: dict, points):
    """Load a cache ONLY if every metadata field agrees. Returns None when absent; RAISES with
    the offending fields when present but incompatible -- never repairs, never partially reuses."""
    root = Path(root)
    mpath, fpath = root.with_suffix(".json"), root.with_suffix(".npz")
    if not (mpath.exists() and fpath.exists()):
        return None
    meta = json.loads(mpath.read_text())
    keys = ("cache_version", "dataset", "split", "checkpoint", "backend", "tirex_version",
            "rollout_mode", "seed", "n_windows", "window_hash", "feature_dtype")
    diff = {k: (meta.get(k), meta_expected.get(k)) for k in keys
            if meta.get(k) != meta_expected.get(k)}
    if meta.get("geometry") != meta_expected.get("geometry"):
        diff["geometry"] = ("<differs>", "<differs>")
    if diff:
        raise RuntimeError(
            f"REFUSING the feature cache {fpath.name}: it was written under a different "
            f"configuration {({k: v[0] for k, v in diff.items()})} than this run "
            f"{({k: v[1] for k, v in diff.items()})}. Delete it and re-extract.")
    missing = [p for p in points if p not in meta["points"]]
    if missing:
        return None                                   # partial cache: re-extract, do not repair
    with np.load(fpath, allow_pickle=False) as z:
        slug = {p.name: p.slug for p in REP_POINTS}
        feats = {p: np.asarray(z[f"rep__{slug[p]}"], dtype=np.float32) for p in points}
        native = np.asarray(z["native"], dtype=np.float32)
    return feats, native, meta


def cached_features(tag, split, X, model, geom: TiRexGeometry, *, cache_dir, checkpoint,
                    backend, points=None, seed=0, batch_size=64, verbose=False,
                    mode: str = ROLLOUT_SINGLE):
    """Extract-or-load. Returns (feats, native, meta, hit)."""
    points = list(points) if points is not None else list(REP_NAMES)
    meta = cache_metadata(tag, split, geom, checkpoint=checkpoint, backend=backend,
                          points=points, seed=seed, X=X, mode=mode)
    root = cache_root(cache_dir, tag, split, mode)
    got = read_cache(root, meta, points)
    if got is not None:
        return got[0], got[1], got[2], True
    if model is None:
        raise RuntimeError(f"no usable cache at {root}.npz and no model was loaded to build one")
    feats, native = extract_window_features(X, model, geom, batch_size=batch_size,
                                            points=points, device=None, verbose=verbose,
                                            mode=mode)
    root.parent.mkdir(parents=True, exist_ok=True)
    slug = {p.name: p.slug for p in REP_POINTS}
    np.savez(root.with_suffix(".npz"), native=native,
             **{f"rep__{slug[p]}": feats[p] for p in points})
    root.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return feats, native, meta, False
