"""Parameter and FLOP accounting for a depth-truncated Chronos-2.

Every constant here is read off the model definition in ``chronos.chronos2`` (v2.3.1), NOT guessed:

    encoder block   Chronos2EncoderBlock = layer[0] TimeSelfAttention + layer[1] GroupSelfAttention
                    + layer[2] FeedForward (model.py). Each attention is an MHA with q,k,v,o
                    ``nn.Linear(d_model, inner_dim, bias=False)`` (inner_dim = num_heads * d_kv =
                    12 * 64 = 768) plus one Chronos2LayerNorm; the FeedForward is MLP(wi, wo,
                    bias=False) plus one Chronos2LayerNorm. The MLP is NON-gated -- layers.py
                    asserts ``not config.is_gated_act`` -- so it is wi + wo, not wi_0 + wi_1 + wo.
                    Position information is RoPE (Chronos2RotaryEmbedding), which is a
                    ``register_buffer`` and therefore contributes NO parameters.
    Chronos2LayerNorm  T5-style RMSNorm: a single weight vector, no bias (layers.py:129).
    input embedding ResidualBlock(in_dim=input_patch_size * 3, h_dim=d_ff, out_dim=d_model)
                    -- the x3 is [time_embedding, patch, patch_mask] (model.py:239).
    native head     ResidualBlock(d_model, d_ff, num_quantiles * output_patch_size) (model.py:265).
    ResidualBlock   hidden_layer + output_layer + residual_layer, three nn.Linear WITH bias
                    (layers.py:414).
    REG embedding   nn.Embedding(vocab_size=2, d_model) when use_reg_token (model.py:236).

The sum of these families equals ``RECORDED_TOTAL_PARAMS`` = 119,477,664, which is not a derived
number: it is the ``trainable_params`` recorded by a real full fine-tuning run of amazon/chronos-2
(results/ft_specialization/*/manifest.json). That run's ``param_drift`` diagnostic also buckets
``model.named_parameters()`` into exactly these families with NO leftover "other" group (see
probing/finetune.py::_param_group), so the families provably partition the whole model. The module
asserts the identity at import time, so a wrong constant cannot pass silently.

``verify_against_model`` re-derives every count from a loaded model's real weight tensors; it needs
a GPU/compute node (loading Chronos-2 is not login-node work). Everything else here is pure
arithmetic and runs anywhere.
"""

from __future__ import annotations

import math

# ---- architecture (amazon/chronos-2; recorded in results/ft_specialization/*/manifest.json) ----
D_MODEL = 768
D_FF = 3072
NUM_BLOCKS = 12
NUM_HEADS = 12
D_KV = 64
NUM_NATIVE_QUANTILES = 21
OUTPUT_PATCH_SIZE = 16
INPUT_PATCH_SIZE = 16
INPUT_PATCH_CHANNELS = 3          # [time_embedding, patch, patch_mask] -> ResidualBlock in_dim
USE_REG_TOKEN = True

# Empirically recorded, NOT derived: sum(p.numel() for p in model.parameters() if p.requires_grad)
# from a real full-FT run of amazon/chronos-2.
RECORDED_TOTAL_PARAMS = 119_477_664


def _residual_block_params(in_dim: int, h_dim: int, out_dim: int) -> int:
    """ResidualBlock = hidden_layer + output_layer + residual_layer, three nn.Linear WITH bias."""
    return (in_dim * h_dim + h_dim) + (h_dim * out_dim + out_dim) + (in_dim * out_dim + out_dim)


INNER_DIM = NUM_HEADS * D_KV                                   # == D_MODEL for chronos-2
MHA_PARAMS = 4 * D_MODEL * INNER_DIM                           # q, k, v, o -- all bias=False
MLP_PARAMS = D_MODEL * D_FF + D_FF * D_MODEL                   # wi, wo -- bias=False, NON-gated
RMSNORM_PARAMS = D_MODEL                                       # weight only, no bias
BLOCK_PARAMS = 2 * (MHA_PARAMS + RMSNORM_PARAMS) + (MLP_PARAMS + RMSNORM_PARAMS)

INPUT_EMBEDDING_PARAMS = _residual_block_params(
    INPUT_PATCH_SIZE * INPUT_PATCH_CHANNELS, D_FF, D_MODEL)
NATIVE_HEAD_PARAMS = _residual_block_params(
    D_MODEL, D_FF, NUM_NATIVE_QUANTILES * OUTPUT_PATCH_SIZE)
REG_EMBEDDING_PARAMS = (2 if USE_REG_TOKEN else 1) * D_MODEL
FINAL_RMSNORM_PARAMS = RMSNORM_PARAMS

# our addition: the shared LinearAdapter, nn.Linear(768, 768) applied to each forecast slot
ADAPTER_PARAMS = D_MODEL * D_MODEL + D_MODEL

TOTAL_PARAMS = (NUM_BLOCKS * BLOCK_PARAMS + NATIVE_HEAD_PARAMS + INPUT_EMBEDDING_PARAMS
                + REG_EMBEDDING_PARAMS + FINAL_RMSNORM_PARAMS)

if TOTAL_PARAMS != RECORDED_TOTAL_PARAMS:                       # fail loud at import
    raise AssertionError(
        f"parameter accounting is inconsistent: the per-family constants sum to {TOTAL_PARAMS:,} but a "
        f"real full-FT run of amazon/chronos-2 recorded {RECORDED_TOTAL_PARAMS:,} trainable parameters. "
        "One of the constants above is wrong -- do NOT publish numbers derived from this module until "
        "it is fixed (run verify_against_model on a compute node to see which family disagrees).")


def param_breakdown() -> dict[str, int]:
    """Per-family parameter counts of the STOCK model (no adapter). Sums to TOTAL_PARAMS."""
    return {"input_patch_embedding": INPUT_EMBEDDING_PARAMS,
            "reg_embedding": REG_EMBEDDING_PARAMS,
            "encoder_blocks": NUM_BLOCKS * BLOCK_PARAMS,
            "final_layer_norm": FINAL_RMSNORM_PARAMS,
            "native_head": NATIVE_HEAD_PARAMS}


def active_params(depth: int, include_adapter: bool = True) -> int:
    """Parameters a model truncated after encoder block ``depth`` must load and execute.

    Retained: the input patch embedding, the REG embedding, blocks 1..depth, the linear adapter,
    the frozen final RMSNorm, and the native forecasting head. Blocks depth+1..12 are dropped.
    ``depth`` is a block index in 0..12 (0 = the embedding output, i.e. no transformer block runs).
    """
    if not 0 <= depth <= NUM_BLOCKS:
        raise ValueError(f"depth must be in 0..{NUM_BLOCKS}, got {depth}")
    return (INPUT_EMBEDDING_PARAMS + REG_EMBEDDING_PARAMS + depth * BLOCK_PARAMS
            + (ADAPTER_PARAMS if include_adapter else 0)
            + FINAL_RMSNORM_PARAMS + NATIVE_HEAD_PARAMS)


def active_fraction(depth: int, include_adapter: bool = True) -> float:
    """active_params(depth) / stock model size.

    NOTE the convention: the numerator includes our adapter, the denominator does not (it is the
    unmodified pretrained model). Hence depth=12 gives 1.005, not 1.0 -- you keep everything and
    add a layer. Report it as such; do not silently renormalise.
    """
    return active_params(depth, include_adapter) / TOTAL_PARAMS


def weight_bytes(depth: int, bytes_per_param: int = 4, include_adapter: bool = True) -> int:
    """Weight memory of the truncated model. 4 = float32 (what probing.extraction loads), 2 = bf16.

    This is weight memory only -- a floor on peak inference memory, not a measurement of it. At the
    token counts used here activations are small next to weights (see forward_macs' token count),
    but allocator overhead and workspace are real; measure with torch.cuda.max_memory_allocated().
    """
    return active_params(depth, include_adapter) * bytes_per_param


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #
def block_flops_fraction(depth: int) -> float:
    """Fraction of the ENCODER BLOCK STACK retained = depth / 12.

    This is exact by construction and needs no FLOP model: Chronos2Encoder builds
    ``nn.ModuleList([Chronos2EncoderBlock(config) for i in range(num_layers)])`` from one config, so
    all 12 blocks are architecturally identical and every block sees the same token sequence.
    Keeping ``depth`` of 12 therefore keeps exactly depth/12 of the block-stack cost.

    It is NOT end-to-end compute (the input embedding and native head do not shrink) -- use
    end_to_end_flops_fraction for that -- and it is NOT a latency claim; see experiments/run_latency.py.
    """
    if not 0 <= depth <= NUM_BLOCKS:
        raise ValueError(f"depth must be in 0..{NUM_BLOCKS}, got {depth}")
    return depth / NUM_BLOCKS


def num_encoder_tokens(context_length: int, horizon: int) -> tuple[int, int, int]:
    """(n_context_patches, n_forecast_slots, total_tokens) for one window.

    Verified against Chronos2Model.encode: input_embeds = [context patches, REG, future patches],
    i.e. ceil(C / input_patch_size) content tokens + 1 REG + K = ceil(H / output_patch_size) slots.
    """
    ncp = math.ceil(context_length / INPUT_PATCH_SIZE)
    k = math.ceil(horizon / OUTPUT_PATCH_SIZE)
    return ncp, k, ncp + (1 if USE_REG_TOKEN else 0) + k


def forward_macs(depth: int, context_length: int = 512, horizon: int = 64,
                 group_size: int = 1, include_adapter: bool = True) -> dict[str, float]:
    """Multiply-accumulate counts for one window through the truncated model.

    ESTIMATE, not a measurement. It counts matmul MACs only and ignores softmax, RMSNorm,
    activations and elementwise work, which understates non-GEMM cost. ``group_size`` = 1 is the
    univariate setting used throughout this project (encode() defaults group_ids to
    arange(batch_size), so each series is its own group and the group attention does no mixing --
    its q,k,v,o projections still run, which is why it costs as much as time attention).
    """
    ncp, k, ntok = num_encoder_tokens(context_length, horizon)
    emb_per_token = ((INPUT_PATCH_SIZE * INPUT_PATCH_CHANNELS) * D_FF + D_FF * D_MODEL
                     + (INPUT_PATCH_SIZE * INPUT_PATCH_CHANNELS) * D_MODEL)
    emb = (ncp + k) * emb_per_token            # applied to context AND future patches (model.py:599, :619)
    proj = 4 * D_MODEL * INNER_DIM
    time_attn = ntok * proj + 2 * ntok * ntok * D_MODEL
    group_attn = ntok * proj + 2 * ntok * group_size * D_MODEL
    ff = ntok * (2 * D_MODEL * D_FF)
    block = time_attn + group_attn + ff
    out_dim = NUM_NATIVE_QUANTILES * OUTPUT_PATCH_SIZE
    head = k * (D_MODEL * D_FF + D_FF * out_dim + D_MODEL * out_dim)
    adapter = k * D_MODEL * D_MODEL if include_adapter else 0
    return {"input_embedding": float(emb), "per_block": float(block),
            "blocks": float(depth * block), "native_head": float(head), "adapter": float(adapter),
            "total": float(emb + depth * block + adapter + head),
            "n_tokens": float(ntok), "n_forecast_slots": float(k)}


def end_to_end_flops_fraction(depth: int, context_length: int = 512, horizon: int = 64,
                              group_size: int = 1) -> float:
    """forward MACs of the truncated model / forward MACs of the full stock model.

    Larger than depth/12 because the input embedding and native head are fixed costs that do not
    shrink. Estimate (see forward_macs).
    """
    trunc = forward_macs(depth, context_length, horizon, group_size, include_adapter=True)["total"]
    full = forward_macs(NUM_BLOCKS, context_length, horizon, group_size, include_adapter=False)["total"]
    return trunc / full


# --------------------------------------------------------------------------- #
# verification against real weights (needs a loaded model -> compute node)
# --------------------------------------------------------------------------- #
def group_parameters(model) -> dict[str, int]:
    """Bucket ``model.named_parameters()` counts the same way probing.finetune._param_group does.

    Anything unmatched lands in an ``other::<name>`` key, so a silently-missing module is visible.
    """
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        if name.startswith("input_patch_embedding"):
            g = "input_patch_embedding"
        elif name.startswith("shared"):
            g = "reg_embedding"
        elif name.startswith("encoder.final_layer_norm"):
            g = "final_layer_norm"
        elif name.startswith("output_patch_embedding"):
            g = "native_head"
        elif name.startswith("encoder.block."):
            g = f"block_{int(name.split('.')[2]):02d}"
        else:
            g = f"other::{name}"
        groups[g] = groups.get(g, 0) + p.numel()
    return groups


def verify_against_model(model) -> dict:
    """Compare this module's constants with a real Chronos-2's weight tensors.

    Returns {"groups", "checks", "ok"}; every check is a (bool, expected, observed) triple so a
    caller can print exactly which family disagrees. Requires a loaded model (compute node).
    """
    groups = group_parameters(model)
    blocks = sorted(v for k, v in groups.items() if k.startswith("block_"))
    total = sum(groups.values())
    checks = {
        "no_unmatched_parameters": (not any(k.startswith("other::") for k in groups), 0,
                                    sum(v for k, v in groups.items() if k.startswith("other::"))),
        "num_blocks": (len(blocks) == NUM_BLOCKS, NUM_BLOCKS, len(blocks)),
        "blocks_identical": (len(set(blocks)) <= 1, 1, len(set(blocks))),
        "block_params": (bool(blocks) and blocks[0] == BLOCK_PARAMS, BLOCK_PARAMS,
                         blocks[0] if blocks else None),
        "input_patch_embedding": (groups.get("input_patch_embedding") == INPUT_EMBEDDING_PARAMS,
                                  INPUT_EMBEDDING_PARAMS, groups.get("input_patch_embedding")),
        "native_head": (groups.get("native_head") == NATIVE_HEAD_PARAMS, NATIVE_HEAD_PARAMS,
                        groups.get("native_head")),
        "reg_embedding": (groups.get("reg_embedding") == REG_EMBEDDING_PARAMS, REG_EMBEDDING_PARAMS,
                          groups.get("reg_embedding")),
        "final_layer_norm": (groups.get("final_layer_norm") == FINAL_RMSNORM_PARAMS,
                             FINAL_RMSNORM_PARAMS, groups.get("final_layer_norm")),
        "total": (total == TOTAL_PARAMS, TOTAL_PARAMS, total),
    }
    return {"groups": groups, "checks": checks, "ok": all(c[0] for c in checks.values())}


if __name__ == "__main__":                                       # python -m probing.model_size
    print(f"amazon/chronos-2 parameter accounting  (d_model={D_MODEL}, d_ff={D_FF}, "
          f"{NUM_BLOCKS} blocks, {NUM_NATIVE_QUANTILES} quantiles, patch {OUTPUT_PATCH_SIZE})\n")
    for k, v in param_breakdown().items():
        print(f"  {k:<24} {v:>12,}")
    print(f"  {'TOTAL (stock model)':<24} {TOTAL_PARAMS:>12,}   "
          f"(recorded: {RECORDED_TOTAL_PARAMS:,})")
    print(f"  {'linear adapter (ours)':<24} {ADAPTER_PARAMS:>12,}\n")
    ncp, k, ntok = num_encoder_tokens(512, 64)
    m = forward_macs(NUM_BLOCKS, include_adapter=False)
    print(f"  encoder tokens at C=512/H=64: {ncp} context + 1 REG + {k} forecast = {ntok}")
    print(f"  block stack share of forward MACs: "
          f"{100 * m['blocks'] / m['total']:.2f}%  (full forward {m['total'] / 1e6:.1f} MMAC)\n")
    print(f"  {'depth':>7}{'active params':>15}{'% of model':>12}{'block FLOPs':>13}"
          f"{'end-to-end':>12}{'fp32 weights':>14}")
    for d in range(NUM_BLOCKS + 1):
        print(f"  {('L%d' % d):>7}{active_params(d):>15,}{100 * active_fraction(d):>11.1f}%"
              f"{block_flops_fraction(d):>12.3f}x{end_to_end_flops_fraction(d):>11.3f}x"
              f"{weight_bytes(d) / 2 ** 20:>11.1f} MB")
