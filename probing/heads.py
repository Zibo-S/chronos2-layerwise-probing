"""Higher-capacity forecasting-probe heads (capacity controls).

These reimplement Chronos-2's native output head STRUCTURE — a nonlinear ResidualBlock
(chronos/chronos2/layers.py:414, `output_patch_embedding` in model.py:265) — as
freshly-initialized probe heads. They are NOT the pretrained native head: the weights are
trained from scratch per layer, so a good forecast reflects a *representation's* decodability
under a higher-capacity readout, not a decoder pre-optimized for the final layer.

One module (`ResidualBlock`) serves two head SHAPES, selected by how the probe applies it:

  - content_mlp_head          : ResidualBlock(768 -> hidden -> Q*H) applied to the (n, 768)
                                mean-pooled content vector -> (n, Q*H) -> (n, Q, H).
                                Not fully native-faithful (one pooled vector, not the K
                                forecast slots) — it isolates "replace the linear map with a
                                nonlinear one" at fixed content-pooled input.
  - forecast_slot_native_head : ONE shared ResidualBlock(768 -> hidden -> Q*P) applied to
                                every one of the K native forecast slots (n, K, 768); the K
                                output patches concatenate to (n, Q, K*P) and trim to H.
                                Weight-sharing across slots matches the native head.

Structure matches the native block EXACTLY (native `use_layer_norm=False`):
    hid = ReLU(Linear(in -> hidden)(x))
    out = Dropout(Linear(hidden -> out)(hid)) + Linear(in -> out)(x)      # NO LayerNorm
`hidden` defaults to the native d_ff = 3072. Dropout defaults to 0 for a deterministic probe
(the per-layer weight-decay grid already regularizes); the native head uses 0.1.

Every op broadcasts over leading dims, so the same instance handles the pooled (n, 768) input
and the shared-slot (n, K, 768) input without change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Chronos-2 config facts (amazon/chronos-2 config.json), verified from the installed package.
NATIVE_D_FF = 3072            # head hidden width (d_ff)
NATIVE_DROPOUT = 0.1          # native head dropout (probe default is 0 — see module docstring)


class ResidualBlock(nn.Module):
    """Structural clone of Chronos-2's ResidualBlock with use_layer_norm=False.

    Broadcasts over any leading dims, so ONE instance serves both the pooled content head
    (input (n, 768)) and the shared forecast-slot head (input (n, K, 768))."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        return out + self.residual_layer(x)


def build_head(in_features: int, out_features: int, hidden_dim: int = NATIVE_D_FF,
               dropout: float = 0.0, device=None) -> ResidualBlock:
    """Construct one probe head. `torch.manual_seed(SEED)` must be called by the FITTER right
    before this (as _fit_quantile_linear does for the linear probe) so every layer / weight-decay
    candidate starts from the identical deterministic init."""
    head = ResidualBlock(in_features, hidden_dim, out_features, dropout=dropout)
    if device is not None:
        head = head.to(device)
    return head


def head_param_count(head: nn.Module) -> int:
    """Total trainable parameters in one head (recorded per family in the results)."""
    return int(sum(p.numel() for p in head.parameters()))


def wd_param_groups(head: nn.Module, weight_decay: float):
    """AdamW param groups mirroring the linear probe's convention: weight-decay on the WEIGHT
    matrices only, none on the biases (the pinball-optimal output bias is the target quantile
    vector, so decaying it shrinks every predicted quantile toward 0). Applies to all three
    Linear sub-layers of the block."""
    decay, no_decay = [], []
    for name, p in head.named_parameters():
        (no_decay if name.endswith("bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
