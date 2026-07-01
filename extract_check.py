"""
Milestone check: load Chronos-2 locally on Apple Silicon (MPS or CPU) and
extract hidden states from EVERY encoder layer for a single time-series sample.

This is an environment + per-layer extraction proof. It does not train,
probe, or plot anything.

Multivariate / group-ID handling is deferred: for tonight we pass a single
1-D univariate series (channel 0 of the sample) through the encoder.
"""

from __future__ import annotations

import importlib.metadata as md
import warnings

import numpy as np
import torch


def pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "<not installed>"


def main() -> None:
    # -------- Step 2: package versions --------
    print("=" * 72)
    print("Installed package versions:")
    for name in [
        "torch",
        "chronos-forecasting",
        "aeon",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
    ]:
        print(f"  {name:>22s} : {pkg_version(name)}")

    # -------- Step 3: MPS check --------
    mps_avail = torch.backends.mps.is_available()
    print(f"\ntorch.backends.mps.is_available() = {mps_avail}")
    device = torch.device("mps") if mps_avail else torch.device("cpu")
    print(f"Selected device: {device}")

    # -------- Step 4: chronos import --------
    import chronos
    from chronos import Chronos2Pipeline
    print(f"chronos-forecasting version: {chronos.__version__}")

    # -------- Step 5: pick checkpoint --------
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    candidates = ["amazon/chronos-2-small", "amazon/chronos-2"]
    checkpoint: str | None = None
    for cand in candidates:
        try:
            api.repo_info(cand)
            checkpoint = cand
            break
        except RepositoryNotFoundError:
            print(f"  checkpoint not on Hub: {cand}")
    if checkpoint is None:
        raise RuntimeError("No Chronos-2 checkpoint found on the Hub")
    print(f"Using checkpoint: {checkpoint}")

    # -------- Step 6: load model (float32, MPS or CPU) --------
    # device_map="mps" can be flaky during from_pretrained; load on CPU then move.
    print("\nLoading pipeline (float32, CPU first)...")
    pipeline = Chronos2Pipeline.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
    )
    inner_model = pipeline.model  # Chronos2Model (PreTrainedModel)

    # Move to target device.
    try:
        inner_model.to(device)
        final_device = device
    except Exception as e:
        warnings.warn(f"Could not move model to {device}: {e}; falling back to CPU.")
        inner_model.to("cpu")
        final_device = torch.device("cpu")
    print(f"Final model device: {next(inner_model.parameters()).device}  (target was {final_device})")

    # -------- Step 7: eval + no_grad --------
    inner_model.eval()
    for p in inner_model.parameters():
        p.requires_grad_(False)

    # -------- Step 8: introspection --------
    cfg = inner_model.config
    print("\n--- Model config (key fields) ---")
    print(f"  model_type   : {cfg.model_type}")
    print(f"  d_model      : {cfg.d_model}")
    print(f"  num_layers   : {cfg.num_layers}")
    print(f"  num_heads    : {cfg.num_heads}")
    print(f"  d_ff         : {cfg.d_ff}")
    print(f"  vocab_size   : {cfg.vocab_size}")
    chronos_cfg = inner_model.chronos_config
    print(f"  context_len  : {chronos_cfg.context_length}")
    print(f"  patch_size   : {chronos_cfg.input_patch_size}")

    print("\n--- Abbreviated module tree (top 2 levels) ---")
    for name, mod in inner_model.named_modules():
        depth = 0 if name == "" else name.count(".") + 1
        if depth > 2:
            continue
        cls = type(mod).__name__
        if name == "":
            print(f"  <root>  {cls}")
        else:
            print(f"  {'  ' * depth}{name}  ({cls})")

    # Identify the repeated encoder-layer modules.
    encoder = inner_model.encoder
    layer_blocks = encoder.block  # nn.ModuleList of Chronos2EncoderBlock
    print(f"\nIdentified encoder layer stack: encoder.block (len={len(layer_blocks)}, "
          f"each {type(layer_blocks[0]).__name__})")

    # -------- Step 9: load BasicMotions --------
    print("\nLoading aeon BasicMotions...")
    from aeon.datasets import load_classification

    X, y = load_classification("BasicMotions")
    print(f"X.shape = {X.shape}   (n_cases x n_channels x n_timepoints)")
    print(f"Unique labels: {sorted(np.unique(y).tolist())}")

    # -------- Step 10: one sample --------
    sample = X[0]  # (n_channels, n_timepoints)
    print(f"sample.shape = {sample.shape}   (channels x timepoints)")

    # -------- Step 11: hook every encoder block --------
    captured: list[torch.Tensor] = []
    hook_handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # Chronos2EncoderBlockOutput is a HF ModelOutput (dict-like) with
            # .hidden_states; the encoder also reads it as layer_outputs[0].
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                hs = output.hidden_states
            elif isinstance(output, (tuple, list)):
                hs = output[0]
            else:
                hs = output[0]  # ModelOutput subscript falls back to first value
            captured.append(hs.detach())
        return hook

    for i, blk in enumerate(layer_blocks):
        hook_handles.append(blk.register_forward_hook(make_hook(i)))

    # Cross-check: does the inner forward path accept output_hidden_states=True?
    import inspect as _inspect
    enc_fwd_params = _inspect.signature(encoder.forward).parameters
    supports_ohs = "output_hidden_states" in enc_fwd_params
    print(f"\nEncoder.forward supports output_hidden_states kwarg: {supports_ohs}")
    print("  (Chronos2Encoder only exposes output_attentions, not output_hidden_states — "
          "hooks are the right path.)")

    # -------- Step 12: single forward pass via pipeline.embed --------
    # Univariate input: a list containing one 1-D float32 tensor (channel 0).
    series_1d = torch.tensor(sample[0], dtype=torch.float32)  # (n_timepoints,)
    print(f"\nUnivariate input tensor: shape={tuple(series_1d.shape)}, dtype={series_1d.dtype}")

    # NOTE: multivariate / group-ID handling is deferred. embed() still runs the
    # full encoder forward, so our hooks fire even though we ignore its return value.
    with torch.no_grad():
        _ = pipeline.embed([series_1d])

    for h in hook_handles:
        h.remove()

    # -------- Step 13: report shapes --------
    print(f"\n--- Captured layers: {len(captured)} ---")
    for i, t in enumerate(captured):
        print(f"  layer[{i}]: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")

    # Mean-pool over the sequence/patch dimension (dim=-2 = num_patches axis).
    pooled = [t.mean(dim=-2) for t in captured]
    print("\nAfter mean-pool over the patch/sequence dim:")
    for i, p in enumerate(pooled):
        print(f"  layer[{i}] pooled shape={tuple(p.shape)}  (expected last dim = d_model={cfg.d_model})")
    assert all(p.shape[-1] == cfg.d_model for p in pooled), \
        "Per-layer pooled vector dim != d_model"

    # -------- Step 14: reconcile against config --------
    n_captured = len(captured)
    n_cfg = cfg.num_layers
    ok = (n_captured == n_cfg) or (n_captured == n_cfg + 1)
    print(f"\nReconciliation: captured={n_captured}, config.num_layers={n_cfg}  -> {'OK' if ok else 'MISMATCH'}")
    assert ok, f"Expected {n_cfg} or {n_cfg + 1} captured layers, got {n_captured}"

    # -------- Step 15: numpy round-trip --------
    # The pooled tensor for the first layer has shape (batch_or_groups, d_model);
    # take the first row as a single per-layer pooled vector.
    v0 = pooled[0]
    while v0.ndim > 1:
        v0 = v0[0]
    v0_np = v0.detach().to("cpu").to(torch.float32).numpy()
    print(f"\nOne pooled vector -> numpy: shape={v0_np.shape}, dtype={v0_np.dtype}")
    assert v0_np.dtype == np.float32 and v0_np.shape == (cfg.d_model,)

    # -------- Step 16: final summary --------
    print("\n" + "=" * 72)
    print("EXTRACTION MILESTONE: PASS")
    print(
        f"  checkpoint={checkpoint}  device={next(inner_model.parameters()).device}  "
        f"#layers_captured={n_captured}  d_model={cfg.d_model}"
    )


if __name__ == "__main__":
    main()
