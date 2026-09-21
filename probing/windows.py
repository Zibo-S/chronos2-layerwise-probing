"""THE window-construction dispatch: one function every model line and audit tool calls.

Which builder a dataset gets is a property of the DATASET — how its series are organized —
so it comes from ``registry.builder(tag)``:

    rolling_within_series : ``id_data.build_windows`` under a ROLLING_SETS dataset set. One
                            series per bootstrap unit; H-spaced non-overlapping origins; per
                            eligible series the last origin is test, the 2nd-last is val.
    rolling_cluster       : ``id_data.build_ood_rolling_windows``. Same per-series protocol,
                            but the bootstrap unit is the PARENT CLUSTER (carpark / station /
                            metric-query), because several series can share one unit.
    legacy_auto_split     : the pre-rolling length-derived within_series/cross_series path.

This used to be written out inside ``run_timesfm3_last_token_probing.windows_for`` and again,
differently spelled, inside ``run_native_head_adapter._windows`` — and both decided the builder
from ``tag in PT_OOD_TAGS``, i.e. from a Chronos-2-relative pretraining label. Two unrelated
facts were tied together, and a dataset outside that label's roster had no defined builder at
all.

This module imports no torch, so window building, the roster audit and the window smoke all run
without a deep-learning environment and without a GPU.
"""

from __future__ import annotations

from probing import registry
from probing.config import SEED

__all__ = ["PAPER7_SET", "PAPER14_SET", "rolling_set_for_suite", "build_for"]

#: Dataset-set names that carry the (1394, 262, 262) rolling budget. The seven keep their own
#: set so their committed windows are rebuilt by exactly the same code path as before.
PAPER7_SET = "extended_v3_rolling"
PAPER14_SET = "paper14_rolling"


def rolling_set_for_suite(suite: str) -> str:
    """The ``probing.config.DATASET_SET`` a paper suite builds its PT-ID windows under."""
    if suite == "paper7":
        return PAPER7_SET
    if suite == "paper14":
        return PAPER14_SET
    raise ValueError(f"{suite!r} is not a paper suite; known: 'paper7', 'paper14'")


def build_for(tag: str, suite: str = "paper14", *, C: int = 512, H: int = 64,
              seed: int = SEED, stride: int = 64):
    """Build one dataset's windows, dispatching on ``registry.builder(tag)``.

    ``suite`` selects which rolling dataset-set (and therefore which budget/namespace) the
    within-series builder runs under. A non-paper suite falls through to the legacy auto-split
    path with the caller's C/H/stride/seed, unchanged.
    """
    from probing import config
    from probing.id_data import build_ood_rolling_windows, build_windows

    if suite not in ("paper7", "paper14"):
        config.set_dataset_set(suite)
        return build_windows(tag, C=C, H=H, stride=stride, seed=seed)

    kind = registry.builder(tag)
    if kind == "rolling_cluster":
        return build_ood_rolling_windows(tag, C=C, H=H, seed=seed)
    if kind != "rolling_within_series":
        raise ValueError(
            f"{tag}: builder {kind!r} has no rolling-origin window path, so it cannot be part "
            f"of the {suite!r} suite. Declare it as rolling_within_series or rolling_cluster in "
            f"probing/registry.py, or keep it out of the paper roster.")
    config.set_dataset_set(rolling_set_for_suite(suite))
    return build_windows(tag)
