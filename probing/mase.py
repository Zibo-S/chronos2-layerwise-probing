"""THE reported MASE: one implementation, one definition string, one seasonal period per dataset.

    MASE is normalized by the seasonal-naive error computed over the 512-step context window
    using the dataset-specific seasonal period.

That sentence (``registry.MASE_DEFINITION``) is what the paper states, and this module is the
only place it is computed.

WHY THIS MODULE EXISTS

1. The formula used to live twice — ``run_id_forecasting._mase_denominator`` (Chronos-2) and
   ``run_timesfm3_probing.mase_denominator`` (TimesFM-3 / TiRex) — as byte-identical copies.
   Two copies of a metric is one copy too many.

2. Both copies defaulted ``m`` to a module-global ``M_SEASON = 24``. That was correct for the
   seven hourly datasets and silently WRONG for every dataset added since: at 5-minute
   sampling, m=24 is a two-hour "season". It would not have crashed. It would have printed a
   number. ``seasonal_denominator`` therefore takes ``m`` as a REQUIRED argument and
   :func:`denominator_for` is the only convenient way to get one — from the registry, keyed by
   dataset tag, with no fallback.

WHAT IS *NOT* CHANGED HERE. The reported denominator stays the IN-CONTEXT seasonal-naive
scale, computed from the finite C=512 context window that the probe itself sees. It is
deliberately NOT ``id_data``'s canonical per-series ``test_denominator`` (the seasonal-naive
scale of the series' history before the test target), which is computed, stored, and consumed
by nothing. Two reasons to keep it that way:

  * reproducibility — every committed Chronos-2 / TimesFM-3 / TiRex number in ``results/`` was
    produced with the in-context denominator, and this refactor must not move them;
  * availability — the canonical denominator uses ``.mean()`` over the full pre-test history,
    so a single NaN anywhere in a series makes it NaN. The 2026-09-21 yield screen measured
    that failure at 96.3% of series for wind_farms_hourly, 98.5% for london_smart_meters and
    100% for kdd_cup_2022_10T. The in-context denominator cannot fail: the window validity
    filter guarantees the context is finite.

The honest consequence, which belongs in the paper: this is not the fev-bench / GIFT-Eval
denominator, and for the NaN-heavy datasets we could not switch to that one even if asked.
"""

from __future__ import annotations

import numpy as np

from probing.registry import MASE_DEFINITION, seasonal_m

__all__ = ["MASE_DEFINITION", "MASE_DEN_FLOOR", "seasonal_denominator", "denominator_for",
           "floored_denominator", "per_window_mase", "mase_definition_for"]

#: Guard against dividing by an exactly-zero denominator. It is a floor, not a fudge: every
#: driver that applies it also records how many windows it touched (``n_denominator_clamped``),
#: so a dataset where it fires often is visible rather than silently rescaled.
MASE_DEN_FLOOR = 1e-8


def seasonal_denominator(X, m: int) -> np.ndarray:
    """In-context seasonal-naive scale, one value per window: ``mean_t |x_t - x_{t-m}|``.

    ``X`` is (n, C) context windows; ``m`` is the seasonal period and is REQUIRED — there is no
    default, because a wrong-but-plausible default produces a wrong number instead of an error.

    The arithmetic is byte-identical to the two implementations this replaces, so every
    committed number reproduces exactly when ``m`` is the 24 those datasets resolve to.
    """
    X64 = np.asarray(X, np.float64)
    if X64.ndim != 2:
        raise ValueError(f"context windows must be 2-D (n, C), got {X64.shape}")
    m = int(m)
    if not 1 <= m < X64.shape[1]:
        raise ValueError(
            f"seasonal period m={m} does not fit a context of length {X64.shape[1]}: it must "
            f"satisfy 1 <= m < C so at least one lagged pair exists. A too-large m would "
            f"average an empty slice and return NaN silently.")
    return np.abs(X64[:, m:] - X64[:, :-m]).mean(axis=1)


def denominator_for(tag: str, X) -> np.ndarray:
    """:func:`seasonal_denominator` with ``m`` resolved from the registry for ``tag``.

    This is the call every driver should make. An unregistered tag raises rather than falling
    back to any default period.
    """
    return seasonal_denominator(X, seasonal_m(tag))


def floored_denominator(tag: str, X, floor: float = MASE_DEN_FLOOR) -> tuple[np.ndarray, int]:
    """``(denominator clamped to >= floor, how many windows were clamped)``.

    Returning the clamp COUNT alongside the values is the point: a driver that records it can
    show the floor was inert, and one where it fires a lot has a dataset problem to report
    rather than a metric to quote.
    """
    den = denominator_for(tag, X)
    n_clamped = int((den < floor).sum())
    return np.maximum(den, floor), n_clamped


def per_window_mase(y_raw, yhat_raw, den) -> np.ndarray:
    """Per-window MASE: mean absolute error over the horizon, divided by that window's scale."""
    return (np.abs(np.asarray(y_raw, np.float64) - np.asarray(yhat_raw, np.float64))
            / np.asarray(den, np.float64)[:, None]).mean(axis=1)


def mase_definition_for(tag: str) -> str:
    """The paper sentence with this dataset's actual period substituted in, for run summaries."""
    return (f"{MASE_DEFINITION} For {tag} the seasonal period is m={seasonal_m(tag)} "
            f"(floor {MASE_DEN_FLOOR}).")
