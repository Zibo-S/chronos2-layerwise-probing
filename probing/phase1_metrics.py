"""Raw-unit Phase-1 test metrics: MASE, MAE and WQL, computed identically for all three models.

The probe loss lives in each model's OWN normalized target space and is therefore not
comparable across models (``phase1.CROSS_MODEL_LOSS_CAVEAT``). These three are: they are
computed on the RAW series values, from the same windows, with the same denominator.

MASE. The reported denominator is the IN-CONTEXT seasonal-naive scale over the C=512 context
window the probe itself sees, with the dataset-specific period from the registry:

    den_w = mean_t | x_{w,t} - x_{w,t-m} |        over the valid lagged pairs of window w
    MASE_w = mean_h | y_{w,h} - yhat_{w,h} | / den_w

That is ``probing.mase`` — the one implementation — and this module only calls it. Two things
about it that belong in the paper and are repeated in every summary:
  * it is NOT the fev-bench / GIFT-Eval denominator (which uses the full pre-test history), and
    a reviewer will notice. It is what every committed number in this repository was computed
    with, and for the NaN-heavy datasets (wind_farms 96.3% of series, london 98.5%) the
    canonical one is not even available — ``id_data``'s ``test_denominator`` uses ``.mean()``
    and is NaN for any series with a single missing reading anywhere in its history.
  * the in-context denominator cannot fail: the window validity filter guarantees a finite
    context. The floor ``MASE_DEN_FLOOR`` therefore fires only on an exactly-constant context,
    and the count of windows it touched is recorded so a dataset where it fires is visible.

The point forecast for MASE and MAE is the MEDIAN (tau = 0.5) quantile — never a mean, never a
neighbouring level. The quantile set must contain an exact 0.5 or no median metric is produced.

WQL (weighted quantile loss) is the scale-free probabilistic metric the forecasting literature
reports. It is a RATIO of sums, so its bootstrap replicate must be formed as
sum(numerator)/sum(denominator) INSIDE each replicate — which is why the numerator and
denominator are returned separately per window and never pre-divided.
"""

from __future__ import annotations

import numpy as np

from probing.mase import MASE_DEN_FLOOR, floored_denominator, per_window_mase

__all__ = ["MASE_DEN_FLOOR", "raw_window_metrics", "wql_parts"]


def wql_parts(y_raw, quant_raw, quantiles):
    """Per-window WQL numerator and denominator, kept SEPARATE for the bootstrap.

    ``quant_raw`` is (n, Q, H) in raw units; ``y_raw`` is (n, H).
        numerator_w   = 2 * sum_{q,h} rho_tau(y - yhat)
        denominator_w = sum_h |y_h|
    WQL = sum_w numerator_w / sum_w denominator_w. Forming the ratio inside each bootstrap
    replicate (rather than averaging per-window ratios) is what makes the CI describe the
    reported statistic. Verbatim the convention of ``run_native_head_adapter._wql_pw_parts``.
    """
    y = np.asarray(y_raw, np.float64)[:, None, :]
    qq = np.asarray(quant_raw, np.float64)
    tau = np.asarray(quantiles, np.float64)[None, :, None]
    pinball = np.where(y >= qq, tau * (y - qq), (1.0 - tau) * (qq - y))
    return 2.0 * pinball.sum(axis=(1, 2)), np.abs(np.asarray(y_raw, np.float64)).sum(axis=1)


def raw_window_metrics(tag: str, contexts, y_raw, quant_raw, quantiles, median_idx: int) -> dict:
    """Per-window MASE / MAE / WQL parts for ONE representation point's raw-unit forecast.

    ``contexts`` (n, C) are the raw context windows the denominator is built from — the SAME
    windows the probe read. ``quant_raw`` is (n, Q, H) in raw units.
    """
    y = np.asarray(y_raw, np.float64)
    qq = np.asarray(quant_raw, np.float64)
    if qq.ndim != 3 or qq.shape[0] != y.shape[0] or qq.shape[2] != y.shape[1]:
        raise ValueError(f"raw quantile forecast {qq.shape} incompatible with targets {y.shape}")
    if qq.shape[1] != len(quantiles):
        raise ValueError(f"forecast has {qq.shape[1]} quantile rows, {len(quantiles)} levels given")
    den, n_clamped = floored_denominator(tag, contexts)
    med = qq[:, median_idx, :]
    num, wden = wql_parts(y, qq, quantiles)
    return {"mase_pw": per_window_mase(y, med, den),
            "mae_pw": np.abs(y - med).mean(axis=1),
            "wql_num_pw": num, "wql_den_pw": wden,
            "denominator": den, "n_denominator_clamped": int(n_clamped),
            "median_raw": med}
