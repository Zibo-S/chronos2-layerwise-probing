"""Smoke tests: the standalone ID-only figures (binned-accuracy and ridge-R²) are produced with
NO UEA dependency — even when results/uea/ is missing or still 12-layer. No GPU/model/cache needed."""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from experiments.run_id_forecasting import make_ridge_r2_plot, make_binned_accuracy_plot
from probing.config import NUM_LAYERS


def _fake_id_results():
    tags = ["monash_electricity_hourly", "uber_tlc_hourly"]
    rng = np.random.default_rng(0)
    return {t: {"poolings": {p: {"ridge_r2": rng.uniform(-0.1, 0.6, NUM_LAYERS).tolist(),
                                 "binned_accuracy": rng.uniform(0.2, 0.9, NUM_LAYERS).tolist()}
                             for p in ("content", "reg")}} for t in tags}


def _stale_12layer_uea(tmp):
    uea = tmp / "uea"; uea.mkdir()
    (uea / "perdataset_summary.json").write_text(
        '{"datasets": {"X": {"per_layer_accuracy": {"ID": %s}}}}' % ([0] * 12))


def test_ridge_r2_plot_created_without_uea():
    # No results/uea/ exists anywhere; the plot must still be written (13-layer curves).
    tmp = Path(tempfile.mkdtemp())
    out = make_ridge_r2_plot(_fake_id_results(), out_dir=tmp)
    assert out.exists() and out.stat().st_size > 0


def test_ridge_r2_plot_ignores_stale_12layer_uea():
    # A stale 12-layer UEA summary on disk must not affect the plot (the fn takes no UEA arg).
    tmp = Path(tempfile.mkdtemp())
    _stale_12layer_uea(tmp)
    out = make_ridge_r2_plot(_fake_id_results(), out_dir=tmp)
    assert out.exists() and out.stat().st_size > 0


def test_binned_accuracy_plot_created_without_uea():
    tmp = Path(tempfile.mkdtemp())
    out = make_binned_accuracy_plot(_fake_id_results(), out_dir=tmp)
    assert out.exists() and out.stat().st_size > 0


def test_binned_accuracy_plot_ignores_stale_12layer_uea():
    tmp = Path(tempfile.mkdtemp())
    _stale_12layer_uea(tmp)
    out = make_binned_accuracy_plot(_fake_id_results(), out_dir=tmp)
    assert out.exists() and out.stat().st_size > 0


TESTS = [test_ridge_r2_plot_created_without_uea,
         test_ridge_r2_plot_ignores_stale_12layer_uea,
         test_binned_accuracy_plot_created_without_uea,
         test_binned_accuracy_plot_ignores_stale_12layer_uea]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(TESTS)} TESTS PASS")
