from __future__ import annotations

import numpy as np

from scripts.run_spatial_persistence_baseline import _pixel_metrics


def test_pixel_metrics_score_only_shared_valid_pixels() -> None:
    metrics = _pixel_metrics(
        np.array([[1.0, 100.0], [3.0, 4.0]]),
        np.array([[2.0, 0.0], [3.0, 6.0]]),
        np.array([[True, False], [False, True]]),
    )

    assert metrics["overlap_pixel_count"] == 2
    assert metrics["mae"] == 1.5
    assert metrics["rmse"] == 1.5811388300841898
