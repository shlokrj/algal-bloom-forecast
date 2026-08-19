from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.validate_training_ready import validate


def test_validate_accepts_a_minimal_leakage_safe_frame(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "forecast_horizon_days",
                "observation_date",
                "predictor_date",
                "target_ci_sum",
                "feature_a",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "forecast_horizon_days": 1,
                "observation_date": "2024-06-11",
                "predictor_date": "2024-06-10",
                "target_ci_sum": 2.0,
                "feature_a": 1.0,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": {
                    "status": "prepared_not_fitted",
                    "id_fields": [
                        "split",
                        "forecast_horizon_days",
                        "observation_date",
                        "predictor_date",
                    ],
                    "target": "target_ci_sum",
                    "feature_names": ["feature_a"],
                    "feature_count": 1,
                    "rows": 1,
                    "split_counts": {"train": 1},
                },
                "validation": {"model_fit_started": False},
                "output_path": str(frame_path),
            }
        ),
        encoding="utf-8",
    )

    report = validate(manifest_path, frame_path)

    assert report["future_predictor_rows"] == 0
    assert report["status"] == "prepared_not_fitted"
