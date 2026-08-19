from __future__ import annotations

import pytest

from algal_bloom_forecast.data.target_audit import audit_target_records


def test_audit_profiles_intervals_and_missing_values() -> None:
    report = audit_target_records(
        [
            {"observation_date": "2024-06-01", "ci_sum": 1.0, "bloom_area_sqkm": 10.0},
            {"observation_date": "2024-06-11", "ci_sum": None, "bloom_area_sqkm": 20.0},
            {"observation_date": "2024-06-21", "ci_sum": 3.0, "bloom_area_sqkm": None},
        ]
    )

    assert report["date_interval_days"] == {10: 2}
    assert report["unexpected_interval_records"] == 0
    assert report["fields"]["ci_sum"]["missing"] == 1
    assert report["fields"]["bloom_area_sqkm"]["missing_rate"] == pytest.approx(1 / 3)


def test_audit_rejects_duplicate_or_unsorted_dates() -> None:
    with pytest.raises(ValueError, match="unique"):
        audit_target_records(
            [
                {"observation_date": "2024-06-01", "ci_sum": 1.0},
                {"observation_date": "2024-06-01", "ci_sum": 2.0},
            ]
        )
    with pytest.raises(ValueError, match="sorted"):
        audit_target_records(
            [
                {"observation_date": "2024-06-11", "ci_sum": 2.0},
                {"observation_date": "2024-06-01", "ci_sum": 1.0},
            ]
        )
