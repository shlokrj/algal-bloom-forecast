import tempfile
import unittest
from pathlib import Path

from algal_bloom_forecast.data.noaa_ci_timeseries import (
    merge_ci_timeseries,
    parse_ci_timeseries,
    profile_ci_timeseries,
)


class NoaaCiTimeseriesTests(unittest.TestCase):
    def test_profiles_fused_series_and_preserves_missing_values(self):
        content = """\
Date,Western Lake Erie_meris,Western Lake Erie_fused
2000-06-06,,0.25
2000-06-16,,
2000-06-26,0.4,0.4
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Western Lake Erie_CIsum_timeseries.csv"
            path.write_text(content, encoding="utf-8")
            metric, rows, _ = parse_ci_timeseries(path)
            profile = profile_ci_timeseries(path)

        self.assertEqual(metric, "ci_sum")
        self.assertEqual(rows[1]["value"], None)
        self.assertEqual(profile["records"], 3)
        self.assertEqual(profile["missing_fused_records"], 1)
        self.assertEqual(profile["date_interval_days"], {10: 2})

    def test_merges_metrics_without_interpolating(self):
        merged = merge_ci_timeseries(
            {
                "ci_sum": [
                    {"observation_date": "2000-06-06", "value": 0.25},
                    {"observation_date": "2000-06-16", "value": None},
                ],
                "bloom_area_sqkm": [
                    {"observation_date": "2000-06-16", "value": 50.0},
                ],
            }
        )

        self.assertEqual(
            merged,
            [
                {"observation_date": "2000-06-06", "ci_sum": 0.25},
                {
                    "observation_date": "2000-06-16",
                    "ci_sum": None,
                    "bloom_area_sqkm": 50.0,
                },
            ],
        )
