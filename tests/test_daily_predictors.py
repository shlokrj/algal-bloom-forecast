import unittest
from pathlib import Path

from algal_bloom_forecast.data.glerl import aggregate_glerl_continuous, parse_glerl_continuous_csv
from algal_bloom_forecast.data.ndbc import aggregate_standard_meteorology
from algal_bloom_forecast.data.usgs import parse_daily_values_payload
from algal_bloom_forecast.features.daily import merge_daily_predictor_records


class DailyPredictorTests(unittest.TestCase):
    def test_usgs_normalization_sorts_and_marks_estimated_values(self):
        records = parse_daily_values_payload(
            {
                "features": [
                    {
                        "properties": {
                            "time": "2024-01-02",
                            "value": "2.5",
                            "qualifier": ["ESTIMATED"],
                        }
                    },
                    {"properties": {"time": "2024-01-01", "value": "1.5", "qualifier": None}},
                ]
            }
        )
        self.assertEqual(records[0]["observation_date"], "2024-01-01")
        self.assertEqual(records[1]["usgs_maumee_discharge_cfs"], 2.5)
        self.assertTrue(records[1]["usgs_maumee_discharge_estimated"])

    def test_ndbc_aggregation_preserves_missing_counts(self):
        daily = aggregate_standard_meteorology(
            [
                {"timestamp": "2024-01-01T00:00:00+00:00", "WSPD": 2.0, "WDIR": 350, "ATMP": None},
                {"timestamp": "2024-01-01T12:00:00+00:00", "WSPD": None, "WDIR": 10, "ATMP": 4.0},
            ]
        )
        self.assertEqual(daily[0]["ndbc_sample_count"], 2)
        self.assertEqual(daily[0]["ndbc_wspd_mean"], 2.0)
        self.assertEqual(daily[0]["ndbc_wspd_valid_count"], 1)
        self.assertAlmostEqual(daily[0]["ndbc_wdir_circular_mean"], 0.0)
        self.assertEqual(daily[0]["ndbc_atmp_valid_count"], 1)

    def test_glerl_annual_summary_skips_metadata_rows(self):
        content = """\
timestamp,water_temperature,water_temperature_flags,turbidity,turbidity_flags
UTC,degrees Celsius,NAN,NTU,NAN
data logger,YSI,NAN,YSI,NAN
1/1/2016 00:00,10.0,1,2.0,1
1/1/2016 00:15,12.0,1,NAN,1
"""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "WE02_2016_annual_summary.csv"
            path.write_text(content, encoding="latin-1")
            rows = parse_glerl_continuous_csv(path)
            daily = aggregate_glerl_continuous([path])

        self.assertEqual(len(rows), 2)
        self.assertEqual(daily[0]["glerl_continuous_record_count"], 2)
        self.assertEqual(daily[0]["glerl_we02_water_temperature_mean"], 11.0)
        self.assertEqual(daily[0]["glerl_we02_turbidity_valid_count"], 1)

    def test_merge_outer_joins_without_interpolation(self):
        merged = merge_daily_predictor_records(
            {
                "usgs": [{"observation_date": "2024-01-01", "discharge": 1.0}],
                "ndbc": [{"observation_date": "2024-01-02", "wind": 2.0}],
            }
        )
        self.assertEqual(
            merged[0], {"observation_date": "2024-01-01", "discharge": 1.0, "wind": None}
        )
        self.assertEqual(
            merged[1], {"observation_date": "2024-01-02", "discharge": None, "wind": 2.0}
        )

    def test_merge_rejects_field_collisions(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            merge_daily_predictor_records(
                {
                    "left": [{"observation_date": "2024-01-01", "value": 1}],
                    "right": [{"observation_date": "2024-01-01", "value": 2}],
                }
            )
