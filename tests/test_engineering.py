import unittest

from algal_bloom_forecast.features.engineering import (
    USGS_CFS_TO_M3S,
    build_feature_records,
    normalize_predictor_records,
)


class EngineeringTests(unittest.TestCase):
    def test_normalization_adds_si_discharge_and_quality_flag(self):
        normalized = normalize_predictor_records(
            [
                {
                    "observation_date": "2024-01-01",
                    "usgs_maumee_discharge_cfs": 10.0,
                    "usgs_maumee_discharge_estimated": True,
                }
            ]
        )
        self.assertAlmostEqual(normalized[0]["usgs_maumee_discharge_m3s"], 10.0 * USGS_CFS_TO_M3S)
        self.assertEqual(normalized[0]["usgs_maumee_discharge_estimated_flag"], 1)

    def test_lags_and_rolling_windows_use_only_pre_cutoff_records(self):
        features = build_feature_records(
            [{"observation_date": "2024-01-10", "ci_sum": 0.2}],
            [
                {"observation_date": "2024-01-01", "signal": 1.0},
                {"observation_date": "2024-01-05", "signal": 5.0},
                {"observation_date": "2024-01-07", "signal": 7.0},
                {"observation_date": "2024-01-08", "signal": 8.0},
            ],
            horizons=(3,),
            lag_days=(1,),
            rolling_windows_days=(3,),
        )
        row = features[0]
        self.assertEqual(row["predictor_date"], "2024-01-07")
        self.assertEqual(row["signal"], 7.0)
        self.assertEqual(row["lag_1d_signal"], 5.0)
        self.assertEqual(row["rolling_3d_signal_mean"], 6.0)
        self.assertEqual(row["rolling_3d_signal_valid_count"], 2)
        self.assertNotEqual(row["signal"], 8.0)

    def test_feature_builder_rejects_target_predictor_collisions(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            build_feature_records(
                [{"observation_date": "2024-01-02", "signal": 1.0}],
                [{"observation_date": "2024-01-01", "signal": 2.0}],
                horizons=(1,),
            )
