import unittest

from algal_bloom_forecast.features.quality import profile_aligned_records, validate_aligned_records


class QualityTests(unittest.TestCase):
    def test_valid_alignment_profiles_missingness(self):
        records = [
            {
                "forecast_horizon_days": 3,
                "observation_date": "2024-01-10",
                "predictor_date": "2024-01-07",
                "feature_lag_days": 3,
                "ci_sum": 0.2,
            },
            {
                "forecast_horizon_days": 3,
                "observation_date": "2024-01-20",
                "predictor_date": None,
                "feature_lag_days": None,
                "ci_sum": None,
            },
        ]
        validate_aligned_records(records)
        profile = profile_aligned_records(records)

        self.assertEqual(profile["horizons"]["3"]["predictor_missing"], 1)
        self.assertEqual(profile["fields"]["ci_sum"]["missing"], 1)

    def test_predictor_after_horizon_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            validate_aligned_records(
                [
                    {
                        "forecast_horizon_days": "3",
                        "observation_date": "2024-01-10",
                        "predictor_date": "2024-01-08",
                        "feature_lag_days": "2",
                    }
                ]
            )
