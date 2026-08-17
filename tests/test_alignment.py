import unittest

from algal_bloom_forecast.features.alignment import align_daily_predictors_to_targets


class AlignmentTests(unittest.TestCase):
    def test_horizon_uses_only_predictors_before_cutoff(self):
        aligned = align_daily_predictors_to_targets(
            [
                {"observation_date": "2024-07-10", "ci_sum": None},
                {"observation_date": "2024-07-12", "ci_sum": 0.5},
            ],
            [
                {"observation_date": "2024-07-09", "discharge": 100},
                {"observation_date": "2024-07-10", "discharge": 110},
                {"observation_date": "2024-07-11", "discharge": 120},
            ],
            horizon_days=1,
        )

        self.assertEqual(aligned[0]["predictor_date"], "2024-07-09")
        self.assertEqual(aligned[0]["feature_lag_days"], 1)
        self.assertEqual(aligned[0]["discharge"], 100)
        self.assertIsNone(aligned[0]["ci_sum"])
        self.assertEqual(aligned[1]["predictor_date"], "2024-07-11")

    def test_no_eligible_predictor_is_explicit(self):
        aligned = align_daily_predictors_to_targets(
            [{"observation_date": "2024-07-01", "ci_sum": 0.1}],
            [{"observation_date": "2024-07-02", "discharge": 100}],
            horizon_days=1,
        )

        self.assertEqual(
            aligned,
            [
                {
                    "observation_date": "2024-07-01",
                    "ci_sum": 0.1,
                    "predictor_date": None,
                    "feature_lag_days": None,
                }
            ],
        )

    def test_duplicate_dates_and_field_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate observation_date"):
            align_daily_predictors_to_targets(
                [{"observation_date": "2024-07-01"}],
                [
                    {"observation_date": "2024-06-30", "x": 1},
                    {"observation_date": "2024-06-30", "x": 2},
                ],
                horizon_days=1,
            )
        with self.assertRaisesRegex(ValueError, "field collision"):
            align_daily_predictors_to_targets(
                [{"observation_date": "2024-07-02", "x": 1}],
                [{"observation_date": "2024-07-01", "x": 2}],
                horizon_days=1,
            )
