import unittest

from algal_bloom_forecast.models.dataset import build_training_frame, build_training_schema


class DatasetTests(unittest.TestCase):
    def test_selects_numeric_predictors_and_preserves_audit_ids(self):
        frame = build_training_frame(
            [
                {
                    "split": "train",
                    "forecast_horizon_days": 1,
                    "observation_date": "2020-07-01",
                    "predictor_date": "2020-06-30",
                    "feature_lag_days": 1,
                    "ci_sum": 2.0,
                    "bloom_area_sqkm": 50.0,
                    "signal": 3.0,
                    "source_note": "kept for provenance only",
                }
            ]
        )
        self.assertEqual(frame.feature_names, ("signal",))
        self.assertEqual(frame.rows[0]["target_ci_sum"], 2.0)
        self.assertEqual(frame.rows[0]["observation_date"], "2020-07-01")
        self.assertNotIn("bloom_area_sqkm", frame.rows[0])
        self.assertEqual(build_training_schema(frame)["status"], "prepared_not_fitted")

    def test_rejects_target_missing_and_future_predictor(self):
        base = {
            "split": "train",
            "forecast_horizon_days": 3,
            "observation_date": "2020-07-10",
            "predictor_date": "2020-07-07",
            "ci_sum": 2.0,
            "signal": 3.0,
        }
        with self.assertRaisesRegex(ValueError, "target ci_sum"):
            build_training_frame([{**base, "ci_sum": None}])
        with self.assertRaisesRegex(ValueError, "after cutoff"):
            build_training_frame([{**base, "predictor_date": "2020-07-08"}])
