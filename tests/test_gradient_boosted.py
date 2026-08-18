import unittest

from algal_bloom_forecast.models.gradient_boosted import (
    build_gradient_boosted_predictions,
    evaluate_gradient_predictions,
)


class GradientBoostedTests(unittest.TestCase):
    def test_model_fits_per_horizon_and_returns_metrics(self):
        train = [
            {
                "split": "train",
                "forecast_horizon_days": 1,
                "observation_date": f"2020-01-{day:02d}",
                "ci_sum": float(day),
                "signal": float(day),
            }
            for day in range(1, 13)
        ]
        evaluation = [
            {
                "split": "test",
                "forecast_horizon_days": 1,
                "observation_date": "2021-01-01",
                "ci_sum": 13.0,
                "signal": 13.0,
            }
        ]
        predictions = build_gradient_boosted_predictions(
            train,
            evaluation,
            feature_names=("signal",),
        )
        self.assertIsNotNone(predictions[0]["gradient_boosted"])
        metrics = evaluate_gradient_predictions(predictions)
        self.assertEqual(metrics[0]["n"], 1)
        self.assertEqual(metrics[0]["model"], "gradient_boosted")
