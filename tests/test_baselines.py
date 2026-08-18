import unittest

from algal_bloom_forecast.evaluation.metrics import regression_metrics
from algal_bloom_forecast.models.baselines import (
    build_baseline_predictions,
    evaluate_baseline_predictions,
)


class BaselineTests(unittest.TestCase):
    def test_metrics_ignore_missing_predictions(self):
        metrics = regression_metrics([(1.0, 1.0), (2.0, None), (3.0, 5.0)])
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae"], 1.0)

    def test_persistence_and_trend_use_only_history_before_cutoff(self):
        train = [
            {
                "forecast_horizon_days": 3,
                "observation_date": "2023-01-01",
                "ci_sum": 1.0,
                "usgs_maumee_discharge_m3s": 1.0,
                "seasonal_day_of_year_sin": 0.0,
                "seasonal_day_of_year_cos": 1.0,
            }
        ]
        evaluation = [
            {
                "split": "validation",
                "forecast_horizon_days": 3,
                "observation_date": "2023-01-10",
                "ci_sum": 10.0,
                "usgs_maumee_discharge_m3s": 2.0,
                "seasonal_day_of_year_sin": 0.1,
                "seasonal_day_of_year_cos": 0.9,
            }
        ]
        history = [
            {"observation_date": "2023-01-01", "ci_sum": 1.0},
            {"observation_date": "2023-01-05", "ci_sum": 5.0},
            {"observation_date": "2023-01-09", "ci_sum": 99.0},
        ]
        predictions = build_baseline_predictions(train, evaluation, history)
        self.assertEqual(predictions[0]["persistence"], 5.0)
        self.assertEqual(predictions[0]["trend"], 10.0)
        self.assertNotEqual(predictions[0]["persistence"], 99.0)

    def test_metrics_group_by_split_horizon_and_model(self):
        predictions = [
            {
                "split": "test",
                "forecast_horizon_days": 1,
                "actual": 2.0,
                "climatology": 1.0,
                "persistence": 2.0,
                "trend": 3.0,
                "linear": 2.5,
            }
        ]
        results = evaluate_baseline_predictions(predictions)
        persistence = next(row for row in results if row["model"] == "persistence")
        self.assertEqual(persistence["n"], 1)
        self.assertEqual(persistence["mae"], 0.0)
