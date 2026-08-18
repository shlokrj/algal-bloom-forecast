import unittest

from algal_bloom_forecast.evaluation.events import (
    event_metrics,
    fit_event_thresholds,
)


class EventTests(unittest.TestCase):
    def test_thresholds_are_fit_from_training_rows_by_horizon(self):
        thresholds = fit_event_thresholds(
            [
                {"forecast_horizon_days": 1, "ci_sum": 1.0},
                {"forecast_horizon_days": 1, "ci_sum": 2.0},
                {"forecast_horizon_days": 1, "ci_sum": 3.0},
                {"forecast_horizon_days": 1, "ci_sum": 4.0},
                {"forecast_horizon_days": 3, "ci_sum": 10.0},
            ],
            quantile=0.5,
        )
        self.assertEqual(thresholds[1], 2.5)
        self.assertEqual(thresholds[3], 10.0)

    def test_event_metrics_report_thresholded_and_rank_metrics(self):
        metrics = event_metrics(
            [(0.0, 0.1), (1.0, 0.8), (1.0, 0.2)],
            event_threshold=0.5,
        )
        self.assertEqual(metrics["positive_events"], 2)
        self.assertEqual(metrics["predicted_events"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 1 / 3)

    def test_quantile_must_be_strictly_between_zero_and_one(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            fit_event_thresholds(
                [{"forecast_horizon_days": 1, "ci_sum": 1.0}],
                quantile=1.0,
            )
