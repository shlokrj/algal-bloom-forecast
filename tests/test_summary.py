import unittest

from algal_bloom_forecast.evaluation.summary import best_by_metric, build_summary_rows


class SummaryTests(unittest.TestCase):
    def test_summary_normalizes_sources_and_selects_best_metric(self):
        rows = build_summary_rows(
            {
                "baseline": [
                    {
                        "split": "test",
                        "forecast_horizon_days": "1",
                        "model": "linear",
                        "n": "2",
                        "mae": "2.0",
                        "rmse": "3.0",
                    }
                ],
                "feature_ablation": [
                    {
                        "split": "test",
                        "forecast_horizon_days": "1",
                        "feature_group": "water_quality",
                        "model": "linear",
                        "n": "2",
                        "mae": "1.0",
                        "rmse": "2.0",
                    }
                ],
            },
            {
                "event": [
                    {
                        "split": "test",
                        "forecast_horizon_days": "1",
                        "model": "linear",
                        "n": "2",
                        "pr_auc": "0.8",
                        "precision": "0.5",
                        "recall": "0.5",
                        "f1": "0.5",
                        "brier": "0.2",
                        "calibration_abs_error": "0.1",
                    }
                ]
            },
        )
        best = best_by_metric(
            rows,
            metric_family="continuous",
            metric="mae",
            lower_is_better=True,
        )
        self.assertEqual(best["test/1"]["model"], "linear:water_quality")
        self.assertEqual(len(rows), 2 * 2 + 6)
