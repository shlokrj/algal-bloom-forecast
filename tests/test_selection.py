import unittest

from algal_bloom_forecast.evaluation.selection import select_by_validation_mae


class SelectionTests(unittest.TestCase):
    def test_selection_uses_validation_only_and_selects_per_horizon(self):
        selected = select_by_validation_mae(
            [
                {"split": "validation", "forecast_horizon_days": 1, "candidate": "persistence", "mae": 1.0},
                {"split": "validation", "forecast_horizon_days": 1, "candidate": "tree", "mae": 2.0},
                {"split": "test", "forecast_horizon_days": 1, "candidate": "tree", "mae": 0.1},
                {"split": "validation", "forecast_horizon_days": 3, "candidate": "tree", "mae": 0.5},
            ]
        )

        self.assertEqual(selected, {1: "persistence", 3: "tree"})
