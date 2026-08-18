import unittest

from algal_bloom_forecast.models.feature_selection import select_by_train_missing_rate


class FeatureSelectionTests(unittest.TestCase):
    def test_selection_uses_train_missingness_only(self):
        selected, rates = select_by_train_missing_rate(
            [
                {"split": "train", "complete": 1.0, "sparse": None},
                {"split": "train", "complete": 2.0, "sparse": 3.0},
                {"split": "validation", "complete": 1.0, "sparse": None},
            ],
            ("complete", "sparse"),
            max_missing_rate=0.5,
        )

        self.assertEqual(selected, ("complete", "sparse"))
        self.assertEqual(rates["sparse"], 0.5)

    def test_threshold_rejects_all_features(self):
        with self.assertRaisesRegex(ValueError, "selected no features"):
            select_by_train_missing_rate(
                [{"split": "train", "sparse": None}],
                ("sparse",),
                max_missing_rate=0.5,
            )
