import unittest

from algal_bloom_forecast.evaluation.splits import build_temporal_splits


class SplitTests(unittest.TestCase):
    def test_splits_eligible_rows_by_year_and_counts_exclusions(self):
        records = [
            {
                "observation_date": "2021-07-01",
                "forecast_horizon_days": 1,
                "ci_sum": 0.1,
                "predictor_available": 1,
            },
            {
                "observation_date": "2023-07-01",
                "forecast_horizon_days": 1,
                "ci_sum": 0.2,
                "predictor_available": "1",
            },
            {
                "observation_date": "2024-07-01",
                "forecast_horizon_days": 1,
                "ci_sum": 0.3,
                "predictor_available": True,
            },
            {
                "observation_date": "2022-07-01",
                "forecast_horizon_days": 1,
                "ci_sum": None,
                "predictor_available": 1,
            },
            {
                "observation_date": "2022-07-11",
                "forecast_horizon_days": 1,
                "ci_sum": 0.4,
                "predictor_available": 0,
            },
        ]
        included, report = build_temporal_splits(
            records,
            validation_years=(2023,),
            test_years=(2024,),
        )

        self.assertEqual([row["split"] for row in included], ["train", "validation", "test"])
        self.assertEqual(
            report["excluded_by_reason"], {"missing_predictor": 1, "missing_target": 1}
        )
        self.assertEqual(report["split_counts"], {"test": 1, "train": 1, "validation": 1})

    def test_overlapping_split_years_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_temporal_splits(
                [],
                validation_years=(2023,),
                test_years=(2023,),
            )

    def test_duplicate_horizon_date_is_rejected(self):
        record = {
            "observation_date": "2024-07-01",
            "forecast_horizon_days": 1,
            "ci_sum": 0.1,
            "predictor_available": 1,
        }
        with self.assertRaisesRegex(ValueError, "duplicate aligned row"):
            build_temporal_splits(
                [record, record],
                validation_years=(),
                test_years=(2024,),
            )
