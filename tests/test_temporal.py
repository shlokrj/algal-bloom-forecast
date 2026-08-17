from datetime import datetime, timezone
import unittest

from algal_bloom_forecast.features.temporal import (
    assert_features_available,
    assert_target_after_origin,
    build_target_timestamp,
)


ORIGIN = datetime(2024, 7, 10, 12, tzinfo=timezone.utc)


class TemporalContractTests(unittest.TestCase):
    def test_features_at_or_before_origin_are_allowed(self):
        assert_features_available(
            [datetime(2024, 7, 10, 11, tzinfo=timezone.utc), ORIGIN],
            ORIGIN,
        )

    def test_future_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after forecast origin"):
            assert_features_available(
                [datetime(2024, 7, 10, 13, tzinfo=timezone.utc)],
                ORIGIN,
            )

    def test_naive_feature_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            assert_features_available([datetime(2024, 7, 10, 11)], ORIGIN)

    def test_target_timestamp_is_strictly_future(self):
        target = build_target_timestamp(ORIGIN, 7)
        self.assertEqual(target, datetime(2024, 7, 17, 12, tzinfo=timezone.utc))
        assert_target_after_origin(target, ORIGIN)

    def test_non_future_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be after forecast origin"):
            assert_target_after_origin(ORIGIN, ORIGIN)

    def test_non_positive_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_target_timestamp(ORIGIN, 0)
