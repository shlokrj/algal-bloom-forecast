import unittest
import math

from algal_bloom_forecast.data.satellite import (
    DN_OFFSET,
    DN_SCALE,
    decode_ci_cyano,
    summarize_ci_cyano,
)


class SatelliteTargetTests(unittest.TestCase):
    def test_valid_data_numbers_are_scaled(self):
        decoded, valid = decode_ci_cyano([[1, 125, 249]])
        expected = [10.0 ** (DN_SCALE * value + DN_OFFSET) for value in [1, 125, 249]]
        self.assertEqual(valid, [[True, True, True]])
        for actual, target in zip(decoded[0], expected):
            self.assertAlmostEqual(actual, target)

    def test_product_flags_are_invalid_and_nan(self):
        decoded, valid = decode_ci_cyano([[0, 250, 251, 252, 253, 254, 255]])
        self.assertEqual(valid, [[False, False, False, False, False, False, False]])
        self.assertTrue(all(math.isnan(value) for value in decoded[0]))

    def test_summary_keeps_coverage_and_mean(self):
        summary = summarize_ci_cyano([[1, 2, 255, 253]])
        expected = sum(10.0 ** (DN_SCALE * value + DN_OFFSET) for value in [1, 2]) / 2
        self.assertAlmostEqual(summary.mean_intensity, expected)
        self.assertEqual(summary.valid_pixel_count, 2)
        self.assertEqual(summary.total_pixel_count, 4)
        self.assertEqual(summary.valid_pixel_fraction, 0.5)
        self.assertIsNone(summary.missing_reason)

    def test_summary_marks_empty_observation_missing(self):
        summary = summarize_ci_cyano([[0, 252, 255]])
        self.assertIsNone(summary.mean_intensity)
        self.assertEqual(summary.valid_pixel_count, 0)
        self.assertEqual(summary.valid_pixel_fraction, 0.0)
        self.assertEqual(summary.missing_reason, "no_valid_pixels")
