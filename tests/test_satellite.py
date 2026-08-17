import unittest
import math
from pathlib import Path
from unittest.mock import patch

from algal_bloom_forecast.data.satellite import (
    DN_OFFSET,
    DN_SCALE,
    build_daily_target_records,
    decode_ci_cyano,
    parse_satellite_filename,
    summarize_ci_cyano,
)


class SatelliteTargetTests(unittest.TestCase):
    def test_filename_date_and_window_are_validated(self):
        metadata = parse_satellite_filename(
            "sentinel-3.2026229.0817.1536_1614C.ab.L3.LE3.CI-CIcyano.WesternLErie.tif"
        )
        self.assertEqual(metadata.observation_date, "2026-08-17")
        self.assertEqual(metadata.acquisition_window, "1536_1614C")
        self.assertIn("timezone_unconfirmed", metadata.timestamp_semantics)

    def test_filename_date_disagreement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            parse_satellite_filename(
                "sentinel-3.2026229.0816.1536_1614C.ab.L3.LE3.CI-CIcyano.WesternLErie.tif"
            )

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

    @patch("algal_bloom_forecast.data.satellite.summarize_ci_cyano_raster")
    def test_daily_records_preserve_timestamp_boundary(self, summarize):
        summarize.return_value = summarize_ci_cyano([[1, 255]])
        records = build_daily_target_records(
            [
                Path(
                    "sentinel-3.2026229.0817.1536_1614C.ab.L3.LE3.CI-CIcyano.WesternLErie.tif"
                )
            ]
        )
        self.assertEqual(records[0]["observation_date"], "2026-08-17")
        self.assertEqual(
            records[0]["timestamp_semantics"],
            "filename_calendar_date_only; acquisition_timezone_unconfirmed",
        )

    @patch("algal_bloom_forecast.data.satellite.summarize_ci_cyano_raster")
    def test_daily_records_reject_duplicate_dates(self, summarize):
        summarize.return_value = summarize_ci_cyano([[1]])
        paths = [
            Path(
                "sentinel-3.2026229.0817.1536_1614C.ab.L3.LE3.CI-CIcyano.WesternLErie.tif"
            ),
            Path(
                "sentinel-3.2026229.0817.1700_1740C.ab.L3.LE3.CI-CIcyano.WesternLErie.tif"
            ),
        ]
        with self.assertRaisesRegex(ValueError, "same observation date"):
            build_daily_target_records(paths)
