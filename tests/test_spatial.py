import unittest

import numpy as np

from algal_bloom_forecast.data.spatial import validate_spatial_arrays


class SpatialValidationTests(unittest.TestCase):
    def test_valid_arrays_match_raw_dn_and_nan_policy(self):
        result = validate_spatial_arrays(
            np.array([[0.01, np.nan], [np.nan, 0.02]], dtype=np.float32),
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
            np.array([[1, 253], [0, 249]], dtype=np.uint8),
        )

        self.assertTrue(result["validation_passed"])
        self.assertTrue(result["mask_matches_raw_dn"])
        self.assertEqual(result["valid_pixel_count"], 2)
        self.assertEqual(result["invalid_intensity_non_nan_count"], 0)

    def test_mask_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_spatial_arrays(
                np.array([[0.01, np.nan]], dtype=np.float32),
                np.array([[0, 0]], dtype=np.uint8),
                np.array([[1, 253]], dtype=np.uint8),
            )

    def test_invalid_pixels_must_be_nan(self):
        with self.assertRaisesRegex(ValueError, "must have NaN"):
            validate_spatial_arrays(
                np.array([[0.01, 0.02]], dtype=np.float32),
                np.array([[1, 0]], dtype=np.uint8),
                np.array([[1, 253]], dtype=np.uint8),
            )

    def test_shapes_must_match(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            validate_spatial_arrays(
                np.zeros((2, 2)),
                np.zeros((2, 1), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
            )
