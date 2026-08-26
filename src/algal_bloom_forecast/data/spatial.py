"""Validation helpers for masked spatial satellite targets."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_spatial_arrays(
    intensity: Any,
    valid_mask: Any,
    raw_dn: Any,
    *,
    valid_dn_range: tuple[int, int] = (1, 249),
) -> dict[str, Any]:
    """Validate the raw-DN, decoded-intensity, and valid-mask contract.

    Valid pixels are exactly raw digital numbers in ``valid_dn_range``. Their
    decoded intensity must be finite; every invalid pixel must retain a NaN
    intensity. The returned counts are JSON-compatible so callers can persist
    them in an immutable quality manifest.
    """
    if len(valid_dn_range) != 2 or valid_dn_range[0] > valid_dn_range[1]:
        raise ValueError("valid_dn_range must contain an ordered lower and upper bound")

    intensity_array = np.asarray(intensity)
    valid_mask_array = np.asarray(valid_mask)
    raw_dn_array = np.asarray(raw_dn)
    shapes = {tuple(array.shape) for array in (intensity_array, valid_mask_array, raw_dn_array)}
    if len(shapes) != 1:
        raise ValueError(
            "intensity, valid_mask, and raw_dn must have identical shapes: "
            f"{tuple(intensity_array.shape)}, {tuple(valid_mask_array.shape)}, "
            f"{tuple(raw_dn_array.shape)}"
        )
    if intensity_array.ndim != 2:
        raise ValueError(f"spatial arrays must be two-dimensional, got {intensity_array.ndim}")

    unique_mask_values = sorted(
        {int(value) for value in np.unique(valid_mask_array).tolist()}
    )
    if not set(unique_mask_values).issubset({0, 1}):
        raise ValueError(f"valid_mask must contain only 0 and 1, got {unique_mask_values}")

    lower_bound, upper_bound = valid_dn_range
    raw_numeric = raw_dn_array.astype(np.float64, copy=False)
    expected_valid = (
        np.isfinite(raw_numeric)
        & (raw_numeric >= lower_bound)
        & (raw_numeric <= upper_bound)
    )
    mask_boolean = valid_mask_array.astype(bool, copy=False)
    mask_matches_raw_dn = bool(np.array_equal(mask_boolean, expected_valid))
    if not mask_matches_raw_dn:
        raise ValueError("valid_mask does not match the valid raw DN range")

    intensity_numeric = intensity_array.astype(np.float64, copy=False)
    finite_intensity = np.isfinite(intensity_numeric)
    valid_intensity_nonfinite_count = int(np.count_nonzero(mask_boolean & ~finite_intensity))
    invalid_intensity_non_nan_count = int(
        np.count_nonzero(~mask_boolean & ~np.isnan(intensity_numeric))
    )
    if valid_intensity_nonfinite_count:
        raise ValueError("valid pixels must have finite decoded intensity")
    if invalid_intensity_non_nan_count:
        raise ValueError("invalid pixels must have NaN decoded intensity")

    total_pixel_count = int(mask_boolean.size)
    valid_pixel_count = int(mask_boolean.sum())
    return {
        "shape": [int(dimension) for dimension in intensity_array.shape],
        "total_pixel_count": total_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "invalid_pixel_count": total_pixel_count - valid_pixel_count,
        "valid_pixel_fraction": (
            valid_pixel_count / total_pixel_count if total_pixel_count else 0.0
        ),
        "valid_mask_unique_values": unique_mask_values,
        "mask_matches_raw_dn": mask_matches_raw_dn,
        "valid_intensity_nonfinite_count": valid_intensity_nonfinite_count,
        "invalid_intensity_non_nan_count": invalid_intensity_non_nan_count,
        "finite_intensity_count": int(finite_intensity.sum()),
        "nan_intensity_count": int(np.isnan(intensity_numeric).sum()),
        "validation_passed": True,
    }
