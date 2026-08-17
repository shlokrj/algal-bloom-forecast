"""Decode and summarize NOAA CI-CIcyano raster pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

VALID_DN_MIN = 1
VALID_DN_MAX = 249
DN_SCALE = 3.0 / 250.0
DN_OFFSET = -4.2


@dataclass(frozen=True)
class SatelliteTargetSummary:
    """Regional summary and observation-quality fields for one raster."""

    mean_intensity: float | None
    valid_pixel_count: int
    total_pixel_count: int
    valid_pixel_fraction: float
    missing_reason: str | None = None

    def as_record(self) -> dict[str, float | int | str | None]:
        """Return a JSON-compatible target record."""
        return asdict(self)


def _as_rows(data_numbers: Any) -> list[list[Any]]:
    """Convert a two-dimensional sequence or array to plain Python rows."""
    values = data_numbers.tolist() if hasattr(data_numbers, "tolist") else data_numbers
    if not isinstance(values, (list, tuple)):
        raise ValueError("raster data must be a two-dimensional sequence")
    if not values:
        return []
    if not all(isinstance(row, (list, tuple)) for row in values):
        raise ValueError("raster data must be a two-dimensional sequence")
    return [list(row) for row in values]


def decode_ci_cyano(data_numbers: Any) -> tuple[list[list[float]], list[list[bool]]]:
    """Decode valid CI-CIcyano data numbers and return values plus a mask.

    DN 0 and DN 250-255 are product flags rather than continuous intensity
    observations. They remain invalid in the returned mask and are represented
    as NaN in the decoded array.
    """
    rows = _as_rows(data_numbers)
    decoded: list[list[float]] = []
    valid: list[list[bool]] = []
    for row in rows:
        decoded_row: list[float] = []
        valid_row: list[bool] = []
        for value in row:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = math.nan
            is_valid = (
                math.isfinite(numeric_value)
                and VALID_DN_MIN <= numeric_value <= VALID_DN_MAX
            )
            valid_row.append(is_valid)
            decoded_row.append(
                10.0 ** (DN_SCALE * numeric_value + DN_OFFSET)
                if is_valid
                else math.nan
            )
        decoded.append(decoded_row)
        valid.append(valid_row)
    return decoded, valid


def summarize_ci_cyano(data_numbers: Any) -> SatelliteTargetSummary:
    """Create the first regional continuous target summary for one raster."""
    decoded, valid = decode_ci_cyano(data_numbers)
    total_pixel_count = sum(len(row) for row in valid)
    valid_values = [
        decoded[row_index][column_index]
        for row_index, row in enumerate(valid)
        for column_index, is_valid in enumerate(row)
        if is_valid
    ]
    valid_pixel_count = len(valid_values)
    valid_pixel_fraction = (
        valid_pixel_count / total_pixel_count if total_pixel_count else 0.0
    )
    if valid_pixel_count == 0:
        return SatelliteTargetSummary(
            mean_intensity=None,
            valid_pixel_count=0,
            total_pixel_count=total_pixel_count,
            valid_pixel_fraction=valid_pixel_fraction,
            missing_reason="no_valid_pixels",
        )

    return SatelliteTargetSummary(
        mean_intensity=sum(valid_values) / valid_pixel_count,
        valid_pixel_count=valid_pixel_count,
        total_pixel_count=total_pixel_count,
        valid_pixel_fraction=valid_pixel_fraction,
    )


def summarize_ci_cyano_raster(path: Path) -> SatelliteTargetSummary:
    """Read and summarize a single-band raster using optional rasterio support."""
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "rasterio is required to read GeoTIFF files; install the data extras"
        ) from error

    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"Expected one raster band, found {dataset.count}")
        return summarize_ci_cyano(dataset.read(1))
