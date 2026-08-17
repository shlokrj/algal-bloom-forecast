"""Decode and summarize NOAA CI-CIcyano raster pixels."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

VALID_DN_MIN = 1
VALID_DN_MAX = 249
DN_SCALE = 3.0 / 250.0
DN_OFFSET = -4.2
FILENAME_PATTERN = re.compile(
    r"^sentinel-3\.(?P<year>\d{4})(?P<day_of_year>\d{3})\."
    r"(?P<month>\d{2})(?P<day>\d{2})\."
    r"(?P<window_start>\d{4})(?:_(?P<window_end>\d{4}))?"
    r"(?P<window_suffix>[A-Za-z])\."
)


@dataclass(frozen=True)
class SatelliteFilenameMetadata:
    """Date and acquisition-window fields encoded in one product filename."""

    observation_date: str
    acquisition_window: str
    timestamp_semantics: str = (
        "filename_calendar_date_only; acquisition_timezone_unconfirmed"
    )


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


def parse_satellite_filename(filename: str) -> SatelliteFilenameMetadata:
    """Validate and parse the date/window prefix of a NOAA satellite filename."""
    match = FILENAME_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"Unrecognized NOAA satellite filename: {filename}")

    fields = match.groupdict()
    year = int(fields["year"])
    day_of_year = int(fields["day_of_year"])
    observation = date(year, 1, 1) + timedelta(days=day_of_year - 1)
    if observation.year != year:
        raise ValueError(f"Invalid day-of-year in NOAA satellite filename: {filename}")
    if (observation.month, observation.day) != (
        int(fields["month"]),
        int(fields["day"]),
    ):
        raise ValueError(f"Filename date fields disagree: {filename}")

    return SatelliteFilenameMetadata(
        observation_date=observation.isoformat(),
        acquisition_window=(
            f"{fields['window_start']}"
            f"{'_' + fields['window_end'] if fields['window_end'] else ''}"
            f"{fields['window_suffix']}"
        ),
    )


def _as_rows(data_numbers: Any) -> list[list[Any]]:
    """Convert a two-dimensional sequence or array to plain Python rows."""
    values = data_numbers.tolist() if hasattr(data_numbers, "tolist") else data_numbers
    if not isinstance(values, (list, tuple)):
        raise TypeError("raster data must be a two-dimensional sequence")
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


def build_daily_target_records(
    raster_paths: list[Path],
) -> list[dict[str, float | int | str | None]]:
    """Summarize rasters into a duplicate-free daily target table."""
    records: list[dict[str, float | int | str | None]] = []
    seen_dates: set[str] = set()
    for raster_path in sorted(raster_paths, key=lambda path: path.name):
        filename_metadata = parse_satellite_filename(raster_path.name)
        if filename_metadata.observation_date in seen_dates:
            raise ValueError(
                "Multiple satellite rasters share the same observation date: "
                f"{filename_metadata.observation_date}"
            )
        seen_dates.add(filename_metadata.observation_date)
        record = {
            "observation_date": filename_metadata.observation_date,
            "acquisition_window": filename_metadata.acquisition_window,
            "timestamp_semantics": filename_metadata.timestamp_semantics,
            "source_filename": raster_path.name,
            **summarize_ci_cyano_raster(raster_path).as_record(),
        }
        records.append(record)
    return sorted(records, key=lambda record: str(record["observation_date"]))
