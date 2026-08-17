"""Parser for NOAA NDBC historical standard meteorological files."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from gzip import open as gzip_open
from pathlib import Path
from typing import Any, TextIO

NDBC_STD_MET_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"

_MISSING_VALUES = {
    "WDIR": {"999"},
    "WSPD": {"99.0"},
    "GST": {"99.0"},
    "WVHT": {"99.00"},
    "DPD": {"99.00"},
    "APD": {"99.00"},
    "MWD": {"999"},
    "PRES": {"9999.0"},
    "ATMP": {"99.0"},
    "WTMP": {"99.0"},
    "DEWP": {"99.0"},
    "VIS": {"99.0"},
    "TIDE": {"99.00"},
}


def build_standard_meteorology_url(station: str, year: int) -> str:
    """Build the official NDBC historical standard-meteorology URL."""
    return NDBC_STD_MET_URL.format(station=station, year=year)


def _parse_value(column: str, raw_value: str) -> Any:
    if raw_value in _MISSING_VALUES.get(column, set()):
        return None
    if column in {"YY", "MM", "DD", "hh", "mm", "WDIR", "MWD"}:
        return int(raw_value)
    return float(raw_value)


def _read_header(handle: TextIO) -> list[str]:
    columns = handle.readline().strip().lstrip("#").split()
    handle.readline()
    if columns[:5] != ["YY", "MM", "DD", "hh", "mm"]:
        raise ValueError(f"Unexpected NDBC header: {columns}")
    return columns


def parse_standard_meteorology(file_path: Path) -> list[dict[str, Any]]:
    """Parse an NDBC gzip file into UTC-timestamped records."""
    records: list[dict[str, Any]] = []
    with gzip_open(file_path, "rt", encoding="utf-8") as handle:
        columns = _read_header(handle)
        for line_number, line in enumerate(handle, start=3):
            values = line.split()
            if not values:
                continue
            if len(values) != len(columns):
                raise ValueError(
                    f"NDBC row {line_number} has {len(values)} values; expected {len(columns)}"
                )
            parsed = {column: _parse_value(column, value) for column, value in zip(columns, values)}
            timestamp = datetime(
                parsed.pop("YY"),
                parsed.pop("MM"),
                parsed.pop("DD"),
                parsed.pop("hh"),
                parsed.pop("mm"),
                tzinfo=UTC,
            )
            records.append({"timestamp": timestamp.isoformat(), **parsed})
    return records


def _utc_date(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        raise ValueError(f"NDBC timestamp must be timezone-aware: {timestamp!r}")
    return parsed.astimezone(UTC).date().isoformat()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _circular_mean(values: list[float]) -> float | None:
    if not values:
        return None
    radians = [math.radians(value) for value in values]
    sine = sum(math.sin(value) for value in radians)
    cosine = sum(math.cos(value) for value in radians)
    if math.isclose(sine, 0.0, abs_tol=1e-12) and math.isclose(cosine, 0.0, abs_tol=1e-12):
        return None
    result = math.degrees(math.atan2(sine, cosine)) % 360
    return 0.0 if math.isclose(result, 360.0, abs_tol=1e-12) else result


def aggregate_standard_meteorology(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate ten-minute NDBC records into UTC calendar-day features."""
    scalar_fields = (
        "WSPD",
        "GST",
        "WVHT",
        "DPD",
        "APD",
        "PRES",
        "ATMP",
        "WTMP",
        "DEWP",
        "VIS",
        "TIDE",
    )
    direction_fields = ("WDIR", "MWD")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        observation_date = _utc_date(str(record["timestamp"]))
        grouped.setdefault(observation_date, []).append(record)

    daily: list[dict[str, Any]] = []
    for observation_date in sorted(grouped):
        day_records = grouped[observation_date]
        output: dict[str, Any] = {
            "observation_date": observation_date,
            "ndbc_sample_count": len(day_records),
        }
        for field in scalar_fields:
            values = [
                float(record[field]) for record in day_records if record.get(field) is not None
            ]
            prefix = f"ndbc_{field.lower()}"
            output[f"{prefix}_mean"] = _mean(values)
            output[f"{prefix}_valid_count"] = len(values)
        for field in direction_fields:
            values = [
                float(record[field]) for record in day_records if record.get(field) is not None
            ]
            prefix = f"ndbc_{field.lower()}"
            output[f"{prefix}_circular_mean"] = _circular_mean(values)
            output[f"{prefix}_valid_count"] = len(values)
        daily.append(output)
    return daily
