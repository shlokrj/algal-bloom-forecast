"""Parser for NOAA NDBC historical standard meteorological files."""

from __future__ import annotations

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
                    f"NDBC row {line_number} has {len(values)} values; "
                    f"expected {len(columns)}"
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
