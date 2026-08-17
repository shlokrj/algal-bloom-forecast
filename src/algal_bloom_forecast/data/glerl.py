"""Profile GLERL/CIGLR CSV observation tables without normalizing source values."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

MISSING_VALUES = {"", "na", "nan", "n/a", "null", "none"}


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _find_column(fieldnames: list[str], *candidates: str) -> str | None:
    normalized = {_normalized_name(name): name for name in fieldnames}
    for candidate in candidates:
        if _normalized_name(candidate) in normalized:
            return normalized[_normalized_name(candidate)]
    return None


def _parse_datetime(value: str) -> datetime | None:
    value = value.strip()
    for format_string in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ):
        try:
            return datetime.strptime(value, format_string)  # noqa: DTZ007
        except ValueError:
            continue
    return None


def _row_datetime(row: dict[str, str | None], fieldnames: list[str]) -> datetime | None:
    datetime_column = _find_column(fieldnames, "datetime")
    if datetime_column:
        return _parse_datetime(row.get(datetime_column) or "")

    timestamp_column = _find_column(fieldnames, "timestamp")
    if timestamp_column:
        return _parse_datetime(row.get(timestamp_column) or "")

    date_column = _find_column(fieldnames, "date")
    time_column = _find_column(fieldnames, "time", "local time (eastern time zone)")
    if date_column and time_column:
        date_value = (row.get(date_column) or "").strip()
        time_value = (row.get(time_column) or "").strip()
        if date_value and time_value:
            return _parse_datetime(f"{date_value} {time_value}")
    return None


def _is_missing(value: str | None) -> bool:
    return (value or "").strip().lower() in MISSING_VALUES


def _time_basis(fieldnames: list[str], source_class: str) -> str:
    if _find_column(fieldnames, "timestamp"):
        return "UTC as documented by the annual-summary timestamp and units rows"
    if _find_column(fieldnames, "local time (eastern time zone)"):
        return "Eastern Time Zone as documented by the source column"
    if _find_column(fieldnames, "datetime"):
        return "Eastern Daylight Time as documented by the fluoroprobe data dictionary"
    if source_class == "moored_buoy_or_continuous":
        return "source time basis not identified from the distributed header"
    return "source local time basis not identified from the distributed header"


def profile_glerl_csv(path: Path, *, source_class: str) -> dict[str, object]:
    """Summarize one GLERL/CIGLR CSV while preserving its source representation."""
    with path.open(newline="", encoding="latin-1") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    timestamps = [
        timestamp for row in rows if (timestamp := _row_datetime(row, fieldnames)) is not None
    ]
    timestamped_rows = [row for row in rows if _row_datetime(row, fieldnames) is not None]
    station_column = _find_column(fieldnames, "station_name", "station", "site", "sample_id")
    depth_category_column = _find_column(fieldnames, "sample_depth_category")
    stations = sorted(
        {
            (row.get(station_column) or "").strip()
            for row in timestamped_rows
            if station_column and (row.get(station_column) or "").strip()
        }
    )
    depth_categories = sorted(
        {
            (row.get(depth_category_column) or "").strip()
            for row in timestamped_rows
            if depth_category_column and (row.get(depth_category_column) or "").strip()
        }
    )
    missing_counts = {
        field: sum(_is_missing(row.get(field)) for row in timestamped_rows)
        for field in fieldnames
        if field
        not in {
            _find_column(fieldnames, "timestamp"),
            _find_column(fieldnames, "datetime"),
            _find_column(fieldnames, "date"),
            _find_column(fieldnames, "time", "local time (eastern time zone)"),
        }
    }
    return {
        "source_kind": source_class,
        "source_filename": path.name,
        "records": len(rows),
        "timestamped_records": len(timestamped_rows),
        "untimestamped_records": len(rows) - len(timestamped_rows),
        "fields": fieldnames,
        "field_count": len(fieldnames),
        "stations": stations,
        "depth_categories": depth_categories,
        "observed_start": min(timestamps).isoformat(sep=" ") if timestamps else None,
        "observed_end": max(timestamps).isoformat(sep=" ") if timestamps else None,
        "time_basis": _time_basis(fieldnames, source_class),
        "missing_counts": dict(sorted(missing_counts.items())),
        "missing_total": sum(missing_counts.values()),
        "missing_fields": sum(count > 0 for count in missing_counts.values()),
        "sample_depth_category_counts": dict(
            Counter(
                (row.get(depth_category_column) or "").strip()
                for row in timestamped_rows
                if depth_category_column and (row.get(depth_category_column) or "").strip()
            )
        ),
    }
