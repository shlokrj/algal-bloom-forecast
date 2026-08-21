"""Profile GLERL/CIGLR CSV observation tables without normalizing source values."""

from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

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


def profile_glerl_flag_codes(path: Path) -> dict[str, object]:
    """Profile raw GLERL flag sequences without assigning undocumented meanings."""
    with path.open(newline="", encoding="latin-1") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        flag_fields = [field for field in fieldnames if field.lower().endswith("_flags")]
        value_counts = {field: Counter() for field in flag_fields}
        tokens: set[str] = set()
        timestamped_records = 0
        for row in reader:
            if _row_datetime(row, fieldnames) is None:
                continue
            timestamped_records += 1
            for field in flag_fields:
                raw_value = (row.get(field) or "").strip()
                value = "<missing>" if _is_missing(raw_value) else raw_value
                value_counts[field][value] += 1
                if value != "<missing>":
                    tokens.update(raw_value.split())
    return {
        "source_filename": path.name,
        "flag_columns": flag_fields,
        "timestamped_records": timestamped_records,
        "flag_value_counts": {
            field: dict(sorted(counts.items())) for field, counts in sorted(value_counts.items())
        },
        "observed_flag_tokens": sorted(tokens),
        "mapping_status": "raw sequences preserved; code meanings not assigned",
    }


def _feature_name(field: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", field.strip().lower()).strip("_")
    if not normalized or normalized.endswith("_flags"):
        return None
    return normalized


def _parse_numeric(value: str | None) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _station_name(path: Path) -> str:
    match = re.match(r"([A-Za-z0-9]+)_\d{4}_annual_summary\.csv$", path.name)
    if not match:
        raise ValueError(f"GLERL annual-summary filename does not identify a station: {path.name}")
    return match.group(1).lower()


def parse_glerl_continuous_csv(path: Path) -> list[dict[str, Any]]:
    """Parse one GLERL annual-summary logger file into timestamped numeric rows."""
    with path.open(newline="", encoding="latin-1") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        timestamp_column = _find_column(fieldnames, "timestamp")
        if timestamp_column is None:
            raise ValueError(f"GLERL continuous file has no timestamp column: {path}")
        numeric_columns = {
            field: feature
            for field in fieldnames
            if (feature := _feature_name(field)) is not None
            and not feature.endswith("_flags")
            and field != timestamp_column
        }
        station = _station_name(path)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed_timestamp = _parse_datetime(row.get(timestamp_column) or "")
            if parsed_timestamp is None:
                continue
            output: dict[str, Any] = {
                "observation_date": parsed_timestamp.date().isoformat(),
            }
            for field, feature in numeric_columns.items():
                value = _parse_numeric(row.get(field))
                if value is not None:
                    output[f"glerl_{station}_{feature}"] = value
            rows.append(output)
    return rows


def aggregate_glerl_continuous(paths: list[Path]) -> list[dict[str, Any]]:
    """Aggregate GLERL continuous logger rows by UTC calendar day."""
    grouped: dict[str, dict[str, list[float]]] = {}
    stations: dict[str, set[str]] = {}
    record_counts: Counter[str] = Counter()
    for path in paths:
        station = _station_name(path)
        for row in parse_glerl_continuous_csv(path):
            observation_date = row["observation_date"]
            record_counts[observation_date] += 1
            stations.setdefault(observation_date, set()).add(station)
            for field, value in row.items():
                if field == "observation_date":
                    continue
                grouped.setdefault(observation_date, {}).setdefault(field, []).append(float(value))

    daily: list[dict[str, Any]] = []
    for observation_date in sorted(record_counts):
        output: dict[str, Any] = {
            "observation_date": observation_date,
            "glerl_continuous_record_count": record_counts[observation_date],
            "glerl_continuous_station_count": len(stations[observation_date]),
        }
        for field, values in sorted(grouped.get(observation_date, {}).items()):
            output[f"{field}_mean"] = sum(values) / len(values)
            output[f"{field}_valid_count"] = len(values)
        daily.append(output)
    return daily
