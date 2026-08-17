"""Parse the historical NOAA Great Lakes CIcyano time-series CSVs."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from itertools import pairwise
from pathlib import Path

FUSED_COLUMN = "Western Lake Erie_fused"


def _parse_value(value: str | None) -> float | None:
    cleaned = (value or "").strip()
    return None if cleaned.lower() in {"", "na", "nan", "n/a", "null"} else float(cleaned)


def metric_name(path: Path) -> str:
    """Return the stable metric identifier encoded by a source filename."""
    name = path.name.lower()
    if "bloomarea" in name:
        return "bloom_area_sqkm"
    if "cisum" in name:
        return "ci_sum"
    raise ValueError(f"Unrecognized NOAA CIcyano time-series filename: {path.name}")


def parse_ci_timeseries(path: Path) -> tuple[str, list[dict[str, object | None]], list[str]]:
    """Read one fused Western Lake Erie time series without filling gaps."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Date" not in fieldnames or FUSED_COLUMN not in fieldnames:
            raise ValueError(f"Missing required CIcyano fields in {path.name}")
        rows: list[dict[str, object | None]] = []
        for row in reader:
            date_value = (row.get("Date") or "").strip()
            if not date_value:
                continue
            observation_date = date.fromisoformat(date_value)
            rows.append(
                {
                    "observation_date": observation_date.isoformat(),
                    "value": _parse_value(row.get(FUSED_COLUMN)),
                }
            )
    dates = [str(row["observation_date"]) for row in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError(f"Dates are not sorted and unique in {path.name}")
    return metric_name(path), rows, fieldnames


def profile_ci_timeseries(path: Path) -> dict[str, object]:
    """Summarize one historical CIcyano time series."""
    metric, rows, fieldnames = parse_ci_timeseries(path)
    dates = [date.fromisoformat(str(row["observation_date"])) for row in rows]
    values = [float(row["value"]) for row in rows if row["value"] is not None]
    intervals = Counter((right - left).days for left, right in pairwise(dates))
    return {
        "metric": metric,
        "source_filename": path.name,
        "fields": fieldnames,
        "records": len(rows),
        "missing_fused_records": len(rows) - len(values),
        "observed_start": dates[0].isoformat() if dates else None,
        "observed_end": dates[-1].isoformat() if dates else None,
        "fused_min": min(values) if values else None,
        "fused_max": max(values) if values else None,
        "date_interval_days": dict(sorted(intervals.items())),
        "time_basis": "date-only 10-day composite center date; no timezone",
    }


def merge_ci_timeseries(
    profiles: dict[str, list[dict[str, object | None]]],
) -> list[dict[str, object | None]]:
    """Merge the two historical metrics by date without interpolation."""
    by_date: dict[str, dict[str, object | None]] = {}
    for metric, rows in profiles.items():
        for row in rows:
            record = by_date.setdefault(
                str(row["observation_date"]),
                {"observation_date": row["observation_date"]},
            )
            record[metric] = row["value"]
    return [by_date[key] for key in sorted(by_date)]
