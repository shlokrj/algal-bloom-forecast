"""Coordinate normalization for fixed GLERL station metadata."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


def canonical_glerl_station(value: Any) -> str:
    """Normalize GLERL station labels such as WE2 and WE02 to WE02."""
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"WE[-_ ]?0*(\d+)", text)
    if match is None:
        raise ValueError(f"unrecognized GLERL station label: {value!r}")
    return f"WE{int(match.group(1)):02d}"


def parse_glerl_station_coordinates(
    path: Path,
    *,
    bbox: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Read and validate a GLERL station coordinate table."""
    with path.open(newline="", encoding="latin-1") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_fields = {field.strip().lower(): field for field in fieldnames}
        station_field = normalized_fields.get("station")
        latitude_field = normalized_fields.get("lat") or normalized_fields.get("latitude")
        longitude_field = (
            normalized_fields.get("long")
            or normalized_fields.get("lon")
            or normalized_fields.get("longitude")
        )
        if not station_field or not latitude_field or not longitude_field:
            raise ValueError("coordinate table must contain station, lat, and long/lon fields")

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in reader:
            source_station = (row.get(station_field) or "").strip()
            if not source_station:
                continue
            station = canonical_glerl_station(source_station)
            if station in seen:
                raise ValueError(f"duplicate GLERL station: {station}")
            try:
                latitude = float((row.get(latitude_field) or "").strip())
                longitude = float((row.get(longitude_field) or "").strip())
            except ValueError as error:
                raise ValueError(f"invalid coordinates for GLERL station {station}") from error
            if not math.isfinite(latitude) or not math.isfinite(longitude):
                raise ValueError(f"non-finite coordinates for GLERL station {station}")
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError(f"coordinates out of range for GLERL station {station}")
            if bbox and not (
                bbox["south"] <= latitude <= bbox["north"]
                and bbox["west"] <= longitude <= bbox["east"]
            ):
                raise ValueError(f"coordinates outside configured bbox for GLERL station {station}")
            seen.add(station)
            records.append(
                {
                    "station": station,
                    "source_station": source_station,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    return sorted(records, key=lambda record: str(record["station"]))
