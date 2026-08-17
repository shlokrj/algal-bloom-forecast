"""Access and profile the NCEI GLERL/CIGLR fluoroprobe subset."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from algal_bloom_forecast.data import ncei

RemoteFile = ncei.RemoteFile
download_remote_file = ncei.download_remote_file
list_ftp_directory = ncei.list_ftp_directory
parse_ftp_listing = ncei.parse_ftp_listing

FLUOROPROBE_DATA_ROOT = (
    "ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0231/0303633/1.1/data/0-data/"
)
FLUOROPROBE_PROFILES_ROOT = urljoin(FLUOROPROBE_DATA_ROOT, "profiles/")
TIME_BASIS = "Eastern Daylight Time as documented by the data dictionary"
NUMERIC_FIELDS = (
    "green_algae",
    "bluegreen",
    "diatoms",
    "cryptophyta",
    "yellow_substances",
    "total_concentration",
    "transmission",
    "depth",
    "temperature",
)


def _parse_source_datetime(value: str) -> datetime:
    """Parse the minute- or second-resolution timestamps used by the archive."""
    return datetime.fromisoformat(value.strip())


def _format_source_datetime(value: datetime) -> str:
    """Keep seconds only when the source provides non-zero seconds."""
    format_string = "%Y-%m-%d %H:%M:%S" if value.second else "%Y-%m-%d %H:%M"
    return value.strftime(format_string)


def parse_fluoroprobe_dictionary(path: Path) -> dict[str, str]:
    """Read the field definitions distributed with the fluoroprobe subset."""
    with path.open(newline="", encoding="latin-1") as handle:
        rows = csv.DictReader(handle)
        return {
            row["measurement_numb"].strip(): row[
                "The number of the measurement collected by the fluoroprobe in acsending order with the downcast."
            ].strip()
            for row in rows
            if row.get("measurement_numb")
        }


def parse_fluoroprobe_coordinates(path: Path) -> list[dict[str, float | str]]:
    """Read station coordinates and discard blank trailing rows."""
    coordinates: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            station = (row.get("station") or "").strip()
            latitude = (row.get("lat") or "").strip()
            longitude = (row.get("long") or "").strip()
            if station and latitude and longitude:
                coordinates.append(
                    {
                        "station": station,
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                    }
                )
    return coordinates


def profile_fluoroprobe_csv(path: Path) -> dict[str, object]:
    """Summarize one depth-profile CSV without changing its source timestamps."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required_fields = {"measurement_number", "datetime", "station", "depth"}
        missing_fields = sorted(required_fields - set(fieldnames))
        if missing_fields:
            raise ValueError(f"Missing fluoroprobe fields: {missing_fields}")
        rows = list(reader)

    timestamps = [_parse_source_datetime(row["datetime"]) for row in rows if row.get("datetime")]
    depths = [float(row["depth"]) for row in rows if row.get("depth")]
    missing_counts = {
        field: sum(not (row.get(field) or "").strip() for row in rows) for field in NUMERIC_FIELDS
    }
    return {
        "source_kind": "fluoroprobe_depth_profile",
        "source_filename": path.name,
        "records": len(rows),
        "fields": fieldnames,
        "stations": sorted({row["station"].strip() for row in rows if row.get("station")}),
        "observed_start": _format_source_datetime(min(timestamps)) if timestamps else None,
        "observed_end": _format_source_datetime(max(timestamps)) if timestamps else None,
        "time_basis": TIME_BASIS,
        "depth_min_m": min(depths) if depths else None,
        "depth_max_m": max(depths) if depths else None,
        "missing_counts": missing_counts,
    }
