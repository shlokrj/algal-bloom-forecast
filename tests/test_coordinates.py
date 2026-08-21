from __future__ import annotations

from pathlib import Path

import pytest

from algal_bloom_forecast.data.coordinates import (
    canonical_glerl_station,
    parse_glerl_station_coordinates,
)


def test_canonical_glerl_station_zero_pads_numeric_suffix() -> None:
    assert canonical_glerl_station("WE2") == "WE02"
    assert canonical_glerl_station("we-13") == "WE13"


def test_coordinate_parser_normalizes_and_validates_bbox(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.csv"
    path.write_text("station,lat,long\nWE2,41.762,-83.330\nWE13,41.741,-83.136\n", encoding="utf-8")

    records = parse_glerl_station_coordinates(
        path,
        bbox={"north": 41.834, "south": 41.617, "east": -83.009, "west": -83.424},
    )

    assert records[0]["station"] == "WE02"
    assert records[1]["longitude"] == pytest.approx(-83.136)


def test_coordinate_parser_rejects_duplicate_canonical_stations(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.csv"
    path.write_text("station,lat,long\nWE2,41.762,-83.330\nWE02,41.763,-83.331\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        parse_glerl_station_coordinates(path)
