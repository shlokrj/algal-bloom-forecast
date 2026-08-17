#!/usr/bin/env python3
"""Summarize one locally downloaded NOAA CI-CIcyano GeoTIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algal_bloom_forecast.data.satellite import summarize_ci_cyano_raster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raster_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_ci_cyano_raster(args.raster_path).as_record(), indent=2))


if __name__ == "__main__":
    main()
