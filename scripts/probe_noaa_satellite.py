#!/usr/bin/env python3
"""List a small, reproducible sample of NOAA western Lake Erie products."""

from __future__ import annotations

import argparse
import json

from algal_bloom_forecast.data.noaa_hab import (
    EXPLORER_ROOT_URL,
    fetch_explorer_listing,
    find_directory_url,
    matching_downloads,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default=r"CI(?:-CI)?cyano.*\.tif$",
        help="regular expression applied to NOAA file names",
    )
    args = parser.parse_args()

    listing_url = find_directory_url(
        EXPLORER_ROOT_URL,
        ["data", "web", "olci_western_le", "tif_archive"],
    )
    entries = matching_downloads(fetch_explorer_listing(listing_url), args.pattern)
    print(
        json.dumps(
            {
                "listing_url": listing_url,
                "match_count": len(entries),
                "matches": [
                    {"filename": entry.label, "url": entry.url, "size": entry.size_label}
                    for entry in entries[:20]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
