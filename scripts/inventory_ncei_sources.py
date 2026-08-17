#!/usr/bin/env python3
"""Inventory historical NCEI source trees without downloading source files."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

from algal_bloom_forecast.data.ncei import InventoryItem, inventory_ftp_tree

ROOT = Path(__file__).resolve().parents[1]
NCEI_ACCESSION_URL = "https://www.ncei.noaa.gov/archive/accession/"

HAB_OFS_ACCESSIONS = (
    ("0172375", "2017", "arc0118", ("1.1",)),
    ("0183933", "2018", "arc0130", ("1.1",)),
    ("0209038", "2019", "arc0152", ("1.1",)),
    ("0239146", "2020", "arc0183", ("1.1",)),
    ("0279234", "2021", "arc0213", ("1.1",)),
    ("0279446", "2022", "arc0213", ("1.1",)),
    ("0288351", "2023", "arc0222", ("1.1",)),
    ("0299571", "2024", "arc0235", ("1.1",)),
)

GLERL_ACCESSIONS = (
    ("0187718", "2012-2018 field sampling", "arc0135", ("1.1", "2.2")),
    ("0190201", "WE02 annual summaries", "arc0140", ("1.1",)),
    ("0190729", "WE04 annual summaries", "arc0140", ("1.1",)),
    ("0194301", "WE08 annual summaries", "arc0142", ("1.1",)),
    ("0194302", "WE13 annual summaries", "arc0142", ("1.1",)),
    ("0209116", "2019 field sampling", "arc0152", ("1.1",)),
    ("0254720", "2020-2021 field sampling", "arc0204", ("1.1",)),
    ("0276355", "2021 transects", "arc0210", ("1.1",)),
    ("0292222", "2022 field sampling", "arc0225", ("1.1",)),
    ("0293514", "2022 transects", "arc0229", ("1.1",)),
    ("0303633", "2017-2022 fluoroprobe profiles", "arc0231", ("1.1",)),
)


def _ftp_root(archive: str, accession: str) -> str:
    return f"ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/{archive}/{accession}/"


def classify_item(collection: str, path: str) -> str:
    """Assign a conservative source class from the distributed path/name."""
    normalized = path.lower()
    segments = set(normalized.split("/"))
    if collection == "noaa_hab_ofs":
        if segments & {"bulletin", "bulletins"} or "bulletin" in normalized:
            return "hab_bulletin"
        if segments & {"image", "imagery"} or "imagery" in normalized:
            return "hab_imagery"
        if "data" in segments:
            return "auxiliary_or_model_data"
        return "archive_metadata"

    if any(
        token in normalized
        for token in ("buoy", "moored", "continuous", "time_series", "annual_summary")
    ):
        return "moored_buoy_or_continuous"
    if any(
        token in normalized
        for token in (
            "field_sampling",
            "annual_summary",
            "transect",
            "fluoroprobe",
            "profiles",
            "results",
        )
    ):
        return "discrete_sampling"
    if any(token in normalized for token in ("dictionary", "coordinates", "iso-19115")):
        return "source_metadata"
    if normalized.endswith((".jpg", ".jpeg", ".png")) or "browse_graphic" in normalized:
        return "browse_graphic"
    return "unclassified"


def _item_record(item: InventoryItem, *, collection: str) -> dict[str, object]:
    return {
        "path": item.path,
        "url": item.url,
        "kind": item.kind,
        "size_bytes": item.size_bytes,
        "classification": classify_item(collection, item.path),
    }


def _summary(items: list[dict[str, object]]) -> dict[str, object]:
    files = [item for item in items if item["kind"] == "file"]
    file_classes = Counter(str(item["classification"]) for item in files)
    suffixes = Counter(Path(str(item["path"])).suffix.lower() or "[none]" for item in files)
    return {
        "item_count": len(items),
        "directory_count": sum(item["kind"] == "directory" for item in items),
        "file_count": len(files),
        "total_file_bytes": sum(int(item["size_bytes"] or 0) for item in files),
        "file_classes": dict(sorted(file_classes.items())),
        "file_suffixes": dict(sorted(suffixes.items())),
        "file_names": [item["path"] for item in files],
    }


def _inventory_accession(
    *,
    collection: str,
    accession: str,
    label: str,
    archive: str,
    versions: tuple[str, ...],
) -> dict[str, object]:
    root_url = _ftp_root(archive, accession)
    version_records: list[dict[str, object]] = []
    for version in versions:
        data_root = urljoin(root_url, f"{version}/data/0-data/")
        raw_items = inventory_ftp_tree(data_root, max_depth=1)
        items = [_item_record(item, collection=collection) for item in raw_items]
        version_records.append(
            {
                "version": version,
                "data_root": data_root,
                "items": items,
                "summary": _summary(items),
            }
        )
    return {
        "collection": collection,
        "accession": accession,
        "label": label,
        "accession_url": f"{NCEI_ACCESSION_URL}{accession}",
        "ftp_root": root_url,
        "versions": version_records,
    }


def run() -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    sources = [
        _inventory_accession(
            collection="noaa_hab_ofs",
            accession=accession,
            label=year,
            archive=archive,
            versions=versions,
        )
        for accession, year, archive, versions in HAB_OFS_ACCESSIONS
    ]
    sources.extend(
        _inventory_accession(
            collection="glerl_ciglr_water_quality",
            accession=accession,
            label=label,
            archive=archive,
            versions=versions,
        )
        for accession, label, archive, versions in GLERL_ACCESSIONS
    )
    report = {
        "source_id": "ncei_historical_source_inventory",
        "retrieved_at": retrieved_at.isoformat(),
        "inventory_scope": "recursive listing of each listed version's data/0-data tree; no source files downloaded",
        "collections": {
            "noaa_hab_ofs": "https://www.ncei.noaa.gov/archive/accession/NOS-HABOFS-LakeErie",
            "glerl_ciglr_water_quality": "https://www.ncei.noaa.gov/archive/accession/GLERL-CIGLR-HAB-LakeErie-water-qual",
        },
        "source_count": len(sources),
        "sources": sources,
    }
    output_path = ROOT / "data/manifests" / f"ncei_source_inventory_{run_id}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manifest: {output_path}")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
