#!/usr/bin/env python3
"""Download and profile NOAA's curated annual Western Lake Erie CI reference."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from algal_bloom_forecast.data.manifest import ArtifactManifest
from algal_bloom_forecast.data.ncei import RemoteFile, download_remote_file
from algal_bloom_forecast.data.noaa_ci_reference import (
    CURATED_WLE_ANNUAL_CI_FILENAME,
    CURATED_WLE_ANNUAL_CI_URL,
    profile_curated_wle_annual_ci,
)

ROOT = Path(__file__).resolve().parents[1]


def run() -> Path:
    retrieved_at = datetime.now(UTC)
    run_id = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    raw_path = ROOT / "data/raw/noaa/satellite/curated_references" / CURATED_WLE_ANNUAL_CI_FILENAME
    remote_file = RemoteFile(
        name=CURATED_WLE_ANNUAL_CI_FILENAME,
        url=CURATED_WLE_ANNUAL_CI_URL,
        size_bytes=0,
    )
    if not raw_path.exists():
        download_remote_file(remote_file, raw_path)

    profile = profile_curated_wle_annual_ci(raw_path)
    artifact = ArtifactManifest.from_file(
        source_id="noaa_curated_wle_annual_ci_reference",
        source_url=CURATED_WLE_ANNUAL_CI_URL,
        file_path=raw_path,
        local_path=str(raw_path.relative_to(ROOT)),
        metadata={
            "reference_role": "annual CI calibration and semantics reference",
            "target_interchangeability": profile["target_interchangeability"],
            "ci_unit_status": profile["ci_unit_status"],
        },
        retrieved_at=retrieved_at,
    )
    manifest = {
        "source_id": "noaa_curated_wle_annual_ci_reference",
        "source_url": CURATED_WLE_ANNUAL_CI_URL,
        "retrieved_at": retrieved_at.isoformat(),
        "reference_role": "annual CI calibration and semantics reference",
        "profile": profile,
        "artifact": asdict(artifact),
        "interpretation": {
            "status": "profiled_reference_only",
            "ci_unit_status": "not stated in workbook; historical ci_sum aggregate unit remains unresolved",
            "target_interchangeability": profile["target_interchangeability"],
            "daily_horizon_labels": "not addressed by this annual reference",
        },
    }
    output = ROOT / "data/manifests" / f"algal_bloom_noaa_curated_wle_annual_ci_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {output}")
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "raw_path": str(raw_path)}, indent=2))
    return output


def main() -> None:
    run()


if __name__ == "__main__":
    main()
