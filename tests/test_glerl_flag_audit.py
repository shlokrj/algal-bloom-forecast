from __future__ import annotations

import hashlib
import json
from pathlib import Path

from algal_bloom_forecast.data.glerl import profile_glerl_flag_codes
from scripts.validate_glerl_flag_audit import run


def test_validates_audit_against_raw_file_and_source_manifest(tmp_path: Path) -> None:
    raw_path = tmp_path / "WE02_2018_annual_summary.csv"
    raw_path.write_text(
        "timestamp,chlorophylla,chlorophylla_flags\n"
        "UTC,RFU,NAN\n"
        "5/3/2018 15:31,159.8,1 2 1 2\n",
        encoding="latin-1",
    )
    checksum = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    source_manifest_path = tmp_path / "source.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "local_path": str(raw_path),
                        "sha256": checksum,
                        "metadata": {"classification": "moored_buoy_or_continuous"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = profile_glerl_flag_codes(raw_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "source_id": "algal_bloom_glerl_flag_audit",
                "source_manifest": str(source_manifest_path),
                "documented_flag_mapping": {
                    "1": "pass",
                    "2": "not evaluated",
                    "3": "suspect",
                    "4": "failed",
                },
                "mapping_references": {
                    "0190201": "https://www.ncei.noaa.gov/archive/accession/0190201",
                    "0190729": "https://www.ncei.noaa.gov/archive/accession/0190729",
                    "0194301": "https://www.ncei.noaa.gov/archive/accession/0194301",
                    "0194302": "https://www.ncei.noaa.gov/archive/accession/0194302",
                },
                "mapping_scope": (
                    "official NCEI metadata for GLERL annual-summary *_flags fields in the audited "
                    "moored-buoy accessions 0190201, 0190729, 0194301, and 0194302"
                ),
                "observed_flag_tokens": profile["observed_flag_tokens"],
                "unmapped_observed_flag_tokens": profile["unmapped_observed_flag_tokens"],
                "profiles": [{"local_path": str(raw_path), "sha256": checksum, **profile}],
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "validation.json"
    output = run(manifest_path=audit_path, output_path=output_path)
    validation = json.loads(output.read_text(encoding="utf-8"))

    assert validation["validation"]["status"] == "flag_audit_validation_complete"
    assert validation["unmapped_observed_flag_tokens"] == []
