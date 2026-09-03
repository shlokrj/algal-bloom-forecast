from __future__ import annotations

from scripts.validate_glerl_annual_summary_scope import AUDITED_ACCESSIONS, inspect_inventory


def test_inspect_inventory_finds_only_audited_annual_summary_accessions() -> None:
    inventory = {
        "sources": [
            {
                "accession": accession,
                "versions": [
                    {
                        "items": [
                            {
                                "path": f"WE_{accession}_annual_summary.csv",
                                "classification": "moored_buoy_or_continuous",
                                "url": "https://example.test/file.csv",
                            }
                        ]
                    }
                ],
            }
            for accession in AUDITED_ACCESSIONS
        ]
    }

    scope = inspect_inventory(inventory)

    assert scope["annual_summary_accessions"] == list(AUDITED_ACCESSIONS)
    assert scope["annual_summary_file_count"] == 4
    assert scope["outside_audited_scope"] == []


def test_inspect_inventory_surfaces_unreviewed_accessions() -> None:
    inventory = {
        "sources": [
            {
                "accession": "9999999",
                "versions": [
                    {
                        "items": [
                            {
                                "path": "WE99_2025_annual_summary.csv",
                                "classification": "moored_buoy_or_continuous",
                            }
                        ]
                    }
                ],
            }
        ]
    }

    scope = inspect_inventory(inventory)

    assert scope["annual_summary_accessions"] == ["9999999"]
    assert len(scope["outside_audited_scope"]) == 1
