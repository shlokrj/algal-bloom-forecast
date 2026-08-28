import tempfile
import unittest
from pathlib import Path

from algal_bloom_forecast.data.glerl import profile_glerl_csv, profile_glerl_flag_codes
from scripts.profile_glerl_observations import _selected_files


class GlerlProfileTests(unittest.TestCase):
    def test_selection_uses_one_fluoroprobe_profile_per_station_year(self):
        inventory = {
            "sources": [
                {
                    "collection": "glerl_ciglr_water_quality",
                    "accession": "0303633",
                    "label": "fluoroprobe",
                    "versions": [
                        {
                            "version": "1.1",
                            "items": [
                                {
                                    "kind": "file",
                                    "path": "profiles/noaa-glerl-fluoroprobe-WE12-20220711.csv",
                                    "classification": "discrete_sampling",
                                    "size_bytes": 1,
                                    "url": "ftp://example.test/early.csv",
                                },
                                {
                                    "kind": "file",
                                    "path": "profiles/noaa-glerl-fluoroprobe-WE12-20220817.csv",
                                    "classification": "discrete_sampling",
                                    "size_bytes": 1,
                                    "url": "ftp://example.test/late.csv",
                                },
                                {
                                    "kind": "file",
                                    "path": "profiles/noaa-glerl-fluoroprobe-WE9-20220817.csv",
                                    "classification": "discrete_sampling",
                                    "size_bytes": 1,
                                    "url": "ftp://example.test/other.csv",
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        selected = _selected_files(inventory)

        self.assertEqual(
            [item["path"] for item in selected],
            [
                "profiles/noaa-glerl-fluoroprobe-WE12-20220711.csv",
                "profiles/noaa-glerl-fluoroprobe-WE9-20220817.csv",
            ],
        )

    def test_profiles_discrete_field_table_and_preserves_local_time_basis(self):
        content = """\
Date,Site,Sample Depth (m),Sample Depth (category),Local Time (Eastern Time Zone),Extracted Chlorophyll a (µg/L)
5/15/2012,WE2,0.75,Surface,10:40,3.67
5/15/2012,WE2,5.1,Bottom,,
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.csv"
            path.write_text(content, encoding="latin-1")
            profile = profile_glerl_csv(path, source_class="discrete_sampling")

        self.assertEqual(profile["records"], 2)
        self.assertEqual(profile["timestamped_records"], 1)
        self.assertEqual(profile["stations"], ["WE2"])
        self.assertEqual(profile["observed_start"], "2012-05-15 10:40:00")
        self.assertIn("Eastern Time Zone", profile["time_basis"])

    def test_skips_annual_summary_units_and_instrument_rows(self):
        content = """\
timestamp,chlorophylla,chlorophylla_flags
UTC,RFU,NAN
data logger,YSI EXO2,NAN
5/3/2018 15:31,159.8,1 1 1 1
5/3/2018 15:46,NAN,1 1 1 1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annual.csv"
            path.write_text(content, encoding="utf-8")
            profile = profile_glerl_csv(path, source_class="moored_buoy_or_continuous")

        self.assertEqual(profile["records"], 4)
        self.assertEqual(profile["timestamped_records"], 2)
        self.assertEqual(profile["untimestamped_records"], 2)
        self.assertEqual(profile["observed_start"], "2018-05-03 15:31:00")
        self.assertIn("UTC", profile["time_basis"])
        self.assertEqual(profile["missing_counts"]["chlorophylla"], 1)

    def test_profiles_flag_sequences_and_documents_known_meanings(self):
        content = """\
timestamp,chlorophylla,chlorophylla_flags
UTC,RFU,NAN
data logger,YSI EXO2,NAN
5/3/2018 15:31,159.8,1 1 1 1
5/3/2018 15:46,NAN,1 NA 1 2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annual.csv"
            path.write_text(content, encoding="latin-1")
            profile = profile_glerl_flag_codes(path)

        assert profile["timestamped_records"] == 2
        assert profile["observed_flag_tokens"] == ["1", "2", "NA"]
        assert profile["documented_flag_mapping"] == {
            "1": "pass",
            "2": "not evaluated",
            "3": "suspect",
            "4": "failed",
        }
        assert profile["mapped_observed_flag_tokens"] == ["1", "2"]
        assert profile["unmapped_observed_flag_tokens"] == ["NA"]
        assert "documented subset mapped" in profile["mapping_status"]
        assert profile["flag_value_counts"]["chlorophylla_flags"]["1 NA 1 2"] == 1
