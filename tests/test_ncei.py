import unittest

from algal_bloom_forecast.data.ncei import parse_ftp_entries, parse_ftp_listing
from scripts.inventory_ncei_sources import classify_item


class NceiTests(unittest.TestCase):
    def test_ftp_entries_preserve_files_and_directories(self):
        listing = (
            "-rw-rw-r--    1 0 0 123 Jan 01 2025 sample.csv\n"
            "drwxrwxr-x    2 0 0 4096 Jan 01 2025 profiles\n"
        )
        entries = parse_ftp_entries(listing, base_url="ftp://example.test/data/")

        self.assertEqual([entry.kind for entry in entries], ["file", "directory"])
        self.assertEqual(entries[0].url, "ftp://example.test/data/sample.csv")
        self.assertEqual(entries[1].url, "ftp://example.test/data/profiles")
        self.assertEqual(entries[1].size_bytes, None)

    def test_file_listing_compatibility_filters_directories(self):
        listing = (
            "drwxrwxr-x    2 0 0 4096 Jan 01 2025 profiles\n"
            "-rw-rw-r--    1 0 0 123 Jan 01 2025 sample.csv\n"
        )

        files = parse_ftp_listing(listing, base_url="ftp://example.test/data/")

        self.assertEqual([(file.name, file.size_bytes) for file in files], [("sample.csv", 123)])

    def test_ftp_urls_quote_spaces_in_source_names(self):
        listing = "-rw-rw-r--    1 0 0 123 Jan 01 2025 Western Lake Erie.csv\n"

        files = parse_ftp_listing(listing, base_url="ftp://example.test/data/")

        self.assertEqual(files[0].url, "ftp://example.test/data/Western%20Lake%20Erie.csv")

    def test_classification_separates_discrete_and_moored_names(self):
        self.assertEqual(
            classify_item(
                "glerl_ciglr_water_quality",
                "profiles/noaa-glerl-fluoroprobe-WE12-20220817.csv",
            ),
            "discrete_sampling",
        )
        self.assertEqual(
            classify_item(
                "glerl_ciglr_water_quality",
                "moored_buoy/continuous_timeseries.csv",
            ),
            "moored_buoy_or_continuous",
        )
        self.assertEqual(
            classify_item(
                "glerl_ciglr_water_quality",
                "WE02_2018_annual_summary.csv",
            ),
            "moored_buoy_or_continuous",
        )
        self.assertEqual(
            classify_item("noaa_hab_ofs", "bulletins/2024-Aug-01_bulletin.pdf"),
            "hab_bulletin",
        )
        self.assertEqual(
            classify_item(
                "glerl_ciglr_water_quality",
                "lake_erie_habs_fluoroprobe_data_dictionary_2017-2022.csv",
            ),
            "source_metadata",
        )
        self.assertEqual(
            classify_item(
                "glerl_ciglr_water_quality",
                "lake_erie_habs_browse_graphic_transect_2022.jpg",
            ),
            "browse_graphic",
        )
