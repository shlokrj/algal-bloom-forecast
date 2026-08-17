import unittest
from unittest.mock import patch

from algal_bloom_forecast.data.noaa_hab import (
    find_directory_url,
    matching_downloads,
    parse_explorer_listing,
)


LISTING_HTML = """
<a href="/habs_explorer/index.php?path=data">data</a>
<div class="row">
  <section class="onecol"><a href="download-one" title="Download"><img src="download.png"></a> sample.CIcyano.tif</section>
  <section class="fourcol last">621.6 KB</section>
</div>
<div class="row">
  <section class="onecol"><a href="download-two" title="Download"><img src="download.png"></a> other.txt</section>
  <section class="fourcol last">12 KB</section>
</div>
"""


class NoaaHabTests(unittest.TestCase):
    def test_parser_extracts_files_and_directory_links(self):
        entries = parse_explorer_listing(
            LISTING_HTML,
            base_url="https://example.test/index.php",
        )
        self.assertEqual(entries[0].label, "data")
        downloads = matching_downloads(entries, r"CIcyano\.tif$")
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0].url, "https://example.test/download-one")
        self.assertEqual(downloads[0].size_label, "621.6 KB")

    def test_directory_traversal_reports_missing_directory(self):
        with patch(
            "algal_bloom_forecast.data.noaa_hab.fetch_explorer_listing",
            return_value=[],
        ), self.assertRaises(LookupError):
            find_directory_url("https://example.test/index.php", ["missing"])
