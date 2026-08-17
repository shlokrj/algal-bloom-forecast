"""Small client for the NOAA HAB Explorer directory listings."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Pattern
from urllib.parse import urljoin
from urllib.request import Request, urlopen


EXPLORER_ROOT_URL = "https://app.coastalscience.noaa.gov/habs_explorer/index.php"
USER_AGENT = "algal-bloom-forecast/0.1"


@dataclass(frozen=True)
class ExplorerEntry:
    """One file or directory link exposed by the explorer."""

    label: str
    url: str
    is_download: bool = False
    size_label: str | None = None


class _ExplorerParser(HTMLParser):
    """Extract directory links and download rows from one explorer page."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[ExplorerEntry] = []
        self._anchor: dict[str, str] | None = None
        self._anchor_text: list[str] = []
        self._row_download_url: str | None = None
        self._row_label: list[str] = []
        self._row_size: list[str] = []
        self._row_section = 0
        self._in_anchor = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a":
            self._anchor = {
                key: value or "" for key, value in attributes.items()
            }
            self._anchor_text = []
            self._in_anchor = True
        elif tag == "section" and self._row_download_url is not None:
            self._row_section += 1

    def handle_data(self, data: str) -> None:
        if self._anchor is not None and self._in_anchor:
            self._anchor_text.append(data)
            return
        if self._row_download_url is None:
            return
        if self._row_section == 1:
            self._row_label.append(data)
        elif self._row_section == 2:
            self._row_size.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None:
            attributes = self._anchor
            label = " ".join("".join(self._anchor_text).split())
            href = attributes.get("href", "")
            if href and attributes.get("title") == "Download":
                self._row_download_url = href
                self._row_section = 1
                self._row_label = []
                self._row_size = []
            elif href and label:
                self.entries.append(ExplorerEntry(label=label, url=href))
            self._anchor = None
            self._anchor_text = []
            self._in_anchor = False
        elif tag == "div" and self._row_download_url is not None:
            label = " ".join("".join(self._row_label).split())
            size_label = " ".join("".join(self._row_size).split()) or None
            if label:
                self.entries.append(
                    ExplorerEntry(
                        label=label,
                        url=self._row_download_url,
                        is_download=True,
                        size_label=size_label,
                    )
                )
            self._row_download_url = None
            self._row_label = []
            self._row_size = []
            self._row_section = 0


def parse_explorer_listing(html: str, *, base_url: str) -> list[ExplorerEntry]:
    """Parse directory and download entries from explorer HTML."""
    parser = _ExplorerParser()
    parser.feed(html)
    return [
        ExplorerEntry(
            label=entry.label,
            url=urljoin(base_url, entry.url),
            is_download=entry.is_download,
            size_label=entry.size_label,
        )
        for entry in parser.entries
    ]


def fetch_explorer_listing(url: str, *, timeout_seconds: int = 60) -> list[ExplorerEntry]:
    """Fetch and parse one live NOAA explorer directory page."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        html = response.read().decode("utf-8", errors="replace")
    return parse_explorer_listing(html, base_url=url)


def find_directory_url(
    root_url: str,
    directory_names: list[str],
    *,
    timeout_seconds: int = 60,
) -> str:
    """Follow explorer directory labels and return the final encoded URL."""
    current_url = root_url
    for directory_name in directory_names:
        entries = fetch_explorer_listing(current_url, timeout_seconds=timeout_seconds)
        matches = [entry for entry in entries if entry.label == directory_name and not entry.is_download]
        if len(matches) != 1:
            raise LookupError(
                f"Expected one NOAA explorer directory named {directory_name!r}, found {len(matches)}"
            )
        current_url = matches[0].url
    return current_url


def matching_downloads(
    entries: list[ExplorerEntry],
    pattern: str | Pattern[str],
) -> list[ExplorerEntry]:
    """Return download entries whose file names match a regular expression."""
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    return [entry for entry in entries if entry.is_download and compiled.search(entry.label)]


def download_entry(
    entry: ExplorerEntry,
    output_path: Path,
    *,
    timeout_seconds: int = 120,
) -> None:
    """Download one explorer file without overwriting an existing artifact."""
    if not entry.is_download:
        raise ValueError("Only downloadable explorer entries can be downloaded")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")

    request = Request(entry.url, headers={"User-Agent": USER_AGENT})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=timeout_seconds) as response:
        output_path.write_bytes(response.read())
