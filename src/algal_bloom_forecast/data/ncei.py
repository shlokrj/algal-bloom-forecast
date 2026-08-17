"""Metadata-only inventory helpers for public NCEI FTP archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urljoin
from urllib.request import urlopen

EntryKind = Literal["file", "directory"]


@dataclass(frozen=True)
class RemoteFile:
    """One file exposed by an NCEI FTP directory listing."""

    name: str
    url: str
    size_bytes: int


@dataclass(frozen=True)
class RemoteEntry:
    """One file or directory exposed by an NCEI FTP directory listing."""

    name: str
    url: str
    kind: EntryKind
    size_bytes: int | None


@dataclass(frozen=True)
class InventoryItem:
    """One item discovered while walking an NCEI FTP tree."""

    path: str
    url: str
    kind: EntryKind
    size_bytes: int | None


def parse_ftp_entries(listing: str, *, base_url: str) -> list[RemoteEntry]:
    """Parse the Unix-style listing returned by the NCEI FTP server."""
    base = base_url if base_url.endswith("/") else f"{base_url}/"
    entries: list[RemoteEntry] = []
    for line in listing.splitlines():
        fields = line.split(maxsplit=8)
        if len(fields) < 9 or fields[0][0] not in {"-", "d"}:
            continue
        kind: EntryKind = "file" if fields[0].startswith("-") else "directory"
        size_bytes: int | None = None
        if kind == "file":
            try:
                size_bytes = int(fields[4])
            except ValueError:
                continue
        name = fields[8]
        entries.append(
            RemoteEntry(
                name=name,
                url=urljoin(base, quote(name)),
                kind=kind,
                size_bytes=size_bytes,
            )
        )
    return entries


def parse_ftp_listing(listing: str, *, base_url: str) -> list[RemoteFile]:
    """Parse only file entries for callers that need downloadable files."""
    return [
        RemoteFile(name=entry.name, url=entry.url, size_bytes=entry.size_bytes or 0)
        for entry in parse_ftp_entries(listing, base_url=base_url)
        if entry.kind == "file"
    ]


def list_ftp_entries(url: str, *, timeout_seconds: int = 60) -> list[RemoteEntry]:
    """List files and directories from a public NCEI FTP directory."""
    with urlopen(url, timeout=timeout_seconds) as response:
        listing = response.read().decode("utf-8", errors="replace")
    return parse_ftp_entries(listing, base_url=url)


def list_ftp_directory(url: str, *, timeout_seconds: int = 60) -> list[RemoteFile]:
    """List files from a public NCEI FTP directory."""
    return [
        RemoteFile(name=entry.name, url=entry.url, size_bytes=entry.size_bytes or 0)
        for entry in list_ftp_entries(url, timeout_seconds=timeout_seconds)
        if entry.kind == "file"
    ]


def inventory_ftp_tree(
    root_url: str,
    *,
    max_depth: int = 2,
    timeout_seconds: int = 60,
) -> list[InventoryItem]:
    """Inventory an FTP tree without downloading any source files."""
    root = root_url if root_url.endswith("/") else f"{root_url}/"
    items: list[InventoryItem] = []

    def walk(url: str, relative_prefix: str, depth: int) -> None:
        for entry in list_ftp_entries(url, timeout_seconds=timeout_seconds):
            relative_path = f"{relative_prefix}{entry.name}"
            items.append(
                InventoryItem(
                    path=relative_path,
                    url=entry.url,
                    kind=entry.kind,
                    size_bytes=entry.size_bytes,
                )
            )
            if entry.kind == "directory" and depth < max_depth:
                walk(entry.url, f"{relative_path}/", depth + 1)

    walk(root, "", 0)
    return items


def download_remote_file(
    remote_file: RemoteFile,
    output_path: Path,
    *,
    timeout_seconds: int = 120,
) -> None:
    """Download one public NCEI file without overwriting local data."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(remote_file.url, timeout=timeout_seconds) as response:
        output_path.write_bytes(response.read())
