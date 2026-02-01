"""Freesound.org scraper — search and download audio previews."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import requests

BASE_URL = "https://freesound.org/apiv2"
DEFAULT_PAGE_SIZE = 15
DEFAULT_FORMAT = "ogg"


def _sanitize_filename(name: str) -> str:
    """Replace non-alphanumeric characters (except hyphens/underscores) with underscores."""
    return re.sub(r"[^\w\-]", "_", name)


def scrape_freesound(
    query: str,
    output_dir: str | Path,
    *,
    api_key: str | None = None,
    num_results: int = 50,
    format: str = DEFAULT_FORMAT,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> list[Path]:
    """Search freesound.org and download audio previews.

    Args:
        query: Search query string.
        output_dir: Directory to save downloaded files.
        api_key: Freesound API key. Falls back to ``FREESOUND_API_KEY`` env var.
        num_results: Maximum number of sounds to download.
        format: Preview format — ``"ogg"`` or ``"mp3"``.
        min_duration: Minimum sound duration in seconds (inclusive).
        max_duration: Maximum sound duration in seconds (inclusive).

    Returns:
        List of paths to downloaded audio files.
    """
    api_key = api_key or os.environ.get("FREESOUND_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Pass api_key= or set FREESOUND_API_KEY env var."
        )

    if format not in ("ogg", "mp3"):
        raise ValueError(f"Unsupported format {format!r}. Use 'ogg' or 'mp3'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_key = f"preview-hq-{format}"

    # Build duration filter
    filter_parts: list[str] = []
    if min_duration is not None or max_duration is not None:
        lo = "*" if min_duration is None else str(min_duration)
        hi = "*" if max_duration is None else str(max_duration)
        filter_parts.append(f"duration:[{lo} TO {hi}]")

    params: dict[str, str | int] = {
        "query": query,
        "token": api_key,
        "fields": "id,name,duration,previews",
        "page_size": min(num_results, DEFAULT_PAGE_SIZE),
    }
    if filter_parts:
        params["filter"] = " ".join(filter_parts)

    saved: list[Path] = []
    url: str | None = f"{BASE_URL}/search/text/"

    while url and len(saved) < num_results:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        for result in data.get("results", []):
            if len(saved) >= num_results:
                break

            sound_id = result["id"]
            name = _sanitize_filename(result["name"])
            preview_url = result.get("previews", {}).get(preview_key)

            if not preview_url:
                warnings.warn(
                    f"No {format} preview for sound {sound_id} ({result['name']}), skipping."
                )
                continue

            dest = output_dir / f"{sound_id}_{name}.{format}"
            try:
                dl = requests.get(preview_url)
                dl.raise_for_status()
                dest.write_bytes(dl.content)
                saved.append(dest)
                print(f"[{len(saved)}/{num_results}] {dest.name}")
            except Exception as exc:
                warnings.warn(f"Failed to download sound {sound_id}: {exc}")

        url = data.get("next")
        # After the first request, params are baked into the 'next' URL
        params = {}

    return saved


def main() -> None:
    """CLI entry point for the freesound scraper."""
    parser = argparse.ArgumentParser(
        description="Search freesound.org and download audio previews."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("output_dir", help="Download destination directory")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Freesound API key (falls back to FREESOUND_API_KEY env var)",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=50,
        help="Maximum number of sounds to download (default: 50)",
    )
    parser.add_argument(
        "--format",
        choices=["ogg", "mp3"],
        default=DEFAULT_FORMAT,
        help=f"Preview format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum sound duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum sound duration in seconds",
    )

    args = parser.parse_args()

    saved = scrape_freesound(
        args.query,
        args.output_dir,
        api_key=args.api_key,
        num_results=args.num_results,
        format=args.format,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    print(f"\nDownloaded {len(saved)} file(s) to {args.output_dir}")
