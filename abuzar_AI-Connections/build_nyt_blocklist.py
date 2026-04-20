"""Build a hash-only blocklist from past NYT Connections puzzle data.

Input should be JSON. The script accepts either:

- a list of puzzle objects
- an object with a "puzzles" list

Each puzzle may include either:

- "groups": [{"words": ["A", "B", "C", "D"]}, ...]
- "groups": [["A", "B", "C", "D"], ...]
- "words": ["A", "B", ... sixteen total words]

The output stores hashes only, not raw puzzle answers.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.nyt_guard import (
    DEFAULT_BLOCKLIST_PATH,
    board_signature,
    group_set_signature,
    group_signatures,
    load_blocklist,
    puzzle_group_word_sets,
    puzzle_words,
)


GROUP_LINE_PATTERN = re.compile(r"^(Yellow|Green|Blue|Purple) Group:\s*(.+)$", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def load_puzzle_entries(path: Path) -> list[dict[str, Any]]:
    """Load puzzle entries from JSON or archive-style text."""

    if path.suffix.lower() in {".txt", ".html", ".htm"}:
        return load_text_archive_entries(path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        entries = payload.get("puzzles", [])
    else:
        entries = payload

    if not isinstance(entries, list):
        raise ValueError("Input must be a JSON list, or an object with a puzzles list.")

    return [entry for entry in entries if isinstance(entry, dict)]


def load_text_archive_entries(path: Path) -> list[dict[str, Any]]:
    """Parse text containing repeated Yellow/Green/Blue/Purple group lines."""

    entries: list[dict[str, Any]] = []
    current_groups: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            without_tags = HTML_TAG_PATTERN.sub("", raw_line)
            line = " ".join(html.unescape(without_tags).strip().split())
            match = GROUP_LINE_PATTERN.match(line)
            if not match:
                continue

            color = match.group(1).lower()
            payload = match.group(2)
            if ":" not in payload:
                continue

            category, words_text = payload.rsplit(":", 1)
            words = [" ".join(word.strip().split()) for word in words_text.split(",")]
            words = [word for word in words if word]

            current_groups.append({"color": color, "category": category.strip(), "words": words})

            if len(current_groups) == 4:
                entries.append({"groups": current_groups})
                current_groups = []

    return entries


def valid_board(entry: dict[str, Any]) -> bool:
    """Check whether an entry has enough information for an exact board hash."""

    words = [word for word in puzzle_words(entry) if word]
    return len(words) == 16 and len(set(words)) == 16


def valid_group_set(entry: dict[str, Any]) -> bool:
    """Check whether an entry has four valid answer groups."""

    groups = puzzle_group_word_sets(entry)
    return len(groups) == 4 and all(len(group) == 4 for group in groups)


def build_blocklist(entries: list[dict[str, Any]], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build or extend a hash-only blocklist."""

    blocked_boards = set((existing or {}).get("blocked_boards", []))
    blocked_group_sets = set((existing or {}).get("blocked_group_sets", []))
    blocked_groups = set((existing or {}).get("blocked_groups", []))
    accepted = 0
    skipped = 0

    for entry in entries:
        added = False

        if valid_board(entry):
            blocked_boards.add(board_signature(entry))
            added = True

        if valid_group_set(entry):
            blocked_group_sets.add(group_set_signature(entry))
            blocked_groups.update(group_signatures(entry))
            added = True

        if added:
            accepted += 1
        else:
            skipped += 1

    existing_archive_count = int((existing or {}).get("archive_count", 0) or 0)

    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_note": (
            "Hash-only NYT Connections guard. Raw NYT answers are intentionally not stored here."
        ),
        "archive_count": max(existing_archive_count, accepted),
        "blocked_boards": sorted(blocked_boards),
        "blocked_group_sets": sorted(blocked_group_sets),
        "blocked_groups": sorted(blocked_groups),
        "build_stats": {
            "accepted_from_input": accepted,
            "skipped_from_input": skipped,
            "total_board_hashes": len(blocked_boards),
            "total_group_set_hashes": len(blocked_group_sets),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hash-only NYT Connections blocklist.")
    parser.add_argument("--input", required=True, type=Path, help="JSON file containing past puzzle data.")
    parser.add_argument("--output", default=DEFAULT_BLOCKLIST_PATH, type=Path, help="Output blocklist path.")
    parser.add_argument("--merge", action="store_true", help="Merge with the existing output blocklist.")
    args = parser.parse_args()

    entries = load_puzzle_entries(args.input)
    existing = load_blocklist(args.output) if args.merge else None
    blocklist = build_blocklist(entries, existing=existing)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(blocklist, file, indent=2)
        file.write("\n")

    stats = blocklist["build_stats"]
    print(f"Accepted input puzzles: {stats['accepted_from_input']}")
    print(f"Skipped input entries: {stats['skipped_from_input']}")
    print(f"Board hashes: {stats['total_board_hashes']}")
    print(f"Group-set hashes: {stats['total_group_set_hashes']}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
