"""Hash-based guard against accidentally generating past NYT Connections boards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BLOCKLIST_PATH = ROOT_DIR / "data" / "nyt_blocklist.json"


def normalize_guard_word(value: Any) -> str:
    """Normalize words for stable comparison without preserving raw source data."""

    return " ".join(str(value).strip().upper().split())


def stable_hash(prefix: str, parts: list[str]) -> str:
    """Hash normalized puzzle parts with a prefix for future-proofing."""

    raw = prefix + "::" + "||".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def puzzle_words(puzzle: dict[str, Any]) -> list[str]:
    """Return normalized board words from either groups or a flat words field."""

    if isinstance(puzzle.get("words"), list):
        return [normalize_guard_word(word) for word in puzzle["words"]]

    words: list[str] = []
    for group in puzzle.get("groups", []):
        if isinstance(group, dict):
            words.extend(normalize_guard_word(word) for word in group.get("words", []))
        elif isinstance(group, list):
            words.extend(normalize_guard_word(word) for word in group)

    return words


def puzzle_group_word_sets(puzzle: dict[str, Any]) -> list[list[str]]:
    """Return normalized group word sets when grouping data is available."""

    groups: list[list[str]] = []
    for group in puzzle.get("groups", []):
        if isinstance(group, dict):
            words = group.get("words", [])
        elif isinstance(group, list):
            words = group
        else:
            continue

        groups.append(sorted(normalize_guard_word(word) for word in words))

    return groups


def board_signature(puzzle: dict[str, Any]) -> str:
    """Hash the exact 16-word board, ignoring order."""

    return stable_hash("connections-board-v1", sorted(puzzle_words(puzzle)))


def group_set_signature(puzzle: dict[str, Any]) -> str:
    """Hash the exact answer grouping, ignoring group and word order."""

    group_parts = ["|".join(group) for group in puzzle_group_word_sets(puzzle)]
    return stable_hash("connections-group-set-v1", sorted(group_parts))


def group_signatures(puzzle: dict[str, Any]) -> list[str]:
    """Hash individual four-word groups for non-blocking diagnostics."""

    return [stable_hash("connections-group-v1", group) for group in puzzle_group_word_sets(puzzle)]


def load_blocklist(path: Path = DEFAULT_BLOCKLIST_PATH) -> dict[str, Any]:
    """Load the hash-only blocklist. Missing files are treated as empty."""

    if not path.exists():
        return {
            "version": 1,
            "archive_count": 0,
            "blocked_boards": [],
            "blocked_group_sets": [],
            "blocked_groups": [],
        }

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    payload.setdefault("blocked_boards", [])
    payload.setdefault("blocked_group_sets", [])
    payload.setdefault("blocked_groups", [])
    payload.setdefault("archive_count", 0)
    return payload


def blocklist_status(path: Path = DEFAULT_BLOCKLIST_PATH) -> dict[str, Any]:
    """Summarize whether the guard has real archive data loaded."""

    payload = load_blocklist(path)
    board_count = len(payload.get("blocked_boards", []))
    group_set_count = len(payload.get("blocked_group_sets", []))

    return {
        "path": str(path),
        "archive_count": int(payload.get("archive_count", 0) or 0),
        "board_count": board_count,
        "group_set_count": group_set_count,
        "group_count": len(payload.get("blocked_groups", [])),
        "ready": board_count > 0 or group_set_count > 0,
    }


def check_puzzle_against_blocklist(
    puzzle: dict[str, Any],
    *,
    require_ready: bool = False,
    path: Path = DEFAULT_BLOCKLIST_PATH,
) -> dict[str, Any]:
    """Return errors when a puzzle matches a blocked NYT signature."""

    errors: list[str] = []
    warnings: list[str] = []
    payload = load_blocklist(path)
    status = blocklist_status(path)

    if require_ready and not status["ready"]:
        errors.append(
            "NYT guard blocklist is empty. Build data/nyt_blocklist.json before accepting generated puzzles."
        )
        return {"ok": False, "errors": errors, "warnings": warnings, "status": status}

    blocked_boards = set(payload.get("blocked_boards", []))
    blocked_group_sets = set(payload.get("blocked_group_sets", []))
    blocked_groups = set(payload.get("blocked_groups", []))

    board_hash = board_signature(puzzle)
    group_set_hash = group_set_signature(puzzle)

    if board_hash in blocked_boards:
        errors.append("Generated board exactly matches a blocked NYT Connections board signature.")

    if group_set_hash in blocked_group_sets:
        errors.append("Generated answer grouping exactly matches a blocked NYT Connections grouping signature.")

    matching_groups = sorted(set(group_signatures(puzzle)) & blocked_groups)
    if matching_groups:
        warnings.append(
            f"{len(matching_groups)} answer group(s) match NYT archive group signatures; review for originality."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "status": status,
        "signatures": {
            "board": board_hash,
            "group_set": group_set_hash,
        },
    }

