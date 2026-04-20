"""Read and write the local puzzle bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.group_bank import GROUP_BANK_SOURCE
from agents.bank_memory import build_bank_memory, repeat_errors
from agents.puzzle_validator import GENERATED_SOURCE, normalize_puzzle, puzzle_fingerprint, validate_puzzle


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PUZZLE_PATH = ROOT_DIR / "data" / "puzzles.json"
LATEST_RUN_PATH = ROOT_DIR / "data" / "latest_agent_run.json"
NYT_GUARDED_SOURCES = {GENERATED_SOURCE, GROUP_BANK_SOURCE}


def load_puzzles(path: Path = DEFAULT_PUZZLE_PATH) -> list[dict[str, Any]]:
    """Load puzzles from disk. Missing banks are treated as empty."""

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        puzzles = payload.get("puzzles", [])
    else:
        puzzles = payload

    return [normalize_puzzle(puzzle, source=puzzle.get("source")) for puzzle in puzzles]


def save_puzzles(puzzles: list[dict[str, Any]], path: Path = DEFAULT_PUZZLE_PATH) -> None:
    """Persist a puzzle list in a stable, readable format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_puzzle(puzzle, source=puzzle.get("source")) for puzzle in puzzles]

    with path.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, indent=2)
        file.write("\n")


def add_puzzles(
    new_puzzles: list[dict[str, Any]],
    path: Path = DEFAULT_PUZZLE_PATH,
    allow_duplicates: bool = False,
) -> dict[str, Any]:
    """Validate and append new puzzles to the local bank."""

    existing = load_puzzles(path)
    memory = build_bank_memory(existing)
    seen = {puzzle_fingerprint(puzzle) for puzzle in existing}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for puzzle in new_puzzles:
        normalized = normalize_puzzle(puzzle, source=puzzle.get("source", "claude-multi-agent"))
        source = normalized.get("source")
        require_nyt_blocklist = source in NYT_GUARDED_SOURCES
        validation = validate_puzzle(
            normalized,
            require_nyt_blocklist=require_nyt_blocklist,
            require_generated_metadata=source == GENERATED_SOURCE,
        )
        bank_repeat_errors = repeat_errors(normalized, memory) if source == GENERATED_SOURCE else []
        fingerprint = puzzle_fingerprint(normalized)

        if not validation["ok"] or bank_repeat_errors:
            rejected.append({"puzzle": normalized, "errors": validation["errors"] + bank_repeat_errors})
            continue

        if not allow_duplicates and fingerprint in seen:
            rejected.append({"puzzle": normalized, "errors": ["Duplicate puzzle fingerprint."]})
            continue

        accepted.append(normalized)
        seen.add(fingerprint)
        memory = build_bank_memory(existing + accepted)

    if accepted:
        save_puzzles(existing + accepted, path)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "total": len(existing) + len(accepted),
    }


def save_latest_run(payload: dict[str, Any], path: Path = LATEST_RUN_PATH) -> None:
    """Save the most recent generation trace for the Agent Lab."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_latest_run(path: Path = LATEST_RUN_PATH) -> dict[str, Any]:
    """Load the most recent generation trace."""

    if not path.exists():
        return {"trace": [], "accepted": [], "rejected": []}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
