"""Verified group bank and no-API puzzle assembly."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from agents.bank_memory import concept_identity, group_key, infer_mechanism_family
from agents.puzzle_validator import (
    DRAFT_EXPLANATION_PATTERN,
    VALID_DIFFICULTIES,
    WORD_PATTERN,
    generated_constraint_errors,
    hidden_claim_errors,
    hidden_substring_mechanism_errors,
    homophone_claim_errors,
    is_phrase_group,
    normalize_category,
    normalize_metadata_key,
    normalize_puzzle,
    normalize_word,
    puzzle_fingerprint,
    validate_puzzle,
    wordplay_leakage_errors,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_BANK_PATH = ROOT_DIR / "data" / "groups.json"
GROUP_BANK_SOURCE = "group-bank-assembler"
GROUP_GENERATOR_SOURCE = "claude-group-agent"
GROUP_BANK_STRATEGIES = {"group-bank", "group_bank", "assembler", "assembled"}
LANES = ("easy", "medium", "hard", "tricky")


def group_fingerprint(group: dict[str, Any]) -> str:
    """Create a stable id for a verified answer group."""

    raw = "|".join(
        [
            normalize_category(group.get("category", "")).casefold(),
            *sorted(normalize_word(word) for word in group.get("words", [])),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def normalize_group(group: dict[str, Any], *, origin: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize one reusable group-bank entry."""

    category = normalize_category(group.get("category", ""))
    difficulty = str(group.get("difficulty", "medium")).strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "medium"

    normalized = {
        "id": str(group.get("id") or ""),
        "category": category,
        "difficulty": difficulty,
        "words": [normalize_word(word) for word in group.get("words", [])],
    }

    explanation = normalize_category(group.get("explanation", ""))
    if explanation:
        normalized["explanation"] = explanation

    family = normalize_metadata_key(group.get("mechanism_family", ""))
    if not family:
        family = infer_mechanism_family(normalized)
    normalized["mechanism_family"] = family

    concept_key = normalize_metadata_key(group.get("concept_key", ""))
    if not concept_key:
        concept_key = normalize_metadata_key(concept_identity(normalized, family, category))
    normalized["concept_key"] = concept_key

    if not normalized["id"]:
        normalized["id"] = group_fingerprint(normalized)

    source = normalize_category(group.get("source", ""))
    if source:
        normalized["source"] = source

    origin_payload = origin if origin is not None else group.get("origin")
    if isinstance(origin_payload, dict):
        normalized["origin"] = {
            key: normalize_category(value)
            for key, value in origin_payload.items()
            if normalize_category(value)
        }

    normalized["verified"] = bool(group.get("verified", True))
    return normalized


def validate_group(group: dict[str, Any]) -> dict[str, list[str] | bool]:
    """Validate one reusable group-bank entry."""

    errors: list[str] = []
    warnings: list[str] = []
    category = normalize_category(group.get("category", ""))

    if not category:
        errors.append("Group is missing a category.")
    elif len(category) > 70:
        warnings.append(f"Category '{category}' is long for the reveal area.")

    if str(group.get("difficulty", "")).strip().lower() not in VALID_DIFFICULTIES:
        errors.append(f"Group '{category or group.get('id', 'unknown')}' has an invalid difficulty.")

    words = [normalize_word(word) for word in group.get("words", [])]
    if len(words) != 4:
        errors.append(f"Group '{category or group.get('id', 'unknown')}' must have exactly 4 words.")

    if len(set(words)) != len(words):
        errors.append(f"Group '{category or group.get('id', 'unknown')}' contains duplicate words.")

    for word in words:
        if not word:
            errors.append(f"Group '{category or group.get('id', 'unknown')}' contains an empty word.")
        elif not WORD_PATTERN.match(word):
            errors.append(f"Word '{word}' has unsupported characters or length.")
        elif len(word) > 14:
            warnings.append(f"Word '{word}' may be tight on small screens.")

    explanation = normalize_category(group.get("explanation", ""))
    if explanation:
        if len(explanation) > 180:
            errors.append(f"Group '{category or group.get('id', 'unknown')}' explanation is too long.")
        if DRAFT_EXPLANATION_PATTERN.search(explanation):
            errors.append(
                f"Group '{category or group.get('id', 'unknown')}' explanation contains draft text."
            )
        errors.extend(hidden_substring_mechanism_errors(group))
        errors.extend(hidden_claim_errors(group))
        errors.extend(homophone_claim_errors(group))
    else:
        errors.extend(hidden_substring_mechanism_errors(group))

    for metadata_field in ("mechanism_family", "concept_key"):
        if not normalize_metadata_key(group.get(metadata_field, "")):
            errors.append(f"Group '{category or group.get('id', 'unknown')}' is missing {metadata_field}.")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def load_group_bank(path: Path = DEFAULT_GROUP_BANK_PATH) -> list[dict[str, Any]]:
    """Load verified groups from disk."""

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    groups = payload.get("groups", payload) if isinstance(payload, dict) else payload
    if not isinstance(groups, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue

        entry = normalize_group(group)
        key = group_key(entry.get("words", []))
        if key in seen:
            continue

        validation = validate_group(entry)
        if validation["ok"]:
            normalized.append(entry)
            seen.add(key)

    return normalized


def save_group_bank(groups: list[dict[str, Any]], path: Path = DEFAULT_GROUP_BANK_PATH) -> None:
    """Save verified groups to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_group(group) for group in groups if validate_group(normalize_group(group))["ok"]]
    normalized.sort(key=lambda group: (group["difficulty"], group["category"].casefold()))

    with path.open("w", encoding="utf-8") as file:
        json.dump({"groups": normalized}, file, indent=2)
        file.write("\n")


def group_repeat_errors(group: dict[str, Any], existing_groups: list[dict[str, Any]]) -> list[str]:
    """Return repeat errors against the verified group bank."""

    errors: list[str] = []
    current_words = {normalize_word(word) for word in group.get("words", [])}
    current_key = group_key(list(current_words))
    current_concept = normalize_metadata_key(group.get("concept_key", ""))
    current_label = normalize_category(group.get("category", "")).casefold()

    for existing in existing_groups:
        existing_words = {normalize_word(word) for word in existing.get("words", [])}
        if current_key == group_key(list(existing_words)):
            errors.append(
                "Answer group repeats an existing verified group: "
                f"{', '.join(sorted(current_words))}."
            )
            break

        overlap = sorted(current_words & existing_words)
        if len(overlap) >= 3:
            errors.append(
                "Answer group overlaps 3+ words with an existing verified group: "
                f"{', '.join(sorted(current_words))} overlaps {', '.join(sorted(existing_words))}."
            )
            break

    for existing in existing_groups:
        existing_concept = normalize_metadata_key(existing.get("concept_key", ""))
        existing_label = normalize_category(existing.get("category", "")).casefold()
        if current_concept and current_concept == existing_concept:
            errors.append(
                "Category concept repeats an existing verified group: "
                f"{group.get('category')} overlaps {existing.get('category')}."
            )
            break

        if current_label and current_label == existing_label:
            errors.append(f"Category label repeats an existing verified group: {group.get('category')}.")
            break

    return errors


def add_groups_to_bank(
    new_groups: list[dict[str, Any]],
    path: Path = DEFAULT_GROUP_BANK_PATH,
    *,
    allow_duplicates: bool = False,
) -> dict[str, Any]:
    """Validate and append new verified groups."""

    existing = load_group_bank(path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for group in new_groups:
        normalized = normalize_group(group)
        validation = validate_group(normalized)
        repeat_errors = [] if allow_duplicates else group_repeat_errors(normalized, existing + accepted)

        if not validation["ok"] or repeat_errors:
            rejected.append(
                {
                    "group": normalized,
                    "errors": list(validation["errors"]) + repeat_errors,
                }
            )
            continue

        accepted.append(normalized)

    if accepted:
        save_group_bank(existing + accepted, path)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "total": len(existing) + len(accepted),
    }


def groups_from_puzzles(
    puzzles: list[dict[str, Any]],
    *,
    include_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create verified group entries from existing puzzles."""

    include_sources = include_sources or {"seed"}
    groups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for puzzle in puzzles:
        source = normalize_category(puzzle.get("source", ""))
        if source not in include_sources:
            continue

        puzzle_id = normalize_category(puzzle.get("id", ""))
        for group in puzzle.get("groups", []):
            if not isinstance(group, dict):
                continue

            entry = normalize_group(
                group,
                origin={"source": source, "puzzle_id": puzzle_id},
            )
            key = group_key(entry.get("words", []))
            validation = validate_group(entry)
            if not validation["ok"]:
                rejected.append({"group": entry, "errors": validation["errors"]})
                continue

            if key in seen:
                continue

            groups.append(entry)
            seen.add(key)

    return groups, rejected


def ensure_group_bank(
    existing_puzzles: list[dict[str, Any]],
    path: Path = DEFAULT_GROUP_BANK_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the group bank, seeding it from starter puzzles when missing."""

    groups = load_group_bank(path)
    if groups:
        return groups, {"created": False, "group_count": len(groups), "rejected": []}

    groups, rejected = groups_from_puzzles(existing_puzzles, include_sources={"seed"})
    save_group_bank(groups, path)
    return groups, {"created": True, "group_count": len(groups), "rejected": rejected}


def candidate_pool(groups: list[dict[str, Any]], lane: str) -> list[dict[str, Any]]:
    """Return group candidates for one reveal lane."""

    if lane == "easy":
        difficulties = {"easy"}
    elif lane == "medium":
        difficulties = {"medium"}
    elif lane == "hard":
        difficulties = {"hard", "medium"}
    else:
        difficulties = {"tricky", "hard"}

    pool = [
        group
        for group in groups
        if group.get("difficulty") in difficulties
        and not hidden_substring_mechanism_errors(group)
    ]
    if lane == "hard" and not any(group.get("difficulty") == "hard" for group in pool):
        return [
            group
            for group in groups
            if group.get("difficulty") in {"medium", "tricky"}
            and not hidden_substring_mechanism_errors(group)
        ]
    return pool


def playable_group(group: dict[str, Any], lane: str) -> dict[str, Any]:
    """Return the puzzle-facing version of a group-bank entry."""

    return {
        "category": group["category"],
        "difficulty": lane,
        "mechanism_family": group["mechanism_family"],
        "concept_key": group["concept_key"],
        "words": list(group["words"]),
        "explanation": group.get("explanation", ""),
    }


def can_add_group(
    selected: list[dict[str, Any]],
    candidate: dict[str, Any],
    used_words: set[str],
    used_concepts: set[str],
) -> bool:
    """Return true when a group can join the current board."""

    words = {normalize_word(word) for word in candidate.get("words", [])}
    if len(words) != 4 or used_words & words:
        return False

    concept = normalize_metadata_key(candidate.get("concept_key", ""))
    if concept and concept in used_concepts:
        return False

    if is_phrase_group(candidate) and any(is_phrase_group(group) for group in selected):
        return False

    return True


def assembled_puzzle_errors(puzzle: dict[str, Any]) -> list[str]:
    """Return deterministic errors for assembled group-bank puzzles."""

    validation = validate_puzzle(
        puzzle,
        require_nyt_blocklist=True,
        enforce_generated_quality=False,
        require_generated_metadata=False,
    )
    errors = list(validation["errors"])
    errors.extend(generated_constraint_errors(puzzle.get("groups", [])))
    errors.extend(wordplay_leakage_errors(puzzle.get("groups", [])))
    return errors


def build_candidate_puzzle(
    groups: list[dict[str, Any]],
    *,
    rng: random.Random,
    difficulty_mode: str,
    usage_counts: Counter[str] | None = None,
) -> dict[str, Any] | None:
    """Assemble one candidate puzzle from verified groups."""

    usage_counts = usage_counts or Counter()
    selected: list[dict[str, Any]] = []
    used_words: set[str] = set()
    used_concepts: set[str] = set()

    for lane in LANES:
        pool = candidate_pool(groups, lane)
        rng.shuffle(pool)
        pool.sort(key=lambda group: usage_counts[group_key(group.get("words", []))])

        chosen: dict[str, Any] | None = None
        for candidate in pool:
            if can_add_group(selected, candidate, used_words, used_concepts):
                chosen = playable_group(candidate, lane)
                break

        if chosen is None:
            return None

        selected.append(chosen)
        used_words.update(normalize_word(word) for word in chosen.get("words", []))
        used_concepts.add(normalize_metadata_key(chosen.get("concept_key", "")))

    puzzle = normalize_puzzle(
        {
            "groups": selected,
            "difficulty_mode": difficulty_mode,
            "decoy": None,
            "source": GROUP_BANK_SOURCE,
        },
        source=GROUP_BANK_SOURCE,
    )
    puzzle["id"] = f"gb-{puzzle_fingerprint(puzzle)}"
    return puzzle


def build_puzzle_from_lane_groups(
    lane_groups: dict[str, dict[str, Any]],
    *,
    difficulty_mode: str = "easy",
    source: str = GROUP_BANK_SOURCE,
) -> dict[str, Any]:
    """Build a puzzle from one selected group for each reveal lane."""

    selected = [
        playable_group(lane_groups[lane], lane)
        for lane in LANES
    ]
    puzzle = normalize_puzzle(
        {
            "groups": selected,
            "difficulty_mode": difficulty_mode,
            "decoy": None,
            "source": source,
        },
        source=source,
    )
    puzzle["id"] = f"gb-{puzzle_fingerprint(puzzle)}"
    return puzzle


def assemble_puzzle_batch(
    *,
    target_count: int,
    existing_puzzles: list[dict[str, Any]],
    difficulty: str = "easy",
    seed: int | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """Assemble valid puzzles from the local verified group bank."""

    started = time.time()
    groups, seed_info = ensure_group_bank(existing_puzzles)
    rng = random.Random(seed)
    max_attempts = max_attempts or max(200, target_count * 300)
    difficulty_mode = "hard" if difficulty == "hard" else "easy"
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    usage_counts: Counter[str] = Counter(
        group_key(group.get("words", []))
        for puzzle in existing_puzzles
        if puzzle.get("source") == GROUP_BANK_SOURCE
        for group in puzzle.get("groups", [])
        if isinstance(group, dict)
    )
    seen_fingerprints = {puzzle_fingerprint(puzzle) for puzzle in existing_puzzles}

    for _ in range(max_attempts):
        if len(accepted) >= target_count:
            break

        candidate = build_candidate_puzzle(
            groups,
            rng=rng,
            difficulty_mode=difficulty_mode,
            usage_counts=usage_counts,
        )
        if candidate is None:
            rejection_counts["Could not assemble four compatible groups."] += 1
            continue

        fingerprint = puzzle_fingerprint(candidate)
        if fingerprint in seen_fingerprints:
            rejection_counts["Duplicate assembled puzzle fingerprint."] += 1
            continue

        errors = assembled_puzzle_errors(candidate)
        if errors:
            for error in errors:
                rejection_counts[error] += 1
            if len(rejected) < 20:
                rejected.append({"puzzle": candidate, "errors": errors, "stage": "group_bank_validator"})
            continue

        accepted.append(candidate)
        seen_fingerprints.add(fingerprint)
        for group in candidate.get("groups", []):
            usage_counts[group_key(group.get("words", []))] += 1

    trace = [
        {
            "agent": "Group Bank",
            "status": "complete",
            "summary": (
                f"Loaded {len(groups)} verified group(s)."
                if not seed_info["created"]
                else f"Seeded {len(groups)} verified group(s) from starter puzzles."
            ),
            "duration_seconds": 0,
            "details": seed_info,
        },
        {
            "agent": "Puzzle Assembler",
            "status": "complete" if accepted else "warning",
            "summary": f"Assembled {len(accepted)} puzzle(s) with no Claude API call.",
            "duration_seconds": round(time.time() - started, 2),
            "details": {
                "target_count": target_count,
                "attempt_limit": max_attempts,
                "accepted": len(accepted),
                "top_rejections": rejection_counts.most_common(8),
            },
        },
    ]

    return {"accepted": accepted, "rejected": rejected, "trace": trace}
