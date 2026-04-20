"""Concept inspiration helpers for group generation."""

from __future__ import annotations

from functools import lru_cache
import json
import random
import re
from pathlib import Path
from typing import Any

from agents.puzzle_validator import normalize_category, normalize_metadata_key


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONCEPT_PATH = ROOT_DIR / "data" / "concept_inspiration.json"
VALID_HINTS = {"easy", "medium", "hard", "tricky"}
NEAR_COPY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "category",
    "common",
    "group",
    "groups",
    "in",
    "kinds",
    "names",
    "of",
    "on",
    "set",
    "sorts",
    "the",
    "things",
    "thing",
    "types",
    "type",
    "with",
    "words",
    "word",
}


def normalize_concept_entry(entry: Any) -> dict[str, str] | None:
    """Normalize a concept seed entry from JSON."""

    if isinstance(entry, str):
        concept = normalize_category(entry)
        family = ""
        difficulty_hint = ""
    elif isinstance(entry, dict):
        concept = normalize_category(entry.get("concept", ""))
        family = normalize_metadata_key(entry.get("family", ""))
        difficulty_hint = normalize_metadata_key(entry.get("difficulty_hint", ""))
    else:
        return None

    if not concept:
        return None

    if difficulty_hint not in VALID_HINTS:
        difficulty_hint = ""

    return {
        "concept": concept,
        "family": family,
        "difficulty_hint": difficulty_hint,
        "concept_key": normalize_metadata_key(concept),
        "near_key": near_copy_key(concept),
    }


@lru_cache(maxsize=4)
def load_concept_inspiration(path_text: str = str(DEFAULT_CONCEPT_PATH)) -> tuple[dict[str, str], ...]:
    """Load cached concept inspiration seeds."""

    path = Path(path_text)
    if not path.exists():
        return ()

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    raw_items = payload.get("concepts", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        return ()

    concepts: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = normalize_concept_entry(raw_item)
        if not item or item["concept_key"] in seen:
            continue
        concepts.append(item)
        seen.add(item["concept_key"])

    return tuple(concepts)


def near_copy_key(value: Any) -> str:
    """Return a coarse key for catching direct renames."""

    words = re.findall(r"[a-z0-9]+", normalize_category(value).casefold())
    filtered = [word for word in words if word not in NEAR_COPY_STOPWORDS]
    if not filtered:
        filtered = words
    return "_".join(filtered)


def sample_concept_inspiration(
    *,
    difficulty: str = "mixed",
    count: int = 18,
) -> list[dict[str, str]]:
    """Return a small sampled concept list for the prompt."""

    concepts = list(load_concept_inspiration())
    if not concepts:
        return []

    difficulty = normalize_metadata_key(difficulty)
    count = max(1, min(count, 40))
    random.shuffle(concepts)

    if difficulty in VALID_HINTS:
        preferred = [item for item in concepts if item.get("difficulty_hint") == difficulty]
        others = [item for item in concepts if item.get("difficulty_hint") != difficulty]
        selected = preferred[:count]
        if len(selected) < count:
            selected.extend(others[: count - len(selected)])
        return selected[:count]

    return concepts[:count]


def format_concept_inspiration_guidance(
    *,
    difficulty: str = "mixed",
    count: int = 18,
) -> str:
    """Format sampled concept seeds as prompt guidance."""

    concepts = sample_concept_inspiration(difficulty=difficulty, count=count)
    if not concepts:
        return "Concept inspiration: no local concept inspiration file found."

    lines = []
    for item in concepts:
        details = []
        if item.get("family"):
            details.append(item["family"])
        if item.get("difficulty_hint"):
            details.append(item["difficulty_hint"])
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {item['concept']}{suffix}")

    return f"""
Concept inspiration:
Treat this list as a style-and-pattern study guide, not as a menu. The sampled
concepts show the kinds of category logic that can work, but every exact concept
and near-rename is off limits. Do not choose one seed and fill in four answers
for it. Instead, understand why the seed works and invent a different nearby
concept, sibling concept, contrast concept, narrower variant, broader variant,
or mechanism-shifted version.

Good transformations use the seed's pattern while changing the actual concept:
- "Types of bread" -> "Words before roll"
- "Planets" -> "Words hiding planet names"
- "Dog breeds" -> "Words that are also boxing terms"
- "Swimming strokes" -> "Words after back"
- "Greek goddesses" -> "Words ending in goddess names"
- "Subatomic particles" -> "Words hiding physics units"

Bad transformations copy or lightly rename the seed:
- "Types of bread" -> "Bread varieties"
- "Planets" -> "Planet names"
- "Dog breeds" -> "Types of dogs"
- "Swimming strokes" -> "Swim strokes"
- "Greek goddesses" -> "Names of Greek goddesses"
- "Subatomic particles" -> "Particles in physics"

Sampled inspiration seeds:
{chr(10).join(lines)}
""".strip()


def inspiration_copy_errors(group: dict[str, Any]) -> list[str]:
    """Return errors when a generated group copies an inspiration seed directly."""

    concepts = load_concept_inspiration()
    if not concepts:
        return []

    category_key = normalize_metadata_key(group.get("category", ""))
    concept_key = normalize_metadata_key(group.get("concept_key", ""))
    category_near_key = near_copy_key(group.get("category", ""))
    errors: list[str] = []

    for concept in concepts:
        seed_key = concept["concept_key"]
        seed_near_key = concept["near_key"]

        if category_key and category_key == seed_key:
            errors.append(
                "Generated group copies an inspiration concept exactly as its category: "
                f"{group.get('category')}."
            )
            break

        if concept_key and concept_key == seed_key:
            errors.append(
                "Generated group copies an inspiration concept exactly as its concept_key: "
                f"{group.get('concept_key')}."
            )
            break

        if (
            category_near_key
            and seed_near_key
            and len(seed_near_key) >= 5
            and category_near_key == seed_near_key
        ):
            errors.append(
                "Generated group appears to be a near-rename of an inspiration concept: "
                f"{group.get('category')} overlaps {concept['concept']}."
            )
            break

    return errors
