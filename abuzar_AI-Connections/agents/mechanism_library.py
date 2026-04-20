"""Local mechanism-library helpers for generator prompt guidance."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_PATH = ROOT_DIR / "data" / "mechanism_library.json"
DEFAULT_INSPIRATION_PATH = ROOT_DIR / "data" / "inspiration_words.json"


def load_mechanism_library(path: Path = DEFAULT_LIBRARY_PATH) -> dict[str, Any]:
    """Load the local mechanism library used as sampled prompt inspiration."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Mechanism library must be a JSON object.")

    families = payload.get("families", [])
    if not isinstance(families, list) or not families:
        raise ValueError("Mechanism library must include a non-empty families list.")

    return payload


def load_inspiration_words(path: Path = DEFAULT_INSPIRATION_PATH) -> list[str]:
    """Load neutral local words used only as diversity seeds."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    words = payload.get("words", []) if isinstance(payload, dict) else []
    cleaned = [
        str(word).strip().upper()
        for word in words
        if str(word).strip()
    ]

    if len(cleaned) < 6:
        raise ValueError("Inspiration word list must contain at least 6 words.")

    return cleaned


def format_inspiration_guidance(count: int = 6) -> str:
    """Return a small random seed that nudges generation away from repeats."""

    words = random.sample(load_inspiration_words(), k=max(1, min(count, 10)))
    return f"""
Diversity seed:
- Neutral random words for loose inspiration only: {", ".join(words)}
- You may imagine a one-sentence scene using these words internally, but do not output the scene.
- Do not force these exact words into the puzzle.
- Do not build a category simply named after these words.
- Use the seed only to escape stale/default category ideas and invent fresher mechanisms.
""".strip()


def select_mechanism_families(
    memory: dict[str, Any],
    *,
    difficulty: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Pick a compact, underused subset of mechanism families for this run."""

    library = load_mechanism_library()
    families = [family for family in library["families"] if isinstance(family, dict)]
    usage = Counter(memory.get("generated_mechanism_family_counts", {}))
    difficulty = difficulty.strip().lower()
    preferred_order = [
        "shared_property",
        "sound",
        "spelling",
        "double_identity",
        "transformation",
        "phrase_completion",
    ]
    if difficulty == "hard":
        preferred_order = [
            "sound",
            "spelling",
            "transformation",
            "double_identity",
            "shared_property",
            "phrase_completion",
        ]
    fallback_order = preferred_order + ["semantic_set", "light_trivia"]
    priority_index = {family_id: index for index, family_id in enumerate(fallback_order)}
    by_id = {str(family.get("id", "")): family for family in families}

    def family_rank(family: dict[str, Any]) -> tuple[int, int, str]:
        family_id = str(family.get("id", ""))
        return (
            int(usage.get(family_id, 0)),
            priority_index.get(family_id, len(fallback_order)),
            family_id,
        )

    selected = [
        by_id[family_id]
        for family_id in preferred_order
        if family_id in by_id
    ][: max(1, limit)]
    selected_ids = {family.get("id") for family in selected}

    if len(selected) < max(1, limit):
        for family in sorted(families, key=family_rank):
            family_id = family.get("id")
            if family_id in selected_ids:
                continue
            selected.append(family)
            selected_ids.add(family_id)
            if len(selected) >= max(1, limit):
                break

    return selected[: max(1, limit)]


def format_mechanism_guidance(
    memory: dict[str, Any],
    *,
    difficulty: str,
    limit: int = 6,
) -> str:
    """Format sampled mechanism families as concise prompt guidance."""

    library = load_mechanism_library()
    families = select_mechanism_families(memory, difficulty=difficulty, limit=limit)
    usage = Counter(memory.get("generated_mechanism_family_counts", {}))
    instruction_lines = "\n".join(f"- {item}" for item in library.get("instructions", []))
    family_blocks: list[str] = []

    for family in families:
        examples = family.get("examples", [])[:2]
        avoid = family.get("avoid", [])[:4]
        family_blocks.append(
            "\n".join(
                [
                    f"- {family.get('id')}: {family.get('name')}",
                    f"  Use: {family.get('description')}",
                    f"  Example patterns to adapt, not copy: {'; '.join(examples)}",
                    f"  Avoid: {'; '.join(avoid)}",
                    f"  Existing generated use count: {usage.get(str(family.get('id')), 0)}",
                ]
            )
        )

    family_text = "\n".join(family_blocks)

    return f"""
Mechanism library guidance:
{instruction_lines}

Sampled underused mechanism families for this run:
{family_text}

Metadata required for every category idea and every final answer group:
- mechanism_family: use one of the sampled family ids, another library family id, or a concise new id if the mechanism is genuinely different.
- concept_key: a stable snake_case concept key. Same concept must use the same key even if the category label is reworded.
- For phrase categories, include direction and target in concept_key, such as phrase_before_board or phrase_after_ice.
- For specific sets, use the underlying set, such as planets, chess_pieces, or programming_languages.
""".strip()
