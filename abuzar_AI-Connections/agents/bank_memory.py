"""Memory helpers for avoiding repetitive generated puzzles."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from agents.puzzle_validator import (
    is_color_shade_group,
    is_generic_group,
    is_phrase_group,
    normalize_category,
    normalize_metadata_key,
    normalize_word,
)


REUSABLE_LABEL_PATTERNS = (
    re.compile(r"^words? before\b", re.IGNORECASE),
    re.compile(r"^words? after\b", re.IGNORECASE),
    re.compile(r"^things? with\b", re.IGNORECASE),
    re.compile(r"^things? that\b", re.IGNORECASE),
    re.compile(r"^anagrams? of\b", re.IGNORECASE),
    re.compile(r"^synonyms? for\b", re.IGNORECASE),
    re.compile(r"^shades? of\b", re.IGNORECASE),
    re.compile(r"^parts? of\b", re.IGNORECASE),
    re.compile(r"^types? of\b", re.IGNORECASE),
    re.compile(r"^kinds? of\b", re.IGNORECASE),
    re.compile(r"^___\b"),
    re.compile(r"\b___$"),
)


def label_key(label: str) -> str:
    """Normalize a category label for repeat checks."""

    return normalize_category(label).casefold()


def group_key(words: list[Any]) -> str:
    """Normalize a four-word group for repeat checks."""

    return "|".join(sorted(normalize_word(word) for word in words))


def normalize_mechanism_target(value: str) -> str:
    """Normalize a phrase mechanism target such as house, WORK, or 'sun'."""

    target = normalize_category(value)
    target = re.sub(r"\s*\([^)]*\)", "", target)
    target = target.strip(" _'\"")
    target = re.sub(r"[^A-Za-z0-9 ]+", "", target)
    target = " ".join(target.casefold().split())
    return target


def mechanism_key(label: str) -> str | None:
    """Return a repeat key for reusable labels that can be written multiple ways."""

    normalized = normalize_category(label)
    normalized = re.sub(r"\s*\([^)]*\)", "", normalized).strip()

    match = re.match(r"^words? before\s+['\"]?(.+?)['\"]?$", normalized, re.IGNORECASE)
    if match:
        target = normalize_mechanism_target(match.group(1))
        return f"phrase-before-target:{target}" if target else None

    match = re.match(r"^words? after\s+['\"]?(.+?)['\"]?$", normalized, re.IGNORECASE)
    if match:
        target = normalize_mechanism_target(match.group(1))
        return f"phrase-after-target:{target}" if target else None

    match = re.match(r"^___\s+(.+?)$", normalized, re.IGNORECASE)
    if match:
        target = normalize_mechanism_target(match.group(1))
        return f"phrase-before-target:{target}" if target else None

    match = re.match(r"^(.+?)\s+___$", normalized, re.IGNORECASE)
    if match:
        target = normalize_mechanism_target(match.group(1))
        return f"phrase-after-target:{target}" if target else None

    match = re.match(r"^anagrams? of\s+(.+?)$", normalized, re.IGNORECASE)
    if match:
        target = normalize_mechanism_target(match.group(1))
        return f"anagrams-of:{target}" if target else None

    match = re.match(r"^homophones? of\s+(.+?)$", normalized, re.IGNORECASE)
    if match:
        target = normalize_mechanism_target(match.group(1))
        return f"homophones-of:{target}" if target else None

    if re.search(r"\bletter sounds?\b", normalized, re.IGNORECASE):
        return "homophones-of:letters"

    match = re.match(r"^shades? of\s+(.+?)$", normalized, re.IGNORECASE)
    if match:
        return "color-shade-set"

    return None


def infer_mechanism_family(group: dict[str, Any]) -> str:
    """Infer a broad mechanism family from metadata, label, and explanation."""

    explicit_family = normalize_metadata_key(group.get("mechanism_family", ""))
    if explicit_family:
        return explicit_family

    label = normalize_category(group.get("category", ""))
    text = " ".join(
        part
        for part in (label, normalize_category(group.get("explanation", "")))
        if part
    ).casefold()

    if is_phrase_group(group):
        return "phrase_completion"

    if is_color_shade_group(group):
        return "color_shade"

    if re.search(r"\b(homophones?|rhymes?|sounds?\s+like|silent)\b", text):
        return "sound"

    if re.search(
        r"\b(anagrams?|palindromes?|hidden|letters?|initials?|prefix|suffix|endings?|"
        r"spelled|spelling)\b",
        text,
    ):
        return "spelling"

    if re.search(r"\b(programming languages?|keyboard keys?|also (?:a|an|means?))\b", text):
        return "double_identity"

    if re.search(r"\b(things? with|things? that|can be|have|has)\b", text):
        return "shared_property"

    return "semantic_set"


def concept_identity(group: dict[str, Any], mechanism_family: str, label: str) -> str:
    """Return a stable concept identity for repeat checks."""

    explicit_key = normalize_metadata_key(group.get("concept_key", ""))
    if explicit_key:
        if mechanism_family == "phrase_completion":
            for prefix in ("phrase_before_", "words_before_", "before_"):
                if explicit_key.startswith(prefix):
                    target = explicit_key.removeprefix(prefix)
                    return f"phrase-before-target:{target}" if target else explicit_key

            for prefix in ("phrase_after_", "words_after_", "after_"):
                if explicit_key.startswith(prefix):
                    target = explicit_key.removeprefix(prefix)
                    return f"phrase-after-target:{target}" if target else explicit_key

        if mechanism_family == "color_shade" or "shade" in explicit_key:
            return "color-shade-set"

        if mechanism_family == "sound" and "letter" in explicit_key and "homophone" in explicit_key:
            return "homophones-of:letters"

        return f"{mechanism_family}:{explicit_key}"

    fallback_key = mechanism_key(label)
    if fallback_key:
        return fallback_key

    return f"{mechanism_family}:{label_key(label)}"


def is_reusable_mechanism(label: str) -> bool:
    """Return true for labels where the mechanism may repeat with a new target."""

    normalized = normalize_category(label)
    return any(pattern.search(normalized) for pattern in REUSABLE_LABEL_PATTERNS)


def group_entries(puzzle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized group entries from a puzzle."""

    entries: list[dict[str, Any]] = []
    for group in puzzle.get("groups", []):
        if not isinstance(group, dict):
            continue

        label = normalize_category(group.get("category", ""))
        words = [normalize_word(word) for word in group.get("words", [])]
        if not label or len(words) != 4:
            continue
        family = infer_mechanism_family(group)

        entries.append(
            {
                "label": label,
                "label_key": label_key(label),
                "mechanism_key": mechanism_key(label),
                "mechanism_family": family,
                "concept_identity": concept_identity(group, family, label),
                "group_key": group_key(words),
                "words": words,
                "reusable": is_reusable_mechanism(label),
            }
        )

    return entries


def purple_mechanism(group: dict[str, Any]) -> str:
    """Classify the purple group mechanism for bank-level variety checks."""

    inferred_family = infer_mechanism_family(group)
    if inferred_family == "phrase_completion":
        return "phrase"

    if inferred_family in {"sound", "spelling", "shared_property", "color_shade"}:
        return inferred_family

    if is_phrase_group(group):
        return "phrase"

    if is_color_shade_group(group):
        return "color_shade"

    if is_generic_group(group):
        return "generic_list"

    text = " ".join(
        part
        for part in (
            normalize_category(group.get("category", "")),
            normalize_category(group.get("explanation", "")),
        )
        if part
    ).casefold()

    if re.search(r"\b(homophones?|rhymes?|sounds?\s+like|silent)\b", text):
        return "sound"

    if re.search(
        r"\b(anagrams?|palindromes?|hidden|letters?|initials?|prefix|suffix|endings?|"
        r"spelled|spelling)\b",
        text,
    ):
        return "spelling"

    if re.search(r"\b(double meanings?|also means?|can be|things? with)\b", text):
        return "shared_property"

    return "set_or_semantic"


def build_bank_memory(puzzles: list[dict[str, Any]], recent_limit: int = 80) -> dict[str, Any]:
    """Build compact memory from the current puzzle bank."""

    label_counts: Counter[str] = Counter()
    label_display: dict[str, str] = {}
    specific_labels: set[str] = set()
    reusable_labels: set[str] = set()
    group_keys: set[str] = set()
    group_display: dict[str, list[str]] = {}
    group_word_sets: list[set[str]] = []
    group_set_display: list[list[str]] = []
    mechanism_keys: dict[str, str] = {}
    concept_identities: dict[str, str] = {}
    mechanism_family_counts: Counter[str] = Counter()
    generated_mechanism_family_counts: Counter[str] = Counter()
    generated_purple_mechanisms: Counter[str] = Counter()
    recent_generated_purple_mechanisms: list[str] = []
    recent_labels: list[str] = []

    for puzzle in puzzles:
        groups = puzzle.get("groups", [])
        if puzzle.get("source") == "claude-multi-agent" and isinstance(groups, list) and len(groups) >= 4:
            purple = groups[3]
            if isinstance(purple, dict):
                mechanism = purple_mechanism(purple)
                generated_purple_mechanisms[mechanism] += 1
                recent_generated_purple_mechanisms.append(mechanism)

        for entry in group_entries(puzzle):
            key = entry["label_key"]
            label_counts[key] += 1
            label_display.setdefault(key, entry["label"])
            group_keys.add(entry["group_key"])
            group_display.setdefault(entry["group_key"], entry["words"])
            group_word_sets.append(set(entry["words"]))
            group_set_display.append(entry["words"])
            recent_labels.append(entry["label"])
            mechanism_family_counts[entry["mechanism_family"]] += 1
            if puzzle.get("source") == "claude-multi-agent":
                generated_mechanism_family_counts[entry["mechanism_family"]] += 1

            if entry["mechanism_key"]:
                mechanism_keys.setdefault(entry["mechanism_key"], entry["label"])

            if entry["concept_identity"]:
                concept_identities.setdefault(entry["concept_identity"], entry["label"])

            if entry["reusable"]:
                reusable_labels.add(key)
            else:
                specific_labels.add(key)

    most_common_labels = [
        label_display[key]
        for key, _ in label_counts.most_common(30)
    ]

    recent_unique_labels = []
    seen_recent: set[str] = set()
    for label in reversed(recent_labels):
        key = label_key(label)
        if key in seen_recent:
            continue
        seen_recent.add(key)
        recent_unique_labels.append(label)
        if len(recent_unique_labels) >= recent_limit:
            break

    return {
        "specific_label_keys": specific_labels,
        "reusable_label_keys": reusable_labels,
        "all_label_keys": set(label_counts),
        "label_display": label_display,
        "group_keys": group_keys,
        "group_display": group_display,
        "group_word_sets": group_word_sets,
        "group_set_display": group_set_display,
        "mechanism_keys": mechanism_keys,
        "concept_identities": concept_identities,
        "mechanism_family_counts": mechanism_family_counts,
        "generated_mechanism_family_counts": generated_mechanism_family_counts,
        "generated_purple_mechanisms": generated_purple_mechanisms,
        "generated_puzzle_count": sum(generated_purple_mechanisms.values()),
        "recent_generated_purple_mechanisms": list(reversed(recent_generated_purple_mechanisms[-12:])),
        "most_common_labels": most_common_labels,
        "recent_labels": recent_unique_labels,
        "total_puzzles": len(puzzles),
        "total_groups": len(recent_labels),
    }


def mechanism_diversity_errors(puzzle: dict[str, Any], memory: dict[str, Any]) -> list[str]:
    """Return errors when a generated puzzle overuses a purple mechanism family."""

    if puzzle.get("source") != "claude-multi-agent":
        return []

    groups = puzzle.get("groups", [])
    if not isinstance(groups, list) or len(groups) < 4 or not isinstance(groups[3], dict):
        return []

    mechanism = purple_mechanism(groups[3])
    if mechanism != "phrase":
        return []

    counts = memory.get("generated_purple_mechanisms", Counter())
    generated_count = int(memory.get("generated_puzzle_count", 0))
    phrase_count = int(counts.get("phrase", 0))
    max_phrase_after_add = max(1, (generated_count + 4) // 4)
    errors: list[str] = []

    if phrase_count + 1 > max_phrase_after_add:
        errors.append(
            "Purple compound/phrase groups are allowed, but the generated bank already "
            "uses them too often. Use a different purple mechanism."
        )

    recent_mechanisms = memory.get("recent_generated_purple_mechanisms", [])
    if recent_mechanisms[:1] == ["phrase"]:
        errors.append(
            "Purple compound/phrase groups should not appear in back-to-back generated puzzles."
        )

    return errors


def repeat_errors(
    puzzle: dict[str, Any],
    memory: dict[str, Any],
    *,
    strict_labels: bool = True,
) -> list[str]:
    """Return deterministic repeat errors against a bank memory snapshot."""

    errors: list[str] = []
    specific_label_keys = memory.get("specific_label_keys", set())
    all_label_keys = memory.get("all_label_keys", set())
    label_display = memory.get("label_display", {})
    group_keys = memory.get("group_keys", set())
    group_word_sets = memory.get("group_word_sets", [])
    group_set_display = memory.get("group_set_display", [])
    mechanism_keys = memory.get("mechanism_keys", {})
    concept_identities = memory.get("concept_identities", {})
    errors.extend(mechanism_diversity_errors(puzzle, memory))

    for entry in group_entries(puzzle):
        if entry["group_key"] in group_keys:
            errors.append(
                f"Answer group repeats an existing bank group: {', '.join(entry['words'])}."
            )
        else:
            current_words = set(entry["words"])
            for existing_words, existing_display in zip(group_word_sets, group_set_display):
                overlap = sorted(current_words & existing_words)
                if len(overlap) >= 3:
                    errors.append(
                        "Answer group overlaps 3+ words with an existing bank group: "
                        f"{', '.join(entry['words'])} overlaps {', '.join(existing_display)}."
                    )
                    break

        if not strict_labels:
            continue

        if entry["concept_identity"] in concept_identities:
            errors.append(
                "Category concept repeats an existing bank concept: "
                f"{entry['label']} overlaps {concept_identities[entry['concept_identity']]}."
            )
            continue

        if entry["reusable"]:
            if entry["label_key"] in all_label_keys:
                errors.append(f"Reusable mechanism label repeats exactly: {entry['label']}.")
            elif entry["mechanism_key"] and entry["mechanism_key"] in mechanism_keys:
                errors.append(
                    "Reusable mechanism target repeats under a different label: "
                    f"{entry['label']} overlaps {mechanism_keys[entry['mechanism_key']]}."
                )
        elif entry["label_key"] in specific_label_keys:
            errors.append(f"Specific category label repeats: {entry['label']}.")
        else:
            for existing_key in specific_label_keys:
                if len(existing_key) < 5 or len(entry["label_key"]) < 5:
                    continue

                if existing_key in entry["label_key"] or entry["label_key"] in existing_key:
                    errors.append(
                        "Specific category label appears to repeat an existing concept: "
                        f"{entry['label']} overlaps {label_display.get(existing_key, existing_key)}."
                    )
                    break

    return errors


def avoid_instructions(memory: dict[str, Any], max_labels: int = 50) -> str:
    """Create concise prompt text that helps Claude avoid repeated concepts."""

    recent_labels = memory.get("recent_labels", [])[:max_labels]

    if not recent_labels:
        return "No existing bank memory yet."

    label_lines = "\n".join(f"- {label}" for label in recent_labels)

    return f"""
Existing bank memory:
- Avoid exact repeated four-word answer groups. The local validator checks the full bank after generation.
- Avoid groups that overlap 3 or more words with an existing answer group. The local validator checks the full bank after generation.
- Every generated group should include mechanism_family and concept_key metadata. The local validator uses this to catch renamed repeats.
- Avoid repeated concept_key values. For example, "Planets", "Planet names", and "Planets in our solar system" should all share concept_key "planets" and should not repeat.
- Avoid repeating specific set categories such as Planets, Flowers, Card suits, Chess pieces, or Dog breeds.
- Reusable mechanisms may repeat only with a new target. For example, "Words before board" and "Words before paper" are different, but do not reuse the exact same full label.
- If a label is a specific set, do not reuse it. If a label is a mechanism, do not reuse the exact same full label.
- Do not rename a specific category to dodge the rule. For example, "Planets" and "Planets in our solar system" count as the same concept.
- Do not rename a reusable phrase target to dodge the rule. For example, "Words before house" and "___ House" count as the same concept.
- Avoid overused safe families already represented in the bank, especially color-shade lists, body/face part lists, and desk/bedroom/beach object lists.
- Simple phrase categories are allowed, including as purple, but diversify them. Do not make purple compound/phrase categories the default; the local validator enforces a rough one-in-four cap across generated puzzles.

Recent/existing category labels to avoid repeating exactly:
{label_lines or "- none"}
""".strip()
