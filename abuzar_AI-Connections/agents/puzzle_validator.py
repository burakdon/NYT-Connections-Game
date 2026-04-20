"""Local validation for Connections-style puzzles.

The validator is intentionally deterministic and cheap. Claude agents judge
taste and fairness; this module rejects objective shape problems before a
puzzle reaches the game bank.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Any

from agents.nyt_guard import check_puzzle_against_blocklist


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_WORDLIST_PATH = ROOT_DIR / "wordlist.txt"
VALID_DIFFICULTIES = {"easy", "medium", "hard", "tricky"}
GENERATED_SOURCE = "claude-multi-agent"
GENERATED_DIFFICULTY_ORDER = ("easy", "medium", "hard", "tricky")
WORD_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9 '&-]{0,23}$")
DRAFT_EXPLANATION_PATTERN = re.compile(
    r"\b(corrected|correction|replace|replacing|final clean set|settled|conflicts?|"
    r"oops|actually|alternatively|alternative|wait|wrong|correct set|scratch that|"
    r"revision|revised)\b",
    re.IGNORECASE,
)
PHRASE_CATEGORY_PATTERN = re.compile(
    r"(___|\bwords?\s+(?:before|after)\b|\bcompound\s+(?:word|phrase)s?\b)",
    re.IGNORECASE,
)
PHRASE_EXPLANATION_PATTERN = re.compile(
    r"\b(?:precedes?|follows?|pairs?\s+with|combines?\s+with|"
    r"placed\s+(?:before|after)|added\s+(?:before|after)|"
    r"forms?\s+(?:a\s+)?(?:compound|phrase))\b",
    re.IGNORECASE,
)
GENERIC_CATEGORY_PATTERN = re.compile(
    r"^(?:things|items|objects|animals|features|elements)\s+"
    r"(?:(?:you\s+)?(?:see|use|find)\b|(?:in|at|on)\b|found\s+(?:in|on|at)\b|"
    r"associated\s+with\b)",
    re.IGNORECASE,
)
COLOR_SHADE_PATTERN = re.compile(
    r"\b(?:shades?\s+of|color\s+shades?|colors?\s+in\s+the)\b",
    re.IGNORECASE,
)
CAN_VERB_CATEGORY_PATTERN = re.compile(
    r"^(?:things?|items?|objects?)\s+"
    r"(?:(?:you|we|people)\s+can|that\s+can(?:\s+be)?)\s+"
    r"[a-z][a-z'-]*\b",
    re.IGNORECASE,
)
PLAIN_SYNONYM_CATEGORY_PATTERN = re.compile(
    r"\b(?:synonyms?\s+(?:of|for)|ways?\s+to\s+say|"
    r"(?:words?|verbs?|nouns?|adjectives?)\s+"
    r"(?:(?:that\s+(?:are\s+also\s+)?)?(?:mean|means)|meaning|for)\b)",
    re.IGNORECASE,
)
PLAIN_SYNONYM_EXPLANATION_PATTERN = re.compile(
    r"\b(?:are\s+synonyms?|all\s+(?:mean|refer\s+to)|"
    r"function\s+as\s+(?:words?|verbs?|nouns?|adjectives?)\s+meaning)\b",
    re.IGNORECASE,
)
EXTRA_TRICK_PATTERN = re.compile(
    r"\b(?:homophones?|rhymes?|sounds?\s+like|silent|hidden|hides|contains|"
    r"letters?|initials?|prefix(?:es)?|suffix(?:es)?|starts?\s+with|ends?\s+with|"
    r"add(?:ing)?|remove|drop|replace|anagrams?|palindromes?|spelled|spelling|"
    r"before|after|compound|programming\s+languages?|keyboard\s+keys?)\b",
    re.IGNORECASE,
)
HIDDEN_CLAIM_PATTERN = re.compile(
    r"\b([A-Z0-9][A-Z0-9 '&-]{0,23})\s+(?:hides|contains)\s+([A-Z0-9]{3,})\b",
    re.IGNORECASE,
)
HOMOPHONE_EQUALS_PATTERN = re.compile(
    r"\b([A-Z][A-Z '&-]{1,23})\s*(?:=|sounds?\s+like)\s*([A-Z][A-Z '&-]{1,23})\b",
    re.IGNORECASE,
)
HOMOPHONE_OF_PATTERN = re.compile(r"\bhomophones?\s+of\s+([A-Za-z ]+)", re.IGNORECASE)
PHRASE_TARGET_PATTERNS = (
    (re.compile(r"^phrase_before_([a-z0-9_]+)$", re.IGNORECASE), "before"),
    (re.compile(r"^words_before_([a-z0-9_]+)$", re.IGNORECASE), "before"),
    (re.compile(r"^before_([a-z0-9_]+)$", re.IGNORECASE), "before"),
    (re.compile(r"^phrase_after_([a-z0-9_]+)$", re.IGNORECASE), "after"),
    (re.compile(r"^words_after_([a-z0-9_]+)$", re.IGNORECASE), "after"),
    (re.compile(r"^after_([a-z0-9_]+)$", re.IGNORECASE), "after"),
)
COMMON_WORD_SUPPLEMENTS = {
    "now",
    "overstep",
    "pets",
    "rats",
    "sunbed",
    "sunstone",
}
HOMOPHONE_TARGET_TO_DISPLAY = {
    "bear": {"bare"},
    "hare": {"hair"},
    "lynx": {"links"},
    "gnu": {"new", "knew"},
    "one": {"won"},
    "two": {"too", "to"},
    "four": {"for", "fore"},
    "eight": {"ate"},
    "beech": {"beach"},
    "fir": {"fur"},
    "plum": {"plumb"},
    "cherry": {"cheery"},
    "yew": {"you", "ewe"},
}
HOMOPHONE_DISPLAY_TO_TARGETS = {
    display: {
        target
        for target, displays in HOMOPHONE_TARGET_TO_DISPLAY.items()
        if display in displays
    }
    for display in {
        display
        for displays in HOMOPHONE_TARGET_TO_DISPLAY.values()
        for display in displays
    }
}
PREFERRED_HOMOPHONE_DISPLAY = {
    "gnu": "new",
    "two": "too",
    "yew": "ewe",
}
BROAD_CATEGORY_PATTERN = re.compile(
    r"^(?:words?|names?|verbs?|nouns?|adjectives?|things?|objects?|items?|"
    r"people|places|foods?|animals?|colors?|colours?|\d+\s*-?\s*letter\s+words?)$",
    re.IGNORECASE,
)
CATEGORY_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "after",
    "add",
    "added",
    "adding",
    "before",
    "can",
    "category",
    "common",
    "different",
    "each",
    "end",
    "find",
    "front",
    "for",
    "from",
    "have",
    "in",
    "inside",
    "make",
    "makes",
    "may",
    "might",
    "new",
    "of",
    "on",
    "or",
    "smaller",
    "that",
    "the",
    "things",
    "thing",
    "types",
    "type",
    "with",
    "word",
    "words",
    "you",
    "your",
}


def normalize_word(value: Any) -> str:
    """Normalize a playable tile label."""

    return " ".join(str(value).strip().upper().split())


def normalize_category(value: Any) -> str:
    """Normalize a category label while preserving readable case."""

    return " ".join(str(value).strip().split())


def normalize_metadata_key(value: Any) -> str:
    """Normalize hidden generator metadata into stable snake_case."""

    normalized = normalize_category(value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")[:80]


def normalize_lookup_word(value: Any) -> str:
    """Return lowercase letters only for local word-list lookups."""

    return re.sub(r"[^a-z]+", "", normalize_category(value).casefold())


def singularize_lookup_word(value: Any) -> str:
    """Return a simple singular lookup form for category targets."""

    word = normalize_lookup_word(value)
    if word.endswith("ies") and len(word) > 4:
        return f"{word[:-3]}y"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def preferred_homophone_display(target: str) -> str:
    """Return a stable example soundalike for a known homophone target."""

    displays = HOMOPHONE_TARGET_TO_DISPLAY.get(target, set())
    preferred = PREFERRED_HOMOPHONE_DISPLAY.get(target)
    if preferred in displays:
        return preferred
    return sorted(displays)[0] if displays else ""


@lru_cache(maxsize=1)
def common_words() -> set[str]:
    """Load a compact local lexicon for high-confidence wordplay leakage checks."""

    words = set(COMMON_WORD_SUPPLEMENTS)

    if not LOCAL_WORDLIST_PATH.exists():
        return words

    for line in LOCAL_WORDLIST_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        word = normalize_lookup_word(line)
        if 2 <= len(word) <= 24:
            words.add(word)

    return words


def is_common_lookup_word(value: Any) -> bool:
    """Return true when a derived word is common enough for puzzle leakage checks."""

    word = normalize_lookup_word(value)
    if not word:
        return False

    words = common_words()
    if word in words:
        return True

    if word.endswith("s") and len(word) > 3 and word[:-1] in words:
        return True

    if word.endswith("es") and len(word) > 4 and word[:-2] in words:
        return True

    return False


def puzzle_fingerprint(puzzle: dict[str, Any]) -> str:
    """Create a stable fingerprint for duplicate detection."""

    words = sorted(
        normalize_word(word)
        for group in puzzle.get("groups", [])
        for word in group.get("words", [])
    )
    categories = sorted(
        normalize_category(group.get("category", "")).lower()
        for group in puzzle.get("groups", [])
    )
    raw = "|".join(words) + "::" + "|".join(categories)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def group_text(group: dict[str, Any]) -> str:
    """Return normalized category and explanation text for quality heuristics."""

    return " ".join(
        part
        for part in (
            normalize_category(group.get("category", "")),
            normalize_category(group.get("explanation", "")),
        )
        if part
    )


def is_phrase_group(group: dict[str, Any]) -> bool:
    """Return true for simple before/after compound-word categories."""

    category = normalize_category(group.get("category", ""))
    explanation = normalize_category(group.get("explanation", ""))
    return bool(
        PHRASE_CATEGORY_PATTERN.search(category)
        or PHRASE_EXPLANATION_PATTERN.search(explanation)
    )


def is_generic_group(group: dict[str, Any]) -> bool:
    """Return true for very broad everyday-list categories."""

    category = normalize_category(group.get("category", ""))
    return bool(GENERIC_CATEGORY_PATTERN.search(category))


def is_color_shade_group(group: dict[str, Any]) -> bool:
    """Return true for color-shade list categories."""

    return bool(COLOR_SHADE_PATTERN.search(group_text(group)))


def is_can_verb_shared_property_group(group: dict[str, Any]) -> bool:
    """Return true for labels like 'Things you can X' or 'Things that can be X'."""

    category = normalize_category(group.get("category", ""))
    return bool(CAN_VERB_CATEGORY_PATTERN.search(category))


def is_plain_synonym_group(group: dict[str, Any]) -> bool:
    """Return true when a group is only a synonym/meaning set with no extra trick."""

    text = group_text(group)
    looks_like_synonyms = bool(
        PLAIN_SYNONYM_CATEGORY_PATTERN.search(text)
        or PLAIN_SYNONYM_EXPLANATION_PATTERN.search(text)
    )
    if not looks_like_synonyms:
        return False

    return not bool(EXTRA_TRICK_PATTERN.search(text))


def compact_letters(value: str) -> str:
    """Return uppercase letters and digits only for substring checks."""

    return re.sub(r"[^A-Z0-9]+", "", normalize_word(value))


def phrase_rule(group: dict[str, Any]) -> tuple[str, str] | None:
    """Return phrase direction and target for categories like 'Words after OVER'."""

    concept_key = normalize_metadata_key(group.get("concept_key", ""))
    for pattern, direction in PHRASE_TARGET_PATTERNS:
        match = pattern.match(concept_key)
        if match:
            target = normalize_lookup_word(match.group(1))
            return (direction, target) if target else None

    category = normalize_category(group.get("category", ""))

    match = re.match(r"^words?\s+before\s+['\"]?(.+?)['\"]?$", category, re.IGNORECASE)
    if match:
        target = normalize_lookup_word(match.group(1))
        return ("before", target) if target else None

    match = re.match(
        r"^words?\s+(?:after|that\s+follow)\s+['\"]?(.+?)['\"]?$",
        category,
        re.IGNORECASE,
    )
    if match:
        target = normalize_lookup_word(match.group(1))
        return ("after", target) if target else None

    match = re.match(r"^___\s+(.+?)$", category, re.IGNORECASE)
    if match:
        target = normalize_lookup_word(match.group(1))
        return ("before", target) if target else None

    match = re.match(r"^(.+?)\s+___$", category, re.IGNORECASE)
    if match:
        target = normalize_lookup_word(match.group(1))
        return ("after", target) if target else None

    return None


def phrase_combination(direction: str, target: str, word: str) -> str:
    """Return the compact phrase-completion candidate word."""

    lookup_word = normalize_lookup_word(word)
    return f"{lookup_word}{target}" if direction == "before" else f"{target}{lookup_word}"


def is_reversal_group(group: dict[str, Any]) -> bool:
    """Return true for categories where answer words reverse into new words."""

    concept_key = normalize_metadata_key(group.get("concept_key", ""))
    if "revers" in concept_key:
        return True

    text = group_text(group)
    return bool(re.search(r"\brevers(?:e|ed|es|ing)\b", text, re.IGNORECASE))


def hidden_claim_errors(group: dict[str, Any]) -> list[str]:
    """Verify simple explanation claims like 'MARSH hides MARS'."""

    explanation = normalize_category(group.get("explanation", ""))
    if not explanation or not re.search(r"\b(?:hides|contains)\b", explanation, re.IGNORECASE):
        return []

    group_words = {
        compact_letters(word): normalize_word(word)
        for word in group.get("words", [])
    }
    errors: list[str] = []

    for match in HIDDEN_CLAIM_PATTERN.finditer(explanation):
        claimed_word = compact_letters(match.group(1))
        hidden = compact_letters(match.group(2))
        if claimed_word.startswith("AND") and claimed_word[3:] in group_words:
            claimed_word = claimed_word[3:]
        display_word = group_words.get(claimed_word, match.group(1).strip().upper())

        if claimed_word not in group_words:
            continue

        if hidden not in claimed_word:
            errors.append(
                f"Group '{normalize_category(group.get('category', ''))}' says "
                f"{display_word} hides {hidden}, but it does not."
            )

    return errors


def homophone_claim_errors(group: dict[str, Any]) -> list[str]:
    """Verify simple homophone explanation claims and direction."""

    explanation = normalize_category(group.get("explanation", ""))
    text = group_text(group)
    if not explanation or not re.search(r"\b(?:homophones?|sounds?\s+like|=)\b", text, re.IGNORECASE):
        return []

    category = normalize_category(group.get("category", ""))
    group_words = {normalize_lookup_word(word): normalize_word(word) for word in group.get("words", [])}
    errors: list[str] = []

    for lookup_word, display_word in group_words.items():
        if lookup_word in HOMOPHONE_TARGET_TO_DISPLAY:
            suggested = preferred_homophone_display(lookup_word)
            errors.append(
                f"Group '{category}' uses target word {display_word} as a tile. "
                f"Homophone categories should display a soundalike such as {suggested.upper()}."
            )

    expected_family = ""
    category_match = HOMOPHONE_OF_PATTERN.search(category)
    if category_match:
        expected_family = singularize_lookup_word(category_match.group(1))

    for match in HOMOPHONE_EQUALS_PATTERN.finditer(explanation):
        left = normalize_lookup_word(match.group(1))
        right = normalize_lookup_word(match.group(2))
        if not left or not right or left not in group_words:
            continue

        allowed_targets = HOMOPHONE_DISPLAY_TO_TARGETS.get(left, set())
        allowed_displays_for_right = HOMOPHONE_TARGET_TO_DISPLAY.get(right, set())

        if left in HOMOPHONE_TARGET_TO_DISPLAY and right in HOMOPHONE_TARGET_TO_DISPLAY[left]:
            errors.append(
                f"Group '{category}' appears reversed: {group_words[left]} is the target word, "
                f"but the displayed answer should be a soundalike such as {right.upper()}."
            )
            continue

        if allowed_targets and right not in allowed_targets:
            errors.append(
                f"Group '{category}' says {group_words[left]} sounds like {right.upper()}, "
                "but the known homophone target does not match."
            )
        elif allowed_displays_for_right and left not in allowed_displays_for_right:
            errors.append(
                f"Group '{category}' should display a soundalike for {right.upper()}, "
                f"not {group_words[left]}."
            )

        if expected_family and expected_family in HOMOPHONE_TARGET_TO_DISPLAY:
            if right == expected_family and left not in HOMOPHONE_TARGET_TO_DISPLAY[expected_family]:
                errors.append(
                    f"Group '{category}' maps {group_words[left]} to {right.upper()}, "
                    "but that displayed word is not a known soundalike for the category target."
                )

    return errors


def category_tokens(category: str) -> set[str]:
    """Return significant normalized tokens from a category label."""

    raw_tokens = re.findall(r"[A-Za-z0-9]+", normalize_category(category).casefold())
    return {
        token
        for token in raw_tokens
        if len(token) >= 3 and token not in CATEGORY_TOKEN_STOPWORDS
    }


def generated_constraint_errors(groups: list[dict[str, Any]]) -> list[str]:
    """Return paper-inspired local constraint errors for generated puzzles."""

    errors: list[str] = []
    category_token_sets: list[tuple[str, set[str]]] = []
    can_verb_categories: list[str] = []

    for group in groups:
        category = normalize_category(group.get("category", ""))
        if not category:
            continue

        if BROAD_CATEGORY_PATTERN.match(category):
            errors.append(f"Generated category '{category}' is too broad.")

        if is_can_verb_shared_property_group(group):
            can_verb_categories.append(category)

        if str(group.get("difficulty", "")).strip().lower() == "tricky" and is_plain_synonym_group(group):
            errors.append(
                f"Generated purple category '{category}' is only a synonym/meaning set. "
                "Use plain synonym groups as yellow/green, or add a real wordplay mechanism."
            )

        tokens = category_tokens(category)
        category_token_sets.append((category, tokens))
        answer_words = {normalize_word(word) for word in group.get("words", [])}
        title_word_collisions = sorted(
            token.upper()
            for token in tokens
            if token.upper() in answer_words
        )
        if title_word_collisions:
            errors.append(
                f"Generated category '{category}' includes category word(s) as answers: "
                f"{', '.join(title_word_collisions)}."
            )

    for left_index, (left_category, left_tokens) in enumerate(category_token_sets):
        if not left_tokens:
            continue

        for right_category, right_tokens in category_token_sets[left_index + 1:]:
            overlap = sorted(left_tokens & right_tokens)
            if overlap:
                errors.append(
                    "Generated puzzle has thematically overlapping category labels: "
                    f"'{left_category}' and '{right_category}' both use {', '.join(overlap)}."
                )

    if len(can_verb_categories) > 1:
        errors.append(
            "Generated puzzle uses multiple same-frame can-verb categories: "
            f"{', '.join(can_verb_categories)}. Use at most one 'Things you can/can be X' "
            "style group per board."
        )

    return errors


def wordplay_leakage_findings(groups: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return hard wordplay errors separately from softer cross-category leaks."""

    if len(groups) != 4:
        return [], []

    hard_errors: list[str] = []
    leak_findings: list[str] = []
    board_words = [
        normalize_word(word)
        for group in groups
        for word in group.get("words", [])
    ]

    for group in groups:
        category = normalize_category(group.get("category", ""))
        group_words = {normalize_word(word) for word in group.get("words", [])}
        rule = phrase_rule(group)
        if not rule:
            continue

        direction, target = rule
        if not target:
            continue

        for word in board_words:
            if word in group_words:
                continue

            combination = phrase_combination(direction, target, word)
            if is_common_lookup_word(combination):
                leak_findings.append(
                    f"Board word {word} also fits phrase category '{category}' "
                    f"as {combination.upper()}, creating a cross-category leak."
                )

    for group in groups:
        if not is_reversal_group(group):
            continue

        category = normalize_category(group.get("category", ""))
        group_words = {normalize_word(word) for word in group.get("words", [])}

        for word in group_words:
            lookup_word = normalize_lookup_word(word)
            reversed_word = lookup_word[::-1]
            if lookup_word == reversed_word or not is_common_lookup_word(reversed_word):
                hard_errors.append(
                    f"Group '{category}' maps {word} to {reversed_word.upper()}, "
                    "which is not a supported common reversed word."
                )

        for word in board_words:
            if word in group_words:
                continue

            lookup_word = normalize_lookup_word(word)
            if len(lookup_word) < 3:
                continue

            reversed_word = lookup_word[::-1]
            if lookup_word != reversed_word and is_common_lookup_word(reversed_word):
                leak_findings.append(
                    f"Board word {word} also fits reversal category '{category}' "
                    f"as {reversed_word.upper()}, creating a cross-category leak."
                )

    return hard_errors, leak_findings


def wordplay_leakage_errors(groups: list[dict[str, Any]]) -> list[str]:
    """Reject broken wordplay and boards with multiple objective cross-fits."""

    hard_errors, leak_findings = wordplay_leakage_findings(groups)
    if len(leak_findings) >= 2:
        return [*hard_errors, *leak_findings]
    return hard_errors


def wordplay_leakage_warnings(groups: list[dict[str, Any]]) -> list[str]:
    """Warn about one minor cross-fit without rejecting an otherwise playable puzzle."""

    _, leak_findings = wordplay_leakage_findings(groups)
    if len(leak_findings) == 1:
        return leak_findings
    return []


def generated_quality_errors(puzzle: dict[str, Any]) -> list[str]:
    """Reject generated puzzles that are valid but too repetitive or bland."""

    groups = [group for group in puzzle.get("groups", []) if isinstance(group, dict)]
    if len(groups) != 4:
        return []

    errors: list[str] = []
    difficulties = tuple(str(group.get("difficulty", "")).strip().lower() for group in groups)
    if difficulties != GENERATED_DIFFICULTY_ORDER:
        errors.append(
            "Generated puzzles must order difficulties as easy, medium, hard, tricky."
        )

    phrase_indexes = [index for index, group in enumerate(groups) if is_phrase_group(group)]
    if len(phrase_indexes) > 1:
        errors.append("Generated puzzles may use at most one compound/before-after group.")

    generic_indexes = [index for index, group in enumerate(groups) if is_generic_group(group)]
    if len(generic_indexes) > 1:
        errors.append("Generated puzzles may use at most one broad everyday-list category.")

    for index in generic_indexes:
        if index >= 2:
            errors.append(
                "Generated blue/purple groups must be more interesting than broad everyday lists."
            )
            break

    if is_color_shade_group(groups[3]):
        errors.append("Generated purple groups must not be simple color-shade lists.")

    errors.extend(generated_constraint_errors(groups))
    errors.extend(wordplay_leakage_errors(groups))

    return errors


def normalize_puzzle(puzzle: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """Return a cleaned puzzle dictionary with normalized words and metadata."""

    normalized_groups = []
    for group in puzzle.get("groups", []):
        difficulty = str(group.get("difficulty", "medium")).strip().lower()
        if difficulty not in VALID_DIFFICULTIES:
            difficulty = "medium"

        normalized_group = {
            "category": normalize_category(group.get("category", "")),
            "difficulty": difficulty,
            "words": [normalize_word(word) for word in group.get("words", [])],
        }

        explanation = normalize_category(group.get("explanation", ""))
        if explanation:
            normalized_group["explanation"] = explanation

        mechanism_family = normalize_metadata_key(group.get("mechanism_family", ""))
        if mechanism_family:
            normalized_group["mechanism_family"] = mechanism_family

        concept_key = normalize_metadata_key(group.get("concept_key", ""))
        if concept_key:
            normalized_group["concept_key"] = concept_key

        normalized_groups.append(normalized_group)

    cleaned = {
        "groups": normalized_groups,
        "source": source or puzzle.get("source", "unknown"),
    }

    difficulty_mode = str(puzzle.get("difficulty_mode", "")).strip().lower()
    if difficulty_mode in {"easy", "hard"}:
        cleaned["difficulty_mode"] = difficulty_mode

    decoy = puzzle.get("decoy")
    if isinstance(decoy, dict):
        cleaned["decoy"] = {
            "label": normalize_category(decoy.get("label", "")),
            "words": [normalize_word(word) for word in decoy.get("words", [])],
            "why_false": normalize_category(decoy.get("why_false", "")),
        }

    if puzzle.get("agent_trace"):
        cleaned["agent_trace"] = puzzle["agent_trace"]

    cleaned["id"] = str(puzzle.get("id") or puzzle_fingerprint(cleaned))
    return cleaned


def validate_puzzle(
    puzzle: dict[str, Any],
    *,
    require_nyt_blocklist: bool = False,
    enforce_generated_quality: bool = True,
    require_generated_metadata: bool = False,
) -> dict[str, Any]:
    """Validate a single puzzle and return errors plus non-blocking warnings."""

    errors: list[str] = []
    warnings: list[str] = []
    groups = puzzle.get("groups")

    if not isinstance(groups, list):
        return {"ok": False, "errors": ["Puzzle must include a groups list."], "warnings": []}

    if len(groups) != 4:
        errors.append("Puzzle must have exactly 4 groups.")

    all_words: list[str] = []
    categories: list[str] = []

    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            errors.append(f"Group {index} must be an object.")
            continue

        category = normalize_category(group.get("category", ""))
        explanation = normalize_category(group.get("explanation", ""))
        categories.append(category.lower())

        if not category:
            errors.append(f"Group {index} is missing a category.")
        elif len(category) > 70:
            warnings.append(f"Category '{category}' is long for the reveal area.")

        if explanation:
            if len(explanation) > 180:
                errors.append(f"Group '{category or index}' explanation is too long.")
            if DRAFT_EXPLANATION_PATTERN.search(explanation):
                errors.append(
                    f"Group '{category or index}' explanation contains draft or self-correction text."
                )
            errors.extend(hidden_claim_errors(group))
            errors.extend(homophone_claim_errors(group))

        difficulty = str(group.get("difficulty", "")).strip().lower()
        if difficulty not in VALID_DIFFICULTIES:
            errors.append(f"Group '{category or index}' has an invalid difficulty.")

        if require_generated_metadata and puzzle.get("source") == GENERATED_SOURCE:
            for metadata_field in ("mechanism_family", "concept_key"):
                if metadata_field not in group:
                    errors.append(
                        f"Generated group '{category or index}' is missing {metadata_field} metadata."
                    )

        for metadata_field in ("mechanism_family", "concept_key"):
            metadata_value = group.get(metadata_field)
            if metadata_value is None:
                continue

            normalized_metadata = normalize_metadata_key(metadata_value)
            if not normalized_metadata:
                errors.append(f"Group '{category or index}' has an empty {metadata_field}.")
            elif len(normalized_metadata) > 64:
                warnings.append(
                    f"Group '{category or index}' has a long {metadata_field} metadata value."
                )

        words = group.get("words")
        if not isinstance(words, list):
            errors.append(f"Group '{category or index}' must include a words list.")
            continue

        if len(words) != 4:
            errors.append(f"Group '{category or index}' must have exactly 4 words.")

        for word in words:
            normalized = normalize_word(word)
            all_words.append(normalized)

            if not normalized:
                errors.append(f"Group '{category or index}' contains an empty word.")
            elif not WORD_PATTERN.match(normalized):
                errors.append(f"Word '{normalized}' has unsupported characters or length.")
            elif len(normalized) > 14:
                warnings.append(f"Word '{normalized}' may be tight on small screens.")

    if len(all_words) != 16:
        errors.append("Puzzle must contain exactly 16 words total.")

    unique_words = set(all_words)
    if len(unique_words) != len(all_words):
        repeated = sorted(word for word in unique_words if all_words.count(word) > 1)
        errors.append(f"Puzzle contains duplicate words: {', '.join(repeated)}.")

    normalized_categories = [category for category in categories if category]
    if len(set(normalized_categories)) != len(normalized_categories):
        errors.append("Puzzle contains duplicate category labels.")

    difficulty_mode = str(puzzle.get("difficulty_mode", "")).strip().lower()
    decoy = puzzle.get("decoy")

    if difficulty_mode == "easy" and decoy:
        errors.append("Easy puzzles must not include an intentional decoy.")

    if difficulty_mode == "hard":
        if not isinstance(decoy, dict):
            errors.append("Hard puzzles must include one intentional decoy.")
        else:
            decoy_words = [normalize_word(word) for word in decoy.get("words", [])]
            if len(decoy_words) not in {3, 4}:
                errors.append("Hard puzzle decoys must contain 3 or 4 words.")

            missing_words = sorted(word for word in decoy_words if word not in unique_words)
            if missing_words:
                errors.append(f"Decoy words are not on the board: {', '.join(missing_words)}.")

            real_groups = {
                tuple(sorted(normalize_word(word) for word in group.get("words", [])))
                for group in groups
                if isinstance(group, dict)
            }
            if tuple(sorted(decoy_words)) in real_groups:
                errors.append("Decoy cannot exactly match a real answer group.")

            if not normalize_category(decoy.get("label", "")):
                errors.append("Decoy must include a label.")

            if not normalize_category(decoy.get("why_false", "")):
                errors.append("Decoy must explain why it is false.")

    if enforce_generated_quality and puzzle.get("source") == GENERATED_SOURCE:
        errors.extend(generated_quality_errors(puzzle))
        warnings.extend(wordplay_leakage_warnings(groups))

    nyt_result = check_puzzle_against_blocklist(
        puzzle,
        require_ready=require_nyt_blocklist,
    )
    errors.extend(nyt_result["errors"])
    warnings.extend(nyt_result["warnings"])

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def validate_bank(puzzles: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a puzzle bank and check duplicate fingerprints."""

    errors: list[str] = []
    fingerprints: dict[str, str] = {}
    prior_puzzles: list[dict[str, Any]] = []

    for index, puzzle in enumerate(puzzles, start=1):
        result = validate_puzzle(puzzle)
        if not result["ok"]:
            errors.extend(f"Puzzle {index}: {error}" for error in result["errors"])

        from agents.bank_memory import build_bank_memory, repeat_errors

        memory = build_bank_memory(prior_puzzles)
        bank_repeat_errors = repeat_errors(puzzle, memory)
        errors.extend(f"Puzzle {index}: {error}" for error in bank_repeat_errors)

        fingerprint = puzzle_fingerprint(puzzle)
        if fingerprint in fingerprints:
            errors.append(
                f"Puzzle {index} duplicates puzzle {fingerprints[fingerprint]} "
                f"(fingerprint {fingerprint})."
            )
        else:
            fingerprints[fingerprint] = str(index)

        prior_puzzles.append(puzzle)

    return {"ok": not errors, "errors": errors}
