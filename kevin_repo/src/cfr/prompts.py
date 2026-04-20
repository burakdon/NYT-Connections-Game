"""Prompt templates for CFR (Category-First Retrieval) — Mode B only.

Unlike src/generator/prompts.py which has 4+ prompt templates for iterative
generation, CFR needs only ONE: a batched prompt asking for 4 category names
in a single API call.
"""

SYSTEM_PROMPT_BATCH_CATEGORIES = """You are an expert at creating New York Times Connections puzzles.

Your task: generate 4 DIVERSE category names for a single puzzle. Each category
will have 4 words that share some property. The 4 categories should be:

- UNRELATED to each other (no semantic overlap)
- Specific enough to have a clear solution (not "THINGS")
- Match NYT Connections style, drawn from these types:
  * Synonyms or slang (e.g., "WAYS TO SAY HELLO")
  * Wordplay / fill-in-the-blank (e.g., "___FISH", "FIRE ___")
  * Hidden pattern (e.g., "WORDS CONTAINING COLORS")
  * Common property (e.g., "THINGS WITH WHEELS")
  * Pop culture (e.g., "TAYLOR SWIFT ALBUMS")

Return JSON with exactly this structure:
{"categories": ["CATEGORY 1", "CATEGORY 2", "CATEGORY 3", "CATEGORY 4"]}

All category names in ALL CAPS. No extra commentary."""


USER_PROMPT_BATCH_CATEGORIES = """Generate 4 diverse, unrelated Connections puzzle category names.

Inspiration words (optional starting point, do not have to use):
{seed_words}

Return only the JSON object."""


def format_batch_categories(seed_words: list[str] = None) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for generating 4 category names."""
    seed_str = ", ".join(seed_words) if seed_words else "(none)"
    return (
        SYSTEM_PROMPT_BATCH_CATEGORIES,
        USER_PROMPT_BATCH_CATEGORIES.format(seed_words=seed_str),
    )
