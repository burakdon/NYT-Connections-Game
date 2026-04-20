"""Prompt templates for the puzzle generation pipeline."""

CATEGORY_STYLES = """Category styles to choose from:
- "Synonyms or Slang" (e.g., WAYS TO SAY HELLO)
- "Wordplay" (e.g., ___FISH)
- "Fill in the blank" (e.g., FIRE ___)
- "Hidden pattern" (e.g., WORDS THAT ARE ALSO COLORS)
- "Common property" (e.g., THINGS WITH WHEELS)
- "Pop culture" (e.g., TAYLOR SWIFT ALBUMS)
"""

GROUP_CREATION_SYSTEM = """You are an expert puzzle designer creating word groups for an NYT Connections-style puzzle.
You produce creative, clever word groupings that are challenging but fair.

{category_styles}

IMPORTANT: Provide exactly 8 candidate words for your category. The best 4 will be selected algorithmically.

Respond with valid JSON only:
{{"category": "CATEGORY NAME IN CAPS", "words": ["WORD1", "WORD2", "WORD3", "WORD4", "WORD5", "WORD6", "WORD7", "WORD8"]}}
"""

GROUP_CREATION_USER = """Create a word group for a Connections puzzle.
{story_injection}
{previous_groups}
Generate a category name and 8 candidate words. All words must be single words or very short phrases, ALL CAPS.
The category should be a clear, concise description that connects exactly the words in the group.
"""

STORY_INJECTION_TEMPLATE = """First, write a short story (2-3 sentences) using these words: {seed_words}.
Then use that story as inspiration for creating your category. The category does NOT need to include those seed words."""

FALSE_GROUP_SYSTEM = """You are an expert puzzle designer. Create a "root group" — a plausible category with 4 words.
Each word should have at least one alternate meaning or usage beyond its meaning in this category.

{category_styles}

Respond with valid JSON:
{{"category": "CATEGORY NAME", "words": ["W1", "W2", "W3", "W4"], "alternate_meanings": {{"W1": "alternate meaning", "W2": "alternate meaning", "W3": "alternate meaning", "W4": "alternate meaning"}}}}
"""

FALSE_GROUP_USER = """Create a root group for the false-group puzzle generation method.
The words should each have clear alternate meanings that could inspire entirely different categories.
{previous_roots}"""

ALTERNATE_MEANING_SYSTEM = """You are an expert puzzle designer. Given a word and its alternate meaning,
create a new category that uses this alternate meaning as inspiration.

{category_styles}

Respond with valid JSON:
{{"category": "CATEGORY NAME IN CAPS", "words": ["WORD1", "WORD2", "WORD3", "WORD4", "WORD5", "WORD6", "WORD7", "WORD8"]}}
"""

ALTERNATE_MEANING_USER = """The word "{word}" has this alternate meaning: "{meaning}".
Create a new category inspired by this alternate meaning. Provide 8 candidate words.
{previous_groups}
Do NOT include the word "{word}" itself in your candidate list."""

EDITOR_SYSTEM = """You are a puzzle editor reviewing a Connections puzzle.
Check that each category name accurately describes its 4 words.
If a category name is inaccurate, suggest a better one.

Respond with valid JSON:
{{"valid": true/false, "changes": [{{"old_category": "...", "new_category": "...", "reason": "..."}}], "notes": "overall assessment"}}
"""

EDITOR_USER = """Review this Connections puzzle for accuracy:

{puzzle_description}

For each group, verify:
1. The category name accurately describes all 4 words
2. No word fits better in a different group
3. The category names are in the style of NYT Connections (concise, ALL CAPS)
"""

LLM_SOLVER_SYSTEM = """You are solving an NYT Connections puzzle. Identify and find 4 groups of 4 words from the 16 words given.

Think step by step:
1. Look for obvious groupings first
2. Consider alternate meanings and wordplay
3. Remember: every word belongs to exactly one group
4. Each group has exactly 4 words

Respond with valid JSON:
{{"groups": [{{"category": "DESCRIPTION", "words": ["W1", "W2", "W3", "W4"]}}, ...], "reasoning": "your step-by-step reasoning"}}
"""

LLM_SOLVER_USER = """Solve this Connections puzzle. Find 4 groups of 4 words.

Words: {words}

Identify the 4 groups and explain your reasoning."""

DIFFICULTY_RANKING_SYSTEM = """You are ranking the difficulty of word groups in a Connections puzzle.
Rank from easiest (1) to hardest (4).

Respond with valid JSON:
{{"ranking": [{{"category": "...", "difficulty": 1}}, ...]}}
"""

DIFFICULTY_RANKING_USER = """Rank these groups from easiest (1) to hardest (4):

{groups_description}
"""


def format_group_creation(seed_words=None, previous_groups=None):
    story = ""
    if seed_words:
        story = STORY_INJECTION_TEMPLATE.format(seed_words=", ".join(seed_words))

    prev = ""
    if previous_groups:
        lines = []
        for g in previous_groups:
            lines.append(f"- {g['category']}: {', '.join(g['words'])}")
        prev = "Already created groups (do NOT reuse any words):\n" + "\n".join(lines)

    system = GROUP_CREATION_SYSTEM.format(category_styles=CATEGORY_STYLES)
    user = GROUP_CREATION_USER.format(story_injection=story, previous_groups=prev)
    return system, user


def format_false_group(previous_roots=None):
    prev = ""
    if previous_roots:
        prev = "Previous root groups (create something different):\n" + \
               "\n".join(f"- {r}" for r in previous_roots)
    system = FALSE_GROUP_SYSTEM.format(category_styles=CATEGORY_STYLES)
    user = FALSE_GROUP_USER.format(previous_roots=prev)
    return system, user


def format_alternate_meaning(word, meaning, previous_groups=None):
    prev = ""
    if previous_groups:
        lines = [f"- {g['category']}: {', '.join(g['words'])}" for g in previous_groups]
        prev = "Already created groups (do NOT reuse words):\n" + "\n".join(lines)
    system = ALTERNATE_MEANING_SYSTEM.format(category_styles=CATEGORY_STYLES)
    user = ALTERNATE_MEANING_USER.format(
        word=word, meaning=meaning, previous_groups=prev
    )
    return system, user


def format_editor(groups):
    lines = []
    for g in groups:
        lines.append(f"Category: {g['category']}\nWords: {', '.join(g['words'])}")
    desc = "\n\n".join(lines)
    return EDITOR_SYSTEM, EDITOR_USER.format(puzzle_description=desc)


def format_solver(words):
    return LLM_SOLVER_SYSTEM, LLM_SOLVER_USER.format(words=", ".join(words))


def format_difficulty_ranking(groups):
    lines = [f"- {g['category']}: {', '.join(g['words'])}" for g in groups]
    desc = "\n".join(lines)
    return DIFFICULTY_RANKING_SYSTEM, DIFFICULTY_RANKING_USER.format(groups_description=desc)
