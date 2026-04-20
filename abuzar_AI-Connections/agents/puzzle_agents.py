"""Claude-backed multi-agent puzzle factory."""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agents.bank_memory import avoid_instructions, build_bank_memory, repeat_errors
from agents.claude_client import call_claude, load_env_files
from agents.mechanism_library import format_inspiration_guidance, format_mechanism_guidance
from agents.nyt_guard import blocklist_status
from agents.puzzle_validator import (
    DRAFT_EXPLANATION_PATTERN,
    WORD_PATTERN,
    generated_constraint_errors,
    normalize_category,
    normalize_puzzle,
    normalize_word,
    validate_puzzle,
)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a Claude response."""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("Response did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])

    raise ValueError("Response contained an incomplete JSON object.")


def normalize_generation_mode(difficulty: str) -> str:
    """Reduce user-facing generation choices to the MVP modes."""

    return "hard" if difficulty.strip().lower() == "hard" else "easy"


QUALITY_RULES = """
Quality bar:
- Do not make a puzzle out of four safe everyday lists. Avoid defaulting to labels like "Things on a Desk", "Animals at the Zoo", "Things You See at the Beach", "Parts of a Shoe", or repeated color-shade sets.
- Use at most one simple compound/phrase category per puzzle.
- Purple may be a simple compound category such as "___ X", "X ___", "Words before X", or "Words after X" only sometimes; do not treat it as the default hardest mechanism.
- Across a generated bank, aim for no more than about one purple compound/phrase puzzle in four.
- Purple should rotate among fair hidden mechanisms: sound, spelling, hidden letters, word endings, double meanings, second identities, or non-obvious shared properties.
- Good purple examples include common words that are also programming languages, keyboard keys that are also common words, things that can be rolled, or words that share a hidden spelling/sound pattern.
- Use at most one same-frame "Things you can X" / "Things that can be X" shared-property category per puzzle. Two of these in one board feel repetitive even when the verbs differ.
- Purple must not be only a plain synonym or definition group, such as "Verbs meaning to look carefully". Put ordinary synonym sets in yellow/green, or give purple a real extra mechanism such as spelling, sound, transformation, hidden letters, or second identity.
- Avoid vague hard categories that are just obscure trivia, random lists, or common object groups.
- Do not use color-shade categories as purple. Use color shades sparingly, and only if they make the whole puzzle better.
- Blue and purple should have a real aha, not just "common objects found in one place."
- Category labels must be specific, and a significant word from the label must not appear as one of that group's answer words.
- The four category labels in a puzzle should be thematically distinct; avoid two labels that share the same important noun, such as two face/body categories.
- For shared-property categories, all four words must pass the same natural phrase test. Reject groups where one answer only fits technically or through awkward wording, such as "rocking boat" beside "rocking chair."
- For objective wordplay groups, check every board word against the rule, not only the four intended answers. Prefer repairing even one off-category fit. Reject or repair when multiple off-category words fit, or when a complete false group is formed. Example: STEP leaks into an OVER- group as OVERSTEP; FLOW leaks into a reversal group as WOLF.
- Before returning JSON, silently verify every word against its category. If one word fails, replace it in the "words" array. Never leave the bad word in place and explain the correction.
- Explanations must contain only the final answer logic. They must not include alternatives, scratch work, "wait", "actually", "correct set", "replace", or rejected mappings.
- For homophone categories, the displayed answer must be the soundalike, not the target category item. Example: use NEW for gnu; do not use GNU as a homophone of animals.
""".strip()


CRITICAL_CRITIC_ISSUE_PATTERN = re.compile(
    r"\b(?:fully valid alternate|valid alternate group|alternate solution|"
    r"alternate answer|fully valid false group|unsolvable|not solvable|"
    r"cannot be solved|zero correct|0\s*/\s*4|no correct groups|"
    r"confidence below 0\.5|confidence under 0\.5)\b",
    re.IGNORECASE,
)
SOLVER_CONFIDENCE_PATTERN = re.compile(
    r"\b(?:solver\s+)?confidence(?:\s+(?:only|is|=|:|of))?\s*"
    r"([01](?:\.\d+)?)\b",
    re.IGNORECASE,
)
TREE_OF_THOUGHT_STRATEGIES = {"tot", "tree-of-thought", "tree_of_thought", "tot-medium", "medium-tot"}
GENERATOR_AGENT_NAMES = {
    "Category Scout",
    "Wordsmith",
    "Category Thought Agent",
    "Board Builder Agent",
    "Misdirection Agent",
    "Editor Agent",
}
REVIEWER_AGENT_NAMES = {"Solver Agent", "Critic Agent"}
DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2, "tricky": 3}
RISK_TEXT_PATTERN = re.compile(
    r"\b(?:placeholder|needs|near-rhyme|uncertain|weak|technical|stretch|swap|"
    r"replacing?|dropping|rebuild(?:ing)?|does\s+not|not\s+clean(?:ly)?|"
    r"scrap(?:ping)?|less\s+common|could\s+confuse|may\s+object)\b",
    re.IGNORECASE,
)
TRICKY_MECHANISMS = {"sound", "spelling", "double_identity", "transformation", "phrase_completion"}
STARTER_MECHANISMS = {"semantic", "semantic_set", "specific_set", "light_trivia"}


@dataclass
class AgentEvent:
    agent: str
    status: str
    summary: str
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "status": self.status,
            "summary": self.summary,
            "duration_seconds": round(self.duration_seconds, 2),
            "details": self.details,
        }


class MultiAgentPuzzleFactory:
    """Coordinate specialized Claude prompts into a puzzle generation run."""

    def __init__(
        self,
        model: str | None = None,
        existing_puzzles: list[dict[str, Any]] | None = None,
        generator_model: str | None = None,
        reviewer_model: str | None = None,
    ):
        load_env_files()
        self.model = model
        self.generator_model = generator_model or os.environ.get("CLAUDE_GENERATOR_MODEL") or None
        self.reviewer_model = reviewer_model or os.environ.get("CLAUDE_REVIEWER_MODEL") or None
        self.existing_puzzles = existing_puzzles or []
        self.bank_memory = build_bank_memory(self.existing_puzzles)
        self.bank_avoid_text = avoid_instructions(self.bank_memory)
        self.mechanism_guidance = ""
        self.inspiration_guidance = ""
        self.trace: list[AgentEvent] = []

    def _model_for_agent(self, name: str) -> str | None:
        """Return the configured model for an agent role."""

        if self.model:
            return self.model

        if name in REVIEWER_AGENT_NAMES:
            return self.reviewer_model

        if name in GENERATOR_AGENT_NAMES:
            return self.generator_model

        return None

    def _run_agent(
        self,
        *,
        name: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        started = time.time()
        text = ""
        parsed: dict[str, Any] | None = None
        parse_error: Exception | None = None
        json_retry_count = 0
        selected_model = self._model_for_agent(name)

        for attempt in range(2):
            retry_user = user
            retry_temperature = temperature
            if attempt:
                retry_temperature = min(temperature, 0.2)
                retry_user = (
                    f"{user}\n\n"
                    "Your previous response was not parseable JSON. Return exactly one valid "
                    "JSON object matching the requested schema. Do not include prose, markdown, "
                    "scratch work, or code fences."
                )

            text = call_claude(
                system=system,
                user=retry_user,
                max_tokens=max_tokens,
                temperature=retry_temperature,
                model=selected_model,
            )

            try:
                parsed = extract_json_object(text)
                json_retry_count = attempt
                parse_error = None
                break
            except ValueError as error:
                parse_error = error

        duration = time.time() - started

        if parsed is None:
            message = str(parse_error or "Response did not contain a JSON object.")
            self.trace.append(
                AgentEvent(
                    agent=name,
                    status="error",
                    summary=f"Invalid JSON response after retry: {message}",
                    duration_seconds=duration,
                    details={
                        "raw_excerpt": text[:900],
                        "json_retry_count": 1,
                        "model": selected_model or "default",
                    },
                )
            )
            raise ValueError(f"{name} returned invalid JSON after retry: {message}") from parse_error

        summary = parsed.get("summary")
        if not summary:
            keys = ", ".join(sorted(parsed.keys()))
            summary = f"Returned JSON fields: {keys}"

        self.trace.append(
            AgentEvent(
                agent=name,
                status="complete",
                summary=str(summary),
                duration_seconds=duration,
                details={
                    "raw_excerpt": text[:900],
                    "json_retry_count": json_retry_count,
                    "model": selected_model or "default",
                },
            )
        )
        return parsed

    def generate_batch(
        self,
        *,
        target_count: int = 4,
        theme: str = "",
        difficulty: str = "easy",
        max_review_rounds: int = 2,
        strategy: str = "standard",
    ) -> dict[str, Any]:
        """Generate, review, repair, and return accepted puzzles."""

        self.trace = []
        target_count = max(1, min(target_count, 12))
        theme_text = theme.strip() or "general knowledge, accessible vocabulary"
        difficulty_text = normalize_generation_mode(difficulty)
        nyt_status = blocklist_status()

        if not nyt_status["ready"]:
            raise RuntimeError(
                "NYT guard blocklist is empty. Build data/nyt_blocklist.json before generating puzzles."
            )

        self.mechanism_guidance = format_mechanism_guidance(
            self.bank_memory,
            difficulty=difficulty_text,
        )
        self.inspiration_guidance = format_inspiration_guidance()
        self._record_bank_memory_event()

        strategy_text = strategy.strip().lower()
        if strategy_text in TREE_OF_THOUGHT_STRATEGIES:
            refined = self._tree_of_thought_candidates(target_count, theme_text, difficulty_text)
        else:
            ideas = self._category_scout(target_count, theme_text, difficulty_text)
            candidates = self._wordsmith(target_count, theme_text, difficulty_text, ideas)
            refined = self._misdirection_agent(candidates, difficulty_text)

        if not refined:
            self.trace.append(
                AgentEvent(
                    agent="Orchestrator",
                    status="complete",
                    summary="Final decision: accepted 0 puzzle(s), no candidate puzzles survived generation/pruning.",
                    duration_seconds=0,
                    details={"accepted": 0, "rejected": 1},
                )
            )
            return {
                "accepted": [],
                "rejected": [
                    {
                        "errors": ["No candidate puzzles survived generation or local pruning."],
                        "stage": "generation_pruning",
                    }
                ],
                "trace": [event.to_dict() for event in self.trace],
            }

        accepted, rejected = self._review_loop(
            refined,
            difficulty=difficulty_text,
            target_count=target_count,
            max_rounds=max(1, min(max_review_rounds, 3)),
        )

        self.trace.append(
            AgentEvent(
                agent="Orchestrator",
                status="complete",
                summary=(
                    f"Final decision: accepted {len(accepted)} puzzle(s), "
                    f"rejected {len(rejected)} candidate(s)."
                ),
                duration_seconds=0,
                details={"accepted": len(accepted), "rejected": len(rejected)},
            )
        )

        return {
            "accepted": accepted,
            "rejected": rejected,
            "trace": [event.to_dict() for event in self.trace],
        }

    def _category_scout(self, target_count: int, theme: str, difficulty: str) -> list[dict[str, Any]]:
        system = (
            "You are Category Scout, a puzzle-design agent for a Connections-style "
            "word grouping game. You invent clean category concepts, not final puzzles."
        )
        user = f"""
Return JSON only.

Goal: create category ideas for {target_count} playable puzzles.
Theme: {theme}
Puzzle mode: {difficulty}

{self.bank_avoid_text}

{QUALITY_RULES}

{self.mechanism_guidance}

{self.inspiration_guidance}

Rules:
- Avoid NYT branding or copied puzzles.
- Mix semantic categories, phrase categories, object sets, and wordplay.
- Use accessible English vocabulary.
- Categories should support exactly 4 answer words.
- Avoid categories that are only arbitrary lists.
- Category labels should feel like polished answer reveals, not clue sentences.
- Prefer concise reveal labels: "Things with scales", "Classroom items", "Words before board", "Silent first letters".
- Avoid awkward sentence labels: "Can have scales", "Things that are used in schools", "Words that have silent first letters".
- Avoid anagram categories for now unless all four anagrams are real, common playable words and the mapping is exact.
- Never include draft language, alternatives, corrections, or self-revisions in explanations.
- Every final puzzle must order its groups as yellow, green, blue, purple: easiest, slightly harder, hard, hardest.
- Green should be meaningfully less obvious than yellow.
- Purple should usually be the most abstract, wordplay-based, or hardest-to-notice category.
- For easy mode, propose clean categories with low overlap and no intentional decoy.
- For hard mode, propose clean real categories that could support exactly one fair decoy later.

Return this exact shape:
{{
  "summary": "one short sentence",
  "ideas": [
    {{
      "category": "category label",
      "mechanism_family": "mechanism family id",
      "concept_key": "stable_snake_case_concept_key",
      "difficulty": "easy | medium | hard | tricky",
      "description": "why the category is fair",
      "sample_words": ["WORD", "WORD", "WORD", "WORD"]
    }}
  ]
}}
"""
        payload = self._run_agent(name="Category Scout", system=system, user=user, temperature=0.8)
        return payload.get("ideas", [])

    def _wordsmith(
        self,
        target_count: int,
        theme: str,
        difficulty: str,
        ideas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        system = (
            "You are Wordsmith, a puzzle construction agent. You turn category ideas "
            "into complete 16-word puzzles with four hidden groups."
        )
        user = f"""
Return JSON only.

Create {target_count + 1} candidate puzzles. Each puzzle must have exactly 4 groups.
Theme: {theme}
Puzzle mode: {difficulty}
Category ideas you may use or adapt:
{json.dumps(ideas, indent=2)}

{self.bank_avoid_text}

{QUALITY_RULES}

{self.mechanism_guidance}

{self.inspiration_guidance}

Puzzle rules:
- No puzzle title.
- Exactly 4 groups per puzzle.
- Exactly 4 words per group.
- Exactly 16 unique playable words per puzzle.
- Words should be short enough for cards.
- Do not include the same answer word twice in a puzzle.
- Include one short explanation per group.
- Include hidden metadata for every group: "mechanism_family" and "concept_key".
- Metadata is for local validation only; it is not a category label and should not be shown to players.
- concept_key must identify the underlying concept, not the wording. For example, "Words before board" and "___ Board" both use phrase_before_board.
- Do not include offensive, adult, or highly obscure content.
- Return groups in exact yellow, green, blue, purple reveal order.
- Yellow should be straightforward. Green should be a little harder than yellow. Blue should be hard. Purple should be the hardest.
- Category labels should be polished answer reveals, not awkward clue sentences.
- Prefer labels like "Things with scales", "Classroom items", "Words before board", "Silent first letters".
- Avoid labels like "Can have scales", "Things that are used in schools", "Words that have silent first letters".
- Avoid anagram categories for now unless every shown word is a real common word and maps cleanly to exactly one answer.
- Explanations must be final polished sentences only. Do not include words like "corrected", "replacing", "final", "settled", "conflict", or scratch alternatives.
- If you notice a bad mapping while drafting, fix the actual "words" list before returning JSON. Do not describe the mistake.
- Easy mode: create clean categories with no intentional decoy. Use "difficulty_mode": "easy" and "decoy": null.
- Hard mode: create clean true categories that can support one fair decoy later. Use "difficulty_mode": "hard"; the Misdirection Agent may add the decoy.

Return this exact shape:
{{
  "summary": "one short sentence",
  "puzzles": [
    {{
      "difficulty_mode": "{difficulty}",
      "decoy": null,
      "groups": [
        {{
          "category": "category revealed after solving",
          "difficulty": "easy | medium | hard | tricky",
          "mechanism_family": "mechanism family id",
          "concept_key": "stable_snake_case_concept_key",
          "words": ["WORD", "WORD", "WORD", "WORD"],
          "explanation": "short fair explanation"
        }}
      ]
    }}
  ]
}}
"""
        payload = self._run_agent(name="Wordsmith", system=system, user=user, temperature=0.9)
        return payload.get("puzzles", [])

    def _tree_of_thought_candidates(
        self,
        target_count: int,
        theme: str,
        difficulty: str,
    ) -> list[dict[str, Any]]:
        """Generate category thoughts, prune them locally, then assemble boards."""

        thoughts = self._thought_pool_agent(target_count, theme, difficulty)
        screened_thoughts, rejected_thoughts = self._screen_group_thoughts(thoughts)
        ranked_thoughts, thought_scores = self._rank_group_thoughts(screened_thoughts, target_count)
        self.trace.append(
            AgentEvent(
                agent="ToT Local Pruner",
                status="complete",
                summary=(
                    f"Kept {len(ranked_thoughts)} category thought(s), "
                    f"rejected {len(rejected_thoughts)} before board assembly."
                ),
                duration_seconds=0,
                details={
                    "kept": len(ranked_thoughts),
                    "rejected": rejected_thoughts[:12],
                },
            )
        )
        self.trace.append(
            AgentEvent(
                agent="ToT Thought Ranker",
                status="complete",
                summary="Ranked surviving thoughts by difficulty lane, mechanism fit, and stated risk.",
                duration_seconds=0,
                details={"top_scores": thought_scores[:12]},
            )
        )

        if len(ranked_thoughts) < 4:
            return []

        candidate_count = min(6, max(target_count + 2, target_count * 2))
        if len(ranked_thoughts) <= 4:
            candidate_count = 1
        elif len(ranked_thoughts) <= 7:
            candidate_count = min(candidate_count, 2)
        candidates = self._board_builder_agent(
            target_count,
            theme,
            difficulty,
            ranked_thoughts,
            candidate_count=candidate_count,
        )
        screened_boards, rejected_boards = self._screen_candidate_boards(
            candidates,
            target_count,
            difficulty=difficulty,
        )
        self.trace.append(
            AgentEvent(
                agent="ToT Board Pruner",
                status="complete",
                summary=(
                    f"Kept {len(screened_boards)} candidate board(s), "
                    f"rejected {len(rejected_boards)} before polish."
                ),
                duration_seconds=0,
                details={"rejected": rejected_boards[:8]},
            )
        )

        if not screened_boards:
            return []

        return self._misdirection_agent(screened_boards, difficulty)

    def _thought_pool_agent(self, target_count: int, theme: str, difficulty: str) -> list[dict[str, Any]]:
        thought_count = max(18, min(24, target_count * 10 + 8))
        system = (
            "You are Category Thought Agent. You generate many independent, "
            "self-contained category-group thoughts for a Connections-style puzzle. "
            "You do not assemble full boards."
        )
        user = f"""
Return JSON only.

Generate {thought_count} independent category thoughts. Each thought is one possible answer group of exactly four words.
Theme: {theme}
Puzzle mode: {difficulty}

{self.bank_avoid_text}

{QUALITY_RULES}

{self.mechanism_guidance}

{self.inspiration_guidance}

Tree-of-thought rules:
- Generate more category thoughts than we need so the local pruner can discard weak or repeated branches.
- Each thought must stand alone as a clean four-word group.
- Use varied mechanism families. Do not make most thoughts semantic lists.
- Prefer thoughts with specific labels and a clear test for all four words.
- Do not assemble a full puzzle here.
- Do not include draft notes, rejected alternatives, or self-corrections.
- For homophone thoughts, the displayed words must be the soundalikes, not the target animals/objects/etc.

Return this exact shape:
{{
  "summary": "one short sentence",
  "thoughts": [
    {{
      "category": "category revealed after solving",
      "difficulty": "easy | medium | hard | tricky",
      "mechanism_family": "mechanism family id",
      "concept_key": "stable_snake_case_concept_key",
      "words": ["WORD", "WORD", "WORD", "WORD"],
      "explanation": "short final explanation",
      "risk": "short note about ambiguity risk"
    }}
  ]
}}
"""
        payload = self._run_agent(
            name="Category Thought Agent",
            system=system,
            user=user,
            temperature=0.85,
        )
        return payload.get("thoughts", [])

    def _board_builder_agent(
        self,
        target_count: int,
        theme: str,
        difficulty: str,
        thoughts: list[dict[str, Any]],
        *,
        candidate_count: int | None = None,
    ) -> list[dict[str, Any]]:
        candidate_total = candidate_count or target_count + 1
        system = (
            "You are Board Builder Agent. You assemble complete 16-word puzzles "
            "from a pruned pool of category thoughts."
        )
        user = f"""
Return JSON only.

Assemble {candidate_total} candidate puzzle board(s) from this ranked category-thought pool:
{json.dumps(thoughts, indent=2)}

Theme: {theme}
Puzzle mode: {difficulty}

{self.bank_avoid_text}

{QUALITY_RULES}

Board assembly rules:
- Pick exactly 4 thoughts per puzzle.
- Use one easy, one medium, one hard, and one tricky group in that order.
- Use exactly 16 unique words per puzzle.
- Prefer diverse mechanism families across the four groups.
- Avoid two category labels with the same important noun or theme.
- Use the thought's existing words unless a small repair is necessary.
- If you repair a thought, repair the metadata and explanation too.
- Do not include any category thought whose risk note suggests real ambiguity.
- Before returning, test every word against every objective wordplay group in the board. Remove boards where an off-category word also fits a phrase, reversal, spelling, sound, or hidden-letter rule.
- Easy mode: no decoy field, or "decoy": null.
- Hard mode: keep the true groups clean; Misdirection Agent may add one fair decoy later.

Return this exact shape:
{{
  "summary": "one short sentence",
  "puzzles": [
    {{
      "difficulty_mode": "{difficulty}",
      "decoy": null,
      "groups": [
        {{
          "category": "category revealed after solving",
          "difficulty": "easy | medium | hard | tricky",
          "mechanism_family": "mechanism family id",
          "concept_key": "stable_snake_case_concept_key",
          "words": ["WORD", "WORD", "WORD", "WORD"],
          "explanation": "short final explanation"
        }}
      ]
    }}
  ]
}}
"""
        payload = self._run_agent(
            name="Board Builder Agent",
            system=system,
            user=user,
            temperature=0.45,
        )
        return payload.get("puzzles", [])

    def _screen_group_thoughts(
        self,
        thoughts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Locally prune category thoughts before board assembly."""

        kept: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_concepts: set[str] = set()

        for index, thought in enumerate(thoughts):
            normalized = normalize_puzzle(
                {"groups": [thought], "source": "claude-multi-agent"},
                source="claude-multi-agent",
            )
            group = normalized["groups"][0] if normalized.get("groups") else {}
            risk = normalize_category(thought.get("risk", ""))
            if risk:
                group["risk"] = risk
            errors = self._group_thought_errors(group)
            bank_errors = repeat_errors(
                {"source": "claude-multi-agent", "groups": [group]},
                self.bank_memory,
            )
            errors.extend(bank_errors)

            concept_key = str(group.get("concept_key", ""))
            if concept_key and concept_key in seen_concepts:
                errors.append(f"Duplicate concept in thought pool: {concept_key}.")

            if errors:
                rejected.append(
                    {
                        "index": index,
                        "category": group.get("category", thought.get("category", "")),
                        "errors": errors,
                    }
                )
                continue

            kept.append(group)
            if concept_key:
                seen_concepts.add(concept_key)

        return kept, rejected

    def _rank_group_thoughts(
        self,
        thoughts: list[dict[str, Any]],
        target_count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Keep a balanced, locally scored pool for medium-depth ToT board assembly."""

        scored = [
            {
                "score": self._thought_score(thought),
                "category": normalize_category(thought.get("category", "")),
                "difficulty": str(thought.get("difficulty", "")).strip().lower(),
                "mechanism_family": str(thought.get("mechanism_family", "")),
                "risk": normalize_category(thought.get("risk", "")),
                "thought": thought,
            }
            for thought in thoughts
        ]
        scored.sort(
            key=lambda item: (
                DIFFICULTY_RANK.get(str(item["difficulty"]), 9),
                -int(item["score"]),
                str(item["category"]).casefold(),
            )
        )

        per_lane_limit = max(3, min(6, target_count * 3))
        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()

        for difficulty in ("easy", "medium", "hard", "tricky"):
            lane = [item for item in scored if item["difficulty"] == difficulty]
            for item in lane[:per_lane_limit]:
                selected.append(item["thought"])
                selected_ids.add(id(item["thought"]))

        if len(selected) < 8:
            for item in sorted(scored, key=lambda value: -int(value["score"])):
                if id(item["thought"]) in selected_ids:
                    continue
                selected.append(item["thought"])
                selected_ids.add(id(item["thought"]))
                if len(selected) >= 8:
                    break

        score_details = [
            {
                "category": item["category"],
                "difficulty": item["difficulty"],
                "mechanism_family": item["mechanism_family"],
                "score": item["score"],
                "risk": item["risk"],
            }
            for item in sorted(scored, key=lambda value: -int(value["score"]))
        ]
        return selected, score_details

    def _thought_score(self, group: dict[str, Any]) -> int:
        """Score one category thought using cheap local quality hints."""

        score = 10
        difficulty = str(group.get("difficulty", "")).strip().lower()
        family = str(group.get("mechanism_family", "")).strip().lower()
        category = normalize_category(group.get("category", ""))
        risk = normalize_category(group.get("risk", ""))
        words = [normalize_word(word) for word in group.get("words", [])]

        if family in TRICKY_MECHANISMS and difficulty in {"hard", "tricky"}:
            score += 2
        if family in STARTER_MECHANISMS and difficulty in {"easy", "medium"}:
            score += 2
        if family == "shared_property" and difficulty in {"medium", "hard"}:
            score += 1
        if family == "phrase_completion" and difficulty == "tricky":
            score -= 1
        if category.casefold().startswith(("things you can ", "things that can ")):
            score -= 1
        if RISK_TEXT_PATTERN.search(risk):
            score -= 4
        if any(len(word) > 12 for word in words):
            score -= 1
        if len({family, category.casefold()}) < 2:
            score -= 1

        return score

    def _screen_candidate_boards(
        self,
        candidates: list[dict[str, Any]],
        target_count: int,
        *,
        difficulty: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run deterministic gates on complete ToT boards before polish/review."""

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        keep_limit = min(6, max(target_count + 1, target_count * 2))

        for index, candidate in enumerate(candidates):
            normalized = normalize_puzzle(candidate, source="claude-multi-agent")
            validation_target = normalized
            if difficulty == "hard":
                validation_target = dict(normalized)
                validation_target["difficulty_mode"] = "easy"
                validation_target.pop("decoy", None)
            validation = validate_puzzle(
                validation_target,
                require_nyt_blocklist=True,
                require_generated_metadata=True,
            )
            bank_repeat_errors = repeat_errors(normalized, self.bank_memory)
            signature = self._candidate_signature(normalized) if len(normalized.get("groups", [])) == 4 else ""
            errors = validation["errors"] + bank_repeat_errors

            if signature and signature in seen_signatures:
                errors.append("Duplicate board inside ToT candidate set.")

            if errors:
                rejected.append(
                    {
                        "index": index,
                        "categories": [
                            normalize_category(group.get("category", ""))
                            for group in normalized.get("groups", [])
                        ],
                        "errors": errors,
                    }
                )
                continue

            accepted.append(normalized)
            if signature:
                seen_signatures.add(signature)

            if len(accepted) >= keep_limit:
                break

        return accepted, rejected

    def _group_thought_errors(self, group: dict[str, Any]) -> list[str]:
        """Return cheap shape and quality errors for a single category thought."""

        errors: list[str] = []
        category = normalize_category(group.get("category", ""))
        difficulty = str(group.get("difficulty", "")).strip().lower()
        words = [normalize_word(word) for word in group.get("words", [])]
        explanation = normalize_category(group.get("explanation", ""))

        if not category:
            errors.append("Missing category.")

        if difficulty not in {"easy", "medium", "hard", "tricky"}:
            errors.append("Invalid difficulty.")

        if len(words) != 4:
            errors.append("Thought must have exactly 4 words.")

        if len(set(words)) != len(words):
            errors.append("Thought contains duplicate words.")

        for word in words:
            if not word or not WORD_PATTERN.match(word):
                errors.append(f"Word '{word}' has unsupported characters or length.")

        if not group.get("mechanism_family"):
            errors.append("Missing mechanism_family metadata.")

        if not group.get("concept_key"):
            errors.append("Missing concept_key metadata.")

        if not explanation:
            errors.append("Missing explanation.")
        elif len(explanation) > 260:
            errors.append("Explanation is too long.")
        elif DRAFT_EXPLANATION_PATTERN.search(explanation):
            errors.append("Explanation contains draft or self-correction text.")

        risk = normalize_category(group.get("risk", ""))
        if risk and RISK_TEXT_PATTERN.search(risk):
            errors.append(f"Risk note signals an unresolved issue: {risk}")

        errors.extend(generated_constraint_errors([group]))
        return errors

    def _misdirection_agent(self, puzzles: list[dict[str, Any]], difficulty: str) -> list[dict[str, Any]]:
        system = (
            "You are Misdirection Agent. You improve challenge in word-group puzzles "
            "while preserving fairness and a single intended solution."
        )
        user = f"""
Return JSON only.

Puzzle mode: {difficulty}

Improve these candidate puzzles:
{json.dumps(puzzles, indent=2)}

{self.bank_avoid_text}

{QUALITY_RULES}

{self.mechanism_guidance}

Your job:
- Preserve yellow, green, blue, purple group order: easiest, slightly harder, hard, hardest.
- Keep exactly 4 groups of 4 words.
- Keep categories fair and explainable.
- Preserve or repair "mechanism_family" and "concept_key" metadata for every group.
- If a group explanation contains scratch work, alternatives, or a correction, repair the group itself and return only the final polished explanation.
- Avoid making a puzzle depend on trivia only experts would know.
- Easy mode: do not add a decoy. Remove accidental 3-word or 4-word false groups if they make the puzzle feel misleading. Set "decoy" to null or omit it.
- Hard mode: add exactly one intentional decoy object. It should contain 3 or 4 board words, pull from at least 2 real groups, feel tempting, and be weaker than the true groups.
- Hard mode decoys must not exactly match a real answer group.
- Hard mode decoys must include "label", "words", and "why_false".
- In hard mode, do not leave "decoy" as null.
- Do not use decoys as an excuse for unfair alternate answers.

Return this exact shape:
{{
  "summary": "one short sentence",
  "puzzles": [same puzzle shape as input]
}}
"""
        payload = self._run_agent(name="Misdirection Agent", system=system, user=user, temperature=0.65)
        return payload.get("puzzles", puzzles)

    def _solver_agent(self, puzzles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        solver_inputs = []
        for index, puzzle in enumerate(puzzles):
            words = [word for group in puzzle["groups"] for word in group["words"]]
            random.shuffle(words)
            solver_inputs.append({"index": index, "words": words})

        system = (
            "You are Solver Agent. You only see shuffled words and try to solve the "
            "puzzles as a player would. Report whether each puzzle feels solvable."
        )
        user = f"""
Return JSON only.

Solve or evaluate these puzzles from shuffled words only:
{json.dumps(solver_inputs, indent=2)}

After finding likely groups, actively look for cross-category leaks:
- Does any word outside a phrase group also combine with the same target?
- Does any word outside a reversal/spelling/sound group also satisfy that rule?
- If a word plausibly fits two groups, list that in issues.

Return this exact shape:
{{
  "summary": "one short sentence",
  "solves": [
    {{
      "index": 0,
      "solvable": true,
      "confidence": 0.0,
      "guessed_groups": [["WORD", "WORD", "WORD", "WORD"]],
      "issues": ["short issue if any"]
    }}
  ]
}}
"""
        payload = self._run_agent(name="Solver Agent", system=system, user=user, temperature=0.2)
        return payload.get("solves", [])

    def _critic_agent(
        self,
        puzzles: list[dict[str, Any]],
        solver_reviews: list[dict[str, Any]],
        difficulty: str,
    ) -> list[dict[str, Any]]:
        system = (
            "You are Critic Agent. You judge puzzle quality, fairness, ambiguity, "
            "vocabulary accessibility, and whether each category has a satisfying aha."
        )
        user = f"""
Return JSON only.

Candidate puzzles:
{json.dumps(puzzles, indent=2)}

Solver reports:
{json.dumps(solver_reviews, indent=2)}

Puzzle mode: {difficulty}

{QUALITY_RULES}

Score each puzzle. Approve only puzzles that are playable and fair.
This is an acceptance gate: puzzles are saved only if you set approved true and score is at least 7.
Reject or flag puzzles when:
- Category labels sound awkward, sentence-like, or too much like hints.
- Explanations contain draft text, self-corrections, rejected alternatives, or "corrected/replacing/final/settled" language.
- Any explanation says a word is wrong, suggests replacing a word, or includes multiple alternate mappings.
- Anagram categories use fake words, non-common words, or incorrect mappings.
- Group order does not climb yellow, green, blue, purple by difficulty.
- Green feels just as obvious as yellow.
- Purple is easier than blue.
- Purple is only a plain synonym or definition group without an extra wordplay/hidden mechanism.
- Purple is another routine compound/phrase category when the puzzle bank already leans on that mechanism.
- Blue or purple is only a broad everyday object list.
- Any group is missing "mechanism_family" or "concept_key" metadata.
- A category label word appears as an answer word in that same group.
- Two category labels in one puzzle are thematically too similar.
- Two groups use the same "Things you can X" / "Things that can be X" shared-property frame.
- A shared-property group has one weak member that only fits by technicality, uncommon phrasing, or a different grammar pattern.
- Multiple board words, or a complete false group, satisfy another group's objective wordplay rule, such as phrase completion, reversal, spelling, hidden letters, or sound.
- Solver issues show a fully valid alternate answer, solver confidence below 0.5, or confusion severe enough that the intended four groups are not recoverable.
- A single minor cross-fit can be a warning if the intended groups remain clear and the puzzle is still fair.
- The puzzle feels bland because most categories are obvious lists from common places.
- The puzzle leans on repeated safe families like color shades, body/face parts, desk/bedroom/beach items, or playground/zoo lists.
- Easy mode includes intentional misdirection.
- Hard mode is missing exactly one fair decoy, or the decoy is a fully valid alternate group.

Return this exact shape:
{{
  "summary": "one short sentence",
  "reviews": [
    {{
      "index": 0,
      "score": 8,
      "approved": true,
      "issues": ["short issue if any"],
      "best_quality": "short note"
    }}
  ]
}}
"""
        payload = self._run_agent(name="Critic Agent", system=system, user=user, temperature=0.25)
        return payload.get("reviews", [])

    def _editor_agent(
        self,
        puzzles: list[dict[str, Any]],
        local_rejections: list[dict[str, Any]],
        critic_reviews: list[dict[str, Any]],
        solver_reviews: list[dict[str, Any]],
        difficulty: str,
    ) -> dict[str, Any]:
        system = (
            "You are Editor Agent. You make final repairs to puzzle candidates and "
            "return only finished, playable puzzles."
        )
        user = f"""
Return JSON only.

Puzzle candidates:
{json.dumps(puzzles, indent=2)}

{self.bank_avoid_text}

{QUALITY_RULES}

{self.mechanism_guidance}

Local validation rejections:
{json.dumps(local_rejections, indent=2)}

Critic reviews:
{json.dumps(critic_reviews, indent=2)}

Solver reviews:
{json.dumps(solver_reviews, indent=2)}

Puzzle mode: {difficulty}

Final repair rules:
- Keep only playable puzzles.
- Repair rejected candidates if the fix is obvious.
- Do not assume your repaired puzzles are accepted; they will be sent back through Validator, Solver, and Critic.
- For every group, first infer the real connection from the four answer words.
- Then check whether the category label accurately names that connection.
- If the label is broad, misleading, or only partly true, rewrite the label before changing words.
- If a category label word appears as an answer word, repair the label or the answer word.
- For shared-property groups, test each word with the same exact phrase frame. If one word needs a different phrase, feels technical, or sounds unnatural, replace that word or rebuild the group.
- Keep answer words when the connection is strong; replace words only when the group itself is weak, ambiguous, or invalid.
- After any repair, update mechanism_family and concept_key so metadata matches the actual connection.
- Exactly 4 groups per puzzle.
- Exactly 4 words per group.
- Exactly 16 unique words per puzzle.
- Include hidden metadata for every group: "mechanism_family" and "concept_key".
- Metadata is for local validation only; it is not a category label and should not be shown to players.
- concept_key must identify the underlying concept, not the wording. For example, "Words before board" and "___ Board" both use phrase_before_board.
- Category labels should be polished hidden-answer reveals, not titles, clues, or awkward sentences.
- Prefer: "Things with scales", "Classroom items", "Words before board", "Silent first letters".
- Avoid: "Can have scales", "Things that are used in schools", "Words that have silent first letters".
- Explanations must be short, polished final answers. Never include scratch work, alternatives, corrections, or self-revision text.
- If the input explanation reveals a bad word or wrong mapping, repair the word group first, then write a clean final explanation.
- Avoid anagram categories unless every anagram word is common and the mapping is exact.
- Return groups in yellow, green, blue, purple order: easiest, slightly harder, hard, hardest.
- Green must be a little harder than yellow. Purple must be the hardest group.
- Easy mode: no decoy field, or "decoy": null.
- Hard mode: exactly one fair decoy object with "label", "words", and "why_false"; it must not exactly match a real group.
- In hard mode, replace the null decoy example below with the required decoy object.
- Use only these difficulty labels: easy, medium, hard, tricky.

Return this exact shape:
{{
  "summary": "one short sentence",
  "puzzles": [
    {{
      "difficulty_mode": "{difficulty}",
      "decoy": null,
      "groups": [
        {{
          "category": "category revealed after solving",
          "difficulty": "easy | medium | hard | tricky",
          "mechanism_family": "mechanism family id",
          "concept_key": "stable_snake_case_concept_key",
          "words": ["WORD", "WORD", "WORD", "WORD"],
          "explanation": "short fair explanation"
        }}
      ]
    }}
  ]
}}
"""
        return self._run_agent(name="Editor Agent", system=system, user=user, temperature=0.35)

    def _record_bank_memory_event(self) -> None:
        """Expose bank memory status in the agent trace."""

        self.trace.append(
            AgentEvent(
                agent="Bank Memory",
                status="complete",
                summary=(
                    f"Loaded {self.bank_memory['total_puzzles']} existing puzzle(s) "
                    f"and {self.bank_memory['total_groups']} category label(s)."
                ),
                duration_seconds=0,
                details={
                    "recent_labels": self.bank_memory.get("recent_labels", [])[:20],
                    "tracked_group_count": len(self.bank_memory.get("group_keys", [])),
                    "generated_mechanism_family_counts": dict(
                        self.bank_memory.get("generated_mechanism_family_counts", {})
                    ),
                },
            )
        )

    def _review_loop(
        self,
        candidates: list[dict[str, Any]],
        *,
        difficulty: str,
        target_count: int,
        max_rounds: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Validate, solve, critique, repair, and only accept critic-approved puzzles."""

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        current = candidates

        for round_index in range(1, max_rounds + 1):
            if not current or len(accepted) >= target_count:
                break

            locally_screened, local_rejections = self._local_screen(current)
            rejected.extend(self._with_round(local_rejections, round_index, "local_validator"))

            if not locally_screened:
                self._record_orchestrator_round(
                    round_index,
                    accepted_count=len(accepted),
                    rejected_count=len(rejected),
                    repair_count=len(current) if round_index < max_rounds else 0,
                    note="No candidates passed local validation.",
                )

                if round_index >= max_rounds:
                    break

                repair_payload = self._editor_agent(
                    current,
                    local_rejections,
                    [],
                    [],
                    difficulty,
                )
                current = repair_payload.get("puzzles", [])
                continue

            solver_reviews = self._solver_agent(locally_screened)
            critic_reviews = self._critic_agent(locally_screened, solver_reviews, difficulty)
            review_map = self._review_map(critic_reviews)
            repair_indexes: list[int] = []

            for index, puzzle in enumerate(locally_screened):
                review = review_map.get(index)

                if self._critic_approved(review):
                    accepted_candidate, rejection = self._accept_candidate(puzzle, seen_signatures)
                    if accepted_candidate:
                        accepted.append(accepted_candidate)
                        if len(accepted) >= target_count:
                            break
                    elif rejection:
                        rejected.append(self._with_single_round(rejection, round_index, "final_validator"))
                else:
                    if round_index < max_rounds:
                        repair_indexes.append(index)
                    else:
                        rejected.append(
                            self._with_single_round(
                                {
                                    "puzzle": puzzle,
                                    "errors": self._critic_reasons(review),
                                },
                                round_index,
                                "critic",
                            )
                        )

            repair_candidates = [locally_screened[index] for index in repair_indexes]

            self._record_orchestrator_round(
                round_index,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                repair_count=len(repair_candidates),
                note="Critic-gated review completed.",
            )

            if len(accepted) >= target_count or not repair_candidates or round_index >= max_rounds:
                break

            repair_payload = self._editor_agent(
                repair_candidates,
                [],
                self._remap_reviews(critic_reviews, repair_indexes),
                self._remap_reviews(solver_reviews, repair_indexes),
                difficulty,
            )
            current = repair_payload.get("puzzles", [])

        return accepted[:target_count], rejected

    def _record_orchestrator_round(
        self,
        round_index: int,
        *,
        accepted_count: int,
        rejected_count: int,
        repair_count: int,
        note: str,
    ) -> None:
        """Add a compact decision event to the agent trace."""

        self.trace.append(
            AgentEvent(
                agent="Orchestrator",
                status="round_complete",
                summary=(
                    f"Round {round_index}: {note} "
                    f"{accepted_count} accepted, {repair_count} sent to repair, "
                    f"{rejected_count} rejected so far."
                ),
                duration_seconds=0,
                details={
                    "round": round_index,
                    "accepted_so_far": accepted_count,
                    "repair_count": repair_count,
                    "rejected_so_far": rejected_count,
                },
            )
        )

    def _review_map(self, reviews: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """Map review indexes to review payloads, ignoring malformed indexes."""

        mapped: dict[int, dict[str, Any]] = {}
        for review in reviews:
            try:
                mapped[int(review.get("index"))] = review
            except (TypeError, ValueError):
                continue
        return mapped

    def _critic_approved(self, review: dict[str, Any] | None) -> bool:
        """Return true only when the Critic explicitly approves with a passing score."""

        if not review or review.get("approved") is not True:
            return False

        if self._critic_has_blocking_issue(review):
            return False

        try:
            score = float(review.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        return score >= 7

    def _critic_has_blocking_issue(self, review: dict[str, Any]) -> bool:
        """Return true only when critic issues describe severe ambiguity despite approval."""

        issues = review.get("issues")
        if not isinstance(issues, list):
            return False

        for issue in issues:
            issue_text = str(issue)
            if CRITICAL_CRITIC_ISSUE_PATTERN.search(issue_text):
                return True

            confidence_match = SOLVER_CONFIDENCE_PATTERN.search(issue_text)
            if confidence_match:
                try:
                    if float(confidence_match.group(1)) < 0.5:
                        return True
                except ValueError:
                    continue

        return False

    def _critic_reasons(self, review: dict[str, Any] | None) -> list[str]:
        """Return readable rejection reasons from a critic review."""

        if not review:
            return ["Critic did not return a review for this candidate."]

        issues = review.get("issues")
        if isinstance(issues, list) and issues:
            return [str(issue) for issue in issues]

        return [
            f"Critic did not approve this candidate "
            f"(score={review.get('score', 'missing')}, approved={review.get('approved', 'missing')})."
        ]

    def _accept_candidate(
        self,
        puzzle: dict[str, Any],
        seen_signatures: set[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Run final deterministic gates before adding a critic-approved puzzle."""

        normalized = normalize_puzzle(puzzle, source="claude-multi-agent")
        validation = validate_puzzle(
            normalized,
            require_nyt_blocklist=True,
            require_generated_metadata=True,
        )
        bank_repeat_errors = repeat_errors(normalized, self.bank_memory)
        signature = self._candidate_signature(normalized)

        if not validation["ok"] or bank_repeat_errors:
            return None, {"puzzle": normalized, "errors": validation["errors"] + bank_repeat_errors}

        if signature in seen_signatures:
            return None, {"puzzle": normalized, "errors": ["Duplicate in generated batch."]}

        seen_signatures.add(signature)
        return normalized, None

    def _candidate_signature(self, puzzle: dict[str, Any]) -> str:
        """Create a simple board signature for batch-level duplicate rejection."""

        return "|".join(sorted(word for group in puzzle["groups"] for word in group["words"]))

    def _with_round(
        self,
        rejections: list[dict[str, Any]],
        round_index: int,
        stage: str,
    ) -> list[dict[str, Any]]:
        """Annotate rejection payloads with loop metadata."""

        return [self._with_single_round(rejection, round_index, stage) for rejection in rejections]

    def _with_single_round(
        self,
        rejection: dict[str, Any],
        round_index: int,
        stage: str,
    ) -> dict[str, Any]:
        """Annotate one rejection payload with loop metadata."""

        enriched = dict(rejection)
        enriched["round"] = round_index
        enriched["stage"] = stage
        return enriched

    def _remap_reviews(
        self,
        reviews: list[dict[str, Any]],
        indexes: list[int],
    ) -> list[dict[str, Any]]:
        """Filter reviews to repaired candidates and renumber them for the Editor input."""

        mapped = self._review_map(reviews)
        remapped: list[dict[str, Any]] = []

        for new_index, old_index in enumerate(indexes):
            if old_index not in mapped:
                continue

            review = dict(mapped[old_index])
            review["index"] = new_index
            remapped.append(review)

        return remapped

    def _local_screen(
        self,
        puzzles: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for puzzle in puzzles:
            normalized = normalize_puzzle(puzzle, source="claude-multi-agent")
            validation = validate_puzzle(
                normalized,
                require_nyt_blocklist=True,
                require_generated_metadata=True,
            )
            bank_repeat_errors = repeat_errors(normalized, self.bank_memory)
            if validation["ok"] and not bank_repeat_errors:
                accepted.append(normalized)
            else:
                rejected.append({"puzzle": normalized, "errors": validation["errors"] + bank_repeat_errors})

        return accepted, rejected

    def _finalize(
        self,
        puzzles: list[dict[str, Any]],
        target_count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        seen: set[str] = set()

        for puzzle in puzzles:
            normalized = normalize_puzzle(puzzle, source="claude-multi-agent")
            validation = validate_puzzle(
                normalized,
                require_nyt_blocklist=True,
                require_generated_metadata=True,
            )
            signature = "|".join(
                sorted(word for group in normalized["groups"] for word in group["words"])
            )

            if not validation["ok"]:
                rejected.append({"puzzle": normalized, "errors": validation["errors"]})
                continue

            if signature in seen:
                rejected.append({"puzzle": normalized, "errors": ["Duplicate in generated batch."]})
                continue

            accepted.append(normalized)
            seen.add(signature)

            if len(accepted) >= target_count:
                break

        return accepted, rejected
