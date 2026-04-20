"""Claude agents for growing the verified category group bank."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from agents.bank_memory import avoid_instructions, build_bank_memory
from agents.claude_client import call_claude, load_env_files
from agents.concept_inspiration import (
    format_concept_inspiration_guidance,
    inspiration_copy_errors,
)
from agents.group_bank import (
    GROUP_GENERATOR_SOURCE,
    add_groups_to_bank,
    assembled_puzzle_errors,
    build_puzzle_from_lane_groups,
    group_repeat_errors,
    load_group_bank,
    normalize_group,
    validate_group,
)
from agents.mechanism_library import format_inspiration_guidance, format_mechanism_guidance
from agents.puzzle_agents import QUALITY_RULES, extract_json_object
from agents.puzzle_validator import normalize_category, puzzle_fingerprint


# Generator default was Opus (very slow per request). Sonnet is much faster; override with
# CLAUDE_GROUP_GENERATOR_MODEL if you want Opus quality back.
DEFAULT_GROUP_GENERATOR_MODEL = "claude-sonnet-4-6"
DEFAULT_GROUP_REVIEWER_MODEL = "claude-sonnet-4-6"
GROUP_DIFFICULTIES = {"easy", "medium", "hard", "tricky"}


@dataclass
class GroupAgentEvent:
    """One group-generation trace event."""

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


class GroupGenerationFactory:
    """Generate reusable verified groups with a small multi-agent loop."""

    def __init__(
        self,
        *,
        existing_groups: list[dict[str, Any]] | None = None,
        existing_puzzles: list[dict[str, Any]] | None = None,
        generator_model: str | None = None,
        reviewer_model: str | None = None,
        model: str | None = None,
    ):
        load_env_files()
        self.model = model
        self.generator_model = (
            generator_model
            or os.environ.get("CLAUDE_GROUP_GENERATOR_MODEL")
            or os.environ.get("CLAUDE_GENERATOR_MODEL")
            or DEFAULT_GROUP_GENERATOR_MODEL
        )
        self.reviewer_model = (
            reviewer_model
            or os.environ.get("CLAUDE_GROUP_REVIEWER_MODEL")
            or os.environ.get("CLAUDE_REVIEWER_MODEL")
            or DEFAULT_GROUP_REVIEWER_MODEL
        )
        self.existing_groups = existing_groups or []
        self.existing_puzzles = existing_puzzles or []
        self.trace: list[GroupAgentEvent] = []

        memory = build_bank_memory(self.existing_puzzles)
        self.bank_avoid_text = avoid_instructions(memory)
        self.mechanism_guidance = format_mechanism_guidance(memory, difficulty="easy", limit=6)
        self.inspiration_guidance = format_inspiration_guidance(count=6)

    def _model_for_agent(self, name: str) -> str:
        if self.model:
            return self.model
        if name == "Group Auditor":
            return self.reviewer_model
        return self.generator_model

    def _run_agent(
        self,
        *,
        name: str,
        system: str,
        user: str,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        started = time.time()
        selected_model = self._model_for_agent(name)
        text = ""
        parsed: dict[str, Any] | None = None
        parse_error: Exception | None = None
        json_retry_count = 0

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
                GroupAgentEvent(
                    agent=name,
                    status="error",
                    summary=f"Invalid JSON response after retry: {message}",
                    duration_seconds=duration,
                    details={
                        "raw_excerpt": text[:900],
                        "json_retry_count": 1,
                        "model": selected_model,
                    },
                )
            )
            raise ValueError(f"{name} returned invalid JSON after retry: {message}") from parse_error

        self.trace.append(
            GroupAgentEvent(
                agent=name,
                status="complete",
                summary=str(parsed.get("summary") or "Returned JSON."),
                duration_seconds=duration,
                details={
                    "raw_excerpt": text[:900],
                    "json_retry_count": json_retry_count,
                    "model": selected_model,
                },
            )
        )
        return parsed

    def generate_groups(
        self,
        *,
        target_count: int,
        difficulty: str = "mixed",
        theme: str = "",
        save: bool = True,
        allow_duplicates: bool = False,
        require_difficulty: str | None = None,
    ) -> dict[str, Any]:
        """Generate, review, locally validate, and optionally save group-bank entries."""

        target_count = max(1, target_count)
        required_difficulty = normalize_category(require_difficulty or "").casefold()
        candidates = self._group_generator(target_count, difficulty, theme)
        local_candidates, local_rejections = self._local_screen(candidates)
        reviews = self._group_auditor(local_candidates, difficulty, theme) if local_candidates else {}

        accepted: list[dict[str, Any]] = []
        rejected = local_rejections
        for index, group in enumerate(local_candidates):
            review = reviews.get(index)
            if not self._review_approved(review):
                rejected.append(
                    {
                        "group": group,
                        "errors": self._review_reasons(review),
                        "stage": "group_auditor",
                    }
                )
                continue

            repaired = review.get("repaired_group") if isinstance(review, dict) else None
            final_group = normalize_group(repaired if isinstance(repaired, dict) else group)
            validation = validate_group(final_group)
            repeat_errors = [] if allow_duplicates else group_repeat_errors(final_group, self.existing_groups + accepted)
            inspiration_errors = inspiration_copy_errors(final_group)
            difficulty_errors: list[str] = []
            if required_difficulty and final_group.get("difficulty") != required_difficulty:
                difficulty_errors.append(
                    "Generated group was accepted by the auditor, but it does not match "
                    f"the requested lane: expected {required_difficulty}, got {final_group.get('difficulty')}."
                )

            if not validation["ok"] or repeat_errors or inspiration_errors or difficulty_errors:
                rejected.append(
                    {
                        "group": final_group,
                        "errors": (
                            list(validation["errors"])
                            + repeat_errors
                            + inspiration_errors
                            + difficulty_errors
                        ),
                        "stage": "final_group_validator",
                    }
                )
                continue

            accepted.append(final_group)
            if len(accepted) >= target_count:
                break

        saved_payload = {"accepted": accepted, "rejected": [], "total": len(self.existing_groups)}
        if save and accepted:
            saved_payload = add_groups_to_bank(accepted, allow_duplicates=allow_duplicates)

        self.trace.append(
            GroupAgentEvent(
                agent="Group Orchestrator",
                status="complete" if saved_payload["accepted"] else "warning",
                summary=(
                    f"Accepted {len(saved_payload['accepted'])} verified group(s), "
                    f"rejected {len(rejected) + len(saved_payload.get('rejected', []))}."
                ),
                duration_seconds=0,
                details={
                    "target_count": target_count,
                    "candidate_count": len(candidates),
                    "locally_screened": len(local_candidates),
                    "saved": save,
                    "bank_total": saved_payload["total"],
                },
            )
        )

        return {
            "accepted": saved_payload["accepted"],
            "rejected": rejected + saved_payload.get("rejected", []),
            "trace": [event.to_dict() for event in self.trace],
            "saved": save,
            "bank_total": saved_payload["total"],
        }

    def _existing_group_context(self, limit: int = 80) -> str:
        lines = []
        for group in self.existing_groups[:limit]:
            lines.append(
                "- "
                f"{group.get('category')} "
                f"({group.get('difficulty')}, {group.get('mechanism_family')}, "
                f"{group.get('concept_key')}): "
                f"{', '.join(group.get('words', []))}"
            )

        return "\n".join(lines) if lines else "- none"

    def _group_generator(self, target_count: int, difficulty: str, theme: str) -> list[dict[str, Any]]:
        requested = max(target_count * 3, target_count + 8)
        difficulty_text = normalize_category(difficulty or "mixed")
        theme_text = normalize_category(theme or "none")
        concept_inspiration_guidance = format_concept_inspiration_guidance(
            difficulty=difficulty,
            count=20,
        )
        system = (
            "You are Group Generator Agent. You create reusable answer groups for a "
            "Connections-style puzzle bank. You do not assemble full puzzles."
        )
        user = f"""
Return JSON only.

Generate {requested} candidate answer groups.
Desired difficulty lane: {difficulty_text}
Optional theme: {theme_text}

Existing verified groups to avoid:
{self._existing_group_context()}

{self.bank_avoid_text}

{QUALITY_RULES}

{self.mechanism_guidance}

{self.inspiration_guidance}

{concept_inspiration_guidance}

Group-generation rules:
- Each candidate is exactly one answer group, not a full puzzle.
- Each group must have exactly four playable answer words.
- All four words must satisfy the same clean rule.
- Use a mix of semantic sets, shared properties, sound, spelling, hidden-letter, transformation, phrase, and double-identity mechanisms.
- Concept inspiration is not a menu. Do not output the sampled concept, a near-rename, or its concept_key.
- When using inspiration, transform the underlying pattern: make a sibling idea, contrast idea, narrower/broader variant, or mechanism shift.
- Do not output broad stale sets like planets, flowers, dog breeds, card suits, or chess pieces unless the theme strongly requires it.
- Do not repeat any existing concept_key or exact category label from the verified groups above.
- Avoid anagrams unless all displayed words are common playable words and the mapping is exact.
- For homophones, display the soundalike answer word, not the target item. Example: use NEW for gnu, not GNU.
- For hidden-letter groups, every explanation claim must be literally true.
- Do not include scratch notes, alternatives, corrections, or rejected mappings.
- If you notice a bad word while drafting, replace it in the words array before returning.

Return this exact shape:
{{
  "summary": "one short sentence",
  "groups": [
    {{
      "category": "polished reveal label",
      "difficulty": "easy | medium | hard | tricky",
      "mechanism_family": "semantic_set | shared_property | phrase_completion | sound | spelling | transformation | double_identity | light_trivia",
      "concept_key": "stable_unique_snake_case_key",
      "words": ["WORD", "WORD", "WORD", "WORD"],
      "explanation": "one short final sentence explaining the rule"
    }}
  ]
}}
"""
        payload = self._run_agent(
            name="Group Generator",
            system=system,
            user=user,
            temperature=0.75,
            max_tokens=6000,
        )
        groups = payload.get("groups", [])
        return groups if isinstance(groups, list) else []

    def _group_auditor(
        self,
        groups: list[dict[str, Any]],
        difficulty: str,
        theme: str,
    ) -> dict[int, dict[str, Any]]:
        system = (
            "You are Group Auditor Agent. You verify whether one category group is "
            "factually correct, fair, and reusable for a Connections-style puzzle bank."
        )
        user = f"""
Return JSON only.

Candidate groups:
{json.dumps(groups, indent=2)}

Desired difficulty lane: {normalize_category(difficulty or "mixed")}
Optional theme: {normalize_category(theme or "none")}

Existing verified groups to avoid:
{self._existing_group_context()}

Audit rules:
- Approve only groups where all four words cleanly satisfy the same rule.
- Reject if one word only fits technically, awkwardly, by a different grammar pattern, or through a false claim.
- Reject fake/non-common anagram words, backwards homophone displays, and hidden-letter claims that are not literal.
- Reject exact repeated concepts, renamed repeats, and groups that overlap heavily with existing verified groups.
- Reject broad stale groups unless they are unusually useful for yellow/easy.
- Score 8 or higher only if the group can be saved without another LLM pass.
- If a group is almost good and one obvious word replacement fixes it, include repaired_group.

Return this exact shape:
{{
  "summary": "one short sentence",
  "reviews": [
    {{
      "index": 0,
      "approved": true,
      "score": 8,
      "issues": ["short issue or warning"],
      "repaired_group": null
    }}
  ]
}}
"""
        payload = self._run_agent(
            name="Group Auditor",
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=6000,
        )
        reviews = payload.get("reviews", [])
        mapped: dict[int, dict[str, Any]] = {}
        if not isinstance(reviews, list):
            return mapped

        for review in reviews:
            if not isinstance(review, dict):
                continue
            try:
                mapped[int(review.get("index"))] = review
            except (TypeError, ValueError):
                continue

        return mapped

    def _local_screen(self, groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for group in groups:
            if not isinstance(group, dict):
                rejected.append({"group": group, "errors": ["Group candidate must be an object."], "stage": "local_group_validator"})
                continue

            normalized = normalize_group(
                {
                    **group,
                    "source": GROUP_GENERATOR_SOURCE,
                    "verified": True,
                    "origin": {"source": GROUP_GENERATOR_SOURCE},
                }
            )
            validation = validate_group(normalized)
            repeat_errors = group_repeat_errors(normalized, self.existing_groups + accepted)
            inspiration_errors = inspiration_copy_errors(normalized)
            if not validation["ok"] or repeat_errors or inspiration_errors:
                rejected.append(
                    {
                        "group": normalized,
                        "errors": list(validation["errors"]) + repeat_errors + inspiration_errors,
                        "stage": "local_group_validator",
                    }
                )
                continue

            accepted.append(normalized)

        self.trace.append(
            GroupAgentEvent(
                agent="Local Group Validator",
                status="complete",
                summary=f"Kept {len(accepted)} candidate group(s), rejected {len(rejected)} before audit.",
                duration_seconds=0,
                details={"rejected": rejected[:8]},
            )
        )
        return accepted, rejected

    @staticmethod
    def _review_approved(review: dict[str, Any] | None) -> bool:
        if not isinstance(review, dict) or review.get("approved") is not True:
            return False

        try:
            score = float(review.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        return score >= 8

    @staticmethod
    def _review_reasons(review: dict[str, Any] | None) -> list[str]:
        if not isinstance(review, dict):
            return ["Group auditor did not return a review."]

        issues = review.get("issues")
        if isinstance(issues, list) and issues:
            return [str(issue) for issue in issues]

        return ["Group auditor rejected this group."]


def generate_and_save_groups(
    *,
    target_count: int,
    existing_puzzles: list[dict[str, Any]],
    difficulty: str = "mixed",
    theme: str = "",
    save: bool = True,
    model: str | None = None,
    generator_model: str | None = None,
    reviewer_model: str | None = None,
    allow_duplicates: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for CLI and API callers."""

    factory = GroupGenerationFactory(
        existing_groups=load_group_bank(),
        existing_puzzles=existing_puzzles,
        model=model,
        generator_model=generator_model,
        reviewer_model=reviewer_model,
    )
    return factory.generate_groups(
        target_count=target_count,
        difficulty=difficulty,
        theme=theme,
        save=save,
        allow_duplicates=allow_duplicates,
    )


def generate_fresh_puzzle_batch(
    *,
    target_count: int,
    existing_groups: list[dict[str, Any]],
    existing_puzzles: list[dict[str, Any]],
    theme: str = "",
    difficulty: str = "easy",
    model: str | None = None,
    generator_model: str | None = None,
    reviewer_model: str | None = None,
) -> dict[str, Any]:
    """Generate fresh lane groups, then build one checked puzzle per target."""

    target_count = max(1, target_count)
    accepted_puzzles: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    generated_groups: list[dict[str, Any]] = []
    working_groups = list(existing_groups)
    seen_puzzle_fingerprints = {puzzle_fingerprint(puzzle) for puzzle in existing_puzzles}
    lanes = ("easy", "medium", "hard", "tricky")
    difficulty_mode = "hard" if difficulty == "hard" else "easy"

    for puzzle_index in range(target_count):
        lane_groups: dict[str, dict[str, Any]] = {}
        puzzle_rejections_before = len(rejected)

        for lane in lanes:
            factory = GroupGenerationFactory(
                existing_groups=working_groups,
                existing_puzzles=existing_puzzles + accepted_puzzles,
                model=model,
                generator_model=generator_model,
                reviewer_model=reviewer_model,
            )
            lane_result = factory.generate_groups(
                target_count=1,
                difficulty=lane,
                theme=theme,
                save=False,
                require_difficulty=lane,
            )
            trace.extend(lane_result["trace"])
            rejected.extend(
                {
                    **item,
                    "lane": lane,
                    "puzzle_index": puzzle_index,
                }
                for item in lane_result["rejected"]
            )

            if not lane_result["accepted"]:
                trace.append(
                    GroupAgentEvent(
                        agent="Fresh Puzzle Orchestrator",
                        status="warning",
                        summary=f"Could not get an accepted {lane} group for puzzle {puzzle_index + 1}.",
                        duration_seconds=0,
                        details={
                            "lane": lane,
                            "puzzle_index": puzzle_index,
                            "rejections_added": len(rejected) - puzzle_rejections_before,
                        },
                    ).to_dict()
                )
                return {
                    "accepted": accepted_puzzles,
                    "rejected": rejected,
                    "trace": trace,
                    "generated_groups": generated_groups,
                }

            selected_group = lane_result["accepted"][0]
            lane_groups[lane] = selected_group
            working_groups.append(selected_group)

        puzzle = build_puzzle_from_lane_groups(
            lane_groups,
            difficulty_mode=difficulty_mode,
        )
        errors = assembled_puzzle_errors(puzzle)
        fingerprint = puzzle_fingerprint(puzzle)
        if fingerprint in seen_puzzle_fingerprints:
            errors.append("Duplicate fresh puzzle fingerprint.")

        if errors:
            rejected.append(
                {
                    "puzzle": puzzle,
                    "errors": errors,
                    "stage": "fresh_puzzle_validator",
                    "puzzle_index": puzzle_index,
                }
            )
            trace.append(
                GroupAgentEvent(
                    agent="Fresh Puzzle Orchestrator",
                    status="warning",
                    summary=f"Generated four fresh groups, but puzzle {puzzle_index + 1} failed final checks.",
                    duration_seconds=0,
                    details={"errors": errors},
                ).to_dict()
            )
            continue

        accepted_puzzles.append(puzzle)
        generated_groups.extend(lane_groups[lane] for lane in lanes)
        seen_puzzle_fingerprints.add(fingerprint)
        trace.append(
            GroupAgentEvent(
                agent="Fresh Puzzle Orchestrator",
                status="complete",
                summary=f"Built fresh puzzle {puzzle_index + 1} from one generated group per lane.",
                duration_seconds=0,
                details={
                    "groups": [
                        {
                            "lane": lane,
                            "category": lane_groups[lane].get("category"),
                            "words": lane_groups[lane].get("words"),
                        }
                        for lane in lanes
                    ]
                },
            ).to_dict()
        )

    return {
        "accepted": accepted_puzzles,
        "rejected": rejected,
        "trace": trace,
        "generated_groups": generated_groups,
    }
