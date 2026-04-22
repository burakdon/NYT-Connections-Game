"""One-click fresh puzzle generator.

Open this file and press Run.

This file has the four color steps written out directly:

1. Generate Yellow Group
2. Generate Green Group
3. Generate Blue Group
4. Generate Purple Group

Then it builds one puzzle from those four fresh groups. If the puzzle passes
local validation and the NYT hash guard, it saves:

- the puzzle to data/puzzles.json
- the four fresh groups to data/groups.json
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import sys
from typing import Any

from agents.claude_client import ClaudeError
from agents.group_agents import GroupGenerationFactory
from agents.group_bank import (
    add_groups_to_bank,
    assembled_puzzle_errors,
    build_puzzle_from_lane_groups,
    can_add_group,
    candidate_pool,
    load_group_bank,
)
from agents.puzzle_store import add_puzzles, load_puzzles, save_latest_run
from agents.puzzle_validator import normalize_metadata_key, normalize_word, puzzle_fingerprint


THEME = ""
LANES = ("easy", "medium", "hard", "tricky")
RESCUE_MAX_ATTEMPTS = 300
RESCUE_CANDIDATES_PER_LANE = 80


def generate_color_group(
    *,
    color: str,
    lane: str,
    existing_groups: list[dict[str, Any]],
    existing_puzzles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate one approved group for a color lane."""

    print(f"\nGenerate {color.title()} Group ({lane})")
    factory = GroupGenerationFactory(
        existing_groups=existing_groups,
        existing_puzzles=existing_puzzles,
    )
    result = factory.generate_groups(
        target_count=1,
        difficulty=lane,
        theme=THEME,
        save=False,
        require_difficulty=lane,
    )

    if not result["accepted"]:
        raise RuntimeError(
            f"No accepted {color} group. "
            "Open data/latest_agent_run.json or Agent Lab to inspect rejection details."
        )

    group = result["accepted"][0]
    print(f"Accepted {color}: {group['category']} | {', '.join(group['words'])}")
    return {
        "group": group,
        "trace": result["trace"],
        "rejected": [
            {
                **item,
                "color": color,
                "lane": lane,
            }
            for item in result["rejected"]
        ],
    }


def print_puzzle(puzzle: dict[str, Any]) -> None:
    """Print a readable accepted puzzle summary."""

    print(f"\nAccepted puzzle: {puzzle.get('id')}")
    for group in puzzle.get("groups", []):
        words = ", ".join(group.get("words", []))
        print(f"- {group.get('difficulty')}: {group.get('category')} | {words}")


def group_signature(group: dict[str, Any]) -> tuple[str, ...]:
    """Return a compact signature for de-duping candidate groups."""

    return tuple(sorted(normalize_word(word) for word in group.get("words", [])))


def rescue_options_for_lane(
    *,
    lane: str,
    fixed_group: dict[str, Any] | None,
    group_bank: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return rescue candidates for one lane, preferring fixed fresh groups."""

    if fixed_group is not None:
        return [fixed_group]

    options: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    # Newer group-bank entries are more likely to come from recent failed calls.
    for group in reversed(candidate_pool(group_bank, lane)):
        signature = group_signature(group)
        if signature in seen:
            continue
        options.append(group)
        seen.add(signature)
        if len(options) >= RESCUE_CANDIDATES_PER_LANE:
            break

    return options


def try_rescue_puzzle(
    *,
    generated_lane_groups: dict[str, dict[str, Any]],
    existing_puzzles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Try to make a valid puzzle by combining fresh groups with saved groups."""

    if not generated_lane_groups:
        return {
            "accepted": [],
            "rejected": [],
            "trace": [],
            "attempts": 0,
            "fresh_groups_used": 0,
            "top_rejections": [],
        }

    group_bank = load_group_bank()
    existing_fingerprints = {puzzle_fingerprint(item) for item in existing_puzzles}
    fresh_lanes = [lane for lane in LANES if lane in generated_lane_groups]
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    attempts = 0
    best_fresh_groups_used = 0

    def search_with_fixed_lanes(fixed_lanes: set[str]) -> dict[str, Any] | None:
        nonlocal attempts

        options_by_lane = {
            lane: rescue_options_for_lane(
                lane=lane,
                fixed_group=generated_lane_groups.get(lane) if lane in fixed_lanes else None,
                group_bank=group_bank,
            )
            for lane in LANES
        }
        if any(not options for options in options_by_lane.values()):
            return None

        selected_by_lane: dict[str, dict[str, Any]] = {}
        selected_groups: list[dict[str, Any]] = []
        used_words: set[str] = set()
        used_concepts: set[str] = set()

        def walk(index: int) -> dict[str, Any] | None:
            nonlocal attempts

            if attempts >= RESCUE_MAX_ATTEMPTS:
                return None

            if index == len(LANES):
                attempts += 1
                puzzle = build_puzzle_from_lane_groups(
                    selected_by_lane,
                    difficulty_mode="easy",
                )
                errors = assembled_puzzle_errors(puzzle)
                if puzzle_fingerprint(puzzle) in existing_fingerprints:
                    errors.append("Duplicate rescue puzzle fingerprint.")

                if not errors:
                    return puzzle

                for error in errors:
                    rejection_counts[error] += 1
                if len(rejected) < 20:
                    rejected.append(
                        {
                            "puzzle": puzzle,
                            "errors": errors,
                            "stage": "fresh_rescue_validator",
                        }
                    )
                return None

            lane = LANES[index]
            for candidate in options_by_lane[lane]:
                if not can_add_group(selected_groups, candidate, used_words, used_concepts):
                    continue

                words = {normalize_word(word) for word in candidate.get("words", [])}
                concept = normalize_metadata_key(candidate.get("concept_key", ""))

                selected_by_lane[lane] = candidate
                selected_groups.append(candidate)
                used_words.update(words)
                if concept:
                    used_concepts.add(concept)

                puzzle = walk(index + 1)
                if puzzle is not None:
                    return puzzle

                selected_groups.pop()
                selected_by_lane.pop(lane, None)
                used_words.difference_update(words)
                if concept:
                    used_concepts.discard(concept)

            return None

        return walk(0)

    for keep_count in range(len(fresh_lanes), 0, -1):
        for keep_lanes_tuple in combinations(fresh_lanes, keep_count):
            if attempts >= RESCUE_MAX_ATTEMPTS:
                break

            puzzle = search_with_fixed_lanes(set(keep_lanes_tuple))
            if puzzle is not None:
                best_fresh_groups_used = keep_count
                trace = [
                    {
                        "agent": "Rescue Assembler",
                        "status": "complete",
                        "summary": (
                            "Recovered a valid puzzle by combining "
                            f"{keep_count} fresh group(s) with saved group-bank entries."
                        ),
                        "duration_seconds": 0,
                        "details": {
                            "attempts": attempts,
                            "fresh_lanes_available": fresh_lanes,
                            "fresh_lanes_used": list(keep_lanes_tuple),
                        },
                    }
                ]
                return {
                    "accepted": [puzzle],
                    "rejected": rejected,
                    "trace": trace,
                    "attempts": attempts,
                    "fresh_groups_used": best_fresh_groups_used,
                    "top_rejections": rejection_counts.most_common(8),
                }

    trace = [
        {
            "agent": "Rescue Assembler",
            "status": "warning",
            "summary": "Could not recover a valid puzzle from fresh partial groups.",
            "duration_seconds": 0,
            "details": {
                "attempts": attempts,
                "fresh_lanes_available": fresh_lanes,
                "top_rejections": rejection_counts.most_common(8),
            },
        }
    ]
    return {
        "accepted": [],
        "rejected": rejected,
        "trace": trace,
        "attempts": attempts,
        "fresh_groups_used": best_fresh_groups_used,
        "top_rejections": rejection_counts.most_common(8),
    }


def save_rescued_puzzle(
    *,
    rescue_result: dict[str, Any],
    trace: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    generated_lane_groups: dict[str, dict[str, Any]],
) -> bool:
    """Save a rescued puzzle and the fresh groups that made it possible."""

    if not rescue_result["accepted"]:
        return False

    saved_puzzles = add_puzzles(rescue_result["accepted"])
    saved_groups = {"accepted": [], "rejected": [], "total": len(load_group_bank())}

    if saved_puzzles["accepted"]:
        saved_groups = add_groups_to_bank(list(generated_lane_groups.values()))

    payload = {
        "accepted": saved_puzzles["accepted"],
        "rejected": (
            rejected
            + rescue_result["rejected"]
            + saved_puzzles["rejected"]
            + saved_groups["rejected"]
        ),
        "trace": trace + rescue_result["trace"],
        "saved": bool(saved_puzzles["accepted"]),
        "bank_total": saved_puzzles["total"],
        "group_bank_total": saved_groups["total"],
        "generated_groups": saved_groups["accepted"],
        "rescue": {
            "attempted": True,
            "success": bool(saved_puzzles["accepted"]),
            "fresh_groups_used": rescue_result["fresh_groups_used"],
            "attempts": rescue_result["attempts"],
        },
    }
    save_latest_run(payload)

    if not saved_puzzles["accepted"]:
        return False

    print(
        "\nRescue assembler saved a puzzle "
        f"using {rescue_result['fresh_groups_used']} fresh group(s)."
    )
    print(f"Rescue attempts: {rescue_result['attempts']}")
    print(f"Rejected rescue candidates: {len(rescue_result['rejected'])}")
    print(f"Saved fresh groups: {len(saved_groups['accepted'])}")
    print(f"Puzzle bank size: {saved_puzzles['total']}")
    print(f"Group bank size: {saved_groups['total']}")

    for accepted_puzzle in saved_puzzles["accepted"]:
        print_puzzle(accepted_puzzle)

    return True


def save_failure_payload(
    *,
    trace: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    message: str,
    generated_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save a failed run for Agent Lab inspection and keep any verified groups."""

    partial_groups = list(generated_groups or [])
    saved_groups = {"accepted": [], "rejected": [], "total": len(load_group_bank())}
    if partial_groups:
        saved_groups = add_groups_to_bank(partial_groups)

    group_bank_rejections = [
        {
            **item,
            "stage": "partial_group_bank",
        }
        for item in saved_groups.get("rejected", [])
    ]

    save_latest_run(
        {
            "accepted": [],
            "rejected": (
                rejected or [{"errors": [message], "stage": "fresh_puzzle_runner"}]
            ) + group_bank_rejections,
            "trace": trace,
            "saved": False,
            "bank_total": len(load_puzzles()),
            "group_bank_total": saved_groups["total"],
            "generated_groups": saved_groups["accepted"],
        }
    )
    return saved_groups


def main() -> int:
    """Run the four color commands and save one accepted puzzle."""

    print("Fresh puzzle generation")
    print(f"Starting puzzle bank size: {len(load_puzzles())}")
    print(f"Starting group bank size: {len(load_group_bank())}")
    print(f"Theme: {THEME or 'none'}")

    trace: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    existing_puzzles = load_puzzles()
    working_groups = load_group_bank()
    generated_lane_groups: dict[str, dict[str, Any]] = {}

    try:
        # 1. Generate Yellow Group
        yellow = generate_color_group(
            color="yellow",
            lane="easy",
            existing_groups=working_groups,
            existing_puzzles=existing_puzzles,
        )
        trace.extend(yellow["trace"])
        rejected.extend(yellow["rejected"])
        working_groups.append(yellow["group"])
        generated_lane_groups["easy"] = yellow["group"]

        # 2. Generate Green Group
        green = generate_color_group(
            color="green",
            lane="medium",
            existing_groups=working_groups,
            existing_puzzles=existing_puzzles,
        )
        trace.extend(green["trace"])
        rejected.extend(green["rejected"])
        working_groups.append(green["group"])
        generated_lane_groups["medium"] = green["group"]

        # 3. Generate Blue Group
        blue = generate_color_group(
            color="blue",
            lane="hard",
            existing_groups=working_groups,
            existing_puzzles=existing_puzzles,
        )
        trace.extend(blue["trace"])
        rejected.extend(blue["rejected"])
        working_groups.append(blue["group"])
        generated_lane_groups["hard"] = blue["group"]

        # 4. Generate Purple Group
        purple = generate_color_group(
            color="purple",
            lane="tricky",
            existing_groups=working_groups,
            existing_puzzles=existing_puzzles,
        )
        trace.extend(purple["trace"])
        rejected.extend(purple["rejected"])
        working_groups.append(purple["group"])
        generated_lane_groups["tricky"] = purple["group"]

    except ClaudeError as error:
        message = f"Claude error: {error}"
        rescue_result = try_rescue_puzzle(
            generated_lane_groups=generated_lane_groups,
            existing_puzzles=existing_puzzles,
        )
        if save_rescued_puzzle(
            rescue_result=rescue_result,
            trace=trace,
            rejected=rejected,
            generated_lane_groups=generated_lane_groups,
        ):
            return 0

        saved_groups = save_failure_payload(
            trace=trace + rescue_result["trace"],
            rejected=rejected + rescue_result["rejected"],
            message=message,
            generated_groups=list(generated_lane_groups.values()),
        )
        if saved_groups["accepted"]:
            print(f"\nSaved partial groups: {len(saved_groups['accepted'])}")
            print(f"Group bank size: {saved_groups['total']}")
        print(f"\n{message}", file=sys.stderr)
        return 1
    except Exception as error:
        message = f"Generation error: {error}"
        rescue_result = try_rescue_puzzle(
            generated_lane_groups=generated_lane_groups,
            existing_puzzles=existing_puzzles,
        )
        if save_rescued_puzzle(
            rescue_result=rescue_result,
            trace=trace,
            rejected=rejected,
            generated_lane_groups=generated_lane_groups,
        ):
            return 0

        saved_groups = save_failure_payload(
            trace=trace + rescue_result["trace"],
            rejected=rejected + rescue_result["rejected"],
            message=message,
            generated_groups=list(generated_lane_groups.values()),
        )
        if saved_groups["accepted"]:
            print(f"\nSaved partial groups: {len(saved_groups['accepted'])}")
            print(f"Group bank size: {saved_groups['total']}")
        print(f"\n{message}", file=sys.stderr)
        return 1

    puzzle = build_puzzle_from_lane_groups(generated_lane_groups, difficulty_mode="easy")

    errors = assembled_puzzle_errors(puzzle)
    existing_fingerprints = {puzzle_fingerprint(item) for item in existing_puzzles}
    if puzzle_fingerprint(puzzle) in existing_fingerprints:
        errors.append("Duplicate fresh puzzle fingerprint.")

    if errors:
        rejected.append({"puzzle": puzzle, "errors": errors, "stage": "fresh_puzzle_validator"})
        rescue_result = try_rescue_puzzle(
            generated_lane_groups=generated_lane_groups,
            existing_puzzles=existing_puzzles,
        )
        if save_rescued_puzzle(
            rescue_result=rescue_result,
            trace=trace,
            rejected=rejected,
            generated_lane_groups=generated_lane_groups,
        ):
            return 0

        saved_groups = save_failure_payload(
            trace=trace + rescue_result["trace"],
            rejected=rejected + rescue_result["rejected"],
            message="Fresh puzzle failed final validation.",
            generated_groups=list(generated_lane_groups.values()),
        )
        print("\nFresh puzzle failed final validation.")
        for error in errors:
            print(f"- {error}")
        if saved_groups["accepted"]:
            print(f"Saved partial groups: {len(saved_groups['accepted'])}")
            print(f"Group bank size: {saved_groups['total']}")
        return 1

    saved_puzzles = add_puzzles([puzzle])
    saved_groups = {"accepted": [], "rejected": [], "total": len(load_group_bank())}

    if saved_puzzles["accepted"]:
        saved_groups = add_groups_to_bank(list(generated_lane_groups.values()))

    payload = {
        "accepted": saved_puzzles["accepted"],
        "rejected": rejected + saved_puzzles["rejected"] + saved_groups["rejected"],
        "trace": trace,
        "saved": bool(saved_puzzles["accepted"]),
        "bank_total": saved_puzzles["total"],
        "group_bank_total": saved_groups["total"],
        "generated_groups": saved_groups["accepted"],
    }
    save_latest_run(payload)

    print(f"\nAccepted puzzles: {len(saved_puzzles['accepted'])}")
    print(f"Rejected items: {len(payload['rejected'])}")
    print(f"Saved fresh groups: {len(saved_groups['accepted'])}")
    print(f"Puzzle bank size: {saved_puzzles['total']}")
    print(f"Group bank size: {saved_groups['total']}")

    for accepted_puzzle in saved_puzzles["accepted"]:
        print_puzzle(accepted_puzzle)

    if not saved_puzzles["accepted"]:
        print("\nNo puzzle was saved. Check data/latest_agent_run.json or Agent Lab.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
