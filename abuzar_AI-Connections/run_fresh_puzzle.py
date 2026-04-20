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

import sys
from typing import Any

from agents.claude_client import ClaudeError
from agents.group_agents import GroupGenerationFactory
from agents.group_bank import (
    add_groups_to_bank,
    assembled_puzzle_errors,
    build_puzzle_from_lane_groups,
    load_group_bank,
)
from agents.puzzle_store import add_puzzles, load_puzzles, save_latest_run
from agents.puzzle_validator import puzzle_fingerprint


THEME = ""


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


def save_failure_payload(
    *,
    trace: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    message: str,
) -> None:
    """Save a failed run for Agent Lab inspection."""

    save_latest_run(
        {
            "accepted": [],
            "rejected": rejected or [{"errors": [message], "stage": "fresh_puzzle_runner"}],
            "trace": trace,
            "saved": False,
            "bank_total": len(load_puzzles()),
            "group_bank_total": len(load_group_bank()),
            "generated_groups": [],
        }
    )


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

    except ClaudeError as error:
        message = f"Claude error: {error}"
        save_failure_payload(trace=trace, rejected=rejected, message=message)
        print(f"\n{message}", file=sys.stderr)
        return 1
    except Exception as error:
        message = f"Generation error: {error}"
        save_failure_payload(trace=trace, rejected=rejected, message=message)
        print(f"\n{message}", file=sys.stderr)
        return 1

    fresh_groups = {
        "easy": yellow["group"],
        "medium": green["group"],
        "hard": blue["group"],
        "tricky": purple["group"],
    }
    puzzle = build_puzzle_from_lane_groups(fresh_groups, difficulty_mode="easy")

    errors = assembled_puzzle_errors(puzzle)
    existing_fingerprints = {puzzle_fingerprint(item) for item in existing_puzzles}
    if puzzle_fingerprint(puzzle) in existing_fingerprints:
        errors.append("Duplicate fresh puzzle fingerprint.")

    if errors:
        rejected.append({"puzzle": puzzle, "errors": errors, "stage": "fresh_puzzle_validator"})
        save_failure_payload(
            trace=trace,
            rejected=rejected,
            message="Fresh puzzle failed final validation.",
        )
        print("\nFresh puzzle failed final validation.")
        for error in errors:
            print(f"- {error}")
        return 1

    saved_puzzles = add_puzzles([puzzle])
    saved_groups = {"accepted": [], "rejected": [], "total": len(load_group_bank())}

    if saved_puzzles["accepted"]:
        saved_groups = add_groups_to_bank(list(fresh_groups.values()))

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
