"""CLI entry point for growing the verified group bank with Claude agents."""

from __future__ import annotations

import argparse
import sys
import time

from agents.claude_client import ClaudeError
from agents.group_agents import GroupGenerationFactory
from agents.group_bank import add_groups_to_bank, load_group_bank
from agents.puzzle_store import load_puzzles, save_latest_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate verified category groups with Claude agents.")
    parser.add_argument("--count", type=int, default=5, help="Number of new groups to accept.")
    parser.add_argument(
        "--difficulty",
        default="mixed",
        choices=("mixed", "easy", "medium", "hard", "tricky"),
        help="Preferred group difficulty lane.",
    )
    parser.add_argument("--theme", default="", help="Optional group-generation theme.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Stop after this many generation batches.")
    parser.add_argument(
        "--max-empty-batches",
        type=int,
        default=1,
        help="Stop after this many consecutive batches save zero groups.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional Claude model for every group agent, overriding role-specific settings.",
    )
    parser.add_argument(
        "--generator-model",
        default="",
        help="Optional Claude model for the Group Generator agent.",
    )
    parser.add_argument(
        "--reviewer-model",
        default="",
        help="Optional Claude model for the Group Auditor agent.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the agents and validation, but do not append accepted groups to data/groups.json.",
    )
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow repeated group concepts.")
    args = parser.parse_args()

    target = max(1, args.count)
    added = 0
    attempts = 0
    empty_batches = 0
    max_empty_batches = max(1, args.max_empty_batches)

    print(f"Starting verified group bank size: {len(load_group_bank())}")
    print(f"Target new groups: {target}")
    print(
        "Safety limits: "
        f"max_attempts={args.max_attempts}, "
        f"max_empty_batches={max_empty_batches}, "
        f"difficulty={args.difficulty}"
    )

    while added < target and attempts < args.max_attempts:
        attempts += 1
        requested = target - added
        print(f"\nBatch {attempts}: requesting {requested} accepted group(s)")
        factory = GroupGenerationFactory(
            existing_groups=load_group_bank(),
            existing_puzzles=load_puzzles(),
            model=args.model or None,
            generator_model=args.generator_model or None,
            reviewer_model=args.reviewer_model or None,
        )

        try:
            result = factory.generate_groups(
                target_count=requested,
                difficulty=args.difficulty,
                theme=args.theme,
                save=False,
                allow_duplicates=args.allow_duplicates,
            )
        except ClaudeError as error:
            save_latest_run(
                {
                    "accepted": [],
                    "rejected": [{"errors": [str(error)], "stage": "claude_error"}],
                    "trace": [event.to_dict() for event in factory.trace],
                    "saved": False,
                    "bank_total": len(load_puzzles()),
                    "group_bank_total": len(load_group_bank()),
                }
            )
            print(f"Claude error: {error}", file=sys.stderr)
            return 1
        except Exception as error:
            save_latest_run(
                {
                    "accepted": [],
                    "rejected": [{"errors": [str(error)], "stage": "generation_error"}],
                    "trace": [event.to_dict() for event in factory.trace],
                    "saved": False,
                    "bank_total": len(load_puzzles()),
                    "group_bank_total": len(load_group_bank()),
                }
            )
            print(f"Generation error: {error}", file=sys.stderr)
            time.sleep(2)
            continue

        saved = {"accepted": result["accepted"], "rejected": [], "total": len(load_group_bank())}
        if not args.no_save and result["accepted"]:
            saved = add_groups_to_bank(
                result["accepted"],
                allow_duplicates=args.allow_duplicates,
            )

        payload = {
            "accepted": saved["accepted"],
            "rejected": result["rejected"] + saved["rejected"],
            "trace": result["trace"],
            "saved": not args.no_save,
            "bank_total": len(load_puzzles()),
            "group_bank_total": saved["total"],
        }
        save_latest_run(payload)

        accepted_count = len(saved["accepted"])
        added += accepted_count

        print(f"Accepted this batch: {accepted_count}")
        print(f"Rejected this batch: {len(result['rejected']) + len(saved['rejected'])}")
        print(f"Group bank size: {saved['total']}")

        if accepted_count == 0:
            empty_batches += 1
            if empty_batches >= max_empty_batches:
                print(
                    f"\nStopped after {empty_batches} consecutive empty batch(es) "
                    "to avoid extra API spend."
                )
                break
            time.sleep(2)
        else:
            empty_batches = 0

    if added < target:
        print(f"\nStopped after {attempts} batches with {added}/{target} new groups.")
        return 1

    print(f"\nDone. Added {added} group(s). Group bank size: {len(load_group_bank())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
