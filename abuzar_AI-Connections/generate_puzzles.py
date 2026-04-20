"""CLI entry point for generating puzzle banks with the multi-agent pipeline."""

from __future__ import annotations

import argparse
import sys
import time

from agents.claude_client import ClaudeError
from agents.group_agents import generate_fresh_puzzle_batch
from agents.group_bank import GROUP_BANK_STRATEGIES, add_groups_to_bank, assemble_puzzle_batch, load_group_bank
from agents.puzzle_agents import MultiAgentPuzzleFactory
from agents.puzzle_store import add_puzzles, load_puzzles, save_latest_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Connections-style puzzles with Claude agents.")
    parser.add_argument("--count", type=int, default=10, help="Number of new puzzles to accept.")
    parser.add_argument("--batch-size", type=int, default=1, help="Puzzles requested per multi-agent run.")
    parser.add_argument("--difficulty", default="easy", choices=("easy", "hard"), help="Puzzle mode.")
    parser.add_argument(
        "--strategy",
        default="group-bank",
        choices=("group-bank", "fresh-puzzle", "standard", "tot", "tot-medium", "medium-tot"),
        help="Generation strategy: no-API assembler, fresh lane-by-lane group generation, standard Claude pipeline, or experimental tree-of-thought search.",
    )
    parser.add_argument("--theme", default="", help="Optional generation theme.")
    parser.add_argument("--max-attempts", type=int, default=6, help="Stop after this many batches.")
    parser.add_argument(
        "--max-empty-batches",
        type=int,
        default=2,
        help="Stop after this many consecutive batches save zero puzzles.",
    )
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Validator/solver/critic/editor rounds per batch.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional Claude model for every agent, overriding role-specific model settings.",
    )
    parser.add_argument(
        "--generator-model",
        default="",
        help="Optional Claude model for generation agents such as scout, wordsmith, and editor.",
    )
    parser.add_argument(
        "--reviewer-model",
        default="",
        help="Optional Claude model for solver and critic agents.",
    )
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate puzzle fingerprints.")
    args = parser.parse_args()

    target = max(1, args.count)
    batch_size = max(1, min(args.batch_size, 12))
    max_empty_batches = max(1, args.max_empty_batches)
    added = 0
    attempts = 0
    empty_batches = 0

    print(f"Starting bank size: {len(load_puzzles())}")
    print(f"Target new puzzles: {target}")
    print(
        "Safety limits: "
        f"max_attempts={args.max_attempts}, "
        f"max_empty_batches={max_empty_batches}, "
        f"max_review_rounds={args.max_review_rounds}, "
        f"strategy={args.strategy}"
    )

    while added < target and attempts < args.max_attempts:
        attempts += 1
        requested = min(batch_size, target - added)
        print(f"\nBatch {attempts}: requesting {requested} accepted puzzle(s)")
        factory = None

        try:
            if args.strategy == "fresh-puzzle":
                result = generate_fresh_puzzle_batch(
                    target_count=requested,
                    existing_groups=load_group_bank(),
                    existing_puzzles=load_puzzles(),
                    theme=args.theme,
                    difficulty=args.difficulty,
                    model=args.model or None,
                    generator_model=args.generator_model or None,
                    reviewer_model=args.reviewer_model or None,
                )
            elif args.strategy in GROUP_BANK_STRATEGIES:
                result = assemble_puzzle_batch(
                    target_count=requested,
                    existing_puzzles=load_puzzles(),
                    difficulty=args.difficulty,
                )
            else:
                factory = MultiAgentPuzzleFactory(
                    model=args.model or None,
                    generator_model=args.generator_model or None,
                    reviewer_model=args.reviewer_model or None,
                    existing_puzzles=load_puzzles(),
                )
                result = factory.generate_batch(
                    target_count=requested,
                    theme=args.theme,
                    difficulty=args.difficulty,
                    max_review_rounds=args.max_review_rounds,
                    strategy=args.strategy,
                )
        except ClaudeError as error:
            if factory is not None:
                save_latest_run(
                    {
                        "accepted": [],
                        "rejected": [
                            {
                                "errors": [str(error)],
                                "round": attempts,
                                "stage": "claude_error",
                            }
                        ],
                        "trace": [event.to_dict() for event in factory.trace],
                        "saved": False,
                        "bank_total": len(load_puzzles()),
                    }
                )
            print(f"Claude error: {error}", file=sys.stderr)
            return 1
        except Exception as error:
            if factory is not None:
                save_latest_run(
                    {
                        "accepted": [],
                        "rejected": [
                            {
                                "errors": [str(error)],
                                "round": attempts,
                                "stage": "generation_error",
                            }
                        ],
                        "trace": [event.to_dict() for event in factory.trace],
                        "saved": False,
                        "bank_total": len(load_puzzles()),
                    }
                )
            print(f"Generation error: {error}", file=sys.stderr)
            time.sleep(2)
            continue

        saved = add_puzzles(result["accepted"], allow_duplicates=args.allow_duplicates)
        group_saved = {"accepted": [], "rejected": [], "total": len(load_group_bank())}
        if saved["accepted"] and result.get("generated_groups"):
            group_saved = add_groups_to_bank(result["generated_groups"])

        save_latest_run(
            {
                "accepted": saved["accepted"],
                "rejected": result["rejected"] + saved["rejected"] + group_saved["rejected"],
                "trace": result["trace"],
                "saved": True,
                "bank_total": saved["total"],
                "group_bank_total": group_saved["total"],
                "generated_groups": group_saved["accepted"],
            }
        )

        accepted_count = len(saved["accepted"])
        added += accepted_count

        print(f"Accepted this batch: {accepted_count}")
        print(f"Rejected this batch: {len(result['rejected']) + len(saved['rejected']) + len(group_saved['rejected'])}")
        print(f"Bank size: {saved['total']}")
        if result.get("generated_groups"):
            print(f"Saved fresh groups: {len(group_saved['accepted'])}")
            print(f"Group bank size: {group_saved['total']}")

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
        print(f"\nStopped after {attempts} batches with {added}/{target} new puzzles.")
        return 1

    print(f"\nDone. Added {added} puzzle(s). Bank size: {len(load_puzzles())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
