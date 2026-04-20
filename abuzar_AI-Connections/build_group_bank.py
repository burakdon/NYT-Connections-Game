"""Seed the verified category group bank from existing starter puzzles."""

from __future__ import annotations

import argparse

from agents.group_bank import groups_from_puzzles, save_group_bank
from agents.puzzle_store import load_puzzles


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/groups.json from existing puzzles.")
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Also import groups from generated puzzles. Starter seed puzzles are always imported.",
    )
    args = parser.parse_args()

    sources = {"seed"}
    if args.include_generated:
        sources.add("claude-multi-agent")

    groups, rejected = groups_from_puzzles(load_puzzles(), include_sources=sources)
    save_group_bank(groups)

    print(f"Saved {len(groups)} verified group(s) to data/groups.json")
    if rejected:
        print(f"Rejected {len(rejected)} group(s) with objective validation errors.")
        for item in rejected[:8]:
            group = item.get("group", {})
            print(f"- {group.get('category', 'Unknown')}: {'; '.join(item.get('errors', []))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
