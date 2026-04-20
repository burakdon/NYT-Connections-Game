#!/usr/bin/env python3
"""
Merge master_connections/output/generated_puzzles.json into Abuzar AI's local banks.

Abuzar's runners use abuzar_AI-Connections/data/puzzles.json and data/groups.json so
the Claude agents avoid repeating boards and answers already present in the shared
canonical archive — including puzzles from Burak, Kevin, Adreama, etc.

Run from master_connections (same folder as run.py):

    python scripts/sync_abuzar_bank_from_master.py

    python scripts/sync_abuzar_bank_from_master.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ABUZAR_PKG = _REPO_ROOT / 'abuzar_AI-Connections'

_COLORS_AND_LANES = (
    ('yellow', 'easy'),
    ('green', 'medium'),
    ('blue', 'hard'),
    ('purple', 'tricky'),
)


def _bootstrap_abuzar_import() -> None:
    if str(_ABUZAR_PKG) not in sys.path:
        sys.path.insert(0, str(_ABUZAR_PKG))


def _load_master_generated(path: Path) -> list:
    if not path.exists():
        return []
    raw = path.read_text(encoding='utf-8').strip()
    if not raw:
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def canonical_to_abuzar_shape(master: dict) -> dict | None:
    """Map master generator schema → Abuzar normalize_puzzle input (four groups + lanes)."""
    mg = master.get('groups')
    if not isinstance(mg, dict):
        return None
    groups_list = []
    for color, lane in _COLORS_AND_LANES:
        block = mg.get(color) or {}
        words = block.get('words') or []
        if len(words) != 4:
            return None
        conn = (block.get('connection') or '').strip() or 'Group'
        groups_list.append(
            {
                'category': conn,
                'difficulty': lane,
                'words': list(words),
                'explanation': conn,
            }
        )
    mid = master.get('id')
    src = master.get('source') or 'master-archive'
    return {
        'groups': groups_list,
        'source': f'synced-{src}',
        'id': str(mid) if mid is not None else '',
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Merge master canonical puzzles into Abuzar data/puzzles.json and refresh groups.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print counts only; do not write files.',
    )
    args = parser.parse_args()

    master_path = _REPO_ROOT / 'master_connections' / 'output' / 'generated_puzzles.json'
    puzzles_path = _ABUZAR_PKG / 'data' / 'puzzles.json'
    groups_path = _ABUZAR_PKG / 'data' / 'groups.json'

    master_list = _load_master_generated(master_path)
    if not master_list:
        print(f'No puzzles in {master_path}; nothing to merge.')
        return 0

    _bootstrap_abuzar_import()
    from agents.group_bank import (  # noqa: E402
        group_key,
        load_group_bank,
        normalize_group,
        save_group_bank,
        validate_group,
    )
    from agents.puzzle_store import load_puzzles, save_puzzles  # noqa: E402
    from agents.puzzle_validator import normalize_puzzle, puzzle_fingerprint  # noqa: E402

    existing = load_puzzles(puzzles_path)
    fp_seen: set[str] = set()
    merged: list = []
    for p in existing:
        np = normalize_puzzle(p, source=p.get('source'))
        merged.append(np)
        fp_seen.add(puzzle_fingerprint(np))

    added_puzzles = 0
    for m in master_list:
        if not isinstance(m, dict):
            continue
        raw_shape = canonical_to_abuzar_shape(m)
        if raw_shape is None:
            continue
        np = normalize_puzzle(raw_shape, source=raw_shape.get('source'))
        fp = puzzle_fingerprint(np)
        if fp in fp_seen:
            continue
        fp_seen.add(fp)
        merged.append(np)
        added_puzzles += 1

    # Refresh group bank from merged puzzles (skip invalid / duplicate word-sets).
    bank = load_group_bank(groups_path)
    seen_keys = {group_key(g.get('words', [])) for g in bank}

    added_groups = 0
    for puzzle in merged:
        for grp in puzzle.get('groups') or []:
            if not isinstance(grp, dict):
                continue
            origin = {
                'source': 'sync-from-master',
                'puzzle_id': str(puzzle.get('id', '')),
            }
            try:
                ng = normalize_group({**grp, 'verified': True}, origin=origin)
            except Exception:
                continue
            if not validate_group(ng)['ok']:
                continue
            key = group_key(ng.get('words', []))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bank.append(ng)
            added_groups += 1

    print(f'Master puzzles read: {len(master_list)}')
    print(f'Abuzar puzzles before: {len(existing)}')
    print(f'New puzzles merged from master: {added_puzzles}')
    print(f'Abuzar puzzles after: {len(merged)}')
    print(f'New groups appended to bank: {added_groups}')
    print(f'Total groups in bank after: {len(bank)}')

    if args.dry_run:
        print('Dry run — no files written.')
        return 0

    puzzles_path.parent.mkdir(parents=True, exist_ok=True)
    save_puzzles(merged, puzzles_path)
    save_group_bank(bank, groups_path)
    print(f'Wrote {puzzles_path}')
    print(f'Wrote {groups_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
