#!/usr/bin/env python3
"""
Convert master_connections/output/generated_puzzles.json (canonical schema)
into webapp/puzzles.json for Kevin's static UI (see webapp/game.js).

Run from repo root (master_connections/):

    python scripts/export_web_puzzles.py

Or after generating puzzles:

    python run.py --only burak --n 5 && python scripts/export_web_puzzles.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_COLORS = ('yellow', 'green', 'blue', 'purple')


def _master_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_generated(path: Path) -> list:
    if not path.exists():
        return []
    raw = path.read_text(encoding='utf-8').strip()
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        print(f'Expected a JSON array in {path}; got {type(data).__name__}.', file=sys.stderr)
        return []
    return data


def _board_from_puzzle(p: dict) -> list[str]:
    board = p.get('board')
    if isinstance(board, list) and len(board) == 16:
        return [str(w).upper() for w in board]
    flat: list[str] = []
    groups = p.get('groups') or {}
    for c in _COLORS:
        g = groups.get(c) or {}
        words = g.get('words') or []
        flat.extend(str(w).upper() for w in words)
    if len(flat) != 16:
        return []
    rng = random.Random(p.get('id') or json.dumps(p.get('groups', {}), sort_keys=True))
    rng.shuffle(flat)
    return flat


def canonical_to_kevin(p: dict) -> dict | None:
    groups_in = p.get('groups')
    if not isinstance(groups_in, dict):
        return None
    out_groups = []
    for c in _COLORS:
        g = groups_in.get(c)
        if not g or len(g.get('words', [])) != 4:
            return None
        conn = (g.get('connection') or 'Group').strip()
        words = [str(w).upper() for w in g['words']]
        out_groups.append(
            {
                'category': conn,
                'words': words,
                'color': c,
            }
        )

    words16 = _board_from_puzzle(p)
    if len(words16) != 16:
        return None

    src = p.get('source')
    if src is not None:
        src = str(src).strip() or None

    entry = {
        'id': p.get('id'),
        'words': words16,
        'groups': out_groups,
        'metadata': {
            'source': src,
            'generated_at': p.get('generated_at'),
        },
    }
    return entry


def export_puzzles(generated_path: Path, out_path: Path, verbose: bool = True) -> int:
    puzzles = _load_generated(generated_path)
    kevin_list = []
    for p in puzzles:
        if not isinstance(p, dict):
            continue
        k = canonical_to_kevin(p)
        if k:
            kevin_list.append(k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(kevin_list, indent=2) + '\n', encoding='utf-8')

    if verbose:
        print(f'Wrote {len(kevin_list)} puzzle(s) → {out_path}')
        if not puzzles:
            print(f'  (no input at {generated_path}; wrote empty array)')
        elif len(kevin_list) < len(puzzles):
            print(f'  (skipped {len(puzzles) - len(kevin_list)} invalid entr(y/ies))')
    return len(kevin_list)


def main() -> None:
    root = _master_root()
    parser = argparse.ArgumentParser(description='Export webapp/puzzles.json from generated puzzles.')
    parser.add_argument(
        '--input',
        type=Path,
        default=root / 'output' / 'generated_puzzles.json',
        help='Canonical puzzle list JSON (array)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=root / 'webapp' / 'puzzles.json',
        help='Output path for static site',
    )
    args = parser.parse_args()
    export_puzzles(args.input.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
