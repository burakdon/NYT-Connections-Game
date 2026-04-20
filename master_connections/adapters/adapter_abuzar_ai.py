# ============================================================
# adapters/adapter_abuzar_ai.py
# ============================================================

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from adapters.base import PuzzleAdapter

# run_fresh_puzzle builds lanes easy → medium → hard → tricky (NYT reveal order).
_LANE_TO_COLOR = {
    'easy': 'yellow',
    'medium': 'green',
    'hard': 'blue',
    'tricky': 'purple',
}


def _groups_dict_from_abuzar(data: dict) -> dict:
    """Abuzar stores groups as a list with per-group difficulty (lane); master uses color keys."""
    raw = data.get('groups')
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, list):
        return {}

    by_color: dict = {}
    for g in raw:
        if not isinstance(g, dict):
            continue
        lane = (g.get('difficulty') or '').strip().lower()
        color = _LANE_TO_COLOR.get(lane)
        if color:
            by_color[color] = g

    if len(by_color) == 4:
        return by_color

    colors = ('yellow', 'green', 'blue', 'purple')
    by_color = {}
    for i, g in enumerate(raw[:4]):
        if isinstance(g, dict):
            by_color[colors[i]] = g
    return by_color


class AbuzarAIAdapter(PuzzleAdapter):

    name = 'abuzar_ai'

    def __init__(self, project_root: str, api_keys: dict = None):
        self.project_root = project_root
        self.api_keys = api_keys or {}

    def _load_generated_puzzle(self, output_file: str) -> Optional[dict]:
        """run_fresh_puzzle.py saves to data/puzzles.json; tempfile is often unused."""
        p = Path(output_file)
        if p.exists() and p.stat().st_size > 0:
            try:
                return json.loads(p.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                pass

        bank = Path(self.project_root) / 'data' / 'puzzles.json'
        if not bank.exists():
            return None
        try:
            raw = json.loads(bank.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return None
        puzzles = raw if isinstance(raw, list) else raw.get('puzzles', [])
        if not puzzles:
            return None
        return puzzles[-1]

    def generate(self) -> Optional[dict]:
        fd, output_file = tempfile.mkstemp(suffix='.json', text=False)
        os.close(fd)
        env = os.environ.copy()
        env.update(self.api_keys)
        env['OUTPUT_FILE'] = output_file
        data = None

        try:
            result = subprocess.run(
                ['python', 'run_fresh_puzzle.py', '--output', output_file],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )

            if result.returncode != 0:
                err = (result.stderr or '').strip()
                out = (result.stdout or '').strip()
                detail = err if len(err) >= len(out) else out
                if err and out and err not in out:
                    detail = err + '\n' + out
                if len(detail) > 600:
                    detail = detail[:600] + '...'
                print(f'  [abuzar_ai] subprocess failed ({result.returncode}): {detail}')
                return None

            data = self._load_generated_puzzle(output_file)

        except subprocess.TimeoutExpired:
            print(f'  [abuzar_ai] timed out')
            return None
        except Exception as e:
            print(f'  [abuzar_ai] error: {e}')
            return None
        finally:
            try:
                if os.path.exists(output_file):
                    os.unlink(output_file)
            except OSError:
                pass

        if data is None:
            print(f'  [abuzar_ai] no puzzle payload (check data/puzzles.json and latest run)')
            return None
        try:
            groups = _groups_dict_from_abuzar(data)

            def get_group(color):
                g = groups.get(color, {})
                if not isinstance(g, dict):
                    return [], '', 'abuzar_ai'
                words = g.get('words', g.get('members', []))
                conn = (
                    g.get('connection')
                    or g.get('category')
                    or g.get('label')
                    or ''
                )
                mech = g.get('mechanism', g.get('mechanism_family', 'abuzar_ai'))
                return words, conn, mech

            yw, yc, ym = get_group('yellow')
            gw, gc, gm = get_group('green')
            bw, bc, bm = get_group('blue')
            pw, pc, pm = get_group('purple')

            return self.canonical(
                yellow_words=yw, yellow_conn=yc, yellow_mech=ym,
                green_words=gw,  green_conn=gc,  green_mech=gm,
                blue_words=bw,   blue_conn=bc,   blue_mech=bm,
                purple_words=pw, purple_conn=pc, purple_mech=pm,
                meta=data.get('meta', {}),
            )
        except Exception as e:
            print(f'  [abuzar_ai] normalisation error: {e}')
            return None
