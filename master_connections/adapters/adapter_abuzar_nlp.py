# ============================================================
# adapters/adapter_abuzar_nlp.py
# ============================================================

import os
import json
import subprocess
import tempfile
from typing import Optional
from adapters.base import PuzzleAdapter


class AbuzarNLPAdapter(PuzzleAdapter):

    name = 'abuzar_nlp'

    def __init__(self, project_root: str):
        self.project_root = project_root

    def generate(self) -> Optional[dict]:
        fd, output_file = tempfile.mkstemp(suffix='.json', text=False)
        os.close(fd)
        data = None
        try:
            result = subprocess.run(
                ['python', 'pick_one_puzzle.py', '--output', output_file],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f'  [abuzar_nlp] subprocess error: {result.stderr[:200]}')
                return None

            if not os.path.exists(output_file):
                print(f'  [abuzar_nlp] no output file produced')
                return None

            with open(output_file) as f:
                data = json.load(f)

        except subprocess.TimeoutExpired:
            print(f'  [abuzar_nlp] timed out')
            return None
        except Exception as e:
            print(f'  [abuzar_nlp] error: {e}')
            return None
        finally:
            try:
                if os.path.exists(output_file):
                    os.unlink(output_file)
            except OSError:
                pass

        if data is None:
            return None
        try:
            # pick_one_puzzle.py writes groups as a list of
            # {category, mechanism, title, words, ...}; other code may use
            # a dict keyed by color.
            raw = data.get('groups', {})
            if isinstance(raw, list):
                groups: dict = {}
                for g in raw:
                    if not isinstance(g, dict):
                        continue
                    color = (g.get('category') or '').strip().lower()
                    if color in ('yellow', 'green', 'blue', 'purple'):
                        groups[color] = g
            else:
                groups = raw

            def get_group(color):
                g = groups.get(color, {})
                if not isinstance(g, dict):
                    return [], '', 'abuzar_nlp'
                words = g.get('words', g.get('members', []))
                conn = (
                    g.get('connection')
                    or g.get('title')
                    or g.get('label')
                    or g.get('category')
                    or ''
                )
                mech = g.get('mechanism', 'abuzar_nlp')
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
            print(f'  [abuzar_nlp] normalisation error: {e}')
            return None