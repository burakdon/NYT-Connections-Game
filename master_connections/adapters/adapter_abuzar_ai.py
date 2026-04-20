# ============================================================
# adapters/adapter_abuzar_ai.py
# ============================================================

import os
import json
import subprocess
import tempfile
from typing import Optional
from adapters.base import PuzzleAdapter


class AbuzarAIAdapter(PuzzleAdapter):

    name = 'abuzar_ai'

    def __init__(self, project_root: str, api_keys: dict = None):
        self.project_root = project_root
        self.api_keys     = api_keys or {}

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
                print(f'  [abuzar_ai] subprocess error: {result.stderr[:200]}')
                return None

            if not os.path.exists(output_file):
                print(f'  [abuzar_ai] no output file produced')
                return None

            with open(output_file) as f:
                data = json.load(f)

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
            return None
        try:
            groups = data.get('groups', {})

            def get_group(color):
                g     = groups.get(color, {})
                words = g.get('words', g.get('members', []))
                conn  = g.get('connection', g.get('category', g.get('label', '')))
                mech  = g.get('mechanism', 'abuzar_ai')
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