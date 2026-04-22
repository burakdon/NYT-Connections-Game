# ============================================================
# adapters/adapter_kevin.py
#
# Wraps Kevin's nyt-connections-generator repo.
# Repo: https://github.com/Kevin2330/nyt-connections-generator
#
# Entry point: scripts/cfr/generate_cfr.py
# Output: data/generated/cfr/<mode>/<timestamp>.json
#
# Requires:
#   - OPENAI_API_KEY env var (for fresh mode)
#   - data/nyt_puzzles/ConnectionsFinalDataset (1).json
#   - pip install -r requirements.txt run inside Kevin's repo
# ============================================================

import os
import json
import glob
import subprocess
import time
from typing import Optional
from adapters.base import PuzzleAdapter


class KevinAdapter(PuzzleAdapter):

    def __init__(self, project_root: str, mode: str = 'fresh',
                 api_key: str = None):
        """
        project_root: path to Kevin's nyt-connections-generator folder
        mode: 'fresh' (1 LLM call) or 'remix' (0 LLM calls, NYT risk)
        """
        self.project_root = project_root
        self.mode         = mode
        self.api_key      = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.name         = f'kevin_{mode}'
        if self.mode == 'fresh':
            self.max_master_attempts = 2

        # Kevin's output directory for this mode
        self.output_dir = os.path.join(
            project_root, 'data', 'generated', 'cfr', mode
        )

    def generate(self) -> Optional[dict]:
        # Record existing files before running so we can find the new one
        before = set(glob.glob(os.path.join(self.output_dir, '*.json')))

        env = os.environ.copy()
        env['OPENAI_API_KEY'] = self.api_key
        env['DRY_RUN']        = 'false' if self.mode == 'fresh' else 'true'
        env['LLM_PROVIDER']   = 'openai'

        try:
            result = subprocess.run(
                [
                    'python',
                    os.path.join('scripts', 'cfr', 'generate_cfr.py'),
                    '--mode',  self.mode,
                    '--count', '1',
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

            if result.returncode != 0:
                print(f'  [{self.name}] subprocess error: {result.stderr[:300]}')
                return None

        except subprocess.TimeoutExpired:
            print(f'  [{self.name}] timed out')
            return None
        except Exception as e:
            print(f'  [{self.name}] error: {e}')
            return None

        # Find the newly created output file
        after   = set(glob.glob(os.path.join(self.output_dir, '*.json')))
        new_files = after - before

        # Also check invalid file — Kevin writes rejects separately
        new_files = {f for f in new_files if 'invalid' not in f}

        if not new_files:
            print(f'  [{self.name}] no output file found in {self.output_dir}')
            return None

        # Take the most recent (leave file on disk — Kevin artifact under kevin_repo/)
        output_file = max(new_files, key=os.path.getmtime)

        try:
            with open(output_file) as f:
                data = json.load(f)

            # Kevin writes a list of puzzles — take the first/only one
            if isinstance(data, list):
                if not data:
                    print(f'  [{self.name}] empty puzzle list')
                    return None
                data = data[0]

        except Exception as e:
            print(f'  [{self.name}] could not read output: {e}')
            return None

        try:
            return self._normalise(data)
        except Exception as e:
            print(f'  [{self.name}] normalisation error: {e}')
            return None

    def _normalise(self, data: dict) -> Optional[dict]:
        """
        Kevin's puzzle schema (from master_demo.ipynb show_puzzle):
        puzzle['groups'] is a LIST of dicts, each with:
          - 'color': 'yellow'|'green'|'blue'|'purple'
          - 'category': str
          - 'words': list of 4 strings
          - 'similarity_score': float (optional)

        Color is assigned by difficulty (yellow=easiest, purple=hardest).
        """
        groups = data.get('groups', [])

        # Build color map
        color_map = {}
        for g in groups:
            color = g.get('color', '').lower()
            if color in ('yellow', 'green', 'blue', 'purple'):
                color_map[color] = g

        # Fallback: assign by order if color field missing
        if len(color_map) < 4:
            for i, color in enumerate(['yellow', 'green', 'blue', 'purple']):
                if i < len(groups) and color not in color_map:
                    color_map[color] = groups[i]

        if len(color_map) < 4:
            print(f'  [{self.name}] could not build full color map: {list(color_map.keys())}')
            return None

        return self.canonical(
            yellow_words = color_map['yellow']['words'],
            yellow_conn  = color_map['yellow']['category'],
            green_words  = color_map['green']['words'],
            green_conn   = color_map['green']['category'],
            blue_words   = color_map['blue']['words'],
            blue_conn    = color_map['blue']['category'],
            purple_words = color_map['purple']['words'],
            purple_conn  = color_map['purple']['category'],
            yellow_mech  = f'kevin_cfr_{self.mode}',
            green_mech   = f'kevin_cfr_{self.mode}',
            blue_mech    = f'kevin_cfr_{self.mode}',
            purple_mech  = f'kevin_cfr_{self.mode}',
            meta         = {
                'mode':              self.mode,
                'similarity_scores': {
                    g.get('color', f'group_{i}'): g.get('similarity_score', 0.0)
                    for i, g in enumerate(groups)
                },
                'solver_validation': data.get('solver_validation', {}),
            }
        )
