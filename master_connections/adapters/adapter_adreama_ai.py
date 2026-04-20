# ============================================================
# adapters/adapter_adreama_ai.py
# ============================================================

import os
import importlib.util
from typing import Optional
from adapters.base import PuzzleAdapter


class AdreamaAIAdapter(PuzzleAdapter):

    name = 'adreama'

    def __init__(self, notebook_dir: str, openai_api_key: str):
        self.notebook_dir    = notebook_dir
        self.openai_api_key  = openai_api_key
        self._loaded         = False
        self._generate_fn    = None

    def _load(self):
        if self._loaded:
            return

        os.environ['OPENAI_API_KEY'] = self.openai_api_key
        original_dir = os.getcwd()
        os.chdir(self.notebook_dir)

        try:
            spec = importlib.util.spec_from_file_location(
                'adreama_module',
                os.path.join(self.notebook_dir, 'adreama_module.py')
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._generate_fn = mod.generate_one_puzzle_entry
        except FileNotFoundError:
            raise RuntimeError(
                'adreama_module.py not found. '
                'Export the v11 notebook cells to adreama_module.py first.'
            )
        finally:
            os.chdir(original_dir)

        self._loaded = True

    def generate(self) -> Optional[dict]:
        try:
            self._load()
        except Exception as e:
            print(f'  [adreama] load error: {e}')
            return None

        try:
            result = self._generate_fn(verbose=False)
        except Exception as e:
            print(f'  [adreama] generation error: {e}')
            return None

        if result is None:
            return None

        try:
            g = result['groups']
            return self.canonical(
                yellow_words = g['yellow']['words'],
                yellow_conn  = g['yellow']['category'],
                green_words  = g['green']['words'],
                green_conn   = g['green']['category'],
                blue_words   = g['blue']['words'],
                blue_conn    = g['blue']['category'],
                purple_words = g['purple']['words'],
                purple_conn  = g['purple']['category'],
                yellow_mech  = 'llm_gpt41',
                green_mech   = 'llm_gpt41',
                blue_mech    = 'llm_gpt41',
                purple_mech  = 'llm_gpt41',
                meta         = {
                    'trap_word':         result.get('trap_word', ''),
                    'false_group':       result.get('false_group', ''),
                    'trap_impersonates': result.get('trap_impersonates', ''),
                }
            )
        except Exception as e:
            print(f'  [adreama] normalisation error: {e}')
            return None