# ============================================================
# adapters/adapter_burak.py
# ============================================================

from typing import Optional, Callable
from adapters.base import PuzzleAdapter


class BurakAdapter(PuzzleAdapter):

    name = 'burak'

    def __init__(self, generate_fn: Callable):
        self.generate_fn = generate_fn

    def generate(self) -> Optional[dict]:
        try:
            result = self.generate_fn()
        except Exception as e:
            print(f'  [burak] generation error: {e}')
            return None

        if result is None:
            return None

        try:
            return self.canonical(
                yellow_words = result['yellow']['group'],
                yellow_conn  = result['yellow']['connection'],
                green_words  = result['green']['group'],
                green_conn   = result['green']['connection'],
                blue_words   = result['blue']['group'],
                blue_conn    = result['blue']['connection'],
                purple_words = result['purple']['group'],
                purple_conn  = result['purple']['connection'],
                yellow_mech  = result['yellow'].get('mechanism', 'word2vec'),
                green_mech   = result['green'].get('mechanism', 'rhyme'),
                blue_mech    = result['blue'].get('mechanism', 'llm_niche'),
                purple_mech  = result['purple'].get('mechanism', ''),
                meta         = {
                    'impostor': result.get('impostor_result', {}),
                }
            )
        except Exception as e:
            print(f'  [burak] normalisation error: {e}')
            return None