# ============================================================
# dedup.py
# ============================================================

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Union


StorePath = Union[str, Path]


class DedupStore:
    """
    Persistent deduplication across all pipelines.

    A puzzle is rejected as a duplicate if:
      - Its board signature (16 words sorted, joined) matches an earlier puzzle, OR
      - Any of its four groups matches an earlier group's word set (same 4 words,
        order ignored). Connection strings are not part of the group signature.
    """

    def __init__(self, store_path: StorePath = 'output/generated_puzzles.json'):
        self.store_path = os.fspath(store_path)
        self._puzzles: list = []
        self._board_sigs: set = set()
        self._group_sigs: set = set()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.store_path):
            print(f'DedupStore: starting empty ({self.store_path})')
            return
        try:
            with open(self.store_path, encoding='utf-8') as f:
                raw = f.read()
            if not raw.strip():
                print(f'DedupStore: empty file, starting fresh ({self.store_path})')
                return
            self._puzzles = json.loads(raw)
        except json.JSONDecodeError as e:
            bak = f'{self.store_path}.corrupt.{int(time.time())}'
            try:
                os.replace(self.store_path, bak)
            except OSError:
                bak = '(could not move corrupt file)'
            print(
                f'DedupStore: invalid JSON ({e}); archived to {bak}. Starting empty.'
            )
            self._puzzles = []

        if not isinstance(self._puzzles, list):
            print('DedupStore: root JSON is not a list — starting empty.')
            self._puzzles = []

        for p in self._puzzles:
            self._index(p)
        print(f'DedupStore: {len(self._puzzles)} puzzles loaded from {self.store_path}')

    def _board_signature(self, puzzle: dict) -> str:
        all_words = [
            w.lower()
            for color in ['yellow', 'green', 'blue', 'purple']
            for w in puzzle['groups'][color]['words']
        ]
        return '|'.join(sorted(all_words))

    def _group_signature(self, words: list, _connection: str) -> str:
        return '|'.join(sorted(w.lower() for w in words))

    def _index(self, puzzle: dict) -> None:
        self._board_sigs.add(self._board_signature(puzzle))
        for color in ['yellow', 'green', 'blue', 'purple']:
            g = puzzle['groups'][color]
            self._group_sigs.add(self._group_signature(g['words'], g['connection']))

    def is_duplicate(self, puzzle: dict) -> tuple:
        board_sig = self._board_signature(puzzle)
        if board_sig in self._board_sigs:
            return True, 'exact board duplicate'

        for color in ['yellow', 'green', 'blue', 'purple']:
            g = puzzle['groups'][color]
            group_sig = self._group_signature(g['words'], g['connection'])
            if group_sig in self._group_sigs:
                return True, f'duplicate {color} group (words): {g["words"]}'

        return False, ''

    def save(self, puzzle: dict) -> None:
        self._puzzles.append(puzzle)
        self._index(puzzle)
        parent = os.path.dirname(os.path.abspath(self.store_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            suffix='.json',
            prefix='generated_puzzles.',
            dir=parent or '.',
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self._puzzles, f, indent=2)
                f.write('\n')
            os.replace(tmp_path, self.store_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @property
    def count(self) -> int:
        return len(self._puzzles)
