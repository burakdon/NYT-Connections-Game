# ============================================================
# master_generator.py
# ============================================================

import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from config import (
    NYT_BLOCKLIST_PATH,
    PIPELINE_WEIGHTS,
    MAX_RETRIES_PER_PIPELINE,
    MAX_TOTAL_ATTEMPTS,
    OUTPUT_PATH,
)
from dedup import DedupStore


def _load_nyt_archive_guard() -> tuple[Optional[Callable[[dict], dict]], str, bool]:
    """
    Import Abuzar's hash-only NYT guard so every pipeline is checked the same way.
    Returns (check_fn, status_message, blocklist_ready) or (None, reason, False).
    """
    repo_root = Path(__file__).resolve().parent.parent
    abuzar_root = repo_root / 'abuzar_AI-Connections'
    if not abuzar_root.is_dir():
        return None, 'abuzar_AI-Connections/ not found beside master_connections/', False
    agents_pkg = abuzar_root / 'agents' / 'nyt_guard.py'
    if not agents_pkg.is_file():
        return None, 'agents/nyt_guard.py missing under abuzar_AI-Connections/', False
    if str(abuzar_root) not in sys.path:
        sys.path.insert(0, str(abuzar_root))
    try:
        from agents.nyt_guard import blocklist_status, check_puzzle_against_blocklist
    except Exception as exc:
        return None, f'import failed: {exc!r}', False

    st = blocklist_status(NYT_BLOCKLIST_PATH)
    msg = (
        f"path={st['path']} ready={st['ready']} "
        f"boards={st['board_count']} group_sets={st['group_set_count']}"
    )

    def _check(puzzle: dict) -> dict:
        return check_puzzle_against_blocklist(
            puzzle,
            path=NYT_BLOCKLIST_PATH,
            require_ready=False,
        )

    return _check, msg, bool(st.get('ready', False))


class MasterGenerator:

    def __init__(self, adapters: dict):
        self.adapters = adapters
        self.store    = DedupStore(OUTPUT_PATH)

        self.weights = {
            name: w
            for name, w in PIPELINE_WEIGHTS.items()
            if name in adapters and w > 0
        }
        total        = sum(self.weights.values())
        if total <= 0:
            raise ValueError(
                'Pipeline weights sum to zero for registered adapters — check config.PIPELINE_WEIGHTS.'
            )
        self.weights = {k: v / total for k, v in self.weights.items()}

        print(f'MasterGenerator ready.')
        print(f'  Pipelines : {list(self.adapters.keys())}')
        ignored = [
            n for n in adapters
            if n in PIPELINE_WEIGHTS and PIPELINE_WEIGHTS[n] <= 0
        ]
        if ignored:
            print(f'  Weight 0 (excluded): {ignored}')
        print(f'  Weights   : {self.weights}')
        print(f'  Store     : {self.store.count} puzzles so far')

        self._nyt_archive_check, nyt_msg, nyt_ready = _load_nyt_archive_guard()
        if self._nyt_archive_check is None:
            print(f'  NYT archive guard: OFF ({nyt_msg})')
        else:
            print(f'  NYT archive guard: ON ({nyt_msg})')
            if not nyt_ready:
                print(
                    '    Hint: build abuzar_AI-Connections/data/nyt_blocklist.json '
                    'with build_nyt_blocklist.py to enable archive matching.'
                )

    def _pick_pipeline(self) -> str:
        names   = list(self.weights.keys())
        weights = [self.weights[n] for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    def _attempt_cap(self, name: str) -> int:
        cap = getattr(self.adapters[name], 'max_master_attempts', None)
        if isinstance(cap, int) and cap > 0:
            return cap
        return MAX_TOTAL_ATTEMPTS

    def generate_one(self, verbose: bool = True) -> Optional[dict]:
        attempts = 0
        tried    = {}

        while attempts < MAX_TOTAL_ATTEMPTS:
            eligible = [
                name for name in self.weights
                if tried.get(name, 0) < self._attempt_cap(name)
            ]
            if not eligible:
                print(f'  Gave up after {attempts} attempts.')
                return None

            attempts += 1
            if len(eligible) == len(self.weights):
                name = self._pick_pipeline()
            else:
                names   = eligible
                weights = [self.weights[n] for n in names]
                name    = random.choices(names, weights=weights, k=1)[0]
            tried[name]   = tried.get(name, 0) + 1
            if tried[name] > MAX_RETRIES_PER_PIPELINE:
                name = min(eligible, key=lambda n: tried.get(n, 0))

            if verbose:
                print(f'  Attempt {attempts}: [{name}]...', end=' ', flush=True)

            t0 = time.time()
            try:
                puzzle = self.adapters[name].generate()
            except Exception as e:
                if verbose:
                    print(f'error — {e!r}')
                continue
            elapsed = time.time() - t0

            if puzzle is None:
                if verbose: print(f'None ({elapsed:.1f}s)')
                continue

            is_dup, reason = self.store.is_duplicate(puzzle)
            if is_dup:
                if verbose: print(f'duplicate — {reason}')
                continue

            ok, reason = self._validate(puzzle)
            if not ok:
                if verbose: print(f'invalid — {reason}')
                continue

            if self._nyt_archive_check is not None:
                nyt = self._nyt_archive_check(puzzle)
                for w in nyt.get('warnings', []):
                    if verbose:
                        print(f'  NYT guard warning: {w}')
                if not nyt.get('ok', True):
                    if verbose:
                        for err in nyt.get('errors', []):
                            print(f'nyt archive — {err}')
                    continue

            self.store.save(puzzle)
            if verbose:
                print(f'OK ({elapsed:.1f}s) → puzzle #{self.store.count}')
                self._preview(puzzle)

            return puzzle

        print(f'  Gave up after {attempts} attempts.')
        return None

    def smoke_all(self, verbose: bool = True) -> dict:
        """
        One generate() per registered adapter; validates shape; does not save to disk.
        """
        results = {}
        for name in sorted(self.adapters.keys()):
            if verbose:
                print(f'  [{name}] ...', end=' ', flush=True)
            t0     = time.time()
            puzzle = self.adapters[name].generate()
            elapsed = time.time() - t0
            if puzzle is None:
                results[name] = 'no_output'
                if verbose:
                    print(f'no puzzle ({elapsed:.1f}s)')
                continue
            ok, reason = self._validate(puzzle)
            if not ok:
                results[name] = f'invalid: {reason}'
                if verbose:
                    print(f'invalid ({elapsed:.1f}s) — {reason}')
                continue
            if self._nyt_archive_check is not None:
                nyt = self._nyt_archive_check(puzzle)
                if not nyt.get('ok', True):
                    err = '; '.join(nyt.get('errors', [])) or 'nyt archive match'
                    results[name] = f'nyt_archive: {err}'
                    if verbose:
                        print(f'nyt archive ({elapsed:.1f}s) — {err}')
                    continue
                for w in nyt.get('warnings', []):
                    if verbose:
                        print(f'    NYT guard warning: {w}')
            results[name] = 'ok'
            if verbose:
                print(f'OK ({elapsed:.1f}s)')
        return results

    def generate_batch(self, n: int, verbose: bool = True) -> list:
        results = []
        for i in range(n):
            print(f'\n[{i+1}/{n}] Generating...')
            puzzle = self.generate_one(verbose=verbose)
            if puzzle:
                results.append(puzzle)
        print(f'\nDone. Generated {len(results)}/{n} puzzles.')
        return results

    def _validate(self, puzzle: dict) -> tuple:
        if 'groups' not in puzzle:
            return False, 'missing groups'
        for color in ['yellow', 'green', 'blue', 'purple']:
            if color not in puzzle['groups']:
                return False, f'missing {color}'
            g = puzzle['groups'][color]
            if len(g.get('words', [])) != 4:
                return False, f'{color} has {len(g.get("words", []))} words'
        all_words = [
            w.lower()
            for color in ['yellow', 'green', 'blue', 'purple']
            for w in puzzle['groups'][color]['words']
        ]
        if len(all_words) != 16:
            return False, f'expected 16 words, got {len(all_words)}'
        if len(set(all_words)) != 16:
            dupes = {w for w in all_words if all_words.count(w) > 1}
            return False, f'duplicate words: {dupes}'
        return True, ''

    def _preview(self, puzzle: dict):
        print(f'  Source  : {puzzle["source"]}')
        for color in ['yellow', 'green', 'blue', 'purple']:
            g = puzzle['groups'][color]
            print(f'  {color.upper():7}: {g["connection"][:40]:40} {g["words"]}')
        print()

    @property
    def stats(self) -> dict:
        counts = {}
        for p in self.store._puzzles:
            src = p.get('source', 'unknown')
            counts[src] = counts.get(src, 0) + 1
        return counts
