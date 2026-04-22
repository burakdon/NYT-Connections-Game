# ============================================================
# config.py
#
# Secrets: copy .env.example → .env in master_connections/ (never commit .env).
#
# Puzzle archive + dedup store (always under this package directory, not CWD):
#
# PIPELINE_WEIGHTS apply only among pipelines that actually registered in run.py
# (missing repos / failed imports are skipped). MasterGenerator renormalizes so
# the relative weights still sum to 1 over active pipelines.
# ============================================================

import os
from pathlib import Path

_MASTER_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MASTER_ROOT.parent
OUTPUT_PATH = _MASTER_ROOT / 'output' / 'generated_puzzles.json'


def nyt_blocklist_path() -> Path:
    """Path to NYT archive hash blocklist (shared with abuzar_AI-Connections)."""

    env = os.environ.get('MASTER_NYT_BLOCKLIST_PATH', '').strip()
    if env:
        return Path(env)
    return _REPO_ROOT / 'abuzar_AI-Connections' / 'data' / 'nyt_blocklist.json'


NYT_BLOCKLIST_PATH = nyt_blocklist_path()

PIPELINE_WEIGHTS = {
    'burak':       0.40,
    'adreama':     0.20,
    # Intentionally 0 — do not enable without review (remix touches real NYT-derived data).
    'kevin_remix': 0,
    'kevin_fresh': 0.15,
    'abuzar_nlp':  0.20,
    'abuzar_ai':   0.05,
}

PIPELINE_TIMEOUTS = {
    'burak':       60,
    'adreama':     45,
    'kevin_remix': 30,
    'kevin_fresh': 45,
    'abuzar_nlp':  120,
    # Wall-clock for subprocess: Abuzar runs ~8 Claude calls (2 per color × 4); each HTTP call
    # allows up to 120s in agents/claude_client.py — 180s was far too low and caused false timeouts.
    'abuzar_ai':   900,
}

MAX_RETRIES_PER_PIPELINE = 3
MAX_TOTAL_ATTEMPTS       = 15
