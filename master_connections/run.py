# ============================================================
# run.py
# ============================================================

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

MASTER_DIR = Path(__file__).resolve().parent
ROOT = MASTER_DIR.parent
load_dotenv(MASTER_DIR / '.env')

from master_generator import MasterGenerator

from adapters.adapter_burak       import BurakAdapter
from adapters.adapter_adreama_ai  import AdreamaAIAdapter
from adapters.adapter_kevin       import KevinAdapter
from adapters.adapter_abuzar_nlp  import AbuzarNLPAdapter
from adapters.adapter_abuzar_ai   import AbuzarAIAdapter


def _repo_ready(root: Path, *parts: str) -> bool:
    """True if root / parts exists (file or directory)."""
    p = root.joinpath(*parts)
    return p.exists()


def _warn_missing_secrets(adapters: dict) -> None:
    """Print warnings when a registered pipeline likely needs keys that are unset."""
    needs = {
        'burak':       ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY'),
        'adreama':     ('OPENAI_API_KEY',),
        'kevin_fresh': ('OPENAI_API_KEY',),
        'abuzar_ai':   ('OPENAI_API_KEY',),
    }
    any_warn = False
    for pipeline, keys in needs.items():
        if pipeline not in adapters:
            continue
        missing = [k for k in keys if not os.environ.get(k, '').strip()]
        if missing:
            any_warn = True
            print(
                f'  Warning: pipeline [{pipeline}] needs env: {", ".join(missing)} '
                f'(see .env.example)'
            )
    if any_warn:
        print('  (Abuzar NLP uses local datasets — no API key required.)\n')


def build_adapters() -> dict:
    adapters = {}

    # ── Burak ─────────────────────────────────────────────────
    try:
        from burak_pipeline import run_full_pipeline
        adapters['burak'] = BurakAdapter(generate_fn=run_full_pipeline)
        print('Registered: burak')
    except ImportError:
        print('Skipped: burak (burak_pipeline.py not found)')

    # ── Adreama (v11, GPT-4.1) ────────────────────────────────
    try:
        adapters['adreama'] = AdreamaAIAdapter(
            notebook_dir=os.path.dirname(os.path.abspath(__file__)),
            openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
        )
        print('Registered: adreama')
    except Exception as e:
        print(f'Skipped: adreama ({e})')

    # Kevin's repo needs to be cloned into the Connections Project folder:
    # git clone https://github.com/Kevin2330/nyt-connections-generator kevin_repo
    # cd kevin_repo && pip install -r requirements.txt

    kevin_root = ROOT / 'kevin_repo'
    if not _repo_ready(kevin_root, 'scripts', 'cfr', 'generate_cfr.py'):
        print('Skipped: kevin_fresh (kevin_repo/scripts/cfr/generate_cfr.py not found)')
    else:
        try:
            adapters['kevin_fresh'] = KevinAdapter(
                project_root=os.fspath(kevin_root),
                mode='fresh',
                api_key=os.environ.get('OPENAI_API_KEY', ''),
            )
            print('Registered: kevin_fresh')
        except Exception as e:
            print(f'Skipped: kevin_fresh ({e})')

    # ── Abuzar NLP ────────────────────────────────────────────
    nlp_root = ROOT / 'abuzar_NLP-Connections-datasets-generators'
    if not _repo_ready(nlp_root, 'pick_one_puzzle.py'):
        print('Skipped: abuzar_nlp (abuzar_NLP-Connections-datasets-generators/pick_one_puzzle.py not found)')
    else:
        try:
            adapters['abuzar_nlp'] = AbuzarNLPAdapter(
                project_root=os.fspath(nlp_root),
            )
            print('Registered: abuzar_nlp')
        except Exception as e:
            print(f'Skipped: abuzar_nlp ({e})')

    # ── Abuzar AI ─────────────────────────────────────────────
    ai_root = ROOT / 'abuzar_AI-Connections'
    if not _repo_ready(ai_root, 'run_fresh_puzzle.py'):
        print('Skipped: abuzar_ai (abuzar_AI-Connections/run_fresh_puzzle.py not found)')
    else:
        try:
            adapters['abuzar_ai'] = AbuzarAIAdapter(
                project_root=os.fspath(ai_root),
                api_keys={'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', '')},
            )
            print('Registered: abuzar_ai')
        except Exception as e:
            print(f'Skipped: abuzar_ai ({e})')

    return adapters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Master NYT Connections generator — multiplexes contributor pipelines.',
    )
    parser.add_argument('--n', type=int, default=1, help='Number of puzzles (normal mode).')
    parser.add_argument('--preview', action='store_true', help='Verbose batch output.')
    parser.add_argument(
        '--only',
        metavar='PIPELINE',
        help='Use only this pipeline name, e.g. burak, adreama, kevin_fresh, abuzar_nlp, abuzar_ai.',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='One generate per adapter; validate only; nothing written to output store.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Register adapters, print weights, exit — no generation, no API calls.',
    )
    args = parser.parse_args()

    if args.dry_run:
        adapters = build_adapters()
        if not adapters:
            print('No pipelines registered.')
            raise SystemExit(1)
        _warn_missing_secrets(adapters)
        if args.only:
            if args.only not in adapters:
                print(f'Unknown pipeline {args.only!r}. Registered: {", ".join(sorted(adapters))}')
                raise SystemExit(1)
            adapters = {args.only: adapters[args.only]}
        MasterGenerator(adapters)
        print('\nDry run complete (no puzzles generated).')
        raise SystemExit(0)

    adapters = build_adapters()
    if not adapters:
        print('No pipelines registered. Fix imports (burak/adreama), .env keys, and clone sibling repos.')
        raise SystemExit(1)

    _warn_missing_secrets(adapters)

    if args.only:
        if args.only not in adapters:
            print(f'Unknown pipeline {args.only!r}. Registered: {", ".join(sorted(adapters))}')
            raise SystemExit(1)
        adapters = {args.only: adapters[args.only]}

    generator = MasterGenerator(adapters)

    if args.smoke:
        print('Smoke test (no saves to output/generated_puzzles.json):\n')
        outcomes = generator.smoke_all(verbose=True)
        bad = [k for k, v in outcomes.items() if v != 'ok']
        if bad:
            print(f'\nSmoke finished with issues: {bad}')
            raise SystemExit(1)
        print('\nSmoke: all adapters returned a valid puzzle.')
        raise SystemExit(0)

    if args.n == 1:
        generator.generate_one(verbose=True)
    else:
        generator.generate_batch(n=args.n, verbose=args.preview)

    print('\nPipeline stats:')
    for source, count in generator.stats.items():
        print(f'  {source:20s}: {count}')