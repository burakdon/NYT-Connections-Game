# Master Connections (NYT-Connections-Game)

Multi-pipeline New York Times Connections-style puzzle generator for ECE682.

This monorepo combines multiple contributor pipelines behind a shared canonical schema,
a weighted orchestrator, and a persistent dedup store.

## Repo layout

- `master_connections/` - main orchestrator, adapters, dedup store, static webapp.
- `kevin_repo/` - Kevin CFR pipeline source.
- `abuzar_AI-Connections/` - Abuzar AI pipeline + local puzzle/group banks.
- `abuzar_NLP-Connections-datasets-generators/` - Abuzar NLP pipeline.

## Quick start

From `master_connections/`:

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in keys in .env
```

Required keys depend on pipeline:

- `ANTHROPIC_API_KEY`: Burak + Abuzar AI
- `OPENAI_API_KEY`: Adreama + Kevin fresh (+ some Burak paths)

## Generate puzzles

### Mixed weighted generation

```bash
cd master_connections
python run.py --n 20
```

### Pipeline-specific generation

```bash
python run.py --only burak --n 5
python run.py --only adreama --n 5
python run.py --only kevin_fresh --n 5
python run.py --only abuzar_nlp --n 5
python run.py --only abuzar_ai --n 1
```

## Publish puzzles to the web app

The browser reads `master_connections/webapp/puzzles.json`.

After generation:

```bash
cd master_connections
python scripts/export_web_puzzles.py
```

This converts canonical puzzles in `output/generated_puzzles.json` to web format.

## Local web test

```bash
cd master_connections
python -m http.server 8080 --directory webapp
```

Open `http://localhost:8080/`.

## GitHub Pages deploy

- Workflow: `/.github/workflows/deploy-pages.yml`
- Published directory: `master_connections/webapp`
- Pages setting: **Settings -> Pages -> Source: GitHub Actions**

Live URL pattern:

`https://<user>.github.io/<repo>/`

## Common gotcha: site shows old or too few puzzles

Do all 3 steps, in order:

1. Generate (`python run.py ...`)
2. Export (`python scripts/export_web_puzzles.py`)
3. Commit + push `master_connections/webapp/puzzles.json`

If the browser still shows stale counts, hard refresh or open in a private window.

## Notes

- `master_connections/output/generated_puzzles.json` is the canonical local archive/dedup source.
- `.env` should never be committed.
- `kevin_remix` stays disabled (`0` weight) by design.
