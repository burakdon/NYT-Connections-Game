# Static playable UI

Vanilla HTML/CSS/JS (same interaction model as Kevin’s `kevin_repo/webapp/`). Puzzle data is loaded from `puzzles.json`.

## Refresh `puzzles.json` from generated puzzles

From the `master_connections/` directory (same folder as `run.py`):

```bash
python scripts/export_web_puzzles.py
```

This reads `output/generated_puzzles.json` (ignored by git — your local archive) and writes `webapp/puzzles.json`. Commit **`webapp/puzzles.json`** when you want the GitHub Pages site to ship new boards.

## Try locally

```bash
python -m http.server 8080 --directory webapp
```

Open http://localhost:8080/

## GitHub Pages

1. Push this repo (with `webapp/` and committed `puzzles.json`).
2. **Settings → Pages → Build and deployment**: source **GitHub Actions**.
3. The workflow `.github/workflows/deploy-pages.yml` publishes the **`webapp/`** folder.

Your site URL will look like `https://<user>.github.io/<repo>/`.

### Monorepo note

If the git root is *above* `master_connections/`, move or adjust the workflow so the upload path points at your `webapp` directory (e.g. `path: master_connections/webapp`), and update any `paths` filters accordingly.
