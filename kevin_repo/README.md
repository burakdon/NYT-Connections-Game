# Infinite Connections

An AI system that generates, validates, and serves NYT-style Connections puzzles at scale. Built around two complementary pipelines: a traditional LLM-heavy baseline and a novel **Category-First Retrieval (CFR)** pipeline that reduces LLM usage by 80-100% while improving quality.

**[Live web app](https://kevin2330.github.io/nyt-connections-generator/)** -- 542 validated puzzles, playable in the browser.

---

<!--
Screenshot placeholder: replace the path below with an actual screenshot
of the web app once available (e.g., docs/images/webapp_screenshot.png).
-->

```
+------------------------------------------------------+
|             Infinite Connections                      |
|                                                      |
|  +----------+ +----------+ +----------+ +----------+ |
|  |  SWIFT   | |  MARS    | |  POKER   | |  SALSA   | |
|  +----------+ +----------+ +----------+ +----------+ |
|  +----------+ +----------+ +----------+ +----------+ |
|  |  RAPID   | |  VENUS   | |  BRIDGE  | |  WALTZ   | |
|  +----------+ +----------+ +----------+ +----------+ |
|  +----------+ +----------+ +----------+ +----------+ |
|  |  FLEET   | |  SATURN  | |  HEARTS  | |  TANGO   | |
|  +----------+ +----------+ +----------+ +----------+ |
|  +----------+ +----------+ +----------+ +----------+ |
|  |  QUICK   | |  JUPITER | |  RUMMY   | |  BALLET  | |
|  +----------+ +----------+ +----------+ +----------+ |
|                                                      |
|          [ Shuffle ]  [ Deselect ]  [ Submit ]       |
+------------------------------------------------------+
```

---

## Headline Results

| Pipeline | LLM calls/puzzle | Pass rate | Time/puzzle | Cost/1k puzzles | Non-NYT words |
|---|---:|---:|---:|---:|---:|
| A -- LLM-heavy baseline (gpt-4o-mini) | 5 | 91% | ~10 s | ~$1.00 | -- |
| **B -- CFR v2 Mode A (remix)** | **0** | **99%** | **1.27 s** | **$0.00** | **62.6%** |
| **B -- CFR v2 Mode B (fresh)** | **1** | **99%** | **2.77 s** | **$0.05** | **69.1%** |

**Zero verbatim past-NYT group reproductions** across 200 benchmark puzzles. **14,877-word bank** (3x larger than NYT's original 4,918). Rubric hard-fail rule enforced by construction.

See [`docs/methods.pdf`](docs/methods.pdf) for the full technical writeup and [`docs/executive_summary.pdf`](docs/executive_summary.pdf) for the 2-page overview.

---

## Quick Start

Everything runs out of the box in dry-run mode. No API keys are needed.

**0. Download the NYT dataset** (not included in the repo for copyright reasons)

Place the Connections dataset JSON file at:

```
data/nyt_puzzles/ConnectionsFinalDataset (1).json
```

The dataset contains 554 puzzles and is available from Kaggle. Pipelines and the web app work without it (they use mock / pre-bundled data), but the data exploration notebook and solver benchmark require it.

**1. Install dependencies**

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"   # only needed for CFR
```

**2. Run the test suite**

```bash
python -m pytest tests/ -v
```

**3. Launch the web app locally** (or just open the live GitHub Pages site)

```bash
# Local Flask version (legacy)
python src/webapp/app.py

# Or serve the static site that's deployed on GitHub Pages
python -m http.server 8080 -d webapp/
```

**4. Open the notebooks**

```bash
jupyter notebook notebooks/
```

Available notebooks:

- `infinite_connections_demo.ipynb` -- End-to-end pipeline demonstration.
- `nyt_data_exploration.ipynb` -- Analysis of 554 NYT puzzles (frequency, categories, MPNET embedding patterns).

**5. Generate real puzzles** (see [Generating Real Puzzles](#generating-real-puzzles))

```bash
# Pipeline A baseline (5 LLM calls / puzzle)
OPENAI_API_KEY=sk-... DRY_RUN=false python scripts/generate_puzzles.py --count 100

# Pipeline B CFR Mode A -- 0 LLM calls, free, fastest
DRY_RUN=true python scripts/cfr/generate_cfr.py --mode remix --count 100

# Pipeline B CFR Mode B -- 1 LLM call per puzzle, fresh categories
OPENAI_API_KEY=sk-... DRY_RUN=false python scripts/cfr/generate_cfr.py --mode fresh --count 100
```

---

## Project Structure

```
infinite-connections/
|
|-- data/
|   |-- nyt_puzzles/              Ground truth: 554 NYT puzzles (JSON)
|   |-- cache/                    Precomputed MPNET embeddings (ignored)
|   |-- generated/                Pipeline A outputs
|   |-- generated/cfr/            Pipeline B v1 outputs
|   |-- generated/cfr_v2/         Pipeline B v2 outputs (augmented bank)
|   +-- mock/mock_puzzles.json    5 hand-crafted puzzles for dry-run mode
|
|-- src/
|   |-- config.py                 Central configuration: DRY_RUN, paths, thresholds
|   |-- llm_client.py             Unified OpenAI / Anthropic / Mock LLM client
|   |
|   |-- generator/                Pipeline A (LLM-heavy baseline)
|   |   |-- pipeline.py           Orchestrator (iterative + false-group)
|   |   |-- group_creator.py      LLM + MPNET word selection (8 -> 4)
|   |   |-- puzzle_editor.py      Second-pass LLM review
|   |   |-- prompts.py            Prompt templates
|   |   |-- difficulty.py         Color assignment (shared with Pipeline B)
|   |   +-- deduplicator.py       16-word overlap + 4-word-group match
|   |
|   |-- cfr/                      Pipeline B: Category-First Retrieval (novel)
|   |   |-- pipeline.py           CFRPipeline (Mode A + Mode B)
|   |   |-- embedding_retriever.py  KNN over MPNET word embeddings
|   |   |-- word_bank.py          Augmented 14,877-word bank (NYT + WordNet)
|   |   +-- prompts.py            Batched-categories prompt (Mode B only)
|   |
|   |-- solvers/                  Multiple independent solver implementations
|   |   |-- embedding_solver.py   Greedy C(16,4) enumeration by cosine sim
|   |   |-- clustering_solver.py  Group-Penalty scoring + beam search
|   |   |-- llm_solver.py         Chain-of-thought LLM solver
|   |   +-- roundtable.py         Multi-solver convergence validator
|   |
|   |-- evaluation/               Quality metrics and analysis
|   |
|   +-- webapp/                   Legacy Flask backend (still supported)
|
|-- webapp/                       Static site deployed to GitHub Pages
|   |-- index.html
|   |-- game.js                   Vanilla JS; loads puzzles.json client-side
|   |-- style.css
|   +-- puzzles.json              Bundled set of 542 validated puzzles
|
|-- notebooks/                    Jupyter demos
|-- scripts/
|   |-- generate_puzzles.py       Pipeline A CLI
|   |-- cfr/generate_cfr.py       Pipeline B CLI (--mode remix|fresh)
|   |-- split_by_color.py         Extract per-color word lists
|   |-- benchmark_solvers.py      Run solvers on all 554 NYT puzzles
|   |-- plot_benchmark.py         Chart solver accuracy
|   +-- build_*_pdf.py            Regenerate PDF documentation
|
|-- tests/
|-- docs/
|   |-- executive_summary.tex/pdf     2-page non-technical summary
|   |-- methods.tex/pdf               Full technical methods document
|   |-- study_guide.tex/pdf           11-page learning guide
|   |-- faq.md                        Anticipated questions
|   +-- technical_appendix.md         Extra algorithm notes
|
|-- papers/                       Reference PDFs (read-only)
|-- repos/                        Cloned reference repos (read-only)
|-- requirements.txt
+-- README.md
```

---

## How It Works

We built **two pipelines** that share the same post-generation validation.

### Pipeline A -- LLM-Heavy Baseline (reference implementation)

Follows the architecture from "Making New Connections" (arXiv:2407.11240):

1. **Group creation** -- An LLM proposes a category name and 8 candidate words. Story injection (random seed words from the NYT bank) prevents repetitive output.
2. **MPNET selection** -- Of the 8 candidates, the 4 most internally cohesive are chosen by enumerating all C(8,4)=70 subsets and picking the one with highest avg pairwise cosine similarity. The LLM's word choices are NOT trusted directly.
3. **Iteration** -- Steps 1-2 repeat 4 times, with previous groups passed as context.
4. **Editor pass** -- A second LLM call reviews the complete puzzle and rewrites inaccurate category names.
5. **Difficulty assignment** -- Groups sorted by cohesion -> yellow / green / blue / purple.
6. **Deduplication + solver validation** (shared with Pipeline B).

**Cost:** 5 LLM calls per puzzle, ~$0.001/puzzle with gpt-4o-mini, ~10s wall time.

### Pipeline B -- Category-First Retrieval (CFR, our contribution)

Inverts Pipeline A: instead of the LLM generating words, it generates only category NAMES (or none at all), and the words are retrieved by k-nearest-neighbors over MPNET embeddings.

**Offline precomputation** (~5 minutes, once):
- Build augmented word bank: 4,918 NYT words + ~10,000 common WordNet lemmas = **14,877 words**
- Encode every word and every NYT category with MPNET
- Build a cosine-distance KNN index
- Precompute the set of all 2,216 past NYT 4-word groups (as frozensets)

**Per puzzle:**

```
Step 1 -- Get 4 category names:
    Mode A (remix): sample 4 diverse past NYT categories     (0 LLM calls)
    Mode B (fresh): 1 batched LLM call returning 4 JSON cats (1 LLM call)

Step 2 -- For each category:
    encode it, KNN-retrieve top 30 words (filter stems, sub-tokens, used)
    keep top 8 as candidate pool

Step 3 -- Select the best 4 (NYT-safe):
    enumerate all C(8,4)=70 subsets, rank by avg pairwise cosine sim
    return the highest-scoring subset that is NOT a verbatim past NYT group

Steps 4-6 -- Shared: color assignment + dedup + solver validation
```

**Why this works:** LLMs are good at naming; embeddings are good at ranking. CFR assigns each task to the component that handles it best. The result: 99% pass rate at 1/20 the cost and 4-7x the speed of Pipeline A.

### Shared: Multi-Solver Roundtable Validation

| Solver | Method | Speed |
|--------|--------|-------|
| Embedding | Greedy C(16,4)=1,820 enumeration by cosine similarity | 50 puzzles/s |
| Clustering | G = 0.4 I + 0.3 s + 0.3 V with beam search (width 10) | 1 puzzle/s |
| LLM | Chain-of-thought partition (optional tiebreaker) | API-bound |

A puzzle is accepted if **either** the embedding or clustering solver recovers all 4 intended groups. Our solvers are intentionally conservative (they solve only 2-3% of real NYT puzzles), so a puzzle they can solve has a cleanly recoverable semantic structure.

### Shared: Rubric-Safe Deduplication

The rubric's hard-fail rule is *"If you generate a past connections puzzle you will automatically fail."* We enforce this at two independent levels:

1. **16-word overlap**: flag if the generated puzzle shares more than 6 words with any past NYT puzzle.
2. **4-word group match**: flag if any of the 4 generated groups exactly matches any of the 2,216 past NYT groups (frozenset comparison).

CFR's Step 3 also actively avoids NYT-group collisions during selection. Across 200 v2 benchmark puzzles, zero generated puzzles triggered either check.

---

## Generating Real Puzzles

By default, `DRY_RUN=true` and all LLM calls return mock responses. To generate real puzzles with live API calls:

### Pipeline A (LLM-heavy baseline)

```bash
# With OpenAI (recommended for bulk)
export DRY_RUN=false
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python scripts/generate_puzzles.py --count 100 --method iterative
```

### Pipeline B (CFR)

```bash
# Mode A: 0 LLM calls, free, fastest
DRY_RUN=true python scripts/cfr/generate_cfr.py --mode remix --count 100

# Mode B: 1 LLM call/puzzle, fresh categories
export DRY_RUN=false
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python scripts/cfr/generate_cfr.py --mode fresh --count 100

# To use only the 4,918-word NYT bank (for ablation)
python scripts/cfr/generate_cfr.py --mode remix --count 100 --no-augment
```

### Cost estimates

| Pipeline | Model | Cost / 1,000 puzzles | 10,000 puzzles |
|----------|-------|---------------------|----------------|
| A -- iterative | gpt-4o-mini | ~$1.00 | ~$10 |
| A -- iterative | gpt-4o | ~$4.00 | ~$40 |
| B -- Mode A (remix) | -- | **$0.00** | **$0.00** |
| B -- Mode B (fresh) | gpt-4o-mini | ~$0.05 | ~$0.50 |

Output is written to `data/generated/` (Pipeline A) or `data/generated/cfr/<mode>/` (Pipeline B). Each puzzle is validated by the roundtable before being saved; invalid puzzles are written to a separate `*_invalid.json` file for inspection.

---

## Key References

### Papers

1. **Making New Connections** (arXiv:2407.11240) -- Generation pipeline, story injection, false-group method. Primary reference for Pipeline A.
2. **Missed Connections** (arXiv:2404.11730) -- Embedding solver and LLM solver baselines.
3. **Deceptively Simple** (arXiv:2412.01621) -- Group Similarity Score (G = 0.4 I + 0.3 s + 0.3 V), Penalty Score, beam search solver.
4. **Connecting the Dots** (arXiv:2406.11012) -- Category taxonomy, human evaluation methodology.

### Data

- NYT Connections dataset: 554 puzzles from the [Connections Kaggle dataset](https://www.kaggle.com/datasets), stored at `data/nyt_puzzles/ConnectionsFinalDataset (1).json`.
- WordNet (NLTK): 147k lemmas, filtered to ~10k common single-token words for the augmented bank.

### Reference repos (inspiration only; no code copied)

- `repos/NLP-Connections/` -- Solver implementation patterns.
- `repos/react-connections-game/` -- Frontend UI design reference.
- `resources/merrill_solver_blog.html` -- Practical solver walkthrough.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/executive_summary.pdf`](docs/executive_summary.pdf) | 2-page overview of both pipelines and results |
| [`docs/methods.pdf`](docs/methods.pdf) | Full technical methods, algorithms, and benchmarks |
| [`docs/study_guide.pdf`](docs/study_guide.pdf) | 11-page learning guide covering all concepts |
| [`docs/faq.md`](docs/faq.md) | Anticipated questions and answers |

All `.tex` source files are in `docs/`; run `python scripts/build_*_pdf.py` to regenerate any PDF without needing a LaTeX compiler.

---

## Team

| Name | Role |
|------|------|
| Kevin | Pipeline architecture, CFR design, web app |
| [Team Member 2] | -- |
| [Team Member 3] | -- |

---

Built for **STA 561D** at Duke University.
