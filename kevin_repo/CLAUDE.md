# Infinite Connections — Claude Code Project Instructions

## Project Overview

Build an AI system that generates NYT-style Connections puzzles at scale. The project has three workstreams that should be prototyped end-to-end first, then iterated:

1. **Puzzle Generator** — Uses LLM APIs (OpenAI GPT-4o-mini for bulk, optionally Claude for editing) to create candidate puzzles
2. **Multi-Solver Validator** — Multiple independent solvers verify each puzzle has a unique solution
3. **Deliverables** — A Jupyter notebook (for grading) + a web app (for demo)

> **CRITICAL: DRY-RUN MODE**
> All code must be built and fully functional WITHOUT making any live API calls. Every module that calls an LLM API must support a `--dry-run` / `DRY_RUN=true` mode that uses mock/fake responses instead. This means:
> - A `MockLLMClient` class that returns realistic hardcoded/template responses matching the expected JSON schema
> - All pipeline stages (generator, editor, LLM solver) must accept a `client` parameter that can be either the real API client or the mock
> - The mock responses should be realistic enough to test the full pipeline: word groups, category names, editor fixes, solver outputs
> - Include 5–10 pre-built mock puzzles in `data/mock/` that exercise all code paths
> - `DRY_RUN` is controlled via environment variable: `export DRY_RUN=true` (default: true)
> - When `DRY_RUN=true`, the code must never import or instantiate any API client
> - All tests, the notebook demo, and the web app must work in dry-run mode out of the box

This is an academic project for STA 561D (Duke). Grading criteria: creativity, solution quality, clarity of exposition. The final submission includes a 2-page executive summary, a 2–5 page FAQ, and a technical appendix with reproducible code and a Jupyter notebook demo.

---

## Directory Structure

```
infinite-connections/
├── CLAUDE.md                    # (this file)
├── data/
│   ├── nyt_puzzles/             # Ground truth NYT puzzles (JSON)
│   │   └── ConnectionsFinalDataset.json
│   ├── generated/               # Output: generated puzzles
│   └── mock/                    # Mock LLM responses for dry-run mode
│       ├── mock_puzzles.json    # 5–10 pre-built puzzles for testing
│       └── mock_responses.json  # Canned LLM responses per pipeline stage
├── papers/                      # Reference PDFs (read-only context)
├── repos/                       # Cloned reference repos (read-only)
├── resources/                   # Blog posts, web references
├── src/
│   ├── llm_client.py            # LLM abstraction: OpenAI, Anthropic, or Mock
│   ├── config.py                # Central config: DRY_RUN flag, model names, API keys
│   ├── generator/               # Puzzle generation pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main orchestrator
│   │   ├── group_creator.py     # LLM calls for word group generation
│   │   ├── puzzle_editor.py     # LLM call to fix category names
│   │   ├── difficulty.py        # MPNET cosine similarity → color assignment
│   │   ├── deduplicator.py      # Check against existing NYT puzzles
│   │   └── prompts.py           # All prompt templates
│   ├── solvers/                 # Multiple solver implementations
│   │   ├── __init__.py
│   │   ├── embedding_solver.py  # MPNET cosine similarity solver
│   │   ├── llm_solver.py        # LLM CoT solver (any provider)
│   │   ├── clustering_solver.py # K-means on embeddings
│   │   └── roundtable.py        # Convergence validator (runs all solvers)
│   ├── evaluation/              # Quality metrics and analysis
│   │   ├── __init__.py
│   │   ├── metrics.py           # Group similarity score, penalty score
│   │   └── analyzer.py          # Statistics, visualizations
│   └── webapp/                  # Flask + React web interface
│       ├── app.py               # Flask backend
│       ├── templates/
│       └── static/
├── notebooks/
│   └── infinite_connections_demo.ipynb  # Main deliverable notebook
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## Phase 1: Working Prototype (Build This First)

Get a minimal version of every piece working end-to-end before optimizing anything. The goal is: generate 1 puzzle → validate it with 2+ solvers → display it in a notebook. Then scale.

### 1.1 Data Loading

- Load `data/nyt_puzzles/ConnectionsFinalDataset.json` (554 puzzles)
- Schema per puzzle: `{ date, contest, words: [16 strings], answers: [{ answerDescription: str, words: [4 strings] }], difficulty: float }`
- Build a word bank from all unique words across all puzzles (~15,000 words)
- This dataset is used for: (a) solver benchmarking, (b) deduplication checks, (c) seeding the story-injection diversity prompt

### 1.2 Puzzle Generator (Multi-Provider LLM)

The generator supports multiple LLM backends via a unified `LLMClient` abstraction in `src/llm_client.py`:

```python
# src/llm_client.py — the key abstraction
import os

class LLMClient:
    """Unified interface for LLM calls. Supports OpenAI, Anthropic, or Mock."""
    
    def __init__(self, provider=None):
        dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        if dry_run:
            provider = "mock"
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        # Only import/instantiate the real client when NOT in dry-run mode
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()  # reads OPENAI_API_KEY from env
            self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
            self.model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        elif self.provider == "mock":
            self.client = None
            self.model = "mock"
    
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 1.0) -> str:
        """Returns the LLM's text response. In mock mode, returns a realistic fake."""
        if self.provider == "mock":
            return self._mock_response(system_prompt, user_prompt)
        elif self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return resp.choices[0].message.content
        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature
            )
            return resp.content[0].text
    
    def _mock_response(self, system_prompt, user_prompt):
        # Return realistic canned responses based on prompt content
        # Must handle: group creation, puzzle editing, LLM solving
        ...
```

**Recommended provider strategy:**
- **GPT-4o-mini** (`openai`, `gpt-4o-mini`) — default for bulk group generation. $0.15/$0.60 per MTok. 10,000 puzzles ≈ $21.
- **Claude Sonnet** (`anthropic`, `claude-sonnet-4-20250514`) — optional for editor pass and LLM solver. Better at nuanced judgment. $3/$15 per MTok.
- **Mock** — default when `DRY_RUN=true`. No API calls, no cost.

Environment variables:
```bash
export DRY_RUN=true                    # Default: no API calls
export LLM_PROVIDER=openai             # or "anthropic" or "mock"  
export LLM_MODEL=gpt-4o-mini           # or "claude-sonnet-4-20250514"
export OPENAI_API_KEY=sk-...           # Only needed when DRY_RUN=false
export ANTHROPIC_API_KEY=sk-ant-...    # Only needed when using anthropic provider
```

Use the iterative pipeline from the "Making New Connections" paper:

**Step 1: Generate word groups iteratively (not all at once)**

Each puzzle is built one group at a time across 4 LLM calls. For each group:
- The LLM proposes a category name + a pool of 8 candidate words
- From the 8 words, the best 4 are selected using MPNET cosine similarity (not by the LLM)
- Previous groups are passed as context to subsequent calls

**Step 2: Story injection for diversity**

To avoid repetitive outputs (the LLM defaults to BOARD GAMES: chess, checkers, monopoly, life), inject randomness:
- System prompt includes: "First, write a short story using these words: [4 random words from the NYT word bank]. Then use that story as inspiration for creating your category."
- This dramatically increases category diversity

**Step 3: Category style guidance**

Include a list of category styles in every prompt so outputs match NYT aesthetics:
- "Synonyms or Slang" (e.g., WAYS TO SAY HELLO)
- "Wordplay" (e.g., ___FISH)  
- "Fill in the blank" (e.g., FIRE ___)
- "Hidden pattern" (e.g., WORDS THAT ARE ALSO COLORS)
- "Common property" (e.g., THINGS WITH WHEELS)
- "Pop culture" (e.g., TAYLOR SWIFT ALBUMS)

**Step 4: False Group pipeline (primary method)**

This produces the highest-quality puzzles per the paper's user study:
1. Generate a "root" group (the false group) — a plausible category with 4 words
2. For each of the 4 root words, find an alternate meaning/definition
3. Generate a new category + 8-word pool based on each alternate meaning
4. The 4 follow-up groups become the actual puzzle; the root group is a decoy
5. Result: players see what looks like 5 possible groupings but only 4 are valid

**Step 5: Puzzle Editor pass**

A second LLM call (recommend Claude Sonnet or GPT-4o for this — quality matters more than cost here) reviews the complete puzzle and:
- Checks if each category name accurately describes its words
- Rewrites inaccurate category names
- Flags any invalid groups (e.g., a word doesn't actually fit)

**Step 6: Difficulty color assignment**

Use MPNET embeddings (sentence-transformers, model: `all-mpnet-base-v2`) to compute average pairwise cosine similarity within each group. Assign colors using these empirically validated thresholds from Table 1 of the paper:

| Color  | Avg Cosine Similarity | Variance |
|--------|----------------------|----------|
| Yellow | ~0.40                | 0.0285   |
| Green  | ~0.35                | 0.0214   |
| Blue   | ~0.29                | 0.0123   |
| Purple | ~0.27                | 0.0108   |

Assign colors by sorting groups by similarity and mapping to: highest → yellow, next → green, next → blue, lowest → purple. If two groups have the same "color" by threshold, also run a final LLM call to rank difficulty by the LLM's judgment.

**Step 7: Deduplication**

Before accepting a puzzle, check it against all known NYT puzzles:
- Compare the 16-word set against every puzzle in the ground truth dataset
- Flag if >6 words overlap with any single existing puzzle
- Also check for exact category name matches

### 1.3 Multi-Solver Validator (Roundtable Approach)

The key insight: a valid puzzle should be solvable, and multiple independent solvers should converge to the **same unique solution**. If solvers disagree, the puzzle is ambiguous.

**Solver 1: Embedding Solver (from "Missed Connections" paper)**
- Enumerate all C(16,4) = 1,820 possible 4-word groups
- Compute average pairwise cosine similarity (MPNET) for each
- Greedily select groups by highest similarity, iterating with remaining words
- This solver is fast and deterministic

**Solver 2: Clustering Solver (from "Deceptively Simple" paper)**
- Compute Group Similarity Score: `G = 0.4·I + 0.3·s + 0.3·V`
  - `I = -K(E)` where K is k-means inertia (k=1) on the group's embeddings
  - `s = min pairwise cosine similarity` among the 4 words
  - `V = mean(P) / (1 + var(P))` where P is all pairwise similarities
- Compute Penalty Score: `P = (1/|R|) * Σ cos(μ_C, r)` for remaining words R
- Use beam search (width=10) to find the best complete 4-group solution maximizing cumulative `S = G - P`

**Solver 3: LLM Solver (any provider, with Chain-of-Thought)**
- Present the 16 words to the LLM with CoT prompting
- Ask it to identify 4 groups of 4, providing reasoning for each
- This is the most "human-like" solver but also the most expensive
- Use sparingly — only on puzzles that pass Solvers 1 & 2
- In dry-run mode, returns a mock solution based on the mock puzzle data

**Roundtable Validation:**
- Run Solver 1 and Solver 2 on every candidate puzzle
- If both converge to the same 4 groups AND those groups match the intended solution → puzzle is VALID
- If they disagree → puzzle is AMBIGUOUS, discard or flag for review  
- Optionally run Solver 3 as a tiebreaker on edge cases
- Track convergence rate as a quality metric

### 1.4 Puzzle Schema

Each generated puzzle should be stored as:

```json
{
  "id": "INF-00001",
  "words": ["WORD1", "WORD2", ... ],  // 16 words, shuffled
  "groups": [
    {
      "category": "CATEGORY NAME",
      "words": ["W1", "W2", "W3", "W4"],
      "color": "yellow",
      "similarity_score": 0.412
    },
    // ... 3 more groups
  ],
  "metadata": {
    "generation_method": "false_group",
    "solver_agreement": true,
    "solvers_used": ["embedding", "clustering", "llm"],
    "solver_results": { ... },
    "dedup_check": true,
    "overall_difficulty": 3.2,
    "created_at": "2026-03-28T12:00:00Z"
  }
}
```

---

## Phase 2: Scale and Evaluate

Once the prototype works end-to-end:

### 2.1 Batch Generation
- Generate puzzles in batches of 50–100
- Use async/parallel API calls where possible
- Target: 10,000 candidate puzzles → expect ~40% pass rate → ~4,000 valid puzzles
- Estimated cost with GPT-4o-mini: ~$21 for 10,000 puzzles
- Recommended: start with 100 candidates (~$0.21) to validate quality, then scale to 1,000 (~$2.10), then full run

### 2.2 Solver Benchmarking
- Run all solvers on the 554 ground-truth NYT puzzles first
- Report accuracy per solver, per difficulty color
- Use this as the baseline to validate solver implementations are working correctly
- Expected baselines from the papers: MPNET embedding solver ~11.6% full-puzzle solve rate, GPT-4 CoT ~38.9%

### 2.3 Quality Analysis
- Compute distribution of cosine similarity scores across generated puzzles vs NYT puzzles
- Plot difficulty distributions (should roughly match NYT spread)
- Track false-group effectiveness: do solvers get "tricked" by the false group?
- Measure category diversity (unique category names across the dataset)

---

## Phase 3: Deliverables

### 3.1 Jupyter Notebook (`notebooks/infinite_connections_demo.ipynb`)

This is the primary graded artifact. It must be fully reproducible. Structure:

1. **Introduction** — problem statement, what Connections is, why generation is hard
2. **Data Exploration** — load and visualize the NYT dataset (word frequency, difficulty distribution, category types)
3. **Generation Pipeline Demo** — generate 5 puzzles live, show each step (group creation, editing, difficulty assignment). In dry-run mode, use mock puzzles and show the pipeline flow with annotated mock responses.
4. **Solver Benchmark** — run all 3 solvers on NYT puzzles, report accuracy tables and plots. Embedding and clustering solvers run locally (no API needed). LLM solver uses mock in dry-run mode.
5. **Validation Demo** — show the roundtable validator in action on generated puzzles
6. **Results at Scale** — load pre-generated dataset, show quality metrics, compare to NYT distribution
7. **Example Puzzles** — display 5–10 of the best generated puzzles in a visual grid format
8. **Discussion** — what worked, what didn't, future improvements

### 3.2 Web App (`src/webapp/`)

A playable Connections interface for demo purposes:

- Flask backend serving puzzles from the generated dataset
- Frontend that replicates the NYT Connections UI:
  - 4×4 grid of word tiles
  - Click to select 4 words, submit a guess
  - Correct group revealed with color, "One away!" feedback for 3/4 matches
  - 4 mistakes allowed before failure
  - Shuffle button to rearrange words
- Puzzle selection: random from validated dataset, or by difficulty
- Mobile-friendly responsive design

Use the `react-connections-game` repo in `repos/` as UI reference, but build fresh with Flask + vanilla JS/HTML/CSS (or React if preferred). Keep it simple — this is a demo, not a production app.

### 3.3 Written Deliverables

Generate scaffolding for:
- `docs/executive_summary.md` — 2-page non-technical summary
- `docs/faq.md` — 2–5 pages of anticipated questions and answers
- `docs/technical_appendix.md` — all technical details, math, algorithm descriptions

---

## Technical Constraints

- **Language**: Python 3.10+
- **LLM Providers** (via unified `LLMClient` in `src/llm_client.py`):
  - **OpenAI** (`openai` Python SDK) — `gpt-4o-mini` for bulk generation. API key from `OPENAI_API_KEY` env var.
  - **Anthropic** (`anthropic` Python SDK) — `claude-sonnet-4-20250514` for editor/solver. API key from `ANTHROPIC_API_KEY` env var.
  - **Mock** — default mode (`DRY_RUN=true`). Returns realistic fake responses. No API keys needed.
- **DRY_RUN=true is the default.** The entire codebase — pipeline, solvers, notebook, web app — must work out of the box with zero API keys configured. Real API calls only happen when `DRY_RUN=false` AND the appropriate API key is set.
- **Embeddings**: `sentence-transformers` library, model `all-mpnet-base-v2` (runs locally, no API needed)
- **Key packages**: `openai`, `anthropic`, `sentence-transformers`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `nltk`, `flask`, `jupyter`, `matplotlib`, `seaborn`, `tqdm`
- **Reproducibility**: set random seeds everywhere, log all API calls, save intermediate outputs

---

## Key Implementation Notes

1. **DRY_RUN=true is the default. No exceptions.** Every single code path must work without API keys. The mock client must return responses realistic enough to exercise all downstream logic (embedding selection, editor fixes, solver CoT). If a function can't work without an API call, it needs a mock fallback.

2. **Don't let the LLM pick the final 4 words from its 8-word pool.** Always use MPNET cosine similarity to select the best 4. LLMs are bad at estimating word group difficulty and make inconsistent selections.

3. **The story injection trick is essential for diversity.** Without it, the LLM produces the same 20 categories repeatedly regardless of temperature/seed settings.

4. **The false group method produces the best puzzles.** In the paper's user study, LLM False Group puzzles beat NYT puzzles in user preference 42.86% of the time. Intentional Overlap puzzles were too hard (31% solve rate). Default to False Group for most generation.

5. **Solver agreement is the quality gate.** Don't rely on any single metric. If two independent solvers can't find the same unique solution, the puzzle is ambiguous and should be discarded.

6. **The embedding solver is your workhorse.** It's fast, deterministic, and solves puzzles reasonably well. Use it as the first filter. Only run the expensive LLM solver on puzzles that pass the embedding check.

7. **Deduplication matters.** The LLM was not trained on Connections puzzles (knowledge cutoff), but could still accidentally generate similar word groupings. Always check against the ground truth dataset.

8. **Color assignment uses empirical thresholds, not LLM judgment.** Use the cosine similarity values from Table 1 first, then optionally refine with an LLM call only when groups have very similar scores.

9. **GPT-4o-mini is the default generator for cost reasons.** At $0.15/$0.60 per MTok, it's ~20x cheaper than Claude Sonnet. Quality may be slightly lower, but the pipeline compensates: MPNET picks the best words, the editor pass fixes names, and solvers filter bad puzzles. Use Claude or GPT-4o for the editor pass if budget allows.

---

## Reference Materials (in project directory)

| File | What it is | Use it for |
|------|-----------|------------|
| `papers/making_new_connections_2407.11240.pdf` | Generation pipeline paper | Generator architecture, prompts, false groups |
| `papers/missed_connections_2404.11730.pdf` | Solver baselines paper | Embedding solver, LLM solver, CoT prompts |
| `papers/deceptively_simple_2412.01621.pdf` | Scoring + beam search paper | Group Similarity Score formula, beam search solver |
| `papers/connecting_the_dots_2406.11012.pdf` | Knowledge taxonomy paper | Category types, human evaluation methodology |
| `data/nyt_puzzles/ConnectionsFinalDataset.json` | 554 NYT puzzles | Ground truth, benchmarking, deduplication |
| `repos/NLP-Connections/` | Solver reference code | Implementation patterns |
| `repos/react-connections-game/` | React frontend reference | UI design patterns |
| `resources/merrill_solver_blog.html` | Practical solver walkthrough | Algorithm implementation details |
