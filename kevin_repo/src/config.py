"""Configuration for the Infinite Connections pipeline."""

import os

# DRY_RUN mode: when True, uses mock LLM responses instead of real API calls.
# Set DRY_RUN=false (case-insensitive) to use real Claude API.
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"

# LLM provider: "openai", "anthropic", or "mock"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")

# OpenAI API settings (cheapest option)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Claude API settings
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
CLAUDE_MAX_TOKENS = 4096

# Embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NYT_PUZZLES_PATH = os.path.join(
    PROJECT_ROOT, "data", "nyt_puzzles", "ConnectionsFinalDataset (1).json"
)
GENERATED_PUZZLES_PATH = os.path.join(PROJECT_ROOT, "data", "generated")
MOCK_PUZZLES_PATH = os.path.join(PROJECT_ROOT, "data", "mock", "mock_puzzles.json")

# Generator settings
WORDS_PER_POOL = 8  # LLM proposes 8 candidates, MPNET picks best 4
WORDS_PER_GROUP = 4
GROUPS_PER_PUZZLE = 4
TOTAL_WORDS = WORDS_PER_GROUP * GROUPS_PER_PUZZLE  # 16

# Deduplication threshold: flag if >6 words overlap with an existing puzzle
DEDUP_WORD_OVERLAP_THRESHOLD = 6

# Difficulty color thresholds (avg cosine similarity from Table 1)
COLOR_THRESHOLDS = {
    "yellow": {"mean": 0.40, "variance": 0.0285},
    "green": {"mean": 0.35, "variance": 0.0214},
    "blue": {"mean": 0.29, "variance": 0.0123},
    "purple": {"mean": 0.27, "variance": 0.0108},
}

# Solver settings
BEAM_WIDTH = 10  # For clustering solver beam search
LLM_SOLVER_TEMPERATURE = 0.3

# Roundtable validation
MIN_SOLVER_AGREEMENT = 2  # At least 2 solvers must agree

# Random seed for reproducibility
RANDOM_SEED = 42
