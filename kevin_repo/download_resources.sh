#!/bin/bash
# ============================================================
# Infinite Connections — Resource Download Script
# Run from your project root directory
# ============================================================

set -e
echo "=== Infinite Connections: Downloading All Resources ==="

# ── Create directory structure ──
mkdir -p data/nyt_puzzles
mkdir -p papers
mkdir -p repos
mkdir -p resources

# ============================================================
# 1. PAPERS (arXiv PDFs)
# ============================================================
echo ""
echo "--- Downloading Papers ---"

# Paper 1: Making New Connections (Generation pipeline - PRIMARY)
echo "Downloading: Making New Connections (2407.11240)..."
curl -sL -o papers/making_new_connections_2407.11240.pdf \
  "https://arxiv.org/pdf/2407.11240v1"

# Paper 2: Missed Connections (Solver baselines)
echo "Downloading: Missed Connections (2404.11730)..."
curl -sL -o papers/missed_connections_2404.11730.pdf \
  "https://arxiv.org/pdf/2404.11730v2"

# Paper 3: Deceptively Simple (Solver scoring / beam search)
echo "Downloading: Deceptively Simple (2412.01621)..."
curl -sL -o papers/deceptively_simple_2412.01621.pdf \
  "https://arxiv.org/pdf/2412.01621v3"

# Paper 4: Connecting the Dots (Evaluation taxonomy)
echo "Downloading: Connecting the Dots (2406.11012)..."
curl -sL -o papers/connecting_the_dots_2406.11012.pdf \
  "https://arxiv.org/pdf/2406.11012v7"

echo "Papers done. Check papers/ directory."

# ============================================================
# 2. DATASETS
# ============================================================
echo ""
echo "--- Downloading Datasets ---"

# Option A: Kaggle NYT Connections dataset (requires kaggle CLI + credentials)
# Uncomment below if you have kaggle set up:
#   pip install kaggle
#   kaggle datasets download -d tm21cy/nyt-connections -p data/nyt_puzzles --unzip

# Option B: Direct download from Kaggle (manual fallback)
echo ""
echo "NOTE: For the Kaggle NYT Connections dataset, do ONE of:"
echo "  Option A (if kaggle CLI is configured):"
echo "    kaggle datasets download -d tm21cy/nyt-connections -p data/nyt_puzzles --unzip"
echo ""
echo "  Option B (manual):"
echo "    Go to https://www.kaggle.com/datasets/tm21cy/nyt-connections"
echo "    Click Download → save into data/nyt_puzzles/"
echo ""

# ============================================================
# 3. NLTK / WordNet (Python)
# ============================================================
echo "--- Installing Python dependencies & downloading WordNet ---"

pip install nltk sentence-transformers numpy scipy scikit-learn pandas tqdm

python3 << 'PYEOF'
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)    # Open Multilingual Wordnet (supplements WordNet)
nltk.download('punkt', quiet=True)       # Tokenizer (useful for text processing)
nltk.download('punkt_tab', quiet=True)
print("WordNet + NLTK resources downloaded successfully.")
PYEOF

echo "NLTK resources done."

# ============================================================
# 4. GITHUB REPOS (reference code)
# ============================================================
echo ""
echo "--- Cloning Reference Repositories ---"

# Repo 1: NLP-Connections (solver implementations)
echo "Cloning: joshbrook/NLP-Connections..."
git clone --depth 1 https://github.com/joshbrook/NLP-Connections.git repos/NLP-Connections 2>/dev/null || echo "  (already exists, skipping)"

# Repo 2: react-connections-game (React frontend reference)
echo "Cloning: and-computers/react-connections-game..."
git clone --depth 1 https://github.com/and-computers/react-connections-game.git repos/react-connections-game 2>/dev/null || echo "  (already exists, skipping)"

# Repo 3: cork89/connections (another solver reference)
echo "Cloning: cork89/connections..."
git clone --depth 1 https://github.com/cork89/connections.git repos/connections 2>/dev/null || echo "  (already exists, skipping)"

echo "Repos done. Check repos/ directory."

# ============================================================
# 5. BLOG POST & WEB RESOURCES
# ============================================================
echo ""
echo "--- Saving Blog Post / Web Resources ---"

# Jeremy Merrill's solver blog post (practical implementation walkthrough)
echo "Downloading: Jeremy Merrill solver blog post..."
curl -sL -o resources/merrill_solver_blog.html \
  "https://jeremybmerrill.com/blog/2024/08/this-algorithm-solves-nyt-connections.html"

# Swellgarfo Connections (custom puzzle creator — UI reference)
echo "Downloading: Swellgarfo connections page..."
curl -sL -o resources/swellgarfo_connections.html \
  "https://connections.swellgarfo.com"

echo "Web resources done. Check resources/ directory."

# ============================================================
# 6. VERIFY DOWNLOADS
# ============================================================
echo ""
echo "=========================================="
echo "  DOWNLOAD SUMMARY"
echo "=========================================="
echo ""
echo "Papers:"
ls -lh papers/ 2>/dev/null || echo "  (none found)"
echo ""
echo "Datasets:"
ls -lh data/nyt_puzzles/ 2>/dev/null || echo "  (remember to download Kaggle dataset manually)"
echo ""
echo "Repos:"
ls -d repos/*/ 2>/dev/null || echo "  (none found)"
echo ""
echo "Resources:"
ls -lh resources/ 2>/dev/null || echo "  (none found)"
echo ""
echo "=========================================="
echo "  REMAINING MANUAL STEPS"
echo "=========================================="
echo "1. Download Kaggle dataset if not done:"
echo "   https://www.kaggle.com/datasets/tm21cy/nyt-connections"
echo "   Save to: data/nyt_puzzles/"
echo ""
echo "2. Copy your existing ConnectionsFinalDataset__1_.json into data/nyt_puzzles/"
echo ""
echo "Done!"
