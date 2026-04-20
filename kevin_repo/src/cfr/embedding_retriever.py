"""EmbeddingRetriever: KNN over word embeddings, plus diverse category sampling.

The core non-LLM component of CFR. Precomputes MPNET embeddings for:
  - The 4,918 NYT word bank
  - The 2,216 NYT category names

Then provides:
  - retrieve(category_name): KNN top-K words closest to a category name
  - sample_diverse_categories(): sample N mutually dissimilar NYT categories
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    """KNN retrieval over precomputed MPNET word + category embeddings."""

    def __init__(
        self,
        word_bank: list[str],
        nyt_categories: list[str],
        model=None,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            word_bank: list of words (e.g., 4,918 NYT words), uppercased.
            nyt_categories: list of category names (e.g., 2,216 NYT categories).
            model: SentenceTransformer, lazily loaded if None.
            cache_path: path to .npz cache file; if exists, loads; if not, writes.
        """
        self.word_bank = [w.upper() for w in word_bank]
        self.nyt_categories = [c.upper() for c in nyt_categories]
        self._model = model
        self.cache_path = Path(cache_path) if cache_path else None

        self._word_embeddings = None  # (N_words, 768)
        self._category_embeddings = None  # (N_cats, 768)
        self._knn = None  # sklearn NearestNeighbors

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from src.config import EMBEDDING_MODEL
            logger.info(f"Loading {EMBEDDING_MODEL}...")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def precompute(self, force: bool = False) -> None:
        """Compute (or load from cache) all embeddings + KNN index."""
        if self.cache_path and self.cache_path.exists() and not force:
            logger.info(f"Loading cached embeddings from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            self._word_embeddings = data["word_embeddings"]
            self._category_embeddings = data["category_embeddings"]
            cached_bank = data["word_bank"].tolist()
            cached_cats = data["nyt_categories"].tolist()
            # Sanity check: cache must match current inputs
            if cached_bank != self.word_bank or cached_cats != self.nyt_categories:
                logger.warning("Cache doesn't match current inputs; recomputing.")
                self._word_embeddings = None
                self._category_embeddings = None
                self.precompute(force=True)
                return
        else:
            logger.info(
                f"Encoding {len(self.word_bank)} words + "
                f"{len(self.nyt_categories)} categories..."
            )
            self._word_embeddings = self.model.encode(
                self.word_bank, convert_to_numpy=True, show_progress_bar=False
            )
            self._category_embeddings = self.model.encode(
                self.nyt_categories, convert_to_numpy=True, show_progress_bar=False
            )
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    self.cache_path,
                    word_embeddings=self._word_embeddings,
                    category_embeddings=self._category_embeddings,
                    word_bank=np.array(self.word_bank, dtype=object),
                    nyt_categories=np.array(self.nyt_categories, dtype=object),
                )
                logger.info(f"Cached embeddings to {self.cache_path}")

        # Build KNN index over word embeddings (cosine distance)
        from sklearn.neighbors import NearestNeighbors
        self._knn = NearestNeighbors(
            n_neighbors=min(50, len(self.word_bank)),
            metric="cosine",
        )
        self._knn.fit(self._word_embeddings)
        logger.info("KNN index ready.")

    def retrieve(
        self,
        category_name: str,
        top_k: int = 30,
        exclude_words: Optional[set] = None,
        filter_morphological_duplicates: bool = True,
        filter_category_tokens: bool = True,
    ) -> list[tuple[str, float]]:
        """Return top-K words most semantically similar to category_name.

        Args:
            category_name: e.g., "BIRDS"
            top_k: how many candidates to return
            exclude_words: words to filter out (already used in other groups)
            filter_morphological_duplicates: skip words sharing a stem with an
                already-chosen word (e.g., block TIMES if TIME already chosen).
            filter_category_tokens: skip words that are substrings of the
                category name or vice versa (e.g., block TIME for category
                "TIME PERIODS").

        Returns:
            List of (word, similarity) tuples, highest similarity first.
        """
        if self._knn is None:
            self.precompute()

        exclude = {w.upper() for w in (exclude_words or [])}
        cat_upper = category_name.upper()
        cat_tokens = {t.strip() for t in cat_upper.replace(",", " ").split() if t.strip()}

        query_vec = self.model.encode(
            [cat_upper], convert_to_numpy=True, show_progress_bar=False
        )

        # Request extra neighbors to allow for filtering
        n_request = min(top_k * 4 + len(exclude) + 10, len(self.word_bank))
        distances, indices = self._knn.kneighbors(query_vec, n_neighbors=n_request)
        distances = distances[0]
        indices = indices[0]

        results = []
        chosen_stems = set()

        for dist, idx in zip(distances, indices):
            word = self.word_bank[idx]
            if word in exclude:
                continue

            # Filter: word is a literal token of the category name
            # (e.g., category "TIME PERIODS" should not include the word "TIME")
            if filter_category_tokens:
                if word in cat_tokens:
                    continue
                # Also skip if word is a near-exact match of the category
                if word == cat_upper or word == cat_upper.rstrip("S"):
                    continue

            # Filter: morphological stem duplicates
            # (e.g., if we've chosen TIME, skip TIMES, TIMER, TIMERS, TIMELY)
            if filter_morphological_duplicates:
                stem = _morph_stem(word)
                if stem in chosen_stems:
                    continue
                chosen_stems.add(stem)

            sim = 1.0 - float(dist)
            results.append((word, sim))
            if len(results) >= top_k:
                break

        return results

    def sample_diverse_categories(
        self,
        n: int = 4,
        min_distance: float = 0.4,
        max_tries: int = 1000,
        rng: Optional[random.Random] = None,
    ) -> list[str]:
        """Sample N NYT categories such that all pairs are MPNET-distant.

        min_distance is cosine distance (1 - cosine sim). Larger = more diverse.
        """
        if self._category_embeddings is None:
            self.precompute()

        rng = rng or random.Random()
        n_cats = len(self.nyt_categories)

        from sklearn.metrics.pairwise import cosine_similarity

        for _ in range(max_tries):
            idxs = rng.sample(range(n_cats), n)
            vecs = self._category_embeddings[idxs]
            sims = cosine_similarity(vecs)
            # Check pairwise distances (diagonal is 1; ignore it)
            max_pair_sim = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    max_pair_sim = max(max_pair_sim, sims[i, j])
            if (1.0 - max_pair_sim) >= min_distance:
                return [self.nyt_categories[i] for i in idxs]

        # If we can't satisfy the constraint, relax and return best sample found
        logger.warning(
            f"Couldn't find {n} categories with min_distance={min_distance} "
            f"in {max_tries} tries. Relaxing."
        )
        idxs = rng.sample(range(n_cats), n)
        return [self.nyt_categories[i] for i in idxs]


# -----------------------------------------------------------------------------
# Helpers

_COMMON_SUFFIXES = ("IEST", "INGLY", "NESS", "MENT", "TION", "SION",
                    "ING", "EST", "ERS", "IES", "IED", "LY",
                    "ED", "ER", "ES", "S")


def _morph_stem(word: str) -> str:
    """Return a crude morphological stem for duplicate detection.

    Not a real stemmer — just strips common English suffixes to collapse
    TIME/TIMES/TIMING into the same token. Good enough for filtering
    near-duplicates in a small word bank.
    """
    w = word.upper()
    for suf in _COMMON_SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def load_nyt_word_bank_and_categories(nyt_path: str) -> tuple[list[str], list[str]]:
    """Load the NYT dataset and extract unique words + unique category names."""
    with open(nyt_path) as f:
        puzzles = json.load(f)

    words = set()
    categories = []

    for p in puzzles:
        for w in p.get("words", []):
            words.add(w.upper())
        for ans in p.get("answers", []):
            cat = ans.get("answerDescription", "").strip().upper()
            if cat:
                categories.append(cat)

    # Deduplicate categories while preserving order
    seen = set()
    unique_cats = []
    for c in categories:
        if c not in seen:
            seen.add(c)
            unique_cats.append(c)

    return sorted(words), unique_cats
