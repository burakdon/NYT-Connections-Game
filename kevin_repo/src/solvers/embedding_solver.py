"""Embedding-based solver: greedy selection by highest cosine similarity."""

import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingSolver:
    """Solves Connections puzzles using MPNET cosine similarity.

    Enumerates C(n, 4) groups, scores each by average pairwise similarity,
    then greedily selects non-overlapping groups.
    """

    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from src.config import EMBEDDING_MODEL
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def solve(self, words: list[str]) -> list[dict]:
        """Find 4 groups of 4 from 16 words using greedy embedding similarity.

        Returns list of {"words": [...], "score": float} dicts.
        """
        words = [w.upper() for w in words]
        embeddings = self.model.encode(words, convert_to_numpy=True)

        # Precompute full similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # Score all C(16,4) = 1820 possible groups
        scored_groups = []
        for combo in combinations(range(len(words)), 4):
            score = self._group_score(combo, sim_matrix)
            scored_groups.append((combo, score))

        # Sort by score descending
        scored_groups.sort(key=lambda x: x[1], reverse=True)

        # Greedy: pick highest-scoring non-overlapping groups
        selected = []
        used_indices = set()

        for combo, score in scored_groups:
            if any(i in used_indices for i in combo):
                continue
            selected.append({
                "words": [words[i] for i in combo],
                "score": score,
            })
            used_indices.update(combo)
            if len(selected) == 4:
                break

        logger.info(f"Embedding solver found {len(selected)} groups")
        return selected

    @staticmethod
    def _group_score(indices: tuple, sim_matrix: np.ndarray) -> float:
        """Average pairwise similarity for a group of indices."""
        total = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total += sim_matrix[indices[i]][indices[j]]
                count += 1
        return total / count if count > 0 else 0.0
