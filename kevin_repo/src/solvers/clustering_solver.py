"""Clustering solver: Group Similarity + Penalty scoring with beam search."""

import logging
from itertools import combinations

import numpy as np

from src.config import BEAM_WIDTH

logger = logging.getLogger(__name__)


class ClusteringSolver:
    """Solves Connections using the scoring formula from 'Deceptively Simple'.

    Group Similarity Score: G = 0.4*I + 0.3*s + 0.3*V
    - I = -K(E): negative k-means inertia (k=1) on group embeddings
    - s = min pairwise cosine similarity
    - V = mean(P) / (1 + var(P)) where P = all pairwise similarities

    Penalty Score: P = (1/|R|) * sum(cos(mu_C, r)) for remaining words R

    Uses beam search (width=10) to find the best complete 4-group solution.
    """

    def __init__(self, model=None, beam_width: int = BEAM_WIDTH):
        self._model = model
        self.beam_width = beam_width

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from src.config import EMBEDDING_MODEL
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def solve(self, words: list[str]) -> list[dict]:
        """Find 4 groups of 4 using beam search with G-P scoring."""
        words = [w.upper() for w in words]
        embeddings = self.model.encode(words, convert_to_numpy=True)

        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # Beam search
        # State: (groups_so_far, remaining_indices, cumulative_score)
        initial_remaining = set(range(len(words)))
        beam = [([], initial_remaining, 0.0)]

        for step in range(4):
            candidates = []
            for groups, remaining, cum_score in beam:
                remaining_list = sorted(remaining)
                if len(remaining_list) < 4:
                    continue

                for combo in combinations(remaining_list, 4):
                    g_score = self._group_similarity(combo, embeddings, sim_matrix)
                    new_remaining = remaining - set(combo)
                    p_score = self._penalty_score(
                        combo, new_remaining, embeddings, sim_matrix
                    )
                    total = g_score - p_score
                    new_groups = groups + [list(combo)]
                    candidates.append(
                        (new_groups, new_remaining, cum_score + total)
                    )

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:self.beam_width]

            if not beam:
                break

        if not beam:
            logger.warning("Beam search found no solution")
            return []

        best_groups, _, best_score = beam[0]
        result = []
        for group_indices in best_groups:
            group_words = [words[i] for i in group_indices]
            score = self._avg_similarity(group_indices, sim_matrix)
            result.append({"words": group_words, "score": score})

        logger.info(f"Clustering solver found {len(result)} groups (score={best_score:.3f})")
        return result

    def _group_similarity(self, indices: tuple, embeddings: np.ndarray,
                          sim_matrix: np.ndarray) -> float:
        """G = 0.4*I + 0.3*s + 0.3*V"""
        group_embs = embeddings[list(indices)]

        # I = negative inertia (k=1 means distance from centroid)
        centroid = group_embs.mean(axis=0)
        distances = np.sum((group_embs - centroid) ** 2, axis=1)
        inertia = distances.sum()
        I = -inertia

        # Pairwise similarities
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.append(sim_matrix[indices[i]][indices[j]])

        if not pairs:
            return 0.0

        s = min(pairs)  # min pairwise similarity
        mean_p = np.mean(pairs)
        var_p = np.var(pairs)
        V = mean_p / (1 + var_p)

        return 0.4 * I + 0.3 * s + 0.3 * V

    def _penalty_score(self, group_indices: tuple, remaining: set,
                       embeddings: np.ndarray, sim_matrix: np.ndarray) -> float:
        """P = (1/|R|) * sum(cos(centroid, r)) for remaining words."""
        if not remaining:
            return 0.0

        group_embs = embeddings[list(group_indices)]
        centroid = group_embs.mean(axis=0)

        # Compute cosine similarity between centroid and remaining words
        remaining_list = list(remaining)
        remaining_embs = embeddings[remaining_list]

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        sims = cos_sim(centroid.reshape(1, -1), remaining_embs)[0]

        return float(sims.mean())

    @staticmethod
    def _avg_similarity(indices, sim_matrix: np.ndarray) -> float:
        total = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total += sim_matrix[indices[i]][indices[j]]
                count += 1
        return total / count if count > 0 else 0.0
