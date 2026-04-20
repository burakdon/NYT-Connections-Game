"""CFRPipeline — Category-First Retrieval puzzle generator.

Two modes:
  - "remix": 0 LLM calls. Samples 4 diverse NYT categories + retrieves words.
  - "fresh": 1 LLM call. Asks LLM for 4 fresh categories + retrieves words.

Imports (but does not modify) existing generator/solver components.
"""

import json
import logging
import random
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

import numpy as np

from src.config import (
    GROUPS_PER_PUZZLE, WORDS_PER_GROUP, WORDS_PER_POOL, RANDOM_SEED,
)
from src.cfr.embedding_retriever import EmbeddingRetriever
from src.cfr.prompts import format_batch_categories

# Reused (but never modified) components from the existing pipeline
from src.generator.group_creator import EmbeddingSelector
from src.generator.difficulty import assign_colors
from src.generator.deduplicator import Deduplicator

logger = logging.getLogger(__name__)


def _rank_combinations(
    words: list[str], embeddings: np.ndarray, n: int = WORDS_PER_GROUP
) -> list[tuple[float, list[str]]]:
    """Score all C(len(words), n) combinations by avg pairwise cosine similarity.

    Returns list of (score, words) pairs sorted by score descending. Parallels
    the existing EmbeddingSelector.select_best() enumeration but returns ALL
    ranked combinations so we can skip the best if it collides with a NYT group.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if len(words) < n:
        return [(0.0, words[:n])]

    sim_matrix = cosine_similarity(embeddings)
    ranked = []
    for combo_indices in combinations(range(len(words)), n):
        # avg of C(n,2) pairwise similarities
        total = 0.0
        count = 0
        for i in range(len(combo_indices)):
            for j in range(i + 1, len(combo_indices)):
                total += sim_matrix[combo_indices[i]][combo_indices[j]]
                count += 1
        score = total / count if count > 0 else 0.0
        combo_words = [words[k] for k in combo_indices]
        ranked.append((float(score), combo_words))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


class CFRPipeline:
    """Category-First Retrieval puzzle generation pipeline."""

    def __init__(
        self,
        retriever: EmbeddingRetriever,
        selector: Optional[EmbeddingSelector] = None,
        llm=None,
        deduplicator: Optional[Deduplicator] = None,
        embedding_model=None,
        seed: int = RANDOM_SEED,
    ):
        self.retriever = retriever
        self.selector = selector or EmbeddingSelector(model=embedding_model)
        self.embedding_model = embedding_model or self.retriever.model
        self.llm = llm  # only needed for Mode B ("fresh")
        self.deduplicator = deduplicator or Deduplicator()
        self.rng = random.Random(seed)
        self._puzzle_counter = 0

        # Ensure retriever precomputed
        if self.retriever._knn is None:
            self.retriever.precompute()

    def generate(self, mode: str = "remix") -> dict:
        """Generate a single puzzle.

        Args:
            mode: "remix" (0 LLM calls) or "fresh" (1 LLM call)

        Returns:
            Puzzle dict matching the project schema.
        """
        if mode == "remix":
            categories = self._get_categories_remix()
        elif mode == "fresh":
            categories = self._get_categories_fresh()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build groups by KNN retrieval + MPNET selection
        groups = self._build_groups_from_categories(categories)

        # Assign difficulty colors (reuses existing non-LLM logic)
        groups = assign_colors(groups, model=self.embedding_model)

        # Collect all words, dedup (word-overlap + exact-group check), shuffle
        all_words = []
        for g in groups:
            all_words.extend(g["words"])

        dedup_result = self.deduplicator.check(all_words, groups=groups)

        shuffled_words = list(all_words)
        self.rng.shuffle(shuffled_words)

        self._puzzle_counter += 1
        puzzle_id = f"CFR-{self._puzzle_counter:05d}"

        clean_groups = [
            {
                "category": g["category"],
                "words": g["words"],
                "color": g.get("color", "yellow"),
                "similarity_score": g.get("similarity_score", 0.0),
            }
            for g in groups
        ]

        puzzle = {
            "id": puzzle_id,
            "words": shuffled_words,
            "groups": clean_groups,
            "metadata": {
                "generation_method": f"cfr_{mode}",
                "solver_agreement": None,
                "solvers_used": [],
                "solver_results": {},
                "dedup_check": not dedup_result["is_duplicate"],
                "dedup_reason": dedup_result.get("reason", ""),
                "exact_group_match": dedup_result.get("exact_group_match", False),
                "overall_difficulty": (
                    sum(g.get("similarity_score", 0) for g in clean_groups)
                    / len(clean_groups)
                    if clean_groups
                    else 0
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        logger.info(f"Generated puzzle {puzzle_id} ({mode})")
        return puzzle

    # ------------------------------------------------------------------
    # Step 1: Get 4 categories

    def _get_categories_remix(self) -> list[str]:
        """Mode A: sample 4 diverse categories from the NYT pool."""
        return self.retriever.sample_diverse_categories(
            n=GROUPS_PER_PUZZLE,
            min_distance=0.35,
            rng=self.rng,
        )

    def _get_categories_fresh(self) -> list[str]:
        """Mode B: one LLM call to generate 4 fresh category names."""
        if self.llm is None:
            raise RuntimeError("Mode 'fresh' requires an LLM client")

        # Use a few random NYT words as optional inspiration
        seed_words = self.rng.sample(
            self.retriever.word_bank, min(4, len(self.retriever.word_bank))
        )
        system, user = format_batch_categories(seed_words=seed_words)
        response = self.llm.complete(system, user)

        data = json.loads(response)
        cats = data.get("categories", [])
        if len(cats) < GROUPS_PER_PUZZLE:
            # Fallback: pad with remix samples
            logger.warning(
                f"LLM returned only {len(cats)} categories; padding from NYT pool."
            )
            extras = self.retriever.sample_diverse_categories(
                n=GROUPS_PER_PUZZLE - len(cats),
                rng=self.rng,
            )
            cats.extend(extras)

        return [c.upper() for c in cats[:GROUPS_PER_PUZZLE]]

    # ------------------------------------------------------------------
    # Step 2: KNN retrieve + MPNET select 4 (with NYT-group-match guard)

    def _pick_best_non_nyt_combo(
        self, candidate_words: list[str]
    ) -> tuple[list[str], float, bool]:
        """Pick the highest-scoring 4-word combo that is NOT a verbatim NYT group.

        Returns (words, score, avoided_collision). `avoided_collision` is True
        if the best combo was skipped because it matched a past NYT group.
        """
        if len(candidate_words) < WORDS_PER_GROUP:
            return candidate_words[:WORDS_PER_GROUP], 0.0, False

        embeddings = self.embedding_model.encode(
            candidate_words, convert_to_numpy=True, show_progress_bar=False
        )
        ranked = _rank_combinations(candidate_words, embeddings, WORDS_PER_GROUP)

        nyt_groups = self.deduplicator.nyt_group_set
        avoided = False
        for score, combo in ranked:
            if frozenset(w.upper() for w in combo) not in nyt_groups:
                return combo, score, avoided
            avoided = True  # best choice was a collision; try next

        # All 70 combos collide with NYT groups (extremely unlikely) -- return top
        return ranked[0][1], ranked[0][0], avoided

    def _build_groups_from_categories(self, categories: list[str]) -> list[dict]:
        """For each category, retrieve candidates and pick best 4.

        Enforces: no generated group may exactly equal a past NYT group.
        """
        groups = []
        used_words = set()
        total_collisions_avoided = 0

        for category in categories:
            # KNN retrieve top candidates (exclude already-used words)
            candidates = self.retriever.retrieve(
                category_name=category,
                top_k=WORDS_PER_POOL + 4,
                exclude_words=used_words,
            )
            candidate_words = [w for (w, _sim) in candidates[:WORDS_PER_POOL]]

            if len(candidate_words) < WORDS_PER_GROUP:
                logger.warning(
                    f"Too few candidates for '{category}' "
                    f"({len(candidate_words)}); padding from word bank."
                )
                unused = [
                    w for w in self.retriever.word_bank if w not in used_words
                ]
                self.rng.shuffle(unused)
                while len(candidate_words) < WORDS_PER_POOL and unused:
                    candidate_words.append(unused.pop())

            # Pick best 4 that is NOT a verbatim NYT group
            selected, score, avoided = self._pick_best_non_nyt_combo(candidate_words)
            if avoided:
                total_collisions_avoided += 1
                logger.info(
                    f"Mode-safe: skipped NYT group collision for '{category}'"
                )

            groups.append(
                {
                    "category": category,
                    "words": selected,
                    "similarity_score": score,
                    "candidate_pool": candidate_words,
                }
            )
            used_words.update(w.upper() for w in selected)

        if total_collisions_avoided > 0:
            logger.info(
                f"Avoided {total_collisions_avoided} NYT-group collisions in this puzzle"
            )
        return groups
