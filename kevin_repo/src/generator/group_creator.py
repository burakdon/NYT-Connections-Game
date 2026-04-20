"""Creates word groups using LLM + MPNET embedding selection."""

import json
import random
import logging
from typing import Optional

import numpy as np

from src.config import WORDS_PER_POOL, WORDS_PER_GROUP, RANDOM_SEED
from src.llm_client import LLMClient
from src.generator.prompts import (
    format_group_creation,
    format_false_group,
    format_alternate_meaning,
)

logger = logging.getLogger(__name__)


class EmbeddingSelector:
    """Selects the best N words from a pool using MPNET cosine similarity."""

    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from src.config import EMBEDDING_MODEL
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def select_best(self, words: list[str], n: int = WORDS_PER_GROUP) -> tuple[list[str], float]:
        """Select the n most internally similar words from the pool.

        Returns (selected_words, avg_similarity_score).
        """
        if len(words) <= n:
            return words[:n], 0.0

        embeddings = self.model.encode(words, convert_to_numpy=True)
        best_score = -1.0
        best_combo = None

        from itertools import combinations
        for combo_indices in combinations(range(len(words)), n):
            combo_embs = embeddings[list(combo_indices)]
            score = self._avg_pairwise_similarity(combo_embs)
            if score > best_score:
                best_score = score
                best_combo = combo_indices

        selected = [words[i] for i in best_combo]
        return selected, float(best_score)

    @staticmethod
    def _avg_pairwise_similarity(embeddings: np.ndarray) -> float:
        """Average cosine similarity across all pairs."""
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        n = len(embeddings)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sim_matrix[i][j]
                count += 1
        return total / count if count > 0 else 0.0


class GroupCreator:
    """Creates word groups for Connections puzzles."""

    def __init__(self, llm: LLMClient, selector: EmbeddingSelector,
                 word_bank: Optional[list[str]] = None):
        self.llm = llm
        self.selector = selector
        self.word_bank = word_bank or []
        self.rng = random.Random(RANDOM_SEED)

    def create_group_iterative(self, previous_groups: list[dict] = None) -> dict:
        """Create one group via iterative method with story injection."""
        seed_words = None
        if self.word_bank:
            seed_words = self.rng.sample(self.word_bank, min(4, len(self.word_bank)))

        system, user = format_group_creation(
            seed_words=seed_words,
            previous_groups=previous_groups,
        )
        response = self.llm.complete(system, user)
        data = json.loads(response)

        candidate_words = [w.upper() for w in data["words"][:WORDS_PER_POOL]]
        category = data["category"].upper()

        # Use MPNET to pick best 4 from the 8 candidates
        selected, score = self.selector.select_best(candidate_words, WORDS_PER_GROUP)

        # Remove any words already used in previous groups
        used_words = set()
        if previous_groups:
            for g in previous_groups:
                used_words.update(w.upper() for w in g["words"])

        selected = [w for w in selected if w not in used_words]
        if len(selected) < WORDS_PER_GROUP:
            remaining = [w for w in candidate_words if w not in used_words and w not in selected]
            selected.extend(remaining[:WORDS_PER_GROUP - len(selected)])

        selected = selected[:WORDS_PER_GROUP]

        return {
            "category": category,
            "words": selected,
            "similarity_score": score,
            "candidate_pool": candidate_words,
        }

    def create_false_group_puzzle(self) -> list[dict]:
        """Create 4 groups via the false-group method.

        1. Generate a root group (the false group / decoy)
        2. For each root word, use its alternate meaning to generate a real group
        3. Return the 4 real groups (root group is discarded as a decoy)
        """
        system, user = format_false_group()
        response = self.llm.complete(system, user)
        root_data = json.loads(response)

        root_words = root_data["words"]
        alt_meanings = root_data.get("alternate_meanings", {})

        groups = []
        for word in root_words[:WORDS_PER_GROUP]:
            meaning = alt_meanings.get(word, f"alternate meaning of {word}")
            sys_prompt, usr_prompt = format_alternate_meaning(
                word=word,
                meaning=meaning,
                previous_groups=groups,
            )
            resp = self.llm.complete(sys_prompt, usr_prompt)
            data = json.loads(resp)

            candidate_words = [w.upper() for w in data["words"][:WORDS_PER_POOL]]
            category = data["category"].upper()

            # Remove words used in previous groups
            used_words = set()
            for g in groups:
                used_words.update(w.upper() for w in g["words"])
            candidate_words = [w for w in candidate_words if w not in used_words]

            selected, score = self.selector.select_best(candidate_words, WORDS_PER_GROUP)

            groups.append({
                "category": category,
                "words": selected[:WORDS_PER_GROUP],
                "similarity_score": score,
                "candidate_pool": candidate_words,
                "inspired_by": {"word": word, "meaning": meaning},
            })

        return groups
