"""Difficulty color assignment using MPNET cosine similarity thresholds."""

import logging

import numpy as np

from src.config import COLOR_THRESHOLDS

logger = logging.getLogger(__name__)

COLORS_BY_DIFFICULTY = ["yellow", "green", "blue", "purple"]


def compute_group_similarity(words: list[str], model) -> float:
    """Compute average pairwise cosine similarity for a word group."""
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings = model.encode(words, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings)
    n = len(words)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i][j]
            count += 1
    return total / count if count > 0 else 0.0


def assign_colors(groups: list[dict], model=None) -> list[dict]:
    """Assign difficulty colors to groups based on cosine similarity.

    Groups are sorted by similarity: highest -> yellow (easiest),
    lowest -> purple (hardest).
    """
    if model is not None:
        for group in groups:
            score = compute_group_similarity(group["words"], model)
            group["similarity_score"] = score

    # Sort by similarity score descending (highest similarity = easiest)
    scored = sorted(groups, key=lambda g: g.get("similarity_score", 0), reverse=True)

    for i, group in enumerate(scored):
        if i < len(COLORS_BY_DIFFICULTY):
            group["color"] = COLORS_BY_DIFFICULTY[i]
        else:
            group["color"] = "purple"

    logger.info(
        "Color assignment: "
        + ", ".join(f"{g['category']}={g['color']} ({g.get('similarity_score', 0):.3f})" for g in scored)
    )
    return scored
