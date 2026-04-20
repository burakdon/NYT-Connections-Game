"""Quality metrics for evaluating generated puzzles."""

import numpy as np


def group_similarity_score(embeddings: np.ndarray) -> float:
    """Compute Group Similarity Score: G = 0.4*I + 0.3*s + 0.3*V

    Args:
        embeddings: (4, D) array of word embeddings for a group.

    Returns:
        Float score (higher = more cohesive group).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if len(embeddings) < 2:
        return 0.0

    # I = negative k-means inertia (k=1)
    centroid = embeddings.mean(axis=0)
    distances = np.sum((embeddings - centroid) ** 2, axis=1)
    I = -distances.sum()

    # Pairwise cosine similarities
    sim_matrix = cosine_similarity(embeddings)
    pairs = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(sim_matrix[i][j])

    if not pairs:
        return 0.0

    s = min(pairs)  # min pairwise similarity
    mean_p = np.mean(pairs)
    var_p = np.var(pairs)
    V = mean_p / (1.0 + var_p)

    return float(0.4 * I + 0.3 * s + 0.3 * V)


def penalty_score(group_embeddings: np.ndarray,
                  remaining_embeddings: np.ndarray) -> float:
    """Compute Penalty Score: P = (1/|R|) * sum(cos(centroid, r)).

    Measures how much the remaining words are attracted to this group's centroid.
    Lower penalty = better separation.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if len(remaining_embeddings) == 0:
        return 0.0

    centroid = group_embeddings.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(centroid, remaining_embeddings)[0]
    return float(sims.mean())


def avg_pairwise_similarity(embeddings: np.ndarray) -> float:
    """Simple average pairwise cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    if len(embeddings) < 2:
        return 0.0

    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i][j]
            count += 1
    return total / count if count > 0 else 0.0


def puzzle_quality_score(puzzle: dict, model=None) -> dict:
    """Compute comprehensive quality metrics for a puzzle.

    Returns dict with per-group and overall metrics.
    """
    if model is None:
        return _quality_from_stored_scores(puzzle)

    groups = puzzle.get("groups", [])
    all_words = []
    group_metrics = []

    for g in groups:
        words = g["words"]
        embs = model.encode(words, convert_to_numpy=True)
        all_words.extend(words)

        g_score = group_similarity_score(embs)
        avg_sim = avg_pairwise_similarity(embs)

        group_metrics.append({
            "category": g.get("category", "UNKNOWN"),
            "color": g.get("color", "unknown"),
            "group_similarity": g_score,
            "avg_pairwise_sim": avg_sim,
            "word_count": len(words),
        })

    overall_sim = np.mean([m["avg_pairwise_sim"] for m in group_metrics]) if group_metrics else 0
    sim_spread = (
        max(m["avg_pairwise_sim"] for m in group_metrics)
        - min(m["avg_pairwise_sim"] for m in group_metrics)
    ) if group_metrics else 0

    return {
        "groups": group_metrics,
        "overall_avg_similarity": float(overall_sim),
        "similarity_spread": float(sim_spread),
        "total_words": len(all_words),
        "unique_words": len(set(w.upper() for w in all_words)),
    }


def _quality_from_stored_scores(puzzle: dict) -> dict:
    """Compute quality metrics from pre-stored similarity scores."""
    groups = puzzle.get("groups", [])
    group_metrics = []
    all_words = []

    for g in groups:
        all_words.extend(g["words"])
        group_metrics.append({
            "category": g.get("category", "UNKNOWN"),
            "color": g.get("color", "unknown"),
            "avg_pairwise_sim": g.get("similarity_score", 0.0),
            "word_count": len(g["words"]),
        })

    sims = [m["avg_pairwise_sim"] for m in group_metrics]

    return {
        "groups": group_metrics,
        "overall_avg_similarity": float(np.mean(sims)) if sims else 0,
        "similarity_spread": float(max(sims) - min(sims)) if sims else 0,
        "total_words": len(all_words),
        "unique_words": len(set(w.upper() for w in all_words)),
    }
