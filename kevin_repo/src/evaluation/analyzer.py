"""Statistical analysis and visualization for puzzle quality."""

import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np

from src.evaluation.metrics import puzzle_quality_score

logger = logging.getLogger(__name__)


class PuzzleAnalyzer:
    """Analyzes collections of puzzles for quality metrics and distributions."""

    def __init__(self, model=None):
        self.model = model

    def analyze_dataset(self, puzzles: list[dict]) -> dict:
        """Compute aggregate statistics across a puzzle dataset.

        Returns:
            {
                "count": int,
                "avg_similarity": float,
                "similarity_std": float,
                "color_distribution": dict,
                "category_diversity": int,
                "method_distribution": dict,
                "solver_agreement_rate": float,
                "per_color_stats": dict,
            }
        """
        if not puzzles:
            return {"count": 0}

        all_sims = []
        color_counts = Counter()
        category_names = set()
        method_counts = Counter()
        agreement_count = 0

        per_color_sims = {"yellow": [], "green": [], "blue": [], "purple": []}

        for puzzle in puzzles:
            quality = puzzle_quality_score(puzzle, model=self.model)

            for gm in quality["groups"]:
                sim = gm.get("avg_pairwise_sim", 0)
                all_sims.append(sim)
                color = gm.get("color", "unknown")
                color_counts[color] += 1
                category_names.add(gm.get("category", ""))

                if color in per_color_sims:
                    per_color_sims[color].append(sim)

            meta = puzzle.get("metadata", {})
            method_counts[meta.get("generation_method", "unknown")] += 1
            if meta.get("solver_agreement"):
                agreement_count += 1

        sims_arr = np.array(all_sims) if all_sims else np.array([0])

        per_color_stats = {}
        for color, sims in per_color_sims.items():
            if sims:
                per_color_stats[color] = {
                    "mean": float(np.mean(sims)),
                    "std": float(np.std(sims)),
                    "count": len(sims),
                }

        return {
            "count": len(puzzles),
            "avg_similarity": float(sims_arr.mean()),
            "similarity_std": float(sims_arr.std()),
            "color_distribution": dict(color_counts),
            "category_diversity": len(category_names),
            "method_distribution": dict(method_counts),
            "solver_agreement_rate": agreement_count / len(puzzles) if puzzles else 0,
            "per_color_stats": per_color_stats,
        }

    def compare_to_nyt(self, generated: list[dict], nyt: list[dict]) -> dict:
        """Compare generated puzzle distribution to NYT ground truth."""
        gen_stats = self.analyze_dataset(generated)
        nyt_stats = self.analyze_dataset(nyt)

        return {
            "generated": gen_stats,
            "nyt": nyt_stats,
            "similarity_diff": gen_stats["avg_similarity"] - nyt_stats["avg_similarity"],
            "diversity_ratio": (
                gen_stats["category_diversity"] / nyt_stats["category_diversity"]
                if nyt_stats.get("category_diversity", 0) > 0 else 0
            ),
        }

    @staticmethod
    def load_puzzles(path: str) -> list[dict]:
        """Load puzzles from a JSON file."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"File not found: {path}")
            return []
        with open(p) as f:
            return json.load(f)

    @staticmethod
    def convert_nyt_format(nyt_puzzles: list[dict]) -> list[dict]:
        """Convert NYT dataset format to our internal puzzle format."""
        converted = []
        for nyt in nyt_puzzles:
            groups = []
            for i, answer in enumerate(nyt.get("answers", [])):
                colors = ["yellow", "green", "blue", "purple"]
                groups.append({
                    "category": answer.get("answerDescription", "UNKNOWN"),
                    "words": [w.upper() for w in answer.get("words", [])],
                    "color": colors[i] if i < len(colors) else "purple",
                    "similarity_score": 0.0,
                })

            converted.append({
                "id": nyt.get("date", "unknown"),
                "words": [w.upper() for w in nyt.get("words", [])],
                "groups": groups,
                "metadata": {
                    "generation_method": "nyt_original",
                    "solver_agreement": None,
                    "solvers_used": [],
                    "solver_results": {},
                    "dedup_check": True,
                    "overall_difficulty": nyt.get("difficulty", 0),
                    "created_at": nyt.get("date", ""),
                },
            })
        return converted
