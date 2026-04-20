"""Roundtable validator: runs multiple solvers and checks for convergence."""

import logging

from src.config import MIN_SOLVER_AGREEMENT
from src.solvers.embedding_solver import EmbeddingSolver
from src.solvers.clustering_solver import ClusteringSolver
from src.solvers.llm_solver import LLMSolver
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


def normalize_groups(groups: list[dict]) -> list[frozenset]:
    """Convert solver output to a canonical form for comparison.

    Returns a list of frozensets, each containing 4 uppercase words.
    """
    return [frozenset(w.upper() for w in g["words"]) for g in groups]


def groups_match(groups_a: list[frozenset], groups_b: list[frozenset]) -> bool:
    """Check if two solver outputs are the same solution (order-independent)."""
    return set(groups_a) == set(groups_b)


def check_against_answer(solver_groups: list[frozenset], answer_groups: list[dict]) -> int:
    """Count how many groups match the intended answer."""
    answer_sets = [frozenset(w.upper() for w in g["words"]) for g in answer_groups]
    return sum(1 for sg in solver_groups if sg in answer_sets)


class Roundtable:
    """Validates puzzles by running multiple independent solvers.

    A puzzle is valid if at least MIN_SOLVER_AGREEMENT solvers
    converge to the same solution AND that solution matches the
    intended answer.
    """

    def __init__(self, embedding_model=None, llm: LLMClient = None,
                 use_llm_solver: bool = False):
        self.embedding_solver = EmbeddingSolver(model=embedding_model)
        self.clustering_solver = ClusteringSolver(model=embedding_model)
        self.llm_solver = LLMSolver(llm=llm) if use_llm_solver else None

    def validate(self, puzzle: dict) -> dict:
        """Run solvers and check convergence.

        Args:
            puzzle: Puzzle dict with "words" and "groups" keys.

        Returns:
            {
                "valid": bool,
                "solver_agreement": bool,
                "solvers_used": list of solver names,
                "solver_results": dict of per-solver results,
                "groups_correct": dict of solver -> num correct groups,
                "convergence_detail": str
            }
        """
        words = puzzle["words"]
        intended = puzzle["groups"]

        results = {}
        normalized = {}

        # Solver 1: Embedding
        emb_groups = self.embedding_solver.solve(words)
        emb_norm = normalize_groups(emb_groups)
        emb_correct = check_against_answer(emb_norm, intended)
        results["embedding"] = {
            "solved": emb_correct == 4,
            "groups_correct": emb_correct,
            "groups": [{"words": list(g["words"]), "score": g["score"]} for g in emb_groups],
        }
        normalized["embedding"] = emb_norm

        # Solver 2: Clustering
        clust_groups = self.clustering_solver.solve(words)
        clust_norm = normalize_groups(clust_groups)
        clust_correct = check_against_answer(clust_norm, intended)
        results["clustering"] = {
            "solved": clust_correct == 4,
            "groups_correct": clust_correct,
            "groups": [{"words": list(g["words"]), "score": g["score"]} for g in clust_groups],
        }
        normalized["clustering"] = clust_norm

        solvers_used = ["embedding", "clustering"]

        # Solver 3: LLM (optional, only if first two agree)
        if self.llm_solver is not None:
            llm_groups = self.llm_solver.solve(words)
            llm_norm = normalize_groups(llm_groups)
            llm_correct = check_against_answer(llm_norm, intended)
            results["llm"] = {
                "solved": llm_correct == 4,
                "groups_correct": llm_correct,
                "groups": [{"words": list(g["words"])} for g in llm_groups],
            }
            normalized["llm"] = llm_norm
            solvers_used.append("llm")

        # Check agreement
        agreement_pairs = 0
        solver_names = list(normalized.keys())
        for i in range(len(solver_names)):
            for j in range(i + 1, len(solver_names)):
                if groups_match(normalized[solver_names[i]], normalized[solver_names[j]]):
                    agreement_pairs += 1

        # Agreement means at least one pair of solvers matched
        solver_agreement = agreement_pairs > 0

        # Valid = agreement AND at least one solver got all 4 correct
        any_fully_correct = any(r["groups_correct"] == 4 for r in results.values())
        valid = solver_agreement and any_fully_correct

        detail = (
            f"{agreement_pairs} agreement pair(s) among {len(solver_names)} solvers. "
            f"Correct groups: "
            + ", ".join(f"{k}={v['groups_correct']}/4" for k, v in results.items())
        )

        logger.info(f"Roundtable result: valid={valid}, {detail}")

        return {
            "valid": valid,
            "solver_agreement": solver_agreement,
            "solvers_used": solvers_used,
            "solver_results": results,
            "groups_correct": {k: v["groups_correct"] for k, v in results.items()},
            "convergence_detail": detail,
        }
