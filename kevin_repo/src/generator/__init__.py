"""Puzzle generation pipeline."""

from src.generator.pipeline import PuzzlePipeline
from src.generator.group_creator import GroupCreator, EmbeddingSelector
from src.generator.puzzle_editor import PuzzleEditor
from src.generator.difficulty import assign_colors, compute_group_similarity
from src.generator.deduplicator import Deduplicator

__all__ = [
    "PuzzlePipeline",
    "GroupCreator",
    "EmbeddingSelector",
    "PuzzleEditor",
    "assign_colors",
    "compute_group_similarity",
    "Deduplicator",
]
