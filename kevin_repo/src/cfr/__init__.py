"""Category-First Retrieval (CFR) — a less-LLM-reliant puzzle generator.

Instead of generating 8 candidate words via LLM and using MPNET to pick 4,
CFR flips the flow:
    1. LLM (or NYT bank) proposes 4 category NAMES
    2. MPNET retrieves candidate words for each name via KNN
    3. MPNET selects the best 4 (existing EmbeddingSelector)

This cuts LLM calls from 5 per puzzle to 0 (Mode A) or 1 (Mode B).

Completely separate from the existing src/generator/ pipeline.
"""

from src.cfr.pipeline import CFRPipeline
from src.cfr.embedding_retriever import EmbeddingRetriever

__all__ = ["CFRPipeline", "EmbeddingRetriever"]
