"""Compatibility wrapper for evaluation helpers."""

from evaluate import evaluate_ranking, ndcg_at_k, recall_at_k

__all__ = ["evaluate_ranking", "ndcg_at_k", "recall_at_k"]
