from evaluation.metrics import (
    recall_at_k,
    recall_at_k_arr,
    ndcg_at_k,
    mrr,
    mrr_arr,
    hard_neg_in_topk,
)

__all__ = [
    "recall_at_k",
    "recall_at_k_arr",
    "ndcg_at_k",
    "mrr",
    "mrr_arr",
    "hard_neg_in_topk",
    "RetrievalContext",
]


def __getattr__(name: str):  # type: ignore[override]
    """Lazy import for heavy modules to avoid faiss/torch import conflicts."""
    if name == "RetrievalContext":
        from evaluation.shared import RetrievalContext
        return RetrievalContext
    if name == "RetrievalEvaluator":
        from evaluation.evaluator import RetrievalEvaluator
        return RetrievalEvaluator
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
