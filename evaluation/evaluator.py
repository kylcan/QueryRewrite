"""High-level evaluator that runs retrieval and computes all metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from retrieval.searcher import Searcher


class RetrievalEvaluator:
    """Runs end-to-end retrieval evaluation on a test set.

    Parameters
    ----------
    searcher : Searcher
        A fully initialised searcher with a built index.
    k_values : list[int]
        Cut-off ranks to evaluate (e.g. ``[1, 5, 10, 20, 50, 100]``).
    metrics : list[str]
        Metric names to compute.  Supported: ``"recall"``, ``"ndcg"``, ``"mrr"``.
    """

    def __init__(
        self,
        searcher: Searcher,
        k_values: Optional[List[int]] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError

    def evaluate(
        self,
        queries: List[str],
        relevant_ids: List[List[int]],
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Run retrieval for all queries and compute metrics.

        Parameters
        ----------
        queries : list[str]
            Raw query strings.
        relevant_ids : list[list[int]]
            Ground-truth relevant doc IDs per query.
        device : str
            Device for encoding.

        Returns
        -------
        dict[str, float]
            Metric name → score mapping
            (e.g. ``{"recall@10": 0.85, "ndcg@10": 0.72, ...}``).
        """
        raise NotImplementedError

    def format_results(self, results: Dict[str, float]) -> str:
        """Pretty-print evaluation results as a table string.

        Parameters
        ----------
        results : dict[str, float]
            Output of ``evaluate()``.

        Returns
        -------
        str
            Human-readable results table.
        """
        raise NotImplementedError
