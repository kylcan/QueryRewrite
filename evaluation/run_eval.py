"""
Unified Evaluation Runner
=========================

Runs all evaluation modules (RQ1–RQ4) with a single data-loading and
encoding pass, saving individual metric files.

Usage::

    python -m evaluation.run_eval
    python -m evaluation.run_eval --dataset data/MS_MARCO/intent_dataset.jsonl
    python -m evaluation.run_eval --only dataset rewrite   # run subset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from evaluation.eval_dataset import evaluate_dataset
from evaluation.eval_gating import evaluate_gating
from evaluation.eval_rewrite import evaluate_rewrite
from evaluation.shared import RetrievalContext, save_metrics

_MODULES = {
    "dataset": ("RQ1+RQ2: Dataset Quality", evaluate_dataset),
    "rewrite": ("RQ3: Rewrite Effect", evaluate_rewrite),
    "gating":  ("RQ4: Gating Strategy", evaluate_gating),
}


def run_all(
    dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 20,
    only: list[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build context once, run selected evaluation modules."""

    ctx = RetrievalContext.build(dataset_path, model_name, top_k)
    out_dir = Path("evaluation/result")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Any]] = {}
    modules = only if only else list(_MODULES.keys())

    for name in modules:
        if name not in _MODULES:
            print(f"  ⚠ Unknown module '{name}', skipping.")
            continue
        label, func = _MODULES[name]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        results = func(ctx)
        save_metrics(results, str(out_dir / f"eval_{name}.json"))
        all_results[name] = results

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--only", nargs="*", choices=list(_MODULES.keys()),
                        help="Run only specific modules")
    args = parser.parse_args()

    run_all(
        dataset_path=args.dataset,
        model_name=args.model,
        top_k=args.top_k,
        only=args.only,
    )


if __name__ == "__main__":
    main()
