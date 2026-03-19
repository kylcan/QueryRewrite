"""
Unified Evaluation Runner
=========================

Runs all evaluation modules (RQ1–RQ5) with a single data-loading and
encoding pass for RQ1–RQ4, plus a standalone alignment eval for RQ5.

Usage::

    python -m evaluation.run_eval
    python -m evaluation.run_eval --dataset data/MS_MARCO/intent_dataset.jsonl
    python -m evaluation.run_eval --only dataset rewrite   # run subset
    python -m evaluation.run_eval --only alignment          # RQ5 only
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

_ALL_CHOICES = list(_MODULES.keys()) + ["alignment", "rewriter"]


def run_all(
    dataset_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 20,
    only: list[str] | None = None,
    checkpoint: str = "checkpoints/alignment/best",
    sft_checkpoint: str = "checkpoints/sft/best",
    dpo_checkpoint: str = "checkpoints/dpo/best",
) -> Dict[str, Dict[str, Any]]:
    """Build context once, run selected evaluation modules."""

    out_dir = Path("evaluation/result")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict[str, Any]] = {}

    modules = only if only else list(_MODULES.keys())

    # RQ1–RQ4: share a single RetrievalContext
    ctx_modules = [m for m in modules if m in _MODULES]
    if ctx_modules:
        ctx = RetrievalContext.build(dataset_path, model_name, top_k)
        for name in ctx_modules:
            label, func = _MODULES[name]
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            results = func(ctx)
            save_metrics(results, str(out_dir / f"eval_{name}.json"))
            all_results[name] = results

    # RQ5: alignment (standalone — builds its own context)
    if "alignment" in modules:
        print(f"\n{'='*60}")
        print("  RQ5: Alignment Training Effect")
        print(f"{'='*60}")
        from evaluation.eval_alignment import evaluate_alignment
        results = evaluate_alignment(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint,
            base_model_name=f"sentence-transformers/{model_name}"
            if "/" not in model_name else model_name,
            top_k=top_k,
        )
        if "error" not in results:
            save_metrics(results, str(out_dir / "eval_alignment.json"))
            all_results["alignment"] = results

    # RQ6: rewriter comparison (standalone)
    if "rewriter" in modules:
        print(f"\n{'='*60}")
        print("  RQ6: Local Rewriter (SFT / DPO vs API)")
        print(f"{'='*60}")
        from evaluation.eval_rewriter import evaluate_rewriter
        embedder_name = (
            f"sentence-transformers/{model_name}"
            if "/" not in model_name else model_name
        )
        results = evaluate_rewriter(
            dataset_path=dataset_path,
            embedder_checkpoint=checkpoint,
            sft_checkpoint=sft_checkpoint,
            dpo_checkpoint=dpo_checkpoint,
            embedder_model_name=embedder_name,
            top_k=top_k,
        )
        if "error" not in results:
            save_metrics(results, str(out_dir / "eval_rewriter.json"))
            all_results["rewriter"] = results

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--dataset", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--checkpoint", default="checkpoints/alignment/best",
                        help="Path to fine-tuned model checkpoint (for RQ5)")
    parser.add_argument("--sft_checkpoint", default="checkpoints/sft/best",
                        help="Path to SFT adapter (for RQ6)")
    parser.add_argument("--dpo_checkpoint", default="checkpoints/dpo/best",
                        help="Path to DPO adapter (for RQ6)")
    parser.add_argument("--only", nargs="*", choices=_ALL_CHOICES,
                        help="Run only specific modules")
    args = parser.parse_args()

    run_all(
        dataset_path=args.dataset,
        model_name=args.model,
        top_k=args.top_k,
        only=args.only,
        checkpoint=args.checkpoint,
        sft_checkpoint=args.sft_checkpoint,
        dpo_checkpoint=args.dpo_checkpoint,
    )


if __name__ == "__main__":
    main()
