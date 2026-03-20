"""
LLM-as-Judge Preference Annotation
=====================================

Replaces cosine-similarity-based preference labels with GPT-4o judgments,
dramatically improving DPO data quality.

The judge is asked to compare two query rewrites and decide which one is
more likely to retrieve a relevant passage. This mimics human annotation
at scale and is the standard industrial practice.

Pipeline
--------
1. Load existing intent_dataset.jsonl (or dpo_dataset.jsonl)
2. For each query, generate K candidate rewrites (or use existing ones)
3. Ask GPT-4o to compare pairs: which rewrite would better find the answer?
4. Output high-quality (prompt, chosen, rejected) pairs

Usage::

    python data/MS_MARCO/llm_judge.py
    python data/MS_MARCO/llm_judge.py --input data/MS_MARCO/intent_dataset.jsonl
    python data/MS_MARCO/llm_judge.py --max_samples 1000 --concurrency 5
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.MS_MARCO.build_preference_data import format_sft_prompt

# ── Judge prompt ──────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert at information retrieval and search quality evaluation.
You will be shown a user's original search query and two rewritten versions.
Your task: decide which rewrite would be MORE EFFECTIVE at retrieving a relevant passage.

A good rewrite:
- Adds specific terminology that would appear in encyclopedia-style passages
- Clarifies ambiguous intent
- Uses noun phrases rather than conversational phrasing
- Does NOT change the meaning of the original query

Respond with EXACTLY one line: either "A" or "B" (the better rewrite), followed by a brief reason.
Format: <choice>|<reason>
Example: A|More specific terminology improves passage matching"""

JUDGE_USER_TEMPLATE = """Original query: {original}

Rewrite A: {rewrite_a}
Rewrite B: {rewrite_b}

Which rewrite is better for information retrieval?"""


# ── OpenAI client helper ──────────────────────────────────────

def _get_openai_client():
    from openai import OpenAI
    api_key = os.getenv("GPT5_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = (
        os.getenv("CHATGPT_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    if not api_key:
        raise EnvironmentError(
            "No OpenAI API key found. Set GPT5_KEY or OPENAI_API_KEY."
        )
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _get_model_name(model_name: Optional[str] = None) -> str:
    return (
        model_name
        or os.getenv("JUDGE_MODEL")
        or os.getenv("CHATGPT_MODEL")
        or os.getenv("AUDIT_MODEL")
        or "gpt-4o"
    )


# ── Single comparison call ────────────────────────────────────

def judge_pair(
    client: Any,
    model: str,
    original: str,
    rewrite_a: str,
    rewrite_b: str,
    max_retries: int = 3,
) -> Tuple[str, str]:
    """Ask the LLM judge to compare two rewrites.

    Returns
    -------
    (winner, reason) — winner is "A" or "B"
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(
                            original=original,
                            rewrite_a=rewrite_a,
                            rewrite_b=rewrite_b,
                        ),
                    },
                ],
                max_tokens=80,
                temperature=0.0,  # deterministic
            )
            text = resp.choices[0].message.content.strip()
            # Parse "A|reason" or "B|reason"
            parts = text.split("|", 1)
            choice = parts[0].strip().upper()
            reason = parts[1].strip() if len(parts) > 1 else ""
            if choice in ("A", "B"):
                return choice, reason
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Judge failed after {max_retries} retries: {e}")
    return "A", "fallback"  # default to A on failure


def _load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load existing judged pairs keyed by query_id."""
    if not path.exists():
        return {}

    existing: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            query_id = record.get("query_id")
            if isinstance(query_id, int):
                existing[query_id] = record
    return existing


def _compute_stats(results: List[Dict[str, Any]], total_records: int) -> Dict[str, Any]:
    """Compute aggregate judge statistics from final judged pairs."""
    stats = {
        "total": total_records,
        "judge_A_wins": 0,
        "judge_B_wins": 0,
        "agree_with_cosine": 0,
        "disagree_with_cosine": 0,
        "cost_approx_usd": 0.0,
        "judged_pairs": len(results),
    }

    for record in results:
        if record.get("judge_prefers_rewrite"):
            stats["judge_A_wins"] += 1
        else:
            stats["judge_B_wins"] += 1

        if record.get("judge_agrees_cosine"):
            stats["agree_with_cosine"] += 1
        else:
            stats["disagree_with_cosine"] += 1

    stats["cost_approx_usd"] = round(len(results) * 200 * 5e-6, 3)
    return stats


def _judge_record(
    item_idx: int,
    record: Dict[str, Any],
    client: Any,
    model: str,
) -> Tuple[int, Dict[str, Any]]:
    """Judge a single reward-annotated record and return the DPO pair."""
    original = record["query_original"]
    rewrite = record["query_rewrite"]
    cosine_delta = record.get("reward_cos_delta", 0.0)

    flip = random.random() < 0.5
    rewrite_a, rewrite_b = (rewrite, original) if flip else (original, rewrite)
    choice, reason = judge_pair(client, model, original, rewrite_a, rewrite_b)

    if flip:
        judge_chosen = rewrite if choice == "A" else original
        judge_rejected = original if choice == "A" else rewrite
    else:
        judge_chosen = original if choice == "A" else rewrite
        judge_rejected = rewrite if choice == "A" else original

    judge_prefers_rewrite = judge_chosen == rewrite
    cosine_prefers_rewrite = cosine_delta > 0

    result = {
        "query_id": record.get("query_id", item_idx),
        "prompt": format_sft_prompt(original),
        "chosen": judge_chosen,
        "rejected": judge_rejected,
        "judge_reason": reason,
        "reward_cos_delta": cosine_delta,
        "judge_agrees_cosine": judge_prefers_rewrite == cosine_prefers_rewrite,
        "judge_prefers_rewrite": judge_prefers_rewrite,
    }
    return item_idx, result


# ── Main annotation pipeline ──────────────────────────────────

def build_judged_dpo_dataset(
    input_path: str = "data/MS_MARCO/reward_annotated.jsonl",
    output_path: str = "data/MS_MARCO/dpo_dataset_judged.jsonl",
    stats_path: str = "data/MS_MARCO/judge_stats.json",
    max_samples: Optional[int] = None,
    min_cosine_gap: float = 0.005,
    judge_model: Optional[str] = None,
    resume: bool = False,
    judge_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Build LLM-judged DPO preference pairs.

    For each query, the judge compares:
      - query_rewrite (API LLM output) vs query_original

    Samples where the judge's verdict agrees with retrieval signal are
    highest confidence; disagreements are still included but flagged.

    Parameters
    ----------
    input_path : str
        Path to reward_annotated.jsonl (must have query_rewrite and
        reward_cos_delta fields).
    output_path : str
        Where to save the judged DPO pairs.
    max_samples : int | None
        Limit for cost control. None = all samples.
    min_cosine_gap : float
        Skip samples where cosine delta is near zero (uninformative).
    """
    # Load data
    records: List[Dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    client = _get_openai_client()
    model = _get_model_name(judge_model)
    print(f"  Using model: {model}")
    print(f"  Judging {len(records)} samples …")

    output_file = Path(output_path)
    existing_results = _load_existing_results(output_file) if resume else {}
    if existing_results:
        print(f"  Resume enabled: loaded {len(existing_results)} cached judged pairs from {output_file}")

    eligible_items: List[Tuple[int, Dict[str, Any]]] = []
    for i, record in enumerate(records):
        cosine_delta = record.get("reward_cos_delta", 0.0)
        if abs(cosine_delta) < min_cosine_gap:
            continue

        query_id = record.get("query_id", i)
        if not isinstance(query_id, int):
            query_id = i

        if query_id in existing_results:
            continue
        eligible_items.append((i, record))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume and output_file.exists() else "w"

    if eligible_items:
        if judge_workers <= 1:
            with open(output_file, write_mode, encoding="utf-8") as out_f:
                with tqdm(total=len(existing_results) + len(eligible_items), initial=len(existing_results), desc="Judging") as progress:
                    for item_idx, record in eligible_items:
                        _, result = _judge_record(item_idx, record, client, model)
                        existing_results[result["query_id"]] = result
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                        progress.update(1)
        else:
            max_workers = min(max(judge_workers, 1), len(eligible_items))
            print(f"  Judging with {max_workers} worker threads …")
            with open(output_file, write_mode, encoding="utf-8") as out_f:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_item: Dict[Future[Tuple[int, Dict[str, Any]]], Tuple[int, Dict[str, Any]]] = {}

                    initial_batch = eligible_items[:max_workers]
                    remaining = eligible_items[max_workers:]
                    for item in initial_batch:
                        future = executor.submit(_judge_record, item[0], item[1], client, model)
                        future_to_item[future] = item

                    with tqdm(total=len(existing_results) + len(eligible_items), initial=len(existing_results), desc="Judging") as progress:
                        while future_to_item:
                            done, _ = wait(list(future_to_item.keys()), return_when=FIRST_COMPLETED)
                            for future in done:
                                future_to_item.pop(future)
                                _, result = future.result()
                                existing_results[result["query_id"]] = result
                                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                out_f.flush()
                                progress.update(1)

                                if remaining:
                                    next_item = remaining.pop(0)
                                    next_future = executor.submit(_judge_record, next_item[0], next_item[1], client, model)
                                    future_to_item[next_future] = next_item
    else:
        print("  All eligible judged pairs already cached; skipping generation.")

    results = [existing_results[qid] for qid in sorted(existing_results)]
    stats = _compute_stats(results, total_records=len(records))

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  ── Judge Statistics ──")
    print(f"  Judged pairs:             {stats['judged_pairs']}")
    print(f"  Judge prefers rewrite:    {stats['judge_A_wins']}")
    print(f"  Judge prefers original:   {stats['judge_B_wins']}")
    print(f"  Agree with cosine signal: {stats['agree_with_cosine']}")
    print(f"  Disagree:                 {stats['disagree_with_cosine']}")
    print(f"  Estimated API cost:       ~${stats['cost_approx_usd']}")
    print(f"\n  Saved to {output_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LLM-as-Judge preference annotation")
    parser.add_argument("--input", default="data/MS_MARCO/reward_annotated.jsonl")
    parser.add_argument("--output", default="data/MS_MARCO/dpo_dataset_judged.jsonl")
    parser.add_argument("--stats", default="data/MS_MARCO/judge_stats.json")
    parser.add_argument(
        "--judge_model",
        default=None,
        help=(
            "LLM model used only for judging. "
            "Overrides JUDGE_MODEL / CHATGPT_MODEL / AUDIT_MODEL."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit for cost control (default: all)")
    parser.add_argument("--min_cosine_gap", type=float, default=0.005)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume judged DPO generation from the existing output file if it exists.",
    )
    parser.add_argument(
        "--judge_workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads for LLM judging.",
    )
    args = parser.parse_args()

    build_judged_dpo_dataset(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        max_samples=args.max_samples,
        min_cosine_gap=args.min_cosine_gap,
        judge_model=args.judge_model,
        resume=args.resume,
        judge_workers=args.judge_workers,
    )


if __name__ == "__main__":
    main()
