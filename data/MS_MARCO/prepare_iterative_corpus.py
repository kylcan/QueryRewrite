"""Prepare corpus/id files for iterative DPO from an existing intent dataset.

This utility backfills the fields expected by scripts/train_iterative_dpo.py:
  - positive_passage_id / pos_id
  - hard_negative_passage_id
  - corpus.jsonl with {id, text}

It is useful when an intent dataset was created before dataset_build.py started
emitting corpus/id information.

Usage::

    python data/MS_MARCO/prepare_iterative_corpus.py \
        --input data/MS_MARCO/15k/intent_dataset.jsonl \
        --output_dataset data/MS_MARCO/15k/intent_dataset_iterative.jsonl \
        --corpus_output data/MS_MARCO/15k/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare corpus/id files for iterative DPO")
    parser.add_argument("--input", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument(
        "--output_dataset",
        default="data/MS_MARCO/intent_dataset_iterative.jsonl",
        help="Augmented intent dataset with positive_passage_id / pos_id fields",
    )
    parser.add_argument(
        "--corpus_output",
        default="data/MS_MARCO/corpus.jsonl",
        help="Deduplicated corpus file with {id, text} rows",
    )
    args = parser.parse_args()

    records = _load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    corpus: List[Dict[str, str]] = []
    doc_to_id: Dict[str, str] = {}

    def ensure_doc_id(text: str) -> str:
        doc_id = doc_to_id.get(text)
        if doc_id is not None:
            return doc_id
        doc_id = str(len(corpus))
        doc_to_id[text] = doc_id
        corpus.append({"id": doc_id, "text": text})
        return doc_id

    augmented: List[Dict[str, Any]] = []
    for record in records:
        pos_doc = record["pos_doc"]
        pos_id = ensure_doc_id(pos_doc)

        updated = dict(record)
        updated["positive_passage_id"] = pos_id
        updated["pos_id"] = pos_id

        hard_neg = record.get("hard_neg_doc")
        if isinstance(hard_neg, str) and hard_neg:
            updated["hard_negative_passage_id"] = ensure_doc_id(hard_neg)

        # Add compatibility aliases for downstream scripts.
        if "query" not in updated and "query_original" in updated:
            updated["query"] = updated["query_original"]
        if "original_query" not in updated and "query_original" in updated:
            updated["original_query"] = updated["query_original"]

        augmented.append(updated)

    _save_jsonl(augmented, args.output_dataset)
    _save_jsonl(corpus, args.corpus_output)

    print(f"Saved augmented dataset → {args.output_dataset}")
    print(f"Saved corpus file → {args.corpus_output}")
    print(f"Corpus size: {len(corpus)}")


if __name__ == "__main__":
    main()
