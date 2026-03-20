

"""
MS MARCO → Intent-Aware Retrieval Dataset Builder
==================================================

Improvements over v1:
- **Large distractor corpus**: all candidate passages (not just pos_docs)
  are indexed, simulating realistic retrieval difficulty.
- **Hard-negative mining with score-band filtering**: negatives are
  selected from a configurable similarity window so they are *confusing
  but incorrect*, not trivially unrelated.
- **LLM query rewrite** with real API path (OpenAI-compatible) +
  robust rule-based fallback.
- **Analytic fields**: per-sample cosine similarities are persisted for
  downstream gap analysis.

Output JSONL schema (one object per line)::

    {
        "query_id":          int,
        "query_original":    str,
        "query_rewrite":     str,
        "pos_doc":           str,
        "hard_neg_doc":      str,
        "cos_q_pos":         float,   # cos(query, pos_doc)
        "cos_q_neg":         float,   # cos(query, hard_neg)
        "cos_q_rewrite":     float,   # cos(query, rewrite)  — gap proxy
        "neg_rank":          int,     # rank of hard_neg among all corpus docs
        "gap_type":          str      # "hard" | "medium" | "easy"
    }
"""

from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ================================================================
# 1. Load MS MARCO  — query + ALL candidate passages
# ================================================================

def load_msmarco_subset(
    n_samples: int = 500,
    n_distractor_pool: int = 5000,
    dataset_version: str = "v1.1",
) -> Tuple[List[Dict], List[str]]:
    """Load queries with positive docs AND build a large distractor corpus.

    Parameters
    ----------
    n_samples : int
        Number of (query, pos_doc) pairs to collect.
    n_distractor_pool : int
        How many extra MS MARCO rows to scan for distractor passages.
        All passages (positive or not) from these rows enter the corpus.
    dataset_version : str
        MS MARCO HuggingFace config name.

    Returns
    -------
    data : list[dict]
        Each dict has keys ``query``, ``pos_doc``, ``all_passages``.
    corpus : list[str]
        De-duplicated global passage pool (for FAISS indexing).
    """
    dataset = load_dataset("ms_marco", dataset_version, split="train")

    data: List[Dict] = []
    corpus_set: set[str] = set()
    corpus_list: List[str] = []

    scan_limit = max(n_samples * 3, n_distractor_pool)
    scanned = 0

    for item in dataset:  # type: ignore[union-attr]
        if scanned >= scan_limit:
            break
        scanned += 1

        passages: List[str] = item["passages"]["passage_text"]  # type: ignore[index]
        labels: List[int] = item["passages"]["is_selected"]  # type: ignore[index]

        # Add ALL passages to the global corpus
        for p in passages:
            if p not in corpus_set:
                corpus_set.add(p)
                corpus_list.append(p)

        # Collect labelled (query, pos_doc) pairs up to n_samples
        if len(data) < n_samples:
            pos_docs = [p for p, l in zip(passages, labels) if l == 1]
            if pos_docs:
                data.append({
                    "query": item["query"],  # type: ignore[index]
                    "pos_doc": pos_docs[0],
                    "all_passages": passages,
                })

    print(f"  Loaded {len(data)} query–pos pairs, corpus size = {len(corpus_list)}")
    return data, corpus_list


# ================================================================
# 2. Query Rewrite — rule-based fallback + LLM path
# ================================================================

# --- Synonym / vocabulary expansion tables ---
_SYNONYMS: Dict[str, List[str]] = {
    "cost": ["price", "expense", "fee"],
    "salary": ["pay", "compensation", "wage"],
    "best": ["top", "greatest", "most popular"],
    "cheap": ["affordable", "low-cost", "budget"],
    "big": ["large", "substantial", "significant"],
    "small": ["little", "minor", "compact"],
    "fast": ["rapid", "quick", "speedy"],
    "old": ["ancient", "historical", "aged"],
    "new": ["recent", "modern", "latest"],
    "important": ["significant", "crucial", "essential"],
    "dangerous": ["hazardous", "risky", "harmful"],
    "part": ["component", "element", "section"],
    "type": ["kind", "category", "variety"],
    "difference": ["distinction", "contrast", "variation"],
}


def _classify_intent(query: str) -> str:
    """Classify a query into one of six intent categories."""
    ql = query.lower().strip().rstrip("?").rstrip(".")

    # Definition
    if re.match(r"^(what is|what are|what does|what do|define|meaning of)\b", ql):
        return "definition"
    # Procedural / how-to
    if re.match(r"^(how to|how do|how can|how does|how should|steps to)\b", ql):
        return "procedural"
    # Factual (who/where/when/which/how many/how much/how long/how far)
    if re.match(r"^(where|when|who|which|how many|how much|how long|how far|how old)\b", ql):
        return "factual"
    # Yes/no
    if re.match(r"^(is|are|do|does|did|can|could|will|would|was|were|has|have|should)\b", ql):
        return "yesno"
    # Comparison
    if re.search(r"\b(vs\.?|versus|compared? to|difference between|similarities)\b", ql):
        return "comparison"

    return "general"


def _expand_vocabulary(query: str) -> str:
    """Add one synonym for the first matching keyword."""
    ql = query.lower()
    for word, syns in _SYNONYMS.items():
        if re.search(rf"\b{word}\b", ql):
            syn = random.choice(syns)
            return f"{query} ({syn})"
    return query


def rule_based_rewrite(query: str) -> str:
    """Intent-aware query rewrite with vocabulary expansion.

    1. Classify query intent (definition / procedural / factual / yesno /
       comparison / general).
    2. Apply an intent-specific structural transform that converts the
       question form into a declarative/noun-phrase form closer to how
       answer passages are written.
    3. Optionally expand vocabulary with a domain synonym.
    """
    q = query.strip().rstrip("?").rstrip(".")
    intent = _classify_intent(query)

    if intent == "definition":
        # "what is X" → "X definition meaning"
        core = re.sub(
            r"^(what is|what are|what does|what do|define|meaning of)\s+",
            "", q, flags=re.IGNORECASE,
        ).strip()
        q = f"{core} definition meaning"

    elif intent == "procedural":
        # "how to X" → "steps method procedure X"
        core = re.sub(
            r"^(how to|how do you|how do i|how can i|how can you|how does one|"
            r"how do|how can|how does|how should i|how should|steps to)\s+",
            "", q, flags=re.IGNORECASE,
        ).strip()
        q = f"{core} steps method procedure"

    elif intent == "factual":
        # "where is X" → "X location"  /  "when was X" → "X date year"
        if re.match(r"^where\b", q, re.IGNORECASE):
            core = re.sub(r"^where\s+(is|are|was|were|do|does)?\s*",
                          "", q, flags=re.IGNORECASE).strip()
            q = f"{core} location geography"
        elif re.match(r"^when\b", q, re.IGNORECASE):
            core = re.sub(r"^when\s+(is|are|was|were|did|do|does)?\s*",
                          "", q, flags=re.IGNORECASE).strip()
            q = f"{core} date year time"
        elif re.match(r"^who\b", q, re.IGNORECASE):
            core = re.sub(r"^who\s+(is|are|was|were|did)?\s*",
                          "", q, flags=re.IGNORECASE).strip()
            q = f"{core} person identity"
        elif re.match(r"^which\b", q, re.IGNORECASE):
            core = re.sub(r"^which\s+", "", q, flags=re.IGNORECASE).strip()
            q = f"{core} specific type"
        else:
            # how many / how much / how long / how far / how old
            core = re.sub(
                r"^(how many|how much|how long|how far|how old)\s+"
                r"(is|are|was|were|do|does|did)?\s*",
                "", q, flags=re.IGNORECASE,
            ).strip()
            measure = re.match(r"^(how many|how much|how long|how far|how old)",
                               q, re.IGNORECASE)
            suffix = {
                "how many": "number count quantity",
                "how much": "amount cost price",
                "how long": "duration length time",
                "how far": "distance range",
                "how old": "age years",
            }.get(measure.group(1).lower() if measure else "", "amount")
            q = f"{core} {suffix}"

    elif intent == "yesno":
        # "is X Y" → "X Y explanation"
        core = re.sub(
            r"^(is|are|do|does|did|can|could|will|would|was|were|"
            r"has|have|should)\s+",
            "", q, flags=re.IGNORECASE,
        ).strip()
        q = f"{core} explanation"

    elif intent == "comparison":
        # "X vs Y" → "comparison X and Y differences similarities"
        core = re.sub(
            r"\b(vs\.?|versus|compared? to)\b", "and",
            q, flags=re.IGNORECASE,
        ).strip()
        q = f"comparison {core} differences similarities"

    else:
        # General: extract as-is, append "information"
        q = f"{q} information"

    # Vocabulary expansion (50% chance to add a synonym)
    if random.random() < 0.5:
        q = _expand_vocabulary(q)

    return q


_LLM_SYSTEM_PROMPT = """You are a search query optimizer. Given a user query, rewrite it to better match the vocabulary of encyclopedia-style answer passages.

Rules:
- Convert questions to declarative/noun-phrase form
- Replace colloquial terms with formal equivalents
- Add domain-specific synonyms that documents would use
- Do NOT add generic filler words like "guide", "tips", "overview", "advice"
- Keep the rewrite concise (under 15 words)
- Return ONLY the rewritten query, nothing else"""

_LLM_FEW_SHOT = [
    ("what is a conifer", "conifer definition evergreen cone-bearing tree"),
    ("how long is a rugby match", "rugby match duration time length minutes"),
    ("best tragedies of ancient greece", "greatest ancient Greek tragedies Sophocles Euripides"),
    ("is mirin halal", "mirin halal status Islamic permissibility rice wine"),
    ("where are precambrian rocks found", "precambrian rock locations geological distribution"),
    ("what does mrna do", "mRNA function role protein synthesis translation"),
    ("cost to treat termites", "termite treatment cost price extermination expense"),
    ("salary for pvt in us army", "US army private salary pay compensation rank E-1"),
]


def llm_rewrite(
    query: str,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> str:
    """Rewrite via LLM if available, else fall back to rules.

    Uses an intent-aware few-shot prompt designed to produce
    declarative, vocabulary-expanded rewrites that align with
    encyclopedia-style answer passages.

    Parameters
    ----------
    query : str
        Original user query.
    llm_callable : callable, optional
        A function ``(prompt: str) -> str`` returning the LLM response.

    Returns
    -------
    str
        Rewritten query.
    """
    if llm_callable is None:
        return rule_based_rewrite(query)

    examples = "\n".join(
        f"  Query: {q}\n  Rewrite: {r}" for q, r in _LLM_FEW_SHOT
    )
    prompt = (
        f"{_LLM_SYSTEM_PROMPT}\n\n"
        f"Examples:\n{examples}\n\n"
        f"Query: {query}\nRewrite:"
    )
    try:
        result = llm_callable(prompt).strip()
        # Reject if LLM returns something too long or empty
        if result and len(result.split()) <= 25:
            return result
        return rule_based_rewrite(query)
    except Exception:
        return rule_based_rewrite(query)


def _load_cached_rewrites(cache_path: Path) -> Dict[int, str]:
    """Load existing rewrite cache keyed by query_id."""
    if not cache_path.exists():
        return {}

    cached: Dict[int, str] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            query_id = record.get("query_id")
            rewrite = record.get("query_rewrite")
            if isinstance(query_id, int) and isinstance(rewrite, str) and rewrite:
                cached[query_id] = rewrite
    return cached


def _save_corpus(corpus: List[str], corpus_output: str) -> None:
    Path(corpus_output).parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_output, "w", encoding="utf-8") as f:
        for idx, text in enumerate(corpus):
            f.write(json.dumps({"id": str(idx), "text": text}, ensure_ascii=False) + "\n")


def _generate_rewrites(
    queries: List[str],
    llm_callable: Optional[Callable[[str], str]],
    cache_path: Path,
    resume: bool = False,
    rewrite_workers: int = 1,
) -> List[str]:
    """Generate rewrites with incremental cache and optional concurrency."""
    cached = _load_cached_rewrites(cache_path) if resume else {}
    rewrites: List[str] = [""] * len(queries)

    for idx, rewrite in cached.items():
        if 0 <= idx < len(queries):
            rewrites[idx] = rewrite

    completed = sum(1 for rewrite in rewrites if rewrite)
    if completed:
        print(f"  Resume enabled: loaded {completed} cached rewrites from {cache_path}")

    pending_indices = [idx for idx, rewrite in enumerate(rewrites) if not rewrite]
    if not pending_indices:
        print("  All rewrites already cached; skipping generation.")
        return rewrites

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume and cache_path.exists() else "w"

    if llm_callable is None or rewrite_workers <= 1:
        with open(cache_path, write_mode, encoding="utf-8") as cache_file:
            progress = tqdm(pending_indices, desc="Rewriting", initial=completed, total=len(queries))
            for idx in progress:
                rewrite = llm_rewrite(queries[idx], llm_callable)
                rewrites[idx] = rewrite
                cache_file.write(
                    json.dumps(
                        {"query_id": idx, "query": queries[idx], "query_rewrite": rewrite},
                        ensure_ascii=False,
                    ) + "\n"
                )
                cache_file.flush()
        return rewrites

    def _rewrite_job(item_idx: int) -> Tuple[int, str]:
        return item_idx, llm_rewrite(queries[item_idx], llm_callable)

    max_workers = min(max(rewrite_workers, 1), len(pending_indices))
    print(f"  Rewriting with {max_workers} worker threads …")

    with open(cache_path, write_mode, encoding="utf-8") as cache_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx: Dict[Future[Tuple[int, str]], int] = {}

            initial_batch = pending_indices[:max_workers]
            remaining = pending_indices[max_workers:]
            for idx in initial_batch:
                future = executor.submit(_rewrite_job, idx)
                future_to_idx[future] = idx

            with tqdm(total=len(queries), initial=completed, desc="Rewriting") as progress:
                while future_to_idx:
                    done, _ = wait(list(future_to_idx.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        idx = future_to_idx.pop(future)
                        _, rewrite = future.result()
                        rewrites[idx] = rewrite
                        cache_file.write(
                            json.dumps(
                                {"query_id": idx, "query": queries[idx], "query_rewrite": rewrite},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        cache_file.flush()
                        progress.update(1)

                        if remaining:
                            next_idx = remaining.pop(0)
                            next_future = executor.submit(_rewrite_job, next_idx)
                            future_to_idx[next_future] = next_idx

    return rewrites


# ================================================================
# 3. Hard Negative Mining (score-band filtering)
# ================================================================

class HardNegativeMiner:
    """Mine negatives at three difficulty tiers using rank-based selection.

    Instead of a single score band (which tends to collapse to one tier),
    we use the **rank** of non-positive passages as the difficulty proxy:

    - **hard**   : rank 1–5 among non-positive passages (very confusing)
    - **medium** : rank 6–30 (topically related but distinguishable)
    - **easy**   : rank 100+ or random (clearly irrelevant)

    All similarity computations are batched for performance.

    Parameters
    ----------
    corpus : list[str]
        All candidate passages.
    model_name : str
        sentence-transformers model for encoding.
    batch_size : int
        Encoding batch size.
    """

    # Rank ranges (0-indexed, among non-positive passages)
    TIER_RANGES: Dict[str, Tuple[int, int]] = {
        "hard":   (0, 5),     # top-5 non-positive
        "medium": (5, 30),    # rank 6–30
        "easy":   (80, 200),  # rank 80+
    }

    def __init__(
        self,
        corpus: List[str],
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
    ) -> None:
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)

        print(f"  Encoding corpus ({len(corpus)} passages) …")
        self.corpus_emb_np: np.ndarray = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def batch_mine(
        self,
        queries: List[str],
        pos_docs: List[str],
        target_tiers: List[str],
        batch_size: int = 128,
    ) -> List[Tuple[str, float, int, str]]:
        """Mine negatives for all queries in one vectorised pass.

        Parameters
        ----------
        queries : list[str]
            Original query strings.
        pos_docs : list[str]
            Corresponding positive documents.
        target_tiers : list[str]
            Desired difficulty tier per query.
        batch_size : int
            Encoding batch size.

        Returns
        -------
        list[tuple]
            For each query: ``(neg_doc, cos_score, rank, tier_label)``.
        """
        # Build a fast pos_doc → corpus-index lookup
        doc_to_idx: Dict[str, int] = {}
        for ci, doc in enumerate(self.corpus):
            if doc not in doc_to_idx:
                doc_to_idx[doc] = ci

        print("  Encoding queries for mining …")
        query_emb_np: np.ndarray = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # (n_queries, corpus_size) similarity matrix via matmul
        print("  Computing similarity matrix …")
        sim_matrix = query_emb_np @ self.corpus_emb_np.T  # cosine (both normalised)

        results: List[Tuple[str, float, int, str]] = []

        for i in tqdm(range(len(queries)), desc="Mining negatives"):
            scores = sim_matrix[i]  # (corpus_size,)
            sorted_indices = np.argsort(-scores)  # descending

            # Mask out the positive doc
            pos_idx = doc_to_idx.get(pos_docs[i], -1)

            # Build ranked non-positive list (index, score, global_rank)
            # Only need up to rank ~200 for any tier
            non_pos: List[Tuple[int, float, int]] = []
            for global_rank_pos, ci in enumerate(sorted_indices[:300]):
                ci_int = int(ci)
                if ci_int == pos_idx:
                    continue
                non_pos.append((ci_int, float(scores[ci_int]), global_rank_pos))
                if len(non_pos) >= 200:
                    break

            target_tier = target_tiers[i]
            chosen = self._pick_from_tier(non_pos, target_tier)
            results.append(chosen)

        return results

    def _pick_from_tier(
        self,
        non_pos: List[Tuple[int, float, int]],
        target_tier: str,
    ) -> Tuple[str, float, int, str]:
        """Select a negative from the target tier, with fallback cascade."""
        lo, hi = self.TIER_RANGES.get(target_tier, (0, 5))
        hi = min(hi, len(non_pos))
        candidates = non_pos[lo:hi]

        if candidates:
            chosen_idx, chosen_score, chosen_rank = random.choice(candidates)
            return self.corpus[chosen_idx], chosen_score, chosen_rank, target_tier

        # Fallback cascade: hard → medium → easy → random
        for fallback_tier in ("hard", "medium", "easy"):
            flo, fhi = self.TIER_RANGES[fallback_tier]
            fhi = min(fhi, len(non_pos))
            fc = non_pos[flo:fhi]
            if fc:
                chosen_idx, chosen_score, chosen_rank = random.choice(fc)
                return self.corpus[chosen_idx], chosen_score, chosen_rank, fallback_tier

        chosen = random.choice(self.corpus)
        return chosen, 0.0, -1, "easy"


# ================================================================
# 4. Build Dataset
# ================================================================

def build_dataset(
    n_samples: int = 500,
    n_distractor_pool: int = 5000,
    output_path: str = "data/MS_MARCO/intent_dataset.jsonl",
    corpus_output_path: Optional[str] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
    neg_ratio: Optional[Dict[str, float]] = None,
    resume: bool = False,
    rewrite_workers: int = 1,
) -> None:
    """End-to-end dataset construction pipeline.

    Parameters
    ----------
    n_samples : int
        Number of query–pos_doc pairs to collect.
    n_distractor_pool : int
        Rows scanned for distractor passages (controls corpus size).
    output_path : str
        Where to write the JSONL output.
    corpus_output_path : str, optional
        Where to write the de-duplicated corpus JSONL with ``{id, text}`` rows.
        If omitted, defaults to ``<output_dir>/corpus.jsonl``.
    llm_callable : callable, optional
        LLM function for query rewriting.
    neg_ratio : dict, optional
        Target distribution of negative difficulty tiers.
        Default: ``{"hard": 0.4, "medium": 0.4, "easy": 0.2}``.
    """
    if neg_ratio is None:
        neg_ratio = {"hard": 0.4, "medium": 0.4, "easy": 0.2}

    print("[1/5] Loading MS MARCO …")
    data, corpus = load_msmarco_subset(n_samples, n_distractor_pool)
    corpus_output = corpus_output_path or str(Path(output_path).with_name("corpus.jsonl"))
    rewrite_cache_path = Path(output_path).with_name(Path(output_path).stem + "_rewrites.jsonl")
    doc_to_idx = {doc: idx for idx, doc in enumerate(corpus)}

    _save_corpus(corpus, corpus_output)
    print(f"      Saved {len(corpus)} corpus passages → {corpus_output}")

    print("[2/5] Building negative miner …")
    miner = HardNegativeMiner(corpus)

    # Pre-assign a difficulty tier to each sample based on target ratio
    tier_sequence: List[str] = []
    for tier, ratio in neg_ratio.items():
        tier_sequence.extend([tier] * int(round(ratio * len(data))))
    while len(tier_sequence) < len(data):
        tier_sequence.append(random.choice(list(neg_ratio.keys())))
    tier_sequence = tier_sequence[: len(data)]
    random.shuffle(tier_sequence)

    # Generate rewrites
    print("[3/5] Generating query rewrites …")
    queries = [d["query"] for d in data]
    pos_docs = [d["pos_doc"] for d in data]
    rewrites = _generate_rewrites(
        queries,
        llm_callable,
        rewrite_cache_path,
        resume=resume,
        rewrite_workers=rewrite_workers,
    )

    # Batch mine negatives
    print("[4/5] Mining negatives (batched) …")
    print(f"       Target ratio: {neg_ratio}")
    neg_results = miner.batch_mine(queries, pos_docs, tier_sequence)

    # Batch encode for analytic similarities
    print("  Computing analytic similarities …")
    all_texts = queries + pos_docs + rewrites
    all_embs = miner.model.encode(
        all_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    n = len(queries)
    q_embs = all_embs[:n]
    p_embs = all_embs[n : 2 * n]
    r_embs = all_embs[2 * n : 3 * n]

    # Per-sample cosines via element-wise dot (both normalised)
    cos_q_pos_arr = np.sum(q_embs * p_embs, axis=1)
    cos_q_rew_arr = np.sum(q_embs * r_embs, axis=1)

    # Assemble results
    results: List[Dict[str, Any]] = []
    gap_counts: Dict[str, int] = {"hard": 0, "medium": 0, "easy": 0}

    for i in range(n):
        hard_neg, cos_q_neg, neg_rank, gap_type = neg_results[i]
        gap_counts[gap_type] += 1
        pos_id = str(doc_to_idx[pos_docs[i]])
        hard_neg_id = str(doc_to_idx.get(hard_neg, -1))

        results.append({
            "query_id": i,
            "query": queries[i],
            "original_query": queries[i],
            "query_original": queries[i],
            "query_rewrite": rewrites[i],
            "positive_passage_id": pos_id,
            "pos_id": pos_id,
            "pos_doc": pos_docs[i],
            "hard_negative_passage_id": hard_neg_id,
            "hard_neg_doc": hard_neg,
            "cos_q_pos": round(float(cos_q_pos_arr[i]), 4),
            "cos_q_neg": round(cos_q_neg, 4),
            "cos_q_rewrite": round(float(cos_q_rew_arr[i]), 4),
            "neg_rank": neg_rank,
            "gap_type": gap_type,
        })

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[5/5] Saved {len(results)} samples → {output_path}")
    print(f"      Rewrite cache saved → {rewrite_cache_path}")
    print(f"      Corpus passages saved → {corpus_output}")
    print(f"  Hard: {gap_counts['hard']}  Medium: {gap_counts['medium']}  Easy: {gap_counts['easy']}")

    cos_neg_arr = [r["cos_q_neg"] for r in results]
    print(f"  Avg cos(Q, P+) = {np.mean(cos_q_pos_arr):.4f}")
    print(f"  Avg cos(Q, N−) = {np.mean(cos_neg_arr):.4f}")
    print(f"  Avg gap        = {np.mean(cos_q_pos_arr) - np.mean(cos_neg_arr):+.4f}")

    # Per-tier stats
    for tier in ("hard", "medium", "easy"):
        tier_negs = [r["cos_q_neg"] for r in results if r["gap_type"] == tier]
        if tier_negs:
            print(f"  {tier:8s} neg: avg cos={np.mean(tier_negs):.4f}  "
                  f"min={np.min(tier_negs):.4f}  max={np.max(tier_negs):.4f}  n={len(tier_negs)}")


# ================================================================
# 5. OpenAI-compatible LLM client
# ================================================================

def make_openai_callable(model_name: Optional[str] = None) -> Optional[Callable[[str], str]]:
    """Build a callable ``(prompt) -> response_text`` from env vars.

    Required env vars (at least one key must be set)::

        GPT5_KEY / OPENAI_API_KEY
        QUERY_REWRITE_MODEL / CHATGPT_MODEL / AUDIT_MODEL   (default: gpt-4o)
        CHATGPT_BASE_URL / OPENAI_BASE_URL / OPENAI_API_BASE

    Returns ``None`` if no API key is found.
    """
    api_key = os.getenv("GPT5_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = (
        model_name
        or os.getenv("QUERY_REWRITE_MODEL")
        or os.getenv("CHATGPT_MODEL")
        or os.getenv("AUDIT_MODEL")
        or "gpt-4o"
    )
    base_url = (
        os.getenv("CHATGPT_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )

    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError:
        print("  [warn] openai package not installed — falling back to rules")
        return None

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)  # type: ignore[arg-type]
    print(f"  LLM rewrite enabled: model={model}")
    if base_url:
        print(f"  base_url={base_url}")

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=60,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    return _call


# ================================================================
# 6. Main
# ================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build MS MARCO intent dataset")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_distractor_pool", type=int, default=5000)
    parser.add_argument("--output", default="data/MS_MARCO/intent_dataset.jsonl")
    parser.add_argument(
        "--corpus_output",
        default=None,
        help="Optional corpus JSONL output path. Defaults to <output_dir>/corpus.jsonl.",
    )
    parser.add_argument(
        "--rewrite_model",
        default=None,
        help=(
            "LLM model used only for query rewriting. "
            "Overrides QUERY_REWRITE_MODEL / CHATGPT_MODEL / AUDIT_MODEL."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume rewrite generation from the cache file if it exists.",
    )
    parser.add_argument(
        "--rewrite_workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads for API-based query rewriting.",
    )
    args = parser.parse_args()

    llm_fn = make_openai_callable(model_name=args.rewrite_model)
    if llm_fn is None:
        print("  [info] No LLM API key found — using rule-based rewrite only")

    build_dataset(
        n_samples=args.n_samples,
        n_distractor_pool=args.n_distractor_pool,
        output_path=args.output,
        corpus_output_path=args.corpus_output,
        llm_callable=llm_fn,
        neg_ratio={"hard": 0.4, "medium": 0.4, "easy": 0.2},
        resume=args.resume,
        rewrite_workers=args.rewrite_workers,
    )


if __name__ == "__main__":
    main()