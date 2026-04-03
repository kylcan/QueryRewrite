"""Prompt templates for query rewriting."""

from __future__ import annotations


def format_sft_prompt(query: str) -> str:
    """Build a minimal-edit rewrite prompt.

    The model should preserve the user's intent and only rewrite when doing so
    materially clarifies the query for retrieval.
    """
    return (
        "Rewrite the search query only if it improves retrieval.\n"
        "Keep edits minimal and preserve the original meaning.\n"
        "If the query is already clear, copy it unchanged.\n"
        f"Query: {query}\n"
        "Rewrite:"
    )