"""Loads BEIR corpora, queries, and relevance judgements from the Hugging Face Hub.

BEIR datasets share a uniform schema, so the same loader serves SciFact, NFCorpus,
and the other small subsets referenced in the project brief.
"""

from __future__ import annotations

from datasets import load_dataset


def _join_title_and_body(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title:
        return f"{title}. {body}".strip()
    return body


def load_corpus(dataset: str, limit: int | None = None) -> tuple[list[str], list[str]]:
    rows = load_dataset(f"BeIR/{dataset}", "corpus")["corpus"]

    doc_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        doc_ids.append(str(row["_id"]))
        texts.append(_join_title_and_body(row.get("title", ""), row.get("text", "")))
        if limit is not None and len(doc_ids) >= limit:
            break
    return doc_ids, texts


def load_queries(dataset: str) -> tuple[list[str], list[str]]:
    rows = load_dataset(f"BeIR/{dataset}", "queries")["queries"]
    query_ids = [str(row["_id"]) for row in rows]
    query_texts = [str(row["text"]) for row in rows]
    return query_ids, query_texts


def load_qrels(dataset: str, split: str = "test") -> dict[str, dict[str, int]]:
    rows = load_dataset(f"BeIR/{dataset}-qrels")[split]

    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        qrels.setdefault(query_id, {})[corpus_id] = int(row["score"])
    return qrels
