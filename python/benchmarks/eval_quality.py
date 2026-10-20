"""Retrieval-quality evaluation on BEIR qrels.

Reports Recall@k and nDCG@10 for the reranked results plus a retrieval-overlap check between
the GPU search stage and an independent exact CPU search over the same stored embeddings.

Point ``--config`` at several engine configs to compare them side by side — that is how the
accuracy cost of int8 or binary storage, and the accuracy recovered by the on-device rescore
stage, are measured.

Import path must include the build output and the tools package, e.g.
``PYTHONPATH=build/bindings:python python eval_quality.py ...``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from zcgvs_tools import beir_loader, config, formats


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    gains = [float(relevant.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(v) for v in relevant.values()), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float | None:
    relevant_set = {doc_id for doc_id, score in relevant.items() if score > 0}
    if not relevant_set:
        return None
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def config_field(config_path: str, field: str) -> str:
    return json.loads(Path(config_path).read_text()).get(field, "")


def resolve(config_path: str, value: str) -> str:
    if not value:
        return value
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    root = Path(config_path).resolve().parent.parent
    return str(root / candidate)


def evaluate(config_path: str, args: argparse.Namespace, queries, qrels,
             cpu_matrix: np.ndarray, cpu_encoder) -> dict:
    engine = config.build_engine(config_path)
    stats = engine.stats()

    recalls: list[float] = []
    ndcgs: list[float] = []
    overlaps: list[float] = []
    gpu_ms: list[float] = []

    options = engine.default_options()
    options.top_k = args.top_k
    options.candidates = args.candidates
    options.hybrid = args.hybrid and engine.has_lexical_index

    for query_id, query_text in queries:
        relevant = qrels[query_id]

        response = engine.search_with(query_text, options)
        ranked_ids = [hit.doc_id for hit in response.hits]
        gpu_ms.append(response.timing.search_gpu_ms)

        recall = recall_at_k(ranked_ids, relevant, args.top_k)
        if recall is not None:
            recalls.append(recall)
        ndcgs.append(ndcg_at_k(ranked_ids, relevant, min(args.top_k, 10)))

        candidates = engine.retrieve(query_text, args.candidates)
        gpu_indices = {candidate.corpus_index for candidate in candidates}
        query_vector = cpu_encoder.encode(query_text, normalize_embeddings=True)
        cpu_scores = cpu_matrix @ query_vector.astype(np.float32)
        cpu_indices = set(np.argsort(-cpu_scores)[: args.candidates].tolist())
        overlaps.append(len(gpu_indices & cpu_indices) / max(len(cpu_indices), 1))

    return {
        "config": Path(config_path).name,
        "storage": stats.storage_dtype,
        "rescore": stats.rescore_dtype if stats.has_rescore_store else "-",
        "hybrid": "yes" if options.hybrid else "no",
        "vram_mb": (stats.corpus_bytes + stats.rescore_bytes + stats.lexical_bytes) / 1e6,
        "recall": float(np.mean(recalls)) if recalls else float("nan"),
        "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "gpu_ms": float(np.median(gpu_ms)) if gpu_ms else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, nargs="+", help="one or more engine configs")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--bi-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    query_ids, query_texts = beir_loader.load_queries(args.dataset)
    qrels = beir_loader.load_qrels(args.dataset)
    queries = [(qid, text) for qid, text in zip(query_ids, query_texts) if qid in qrels]
    queries = queries[: args.max_queries]

    # The CPU parity check always runs against the float32 corpus of the first config, which
    # is the ground truth the quantized variants are compared to.
    reference_config = args.config[0]
    embeddings_path = resolve(reference_config, config_field(reference_config, "embeddings_path"))
    rescore_path = resolve(
        reference_config, config_field(reference_config, "rescore_embeddings_path")
    )
    exact_path = rescore_path or embeddings_path

    header = formats.read_embedding_header(exact_path)
    if header["dtype"] == "binary":
        raise SystemExit(
            "the CPU parity check needs a float corpus; point the first config's "
            "rescore_embeddings_path at the float32 file"
        )
    cpu_matrix, _ = formats.read_embeddings(exact_path)
    cpu_matrix = np.ascontiguousarray(cpu_matrix, dtype=np.float32)
    cpu_encoder = SentenceTransformer(args.bi_encoder)

    results = [
        evaluate(path, args, queries, qrels, cpu_matrix, cpu_encoder) for path in args.config
    ]

    print(f"\nqueries evaluated: {len(queries)} from BEIR/{args.dataset}\n")
    header_row = (
        f"{'config':<28} {'storage':<9} {'rescore':<9} {'hybrid':<7} {'VRAM MB':>9} "
        f"{'Recall@' + str(args.top_k):>10} {'nDCG@10':>8} {'overlap':>8} {'gpu ms':>8}"
    )
    print(header_row)
    print("-" * len(header_row))
    for row in results:
        print(
            f"{row['config']:<28} {row['storage']:<9} {row['rescore']:<9} {row['hybrid']:<7} "
            f"{row['vram_mb']:>9.1f} {row['recall']:>10.4f} {row['ndcg']:>8.4f} "
            f"{row['overlap']:>8.4f} {row['gpu_ms']:>8.3f}"
        )


if __name__ == "__main__":
    main()
