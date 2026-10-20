"""Minimal usage example for the zcgvs engine.

Assumes the offline artifacts already exist (see ``export_models.py`` and ``embed_corpus.py``)
and that ``zcgvs_engine`` is importable from the build tree. Pass ``--config`` to load a JSON
engine config instead of individual paths, which is the only way to reach the rescore and
hybrid stages.
"""

from __future__ import annotations

import argparse

import zcgvs_engine


def build_from_flags(args: argparse.Namespace) -> "zcgvs_engine.Engine":
    config = zcgvs_engine.EngineConfig()
    config.embeddings_path = args.embeddings
    config.metadata_path = args.metadata
    config.vocab_path = args.vocab
    config.encoder_model_path = args.encoder
    config.reranker_model_path = args.reranker
    return zcgvs_engine.Engine(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="engine config JSON (overrides the paths)")
    parser.add_argument("--embeddings", default="data/scifact.bin")
    parser.add_argument("--metadata", default="data/scifact.meta")
    parser.add_argument("--vocab", default="models/vocab.txt")
    parser.add_argument("--encoder", default="models/encoder/model.onnx")
    parser.add_argument("--reranker", default="models/reranker/model.onnx")
    parser.add_argument("--query", default="what causes type 2 diabetes")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()

    if args.config:
        from zcgvs_tools.config import build_engine

        engine = build_engine(args.config)
    else:
        engine = build_from_flags(args)

    stats = engine.stats()
    print(f"corpus  : {stats.corpus_size} docs, dim {stats.dim}, storage {stats.storage_dtype} "
          f"({stats.corpus_bytes / 1e6:.1f} MB)")
    if stats.has_rescore_store:
        print(f"rescore : {stats.rescore_dtype} ({stats.rescore_bytes / 1e6:.1f} MB)")
    if stats.has_lexical_index:
        print(f"lexical : {stats.lexical_postings} postings")
    print(f"device  : {stats.device_name}")

    options = engine.default_options()
    options.top_k = args.top_k
    options.candidates = args.candidates
    options.hybrid = args.hybrid and engine.has_lexical_index

    response = engine.search_with(args.query, options)
    print(f"\nquery: {args.query}")
    for rank, hit in enumerate(response.hits, start=1):
        print(f"{rank:2d}. [{hit.doc_id}] rerank={hit.rerank_score:.3f} sim={hit.similarity:.3f}")
        print(f"    {hit.text[:120]}")

    timing = response.timing
    print(
        f"\ntiming (ms): embed={timing.embed_ms:.2f} gpu_search={timing.search_gpu_ms:.3f} "
        f"copy={timing.copy_ms:.3f} rerank={timing.rerank_ms:.2f} total={timing.total_ms:.2f}"
    )


if __name__ == "__main__":
    main()
