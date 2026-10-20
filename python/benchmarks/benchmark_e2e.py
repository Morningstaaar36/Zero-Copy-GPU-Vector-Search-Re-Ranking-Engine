"""End-to-end latency and throughput comparison.

Times the C++/CUDA engine against a pure-Python baseline (sentence-transformers
``util.semantic_search`` for retrieval + a ``CrossEncoder`` for reranking) over the same
queries, corpus and models, then sweeps the engine's own knobs — batch size, CUDA graphs,
retrieval-only — so the numbers in docs/BENCHMARKS.md can be reproduced in one run.

Reports the median of many timed runs after warmup.

The build directory containing ``zcgvs_engine`` must be importable, e.g.
``PYTHONPATH=build/bindings:python python benchmark_e2e.py ...``.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

import zcgvs_engine
from zcgvs_tools import formats


def build_engine(args: argparse.Namespace) -> "zcgvs_engine.Engine":
    config = zcgvs_engine.EngineConfig()
    config.embeddings_path = args.embeddings
    config.metadata_path = args.metadata
    config.vocab_path = args.vocab
    config.encoder_model_path = args.encoder
    config.reranker_model_path = args.reranker
    config.rescore_embeddings_path = args.rescore_embeddings
    config.lexical_index_path = args.lexical
    config.rescore_candidates = args.rescore_candidates
    config.device_id = args.device_id
    return zcgvs_engine.Engine(config)


def median_ms(callable_under_test, repeats: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        callable_under_test()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_under_test()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def time_baseline(args: argparse.Namespace, queries: list[str], repeats: int) -> float:
    # The baseline does a dense matmul, so it needs float vectors. A binary corpus would
    # silently be reinterpreted as packed words and produce meaningless timings.
    header = formats.read_embedding_header(args.embeddings)
    if header["dtype"] == "binary":
        raise SystemExit(
            "the Python baseline needs a float corpus; point --embeddings at the float32 file "
            "or pass --skip-baseline"
        )

    matrix, _ = formats.read_embeddings(args.embeddings)
    _, texts = formats.read_metadata(args.metadata)

    device = f"cuda:{args.device_id}"
    corpus = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32)).to(device)
    bi_encoder = SentenceTransformer(args.bi_encoder, device=device)
    cross_encoder = CrossEncoder(args.cross_encoder, device=device)

    def run_once() -> None:
        for query in queries:
            query_vector = bi_encoder.encode(
                query, normalize_embeddings=True, convert_to_tensor=True
            )
            hits = util.semantic_search(query_vector, corpus, top_k=args.candidates)[0]
            pairs = [[query, texts[hit["corpus_id"]]] for hit in hits]
            cross_encoder.predict(pairs)
        torch.cuda.synchronize(device)

    return median_ms(run_once, repeats) / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--reranker", required=True)
    parser.add_argument("--rescore-embeddings", default="")
    parser.add_argument("--rescore-candidates", type=int, default=0)
    parser.add_argument("--lexical", default="")
    parser.add_argument("--bi-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--queries", nargs="+", default=[
        "how does crispr gene editing work",
        "what causes type 2 diabetes",
        "effects of vitamin d on immune response",
    ])
    args = parser.parse_args()

    engine = build_engine(args)
    stats = engine.stats()
    print(f"device   : {stats.device_name}")
    print(f"corpus   : {stats.corpus_size} docs, dim {stats.dim}, storage {stats.storage_dtype}"
          f" ({stats.corpus_bytes / 1e6:.1f} MB)")
    if stats.has_rescore_store:
        print(f"rescore  : {stats.rescore_dtype} ({stats.rescore_bytes / 1e6:.1f} MB), "
              f"width {args.rescore_candidates}")
    if stats.has_lexical_index:
        print(f"lexical  : {stats.lexical_postings} postings")
    print()

    base = engine.default_options()
    base.top_k = args.top_k
    base.candidates = args.candidates
    base.hybrid = False

    rows: list[tuple[str, float, str]] = []

    for batch in args.batch_sizes:
        queries = [args.queries[i % len(args.queries)] for i in range(batch)]
        for graphs in (False, True):
            options = engine.default_options()
            options.top_k = args.top_k
            options.candidates = args.candidates
            options.use_config_graphs = False
            options.use_graph = graphs
            elapsed = median_ms(lambda: engine.search_batch(queries, options), args.repeats)
            label = f"engine rerank batch={batch} graphs={'on' if graphs else 'off'}"
            rows.append((label, elapsed / batch, f"{batch / (elapsed / 1000.0):8.1f} q/s"))

    retrieval_only = engine.default_options()
    retrieval_only.top_k = args.top_k
    retrieval_only.candidates = args.candidates
    retrieval_only.rerank = False
    elapsed = median_ms(
        lambda: engine.search_batch(args.queries, retrieval_only), args.repeats
    )
    rows.append(("engine retrieval-only", elapsed / len(args.queries),
                 f"{len(args.queries) / (elapsed / 1000.0):8.1f} q/s"))

    if engine.has_lexical_index:
        hybrid = engine.default_options()
        hybrid.top_k = args.top_k
        hybrid.candidates = args.candidates
        hybrid.hybrid = True
        elapsed = median_ms(lambda: engine.search_batch(args.queries, hybrid), args.repeats)
        rows.append(("engine hybrid (BM25 + dense)", elapsed / len(args.queries),
                     f"{len(args.queries) / (elapsed / 1000.0):8.1f} q/s"))

    if not args.skip_baseline:
        baseline = time_baseline(args, args.queries, args.repeats)
        rows.append(("python baseline (ST + CrossEncoder)", baseline, ""))

    width = max(len(label) for label, _, _ in rows)
    print(f"{'configuration'.ljust(width)}   ms/query   throughput")
    for label, ms, throughput in rows:
        print(f"{label.ljust(width)}   {ms:8.3f}   {throughput}")

    engine_row = next((ms for label, ms, _ in rows if label.startswith("engine rerank batch=1")),
                      None)
    baseline_row = next((ms for label, ms, _ in rows if label.startswith("python baseline")), None)
    if engine_row and baseline_row and engine_row > 0:
        print(f"\nspeedup at batch=1 vs the Python baseline: {baseline_row / engine_row:.2f}x")


if __name__ == "__main__":
    main()
