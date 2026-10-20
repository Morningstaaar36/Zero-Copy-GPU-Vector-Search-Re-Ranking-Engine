# Features

Every capability here is built on the same hand-written kernels — there is no third-party
index anywhere in the stack, and nothing is trained. The Llama reasoning layer on top is
documented separately in [LLAMA.md](LLAMA.md). Each feature is a composable addition
that can be turned on from a config file or a per-request option.

| Feature | Where | One-liner |
| --- | --- | --- |
| [Storage precision ladder](#storage-precision-ladder) | `ScalarType`, `similarity.cu` | fp32 / fp16 / int8 / binary — up to 32× less VRAM and bandwidth |
| [On-device rescoring](#on-device-rescoring) | `rescore.cu`, `Searcher` | Coarse candidates re-scored exactly, without a host round trip |
| [Batched multi-query scan](#batched-multi-query-scan) | `similarity.cu` | One corpus read serves 8 queries |
| [Recursive top-K merge](#recursive-top-k-merge) | `topk.cu` | No corpus-size cap; `k` up to 1024 |
| [CUDA graph replay](#cuda-graph-replay) | `Searcher` | The device pipeline replays from one instantiated graph |
| [Hybrid BM25 + dense](#hybrid-bm25--dense-retrieval) | `lexical.cu` | GPU scatter-add over a CSR inverted index, fused on-device |
| [cuBLAS reference backend](#cublas-reference-backend) | `cublas_similarity.cpp` | Benchmark the hand-written kernel against the vendor GEMM |
| [Metrics](#similarity-metrics) | `Metric`, `similarity.cu` | Cosine / dot / L2, selectable per request |
| [Filtered search](#filtered-search) | device mask, `Engine::buildMask` | Restrict the scan to a subset of documents |
| [Incremental indexing](#incremental-indexing) | `EmbeddingStore::append` | Append to the live VRAM index, quantizing on the device |
| [Stage timing](#stage-timing-and-profiling) | CUDA events, `SearchTiming` | Per-stage latency through the API |
| [NVTX ranges](#stage-timing-and-profiling) | `core/nvtx.hpp` | Nsight Systems annotations behind a build flag |
| [Retrieval-only](#retrieval-only-queries) | `Engine::retrieve` | The search stage without the reranker |
| [HTTP service](#http-service) | `zcgvs_tools.service` | FastAPI: search, batch, retrieve, documents, stats, verify, ask |
| [Engine statistics](#engine-statistics) | `Engine::stats` | What is resident in VRAM and which stages are wired up |
| [Kernel roofline bench](#kernel-roofline-benchmark) | `apps/bench_kernels.cu` | GB/s vs peak on synthetic data, no artifacts needed |
| [Llama claim verification](LLAMA.md#job-1--the-verdict-is-a-classifier-not-a-parser) | `factcheck.ClaimVerifier` | Calibrated SUPPORT / CONTRADICT / NEI in one forward pass, plus a streamed cited explanation |
| [Llama reranking](LLAMA.md#job-3--llama-as-a-zero-shot-reranker) | `factcheck.LlamaReranker` | Zero-shot query likelihood (UPR) or yes/no, fusable with the cross-encoder |
| [Grounded Q&A](LLAMA.md#job-4--grounded-question-answering) | `factcheck.GroundedAnswerer` | Answers only from retrieved abstracts, with citations |

---

## Storage precision ladder

Brute-force GPU search is **memory-bandwidth bound**: every query streams the entire corpus
matrix through the SMs. The single most effective optimization is therefore to make the
matrix smaller. The engine supports four storage forms, chosen when the artifact is written
and dispatched automatically at load time from the file header.

| dtype | Bytes/dim | vs fp32 | How the kernel scores it |
| --- | --- | --- | --- |
| `float32` | 4 | 1× | Direct load |
| `float16` | 2 | 2× smaller | Load and widen in registers |
| `int8` | 1 | 4× smaller | Dequantize with a matrix-wide scale; query stays fp32 (asymmetric) |
| `binary` | 0.125 | **32× smaller** | `__popc(doc ^ query)` over packed sign bits |

`int8` uses symmetric quantization with one scale for the whole matrix, `scale = max|x| / 127`,
recorded in the file header. Because the query is never quantized, the only precision loss is
on the corpus side — the standard asymmetric arrangement, and materially more accurate than
quantizing both sides.

`binary` packs one sign bit per dimension into `ceil(dim/32)` uint32 words. Bit `b` of word
`w` carries dimension `w*32 + b` and is set when the value is non-negative; dimensions past
`dim` stay zero so they cancel in any XOR. Scoring is
`score = (dim - 2 * hamming) / dim ∈ [-1, 1]`, which behaves like a cosine proxy. This is the
1-bit LSH that production vector databases ship as "binary quantization"; on its own it is
lossy, which is exactly what the rescore stage is for.

Derive every variant from an existing float32 corpus without re-embedding anything:

```bash
python -m zcgvs_tools.quantize --input data/scifact.bin --dtype float16 int8 binary
```

`engine.storage_dtype` reports which form is in use; `engine.stats().corpus_bytes` reports the
resident VRAM.

## On-device rescoring

Quantized retrieval is only useful if you can buy the accuracy back. The standard recipe —
scan cheap, then re-score the survivors at full precision — normally costs a host round trip:
copy candidate ids back, look up full-precision vectors, re-score. This engine keeps the whole
thing on the GPU.

```
binary scan  →  scores [batch, N]
                    │
              top-K (M candidates)   →  indices [batch, M]   ── device buffer
                    │                            │
                    └──────────────────────────► gather + rescore against the fp32 store
                                                 │
                                           top-K (k)  →  local positions [batch, k]
                                                 │
                                           remap → corpus indices [batch, k]
```

The gather kernel reads the index buffer the top-K pass just wrote. Nothing is copied,
nothing is synchronized; the coarse-to-exact hand-off is a device pointer. Configure it with
two fields:

```json
{
    "embeddings_path": "data/scifact.binary.bin",
    "rescore_embeddings_path": "data/scifact.bin",
    "rescore_candidates": 512
}
```

The rescore store must hold the same documents in the same order. It may be `float32`,
`float16` or `int8` — anything but binary. VRAM cost is the sum of both stores, so a binary
primary plus a float32 rescore store is still smaller than fp32 alone would be... only
marginally (33/32), which is the honest trade: the win is that the **scan** — the part that
runs over every document on every query — reads 32× less. Pair binary with an `int8` rescore
store for a 4.1× total reduction, or keep the exact store on the host if you can afford one
copy per query (not implemented here; the device-resident version is the point).

`tests/cpp/test_pipeline.cu` pins this down: with a lossless (float16) coarse pass the
rescored top-K matches the CPU float32 ranking exactly.

## Batched multi-query scan

A single-query brute-force scan reads the whole corpus and does one multiply-add per element
loaded — arithmetic intensity of about 1 FLOP per 4 bytes, hopelessly bandwidth-bound. Batching
fixes that: each resident block stages up to `kQueryTile = 8` query vectors in shared memory
once, and its warps reuse every loaded corpus element across all of them while striding over
the corpus.

```cpp
// one warp per document, persistent over the corpus
for (int doc = blockIdx.x * warps + warp; doc < doc_count; doc += gridDim.x * warps) {
    for (int base = lane * kVec; base < dim; base += 32 * kVec) {
        float values[kVec];
        loadRow<Storage, kVec>(row + base, scale, values);   // one 128-bit global load
        #pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {               // eight uses of each element
            if (q < tile) { /* float4 shared reads of query q, accumulate into partial[q] */ }
        }
    }
    // butterfly-reduce partial[q] across the warp; lane q stores query q's score
}
```

Corpus traffic per query drops by up to 8×, and the launch overhead of the whole pipeline is
amortized across the batch. The `partial[]` accumulators are unrolled to compile-time indices
so they stay in registers.

```python
response = engine.search_batch(["query one", "query two", "query three"], options)
for hits in response.results:
    ...
print(response.timing.batch, response.timing.search_gpu_ms)
```

The top-K passes are batched the same way, with the query index on `blockIdx.y`.

## Recursive top-K merge

The first version of this engine merged all block-local candidates in a single block, which
capped the padded candidate set at 4096 and therefore capped the corpus. The merge now
recurses:

```
scores [N]
   │  local pass: ceil(N/1024) blocks, 1024-element bitonic tiles
   ▼
candidates [B·k]
   │  merge pass: ceil(C/4096) blocks, 4096-element bitonic tiles      ◄─┐
   ▼                                                                     │ ping-pong
candidates [B'·k]  ── while more than one block remains ─────────────────┘
   │
   ▼  final merge: 1 block
indices/scores [k]
```

Because a merge tile holds 4096 elements and `k ≤ 1024`, each merge pass shrinks the candidate
set by at least 4×, so the recursion is guaranteed to terminate. The bitonic network sorts
ascending; the largest `k` occupy the tail and are read out in descending order. Padding with
`-inf` and index `-1` means pad entries can never displace a real document.

Two scratch buffers of `ceil(N/1024) · k` elements per query are ping-ponged between passes;
`cuda::topKScratchElements(N, k)` sizes them.

`tests/cpp/test_topk_large.cu` runs one million scores at `k = 512` — several merge passes —
against a `std::stable_sort` reference.

## CUDA graph replay

The device pipeline is a fixed sequence of kernel launches whose shapes depend only on
`(corpus size, batch, k, rescore width, metric, mask presence)`. That makes it a textbook
candidate for CUDA graphs: capture once, then replay the whole thing with a single
`cudaGraphLaunch` instead of paying per-launch CPU overhead every query.

```python
options = engine.default_options()
options.use_config_graphs = False
options.use_graph = True
response = engine.search_with(query, options)
assert response.timing.graph_replay
```

The searcher caches one instantiated graph keyed by every value that gets baked into it —
buffer pointers, corpus pointer, mask pointer, an allocation generation counter, and each
by-value kernel argument. Any change invalidates the cache and triggers a recapture, so a
stale graph can never run. Capture happens **outside** the CUDA event bracket, so the reported
`search_gpu_ms` is pure replay time.

This matters most where it is easiest to overlook: small corpora and small batches, where
launch overhead is a real fraction of the query. `tests/cpp/test_pipeline.cu` asserts that a
replayed graph produces bit-identical indices and scores to direct launches.

Graphs cover the custom-kernel path. The cuBLAS backend always launches directly.

## Hybrid BM25 + dense retrieval

Dense retrieval misses exact-match queries — product codes, rare proper nouns, numbers. Every
production vector database therefore ships hybrid search. This engine does the lexical half on
the GPU too.

**Offline**, `zcgvs_tools.build_lexical` builds a CSR inverted index over WordPiece term ids —
the same vocabulary the engine tokenizes queries with, so no separate analyzer has to be kept
in sync across the Python/C++ boundary. Crucially, each posting stores its *complete* BM25
contribution:

```
w(t, d) = idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
idf(t)  = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

**At query time** the device work is then a pure scatter-add — no corpus statistics, no
division, no branching:

```cpp
const int term = query_terms[blockIdx.x];
for (int64_t p = offsets[term] + threadIdx.x; p < offsets[term + 1]; p += blockDim.x) {
    atomicAdd(&scores[docs[p]], weights[p]);
}
```

Fusion is also on-device: a block-reduced `atomicMax` finds the maximum lexical score, then an
element-wise kernel forms the convex combination

```
fused[d] = dense_weight · dense[d] + (1 - dense_weight) · lexical[d] / max(lexical)
```

before the single top-K pass runs over the fused scores. The alternative — reciprocal rank
fusion over two separate top-K lists — needs two selections and a host-side merge; the
device-side linear fusion keeps the zero-copy property intact.

```bash
python -m zcgvs_tools.build_lexical \
    --metadata data/scifact.meta --vocab models/vocab.txt \
    --tokenizer-dir models/encoder --output data/scifact.lex
```

```python
options.hybrid = True
options.dense_weight = 0.7      # 1.0 = dense only, 0.0 = BM25 only
engine.search_with("crispr cas9", options)
```

`--tokenizer-dir` uses the exported HuggingFace tokenizer; omit it and the builder falls back
to a pure-Python WordPiece pass that mirrors `src/text/tokenizer.cpp`, so the index can be
built with nothing but NumPy. The two tokenizers agree on ordinary English text; where they
differ, a term simply fails to match and that query term contributes nothing — the failure
mode is graceful.

Masked documents keep their `-FLT_MAX` dense score through fusion, so filtering still wins.

## cuBLAS reference backend

A hand-written kernel is only interesting if you know what it is being compared against.
`SimilarityBackend::Cublas` computes the identical `[batch, N]` score matrix as a GEMM:

```cpp
// row-major [N, D] corpus == column-major [D, N]; scores = corpusᵀ · queries
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, batch, D, &alpha,
            corpus, D, queries, D, &beta, scores, N);
```

The GEMM path has no mask predicate and no quantized decode, so it covers float32
inner-product metrics only; anything else falls back to the custom kernel automatically
(`CublasSimilarity::supports`). Select it per request or per config:

```python
options.use_config_backend = False
options.backend = zcgvs_engine.SimilarityBackend.Cublas
```

`zcgvs_bench_kernels` times both in the same harness. Expect cuBLAS to win on plain fp32 —
that is the honest result, and the interesting numbers are how close the custom kernel gets on
fp32 and how far ahead it is on int8 and binary, which cuBLAS cannot express at all.

## Similarity metrics

`Metric.Cosine` (dot product over L2-normalized vectors), `Metric.DotProduct` (raw inner
product) and `Metric.L2` (negated squared Euclidean distance, so larger is still better) are
selected per call and handled inside the one scan kernel.

```python
options.use_config_metric = False
options.metric = zcgvs_engine.Metric.L2
```

Binary storage always scores by Hamming distance; the metric selector does not apply to it.

## Filtered search

Pass an allow-list of corpus indices; the engine builds a device-resident `uint8` mask and the
scan assigns masked-out documents `-FLT_MAX` so they can never enter the top-K. This is the
metadata-filter primitive of a vector database — build the allow-list from whatever predicate
you keep host-side (category, recency, ACLs).

```python
options.allow = [3, 17, 42, 128]
```

The mask is evaluated inside the scan, before any scoring work, so filtered queries are
cheaper rather than more expensive.

## Incremental indexing

Add documents to the live index without reloading. The engine embeds each passage with the
bi-encoder and appends the vectors to VRAM (growing the device buffer geometrically),
converting to the store's precision **on the device** — half conversion, int8 quantization or
sign-bit packing, whichever applies. The searcher reads the current corpus size on each query,
so new documents are searchable immediately, and the graph cache invalidates automatically
when the corpus pointer moves.

```python
engine.add_documents(ids=["d1", "d2"], texts=["first passage", "second passage"])
```

Refused while a rescore store or lexical index is attached: both are built offline over a
fixed document set and would silently go stale.

## Stage timing and profiling

`search_with` and `search_batch` return a `SearchTiming` alongside the hits. GPU time comes
from CUDA events bracketing the device pipeline; host stages are wall clock.

```python
timing = response.timing
print(timing.tokenize_ms, timing.embed_ms, timing.search_gpu_ms,
      timing.copy_ms, timing.rerank_ms, timing.total_ms,
      timing.batch, timing.graph_replay)
```

For kernel-level attribution, build with `-DZCGVS_ENABLE_NVTX=ON` and run under Nsight
Systems — `zcgvs::search`, `zcgvs::rerank` and `zcgvs::copyResultToHost` ranges appear on the
timeline. The macro compiles to nothing when the option is off, so the default build carries
no profiler dependency.

## Retrieval-only queries

`retrieve` runs embed → scan → top-K (→ rescore) and returns `Candidate`s (doc id, corpus
index, score) without the cross-encoder. Useful for evaluating the search stage in isolation
and for the GPU/CPU parity check in `eval_quality.py`. Setting `options.rerank = False` does
the same for the full-fat API. Leaving `reranker_model_path` empty in the config runs the
whole engine as a pure retrieval service.

## HTTP service

```bash
PYTHONPATH=build/bindings:python python -m zcgvs_tools.service --config configs/scifact.json
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, corpus size, storage dtype, which stages are available |
| `GET /stats` | Full engine, index and device statistics |
| `POST /search` | One query; every per-request option is exposed |
| `POST /search/batch` | Many queries in a single device pass |
| `POST /retrieve` | Search stage only |
| `POST /documents` | Append to the live VRAM index |
| `POST /verify` | Fact-check a claim with Llama: verdict, probabilities, cited explanation (`--llama`) |
| `POST /ask` | Grounded question answering with Llama, cited to retrieved sources (`--llama`) |

Every knob — metric, allow-list, hybrid, dense weight, rescore width, backend, CUDA graphs —
is settable per request.

## Engine statistics

```python
stats = engine.stats()
stats.corpus_size, stats.dim, stats.storage_dtype, stats.corpus_bytes
stats.has_rescore_store, stats.rescore_dtype, stats.rescore_bytes
stats.has_lexical_index, stats.lexical_postings, stats.lexical_bytes
stats.graph_captures, stats.graph_launches
stats.device_name, stats.device_total_memory, stats.compute_major, stats.compute_minor
```

`graph_captures` vs `graph_launches` is the diagnostic that matters: if captures keep climbing
alongside launches, something in the request shape is changing every query and the graph is
being rebuilt instead of replayed.

## Kernel roofline benchmark

`build/apps/zcgvs_bench_kernels` runs on synthetic data — no models, no corpus, no network —
so it works the moment the build succeeds:

```bash
./build/apps/zcgvs_bench_kernels --docs 1000000 --dim 384 --k 100 --markdown 1
```

It reports, against the device's theoretical peak DRAM bandwidth: the scan at each storage
precision, `cublasSgemv` on the same data, top-K at several widths, and pipeline throughput
across batch sizes with and without CUDA graphs. Output is markdown-ready for
[`BENCHMARKS.md`](BENCHMARKS.md).

## Python API reference

See [`API.md`](API.md) for the complete surface, and [`FORMATS.md`](FORMATS.md) for the
byte-level wire specifications.
