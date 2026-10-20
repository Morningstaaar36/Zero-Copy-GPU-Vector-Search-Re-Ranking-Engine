# API reference

Python surface first, since that is what most callers use; the C++ headers follow the same
shapes.

## Python

```python
import zcgvs_engine
from zcgvs_tools.config import build_engine, describe
```

### Construction

```python
engine = build_engine("configs/scifact.json")   # from JSON
engine = build_engine({...})                    # from a dict
engine = zcgvs_engine.Engine(config)            # from an EngineConfig
```

### `EngineConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `embeddings_path` | str | — | Primary corpus matrix (`.bin`) |
| `metadata_path` | str | — | Corpus metadata (`.meta`) |
| `vocab_path` | str | — | WordPiece `vocab.txt` |
| `rescore_embeddings_path` | str | `""` | Exact-precision corpus for the rescore stage |
| `lexical_index_path` | str | `""` | CSR inverted index (`.lex`) enabling hybrid retrieval |
| `encoder_model_path` | str | — | Bi-encoder ONNX model |
| `encoder_output_name` | str | `"last_hidden_state"` | Output tensor to mean-pool |
| `encoder_max_seq_len` | int | 256 | Query truncation length |
| `reranker_model_path` | str | `""` | Cross-encoder ONNX model; empty runs retrieval-only |
| `reranker_output_name` | str | `"logits"` | Output tensor to read |
| `reranker_max_seq_len` | int | 256 | Pair truncation length |
| `reranker_batch_size` | int | 32 | Pairs per ONNX call |
| `device_id` | int | 0 | CUDA device |
| `lowercase` | bool | True | Tokenizer case folding |
| `metric` | `Metric` | `Cosine` | Default similarity metric |
| `similarity_backend` | `SimilarityBackend` | `Custom` | Default scan implementation |
| `default_top_k` | int | 10 | Results returned when unspecified |
| `default_candidates` | int | 100 | Candidates fed to the reranker |
| `rescore_candidates` | int | 0 | Coarse candidate width; 0 disables rescoring |
| `hybrid` | bool | False | Run the BM25 stage by default; attaching an index only makes it *available* |
| `dense_weight` | float | 0.7 | Convex weight on the dense score in hybrid mode |
| `max_query_terms` | int | 64 | Cap on distinct query terms fed to BM25 |
| `use_cuda_graphs` | bool | False | Replay the device pipeline from a captured graph |

In JSON, `metric` and `similarity_backend` are strings (`"cosine"`, `"dot"`, `"l2"`;
`"custom"`, `"cublas"`). Path fields are resolved against the working directory, then the
config's directory, then its parent.

### `SearchOptions`

Get one pre-filled from the config with `engine.default_options()`, then override.

| Field | Type | Sentinel | Meaning |
| --- | --- | --- | --- |
| `top_k` | int | `<= 0` → config | Results to return |
| `candidates` | int | `<= 0` → config | Candidates selected before reranking |
| `use_config_metric` | bool | `True` | Set `False` to honour `metric` |
| `metric` | `Metric` | — | Cosine / DotProduct / L2 |
| `allow` | list[int] | `[]` | Corpus indices the scan is restricted to |
| `rescore_candidates` | int | `< 0` → config | Coarse candidate width |
| `dense_weight` | float | `< 0` → config | 1.0 dense-only, 0.0 BM25-only |
| `hybrid` | bool | from config | Enable the BM25 stage (needs a lexical index) |
| `rerank` | bool | `True` | Run the cross-encoder |
| `use_config_backend` | bool | `True` | Set `False` to honour `backend` |
| `backend` | `SimilarityBackend` | — | Custom / Cublas |
| `use_config_graphs` | bool | `True` | Set `False` to honour `use_graph` |
| `use_graph` | bool | `False` | Replay from a CUDA graph |

`candidates` is clamped up to `top_k` and down to the corpus size; `rescore_candidates` is
clamped up to `candidates`. Both are capped at 1024 by the bitonic network, and exceeding that
raises rather than silently truncating.

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `search(query, top_k, candidates)` | `list[SearchHit]` | Simple form |
| `search(query)` | `list[SearchHit]` | Uses the config defaults |
| `search_timed(query, top_k, candidates, metric=Cosine, allow=[])` | `SearchResponse` | Explicit metric and filter |
| `search_with(query, options)` | `SearchResponse` | Full option surface |
| `search_batch(queries, options)` | `BatchSearchResponse` | One device pass for all queries |
| `retrieve(query, k, metric=Cosine, allow=[])` | `list[Candidate]` | No reranker |
| `add_documents(ids, texts)` | `None` | Appends to the live VRAM index |
| `default_options()` | `SearchOptions` | Pre-filled from the config |
| `stats()` | `EngineStats` | Resident sizes, wired stages, device info |

Properties: `corpus_size`, `dim`, `storage_dtype`, `has_lexical_index`, `has_rescore_store`,
`has_reranker`.

Every search method releases the GIL for the CUDA and ONNX work.

### Result types

```python
hit.doc_id        # str — corpus doc id
hit.text          # str — passage text
hit.similarity    # float — retrieval score (exact, if rescoring ran)
hit.rerank_score  # float — cross-encoder logit; equals similarity when rerank is off
hit.corpus_index  # int — row index in the corpus

candidate.doc_id, candidate.score, candidate.corpus_index

response.hits             # list[SearchHit]
response.timing           # SearchTiming
batch_response.results    # list[list[SearchHit]], aligned with the query list
batch_response.timing     # SearchTiming, with .batch set
```

`SearchTiming`: `tokenize_ms`, `embed_ms`, `search_gpu_ms`, `copy_ms`, `rerank_ms`,
`total_ms`, `batch`, `graph_replay`.

`EngineStats`: `corpus_size`, `dim`, `storage_dtype`, `corpus_bytes`, `has_rescore_store`,
`rescore_dtype`, `rescore_bytes`, `has_lexical_index`, `lexical_postings`, `lexical_bytes`,
`has_reranker`, `graph_captures`, `graph_launches`, `device_name`, `device_total_memory`,
`compute_major`, `compute_minor`.

### Enums

```python
zcgvs_engine.Metric.Cosine | DotProduct | L2
zcgvs_engine.SimilarityBackend.Custom | Cublas
zcgvs_engine.RerankerBackend.Onnx
```

### Errors

Invalid arguments raise `ValueError`; CUDA and runtime failures raise `RuntimeError`, with the
CUDA error name, string, and the source location of the failing call in the message. The HTTP
service maps both to `400` with the message in `detail`.

---

## Offline tooling

| Module | Entry point | Purpose |
| --- | --- | --- |
| `zcgvs_tools.export_models` | `zcgvs-export-models` | PyTorch checkpoints → ONNX + `vocab.txt` |
| `zcgvs_tools.embed_corpus` | `zcgvs-embed-corpus` | BEIR subset → `.bin` + `.meta` |
| `zcgvs_tools.quantize` | `zcgvs-quantize` | float32 `.bin` → float16 / int8 / binary |
| `zcgvs_tools.build_lexical` | `zcgvs-build-lexical` | `.meta` + `vocab.txt` → `.lex` |
| `zcgvs_tools.service` | `zcgvs-serve` | FastAPI server |
| `zcgvs_tools.formats` | — | Wire-format readers and writers (NumPy only) |
| `zcgvs_tools.config` | — | JSON → `EngineConfig`; `describe(engine)` |
| `zcgvs_tools.beir_loader` | — | BEIR corpus / queries / qrels |

`formats` and `quantize` import nothing but NumPy, so the artifact tooling works without torch
or a built extension module.

---

## C++

The header layout mirrors the Python surface.

```cpp
#include "zcgvs/engine/engine.hpp"

zcgvs::EngineConfig config;
config.embeddings_path = "data/scifact.binary.bin";
config.rescore_embeddings_path = "data/scifact.bin";
config.rescore_candidates = 512;
config.use_cuda_graphs = true;
// ... encoder, reranker, vocab, metadata

zcgvs::Engine engine(config);

zcgvs::SearchOptions options = engine.defaultOptions();
options.top_k = 10;
options.candidates = 100;

const zcgvs::SearchResponse response = engine.searchTimed("crispr gene editing", options);
for (const zcgvs::SearchHit& hit : response.hits) { /* ... */ }

const zcgvs::BatchSearchResponse batch =
    engine.searchBatch({"query one", "query two"}, options);

const zcgvs::EngineStats stats = engine.stats();
```

### Layer boundaries

The dependency direction is strictly downward:

```
engine → {embed, rerank, search, data, text} → {onnx, cuda, core}
```

`QueryEmbedder` and `Reranker` are pure interfaces, which is why neither the engine nor the
pybind layer includes an ONNX header.

### Kernel launchers

Useful if you want the kernels without the engine — this is what the benchmark and the tests
do.

```cpp
#include "zcgvs/cuda/similarity.cuh"
#include "zcgvs/cuda/topk.cuh"
#include "zcgvs/cuda/rescore.cuh"
#include "zcgvs/cuda/lexical.cuh"
#include "zcgvs/cuda/convert.cuh"
```

| Function | Purpose |
| --- | --- |
| `launchSimilarity(corpus, dtype, scale, N, D, queries, batch, mask, metric, scores, stream)` | Brute-force scan → `[batch, N]` |
| `launchTopK(scores, N, k, batch, out_idx, out_val, scratch_a…, scratch_b…, stream)` | Recursive bitonic selection → `[batch, k]` |
| `topKScratchElements(N, k)` | Per-query size of each ping-pong scratch buffer |
| `topKMaxK()` | Largest supported `k` (1024) |
| `launchGatherRescore(corpus, dtype, scale, N, D, queries, batch, indices, M, metric, values, stream)` | Exact re-scoring of a device-resident candidate set |
| `launchRemapIndices(candidates, local, M, k, batch, out, stream)` | Candidate positions → corpus indices |
| `launchBm25(offsets, docs, weights, terms, term_count, N, scores, stream)` | CSR scatter-add; caller zeroes `scores` |
| `launchMaxPositive(values, N, max, stream)` | Max reduction; caller zeroes `max` |
| `launchFuseScores(dense, lexical, max, weight, N, stream)` | In-place convex combination |
| `launchConvertF32ToF16 / ToI8 / launchPackBinary` | Device-side storage conversion |

Building a store without an artifact file, for tests and benchmarks:

```cpp
const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
    host_float32, doc_count, dim, zcgvs::ScalarType::Binary, 1.0f, true);
```

### Driving `Searcher` directly

```cpp
zcgvs::Searcher searcher(store);
searcher.setRescoreStore(&exact_store);      // optional
searcher.setLexicalIndex(&index);            // optional

zcgvs::SearchParams params;
params.k = 100;
params.metric = zcgvs::Metric::Cosine;
params.rescore_candidates = 512;
params.use_graph = true;

searcher.search(host_queries, dim, batch, params, stream);
const zcgvs::HostTopK selection = searcher.copyResultToHost(stream);

selection.indices[selection.offset(query) + rank];   // corpus index, or -1 for padding
searcher.lastSearchGpuMs();
searcher.lastSearchUsedGraph();
```

`SearchParams` carries `host_query_terms` and `lexical_stride` for the hybrid path: a flattened
`[batch, lexical_stride]` array of WordPiece ids, `-1` padded. A `-1` slot is skipped by the
BM25 kernel, which is what makes the launch shape constant across queries and therefore
capturable into a graph.

### Ownership

Every device allocation is RAII: `DeviceBuffer<T>`, `PinnedBuffer<T>`, `CudaStream`, and the
searcher's `cudaGraphExec_t`. Nothing above `core` calls `cudaMalloc` or `cudaFree` directly.
The `Engine` is a single object owned by Python; `~Engine()` is declared out-of-line so the
`unique_ptr`s to forward-declared types see complete types in the translation unit.

## Llama layer (Python)

| Object | Purpose |
| --- | --- |
| `zcgvs_tools.llama.Llama(model_id, device="cuda:0")` | Loads a Llama 3.x Instruct checkpoint in bf16 (Hub id or local dir; gated repos fall back to an identical mirror) |
| `Llama.stream(messages, max_new_tokens, on_text)` | Chat generation; returns `Generation(text, prompt_tokens, new_tokens, ttft_ms, total_ms, tokens_per_s)` |
| `Llama.choice_logprobs(prompts, choices)` | `[n, len(choices)]` log-probabilities of each choice's first token, one forward pass, fp32 head |
| `Llama.continuation_logprob(prompts, continuation)` | Mean per-token log-probability of a shared continuation, teacher-forced |
| `factcheck.ClaimVerifier(engine, llama, evidence_k=3)` | `.verify(claim, explain=True, on_text=None) -> Verdict`; `.classify(claims, evidence_sets)` for batches |
| `factcheck.Verdict` | `label`, `probabilities`, `confidence`, `evidence: list[Evidence]`, `explanation`, `quote_checks`, `quotes_verified`, `timing_ms` |
| `factcheck.LlamaReranker(llama, mode="query_likelihood")` | `.score(query, passages)`, `.rerank(query, evidence)`; `mode="yes_no"` for pointwise judging |
| `factcheck.GroundedAnswerer(engine, llama, k=5)` | `.answer(question, on_text=None) -> (Generation, evidence, timing)` |
| `factcheck.check_quotes(explanation, evidence)` | Verifies each quoted passage is verbatim in the abstract it cites; returns `QuoteCheck(quote, cited, found_in)` with `.verified` |
| `factcheck.fit_label_bias(logprobs, gold)` | Contextual-calibration offsets maximizing macro-F1; saved per model in `configs/calibration/` |

Details and measured results: [LLAMA.md](LLAMA.md).

