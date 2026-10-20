# Learning path — understand this codebase line by line

A dependency-ordered map from *concept* → *the exact file it explains* → *the exact resource
to learn it from* → *the question you must be able to answer before moving on*.

**The rule: never read or watch a resource cover to cover.** Every entry below says precisely
what to take from it. Most are 10–40 minutes. If an entry says "§5.3", read §5.3 and stop.

Total: **≈ 40 focused hours**. Five hard days, or three weeks at two hours an evening. That is
the honest cost of being able to defend every line of this repo in an interview.

---

## How to use this document

Work one tier at a time, and for each tier run this loop:

1. **Read the concept list.** If you already know all of it, skip to the checkpoint.
2. **Take exactly the resource slice listed.** Nothing more.
3. **Open the files in the "unlocks" list and read them top to bottom.** Not skim — read.
4. **Answer the checkpoint questions out loud, without looking.** If you can't, you haven't
   got it yet; go back to step 2.
5. **Run the experiment** where one is given. Predicting a failure correctly is the strongest
   possible evidence you understand the mechanism.

The checkpoint questions are not trivia. Every one of them is answerable *only* from this
codebase, and every one of them is something an interviewer would actually ask.

### Where to start reading code

Start with **`include/zcgvs/core/scalar_type.hpp`**. It is 70 lines, has no dependencies, and
already contains three ideas the whole engine rests on: a strongly-typed enum with an explicit
underlying type (because it is serialized to disk), ceiling division for bit packing
(`(dim + 31) / 32`), and the fact that "bytes per row" is a *function of dtype*, not a
constant. Everything else grows out of that file.

---

## Tier 0 — The mental model (≈ 4 h)

Before any file. If your model of a GPU is "a CPU with more cores", nothing below will land.

**Concepts**

- SIMT: threads → warps (32) → blocks → grid; blocks are scheduled to SMs and cannot
  communicate across blocks
- Latency hiding by oversubscription, not by caches — why a GPU wants thousands of threads
- The memory hierarchy: registers → shared memory → L2 → DRAM, and the ~100× latency cliff
- **Memory-bound vs compute-bound**, and the roofline model
- Occupancy, and why more occupancy is not automatically better

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| The whole talk. This is the single highest-value hour in GPU computing. | Stephen Jones, **"How GPU Computing Works"** (NVIDIA GTC) — [nvidia.com/on-demand](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/) | 50 min |
| Follow-up from the same author on writing kernels | Stephen Jones, **"How to Write a CUDA Program"** — search NVIDIA On-Demand for the title | 50 min |
| §1–2 only (programming model, thread hierarchy). Bookmark the rest as reference. | **CUDA C++ Programming Guide** — [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 60 min |
| The roofline section only | **CUDA C++ Best Practices Guide**, §"Performance Metrics" — [docs.nvidia.com/cuda/cuda-c-best-practices-guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 30 min |
| You are already doing this — finish through the shared-memory section and stop. | freeCodeCamp, **"CUDA Programming Course – High-Performance Computing with GPUs"** — [youtube.com/watch?v=86FAWCzIe_4](https://www.youtube.com/watch?v=86FAWCzIe_4) | as needed |
| PMPP Ch. 4 (compute architecture) and Ch. 5 (memory architecture) | *Programming Massively Parallel Processors*, 4th ed. | 3 h |

**Checkpoint**

1. This engine's scan reads the entire corpus for every query and does one multiply-add per
   element. Compute its arithmetic intensity in FLOP/byte for `float32` and for `binary`.
   Which one is closer to compute-bound, and why?
2. Why does `docs/BENCHMARKS.md` insist the meaningful figure is GB/s against theoretical
   peak, rather than milliseconds?
3. `DeviceInfo::peakBandwidthGBs()` in `include/zcgvs/core/cuda_context.hpp` is
   `2 × memory_clock × bus_width / 8`. Where does the factor of 2 come from?

---

## Tier 1 — CUDA plumbing and C++ RAII (≈ 4 h)

**Unlocks:** `core/cuda_check.hpp`, `core/device_buffer.hpp`, `core/pinned_buffer.hpp`,
`core/cuda_stream.hpp`, `core/cuda_context.{hpp,cpp}`, `cuda/convert.cu`

**Concepts**

- `cudaMalloc` / `cudaFree` / `cudaMemcpyAsync`; why kernel launches are asynchronous
- Why you must call `cudaGetLastError()` *after* a launch (launch errors are async)
- Streams as ordered queues; `cudaStreamSynchronize`
- **Pinned (page-locked) host memory** and why a pageable DtoH copy is slower *and* blocking
- C++ RAII, the rule of five, move semantics, `= delete`, `noexcept`
- Macro hygiene: the `do { … } while (0)` idiom, `__FILE__` / `__LINE__`
- Grid-stride loops
- Bit manipulation: `1u << b`, ceiling division

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| Whole post — it is short and is exactly what `PinnedBuffer` exists for | **"How to Optimize Data Transfers in CUDA C/C++"** — [developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) | 15 min |
| Whole post | **"How to Overlap Data Transfers in CUDA C/C++"** — [developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) | 15 min |
| Whole post | **"CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"** — [developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) | 10 min |
| The "Rule of five" section, and the move-constructor page | **cppreference**, [rule of three/five/zero](https://en.cppreference.com/w/cpp/language/rule_of_three) and [RAII](https://en.cppreference.com/w/cpp/language/raii) | 30 min |
| Whole talk — the clearest explanation of move semantics available | Klaus Iglberger, **"Back to Basics: Move Semantics"** (CppCon) — [youtube.com/watch?v=St0MNEU5b0o](https://www.youtube.com/watch?v=St0MNEU5b0o) | 60 min |

**Checkpoint**

1. `DeviceBuffer` deletes its copy constructor but defines a move constructor. What would go
   wrong at runtime — concretely — if the copy constructor were defaulted instead?
2. `EmbeddingStore` never declares a destructor, copy constructor or move constructor, yet
   `std::make_unique<EmbeddingStore>(EmbeddingStore::loadFromFile(path))` in `engine.cpp`
   compiles and does the right thing. Explain the exact rule that makes that work.
3. `ZCGVS_CUDA_CHECK_LAST()` is called after every kernel launch in this codebase. Why is
   checking the return value of the launch itself not sufficient?
4. `Searcher::copyResultToHost` copies through `PinnedBuffer`. The transfer is a few hundred
   bytes. Give the real reason this matters that is *not* "pinned memory is faster".

---

## Tier 2 — The scan kernel (≈ 5 h)

**Unlocks:** `cuda/reduction.cuh`, `cuda/decode.cuh`, `cuda/similarity.cu`

This is the heart of the project. Budget real time here.

**Concepts**

- **Memory coalescing** — why `for (t = threadIdx.x; t < dim; t += blockDim.x)` is the
  correct loop and `for (t = threadIdx.x * n; ...)` would be catastrophic
- **Parallel reduction**: the tree pattern, and why naive versions have warp divergence
- **Warp shuffle** (`__shfl_down_sync`) — register-to-register exchange with no shared memory
- Two-level reduction: warp-level shuffle, then a shared-memory fan-in across warps
- `__syncthreads()` and the **block-uniform branching** requirement — divergent barriers
  deadlock
- Static vs **dynamic shared memory** (`extern __shared__`), and how the size is passed at
  launch
- Templates in device code; one kernel, four storage types
- `#pragma unroll` and the **register vs local-memory** consequence of runtime array indexing
- 2D grids (`blockIdx.y`)
- `__popc` (population count) for Hamming distance
- Shared-memory tiling for **data reuse** — the difference between a bandwidth-bound and an
  intensity-bound kernel

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| Whole post | Mark Harris, **"How to Access Global Memory Efficiently in CUDA C/C++ Kernels"** — [developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/) | 20 min |
| Whole post | Mark Harris, **"Using Shared Memory in CUDA C/C++"** — [developer.nvidia.com/blog/using-shared-memory-cuda-cc](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) | 25 min |
| The classic. Slides 1–35: the seven successive optimizations of a reduction. | Mark Harris, **"Optimizing Parallel Reduction in CUDA"** — [developer.download.nvidia.com/assets/cuda/files/reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) | 45 min |
| Whole post — this is literally `blockReduceSum` in `reduction.cuh` | **"Faster Parallel Reductions on Kepler"** — [developer.nvidia.com/blog/faster-parallel-reductions-kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) | 25 min |
| Reductions lecture, and the "CUDA performance checklist" lecture | **GPU MODE** lecture series — [youtube.com/@GPUMODE](https://www.youtube.com/@GPUMODE) | 2 × 60 min |
| PMPP Ch. 10 (reduction) — the whole chapter. Ch. 6 §6.1–6.3 (coalescing). | *PMPP* 4th ed. | 2 h |
| §"Shared Memory" and §"Warp Shuffle Functions" only | CUDA C++ Programming Guide | 30 min |

**Checkpoint**

1. In `similarityScanKernel`, the accumulation loop is

   ```cpp
   #pragma unroll
   for (int q = 0; q < kQueryTile; ++q) {
       if (q < tile) { /* accumulate query q against values[0..kVec) into partial[q] */ }
   }
   ```

   Why is it written with a compile-time bound plus a runtime `if`, instead of the obvious
   `for (int q = 0; q < tile; ++q)`? What exactly does the compiler do differently, and what
   is the performance consequence? (`docs/ARCHITECTURE.md` states the answer — make sure you
   can derive it rather than recite it.)
2. The kernel `continue`s past a masked document, then reduces with `__shfl_xor_sync` over a
   full-warp mask. Explain why that is safe. Then construct a one-line change to the mask
   check that would make the shuffle undefined behaviour.
3. Query tiling reads each corpus row once for up to 8 queries. Derive the new arithmetic
   intensity and explain why the speedup flattens out past `kQueryTile`.
4. `hammingScanKernel` gives each *thread* a document, while the float kernels give each
   *warp* one. At `dim = 384`, how many bytes is a binary row, and what fraction of a warp's
   lanes would sit idle under warp-per-document?
5. `launchSimilarity` chunks the batch when shared memory would be exceeded. Compute, for
   `dim = 1536`, how many queries fit per block under `kMaxQuerySharedBytes = 40960`.
6. Both kernels declare `extern __shared__ float4 zcgvs_shared_vec[]` and then cast it to
   `float*` or `unsigned int*`. Why `float4` rather than `unsigned int`?

**Experiment.** In `launchScan`, replace `persistentGrid(...)` with `needed` (one block per 8
documents, no striding). Every block now re-stages the query tile. Predict the slowdown for
float32 *and* for int8 from the bytes staged per block versus the bytes of corpus it scores,
before running `zcgvs_bench_kernels`. Then run it. (Revert afterwards.)

---

## Tier 3 — Selection: bitonic sort and the recursive merge (≈ 5 h)

**Unlocks:** `cuda/bitonic.cuh`, `cuda/topk.cu`, `tests/cpp/test_topk_large.cu`

**Concepts**

- Why **selection is harder to parallelize than reduction** — the answer depends on a global
  ordering
- **Sorting networks**; bitonic sequences; compare-exchange
- The `i ^ step` partner-index trick and why `(i & stage) == 0` gives the sort direction
- O(n log²n) compare-exchanges, and why that is acceptable for a 1024-element tile
- Padding with `-inf` / index `-1` as a correctness device
- Multi-pass reduction with **ping-pong buffers**
- **Termination arguments** — proving a recursion shrinks

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| The whole page — it is the clearest CUDA-specific walkthrough | **CUDA bitonic sort tutorial** (Rutgers) — [people.cs.rutgers.edu/~venugopa/parallel_summer2012/cuda_bitonic.html](https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/cuda_bitonic.html) | 40 min |
| The diagrams. Trace an 8-element network by hand on paper — do not skip this. | **Bitonic sorter** — [en.wikipedia.org/wiki/Bitonic_sorter](https://en.wikipedia.org/wiki/Bitonic_sorter) | 40 min |
| §1–4 only (problem statement and the bitonic top-k approach). The rest is beyond scope. | Shanbhag et al., **"Efficient Top-K Query Processing on Massively Parallel Hardware"**, SIGMOD 2018 — [doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf](https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf) | 60 min |
| PMPP Ch. 12 (merge) and Ch. 13 (sorting) | *PMPP* 4th ed. | 2 h |

**Checkpoint**

1. Trace `bitonicSortSharedAscending` on 8 elements by hand. How many compare-exchange stages
   run, and how many `__syncthreads()` calls execute?
2. `kTileSize` is 1024 for the local pass but `kMergeTile` is 4096 for merge passes. Prove
   that with `k ≤ 1024` the merge recursion terminates, and show what breaks if `kMergeTile`
   were 1024.
3. For `N = 1,000,000` and `k = 512`, work out every pass `launchTopK` performs: how many
   blocks, and the candidate count after each. (Check yourself against
   `tests/cpp/test_topk_large.cu`.)
4. Padding uses `-FLT_MAX` with index `-1`. Under what corpus condition could a `-1` index
   actually reach the final output, and where does the calling code defend against it?
5. `topKScratchElements(N, k)` sizes *one* ping-pong buffer. Why are two needed, and why is
   the pass-0 output size a safe upper bound for every later pass?

---

## Tier 4 — Quantization and binary embeddings (≈ 4 h)

**Unlocks:** `core/scalar_type.hpp`, `data/binary_format.hpp`, `data/embedding_store.cpp`,
`python/zcgvs_tools/formats.py`, `tests/python/test_formats.py`,
`tests/cpp/test_quantized.cu`

**Concepts**

- Fixed-point / symmetric quantization; scale selection; **symmetric vs asymmetric** schemes
- Why quantizing only the corpus (not the query) is both more accurate and simpler
- IEEE 754 binary16 vs binary32 — range and precision, and where fp16 rounding bites
- **Binary quantization as 1-bit LSH**; why sign bits preserve angular information
- Hamming distance ↔ cosine similarity, and the `(D − 2·hamming)/D` mapping
- Bit packing: layout conventions, padding bits, why they must be zero
- Binary file formats: endianness, versioning, self-describing headers
- The **over-fetch-and-rescore** pattern from production vector databases

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| Whole post. This is exactly the technique implemented here, by the people who ship it. | HuggingFace, **"Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval"** — [huggingface.co/blog/embedding-quantization](https://huggingface.co/blog/embedding-quantization) | 30 min |
| Whole article — the rescoring/oversampling section especially | Qdrant, **"Binary Quantization"** — [qdrant.tech/articles/binary-quantization](https://qdrant.tech/articles/binary-quantization/) | 30 min |
| §"Scalar quantization" | Qdrant, **"Scalar Quantization"** — [qdrant.tech/articles/scalar-quantization](https://qdrant.tech/articles/scalar-quantization/) | 20 min |
| The bit-layout section only | **Half-precision floating-point format** — [en.wikipedia.org/wiki/Half-precision_floating-point_format](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) | 15 min |
| Skim for the *idea* that random hyperplane signs preserve angle | **Locality-sensitive hashing / SimHash** — [en.wikipedia.org/wiki/SimHash](https://en.wikipedia.org/wiki/SimHash) | 20 min |
| Read the whole thing — it is the spec this codebase implements twice, in two languages | **`docs/FORMATS.md`** (this repo) | 30 min |

**Checkpoint**

1. `quantize_int8` in `formats.py` computes `inverse_scale = 1/scale` and evaluates
   `rint(x * inverse_scale)`. The CPU test reference in `tests/cpp/cpu_reference.hpp` does the
   *same* thing rather than `rint(x / scale)`. Why does that distinction matter, and what
   would the symptom be if the two disagreed?
2. Binary rows pad up to a whole 32-bit word. Prove that a nonzero padding bit in the corpus
   would corrupt results, and identify the two places (one Python, one CUDA) that guarantee
   zeros.
3. `dim = 130` gives 5 words per row. Which word and which bit hold dimension 129? Verify
   against `test_pack_binary_bit_order_and_padding`.
4. `test_quantized.cu` uses tolerance `1e-3` for int8 but `1e-5` for binary. Justify both
   numbers from first principles.
5. A binary primary store plus a float32 rescore store uses *more* total VRAM than float32
   alone (33/32). So what is actually being saved? Answer in terms of what happens per query
   versus what happens once.
6. The embedding header stores `row_stride` even though it is derivable from dtype and dim.
   What class of bug does that redundancy catch?

---

## Tier 5 — Chaining device stages: the rescore path (≈ 1.5 h)

**Unlocks:** `cuda/rescore.cu`, the rescore branch of `search/searcher.cpp`,
`tests/cpp/test_pipeline.cu`

**Concepts**

- **Gather vs scatter** memory access patterns
- Indirection: one kernel's output buffer used directly as another's index list
- Index-space translation (candidate positions → corpus indices)
- Why keeping this on-device is the project's second zero-copy claim

No new external reading. Instead:

**Checkpoint**

1. Write out the four launches of the rescore path in order, and for each state which device
   buffer it reads and which it writes. Then state exactly where a naive implementation would
   have inserted a `cudaMemcpy` and a `cudaStreamSynchronize`.
2. Why is a separate `launchRemapIndices` kernel necessary? What index space does the second
   `launchTopK` operate in, and why can it not emit corpus indices directly?
3. `test_pipeline.cu` asserts an *exact* match against the CPU top-K for a float16 coarse pass
   but only structural properties for a binary coarse pass. Explain why testing exact recovery
   for the binary case would be a flaky test.
4. `setRescoreStore` rejects a binary store. Why is that restriction fundamental rather than
   an implementation shortcut?

---

## Tier 6 — Sparse retrieval, atomics, and BM25 (≈ 4 h)

**Unlocks:** `cuda/lexical.cu`, `data/lexical_index.{hpp,cpp}`,
`python/zcgvs_tools/build_lexical.py`, `tests/cpp/test_lexical.cu`

**Concepts**

- **CSR (compressed sparse row)** format: row pointers + column indices + values
- Inverted indexes: term → posting list
- **BM25**: idf, term-frequency saturation (`k1`), length normalization (`b`)
- Why the Lucene idf variant stays positive where the classic one goes negative
- `atomicAdd` on floats: correctness, contention, and **non-determinism from reordering**
- `atomicMax` on `__float_as_uint` — and the precondition that makes it valid
- Shared-memory tree reduction for a max
- Hybrid retrieval: score fusion vs reciprocal rank fusion
- Why score normalization is required before fusing two different score scales

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| Parts 1–3. The best practical explanation of BM25's parameters. Part 2's direct URL has rotted; follow the in-page link from Part 1. | Elastic, **"Practical BM25"** — [elastic.co/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch](https://www.elastic.co/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch) | 45 min |
| §3 (the BM25 derivation) only. Skip the probabilistic foundations unless curious. | Robertson & Zaragoza, **"The Probabilistic Relevance Framework: BM25 and Beyond"** — [staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) | 60 min |
| The CSR section | **Sparse matrix — CSR** — [en.wikipedia.org/wiki/Sparse_matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) | 15 min |
| §"Atomic Functions" | CUDA C++ Programming Guide | 20 min |
| PMPP Ch. 14 (sparse matrix computation) — the SpMV/CSR sections | *PMPP* 4th ed. | 60 min |
| Skim — how production systems combine lexical and vector search | Weaviate or Elastic docs on **hybrid search / RRF** (search "reciprocal rank fusion hybrid search") | 20 min |

**Checkpoint**

1. Every posting stores a fully precomputed BM25 weight. Write out what the query-time kernel
   would have to do if `idf`, `tf` and `|d|` were stored raw instead — and say which of those
   extra operations would hurt most on a GPU, and why.
2. `launchMaxPositive` uses `atomicMax` on the reinterpreted bit pattern of a float. State the
   precondition that makes this correct, prove it holds for BM25 scores, and give a concrete
   input for which the same trick would silently produce the wrong answer.
3. Masked-out documents carry `-FLT_MAX` in the dense scores. Prove that a masked document
   can never be resurrected by a strong BM25 match after `fuseScoresKernel` runs.
4. `test_lexical.cu` compares BM25 with a *relative* tolerance while the fusion check uses an
   absolute one. Explain the asymmetry.
5. Query terms are WordPiece ids from the same `vocab.txt` the engine tokenizes with. Name the
   classic hybrid-search failure mode this design choice eliminates.
6. The BM25 kernel launches one block per query term and skips `term < 0`. What does that
   padding buy, beyond tidiness? (Hint: Tier 7.)

---

## Tier 7 — CUDA Graphs (≈ 2.5 h)

**Unlocks:** the graph machinery in `search/searcher.cpp`, `GraphKey`,
`checkGraphEquivalence` in `tests/cpp/test_pipeline.cu`

**Concepts**

- Per-launch CPU overhead, and when it becomes a meaningful fraction of a query
- Stream capture → `cudaGraph_t` → `cudaGraphExec_t` → `cudaGraphLaunch`
- **What a captured graph bakes in**: pointers *and* by-value kernel arguments
- Cache invalidation — the hard part, and why the `GraphKey` has twelve fields
- Capture modes; what is illegal inside a capture region

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| Whole post | **"Getting Started with CUDA Graphs"** — [developer.nvidia.com/blog/cuda-graphs](https://developer.nvidia.com/blog/cuda-graphs/) | 30 min |
| Whole post — directly about the invalidation problem this repo solves with a key | **"Employing CUDA Graphs in a Dynamic Environment"** — [developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/) | 30 min |
| §"CUDA Graphs" | CUDA C++ Programming Guide | 40 min |

**Checkpoint**

1. `GraphKey` includes `float dense_weight`. Why is a *scalar tuning parameter* part of a
   graph-cache key? What exactly goes wrong if you drop it?
2. It also includes a `generation` counter bumped by `ensureCapacity`. What failure does that
   prevent that the pointer fields alone would not?
3. `captureGraph` calls `stream.synchronize()` before `cudaStreamBeginCapture`. What is being
   drained, and why must it not end up inside the graph?
4. Capture happens *outside* the `cudaEventRecord` bracket in `Searcher::search`. What would
   the reported `search_gpu_ms` measure if it were inside, and why would that be misleading?
5. `stats().graph_captures` climbing in step with `graph_launches` is a symptom. Name three
   distinct request-shape changes that would cause it.
6. Why is the cuBLAS backend excluded from graph capture in this implementation?

---

## Tier 8 — Vendor libraries and layout conventions (≈ 2 h)

**Unlocks:** `cuda/cublas_similarity.{hpp,cpp}`

**Concepts**

- BLAS levels 1/2/3; GEMV vs GEMM
- **Column-major vs row-major**, leading dimension (`lda`), and the transpose trick
- Why "beat the vendor library" is usually the wrong goal, and what the right goal is
- Handles, streams and workspaces in cuBLAS

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| §"cublas<t>gemm" and §"cublas<t>gemv" — the parameter tables specifically | **cuBLAS docs** — [docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/) | 40 min |
| Whole page — the definitive explanation of the row/column-major mapping | **"Row-major vs column-major"** — [en.wikipedia.org/wiki/Row-_and_column-major_order](https://en.wikipedia.org/wiki/Row-_and_column-major_order) | 20 min |
| PMPP Ch. 3 (multidimensional grids and data) | *PMPP* 4th ed. | 45 min |

**Checkpoint**

1. Derive, on paper, why
   `cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, batch, D, …, corpus, D, queries, D, …, scores, N)`
   produces exactly the row-major `[batch, N]` score matrix the top-K pass expects. Do not
   read the comment first.
2. `CublasSimilarity::supports` returns false for a mask, for L2, and for anything but
   float32. Explain each of the three, in terms of what a GEMM can and cannot express.
3. If cuBLAS beats the custom kernel on float32, is that a failure of this project? Give the
   argument you would actually make in an interview.

---

## Tier 9 — C++ architecture (≈ 4 h)

**Unlocks:** `engine/`, `rerank/`, `embed/`, `search/search_params.hpp`,
`bindings/python/module.cpp` structure

**Concepts**

- Abstract interfaces and **dependency inversion** — why the engine never includes an ONNX
  header
- Forward declaration + `unique_ptr` + out-of-line destructor (the incomplete-type rule)
- Composition roots
- `const` correctness, `mutable`, and why `Engine::searchBatch` is `const` yet mutates caches
- Parameter objects instead of ten-argument functions
- Sentinel-based option resolution (`< 0` means "use the config")
- Guaranteed copy elision (C++17) — why `const EmbeddingStore s = fromHostMatrix(...)` compiles
  with a deleted copy constructor

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| The "incomplete type" section | **cppreference**, [std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr) — note the requirements on `T`'s completeness at destruction | 20 min |
| Whole page | **Copy elision** — [en.cppreference.com/w/cpp/language/copy_elision](https://en.cppreference.com/w/cpp/language/copy_elision) | 20 min |
| Whole talk | Klaus Iglberger, **"Back to Basics: Designing Classes"** (CppCon) — search the CppCon channel | 60 min |
| Read `docs/ARCHITECTURE.md` §"Module responsibilities" and §"The Python/C++ boundary" | this repo | 30 min |

**Checkpoint**

1. `~Engine()` is declared in the header but defined as `= default` in the `.cpp`. Delete the
   `.cpp` definition and predict the exact compiler error. Then explain the rule.
2. `Engine::searchBatch` is `const`, calls `searcher_->search(...)` which is non-`const`, and
   writes to `host_terms_`. Explain both, precisely.
3. `SearchOptions` uses sentinels (`top_k <= 0`, `dense_weight < 0`) rather than
   `std::optional`. Argue both sides, then say which you would ship and why.
4. Trace how a `Reranker` gets constructed without `engine.cpp` ever including
   `onnxruntime_cxx_api.h`. Which file finally does include it?

---

## Tier 10 — The Python/C++ boundary (≈ 2 h)

**Unlocks:** `bindings/python/module.cpp`, `pyproject.toml`, `bindings/CMakeLists.txt`

**Concepts**

- pybind11: `py::class_`, `def_readwrite` vs `def_readonly`, `py::arg`, default arguments
- The **GIL**, and `py::call_guard<py::gil_scoped_release>` — when it is required and when it
  is a bug
- Automatic STL conversion via `pybind11/stl.h` — and that it *copies*
- Object lifetime and ownership across the boundary
- scikit-build-core: building a C++ extension inside a wheel

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| "First steps", "Object-oriented code", and "Functions → Keyword arguments / Default arguments" | **pybind11 docs** — [pybind11.readthedocs.io](https://pybind11.readthedocs.io/) | 60 min |
| §"Global Interpreter Lock" in the advanced section | pybind11 docs | 20 min |
| Skim the configuration reference | **scikit-build-core** — [scikit-build-core.readthedocs.io](https://scikit-build-core.readthedocs.io/) | 20 min |

**Checkpoint**

1. Every search method uses `py::call_guard<py::gil_scoped_release>`. Name a method in
   `module.cpp` where adding that guard would be an outright bug, and say why.
2. `SearchHit` uses `def_readonly` while `EngineConfig` uses `def_readwrite`. Justify the
   asymmetry from the API's semantics.
3. `BatchSearchResponse::results` is a `vector<vector<SearchHit>>`. What does pybind11 do with
   it, and what is the cost for a batch of 32 × 100 candidates?

---

## Tier 11 — The retrieval/ML side (≈ 4 h)

**Unlocks:** `text/tokenizer.cpp`, `embed/onnx_query_embedder.cpp`, `rerank/onnx_reranker.cpp`,
`onnx/onnx_session.cpp`, `python/benchmarks/eval_quality.py`

**Concepts**

- **Bi-encoder vs cross-encoder**, and why the two-stage retrieve-then-rerank pattern exists
- Mean pooling over token embeddings; L2 normalization; why cosine becomes a dot product
- **WordPiece** tokenization: greedy longest-match-first, the `##` continuation marker,
  `[CLS]` / `[SEP]` / `[PAD]`
- Sequence-pair encoding and `token_type_ids`
- ONNX as an interchange IR; execution providers; why TorchScript is the deprecated path
- IR evaluation: Recall@k, **nDCG@k**, qrels
- BEIR as a benchmark suite

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| Whole page — the clearest statement of why both encoder types exist | **SBERT: Cross-Encoders** — [sbert.net/examples/applications/cross-encoder/README.html](https://www.sbert.net/examples/applications/cross-encoder/README.html) | 20 min |
| §3 (model) and §4 (evaluation) | Reimers & Gurevych, **"Sentence-BERT"** — [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) | 45 min |
| The WordPiece chapter — implement the algorithm on paper for one word | **HuggingFace NLP Course, Ch. 6.6** — [huggingface.co/learn/nlp-course/chapter6/6](https://huggingface.co/learn/nlp-course/chapter6/6) | 40 min |
| §8.4 "Evaluation of ranked retrieval results" — DCG and nDCG | Manning et al., **Introduction to Information Retrieval** — [nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) | 40 min |
| §2 (the benchmark's design) and Table 1 | Thakur et al., **"BEIR"** — [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) | 40 min |
| The CUDA execution provider page | **ONNX Runtime docs** — [onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | 25 min |

**Checkpoint**

1. `OnnxQueryEmbedder::embed` mean-pools then L2-normalizes. Show that after normalization,
   cosine similarity and dot product are the same operation — and find the flag in the
   embedding file header that lets the engine rely on this.
2. Hand-trace `wordpieceInto` on a word not in the vocabulary. How many `lookup` calls does
   the greedy longest-match-first loop make in the worst case for an `n`-character word?
3. `encodePair` assigns `token_type_ids` of 0 and 1. What does the cross-encoder do with them,
   and why does the bi-encoder pass all zeros?
4. `eval_quality.py` reports both nDCG@10 and a "GPU/CPU overlap". They measure different
   things. What can overlap tell you that nDCG cannot, and vice versa?
5. Why can the reranker only improve results *within* the candidate set, and what does that
   imply about choosing `candidates`?

---

## Tier 12 — Build system, testing and profiling (≈ 3 h)

**Unlocks:** all `CMakeLists.txt`, `cmake/FindONNXRuntime.cmake`, `tests/`, `core/nvtx.hpp`,
`scripts/`

**Concepts**

- Modern CMake: targets not variables; `PUBLIC` vs `PRIVATE` vs `INTERFACE`
- `find_package`, `IMPORTED` targets, writing a `FindXxx.cmake`
- CUDA as a first-class CMake language; `CMAKE_CUDA_ARCHITECTURES`
- `compile_commands.json` and how editors consume it
- **Reference-implementation testing** and choosing tolerances
- Floating-point **non-associativity** — why GPU and CPU sums differ, legitimately
- NVTX ranges; reading an Nsight Systems timeline
- Bash: `set -euo pipefail`, and why `verify.sh` deliberately omits `-e`

**Resources**

| Take | Resource | Time |
| --- | --- | --- |
| "Introduction", "Targets", "Properties". Stop there. | **An Introduction to Modern CMake** — [cliutils.gitlab.io/modern-cmake](https://cliutils.gitlab.io/modern-cmake/) | 60 min |
| Whole post | NVIDIA, **"Building Cross-Platform CUDA Applications with CMake"** — [developer.nvidia.com/blog/building-cuda-applications-cmake](https://developer.nvidia.com/blog/building-cuda-applications-cmake/) | 25 min |
| The "Profiling from the CLI" and "Timeline View" sections | **Nsight Systems User Guide** — [docs.nvidia.com/nsight-systems/UserGuide](https://docs.nvidia.com/nsight-systems/UserGuide/) | 40 min |
| The README | **NVTX** — [github.com/NVIDIA/NVTX](https://github.com/NVIDIA/NVTX) | 15 min |
| Whole post — the definitive treatment of float comparison tolerances | Bruce Dawson, **"Comparing Floating Point Numbers, 2012 Edition"** — [randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/) | 40 min |

**Checkpoint**

1. `zcgvs_core` links `CUDA::cudart` and `ONNXRuntime::ONNXRuntime` as `PUBLIC` but sets
   `POSITION_INDEPENDENT_CODE` as a property. Why `PUBLIC`, and what would break with
   `PRIVATE`?
2. `verify.sh` uses `set -uo pipefail` — deliberately without `-e`. What behaviour does that
   enable, and why is it the right choice for this specific script?
3. GPU and CPU produce different last bits for the same dot product. Explain the mechanism,
   and then justify why `test_pipeline.cu` is nonetheless allowed to demand *bit-identical*
   results for the CUDA-graph comparison.
4. `ZCGVS_NVTX_RANGE` compiles to nothing when the option is off. Show how, and explain why
   the `__LINE__` concatenation in the macro is necessary.

---

## Reading the codebase: three passes

Do not read files in alphabetical order. Do three deliberate passes.

**Pass 1 — the spine (2 h).** Follow one query end to end, ignoring every branch:

```
python/examples/quickstart.py
  → bindings/python/module.cpp          (Engine::searchTimed binding)
  → src/engine/engine.cpp               (searchBatch: embed → params → search → assemble)
  → src/search/searcher.cpp             (recordPipeline)
  → src/cuda/similarity.cu              (the scan)
  → src/cuda/topk.cu                    (selection)
  → src/rerank/onnx_reranker.cpp        (rerank)
```

Goal: know *where things happen*. Do not try to understand the kernels yet.

**Pass 2 — bottom up (8 h).** Now read for real, in dependency order, so nothing forward-
references:

```
core/scalar_type.hpp → core/cuda_check.hpp → core/device_buffer.hpp
  → core/pinned_buffer.hpp → core/cuda_stream.hpp → core/cuda_context.*
  → cuda/decode.cuh → cuda/reduction.cuh → cuda/bitonic.cuh
  → cuda/convert.cu → cuda/similarity.cu → cuda/topk.cu
  → cuda/rescore.cu → cuda/lexical.cu → cuda/cublas_similarity.cpp
  → data/binary_format.hpp → data/embedding_store.cpp → data/corpus.cpp
  → data/lexical_index.cpp
  → search/search_params.hpp → search/searcher.cpp
  → text/tokenizer.cpp → onnx/onnx_session.cpp → embed/ → rerank/
  → engine/engine.cpp → bindings/python/module.cpp
```

**Pass 3 — the mirror (3 h).** Every C++ file that touches disk or bits has a Python
counterpart. Read them side by side and confirm they agree:

| C++ | Python |
| --- | --- |
| `data/binary_format.hpp` | `zcgvs_tools/formats.py` |
| `cuda/convert.cu` (`launchConvertF32ToI8`) | `formats.quantize_int8` |
| `cuda/convert.cu` (`launchPackBinary`) | `formats.pack_binary` |
| `cuda/lexical.cu` | `zcgvs_tools/build_lexical.py` |
| `text/tokenizer.cpp` | the fallback tokenizer in `build_lexical.py` |
| `tests/cpp/test_quantized.cu` | `tests/python/test_formats.py` |

If you can explain why each pair must agree and what breaks if they drift, you understand the
riskiest part of this repo.

---

## Experiments: break it and predict the failure

Understanding is proven by prediction. For each, **write down your prediction first**, then
run it, then revert.

| # | Change | Predict |
| --- | --- | --- |
| 1 | In `topk.cu`, set `kMergeTile = 1024` | Which test hangs or fails, and at what `k`? |
| 2 | In `similarity.cu`, remove `#pragma unroll` from the accumulation loop | Direction and rough size of the throughput change |
| 3 | In `formats.py`, flip `pack_binary` to set bits on `< 0` instead of `>= 0` | Which test fails first — the Python one or the CUDA one? Why? |
| 4 | In `searcher.cpp`, delete `dense_weight` from `GraphKey` | Construct a two-query sequence that returns wrong results |
| 5 | In `lexical.cu`, change `atomicAdd` to a plain `+=` | Will it crash, or silently under-count? |
| 6 | In `engine.cpp`, remove `stream_->synchronize()` implicitly by returning before `copyResultToHost` | What do you read out of the buffers? |
| 7 | Set `rescore_candidates` equal to `top_k` | What quality change, and why is it *almost* pointless? |
| 8 | Build with `-DZCGVS_ENABLE_NVTX=ON` and profile with and without `--graphs` | What visually changes on the Nsight timeline? |

Experiment 4 is the best interview story in the list. Make sure you can tell it.

---

## Time budget

| Tier | Topic | Hours |
| ---: | --- | ---: |
| 0 | GPU mental model | 4 |
| 1 | CUDA plumbing + RAII | 4 |
| 2 | The scan kernel | 5 |
| 3 | Bitonic selection | 5 |
| 4 | Quantization + formats | 4 |
| 5 | Chained device stages | 1.5 |
| 6 | Sparse + atomics + BM25 | 4 |
| 7 | CUDA Graphs | 2.5 |
| 8 | cuBLAS + layouts | 2 |
| 9 | C++ architecture | 4 |
| 10 | Python boundary | 2 |
| 11 | Retrieval/ML | 4 |
| 12 | Build, test, profile | 3 |
| — | Three code-reading passes | 13 |
| — | Experiments | 4 |
| | **Total** | **≈ 62 h** |

Concepts and code reading overlap, so the realistic figure is **40–45 hours** if you read each
file immediately after its tier rather than as a separate block. That is the schedule this
document is designed around.

### If you only have one weekend

Tiers 0, 2, 3 and the Pass 1 spine. That covers the scan kernel, the selection kernel, and
where they sit — which is 80% of what anyone will ask you about, because the custom kernels
are the actual content.

---

## The four conversations this repo earns you

Every one of these is a distinct MLSys interview thread. Prepare them as stories, with a
number attached:

1. **"Why is Top-K harder than a dot product?"** Reduction is associative and order-free;
   selection depends on a global ordering. Walk through the recursive bitonic merge and the
   termination argument (Tier 3).
2. **"What did you do about host↔device transfers?"** Two transfers removed, not one — the
   score vector, and the quantized-rescore round trip. Show the profile (Tiers 5, 12).
3. **"How do you make a bandwidth-bound kernel faster?"** You do not tune it — you read fewer
   bytes (quantization) or reuse each byte more (query tiling). Both, with the roofline
   numbers (Tiers 2, 4).
4. **"How do you know it is correct?"** Every kernel against a CPU reference that reproduces
   the exact arithmetic; the same wire-format contract tested from both languages; the
   float16-coarse case pinned to an exact CPU ranking (Tiers 4, 12).

---

## Link index

Everything above, in one place. Links were correct at the time of writing; where one has
rotted, the **bolded title** is the search term.

**GPU fundamentals**
- **How GPU Computing Works** — https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/
- **CUDA C++ Programming Guide** — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA C++ Best Practices Guide** — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **CUDA Programming Course – High-Performance Computing with GPUs** (freeCodeCamp) — https://www.youtube.com/watch?v=86FAWCzIe_4
- **GPU MODE** lecture series — https://www.youtube.com/@GPUMODE
- *Programming Massively Parallel Processors*, 4th ed. — Hwu, Kirk, El Hajj

**Memory, reductions, kernels**
- **How to Access Global Memory Efficiently in CUDA C/C++ Kernels** — https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
- **Using Shared Memory in CUDA C/C++** — https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
- **Optimizing Parallel Reduction in CUDA** — https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- **Faster Parallel Reductions on Kepler** — https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
- **CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops** — https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
- **How to Optimize Data Transfers in CUDA C/C++** — https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- **How to Overlap Data Transfers in CUDA C/C++** — https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/

**Sorting and selection**
- **CUDA bitonic sort walkthrough** — https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/cuda_bitonic.html
- **Bitonic sorter** — https://en.wikipedia.org/wiki/Bitonic_sorter
- **Efficient Top-K Query Processing on Massively Parallel Hardware** — https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf

**Quantization**
- **Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval** — https://huggingface.co/blog/embedding-quantization
- **Binary Quantization** (Qdrant) — https://qdrant.tech/articles/binary-quantization/
- **Scalar Quantization** (Qdrant) — https://qdrant.tech/articles/scalar-quantization/
- **Half-precision floating-point format** — https://en.wikipedia.org/wiki/Half-precision_floating-point_format
- **SimHash** — https://en.wikipedia.org/wiki/SimHash

**Sparse retrieval**
- **Practical BM25** (Elastic) — https://www.elastic.co/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch
- **The Probabilistic Relevance Framework: BM25 and Beyond** — https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
- **Sparse matrix — CSR** — https://en.wikipedia.org/wiki/Sparse_matrix

**CUDA Graphs and cuBLAS**
- **Getting Started with CUDA Graphs** — https://developer.nvidia.com/blog/cuda-graphs/
- **Employing CUDA Graphs in a Dynamic Environment** — https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/
- **cuBLAS documentation** — https://docs.nvidia.com/cuda/cublas/
- **Row- and column-major order** — https://en.wikipedia.org/wiki/Row-_and_column-major_order

**C++ and build**
- **cppreference: RAII** — https://en.cppreference.com/w/cpp/language/raii
- **cppreference: rule of three/five/zero** — https://en.cppreference.com/w/cpp/language/rule_of_three
- **cppreference: copy elision** — https://en.cppreference.com/w/cpp/language/copy_elision
- **Back to Basics: Move Semantics** (Klaus Iglberger, CppCon) — https://www.youtube.com/watch?v=St0MNEU5b0o
- **An Introduction to Modern CMake** — https://cliutils.gitlab.io/modern-cmake/
- **Building Cross-Platform CUDA Applications with CMake** — https://developer.nvidia.com/blog/building-cuda-applications-cmake/
- **pybind11 documentation** — https://pybind11.readthedocs.io/
- **scikit-build-core** — https://scikit-build-core.readthedocs.io/
- **Comparing Floating Point Numbers, 2012 Edition** — https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

**Retrieval and evaluation**
- **SBERT: Cross-Encoders** — https://www.sbert.net/examples/applications/cross-encoder/README.html
- **Sentence-BERT** — https://arxiv.org/abs/1908.10084
- **HuggingFace NLP Course, Ch. 6.6 (WordPiece)** — https://huggingface.co/learn/nlp-course/chapter6/6
- **Introduction to Information Retrieval §8.4** — https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
- **BEIR** — https://arxiv.org/abs/2104.08663
- **ONNX Runtime CUDA Execution Provider** — https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

**Profiling**
- **Nsight Systems User Guide** — https://docs.nvidia.com/nsight-systems/UserGuide/
- **NVTX** — https://github.com/NVIDIA/NVTX
