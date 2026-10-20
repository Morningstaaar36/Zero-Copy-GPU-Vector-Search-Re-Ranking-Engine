# Rebuild path — write this engine yourself, from scratch

This is not a reading list. It is a **build order**. Every stage gives you the minimum theory
to write the next piece, then you write it, then you prove it works. At the end you will have
typed every line of this engine and be able to defend all of it as your own.

**Who this is for:** you know C and basic C++ (functions, classes, pointers, `for` loops). You
have never written CUDA. You do not know what an embedding is. That is fine — everything else
is introduced exactly when you need it, and never earlier.

**The rule that makes this work:** never finish a course. Every resource below has a *Take*
column saying precisely what to extract. Take that and leave. If a video is 50 minutes and the
Take says "first 20 minutes", watch 20 minutes and stop. You are learning to build one specific
thing, not to pass an exam.

**Total: ≈ 110 hours.** At 2 hours an evening that is about 4 months. Full-time, about 3 weeks.
Roughly 35 hours is theory, 75 is you typing code.

---

## How to run this

Work in a **separate directory** from this repo. Call it `myengine/`. You are rebuilding, not
editing.

```bash
mkdir ~/myengine && cd ~/myengine && git init
```

Keep this repo open in a second window as the **answer key**. The honest way to use it:

1. Read the stage's theory Take.
2. Write the file yourself, from the spec in the stage.
3. **Only then** open the reference file and compare.
4. Where you differ, decide which is better. Sometimes yours will be. Write down why.

If you copy first and understand later you will not be able to defend it, and you will know.

Every stage ends with a **Ship it** — a command that must print the right answer. Do not move
on until it does. This is the whole discipline: you always have something that runs.

---

## Stage 0 — See the finished thing, then delete it from your mind (1 h)

You already have this repo built and working. Run it once so you know what you are aiming at:

```bash
bash scripts/run_demo.sh
./build/apps/zcgvs_bench_kernels --docs 1000000 --dim 384 --k 100
```

Watch `gpu_search` come back at ~0.2 ms and the scan hit ~205 GB/s. That number is the target.

**Then write down, in one paragraph, what you think the program did.** Keep it. At Stage 12 you
will read it again and see how wrong it was — that is the clearest measure of progress you will
get.

**Ship it:** a paragraph in `myengine/NOTES.md`.

---

## Stage 1 — The GPU mental model (4 h, all video)

You cannot write a good kernel while thinking "GPU = fast CPU". Fix that first. This stage has
no code.

| Take | Resource | Time |
| --- | --- | --- |
| **The whole talk.** The single highest-value hour in GPU computing. | Stephen Jones, *How GPU Computing Works* — [nvidia.com/on-demand](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/) | 50 min |
| **First 90 minutes only** — up to and including shared memory. Stop there; come back at Stage 3. | freeCodeCamp, *CUDA Programming Course* — [youtube.com/watch?v=86FAWCzIe_4](https://www.youtube.com/watch?v=86FAWCzIe_4) | 90 min |
| Puzzles **1–8 only**. Python, runs in a notebook, no CUDA install. This teaches thread indexing without C++ syntax in the way. | *GPU Puzzles* — [github.com/srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | 90 min |

**The five things you must be able to say out loud:**

1. Threads run in **warps of 32**, in lockstep. If they branch differently, both paths run.
2. Blocks are independent. **Blocks cannot talk to each other.** Only threads within a block can.
3. A GPU hides memory latency by having thousands of threads ready, not by caching.
4. **Coalescing:** when the 32 threads of a warp read 32 *consecutive* addresses, that is one
   memory transaction. Strided reads cost up to 32×.
5. **Memory-bound vs compute-bound.** Most of this engine is memory-bound: the answer to "make
   it faster" is almost always *read fewer bytes* or *reuse each byte more*, never *optimize
   the arithmetic*.

**Ship it:** explain #4 to someone (or a rubber duck) without notes.

---

## Stage 2 — Your first kernel, and C++ RAII (8 h)

### Theory first (2 h)

| Take | Resource | Time |
| --- | --- | --- |
| The whole post. Written for exactly your starting point. | NVIDIA, *An Even Easier Introduction to CUDA* — [developer.nvidia.com/blog/even-easier-introduction-cuda](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) | 40 min |
| Just the page on **destructors**. | learncpp — [destructors](https://www.learncpp.com/cpp-tutorial/destructors/) | 20 min |
| **Only** "Intro to smart pointers / move semantics" and the `std::unique_ptr` page. Skip the rest of the chapter. | learncpp — [smart pointers & move semantics](https://www.learncpp.com/cpp-tutorial/introduction-to-smart-pointers-move-semantics/) | 45 min |

The one C++ idea that matters here: **RAII**. A class grabs a resource in its constructor and
releases it in its destructor. You cannot leak, because leaving scope runs the destructor. GPU
memory is a resource. So `cudaMalloc` goes in a constructor and `cudaFree` in a destructor, and
then you never write `cudaFree` again.

A buffer that owns GPU memory must **not** be copyable — two objects freeing the same pointer
is a double-free crash. So you `= delete` the copy constructor and write a move constructor
instead. That is the whole of Stage 2's C++ content.

### Build (6 h)

Write these four files, in this order:

**1. `cuda_check.hpp`** — a macro wrapping every CUDA call so errors throw instead of being
silently ignored.

```cpp
#define MY_CUDA_CHECK(expr) do {                                  \
    cudaError_t err = (expr);                                     \
    if (err != cudaSuccess) throw std::runtime_error(             \
        std::string("cuda: ") + cudaGetErrorString(err));         \
} while (0)
```

The `do { … } while (0)` is so the macro behaves like a statement inside an `if` without braces.
Add a second macro `MY_CUDA_CHECK_LAST()` calling `cudaGetLastError()`. **Kernel launches are
asynchronous and do not return errors** — you must check separately, after the launch.

**2. `device_buffer.hpp`** — RAII around `cudaMalloc`/`cudaFree`. Constructor allocates,
destructor frees, copy deleted, move implemented. Methods: `data()`, `size()`, `copyFromHost()`,
`copyToHost()`.

**3. `vector_add.cu`** — the hello world. Add two arrays of a million floats on the GPU. Use
your `DeviceBuffer` for all three arrays. Verify every element on the CPU.

**4. `dot_product.cu`** — one block, one dot product of two 384-element vectors. Do it the naive
way: each thread multiplies one pair into shared memory, then thread 0 loops over shared memory
and sums. **It will be correct and slow.** That is deliberate — Stage 3 makes it fast, and you
need the slow version to measure against.

**Ship it:**

```bash
nvcc -std=c++17 -arch=native dot_product.cu -o dot && ./dot
# must print a dot product matching your CPU loop to within 1e-4
```

**Answer key:** `include/zcgvs/core/cuda_check.hpp`, `core/device_buffer.hpp`.

**Predict-then-check:** before running, predict what happens if you delete the destructor's
`cudaFree`. Then run `nvidia-smi` in a loop while your program runs 1000 times.

---

## Stage 3 — The scan kernel: this is the heart (14 h)

This is the most important stage in the document. Budget real time. Everything else in the
engine is scaffolding around this kernel.

### Theory (4 h)

| Take | Resource | Time |
| --- | --- | --- |
| Whole post. | Mark Harris, *How to Access Global Memory Efficiently* — [developer.nvidia.com/blog](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/) | 20 min |
| Whole post. | Mark Harris, *Using Shared Memory in CUDA C/C++* — [developer.nvidia.com/blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) | 25 min |
| **Slides 1–35.** The seven successive optimizations of a reduction. This is the classic; read it slowly. | *Optimizing Parallel Reduction in CUDA* — [reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) | 60 min |
| Whole post. This *is* the `blockReduceSum` you are about to write. | *Faster Parallel Reductions on Kepler* — [developer.nvidia.com/blog](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) | 25 min |
| The **reductions** lecture only. | GPU MODE — [youtube.com/@GPUMODE](https://www.youtube.com/@GPUMODE) | 60 min |

The key trick is the **warp shuffle**: `__shfl_down_sync` lets threads in a warp swap registers
directly, no shared memory, no `__syncthreads()`. So a block reduction becomes two levels —
shuffle within each warp, then one small shared-memory pass across warps.

### Build (10 h)

**1. `reduction.cuh`** — `warpReduceSum` then `blockReduceSum`. Test it standalone: sum
1..1024 in one block, compare against `n(n+1)/2`.

**2. `similarity.cu`, float32 only.** One block per document. The loop that matters:

```cpp
for (int t = threadIdx.x; t < dim; t += blockDim.x)   // coalesced
```

Consecutive threads touch consecutive addresses. Write it, then write the wrong version
(`t = threadIdx.x * chunk`, each thread a contiguous slice), and **measure both**. The gap is
your Stage 1 lesson made physical.

**3. Benchmark it.** Compute achieved GB/s = `bytes_read / seconds`, and theoretical peak =
`2 × memory_clock_khz × bus_width_bits / 8`. The factor of 2 is DDR — double data rate,
transfers on both clock edges. On your RTX 5060 Ti peak is **448 GB/s**. Getting 40–50% on a
first correct kernel is normal and good.

**4. Query tiling — the part that makes this project interesting.** Right now one corpus read
serves one query, so you are purely bandwidth-bound and can never beat memory. Stage up to 8
query vectors in shared memory and score all of them against each corpus row you load. Each
loaded byte now does 8× the work. Re-measure.

**5. Make it persistent.** Now run the benchmark on fp16 and int8 too. If they are no faster
than fp32, your kernel is not bandwidth-bound any more — it is paying a fixed cost per block.
With one block per document, every block re-stages the whole query tile: at 1M documents that
is ~12 GB of staging to scan a 0.4 GB int8 corpus. Size the grid to the GPU's residency
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor × SM count`), give each *warp* a document,
stride the warps over the corpus, and load rows in 128-bit vectors. Every precision should then
land within a few percent of peak, and fp32 should tie `cublasSgemv`.

This is the difference between a demo and an engineered kernel, and it is the thing to talk
about when someone asks what you built.

**Ship it:** your scan over 1M×384 float32, checked against a CPU reference, printing GB/s and
% of peak.

**Answer key:** `src/cuda/similarity.cu`, `include/zcgvs/cuda/reduction.cuh`.

---

## Stage 4 — Four storage precisions, and a file format (12 h)

Now make the corpus smaller. This is the "read fewer bytes" half of going faster.

### Theory (3 h)

| Take | Resource | Time |
| --- | --- | --- |
| Whole post. Exactly this technique, by the people who ship it. | HuggingFace, *Binary and Scalar Embedding Quantization* — [huggingface.co/blog/embedding-quantization](https://huggingface.co/blog/embedding-quantization) | 30 min |
| The **rescoring / oversampling** section especially. | Qdrant, *Binary Quantization* — [qdrant.tech/articles/binary-quantization](https://qdrant.tech/articles/binary-quantization/) | 30 min |
| **Only** "Function templates" and "Template non-type parameters". | learncpp — [function templates](https://www.learncpp.com/cpp-tutorial/function-templates/) | 60 min |
| Read this repo's `docs/FORMATS.md` — it is the spec you are implementing twice. | this repo | 30 min |

The ideas, compressed:

- **int8:** pick one scale for the whole matrix, store `round(x / scale)` as a byte. 4× smaller.
- **binary:** keep only the **sign bit** of each dimension, 32 dims per `uint32`. 32× smaller.
  Similarity becomes `__popc(a ^ b)` — count differing bits. Fewer differing bits = more similar.
  Map to a cosine-like score with `(D − 2·hamming) / D`.
- Only the **corpus** is quantized; the query stays float32. Asymmetric, more accurate, simpler.

### Build (9 h)

**1. `scalar_type.hpp`** — an `enum class` with an explicit underlying type (it gets written to
disk, so the width must be pinned), plus `bytesPerRow(dtype, dim)`. Note bit packing needs
ceiling division: `(dim + 31) / 32`.

**2. `formats.py`** — the Python writer: a self-describing header (magic number, version, dtype,
doc count, dim, row stride, scale) then the matrix. Include the magic so a wrong file fails
loudly instead of producing garbage.

**3. `convert.cu`** — the CUDA converters, `float32 → int8` and `float32 → binary`.

**4. Template your scan.** One kernel, four storage types, decoding to float in registers. This
is why you read the templates page.

**Now the trap, and it is the most valuable bug in this project.** Your Python writer and your
CUDA converter implement the *same* contract in two languages. If they disagree by one bit —
say one sets bits on `>= 0` and the other on `> 0` — everything still runs and results are
quietly wrong. So write **both** tests: a Python round-trip and a CUDA-vs-CPU check. A change to
one side without the other is exactly the bug that pair exists to catch.

**Ship it:** all four dtypes scanning correctly; print MB and GB/s per dtype. Binary should be
1/32 the bytes of float32.

**Answer key:** `core/scalar_type.hpp`, `python/zcgvs_tools/formats.py`, `cuda/convert.cu`,
`tests/python/test_formats.py`.

---

## Stage 5 — Top-K selection (12 h)

You have `N` scores on the device. You need the best `k`. Copying all `N` to the CPU and sorting
is what a naive stack does — and it is the transfer this whole project exists to delete.

Selection is genuinely harder than reduction: a sum is associative and order-free, but "the top
100" depends on a **global ordering**. That framing is the answer to the most common interview
question about this repo.

### Theory (3.5 h)

| Take | Resource | Time |
| --- | --- | --- |
| **Trace an 8-element network by hand, on paper.** Do not skip this — 20 minutes of drawing replaces hours of confusion. | *Bitonic sorter* — [en.wikipedia.org/wiki/Bitonic_sorter](https://en.wikipedia.org/wiki/Bitonic_sorter) | 45 min |
| The whole page, the clearest CUDA-specific walkthrough. | Rutgers, *CUDA bitonic sort* — [people.cs.rutgers.edu](https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/cuda_bitonic.html) | 40 min |
| **§1–4 only.** | Shanbhag et al., *Efficient Top-K on Massively Parallel Hardware* — [doc.ic.ac.uk](https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf) | 60 min |

A sorting network is a **fixed** sequence of compare-exchange operations — same comparisons
regardless of the data, which is exactly what a GPU wants. Two tricks carry the implementation:
partner index is `i ^ step` (XOR), and sort direction comes from `(i & stage) == 0`.

### Build (8.5 h)

**1. `bitonic.cuh`** — sort 1024 elements in shared memory. Pad to a power of two with `-INFINITY`
and index `-1`.

**2. `topk.cu`, single pass** — each block sorts a 1024-tile and writes its top-`k`.

**3. Then the recursive merge.** One pass leaves you with `k × num_blocks` candidates, still too
many. So merge: blocks load **4096**-element tiles of survivors, sort, emit top-`k` each,
ping-ponging between two scratch buffers until one block remains.

Why 4096 and not 1024? Because a merge tile must be **strictly larger** than `k` or the set
never shrinks and you loop forever. With tile 4096 and `k ≤ 1024` each pass shrinks the
candidate set by at least 4×, so the recursion provably terminates. **This termination argument
is the thing to be able to state.** Set the tile to 1024 and watch it hang — then you own it.

**Ship it:** top-100 of 1,000,000 random floats, matching `std::partial_sort` on the CPU exactly.

**Answer key:** `cuda/bitonic.cuh`, `cuda/topk.cu`, `tests/cpp/test_topk_large.cu`.

---

## Stage 6 — Chaining stages on-device: the rescore path (6 h)

No new external reading. This stage is the project's actual thesis.

Binary storage is 32× smaller but lossy. The fix: scan **binary** to get 512 rough candidates,
then re-score just those 512 against a **float32** copy, and take the real top-10 from that.

The point is *how* the two stages connect. The top-K kernel wrote its winning indices into a
device buffer. The rescore kernel reads **that same buffer** directly as its gather list. No
copy to the host, no synchronize, no round trip. The hand-off between "coarse" and "exact" is a
**device pointer**.

Write the four launches and, for each, name the buffer it reads and the buffer it writes:

1. scan (binary) → scores
2. top-K → candidate indices
3. rescore (gather those rows from the float32 store) → new scores
4. top-K again → **positions within the candidate set**, then a remap kernel turning those back
   into corpus indices

Step 4's remap is the subtle one. The second top-K works in candidate-index space (0..511), not
corpus space, so it cannot emit corpus ids directly. You need the extra kernel.

**Ship it:** binary + rescore returning results measurably closer to the float32 ranking than
binary alone. Measure overlap@10 against your float32 result — expect a jump from roughly 0.6 to
0.95.

**Answer key:** `cuda/rescore.cu`, the rescore branch of `search/searcher.cpp`.

---

## Stage 7 — BM25 and atomics (10 h)

Dense vectors miss exact keyword matches. Add a classical lexical score and fuse the two.

### Theory (3 h)

| Take | Resource | Time |
| --- | --- | --- |
| Parts 1–3. Best practical explanation of BM25's knobs. (Part 2's direct URL has rotted — follow the in-page link from Part 1.) | Elastic, *Practical BM25* — [part 1](https://www.elastic.co/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch) | 45 min |
| The **CSR** section only. | *Sparse matrix* — [en.wikipedia.org/wiki/Sparse_matrix](https://en.wikipedia.org/wiki/Sparse_matrix) | 15 min |
| §"Atomic Functions". | CUDA C++ Programming Guide — [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 20 min |

BM25 scores a document by term frequency, damped so the 20th occurrence adds little (`k1`), and
normalized by document length (`b`), weighted by how rare the term is (idf).

### Build (7 h)

**1. `build_lexical.py`** — an inverted index in **CSR**: term → list of (doc, weight).
**Precompute the entire BM25 weight offline.** Then the GPU does a pure scatter-add at query
time, no arithmetic. Doing the idf/tf/length math per-posting on the device would be far worse.

**2. `lexical.cu`** — one block per query term, threads stride over that term's postings,
`atomicAdd` into the score vector. Then `atomicMax` (on the reinterpreted bit pattern) to find
the maximum for normalization, then an element-wise fuse into the dense scores.

**Two things worth understanding rather than copying.** `atomicMax` on a float's bit pattern
only works because positive IEEE-754 floats compare correctly as integers — it silently breaks
on negatives. And `atomicAdd` makes results **non-deterministic**: floating-point addition is
not associative, so a different accumulation order changes the last bits and can flip documents
whose scores are nearly tied. That is inherent, not a bug — but you must be able to say it.

**3. Fuse and evaluate.** Normalize both score vectors, blend with `dense_weight`.

**Ship it:** hybrid returning different results from dense-only, and a measured Recall@10 for
both. **Be ready for hybrid to be *worse*** — on SciFact, `dense_weight = 0.7` scores 0.7940
against 0.8240 dense-only. Report it honestly and sweep the weight. Reporting a negative result
you understand is worth more than a positive one you don't.

**Answer key:** `cuda/lexical.cu`, `python/zcgvs_tools/build_lexical.py`.

---

## Stage 8 — CUDA graphs (6 h)

Your pipeline is now ~6 kernel launches per query. Each launch costs CPU time, and when kernels
are this short that overhead is a real fraction of the query.

| Take | Resource | Time |
| --- | --- | --- |
| Whole post. | *Getting Started with CUDA Graphs* — [developer.nvidia.com/blog/cuda-graphs](https://developer.nvidia.com/blog/cuda-graphs/) | 30 min |
| Whole post — directly about the invalidation problem you are about to hit. | *Employing CUDA Graphs in a Dynamic Environment* — [developer.nvidia.com/blog](https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/) | 30 min |

Capture the whole sequence once into a graph, then replay it with a single launch.

**The hard part is not capture — it is cache invalidation.** A captured graph bakes in device
pointers *and* by-value kernel arguments. Change the batch size, reallocate a buffer, or change
`dense_weight`, and replaying the old graph silently computes the **wrong answer**. So you need
a `GraphKey` of every input that got baked in, and you recapture when it changes.

Build it, then deliberately drop `dense_weight` from the key and construct a two-query sequence
that returns wrong results. **That bug is the best story this project gives you** — it is a real
production failure mode and it is entirely yours once you have caused it on purpose.

**Ship it:** `graph_replay = true` on the second identical query, bit-identical results to the
direct path, and captures staying at 1 while launches climb.

**Answer key:** the graph machinery in `search/searcher.cpp`, `checkGraphEquivalence` in
`tests/cpp/test_pipeline.cu`.

---

## Stage 9 — Compare against cuBLAS (4 h)

Your scan is a matrix-vector product. NVIDIA ships a hand-tuned one. Wire it up as a second
backend and race it.

| Take | Resource | Time |
| --- | --- | --- |
| The `gemm`/`gemv` **parameter tables**. | cuBLAS docs — [docs.nvidia.com/cuda/cublas](https://docs.nvidia.com/cuda/cublas/) | 40 min |
| Whole page. cuBLAS is **column-major**; your data is row-major. This is the whole difficulty. | *Row- and column-major order* — [wikipedia](https://en.wikipedia.org/wiki/Row-_and_column-major_order) | 20 min |

**cuBLAS will probably win on float32** — on this machine 416 GB/s against 205. That is not a
failure, and the reason matters: cuBLAS only handles float32, no mask, inner-product metrics
only. Your kernel does int8, binary, masking, L2, and fuses BM25 — none of which a GEMM can
express. **Having the honest comparison is the mark of an engineer; omitting it is the mark of
someone hiding from it.**

**Ship it:** both backends, same ranking, scores agreeing to ~1e-7, GB/s for each.

**Answer key:** `cuda/cublas_similarity.cpp`.

---

## Stage 10 — Text in, results out (12 h)

Everything so far assumed vectors appear by magic. Now handle real queries.

### Theory (4 h)

| Take | Resource | Time |
| --- | --- | --- |
| Whole page. The clearest statement of why two model types exist. | SBERT, *Cross-Encoders* — [sbert.net](https://www.sbert.net/examples/applications/cross-encoder/README.html) | 20 min |
| The WordPiece chapter. **Tokenize one word on paper.** | HF NLP Course 6.6 — [huggingface.co](https://huggingface.co/learn/nlp-course/chapter6/6) | 40 min |
| §8.4, DCG and nDCG. You cannot claim quality without this. | *Introduction to Information Retrieval* — [nlp.stanford.edu](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) | 40 min |
| The CUDA execution provider page. | ONNX Runtime — [onnxruntime.ai/docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | 25 min |

The two-stage pattern, which is the entire architecture of modern retrieval:

- **Bi-encoder** encodes query and document *separately* into vectors. Documents can be encoded
  once, offline. Fast enough to scan millions — this is what your CUDA kernels consume.
- **Cross-encoder** reads query and document *together* and scores the pair. Far more accurate,
  far too slow to run on the whole corpus. So you run it on the top ~100 only.

Mean-pool the token embeddings, then L2-normalize. After normalizing, **cosine similarity and
dot product are the same operation** — which is why the kernel can just do a dot product.

### Build (8 h)

**1. `tokenizer.cpp`** — BERT WordPiece in C++. Lowercase, split on punctuation, then greedy
longest-match-first against the vocab, marking continuations with `##`. Check it against the
Python tokenizer on a few hundred strings; they must agree exactly.

**2. `onnx_session.cpp`** — load a model, run it on the CUDA provider.

> **Watch for this one.** ONNX Runtime registers its default logger when you create the `Env`.
> If you write `Session session(makeEnv(), path, makeOptions())`, C++ does **not** guarantee
> which argument is evaluated first, and you can touch the logger before it exists. It throws
> `Attempt to use DefaultLogger but none has been registered`. Force the `Env` first. This
> repo hit exactly that bug on ONNX Runtime 1.28.

**3. `eval_quality.py`** — Recall@10 and nDCG@10 against BEIR qrels, plus overlap with a CPU
baseline. Overlap tells you your *kernel* is right; nDCG tells you your *system* is good. They
are different claims and you need both.

**Ship it:** type a question, get relevant passages back. Then the quality table across all four
dtypes. Aim for int8 matching float32 almost exactly and binary+rescore within ~2%.

**Answer key:** `text/tokenizer.cpp`, `onnx/onnx_session.cpp`, `python/benchmarks/eval_quality.py`.

---

## Stage 11 — Make it a library, not a script (10 h)

| Take | Resource | Time |
| --- | --- | --- |
| **Only** the pure-virtual / abstract-base-class page. | learncpp — [abstract base classes](https://www.learncpp.com/cpp-tutorial/pure-virtual-functions-abstract-base-classes-and-interface-classes/) | 40 min |
| "Introduction", "Targets", "Properties". **Stop there.** | *An Introduction to Modern CMake* — [cliutils.gitlab.io/modern-cmake](https://cliutils.gitlab.io/modern-cmake/) | 60 min |
| "First steps" and "Object-oriented code" only. | pybind11 — [pybind11.readthedocs.io](https://pybind11.readthedocs.io/en/stable/basics.html) | 60 min |

Three things to build:

**1. Interfaces.** Define an abstract `Reranker` with a pure virtual `score()`. The ONNX one
implements it. Now your `Engine` never includes an ONNX header — swap the backend without
touching the engine or the kernels. This is **dependency inversion**, and it is the single most
useful design idea in the file.

**2. CMake.** Targets, not variables. `PUBLIC` vs `PRIVATE` on link libraries.

**3. pybind11.** Expose `Engine` to Python. One subtlety: **release the GIL** around the search
call (`py::call_guard<py::gil_scoped_release>`) so other Python threads run while the GPU works.
Do *not* release it around anything touching Python objects.

**Ship it:** `import myengine; myengine.Engine(cfg).search("...")` from Python.

**Answer key:** `engine/engine.cpp`, `bindings/python/module.cpp`, the `CMakeLists.txt` files.

---

## Stage 12 — Prove it, and write it up (8 h)

Anyone can claim a fast kernel. What separates this from a toy:

**1. Every kernel against a CPU reference** that reproduces the *exact same arithmetic* —
including the quantization rounding and the bit-packing order. Then a mismatch means a real
divergence, not a tolerance argument.

**2. Know why GPU and CPU differ in the last bits.** Floating-point addition is not associative;
a different summation order gives a different answer. Both are "correct". So you pick tolerances
deliberately and can justify each one.

**3. Report the thing you removed, and what it cost.** *"Here is the transfer I deleted, here is
its measured effect, and here is what it costs in recall"* is a far stronger claim than a round
speedup multiple. Include the case where quantization **loses** accuracy, and the case where
hybrid **hurts**. Numbers you can defend beat numbers that flatter you.

**4. Re-read the paragraph you wrote in Stage 0.**

Finally, write your own README: the zero-copy accounting (what stays on device, what crosses and
why it is small), the kernel descriptions, the benchmark tables, and an honest limitations
section. If you cannot write the limitations you do not understand the design.

---

## Time budget

| Stage | What | Hours |
| ---: | --- | ---: |
| 0 | See it run | 1 |
| 1 | GPU mental model | 4 |
| 2 | First kernel + RAII | 8 |
| 3 | **The scan kernel** | **14** |
| 4 | Quantization + formats | 12 |
| 5 | **Top-K selection** | **12** |
| 6 | On-device rescore chain | 6 |
| 7 | BM25 + atomics | 10 |
| 8 | CUDA graphs | 6 |
| 9 | cuBLAS comparison | 4 |
| 10 | Tokenizer, ONNX, evaluation | 12 |
| 11 | Library, CMake, bindings | 10 |
| 12 | Testing and write-up | 8 |
| | **Total** | **≈ 107 h** |

Stages **3 and 5 are 25% of the time and 80% of the value.** The custom kernels are the actual
content; everything else is plumbing an employer expects but will not ask about.

**If you have one weekend:** Stages 1, 2, 3. A working, measured, coalesced scan kernel with
query tiling is a real artifact on its own.

**Minimum honest "I built a GPU search engine":** Stages 1–6. That is scan, quantization,
selection, and the on-device rescore hand-off — roughly 55 hours, and it is a complete story.

---

## The four questions this earns you

Prepare each as a story with a number attached:

1. **"Why is top-K harder than a dot product?"** Reduction is associative and order-free;
   selection depends on a global ordering. Walk the recursive bitonic merge and the termination
   argument. (Stage 5)
2. **"What did you do about host↔device transfers?"** Two removed, not one: the score vector,
   and the quantized-rescore round trip. (Stages 5, 6)
3. **"How do you speed up a bandwidth-bound kernel?"** You don't tune the arithmetic — you read
   fewer bytes (quantize) or reuse each one more (query tiling). Both, with roofline numbers.
   (Stages 3, 4)
4. **"How do you know it's correct?"** Every kernel against a CPU reference reproducing the exact
   arithmetic; the same wire format tested from both languages. (Stages 4, 12)

---

## A note on honesty

The goal is to be able to defend every line as your own, which requires that it *is* your own.
Type it. Get it wrong. Measure the wrong version. The bugs you cause deliberately — the merge
tile that hangs, the graph key that returns stale results — are worth more in an interview than
any amount of working code you transcribed, because you can explain the failure and nobody who
copied can.

`docs/LEARNING.md` remains the deep reference: same material, organised for *reading* this
codebase rather than rebuilding it, with harder checkpoint questions per file. Use it when a
stage here leaves you wanting more depth on one file.
