# Benchmarks

Methodology first, then the numbers measured on the reference machine below. Numbers are
only worth quoting if they came off the machine they are quoted from, and this project's whole
benchmarking stance is that an honest measured delta beats a round speedup multiple. Re-run on
your own hardware and replace them.

```bash
bash scripts/bench_all.sh
```

Runs all three layers and prints markdown-ready output.

## Methodology

**Timing.** Device work is timed with CUDA events bracketing the pipeline, so it excludes
launch setup and graph capture. Host stages are `steady_clock`. Every measurement takes the
**median** of many iterations after warmup runs — never a single sample, and never a mean,
which one cold outlier dominates.

**Warmup matters more than usual here.** The first query pays ONNX Runtime session warmup,
CUDA context and module load, lazy scratch allocation, and (if enabled) CUDA graph capture.
The kernel benchmark warms up 5 iterations by default; `--repeat` on the CLI and `--repeats`
in the Python benchmark exist for the same reason.

**What the baseline is.** `sentence-transformers`' `util.semantic_search` for retrieval plus a
`CrossEncoder` for reranking, over the identical corpus, models and queries, on the same GPU.
It is a fair baseline: it is what you would actually write in Python, and it also runs on the
GPU. It is not a strawman CPU loop.

**Roofline.** The scan is memory-bandwidth bound, so the meaningful figure is achieved GB/s
against the device's theoretical peak (`2 × memory_clock × bus_width / 8`). A kernel at 80% of
peak is essentially done; chasing the last 20% is not where the remaining wins are. The wins
are in reading *fewer bytes* — which is what the storage ladder is for — and in reusing each
byte across more queries, which is what batching is for.

## 1. Kernel microbenchmark

No models, no corpus, no network. Works immediately after a build.

```bash
./build/apps/zcgvs_bench_kernels --docs 1000000 --dim 384 --k 100 --markdown 1
```

The synthetic corpus is materialized in host memory first, so `--docs × --dim × 4` bytes of
RAM and up to the same again in VRAM. One million 384-dim rows is 1.5 GB of each; the default
is 200 000 rows (~300 MB), which already far exceeds any L2 and therefore still measures DRAM
bandwidth honestly. Scale up if you have the headroom — the numbers get cleaner.

Environment:

```
GPU              : NVIDIA GeForce RTX 5060 Ti 16 GB (sm_120, 36 SMs)
Driver / CUDA    : 580.178.04 / CUDA 13.3
Peak DRAM (GB/s) : 448 (theoretical)
Workload         : 1 000 000 docs × 384 dims, k = 100, median of 50 iterations
```

### Scan throughput by storage precision

Bytes moved is `doc_count × row_stride`, so the smaller dtypes do strictly less work. The
question the table answers is whether the kernel stays near roofline as the rows get smaller.

| Storage | Bytes scanned | Median ms | GB/s | % of peak |
| --- | ---: | ---: | ---: | ---: |
| float32 | 1.54 GB | 3.648 | 421.0 | 94% |
| float16 | 0.77 GB | 1.870 | 410.6 | 92% |
| int8 | 0.38 GB | 0.957 | 401.1 | 90% |
| binary | 0.05 GB | 0.122 | 392.6 | 88% |
| cuBLAS `sgemv` (float32) | 1.54 GB | 3.625 | 423.7 | 95% |

**Before → after the persistent-kernel rewrite.** The first version of the scan launched one
256-thread block per document, and every block re-staged the full query tile into shared
memory. At 1M documents that is ~12 GB of redundant query traffic to scan a 0.05–1.5 GB corpus
— a fixed cost that swamped the corpus bytes, so every precision ran at the same speed:

| Storage | Block-per-doc (v1) | Persistent warp-per-doc (v2) | Speedup |
| --- | ---: | ---: | ---: |
| float32 | 7.279 ms · 47% of peak | 3.648 ms · 94% | 2.0× |
| float16 | 7.769 ms · 22% | 1.870 ms · 92% | 4.2× |
| int8 | 7.938 ms · 11% | 0.957 ms · 90% | 8.3× |
| binary | 5.425 ms · 2% | 0.122 ms · 88% | 44× |
| full pipeline, batch 8 | 334 q/s | 1428 q/s | 4.3× |

v2 stages the queries once per *resident* block, sizes the grid to exactly fill the GPU
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor`), gives each warp a document with 128-bit
loads (`float4` / 8×`half` / 16×`int8`) and float4 shared-memory query reads, and gives each
*thread* a document on the binary path, where a row is only 48 bytes. The custom fp32 kernel
now ties cuBLAS while also handling fp16, int8, binary, L2 and the allow-mask, which cuBLAS
cannot. `tests/cpp/test_scan_paths.cu` checks all 44 dtype × dim × metric × mask combinations,
including the scalar fallbacks, against CPU references.

Read it like this: fp16 should be roughly 2× faster than fp32 and int8 roughly 4×, because
they move that much less. Where the ratio falls short of the size ratio, the kernel has stopped
being bandwidth-bound and started being latency- or occupancy-bound — binary is the likeliest
place to see that, since its rows are only 48 bytes.

The cuBLAS row is the honest yardstick. Expect it to win on plain fp32; that is a
well-optimized vendor GEMM. The interesting facts are how close the custom kernel gets, and
that cuBLAS cannot express the int8, binary, masked or L2 cases at all.

### Top-K cost

| k | Median ms | Effective GB/s over the score array |
| --- | ---: | ---: |
| 10 | 0.430 | 9.3 |
| 100 | 0.532 | 7.5 |
| 1000 | 1.185 | 3.4 |

Selection should cost a small fraction of the scan. If top-K dominates, the merge recursion is
running more passes than expected — check `topKScratchElements` against the corpus size.

Now that the scan is at roofline, this table is the next bottleneck: at k = 100 top-K costs
more than half of an fp16 scan and four times a binary one. A radix-select first pass is the
obvious next kernel.

### Batching and CUDA graphs

| Batch | Direct launch (ms) | Graph replay (ms) | Direct q/s | Graph q/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.405 | 2.402 | 416 | 416 |
| 8 | 5.602 | 5.650 | 1428 | 1416 |
| 32 | 22.104 | 22.094 | 1448 | 1448 |

At a million documents the scan is milliseconds long, so launch overhead is noise and graph
replay buys nothing measurable; it matters on small corpora (see the end-to-end table). The
query-tiling gain is real: batch 8 serves 3.4× the queries per second of batch 1.

Two effects to separate:

- **Query tiling** — per-query time should fall as batch grows, because each corpus row is
  read once for up to 8 queries. The gain flattens past the tile width.
- **Graph replay** — a roughly constant time saving per query, so it is proportionally largest
  at batch 1 and on small corpora, where launch overhead is a real share of the total.

## 2. End-to-end latency

```bash
python python/benchmarks/benchmark_e2e.py \
  --embeddings data/scifact.bin --metadata data/scifact.meta \
  --vocab models/vocab.txt \
  --encoder models/encoder/model.onnx --reranker models/reranker/model.onnx \
  --lexical data/scifact.lex
```

SciFact, 5 183 abstracts, float32, 100 candidates reranked, median over the 300 test queries:

| Configuration | ms/query | queries/s |
| --- | ---: | ---: |
| engine rerank batch=1 graphs=off | 74.50 | 13.4 |
| engine rerank batch=1 graphs=on | 74.51 | 13.4 |
| engine rerank batch=8 graphs=on | 75.59 | 13.2 |
| engine rerank batch=32 graphs=on | 75.73 | 13.2 |
| engine retrieval-only | 0.727 | 1375 |
| engine hybrid (BM25 + dense) | 75.86 | 13.2 |
| python baseline (ST + CrossEncoder) | 148.74 | — |

2.0× faster end to end than the Python baseline at batch 1 — and as predicted below, almost all
of the remaining 74 ms is the cross-encoder scoring 100 passages.

**Expect the cross-encoder to dominate.** Reranking 100 passages through a transformer is far
more work than scanning 5 000 short vectors. That is the honest headline: on a small corpus the
GPU search is already close to free, and the end-to-end speedup mostly reflects avoiding
Python overhead in the retrieval stage. The `retrieval-only` row is where the kernel work is
actually visible — quote both.

Per-stage breakdown for one query:

Medians over the 300 SciFact test queries, float32 corpus, 100 candidates:

| Stage | ms | Note |
| --- | ---: | --- |
| tokenize | 0.006 | Host, hybrid path only (measured on the hybrid config) |
| embed | 0.785 | ONNX bi-encoder, host-visible |
| gpu_search | 0.076 | CUDA events: scan + fuse + top-K + rescore |
| copy | 0.076 | The one device→host transfer, pinned |
| rerank | 73.973 | ONNX cross-encoder over the candidates |
| **total** | **74.928** | |

The `copy` row is the point of the whole project. It should be a rounding error, and it does
not grow with corpus size — only with `k`.

## 3. Retrieval quality across the storage ladder

```bash
python python/benchmarks/eval_quality.py \
  --config configs/scifact.json configs/scifact_int8.json \
           configs/scifact_binary_rescore.json configs/scifact_hybrid.json \
  --dataset scifact --max-queries 100
```

All 300 BEIR SciFact test queries, top-100 candidates reranked by the cross-encoder:

| Config | Storage | Rescore | Hybrid | VRAM MB | Recall@10 | nDCG@10 | GPU/CPU overlap | gpu ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| scifact.json | float32 | – | no | 8.0 | 0.7982 | 0.6807 | 0.9994 | 0.075 |
| scifact_int8.json | int8 | – | no | 2.0 | 0.7982 | 0.6807 | 0.9933 | 0.061 |
| scifact_binary_rescore.json | binary | float32 | no | 8.2 | 0.7932 | 0.6787 | 0.9458 | 0.164 |
| scifact_hybrid.json | float16 | – | yes | 10.7 | 0.7882 | 0.6778 | 0.9994 | 0.086 |

int8 is free: 4× less VRAM, identical Recall@10 and nDCG@10. Binary + float32 rescore loses
0.002 nDCG while its coarse pass reads 32× fewer bytes. Hybrid BM25 at `dense_weight = 0.7` is
slightly *worse* on SciFact — its claims are paraphrastic, so the lexical signal mostly adds
noise to an already strong dense ranking. (An earlier version of `eval_quality.py` silently
evaluated the hybrid config with hybrid off; it now honours the config's own flag.)

This is the table that makes the storage ladder a real engineering result rather than a
feature list. What it should show:

- **int8** costs very little accuracy for 4× less scanned data. Cosine similarity over
  L2-normalized 384-dim embeddings has values in a narrow range, so 8 bits is generous.
- **binary alone** loses real accuracy — 1 bit per dimension is a genuinely lossy LSH. Run it
  without a rescore store once to see how much.
- **binary + rescore** recovers most of it, because the final ordering is computed from exact
  float32 vectors. The residual loss is entirely recall of the *coarse* stage: anything the
  Hamming pass never puts in the candidate window cannot be recovered. Widen
  `rescore_candidates` and it improves.
- **hybrid** helps on keyword-ish queries and can hurt on purely semantic ones. Sweep
  `dense_weight` from 1.0 down to 0.5 before concluding anything.

Suggested sweep, since one `rescore_candidates` value proves nothing:

| rescore_candidates | Recall@10 | nDCG@10 | gpu ms |
| --- | ---: | ---: | ---: |
| 0 (binary only) | 0.7946 | 0.6796 | 0.055 |
| 128 | 0.8019 | 0.6824 | 0.082 |
| 256 | 0.7972 | 0.6809 | 0.109 |
| 512 | 0.7932 | 0.6787 | 0.165 |
| 1024 | 0.7982 | 0.6807 | 0.235 |

On SciFact the sweep is flat within noise, and the reason is instructive: the cross-encoder
re-orders the top 100 anyway, so the coarse stage only has to land the relevant abstracts
somewhere in that window — and on a 5 183-document corpus, even 1-bit Hamming does. Rescoring
earns its keep when there is no reranker (`options.rerank = False`), when `k` is close to the
candidate count, or on corpora large enough that the window is a tiny fraction of the index.

## What to claim, and what not to

**Claim:** "Binary storage scans 32× fewer bytes; measured GB/s went from X to Y; Recall@10
went from A to B without rescoring and back to C with a 512-wide on-device rescore, at a cost
of D ms."

**Claim:** "The candidate selection never leaves the device — here is the profile with no DtoH
between the scan and the rescore, and here is the one 800-byte transfer that remains."

**Do not claim:** a single speedup multiple over the Python baseline without saying which stage
it came from. Most of an end-to-end number on a 5k-document corpus is cross-encoder time and
Python overhead, not kernel quality — and anyone qualified to be impressed will ask.

**Do not claim** anything measured at batch 1 on a cold cache, or a mean over runs that
included the first one.

## Scaling the corpus

SciFact is 5 183 documents — small enough that the scan is nearly free and launch overhead
dominates. To exercise the kernels properly:

```bash
DATASET=fiqa bash scripts/generate_artifacts.sh       # ~57k documents
./build/apps/zcgvs_bench_kernels --docs 10000000      # synthetic, 10M rows
```

The synthetic benchmark is the better instrument for kernel work: it scales to whatever fits in
VRAM without needing a real corpus, and it isolates the scan from every ONNX cost.
