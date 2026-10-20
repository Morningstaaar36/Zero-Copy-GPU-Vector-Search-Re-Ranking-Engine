# Benchmarks

Methodology first, then tables to fill in from your own hardware. **The tables below are
empty on purpose.** Numbers are only worth quoting if they came off the machine they are
quoted from, and this project's whole benchmarking stance is that an honest measured delta
beats a round speedup multiple.

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
GPU              :
Driver / CUDA    :
Peak DRAM (GB/s) :
Workload         : 1 000 000 docs × 384 dims
```

### Scan throughput by storage precision

Bytes moved is `doc_count × row_stride`, so the smaller dtypes do strictly less work. The
question the table answers is whether the kernel stays near roofline as the rows get smaller.

| Storage | Bytes scanned | Median ms | GB/s | % of peak |
| --- | ---: | ---: | ---: | ---: |
| float32 | 1.54 GB | | | |
| float16 | 0.77 GB | | | |
| int8 | 0.38 GB | | | |
| binary | 0.05 GB | | | |
| cuBLAS `sgemv` (float32) | 1.54 GB | | | |

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
| 10 | | |
| 100 | | |
| 1000 | | |

Selection should cost a small fraction of the scan. If top-K dominates, the merge recursion is
running more passes than expected — check `topKScratchElements` against the corpus size.

### Batching and CUDA graphs

| Batch | Direct launch (ms) | Graph replay (ms) | Direct q/s | Graph q/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |

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

| Configuration | ms/query | queries/s |
| --- | ---: | ---: |
| engine rerank batch=1 graphs=off | | |
| engine rerank batch=1 graphs=on | | |
| engine rerank batch=8 graphs=on | | |
| engine rerank batch=32 graphs=on | | |
| engine retrieval-only | | |
| engine hybrid (BM25 + dense) | | |
| python baseline (ST + CrossEncoder) | | — |

**Expect the cross-encoder to dominate.** Reranking 100 passages through a transformer is far
more work than scanning 5 000 short vectors. That is the honest headline: on a small corpus the
GPU search is already close to free, and the end-to-end speedup mostly reflects avoiding
Python overhead in the retrieval stage. The `retrieval-only` row is where the kernel work is
actually visible — quote both.

Per-stage breakdown for one query:

| Stage | ms | Note |
| --- | ---: | --- |
| tokenize | | Host, hybrid path only |
| embed | | ONNX bi-encoder, host-visible |
| gpu_search | | CUDA events: scan + fuse + top-K + rescore |
| copy | | The one device→host transfer, pinned |
| rerank | | ONNX cross-encoder over the candidates |
| **total** | | |

The `copy` row is the point of the whole project. It should be a rounding error, and it does
not grow with corpus size — only with `k`.

## 3. Retrieval quality across the storage ladder

```bash
python python/benchmarks/eval_quality.py \
  --config configs/scifact.json configs/scifact_int8.json \
           configs/scifact_binary_rescore.json configs/scifact_hybrid.json \
  --dataset scifact --max-queries 100
```

| Config | Storage | Rescore | Hybrid | VRAM MB | Recall@10 | nDCG@10 | GPU/CPU overlap | gpu ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| scifact.json | float32 | – | no | | | | | |
| scifact_int8.json | int8 | – | no | | | | | |
| scifact_binary_rescore.json | binary | float32 | no | | | | | |
| scifact_hybrid.json | float16 | – | yes | | | | | |

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

| rescore_candidates | Recall@10 | gpu ms |
| --- | ---: | ---: |
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |

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
