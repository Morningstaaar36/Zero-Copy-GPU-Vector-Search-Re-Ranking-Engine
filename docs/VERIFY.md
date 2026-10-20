# Verify

How to prove the engine is correct, what each check actually proves, and what you should see.

```bash
bash scripts/verify.sh
```

Runs every check in dependency order and prints a pass/fail/skip summary. It is deliberately
tiered, so a partial setup still verifies what it can rather than failing outright.

| Tier | Check | Needs | Skip with |
| --- | --- | --- | --- |
| 1 | Wire-format round-trips | NumPy only | — |
| 2 | CUDA kernel correctness | A build + a GPU | `SKIP_KERNELS=1` |
| 3 | End-to-end query | Models + corpus | `SKIP_ENGINE=1` |
| 4 | Storage / rescore / hybrid variants | Derived artifacts | `SKIP_VARIANTS=1` |

Expected tail:

```
==============================================================
 summary
==============================================================
  PASS     formats round-trip
  PASS     CUDA kernel tests
  PASS     end-to-end query
  PASS     int8 storage
  PASS     binary + on-device rescore
  PASS     hybrid BM25 + dense

  6 passed, 0 failed, 0 skipped
```

---

## Tier 1 — wire formats (no GPU, no build)

```bash
python tests/python/test_formats.py
```

Seven checks, all pure NumPy:

| Test | What it pins down |
| --- | --- |
| `test_row_stride_matches_dtype` | Bytes per row for all four dtypes, including binary word padding |
| `test_pack_binary_bit_order_and_padding` | Bit `b` of word `w` ↔ dimension `w*32+b`; padding dims stay zero |
| `test_int8_quantization_is_symmetric_and_bounded` | Quantization stays in `[-127, 127]`; error bounded by the scale |
| `test_embedding_roundtrip_every_dtype` | Header fields, file size, and value fidelity for all four dtypes |
| `test_binary_storage_is_32x_smaller_than_float32` | The 2× / 4× / 32× size claims |
| `test_metadata_roundtrip_handles_utf8` | Length-prefixed UTF-8 doc ids and passages |
| `test_lexical_index_roundtrip` | CSR offsets, postings, BM25 parameters |

This matters more than it looks: the bit-packing convention and the quantization arithmetic are
duplicated in Python and in CUDA. If these two ever disagree, every binary or int8 search
silently returns garbage. `tests/cpp/test_quantized.cu` checks the other side of the same
contract.

Expected:

```
  PASS  test_binary_storage_is_32x_smaller_than_float32
  PASS  test_embedding_roundtrip_every_dtype
  ...
formats: PASS
```

## Tier 2 — CUDA kernel correctness

```bash
ctest --test-dir build --output-on-failure
```

Nine harnesses, each comparing a kernel against a CPU reference:

| Test | Compares |
| --- | --- |
| `similarity` | float32 cosine scan vs a sequential dot-product loop |
| `topk` | Top-100 of 5 000 scores vs `std::stable_sort` |
| `metrics_filter` | L2 metric vs a CPU reference; masked docs pinned to `-FLT_MAX` |
| `quantized` | int8 scan vs a CPU reference that reproduces the *exact* quantization; binary scan vs a CPU popcount Hamming |
| `topk_large` | 1 000 000 scores at `k=512` — several recursive merge passes |
| `batched` | 11 queries (deliberately not a multiple of the 8-wide query tile) vs per-query references, scores *and* rankings |
| `pipeline` | Lossless coarse pass + rescore reproduces the exact CPU top-K; binary coarse pass stays well-formed; CUDA-graph replay is bit-identical to direct launches |
| `lexical` | BM25 scatter-add, the max reduction, and score fusion vs CPU references |
| `scan_paths` | 44 cases: every storage type × dims 384/768/100/97 (vectorized and scalar-fallback paths) × dot/L2 × with and without a mask, 11 queries across two tiles, 20 000 docs so every warp strides the corpus |

Expected:

```
Test #1: similarity ..... Passed
Test #2: topk ........... Passed
Test #3: metrics_filter . Passed
Test #4: quantized ...... Passed
Test #5: topk_large ..... Passed
Test #6: batched ........ Passed
Test #7: pipeline ....... Passed
Test #8: lexical ........ Passed
Test #9: scan_paths ..... Passed

100% tests passed, 0 tests failed out of 9
```

Run one directly for its diagnostic output:

```bash
./build/tests/zcgvs_test_pipeline
```

### What the tolerances mean

The GPU reduces with a warp-shuffle tree, the CPU sums sequentially, so bit-exact agreement is
not expected and not required. Dot products over 256–384 normalized dimensions are compared at
`1e-4`; the int8 path at `1e-3` because a single quantization step is worth `scale ≈ 1/127`;
the binary path at `1e-5` because popcounts are exact integers. BM25 uses a relative tolerance
because `atomicAdd` reorders the additions non-deterministically.

Two properties are checked *exactly*, because they must be: CUDA-graph replay must produce
identical bits to direct launches, and a lossless coarse pass must reproduce the CPU's exact
top-K ordering.

## Tier 3 — end-to-end query

```bash
python python/examples/quickstart.py --query "what causes type 2 diabetes"
```

Expected shape:

```
corpus  : 5183 docs, dim 384, storage float32 (8.0 MB)
device  : NVIDIA GeForce RTX ...

query: what causes type 2 diabetes
 1. [4983123] rerank=8.421 sim=0.612
    Type 2 diabetes mellitus is characterized by insulin resistance ...
 ...
timing (ms): embed=2.31 gpu_search=0.184 copy=0.041 rerank=11.02 total=13.6
```

What this proves: the binary format loads into VRAM, the tokenizer and ONNX bi-encoder produce
a usable query vector, the CUDA scan and top-K run over real data, the device→host transfer is
correct, the ONNX cross-encoder scores the candidates, and the pybind11 boundary passes objects
both ways.

Sanity, not just absence of crashes: the top hits should be topically relevant, `sim` should be
in `[-1, 1]` for cosine and monotonically decreasing before reranking, and `rerank` should
reorder the list (identical ordering to `sim` every time means the reranker is not running).

## Tier 4 — the optional stages

```bash
python python/examples/quickstart.py --config configs/scifact_int8.json --query "crispr"
python python/examples/quickstart.py --config configs/scifact_binary_rescore.json --query "crispr"
python python/examples/quickstart.py --config configs/scifact_hybrid.json --hybrid --query "crispr"
```

What to look for in each:

- **int8** — the header line reports `storage int8` and roughly a quarter of the float32
  `corpus_bytes`. Top hits should be close to the float32 run, not identical.
- **binary + rescore** — reports `storage binary` with about 1/32 of the bytes, plus a
  `rescore : float32` line. The `sim` values are the *exact* float32 similarities, because the
  rescore stage recomputed them; if they were raw Hamming proxies they would be coarse
  multiples of `2/dim`.
- **hybrid** — with `--hybrid`, an exact-keyword query should promote documents containing that
  literal token. Compare with and without the flag on a query with a rare term; if the ordering
  never changes, `dense_weight` is at 1.0 or the index did not load.

## Cross-checking retrieval quality

The strongest correctness argument is agreement with an independent implementation:

```bash
python python/benchmarks/eval_quality.py \
    --config configs/scifact.json configs/scifact_int8.json \
             configs/scifact_binary_rescore.json configs/scifact_hybrid.json \
    --dataset scifact --max-queries 100
```

For every config this reports Recall@10 and nDCG@10 against BEIR qrels, and — the important
column — **GPU/CPU overlap**: the fraction of the GPU's top-`candidates` that a plain NumPy
`matrix @ query` argsort also returns.

Interpreting overlap:

| Config | Expected overlap | Why |
| --- | --- | --- |
| float32 | ≈ 1.00 | Same arithmetic, only reduction order differs |
| float16 | ≈ 0.99+ | Half rounding occasionally swaps near-ties |
| int8 | high 0.9x | One quantization step of error per dimension |
| binary (no rescore) | much lower | 1-bit LSH is genuinely lossy — this is expected, not a bug |
| binary + rescore | high again | The rescore stage recovers the exact ordering within its candidate window |

That last row is the whole argument for the rescore stage, in one number.

## Verifying the zero-copy claim itself

The claim is structural, so read it in the code and confirm it in a profile.

In code — `src/search/searcher.cpp`, `recordPipeline` is a pure sequence of kernel launches and
async memsets. The only host transfer is `copyResultToHost`, which moves `batch × k` ints and
floats through pinned staging. Grep proves the negative:

```bash
grep -rn "cudaMemcpy" src/ | grep -v Async
```

The only synchronous copies are in `loadFromFile` / `fromHostMatrix` (startup) and
`lexical_index.cpp` (startup). Nothing in the query path.

In a profile — build with `-DZCGVS_ENABLE_NVTX=ON` and run under Nsight Systems. Inside a
`zcgvs::search` range you should see kernels back to back with no `Memcpy DtoH` between the
scan, the top-K and the rescore. The only DtoH sits inside `zcgvs::copyResultToHost`, and it is
a few hundred bytes.

With `--graphs`, the individual launch markers collapse into a single graph launch — which is
also visual proof the graph is being replayed rather than recaptured.

## Continuous checks worth running after any change

```bash
python tests/python/test_formats.py     # seconds, no GPU — run this constantly
ctest --test-dir build --output-on-failure
bash scripts/verify.sh                  # full sweep before committing
```

If you touch the bit-packing or quantization arithmetic, tiers 1 and 2 must **both** pass —
they are the two halves of the same contract, and a change to one without the other is exactly
the bug the pair is designed to catch.

Troubleshooting failures: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
