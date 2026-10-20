# Testing Guide

The documentation set is split by task. This file is the index.

| I want to… | Read |
| --- | --- |
| Understand the codebase line by line | [`docs/LEARNING.md`](docs/LEARNING.md) |
| Get every dependency, model and dataset onto disk | [`docs/DOWNLOAD.md`](docs/DOWNLOAD.md) |
| Build it and run something | [`docs/RUN.md`](docs/RUN.md) |
| Prove it is correct | [`docs/VERIFY.md`](docs/VERIFY.md) |
| Measure it honestly | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| Fix a failure | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| Understand what it does | [`docs/FEATURES.md`](docs/FEATURES.md) |
| Understand how it works | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Call it from code | [`docs/API.md`](docs/API.md) |
| Read or change a file format | [`docs/FORMATS.md`](docs/FORMATS.md) |

## The short version

```bash
bash scripts/setup_all.sh     # download + build + verify, from a clean checkout
```

or step by step:

```bash
bash scripts/download_all.sh  # ONNX Runtime, Python deps, models, corpus, indexes
bash scripts/build.sh         # compile the library, bindings, CLI, benchmark, tests
bash scripts/verify.sh        # every correctness check, tiered
bash scripts/run_demo.sh      # one query through every configured variant
bash scripts/bench_all.sh     # kernel roofline + end-to-end + retrieval quality
```

The scripts fetch large dependencies and compile CUDA code. They were written for you to run —
nothing has been executed for you except the pure-Python wire-format tests, which need only
NumPy.

## What "everything" covers

| Layer | Verified by |
| --- | --- |
| Wire formats, all four storage dtypes | `tests/python/test_formats.py` (NumPy only, no GPU) |
| float32 scan | `zcgvs_test_similarity` vs a CPU dot-product loop |
| L2 metric and the device allow-mask | `zcgvs_test_metrics_filter` vs a CPU reference |
| int8 and binary scans | `zcgvs_test_quantized` vs CPU references reproducing the exact quantization and bit-packing |
| Top-K selection | `zcgvs_test_topk` vs `std::stable_sort` |
| Recursive merge at scale | `zcgvs_test_topk_large` — 1M scores, k=512 |
| Batched scan and selection | `zcgvs_test_batched` — 11 queries vs per-query references |
| Rescore stage, graph replay | `zcgvs_test_pipeline` — exact top-K recovery and bit-identical replay |
| BM25, max reduction, fusion | `zcgvs_test_lexical` vs CPU references |
| Corpus load into VRAM | Engine startup in the end-to-end run |
| Tokenizer + ONNX bi-encoder | An end-to-end query returns ranked results |
| ONNX cross-encoder rerank | Rerank scores in the CLI / quickstart output |
| pybind11 boundary | `import zcgvs_engine` in quickstart and the benchmarks |
| Retrieval quality vs qrels | `benchmarks/eval_quality.py`, across the whole storage ladder |
| Zero-copy search→rescore→rerank path | Exercised on every query; see [`docs/VERIFY.md`](docs/VERIFY.md#verifying-the-zero-copy-claim-itself) |

## Quick smoke test on a fresh machine

```bash
python tests/python/test_formats.py            # seconds, needs only NumPy
bash scripts/build.sh
./build/apps/zcgvs_bench_kernels --docs 100000 # synthetic, no artifacts needed
ctest --test-dir build --output-on-failure
```

That covers the wire formats and every CUDA kernel without downloading a single model.
