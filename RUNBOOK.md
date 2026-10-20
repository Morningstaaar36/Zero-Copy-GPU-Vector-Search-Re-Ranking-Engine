# Runbook — this machine

Setup is complete and verified. This file is the short version of what to type.

Machine: RTX 5060 Ti (sm_120, 16 GB) · CUDA 13.3 · GCC 15.2 · Ubuntu 26.04
Python: `.venv` built from anaconda Python 3.14.6 · ONNX Runtime 1.28.0 (CUDA 13 build)

Everything below is run from the project root:

```bash
cd /home/lucifer/BUET/CUDA/Zero-Copy-GPU-Vector-Search-Re-Ranking-Engine
```

## The scripts (they set up their own environment)

```bash
bash scripts/run_demo.sh     # one query through all four variants + the C++ CLI
bash scripts/verify.sh       # all checks: formats, 9 CUDA kernel tests, engine, variants
bash scripts/bench_all.sh    # kernel roofline + end-to-end latency + retrieval quality
bash scripts/build.sh        # recompile after editing C++/CUDA
bash scripts/run_factcheck.sh  # Llama 3.2 fact-check of one claim (CLAIM="..." to change it)
python scripts/make_figures.py # regenerate docs/assets/*.svg from docs/results/*.json
```

Llama weights live in `models/llama/` (fetched by `scripts/download_llama.sh`). The Llama
evaluation that produced `docs/results/*.json` and `configs/calibration/*.json`:

```bash
python python/benchmarks/eval_llama.py --llama models/llama/Llama-3.2-3B-Instruct models/llama/Llama-3.2-1B-Instruct
```

Change the query:

```bash
QUERY="how does crispr gene editing work" bash scripts/run_demo.sh
```

## Running commands by hand

Anything that loads the engine directly needs these three lines first:

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$PWD/third_party/onnxruntime/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/build/bindings:$PWD/python:$PYTHONPATH
```

### Kernel benchmark — needs no models or corpus

```bash
./build/apps/zcgvs_bench_kernels --docs 1000000 --dim 384 --k 100
```

### Python quickstart

```bash
python python/examples/quickstart.py --query "what causes type 2 diabetes" --top-k 5

# configs are the only way to reach the rescore and hybrid stages
python python/examples/quickstart.py --config configs/scifact_int8.json --query "crispr"
python python/examples/quickstart.py --config configs/scifact_binary_rescore.json --query "crispr"
python python/examples/quickstart.py --config configs/scifact_hybrid.json --hybrid --query "crispr"
```

### C++ CLI

```bash
./build/apps/zcgvs_search_cli \
  --embeddings data/scifact.bin --metadata data/scifact.meta \
  --vocab models/vocab.txt \
  --encoder models/encoder/model.onnx --reranker models/reranker/model.onnx \
  --query "how does crispr gene editing work" --top-k 10 --stats
```

Binary storage with the on-device rescore pass, timed after warmup:

```bash
./build/apps/zcgvs_search_cli \
  --embeddings data/scifact.binary.bin --rescore-embeddings data/scifact.bin \
  --rescore-candidates 512 --graphs --repeat 20 \
  --metadata data/scifact.meta --vocab models/vocab.txt \
  --encoder models/encoder/model.onnx --reranker models/reranker/model.onnx \
  --query "crispr gene editing"
```

### CUDA kernel tests

```bash
ctest --test-dir build --output-on-failure
```

### HTTP service

```bash
python -m zcgvs_tools.service --config configs/scifact_hybrid.json --port 8000
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/stats
curl -s -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"crispr gene editing","top_k":5,"candidates":50,"hybrid":true,"dense_weight":0.7}'
```

### Python API

```python
from zcgvs_tools.config import build_engine

engine = build_engine("configs/scifact_binary_rescore.json")
options = engine.default_options()
options.top_k, options.candidates = 10, 100

response = engine.search_with("what causes type 2 diabetes", options)
for hit in response.hits:
    print(hit.doc_id, round(hit.rerank_score, 3), hit.text[:80])
print(response.timing.search_gpu_ms, response.timing.total_ms)
```

## Rebuilding artifacts

A bigger corpus exercises the kernels far more than SciFact's 5.2k documents:

```bash
DATASET=fiqa bash scripts/generate_artifacts.sh   # ~57k docs
DATASET=fiqa bash scripts/run_demo.sh
```

The configs in `configs/` are named per dataset, so a new dataset needs its own config files
(copy `configs/scifact*.json` and change the paths).

## What was changed to make this machine work

Three edits, all upstream-safe:

1. `src/core/cuda_context.cpp` — CUDA 13 removed `cudaDeviceProp::memoryClockRate`; now read
   through `cudaDeviceGetAttribute` behind a `CUDART_VERSION >= 13000` guard.
2. `src/onnx/onnx_session.cpp` — force the `Ort::Env` into existence before building session
   options. Argument evaluation order is unspecified, so ONNX Runtime 1.28 could reach its
   default logger before the env registered one.
3. `scripts/download_onnxruntime.sh` — pick the ONNX Runtime GPU build from the installed
   `nvcc` major version (CUDA 13 → `gpu_cuda13-1.28.0`, otherwise the original CUDA 12
   `1.20.1`), with `ONNXRUNTIME_PKG` as an override.

Note that `scripts/setup_python.sh` defaults to `python3`, which on this machine is a system
Python with no `pip` and no headers. `.venv` is already built from anaconda's Python, so that
script now just reuses it. If you ever delete `.venv`, recreate it with:

```bash
PYTHON=/home/lucifer/anaconda3/bin/python bash scripts/setup_python.sh
pip install cmake ninja        # cmake is not installed system-wide
```
