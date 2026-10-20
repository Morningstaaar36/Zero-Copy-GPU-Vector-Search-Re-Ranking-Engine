# Run

Everything you can execute, in the order you would reach for it. Assumes
[DOWNLOAD.md](DOWNLOAD.md) is done.

## Zero to running

```bash
bash scripts/setup_all.sh
```

Downloads dependencies, exports models, embeds the corpus, derives the quantized variants and
the lexical index, builds, and verifies. If it finishes cleanly, everything works.

Then:

```bash
bash scripts/run_demo.sh     # one query through every variant, side by side
bash scripts/bench_all.sh    # every benchmark
```

## Environment

Two things must be set for anything that loads the engine:

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=third_party/onnxruntime/lib:$LD_LIBRARY_PATH
export PYTHONPATH=build/bindings:python:$PYTHONPATH
```

Every script in `scripts/` does this for you. Do it by hand only when running commands
directly.

## Build

```bash
bash scripts/build.sh
```

or explicitly:

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$PWD/third_party/onnxruntime \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

| Option | Default | Effect |
| --- | --- | --- |
| `ZCGVS_BUILD_PYTHON` | ON | The pybind11 module |
| `ZCGVS_BUILD_APPS` | ON | CLI and kernel benchmark |
| `ZCGVS_BUILD_TESTS` | ON | The nine CUDA harnesses |
| `ZCGVS_ENABLE_NVTX` | OFF | Nsight Systems ranges |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Set explicitly for cross-compilation, e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` |

Outputs:

```
build/bindings/zcgvs_engine.*.so   Python module
build/apps/zcgvs_search_cli        C++ CLI
build/apps/zcgvs_bench_kernels     synthetic kernel benchmark
build/tests/zcgvs_test_*           correctness harnesses
```

## The kernel benchmark — no artifacts required

The fastest way to confirm the CUDA side works, straight after a build:

```bash
./build/apps/zcgvs_bench_kernels --docs 1000000 --dim 384 --k 100
```

Synthetic data, no models, no corpus, no network. Flags: `--docs`, `--dim`, `--k`,
`--warmup`, `--iterations`, `--device`, `--markdown 1`.

## Python

### Quickstart

```bash
python python/examples/quickstart.py --query "how does crispr gene editing work"
```

Or through a config, which is the only way to reach the rescore and hybrid stages:

```bash
python python/examples/quickstart.py --config configs/scifact_binary_rescore.json \
    --query "how does crispr gene editing work"

python python/examples/quickstart.py --config configs/scifact_hybrid.json --hybrid \
    --query "crispr cas9"
```

### The API

```python
from zcgvs_tools.config import build_engine

engine = build_engine("configs/scifact.json")

# Start from the configured defaults and override what you need.
options = engine.default_options()
options.top_k = 10
options.candidates = 100

response = engine.search_with("what causes type 2 diabetes", options)
for hit in response.hits:
    print(hit.doc_id, round(hit.rerank_score, 3), hit.text[:80])

t = response.timing
print(f"embed {t.embed_ms:.2f}  gpu {t.search_gpu_ms:.3f}  copy {t.copy_ms:.3f} "
      f"rerank {t.rerank_ms:.2f}  total {t.total_ms:.2f}  graph={t.graph_replay}")
```

Batched — one device pass for many queries:

```python
batch = engine.search_batch(["query one", "query two", "query three"], options)
for query, hits in zip(["query one", "query two", "query three"], batch.results):
    print(query, [h.doc_id for h in hits])
print("batch", batch.timing.batch, "gpu ms", batch.timing.search_gpu_ms)
```

Metric, filter, hybrid, rescore width, backend, graphs:

```python
import zcgvs_engine

options.use_config_metric = False
options.metric = zcgvs_engine.Metric.L2

options.allow = [3, 17, 42, 128]        # device-side allow-mask

options.hybrid = True                   # needs lexical_index_path in the config
options.dense_weight = 0.7

options.rescore_candidates = 512        # needs rescore_embeddings_path in the config

options.use_config_backend = False
options.backend = zcgvs_engine.SimilarityBackend.Cublas

options.use_config_graphs = False
options.use_graph = True
```

Retrieval only, and live indexing:

```python
for candidate in engine.retrieve("crispr", 50):
    print(candidate.doc_id, candidate.score)

engine.add_documents(ids=["new-1"], texts=["a passage added at runtime"])
```

What is resident:

```python
from zcgvs_tools.config import describe
import json
print(json.dumps(describe(engine), indent=2))
```

## C++ CLI

```bash
./build/apps/zcgvs_search_cli \
  --embeddings data/scifact.bin --metadata data/scifact.meta \
  --vocab models/vocab.txt \
  --encoder models/encoder/model.onnx --reranker models/reranker/model.onnx \
  --query "how does crispr gene editing work" --top-k 10 --candidates 100 --stats
```

| Flag | Meaning |
| --- | --- |
| `--embeddings --metadata --vocab --encoder --query` | Required |
| `--reranker PATH` | Omit for retrieval-only |
| `--rescore-embeddings PATH` `--rescore-candidates N` | On-device rescore stage |
| `--lexical PATH` `--hybrid` `--dense-weight F` | Hybrid BM25 + dense |
| `--metric cosine\|dot\|l2` | Similarity metric |
| `--backend custom\|cublas` | Scan implementation |
| `--graphs` | Replay from a CUDA graph |
| `--repeat N` | Re-run the query N times, report the last timing (warms caches and graphs) |
| `--device N` `--stats` | Device selection; print device info |
| `--help` | Full usage |

Binary corpus with an exact rescore pass, timed after warmup:

```bash
./build/apps/zcgvs_search_cli \
  --embeddings data/scifact.binary.bin --rescore-embeddings data/scifact.bin \
  --rescore-candidates 512 --graphs --repeat 20 \
  --metadata data/scifact.meta --vocab models/vocab.txt \
  --encoder models/encoder/model.onnx --reranker models/reranker/model.onnx \
  --query "crispr gene editing"
```

## HTTP service

```bash
pip install -r python/requirements-serve.txt
python -m zcgvs_tools.service --config configs/scifact_hybrid.json --port 8000
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/stats

curl -s -X POST localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "crispr gene editing", "top_k": 5, "candidates": 50,
  "hybrid": true, "dense_weight": 0.7
}'

curl -s -X POST localhost:8000/search/batch -H 'content-type: application/json' -d '{
  "queries": ["crispr gene editing", "type 2 diabetes"], "top_k": 5, "candidates": 50
}'

curl -s -X POST localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query": "crispr", "k": 20}'

curl -s -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"ids": ["x1"], "texts": ["a passage added at runtime"]}'
```

Start the service with `--llama models/llama/Llama-3.2-3B-Instruct` (or a Hub id) to add the
Llama endpoints:

```bash
curl -s -X POST localhost:8000/verify -H 'content-type: application/json' \
  -d '{"claim": "Aspirin reduces colorectal cancer risk.", "evidence_k": 3, "explain": true}'

curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What links obesity to insulin resistance?", "k": 5}'
```

`/verify` returns the verdict, calibrated label probabilities, the explanation, the evidence
and a per-stage timing breakdown; `/ask` returns the answer and its sources. See
[LLAMA.md](LLAMA.md).

Every per-request option is exposed: `metric`, `allow`, `hybrid`, `dense_weight`,
`rescore_candidates`, `rerank`, `backend`, `use_graph`.

## Configs

| Config | Storage | Extras |
| --- | --- | --- |
| `configs/scifact.json` | float32 | Baseline |
| `configs/scifact_int8.json` | int8 | 4× smaller scan |
| `configs/scifact_binary_rescore.json` | binary | float32 rescore store, 512 candidates, CUDA graphs on |
| `configs/scifact_hybrid.json` | float16 | BM25 inverted index, `dense_weight` 0.7 |

Paths inside a config are resolved against the working directory first, then the config's own
directory, then its parent — so the same file works from anywhere in the tree.

## Profiling

```bash
cmake -B build -S . -DZCGVS_ENABLE_NVTX=ON ...   # reconfigure
cmake --build build -j

nsys profile -o zcgvs --trace=cuda,nvtx \
  ./build/apps/zcgvs_search_cli --repeat 50 ... --query "crispr"
```

`zcgvs::search`, `zcgvs::rerank` and `zcgvs::copyResultToHost` ranges show on the timeline
alongside the kernels. Use `--repeat` so the trace contains warm iterations, and compare a run
with and without `--graphs` to see the launch gaps close.

## Installing as a package

```bash
pip install . -C cmake.define.ONNXRUNTIME_ROOT=$PWD/third_party/onnxruntime
```

scikit-build-core compiles the extension and installs `zcgvs_tools`, exposing the console
scripts `zcgvs-export-models`, `zcgvs-embed-corpus`, `zcgvs-quantize`,
`zcgvs-build-lexical` and `zcgvs-serve`.

Next: [VERIFY.md](VERIFY.md).
