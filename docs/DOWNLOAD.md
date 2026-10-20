# Download — everything needed to run

Every external artifact this project needs, what it is, how big it is, and where it comes
from. **Nothing here is trained.** Models are downloaded checkpoints exported to ONNX; the
dataset is pulled from the Hugging Face Hub; the quantized corpora and the inverted index are
deterministic transforms of files already on disk.

## The one command

```bash
bash scripts/download_all.sh
```

That runs, in order: ONNX Runtime → Python environment → model export → corpus embedding →
quantized variants → BM25 index. Roughly **4–5 GB** of downloads and 5–15 minutes on a warm
connection, dominated by PyTorch.

Skip the models and corpus (dependencies only):

```bash
SKIP_ARTIFACTS=1 bash scripts/download_all.sh
```

## Prerequisites you must install yourself

| Requirement | Why | Check |
| --- | --- | --- |
| NVIDIA GPU + recent driver | Everything runs on the device | `nvidia-smi` |
| CUDA Toolkit 12.x with `nvcc` on `PATH` | Compiles the kernels; supplies cuBLAS and NVTX | `nvcc --version` |
| CMake ≥ 3.24 | Build system | `cmake --version` |
| C++17 host compiler | Builds the host side | `g++ --version` |
| Python ≥ 3.9 with `venv` | Offline tooling and the bindings | `python3 -m venv --help` |
| `curl`, `tar` | Fetching ONNX Runtime | `curl --version` |
| Python development headers | Building the pybind11 module | `python3-dev` / `python3-devel` |

The default ONNX Runtime GPU 1.20.1 package targets CUDA 12 + cuDNN 9. On a different CUDA
line, override the version — see [TROUBLESHOOTING](TROUBLESHOOTING.md).

## What gets downloaded

### 1. ONNX Runtime C++ SDK — ~450 MB

```bash
bash scripts/download_onnxruntime.sh
```

Fetches `onnxruntime-linux-x64-gpu-1.20.1` from the official GitHub release and extracts it to
`third_party/`, with a stable `third_party/onnxruntime` symlink that CMake and
`LD_LIBRARY_PATH` point at.

- Source: <https://github.com/microsoft/onnxruntime/releases>
- Override: `ONNXRUNTIME_VERSION=1.19.2 bash scripts/download_onnxruntime.sh`

This is the C++ inference runtime. It is not on PyPI as an SDK, which is why its location has
to be supplied to CMake explicitly.

### 2. Python environment — ~3 GB

```bash
bash scripts/setup_python.sh
source .venv/bin/activate
```

Creates `.venv` and installs `python/requirements.txt` plus `pybind11`:

| Package | Used for |
| --- | --- |
| `numpy` | Wire formats, quantization, evaluation |
| `torch` | Backs sentence-transformers and the ONNX export |
| `sentence-transformers` | Embedding the corpus, and the baseline in the benchmark |
| `transformers` | Tokenizer for the lexical index builder |
| `optimum[onnxruntime-gpu]` | One-line PyTorch → ONNX export |
| `datasets` | Pulling BEIR corpora, queries and qrels |
| `onnx` | Model graph handling |
| `pybind11` | Needed by CMake to build the extension module |

The HTTP service needs three more, installed separately so the core path stays lean:

```bash
pip install -r python/requirements-serve.txt   # fastapi, uvicorn, pydantic
```

### 3. Models — ~170 MB, no training

```bash
python -m zcgvs_tools.export_models --output-dir models
```

| Model | Role | Size | Link |
| --- | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder; 384-dim embeddings for corpus and queries | ~90 MB | [hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder; scores (query, passage) pairs | ~80 MB | [hub](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |

Both are pretrained checkpoints downloaded from the Hub and exported to ONNX with `optimum`.
They share the same BERT uncased vocabulary, which is copied to `models/vocab.txt` and used by
the C++ tokenizer *and* by the lexical index builder — that shared vocabulary is what keeps
term ids consistent across the Python/C++ boundary.

Produces:

```
models/encoder/model.onnx      bi-encoder
models/reranker/model.onnx     cross-encoder
models/vocab.txt               WordPiece vocabulary
```

### 4. Corpus — ~10 MB

```bash
python -m zcgvs_tools.embed_corpus --dataset scifact --output-dir data
```

BEIR datasets share a uniform schema (`corpus`, `queries`, `qrels`), so the same loader serves
any of them. Defaults to SciFact.

| Dataset | Documents | Link |
| --- | --- | --- |
| SciFact (default) | ~5.2k | [hub](https://huggingface.co/datasets/BeIR/scifact) |
| NFCorpus | ~3.6k | [hub](https://huggingface.co/datasets/BeIR/nfcorpus) |
| SciDocs | ~25k | [hub](https://huggingface.co/datasets/BeIR/scidocs) |
| FiQA | ~57k | [hub](https://huggingface.co/datasets/BeIR/fiqa) |

The `qrels` splits give ground-truth relevance, which is what `eval_quality.py` measures
against. Useful overrides:

```bash
DATASET=nfcorpus  bash scripts/generate_artifacts.sh   # different corpus
CORPUS_LIMIT=500  bash scripts/generate_artifacts.sh   # quick smoke subset
EMBED_DEVICE=cpu  bash scripts/generate_artifacts.sh   # embed without a GPU
```

Produces `data/scifact.bin` (the float32 embedding matrix) and `data/scifact.meta` (doc ids
and passage text).

### 5. Derived artifacts — no download, no model

These are computed from files already on disk. `generate_artifacts.sh` runs both steps.

```bash
python -m zcgvs_tools.quantize --input data/scifact.bin --dtype float16 int8 binary

python -m zcgvs_tools.build_lexical \
    --metadata data/scifact.meta --vocab models/vocab.txt \
    --tokenizer-dir models/encoder --output data/scifact.lex
```

| File | Contents | Size vs float32 |
| --- | --- | --- |
| `data/scifact.bin` | float32 matrix | 1× (≈ 8 MB for SciFact) |
| `data/scifact.float16.bin` | half precision | 1/2 |
| `data/scifact.int8.bin` | symmetric int8 + scale | 1/4 |
| `data/scifact.binary.bin` | packed sign bits | **1/32** |
| `data/scifact.meta` | doc ids + passage text | — |
| `data/scifact.lex` | CSR BM25 inverted index | — |

`quantize` refuses to run on anything but a float32 source, so a quantized file can never be
re-quantized by accident.

## Verifying the download

```bash
tree -L 2 third_party models data
```

Expected:

```
third_party/
  onnxruntime -> onnxruntime-linux-x64-gpu-1.20.1
models/
  encoder/model.onnx
  reranker/model.onnx
  vocab.txt
data/
  scifact.bin  scifact.float16.bin  scifact.int8.bin  scifact.binary.bin
  scifact.meta  scifact.lex
```

Inspect any embedding file's header without loading it into VRAM:

```bash
PYTHONPATH=python python -c "
from zcgvs_tools import formats
for name in ['', '.float16', '.int8', '.binary']:
    print(name or '.float32', formats.read_embedding_header(f'data/scifact{name}.bin'))"
```

## Disk and VRAM budget

| Item | Disk | VRAM when loaded |
| --- | --- | --- |
| ONNX Runtime | ~1.5 GB extracted | — |
| Python `.venv` | ~3 GB | — |
| Models | ~170 MB | ~200 MB (both ONNX sessions) |
| SciFact float32 corpus | ~8 MB | ~8 MB |
| SciFact binary + float32 rescore | ~8.3 MB | ~8.3 MB |
| SciFact lexical index | ~10 MB | ~10 MB |
| Search scratch | — | grows with `batch × corpus_size × 4 B` |

An 8 GB card handles SciFact comfortably; the scan scratch is the term that grows, at
`batch × N × 4` bytes for the dense scores and the same again if hybrid is on.

## Offline / air-gapped

Once `third_party/`, `.venv`, `models/` and `data/` exist, nothing else reaches the network.
To move the setup to an offline machine, copy those four directories. Set `HF_HOME` before
the first run if you want the Hugging Face cache somewhere specific, and
`HF_HUB_OFFLINE=1` afterwards to guarantee no further calls are attempted.

Next: [RUN.md](RUN.md).
