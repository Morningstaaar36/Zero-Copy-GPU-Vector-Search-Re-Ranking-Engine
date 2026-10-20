# Troubleshooting

Grouped by where the failure happens. Each entry states the symptom, the cause, and the fix.

## Setup and download

**`missing required tool: cmake` (or `curl`, `tar`, `python3`).**
`setup_all.sh` checks prerequisites up front. Install the tool and re-run; every script is
idempotent.

**Model download is slow or rate-limited.**
`generate_artifacts.sh` pulls weights from the Hugging Face Hub. Set `HF_HOME` to a persistent
path so the cache survives between runs, or authenticate with `huggingface-cli login` if you
hit anonymous limits. Once `models/` and `data/` exist, nothing re-downloads.

**BEIR dataset download fails.**
`datasets` needs network access on first use. Set `HF_HOME` and retry; for a quick smoke test
use `CORPUS_LIMIT=500 bash scripts/generate_artifacts.sh`, which embeds only the first 500
documents.

## Build

**`Could not find pybind11` during CMake configure.**
Activate the venv first (`source .venv/bin/activate`) so `python -m pybind11 --cmakedir`
resolves. `build.sh` sources it automatically when `.venv` exists.

**`Python.h: No such file or directory`.**
Install the Python development headers for your interpreter: `python3-dev` on Debian/Ubuntu,
`python3-devel` on Fedora/RHEL. Then re-run `scripts/build.sh`.

**`nvcc` not found.**
Install the CUDA Toolkit and put `nvcc` on `PATH` (commonly `/usr/local/cuda/bin`).

**`cannot open source file "onnxruntime_cxx_api.h"` — editor squiggle only.**
ONNX Runtime is not downloaded yet, or IntelliSense has not reloaded. Run
`scripts/download_onnxruntime.sh`, then `scripts/build.sh` (which writes
`compile_commands.json`), then reload the window. `.vscode/c_cpp_properties.json` already points
at `third_party/onnxruntime/include`.

**`cannot find -lcublas` or cuBLAS headers missing.**
cuBLAS ships with the CUDA Toolkit. If `find_package(CUDAToolkit)` succeeds but the target is
missing, the Toolkit install is partial — install the `cuda-libraries-dev` package for your
CUDA version.

**NVTX headers not found with `-DZCGVS_ENABLE_NVTX=ON`.**
`nvtx3/nvToolsExt.h` ships with CUDA 11.4+. On an older Toolkit, leave the option off; it is
off by default and the macro compiles to nothing.

**Compiling takes a long time.**
`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which compiles for your GPU only. If it was
overridden to a list, narrow it: `-DCMAKE_CUDA_ARCHITECTURES=86` for one target.

## Llama layer

**`GatedRepoError` / 401 / 403 downloading `meta-llama/...`.**
Those repositories need you to accept Meta's license on the Hugging Face model page and be
logged in (`hf auth login`). You do not have to: `Llama()` and `download_llama.sh` fall back to
the `unsloth/*` mirror of the same weights and say so on stderr.

**`CUDA out of memory` loading Llama.**
The 3B model needs ~6.8 GB for PyTorch on top of the index and the ONNX sessions (~8.5 GB in
total). On an 8 GB card use `models/llama/Llama-3.2-1B-Instruct` (~3.4 GB), and close other GPU
programs. Do not load two Llama models at once.

**Downloads from the Hub crawl at a few KB/s.**
`hf download` uses a single stream per file and some networks throttle it. Fetching the
`.safetensors` in ranged chunks with several parallel `curl` connections is many times faster; the
files only need to end up in `models/llama/<name>/`.

**The first `verify` is slow, later ones are fast.**
The first call pays PyTorch/CUDA kernel warmup. `fact_check.py` runs one throwaway claim before
printing timings; do the same in your own code before measuring.

**`✗ NOT VERIFIED` in the quote check.**
Expected sometimes — it is the checker doing its job. A quotation was not found word for word in
any retrieved abstract (invented, or paraphrase in quote marks), or it is real text credited to a
different abstract than the `[n]` next to it. About one quote in eight fails for the 3B model
([LLAMA.md](LLAMA.md#the-quote-checker)). Read the evidence printed above the verdict.

**Verdict is `NOT ENOUGH INFO` for a claim you know is true.**
The corpus is 5 183 SciFact abstracts. If none of them addresses the claim, "not enough
information" is the correct answer — add documents with `POST /documents` or build your own
corpus with `scripts/generate_artifacts.sh`.

## Runtime — loading

**`libonnxruntime.so: cannot open shared object file`.**
Put the runtime on the loader path:
`export LD_LIBRARY_PATH=third_party/onnxruntime/lib:$LD_LIBRARY_PATH`. Every script in
`scripts/` does this for you.

**CUDA / cuDNN version mismatch when creating an ONNX session.**
The ONNX Runtime GPU build must match your installed CUDA and cuDNN. Pick a matching release,
e.g. `ONNXRUNTIME_VERSION=1.19.2 bash scripts/download_onnxruntime.sh`, and rebuild. Consult
the ONNX Runtime CUDA execution-provider compatibility matrix for the exact pairings.

**`ModuleNotFoundError: No module named 'zcgvs_engine'`.**
The build output is not on the import path:
`export PYTHONPATH=build/bindings:python:$PYTHONPATH`.

**`zcgvs binary: magic mismatch`.**
A path points at the wrong kind of file — an embedding path pointed at a `.meta`, or a stale
artifact from a different tool. Check the header:
`PYTHONPATH=python python -c "from zcgvs_tools import formats; print(formats.read_embedding_header('data/scifact.bin'))"`

**`zcgvs: unsupported embedding file version 1` / `quantized embeddings require file version 2`.**
The artifact predates the quantization support. Regenerate it:
`python -m zcgvs_tools.embed_corpus --dataset scifact --output-dir data`. Version-1 float files
still load; only quantized dtypes require version 2.

**`zcgvs: metadata doc count does not match embeddings`.**
The `.bin` and `.meta` came from different runs — most often after re-embedding with a
different `CORPUS_LIMIT`. Regenerate both together.

**`zcgvs: embedding row stride does not match dtype and dim`.**
The file was written by a mismatched writer version. Regenerate with the current
`zcgvs_tools`.

**`zcgvs: the rescore store holds fewer documents than the primary corpus`.**
`rescore_embeddings_path` points at a corpus built from a different (or truncated) document
set. Both must come from the same embedding run — `quantize` derives them from one float32
file precisely so this cannot happen by accident.

**`zcgvs: the lexical index holds fewer documents than the corpus`.**
The `.lex` was built against an older `.meta`. Rebuild it with `build_lexical`.

## Runtime — queries

**`zcgvs: k=… exceeds kernel capacity 1024`.**
`top_k`, `candidates` and `rescore_candidates` are capped at 1024 by the shared-memory bitonic
network. Lower the value; there is no configuration that lifts it without changing the kernel.

**`zcgvs: rescore_candidates must be at least k`.**
The coarse stage has to select at least as many candidates as the final selection returns. The
engine clamps this for you when going through `SearchOptions`; the raw `Searcher` API does not.

**`zcgvs: k=… exceeds corpus size`.**
The corpus is smaller than the requested width — common with `CORPUS_LIMIT` set.

**`zcgvs: incremental indexing is unavailable while a rescore store or lexical index is attached`.**
Both artifacts are built offline over a fixed document set and would go stale. Rebuild them
offline, or use a config without them for the live-indexing workflow.

**`zcgvs rescore: the rescore store must hold float32, float16 or int8 rows`.**
`rescore_embeddings_path` points at a binary file. The rescore stage exists to recover
precision, so its store must actually carry some.

**Hybrid mode changes nothing.**
Check three things in order: the config has `lexical_index_path` set (`engine.has_lexical_index`
is `True`); `options.hybrid` is `True`; and `dense_weight` is below 1.0 — at 1.0 the lexical
contribution is multiplied by zero.

**Binary search returns poor results.**
Expected without a rescore store — 1-bit quantization is genuinely lossy. Add
`rescore_embeddings_path` and `rescore_candidates` (start at 512). If it is still poor with
rescoring, the coarse pass is not surfacing the right documents: widen `rescore_candidates`.

**`similarity` scores look like coarse multiples rather than smooth values.**
You are reading raw Hamming proxies, which means the rescore stage did not run. Confirm
`rescore_candidates > 0` and that `rescore_embeddings_path` is set and resolvable.

**Rerank scores are identical to similarity scores.**
The reranker is not running: either `reranker_model_path` is empty or `options.rerank` is
`False`. `engine.has_reranker` tells you which.

**`graph_replay` is False even with `use_graph = True`.**
Graph capture is skipped for the cuBLAS backend. Set `options.backend` to `Custom`, or leave
`use_config_backend` at `True` with a config whose `similarity_backend` is `"custom"`.

**`graph_captures` climbs alongside `graph_launches`.**
Something in the request shape changes every query, so the graph is rebuilt instead of
replayed. The usual causes are a varying `allow` list (the mask pointer is part of the key), a
varying `batch`, or a varying `dense_weight`. Batch queries of the same shape together.

**Results differ slightly between runs.**
The BM25 stage uses `atomicAdd`, so accumulation order — and therefore the last bits of a
float sum — varies. This can flip documents whose fused scores are within rounding distance of
each other. It is inherent to the atomic scatter-add; the dense-only path is deterministic.

## Correctness failures

**`ctest` reports a kernel test failing.**
Run the binary directly for its diagnostic line: `./build/tests/zcgvs_test_quantized` prints
the first mismatching index and both values. Note which side is wrong before changing anything
— the CPU references in `tests/cpp/cpu_reference.hpp` deliberately reproduce the kernel's exact
arithmetic, so a mismatch means a real divergence rather than a tolerance problem.

**`test_formats.py` fails after touching quantization or bit-packing.**
The Python writer and the CUDA converter implement the same contract. Both
`tests/python/test_formats.py` and `tests/cpp/test_quantized.cu` must pass; a change to one
side without the other is exactly the bug that pair exists to catch.

**`pipeline` fails on the graph-equivalence check.**
Replay must be bit-identical to direct launches. A failure means the captured graph is being
replayed with stale pointers — check that any new device buffer added to `recordPipeline` is
covered by the `GraphKey` or by the generation counter in `ensureCapacity`.

## No GPU available

The CUDA kernels and the ONNX CUDA execution provider both require an NVIDIA GPU. Without one
you can still:

- run the wire-format tests: `python tests/python/test_formats.py`
- embed a corpus on CPU: `EMBED_DEVICE=cpu bash scripts/generate_artifacts.sh`
- derive quantized corpora and build the lexical index (both are pure NumPy/Python)
- inspect artifacts with `zcgvs_tools.formats`

`scripts/verify.sh` skips the GPU tiers automatically rather than failing.

## Performance is disappointing

**Check the roofline first.** `./build/apps/zcgvs_bench_kernels` reports achieved GB/s against
theoretical peak. If the scan is near peak, the kernel is done and the only remaining wins are
reading fewer bytes (quantize) or reusing them more (batch).

**Confirm you are measuring the warm path.** The first query pays ONNX session warmup, CUDA
context creation, lazy scratch allocation and graph capture. Use `--repeat` on the CLI and
`--repeats` in the Python benchmark, and take the median.

**Look at the stage breakdown.** On a small corpus the cross-encoder dominates by a wide
margin. `search_gpu_ms` is the number that reflects kernel quality; `total_ms` mostly reflects
the reranker. Both are worth quoting, but only if you say which is which.

**A 5k-document corpus does not exercise the kernels.** Use `DATASET=fiqa` (~57k) or the
synthetic benchmark at `--docs 10000000`.
