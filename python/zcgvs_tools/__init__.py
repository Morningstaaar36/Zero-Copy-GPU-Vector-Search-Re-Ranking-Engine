"""Offline tooling for the Zero-Copy GPU Vector Search & Re-Ranking Engine.

These modules run in Python and produce the artifacts the C++/CUDA engine loads at startup:
the ONNX models, the WordPiece vocabulary, the embedding matrix (in any of the four storage
precisions), the corpus metadata, and the CSR inverted index behind the hybrid stage.

Submodules are imported on demand rather than eagerly, so ``from zcgvs_tools import formats``
works with nothing but NumPy installed — the wire-format code and its tests must not drag in
torch, datasets or the built extension module.
"""

__all__ = [
    "beir_loader",     # needs `datasets`
    "build_lexical",   # needs `transformers` only with --tokenizer-dir
    "config",          # needs the built `zcgvs_engine` module
    "embed_corpus",    # needs `sentence-transformers`
    "export_models",   # needs `optimum`
    "formats",         # needs `numpy` only
    "quantize",        # needs `numpy` only
    "service",         # needs `fastapi` and the built `zcgvs_engine` module
]
