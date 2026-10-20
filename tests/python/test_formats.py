"""Wire-format round-trip checks. Pure NumPy — no GPU, no models, no build required.

Runs standalone (``python tests/python/test_formats.py``) or under pytest. The point is to
pin the byte layout and the quantization conventions that ``include/zcgvs/data/binary_format.hpp``
and the CUDA converters depend on.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from zcgvs_tools import formats  # noqa: E402


def _matrix(rows: int = 64, dim: int = 130, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_row_stride_matches_dtype():
    assert formats.row_stride(formats.DTYPE_FLOAT32, 384) == 384 * 4
    assert formats.row_stride(formats.DTYPE_FLOAT16, 384) == 384 * 2
    assert formats.row_stride(formats.DTYPE_INT8, 384) == 384
    assert formats.row_stride(formats.DTYPE_BINARY, 384) == 12 * 4
    # A dimension that is not a multiple of 32 pads up to a whole word.
    assert formats.row_stride(formats.DTYPE_BINARY, 130) == 5 * 4


def test_pack_binary_bit_order_and_padding():
    dim = 130
    matrix = np.full((1, dim), -1.0, dtype=np.float32)
    matrix[0, 0] = 1.0     # word 0, bit 0
    matrix[0, 33] = 1.0    # word 1, bit 1
    matrix[0, 129] = 1.0   # word 4, bit 1

    packed = formats.pack_binary(matrix)
    assert packed.shape == (1, 5)
    assert packed[0, 0] == 1
    assert packed[0, 1] == 2
    assert packed[0, 2] == 0
    assert packed[0, 3] == 0
    assert packed[0, 4] == 2

    # Padding dimensions 130..159 must stay zero or Hamming distances would be wrong.
    assert packed[0, 4] >> 2 == 0


def test_int8_quantization_is_symmetric_and_bounded():
    matrix = _matrix()
    quantized, scale = formats.quantize_int8(matrix)
    assert quantized.dtype == np.int8
    assert quantized.min() >= -127 and quantized.max() <= 127
    assert scale > 0.0

    dequantized = quantized.astype(np.float32) * np.float32(scale)
    assert np.max(np.abs(dequantized - matrix)) <= scale


def test_embedding_roundtrip_every_dtype():
    matrix = _matrix()
    with tempfile.TemporaryDirectory() as directory:
        for dtype in ("float32", "float16", "int8", "binary"):
            path = Path(directory) / f"corpus.{dtype}.bin"
            formats.write_embeddings(path, matrix, normalized=True, dtype=dtype)

            header = formats.read_embedding_header(path)
            assert header["version"] == formats.EMBEDDING_VERSION
            assert header["dtype"] == dtype
            assert header["doc_count"] == matrix.shape[0]
            assert header["dim"] == matrix.shape[1]
            assert header["normalized"] is True
            assert header["row_stride"] == formats.row_stride(header["dtype_code"], header["dim"])
            assert path.stat().st_size == 36 + header["bytes"]

            stored, normalized = formats.read_embeddings(path)
            assert normalized is True
            if dtype == "binary":
                assert np.array_equal(stored, formats.pack_binary(matrix))
            elif dtype == "int8":
                assert np.max(np.abs(stored - matrix)) <= header["scale"]
            elif dtype == "float16":
                assert np.max(np.abs(stored.astype(np.float32) - matrix)) < 1e-3
            else:
                assert np.array_equal(stored, matrix)


def test_binary_storage_is_32x_smaller_than_float32():
    matrix = _matrix(rows=256, dim=384)
    with tempfile.TemporaryDirectory() as directory:
        sizes = {}
        for dtype in ("float32", "float16", "int8", "binary"):
            path = Path(directory) / f"c.{dtype}.bin"
            formats.write_embeddings(path, matrix, normalized=True, dtype=dtype)
            sizes[dtype] = formats.read_embedding_header(path)["bytes"]

    assert sizes["float32"] / sizes["float16"] == 2
    assert sizes["float32"] / sizes["int8"] == 4
    assert sizes["float32"] / sizes["binary"] == 32


def test_metadata_roundtrip_handles_utf8():
    doc_ids = ["a", "b-2", "ünïcode"]
    texts = ["first passage", "sécond", "third — with an em dash"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "corpus.meta"
        formats.write_metadata(path, doc_ids, texts)
        loaded_ids, loaded_texts = formats.read_metadata(path)
    assert loaded_ids == doc_ids
    assert loaded_texts == texts


def test_lexical_index_roundtrip():
    vocab_size = 40
    offsets = np.zeros(vocab_size + 1, dtype=np.int64)
    docs: list[int] = []
    weights: list[float] = []
    rng = np.random.default_rng(7)
    for term in range(vocab_size):
        for _ in range(int(rng.integers(0, 5))):
            docs.append(int(rng.integers(0, 100)))
            weights.append(float(rng.random()))
        offsets[term + 1] = len(docs)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "corpus.lex"
        formats.write_lexical_index(
            path,
            doc_count=100,
            vocab_size=vocab_size,
            offsets=offsets,
            posting_docs=np.asarray(docs, dtype=np.int32),
            posting_weights=np.asarray(weights, dtype=np.float32),
            k1=1.2,
            b=0.75,
            average_doc_length=42.5,
        )
        index = formats.read_lexical_index(path)

    assert index["doc_count"] == 100
    assert index["vocab_size"] == vocab_size
    assert index["nnz"] == len(docs)
    assert np.array_equal(index["offsets"], offsets)
    assert np.array_equal(index["posting_docs"], np.asarray(docs, dtype=np.int32))
    assert np.allclose(index["posting_weights"], np.asarray(weights, dtype=np.float32))
    assert abs(index["average_doc_length"] - 42.5) < 1e-5


def main() -> int:
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL  {name}: {error}")
    print(f"\nformats: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
