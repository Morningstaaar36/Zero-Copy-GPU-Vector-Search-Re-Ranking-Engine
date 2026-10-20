"""Readers and writers for the zcgvs on-disk binary formats.

The byte layout here is the exact counterpart of ``include/zcgvs/data/binary_format.hpp``.
All scalars are little-endian.

Embedding files are versioned: version 1 held only float32/float16 matrices, version 2 adds
a dequantization scale and an explicit row stride so int8 and packed-binary matrices can be
described. Readers accept both; writers always emit version 2.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

EMBEDDING_MAGIC = b"ZCVE"
METADATA_MAGIC = b"ZCVM"
LEXICAL_MAGIC = b"ZCVL"

EMBEDDING_VERSION = 2
METADATA_VERSION = 1
LEXICAL_VERSION = 1

FLAG_NORMALIZED = 1

DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_INT8 = 2
DTYPE_BINARY = 3

DTYPE_NAMES = {
    "float32": DTYPE_FLOAT32,
    "float16": DTYPE_FLOAT16,
    "int8": DTYPE_INT8,
    "binary": DTYPE_BINARY,
}
DTYPE_CODE_NAMES = {code: name for name, code in DTYPE_NAMES.items()}

_STORAGE_NUMPY = {
    DTYPE_FLOAT32: np.float32,
    DTYPE_FLOAT16: np.float16,
    DTYPE_INT8: np.int8,
    DTYPE_BINARY: np.uint32,
}


def binary_words(dim: int) -> int:
    return (dim + 31) // 32


def row_stride(dtype_code: int, dim: int) -> int:
    if dtype_code == DTYPE_FLOAT32:
        return dim * 4
    if dtype_code == DTYPE_FLOAT16:
        return dim * 2
    if dtype_code == DTYPE_INT8:
        return dim
    if dtype_code == DTYPE_BINARY:
        return binary_words(dim) * 4
    raise ValueError(f"unknown dtype code {dtype_code}")


def pack_binary(matrix: np.ndarray) -> np.ndarray:
    """Sign-bit packing matching ``cuda::launchPackBinary``.

    Bit ``b`` of word ``w`` carries dimension ``w * 32 + b`` and is set when the value is
    non-negative. Dimensions past ``dim`` stay zero so they cancel in any XOR.
    """
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, dim = matrix.shape
    words = binary_words(dim)

    padded = np.zeros((rows, words * 32), dtype=bool)
    padded[:, :dim] = matrix >= 0.0
    bits = padded.reshape(rows, words, 32)

    packed = np.zeros((rows, words), dtype=np.uint32)
    for bit in range(32):
        packed |= bits[:, :, bit].astype(np.uint32) << np.uint32(bit)
    return packed


def quantize_int8(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-matrix quantization matching ``cuda::launchConvertF32ToI8``."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    peak = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    scale = (peak / 127.0) if peak > 0.0 else 1.0
    quantized = np.rint(matrix / np.float32(scale))
    return np.clip(quantized, -127, 127).astype(np.int8), scale


def encode_matrix(matrix: np.ndarray, dtype: str) -> tuple[np.ndarray, int, float]:
    """Converts a float32 matrix into the requested storage form.

    Returns the storage array, its dtype code and the dequantization scale.
    """
    if dtype not in DTYPE_NAMES:
        raise ValueError(f"unknown storage dtype '{dtype}'")
    code = DTYPE_NAMES[dtype]

    if code == DTYPE_FLOAT32:
        return np.ascontiguousarray(matrix, dtype=np.float32), code, 1.0
    if code == DTYPE_FLOAT16:
        return np.ascontiguousarray(matrix, dtype=np.float16), code, 1.0
    if code == DTYPE_INT8:
        quantized, scale = quantize_int8(matrix)
        return quantized, code, scale
    return pack_binary(matrix), code, 1.0


def write_embeddings(
    path: str | Path,
    matrix: np.ndarray,
    normalized: bool,
    dtype: str | None = None,
) -> None:
    """Writes a version-2 embedding file.

    ``matrix`` is a float [doc_count, dim] array; ``dtype`` selects the on-disk storage form
    and the conversion (half rounding, int8 quantization, sign-bit packing) happens here.
    Defaults to the storage form that matches the input array's precision.
    """
    source = np.asarray(matrix)
    if source.ndim != 2:
        raise ValueError("embedding matrix must be 2-D [doc_count, dim]")
    if dtype is None:
        dtype = "float16" if source.dtype == np.float16 else "float32"

    source = np.ascontiguousarray(source, dtype=np.float32)
    doc_count, dim = int(source.shape[0]), int(source.shape[1])

    storage, code, scale = encode_matrix(source, dtype)
    stride = row_stride(code, dim)
    if stride != int(storage.shape[1]) * storage.dtype.itemsize:
        raise ValueError("computed row stride does not match the storage array")

    flags = FLAG_NORMALIZED if normalized else 0
    with open(path, "wb") as stream:
        stream.write(EMBEDDING_MAGIC)
        stream.write(struct.pack("<I", EMBEDDING_VERSION))
        stream.write(struct.pack("<I", flags))
        stream.write(struct.pack("<q", doc_count))
        stream.write(struct.pack("<i", dim))
        stream.write(struct.pack("<i", code))
        stream.write(struct.pack("<f", float(scale)))
        stream.write(struct.pack("<i", stride))
        stream.write(storage.tobytes(order="C"))


def read_embedding_header(path: str | Path) -> dict:
    with open(path, "rb") as stream:
        if stream.read(4) != EMBEDDING_MAGIC:
            raise ValueError("not a zcgvs embedding file")
        (version,) = struct.unpack("<I", stream.read(4))
        (flags,) = struct.unpack("<I", stream.read(4))
        (doc_count,) = struct.unpack("<q", stream.read(8))
        (dim,) = struct.unpack("<i", stream.read(4))
        (code,) = struct.unpack("<i", stream.read(4))
        if version == 1:
            scale, stride = 1.0, row_stride(code, dim)
        elif version == EMBEDDING_VERSION:
            (scale,) = struct.unpack("<f", stream.read(4))
            (stride,) = struct.unpack("<i", stream.read(4))
        else:
            raise ValueError(f"unsupported embedding file version {version}")

    return {
        "version": version,
        "normalized": bool(flags & FLAG_NORMALIZED),
        "doc_count": doc_count,
        "dim": dim,
        "dtype": DTYPE_CODE_NAMES.get(code, str(code)),
        "dtype_code": code,
        "scale": scale,
        "row_stride": stride,
        "bytes": doc_count * stride,
    }


def read_embeddings(path: str | Path) -> tuple[np.ndarray, bool]:
    """Returns the stored matrix in its storage dtype plus the normalized flag.

    Int8 matrices come back dequantized to float32; binary matrices come back as the packed
    uint32 words, since unpacking them to floats would be lossy in the other direction.
    """
    header = read_embedding_header(path)
    offset = 28 if header["version"] == 1 else 36
    code = header["dtype_code"]
    np_dtype = _STORAGE_NUMPY[code]

    columns = header["row_stride"] // np.dtype(np_dtype).itemsize
    with open(path, "rb") as stream:
        stream.seek(offset)
        raw = stream.read(header["doc_count"] * header["row_stride"])
    matrix = np.frombuffer(raw, dtype=np_dtype).reshape(header["doc_count"], columns)

    if code == DTYPE_INT8:
        matrix = matrix.astype(np.float32) * np.float32(header["scale"])
    return matrix, header["normalized"]


def write_metadata(path: str | Path, doc_ids: list[str], texts: list[str]) -> None:
    if len(doc_ids) != len(texts):
        raise ValueError("doc_ids and texts must have equal length")

    with open(path, "wb") as stream:
        stream.write(METADATA_MAGIC)
        stream.write(struct.pack("<I", METADATA_VERSION))
        stream.write(struct.pack("<q", len(doc_ids)))
        for doc_id, text in zip(doc_ids, texts):
            id_bytes = doc_id.encode("utf-8")
            text_bytes = text.encode("utf-8")
            stream.write(struct.pack("<I", len(id_bytes)))
            stream.write(id_bytes)
            stream.write(struct.pack("<I", len(text_bytes)))
            stream.write(text_bytes)


def read_metadata(path: str | Path) -> tuple[list[str], list[str]]:
    with open(path, "rb") as stream:
        if stream.read(4) != METADATA_MAGIC:
            raise ValueError("not a zcgvs metadata file")
        (version,) = struct.unpack("<I", stream.read(4))
        (doc_count,) = struct.unpack("<q", stream.read(8))
        if version != METADATA_VERSION:
            raise ValueError("unsupported metadata version")

        doc_ids: list[str] = []
        texts: list[str] = []
        for _ in range(doc_count):
            (id_len,) = struct.unpack("<I", stream.read(4))
            doc_ids.append(stream.read(id_len).decode("utf-8"))
            (text_len,) = struct.unpack("<I", stream.read(4))
            texts.append(stream.read(text_len).decode("utf-8"))
        return doc_ids, texts


def write_lexical_index(
    path: str | Path,
    doc_count: int,
    vocab_size: int,
    offsets: np.ndarray,
    posting_docs: np.ndarray,
    posting_weights: np.ndarray,
    k1: float,
    b: float,
    average_doc_length: float,
) -> None:
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    posting_docs = np.ascontiguousarray(posting_docs, dtype=np.int32)
    posting_weights = np.ascontiguousarray(posting_weights, dtype=np.float32)

    if offsets.shape[0] != vocab_size + 1:
        raise ValueError("offsets must hold vocab_size + 1 entries")
    if posting_docs.shape[0] != posting_weights.shape[0]:
        raise ValueError("posting arrays must have equal length")
    nnz = int(posting_docs.shape[0])
    if int(offsets[0]) != 0 or int(offsets[-1]) != nnz:
        raise ValueError("CSR offsets are inconsistent with the posting count")

    with open(path, "wb") as stream:
        stream.write(LEXICAL_MAGIC)
        stream.write(struct.pack("<I", LEXICAL_VERSION))
        stream.write(struct.pack("<q", int(doc_count)))
        stream.write(struct.pack("<i", int(vocab_size)))
        stream.write(struct.pack("<q", nnz))
        stream.write(struct.pack("<f", float(k1)))
        stream.write(struct.pack("<f", float(b)))
        stream.write(struct.pack("<f", float(average_doc_length)))
        stream.write(offsets.tobytes(order="C"))
        stream.write(posting_docs.tobytes(order="C"))
        stream.write(posting_weights.tobytes(order="C"))


def read_lexical_index(path: str | Path) -> dict:
    with open(path, "rb") as stream:
        if stream.read(4) != LEXICAL_MAGIC:
            raise ValueError("not a zcgvs lexical index")
        (version,) = struct.unpack("<I", stream.read(4))
        if version != LEXICAL_VERSION:
            raise ValueError(f"unsupported lexical index version {version}")
        (doc_count,) = struct.unpack("<q", stream.read(8))
        (vocab_size,) = struct.unpack("<i", stream.read(4))
        (nnz,) = struct.unpack("<q", stream.read(8))
        (k1,) = struct.unpack("<f", stream.read(4))
        (b,) = struct.unpack("<f", stream.read(4))
        (average_doc_length,) = struct.unpack("<f", stream.read(4))

        offsets = np.frombuffer(stream.read((vocab_size + 1) * 8), dtype=np.int64)
        posting_docs = np.frombuffer(stream.read(nnz * 4), dtype=np.int32)
        posting_weights = np.frombuffer(stream.read(nnz * 4), dtype=np.float32)

    return {
        "doc_count": doc_count,
        "vocab_size": vocab_size,
        "nnz": nnz,
        "k1": k1,
        "b": b,
        "average_doc_length": average_doc_length,
        "offsets": offsets,
        "posting_docs": posting_docs,
        "posting_weights": posting_weights,
    }
