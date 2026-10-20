# Wire formats

Three little-endian binary formats, each written by `python/zcgvs_tools/formats.py` and read by
the C++ side. The Python writer and the C++ reader are the two halves of one contract; the
tests in `tests/python/test_formats.py` and `tests/cpp/test_quantized.cu` exist to keep them
honest.

| File | Magic | Written by | Read by |
| --- | --- | --- | --- |
| `*.bin` | `ZCVE` | `formats.write_embeddings` | `EmbeddingStore::loadFromFile` |
| `*.meta` | `ZCVM` | `formats.write_metadata` | `Corpus::load` |
| `*.lex` | `ZCVL` | `formats.write_lexical_index` | `LexicalIndex::loadFromFile` |

Keeping metadata out of the embedding file means the reranker's passage lookup is a plain host
array index by the device-selected candidate index — no parsing, no offset table.

---

## Embedding matrix — `ZCVE`

Header is 36 bytes at version 2, then the row-major matrix.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | char[4] | magic `'Z','C','V','E'` |
| 4 | 4 | uint32 | version (currently 2) |
| 8 | 4 | uint32 | flags — bit 0 set means rows are L2-normalized |
| 12 | 8 | int64 | `doc_count` (N) |
| 20 | 4 | int32 | `dim` (D) |
| 24 | 4 | int32 | dtype code |
| 28 | 4 | float | `scale` — int8 dequantization factor, 1.0 otherwise |
| 32 | 4 | int32 | `row_stride` — bytes per document row |
| 36 | N × row_stride | — | the matrix |

| dtype code | Name | `row_stride` | Element |
| ---: | --- | --- | --- |
| 0 | `float32` | `4·D` | IEEE binary32 |
| 1 | `float16` | `2·D` | IEEE binary16 |
| 2 | `int8` | `D` | signed byte, dequantized as `q · scale` |
| 3 | `binary` | `4·ceil(D/32)` | packed sign bits, uint32 words |

`row_stride` is redundant — it is derivable from dtype and dim — and is stored anyway so the
reader can reject a file whose header does not agree with itself. `EmbeddingStore::loadFromFile`
does exactly that.

The `normalized` flag records that rows are unit vectors, which is what lets the scan treat a
dot product as a cosine without renormalizing.

### Version 1 compatibility

Version 1 had no `scale` or `row_stride` fields (28-byte header) and only supported dtype codes
0 and 1. The reader still accepts it, deriving the stride and defaulting the scale to 1.0, and
rejects any version-1 file claiming a quantized dtype. Writers always emit version 2.

### int8 quantization

Symmetric, one scale for the whole matrix:

```
scale = max|x| / 127
q     = clamp(round(x / scale), -127, 127)
x̂     = q · scale
```

`-128` is never emitted, which keeps the range symmetric around zero. The query is **not**
quantized — the scan dequantizes each corpus element into a register and multiplies by an fp32
query component. Asymmetric quantization like this is both more accurate and simpler than
quantizing both sides.

The CUDA converter (`launchConvertF32ToI8`, used by `add_documents`) computes
`inverse = 1/scale` once and evaluates `rintf(x · inverse)`; the CPU reference in
`tests/cpp/cpu_reference.hpp` reproduces that exact sequence rather than `x / scale`, so the
comparison is bit-for-bit rather than merely close.

### Binary packing

One bit per dimension, packed into `ceil(D/32)` uint32 words per row:

```
bit b of word w  ↔  dimension w·32 + b
bit is set       ⟺  value ≥ 0
dimensions ≥ D   →  left zero
```

Both the corpus and the query are packed the same way, so scoring is

```
hamming = Σ popcount(doc_word ^ query_word)
score   = (D - 2·hamming) / D        ∈ [-1, 1]
```

The zero padding is load-bearing: pad bits are zero on both sides, so they XOR to zero and
never contribute to the distance. `test_pack_binary_bit_order_and_padding` and
`test_quantized.cu` both check this, from opposite sides.

`Engine::addDocuments` packs new rows on the device with `launchPackBinary`, using the same
convention.

---

## Corpus metadata — `ZCVM`

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | char[4] | magic `'Z','C','V','M'` |
| 4 | 4 | uint32 | version (currently 1) |
| 8 | 8 | int64 | `doc_count` — must match the embedding file |

Then, per document in corpus order:

| Size | Type | Field |
| ---: | --- | --- |
| 4 | uint32 | `id_len` |
| `id_len` | bytes | doc id, UTF-8 |
| 4 | uint32 | `text_len` |
| `text_len` | bytes | passage text, UTF-8 |

Document order is the index space everything else uses: the row index in the embedding matrix,
the value emitted by the top-K kernel, the doc index in a lexical posting, and the position in
an allow-mask are all the same integer.

`Corpus::load` refuses to proceed if `doc_count` disagrees with the embedding file, which
catches the most likely artifact-mismatch mistake immediately rather than as garbled results.

---

## Lexical index — `ZCVL`

A CSR inverted index over WordPiece term ids. Header is 40 bytes.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | char[4] | magic `'Z','C','V','L'` |
| 4 | 4 | uint32 | version (currently 1) |
| 8 | 8 | int64 | `doc_count` — must be ≥ the corpus size |
| 16 | 4 | int32 | `vocab_size` (V) |
| 20 | 8 | int64 | `nnz` — total postings |
| 28 | 4 | float | `k1` |
| 32 | 4 | float | `b` |
| 36 | 4 | float | `avg_doc_len` |
| 40 | 8(V+1) | int64[] | `term_offsets` — CSR row pointers |
| … | 4·nnz | int32[] | `posting_doc` |
| … | 4·nnz | float[] | `posting_weight` |

Total size is exactly `40 + 8(V+1) + 8·nnz`.

`term_offsets` is ascending with `term_offsets[0] == 0` and `term_offsets[V] == nnz`; the reader
validates both and rejects the file otherwise. Postings for term `t` occupy
`[term_offsets[t], term_offsets[t+1])`.

### Why the weights are precomputed

`posting_weight` holds the **complete** BM25 contribution of a (term, document) pair:

```
w(t,d) = idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

Everything — idf, term-frequency saturation, document-length normalization — is folded in
offline. The device kernel therefore does nothing but

```cpp
atomicAdd(&scores[docs[p]], weights[p]);
```

no divisions, no logarithms, no per-document length lookups, no branching. `k1`, `b` and
`avg_doc_len` are stored purely so the index is self-describing and reproducible; the query
path never reads them.

The idf form above is the Lucene/BM25+ variant, which stays positive even for terms appearing
in more than half the corpus — the classic BM25 idf goes negative there and lets a common term
*subtract* from a document's score.

### Terms are WordPiece ids

Terms are ids from the same `vocab.txt` the engine tokenizes queries with. That choice removes
the usual hybrid-search failure mode where an offline analyzer and an online analyzer drift
apart: there is only one vocabulary, shipped with the models.

`build_lexical.py` will use the exported HuggingFace tokenizer when given `--tokenizer-dir`,
and otherwise falls back to a pure-Python WordPiece pass mirroring `src/text/tokenizer.cpp`.
The two agree on ordinary English text. Where they differ, a query term simply fails to find
postings and contributes nothing — the degradation is graceful, never incorrect.

---

## Reading a file without a GPU

```python
from zcgvs_tools import formats

print(formats.read_embedding_header("data/scifact.binary.bin"))
# {'version': 2, 'normalized': True, 'doc_count': 5183, 'dim': 384,
#  'dtype': 'binary', 'scale': 1.0, 'row_stride': 48, 'bytes': 248784}

matrix, normalized = formats.read_embeddings("data/scifact.bin")
doc_ids, texts = formats.read_metadata("data/scifact.meta")
index = formats.read_lexical_index("data/scifact.lex")
```

`read_embeddings` returns int8 matrices already dequantized to float32, and binary matrices as
the raw packed uint32 words — unpacking those to floats would invent precision that is not
there.

## Changing a format

Both sides and both test suites move together:

1. `python/zcgvs_tools/formats.py` — writer and reader
2. `include/zcgvs/data/binary_format.hpp` — the layout comment and the constants
3. The corresponding `loadFromFile`
4. `tests/python/test_formats.py` — round-trip and size assertions
5. `tests/cpp/*.cu` — if the change touches quantization or packing

Bump the version constant and keep the old branch readable. A silently reinterpreted file is
far worse than a rejected one.
