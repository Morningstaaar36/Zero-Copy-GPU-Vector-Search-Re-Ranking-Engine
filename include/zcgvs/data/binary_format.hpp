#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <stdexcept>
#include <string>

namespace zcgvs {

// Embedding file layout (little-endian):
//   [4]  magic  = 'Z','C','V','E'
//   [4]  uint32 version
//   [4]  uint32 flags        (bit 0 = rows are L2-normalized)
//   [8]  int64  doc_count    (N)
//   [4]  int32  dim          (D)
//   [4]  int32  dtype        (0 = float32, 1 = float16, 2 = int8, 3 = binary)
//   --- version 2 and later only ---
//   [4]  float  scale        (int8 dequantization scale; 1.0 for the other dtypes)
//   [4]  int32  row_stride   (bytes per document row)
//   ---
//   [N*row_stride] row-major embedding matrix
//
// Version 1 files carry no scale/row_stride and are limited to float32 and float16; they are
// still readable, with scale = 1.0 and row_stride derived from dtype and dim.
//
// Metadata file layout (little-endian):
//   [4]  magic  = 'Z','C','V','M'
//   [4]  uint32 version
//   [8]  int64  doc_count    (N, must match the embedding file)
//   per document, in corpus order:
//     [4] uint32 id_len,   [id_len]   bytes  doc id (utf-8)
//     [4] uint32 text_len, [text_len] bytes  passage text (utf-8)
//
// Lexical index layout (little-endian), a CSR inverted index over WordPiece term ids:
//   [4]  magic  = 'Z','C','V','L'
//   [4]  uint32 version
//   [8]  int64  doc_count    (N, must match the embedding file)
//   [4]  int32  vocab_size   (V)
//   [8]  int64  nnz          (total postings)
//   [4]  float  k1
//   [4]  float  b
//   [4]  float  avg_doc_len
//   [(V+1)*8]  int64  term_offsets    (CSR row pointers, ascending, term_offsets[V] == nnz)
//   [nnz*4]    int32  posting_doc     (corpus index of each posting)
//   [nnz*4]    float  posting_weight  (full BM25 contribution of the posting)

constexpr char kEmbeddingMagic[4] = {'Z', 'C', 'V', 'E'};
constexpr char kMetadataMagic[4] = {'Z', 'C', 'V', 'M'};
constexpr char kLexicalMagic[4] = {'Z', 'C', 'V', 'L'};

constexpr std::uint32_t kEmbeddingVersion = 2;
constexpr std::uint32_t kMetadataVersion = 1;
constexpr std::uint32_t kLexicalVersion = 1;

constexpr std::uint32_t kFlagNormalized = 1u << 0;

constexpr std::int32_t kDtypeFloat32 = 0;
constexpr std::int32_t kDtypeFloat16 = 1;
constexpr std::int32_t kDtypeInt8 = 2;
constexpr std::int32_t kDtypeBinary = 3;

class BinaryReader {
public:
    explicit BinaryReader(std::istream& stream) : stream_(stream) {}

    template <typename T>
    T readScalar() {
        T value{};
        readBytes(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    void readBytes(char* destination, std::size_t count) {
        stream_.read(destination, static_cast<std::streamsize>(count));
        if (!stream_) {
            throw std::runtime_error("zcgvs binary: unexpected end of stream");
        }
    }

    std::string readString(std::size_t length) {
        std::string value;
        value.resize(length);
        if (length > 0) {
            readBytes(value.data(), length);
        }
        return value;
    }

    void expectMagic(const char (&expected)[4]) {
        char actual[4];
        readBytes(actual, 4);
        if (std::memcmp(actual, expected, 4) != 0) {
            throw std::runtime_error("zcgvs binary: magic mismatch");
        }
    }

private:
    std::istream& stream_;
};

}  // namespace zcgvs
