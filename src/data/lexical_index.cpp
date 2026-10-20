#include "zcgvs/data/lexical_index.hpp"

#include <cuda_runtime.h>

#include <fstream>
#include <stdexcept>
#include <vector>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/data/binary_format.hpp"

namespace zcgvs {

LexicalIndex LexicalIndex::loadFromFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("zcgvs: cannot open lexical index '" + path + "'");
    }

    BinaryReader reader(stream);
    reader.expectMagic(kLexicalMagic);

    const std::uint32_t version = reader.readScalar<std::uint32_t>();
    if (version != kLexicalVersion) {
        throw std::runtime_error("zcgvs: unsupported lexical index version " +
                                 std::to_string(version));
    }

    LexicalIndex index;
    index.doc_count_ = reader.readScalar<std::int64_t>();
    index.vocab_size_ = reader.readScalar<std::int32_t>();
    index.posting_count_ = reader.readScalar<std::int64_t>();
    index.k1_ = reader.readScalar<float>();
    index.b_ = reader.readScalar<float>();
    index.average_doc_length_ = reader.readScalar<float>();

    if (index.doc_count_ <= 0 || index.vocab_size_ <= 0 || index.posting_count_ < 0) {
        throw std::runtime_error("zcgvs: lexical index header describes an empty index");
    }

    const std::size_t offset_count = static_cast<std::size_t>(index.vocab_size_) + 1;
    std::vector<std::int64_t> offsets(offset_count);
    reader.readBytes(reinterpret_cast<char*>(offsets.data()),
                     offset_count * sizeof(std::int64_t));
    if (offsets.front() != 0 || offsets.back() != index.posting_count_) {
        throw std::runtime_error("zcgvs: lexical index CSR offsets are inconsistent with nnz");
    }

    const std::size_t postings = static_cast<std::size_t>(index.posting_count_);
    std::vector<int> posting_docs(postings);
    std::vector<float> posting_weights(postings);
    if (postings > 0) {
        reader.readBytes(reinterpret_cast<char*>(posting_docs.data()), postings * sizeof(int));
        reader.readBytes(reinterpret_cast<char*>(posting_weights.data()), postings * sizeof(float));
    }

    index.offsets_.allocate(offset_count);
    ZCGVS_CUDA_CHECK(cudaMemcpy(index.offsets_.data(), offsets.data(),
                                offset_count * sizeof(std::int64_t), cudaMemcpyHostToDevice));

    if (postings > 0) {
        index.posting_docs_.allocate(postings);
        index.posting_weights_.allocate(postings);
        ZCGVS_CUDA_CHECK(cudaMemcpy(index.posting_docs_.data(), posting_docs.data(),
                                    postings * sizeof(int), cudaMemcpyHostToDevice));
        ZCGVS_CUDA_CHECK(cudaMemcpy(index.posting_weights_.data(), posting_weights.data(),
                                    postings * sizeof(float), cudaMemcpyHostToDevice));
    }

    return index;
}

}  // namespace zcgvs
