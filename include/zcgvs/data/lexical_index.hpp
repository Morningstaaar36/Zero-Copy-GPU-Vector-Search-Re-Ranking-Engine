#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "zcgvs/core/device_buffer.hpp"

namespace zcgvs {

// A device-resident CSR inverted index over WordPiece term ids. Posting weights carry the
// complete BM25 contribution of a (term, document) pair, so query-time scoring is a pure
// scatter-add and needs no corpus statistics on the device.
class LexicalIndex {
public:
    LexicalIndex() = default;

    static LexicalIndex loadFromFile(const std::string& path);

    const std::int64_t* deviceOffsets() const { return offsets_.data(); }
    const int* devicePostingDocs() const { return posting_docs_.data(); }
    const float* devicePostingWeights() const { return posting_weights_.data(); }

    std::int64_t docCount() const { return doc_count_; }
    std::int64_t postingCount() const { return posting_count_; }
    int vocabSize() const { return vocab_size_; }
    float k1() const { return k1_; }
    float b() const { return b_; }
    float averageDocLength() const { return average_doc_length_; }

    std::size_t deviceBytes() const {
        return offsets_.bytes() + posting_docs_.bytes() + posting_weights_.bytes();
    }

private:
    DeviceBuffer<std::int64_t> offsets_;
    DeviceBuffer<int> posting_docs_;
    DeviceBuffer<float> posting_weights_;

    std::int64_t doc_count_ = 0;
    std::int64_t posting_count_ = 0;
    int vocab_size_ = 0;
    float k1_ = 1.2f;
    float b_ = 0.75f;
    float average_doc_length_ = 0.0f;
};

}  // namespace zcgvs
