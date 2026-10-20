#include "zcgvs/embed/onnx_query_embedder.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "zcgvs/onnx/onnx_session.hpp"
#include "zcgvs/text/tokenizer.hpp"

namespace zcgvs {

OnnxQueryEmbedder::OnnxQueryEmbedder(const std::string& model_path,
                                     int device_id,
                                     const Tokenizer& tokenizer,
                                     int max_seq_len,
                                     std::string output_name,
                                     int expected_dim)
    : session_(std::make_unique<OnnxSession>(model_path, device_id)),
      tokenizer_(tokenizer),
      max_seq_len_(max_seq_len),
      output_name_(std::move(output_name)),
      dim_(expected_dim) {}

OnnxQueryEmbedder::~OnnxQueryEmbedder() = default;

std::vector<float> OnnxQueryEmbedder::embed(const std::string& text) const {
    const Encoding encoding = tokenizer_.encode(text, max_seq_len_);
    const std::int64_t length = static_cast<std::int64_t>(encoding.length());

    std::vector<OnnxInt64Input> inputs;
    inputs.push_back({"input_ids", {1, length}, encoding.input_ids});
    inputs.push_back({"attention_mask", {1, length}, encoding.attention_mask});
    inputs.push_back({"token_type_ids", {1, length}, encoding.token_type_ids});

    const OnnxTensorF output = session_->run(inputs, output_name_);
    if (output.shape.size() != 3) {
        throw std::runtime_error("zcgvs embedder: expected a [1, L, H] hidden-state tensor");
    }

    const std::int64_t seq = output.shape[1];
    const std::int64_t hidden = output.shape[2];
    if (hidden != dim_) {
        throw std::runtime_error("zcgvs embedder: model hidden size does not match corpus dim");
    }

    std::vector<float> pooled(static_cast<std::size_t>(hidden), 0.0f);
    float denominator = 0.0f;
    for (std::int64_t t = 0; t < seq; ++t) {
        if (encoding.attention_mask[static_cast<std::size_t>(t)] == 0) {
            continue;
        }
        denominator += 1.0f;
        const std::size_t base = static_cast<std::size_t>(t * hidden);
        for (std::int64_t h = 0; h < hidden; ++h) {
            pooled[static_cast<std::size_t>(h)] += output.data[base + static_cast<std::size_t>(h)];
        }
    }
    if (denominator <= 0.0f) {
        denominator = 1.0f;
    }

    float norm = 0.0f;
    for (float& value : pooled) {
        value /= denominator;
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& value : pooled) {
            value /= norm;
        }
    }
    return pooled;
}

}  // namespace zcgvs
