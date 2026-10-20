#include "zcgvs/rerank/onnx_reranker.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "zcgvs/onnx/onnx_session.hpp"
#include "zcgvs/text/tokenizer.hpp"

namespace zcgvs {

OnnxReranker::OnnxReranker(const std::string& model_path,
                           int device_id,
                           const Tokenizer& tokenizer,
                           int max_seq_len,
                           std::string output_name,
                           int batch_size)
    : session_(std::make_unique<OnnxSession>(model_path, device_id)),
      tokenizer_(tokenizer),
      max_seq_len_(max_seq_len),
      output_name_(std::move(output_name)),
      batch_size_(batch_size > 0 ? batch_size : 1) {}

OnnxReranker::~OnnxReranker() = default;

std::vector<float> OnnxReranker::scoreBatch(const std::vector<Encoding>& batch) const {
    const std::size_t rows = batch.size();
    std::size_t max_length = 0;
    for (const Encoding& encoding : batch) {
        max_length = std::max(max_length, encoding.length());
    }

    const std::int64_t pad = tokenizer_.padId();
    std::vector<std::int64_t> input_ids(rows * max_length, pad);
    std::vector<std::int64_t> attention_mask(rows * max_length, 0);
    std::vector<std::int64_t> token_type_ids(rows * max_length, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        const Encoding& encoding = batch[r];
        const std::size_t base = r * max_length;
        for (std::size_t t = 0; t < encoding.length(); ++t) {
            input_ids[base + t] = encoding.input_ids[t];
            attention_mask[base + t] = encoding.attention_mask[t];
            token_type_ids[base + t] = encoding.token_type_ids[t];
        }
    }

    const std::int64_t shape_rows = static_cast<std::int64_t>(rows);
    const std::int64_t shape_cols = static_cast<std::int64_t>(max_length);

    std::vector<OnnxInt64Input> inputs;
    inputs.push_back({"input_ids", {shape_rows, shape_cols}, std::move(input_ids)});
    inputs.push_back({"attention_mask", {shape_rows, shape_cols}, std::move(attention_mask)});
    inputs.push_back({"token_type_ids", {shape_rows, shape_cols}, std::move(token_type_ids)});

    const OnnxTensorF output = session_->run(inputs, output_name_);

    std::int64_t labels = 1;
    if (output.shape.size() == 2) {
        labels = output.shape[1];
    }
    if (labels <= 0) {
        labels = 1;
    }

    std::vector<float> scores(rows, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        scores[r] = output.data[static_cast<std::size_t>(static_cast<std::int64_t>(r) * labels)];
    }
    return scores;
}

std::vector<float> OnnxReranker::score(const std::string& query,
                                       const std::vector<std::string>& passages) const {
    std::vector<float> scores;
    scores.reserve(passages.size());

    std::vector<Encoding> batch;
    batch.reserve(static_cast<std::size_t>(batch_size_));

    for (std::size_t start = 0; start < passages.size(); start += static_cast<std::size_t>(batch_size_)) {
        const std::size_t end = std::min(passages.size(), start + static_cast<std::size_t>(batch_size_));
        batch.clear();
        for (std::size_t i = start; i < end; ++i) {
            batch.push_back(tokenizer_.encodePair(query, passages[i], max_seq_len_));
        }
        const std::vector<float> batch_scores = scoreBatch(batch);
        scores.insert(scores.end(), batch_scores.begin(), batch_scores.end());
    }
    return scores;
}

}  // namespace zcgvs
