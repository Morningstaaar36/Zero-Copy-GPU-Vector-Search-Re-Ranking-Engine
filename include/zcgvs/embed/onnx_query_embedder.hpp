#pragma once

#include <memory>
#include <string>
#include <vector>

#include "zcgvs/embed/query_embedder.hpp"

namespace zcgvs {

class OnnxSession;
class Tokenizer;

class OnnxQueryEmbedder : public QueryEmbedder {
public:
    OnnxQueryEmbedder(const std::string& model_path,
                      int device_id,
                      const Tokenizer& tokenizer,
                      int max_seq_len,
                      std::string output_name,
                      int expected_dim);

    ~OnnxQueryEmbedder() override;

    int dim() const override { return dim_; }

    std::vector<float> embed(const std::string& text) const override;

private:
    std::unique_ptr<OnnxSession> session_;
    const Tokenizer& tokenizer_;
    int max_seq_len_;
    std::string output_name_;
    int dim_;
};

}  // namespace zcgvs
