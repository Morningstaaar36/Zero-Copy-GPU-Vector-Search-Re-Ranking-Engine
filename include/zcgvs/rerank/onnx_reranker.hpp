#pragma once

#include <memory>
#include <string>
#include <vector>

#include "zcgvs/rerank/reranker.hpp"

namespace zcgvs {

class OnnxSession;
class Tokenizer;
struct Encoding;

class OnnxReranker : public Reranker {
public:
    OnnxReranker(const std::string& model_path,
                 int device_id,
                 const Tokenizer& tokenizer,
                 int max_seq_len,
                 std::string output_name,
                 int batch_size);

    ~OnnxReranker() override;

    std::vector<float> score(const std::string& query,
                             const std::vector<std::string>& passages) const override;

private:
    std::vector<float> scoreBatch(const std::vector<Encoding>& batch) const;

    std::unique_ptr<OnnxSession> session_;
    const Tokenizer& tokenizer_;
    int max_seq_len_;
    std::string output_name_;
    int batch_size_;
};

}  // namespace zcgvs
