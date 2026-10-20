#include "zcgvs/rerank/reranker_factory.hpp"

#include <stdexcept>

#include "zcgvs/engine/engine_config.hpp"
#include "zcgvs/rerank/onnx_reranker.hpp"

namespace zcgvs {

std::unique_ptr<Reranker> makeReranker(const EngineConfig& config, const Tokenizer& tokenizer) {
    switch (config.reranker_backend) {
        case RerankerBackend::Onnx:
            return std::make_unique<OnnxReranker>(config.reranker_model_path, config.device_id,
                                                  tokenizer, config.reranker_max_seq_len,
                                                  config.reranker_output_name,
                                                  config.reranker_batch_size);
    }
    throw std::runtime_error("zcgvs: unsupported reranker backend");
}

}  // namespace zcgvs
