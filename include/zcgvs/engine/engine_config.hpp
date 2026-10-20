#pragma once

#include <string>

#include "zcgvs/core/metric.hpp"
#include "zcgvs/search/search_params.hpp"

namespace zcgvs {

enum class RerankerBackend { Onnx };

struct EngineConfig {
    std::string embeddings_path;
    std::string metadata_path;
    std::string vocab_path;

    // Optional exact-precision copy of the same corpus in the same document order. When set,
    // the primary (quantized) scan produces a coarse candidate set that a device-side gather
    // kernel re-scores against these vectors before the final top-k.
    std::string rescore_embeddings_path;

    // Optional CSR inverted index; enables the hybrid BM25 + dense retrieval path.
    std::string lexical_index_path;

    std::string encoder_model_path;
    std::string encoder_output_name = "last_hidden_state";
    int encoder_max_seq_len = 256;

    // Leaving reranker_model_path empty runs the engine as a pure retrieval service.
    RerankerBackend reranker_backend = RerankerBackend::Onnx;
    std::string reranker_model_path;
    std::string reranker_output_name = "logits";
    int reranker_max_seq_len = 256;
    int reranker_batch_size = 32;

    int device_id = 0;
    bool lowercase = true;

    Metric metric = Metric::Cosine;
    SimilarityBackend similarity_backend = SimilarityBackend::Custom;

    int default_top_k = 10;
    int default_candidates = 100;

    // Coarse candidate width for the rescore stage; 0 disables it even when a rescore store
    // is configured.
    int rescore_candidates = 0;

    // Whether the BM25 stage runs by default. Attaching a lexical index makes hybrid
    // retrieval *available*; this makes it the default, so a config states its intent rather
    // than having it inferred from which files it happens to reference.
    bool hybrid = false;

    // Convex-combination weight on the dense score when the hybrid path is active.
    float dense_weight = 0.7f;

    // Upper bound on distinct query terms fed to the BM25 stage.
    int max_query_terms = 64;

    // Replay the device pipeline from an instantiated CUDA graph instead of relaunching each
    // kernel individually.
    bool use_cuda_graphs = false;
};

}  // namespace zcgvs
