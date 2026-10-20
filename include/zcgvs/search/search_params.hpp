#pragma once

#include <cstdint>

#include "zcgvs/core/metric.hpp"

namespace zcgvs {

// Selects who computes the brute-force score matrix. Custom is the hand-written kernel and
// the only path that supports quantized storage, masks and L2; Cublas is the vendor GEMM
// reference used to keep the benchmark honest.
enum class SimilarityBackend : std::int32_t {
    Custom = 0,
    Cublas = 1,
};

inline const char* similarityBackendName(SimilarityBackend backend) {
    return backend == SimilarityBackend::Cublas ? "cublas" : "custom";
}

// One device-side search request. Everything here is stable for the lifetime of a call and
// participates in the CUDA graph cache key, because kernel arguments captured by value are
// baked into an instantiated graph.
struct SearchParams {
    int k = 10;
    Metric metric = Metric::Cosine;

    // Optional [doc_count] uint8 allow-mask; masked documents can never enter the top-k.
    const std::uint8_t* d_mask = nullptr;

    // When positive and a rescore store is attached, the primary (typically quantized) scan
    // selects this many candidates and an exact-precision gather kernel re-scores them on the
    // device before the final top-k.
    int rescore_candidates = 0;

    // Flattened [batch, lexical_stride] WordPiece term ids, -1 padded. Ignored unless a
    // lexical index is attached and lexical_stride is positive.
    const int* host_query_terms = nullptr;
    int lexical_stride = 0;

    // Convex-combination weight on the dense score; 1.0 is dense-only.
    float dense_weight = 1.0f;

    SimilarityBackend backend = SimilarityBackend::Custom;
    bool use_graph = false;
};

}  // namespace zcgvs
