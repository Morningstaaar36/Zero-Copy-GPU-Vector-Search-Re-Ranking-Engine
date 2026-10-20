#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "zcgvs/core/metric.hpp"
#include "zcgvs/core/scalar_type.hpp"

namespace zcgvs {

// Vendor-library reference path for the brute-force scan: the same [batch, doc_count] score
// matrix the hand-written kernel produces, computed as a GEMM. Exists so the custom kernel
// can be benchmarked against a strong baseline rather than against nothing.
class CublasSimilarity {
public:
    CublasSimilarity();
    ~CublasSimilarity();

    CublasSimilarity(const CublasSimilarity&) = delete;
    CublasSimilarity& operator=(const CublasSimilarity&) = delete;

    // The GEMM path has no mask predicate and no quantized-storage decode, so it only covers
    // float32 inner-product metrics. Everything else falls back to the custom kernel.
    static bool supports(ScalarType dtype, Metric metric, bool has_mask);

    void run(const float* d_corpus,
             int doc_count,
             int dim,
             const float* d_queries,
             int batch,
             float* d_scores,
             cudaStream_t stream);

private:
    cublasHandle_t handle_ = nullptr;
};

}  // namespace zcgvs
