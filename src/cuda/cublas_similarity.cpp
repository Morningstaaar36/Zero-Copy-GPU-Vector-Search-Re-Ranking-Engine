#include "zcgvs/cuda/cublas_similarity.hpp"

#include "zcgvs/core/cublas_check.hpp"

namespace zcgvs {

CublasSimilarity::CublasSimilarity() { ZCGVS_CUBLAS_CHECK(cublasCreate(&handle_)); }

CublasSimilarity::~CublasSimilarity() {
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
    }
}

bool CublasSimilarity::supports(ScalarType dtype, Metric metric, bool has_mask) {
    if (has_mask || dtype != ScalarType::Float32) {
        return false;
    }
    return metric == Metric::Cosine || metric == Metric::DotProduct;
}

void CublasSimilarity::run(const float* d_corpus,
                           int doc_count,
                           int dim,
                           const float* d_queries,
                           int batch,
                           float* d_scores,
                           cudaStream_t stream) {
    if (doc_count <= 0 || batch <= 0 || dim <= 0) {
        return;
    }

    ZCGVS_CUBLAS_CHECK(cublasSetStream(handle_, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // The row-major [doc_count, dim] corpus is a column-major [dim, doc_count] matrix with
    // leading dimension `dim`, and likewise for the queries. Scoring is therefore
    // corpus^T * queries, whose column-major [doc_count, batch] result is exactly the
    // row-major [batch, doc_count] score matrix the top-k pass expects.
    if (batch == 1) {
        ZCGVS_CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_T, dim, doc_count, &alpha, d_corpus, dim,
                                       d_queries, 1, &beta, d_scores, 1));
        return;
    }

    ZCGVS_CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, doc_count, batch, dim, &alpha,
                                   d_corpus, dim, d_queries, dim, &beta, d_scores, doc_count));
}

}  // namespace zcgvs
