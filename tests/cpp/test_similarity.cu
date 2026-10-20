#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_reference.hpp"
#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/similarity.cuh"

int main() {
    const int doc_count = 4096;
    const int dim = 384;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> corpus(static_cast<std::size_t>(doc_count) * dim);
    std::vector<float> query(static_cast<std::size_t>(dim));
    for (float& value : corpus) {
        value = dist(rng);
    }
    for (float& value : query) {
        value = dist(rng);
    }
    for (int i = 0; i < doc_count; ++i) {
        zcgvs::testing::l2Normalize(corpus, i * dim, dim);
    }
    zcgvs::testing::l2Normalize(query, 0, dim);

    float* d_corpus = nullptr;
    float* d_query = nullptr;
    float* d_scores = nullptr;
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_corpus, sizeof(float) * corpus.size()));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_query, sizeof(float) * query.size()));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * doc_count));
    ZCGVS_CUDA_CHECK(cudaMemcpy(d_corpus, corpus.data(), sizeof(float) * corpus.size(),
                                cudaMemcpyHostToDevice));
    ZCGVS_CUDA_CHECK(cudaMemcpy(d_query, query.data(), sizeof(float) * query.size(),
                                cudaMemcpyHostToDevice));

    zcgvs::cuda::launchSimilarity(d_corpus, zcgvs::ScalarType::Float32, 1.0f, doc_count, dim,
                                  d_query, 1, nullptr, zcgvs::Metric::Cosine, d_scores, 0);
    ZCGVS_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu(static_cast<std::size_t>(doc_count));
    ZCGVS_CUDA_CHECK(cudaMemcpy(gpu.data(), d_scores, sizeof(float) * doc_count,
                                cudaMemcpyDeviceToHost));

    const std::vector<float> reference =
        zcgvs::testing::cpuCosineSimilarity(corpus, doc_count, dim, query);

    float max_abs_error = 0.0f;
    for (int i = 0; i < doc_count; ++i) {
        max_abs_error = std::max(max_abs_error, std::fabs(gpu[static_cast<std::size_t>(i)] -
                                                          reference[static_cast<std::size_t>(i)]));
    }

    cudaFree(d_corpus);
    cudaFree(d_query);
    cudaFree(d_scores);

    const float tolerance = 1e-3f;
    if (max_abs_error > tolerance) {
        std::printf("similarity FAIL max_abs_error=%g\n", static_cast<double>(max_abs_error));
        return 1;
    }
    std::printf("similarity PASS max_abs_error=%g\n", static_cast<double>(max_abs_error));
    return 0;
}
