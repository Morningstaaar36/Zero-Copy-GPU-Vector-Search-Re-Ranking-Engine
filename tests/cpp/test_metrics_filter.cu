#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/similarity.cuh"

namespace {

std::vector<float> cpuL2(const std::vector<float>& corpus, int doc_count, int dim,
                         const std::vector<float>& query) {
    std::vector<float> scores(static_cast<std::size_t>(doc_count));
    for (int i = 0; i < doc_count; ++i) {
        float sum = 0.0f;
        const std::size_t base = static_cast<std::size_t>(i) * dim;
        for (int j = 0; j < dim; ++j) {
            const float diff = query[static_cast<std::size_t>(j)] - corpus[base + j];
            sum += diff * diff;
        }
        scores[static_cast<std::size_t>(i)] = -sum;
    }
    return scores;
}

}  // namespace

int main() {
    const int doc_count = 2048;
    const int dim = 128;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> corpus(static_cast<std::size_t>(doc_count) * dim);
    std::vector<float> query(static_cast<std::size_t>(dim));
    for (float& v : corpus) {
        v = dist(rng);
    }
    for (float& v : query) {
        v = dist(rng);
    }

    std::vector<std::uint8_t> mask(static_cast<std::size_t>(doc_count), 1);
    for (int i = 0; i < doc_count; i += 2) {
        mask[static_cast<std::size_t>(i)] = 0;
    }

    float* d_corpus = nullptr;
    float* d_query = nullptr;
    float* d_scores = nullptr;
    std::uint8_t* d_mask = nullptr;
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_corpus, sizeof(float) * corpus.size()));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_query, sizeof(float) * query.size()));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * doc_count));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_mask, sizeof(std::uint8_t) * doc_count));
    ZCGVS_CUDA_CHECK(cudaMemcpy(d_corpus, corpus.data(), sizeof(float) * corpus.size(), cudaMemcpyHostToDevice));
    ZCGVS_CUDA_CHECK(cudaMemcpy(d_query, query.data(), sizeof(float) * query.size(), cudaMemcpyHostToDevice));
    ZCGVS_CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), sizeof(std::uint8_t) * doc_count, cudaMemcpyHostToDevice));

    zcgvs::cuda::launchSimilarity(d_corpus, zcgvs::ScalarType::Float32, 1.0f, doc_count, dim,
                                  d_query, 1, nullptr, zcgvs::Metric::L2, d_scores, 0);
    ZCGVS_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gpu_l2(static_cast<std::size_t>(doc_count));
    ZCGVS_CUDA_CHECK(cudaMemcpy(gpu_l2.data(), d_scores, sizeof(float) * doc_count, cudaMemcpyDeviceToHost));

    const std::vector<float> ref_l2 = cpuL2(corpus, doc_count, dim, query);
    float max_error = 0.0f;
    for (int i = 0; i < doc_count; ++i) {
        max_error = std::max(max_error, std::fabs(gpu_l2[static_cast<std::size_t>(i)] -
                                                  ref_l2[static_cast<std::size_t>(i)]));
    }

    zcgvs::cuda::launchSimilarity(d_corpus, zcgvs::ScalarType::Float32, 1.0f, doc_count, dim,
                                  d_query, 1, d_mask, zcgvs::Metric::DotProduct, d_scores, 0);
    ZCGVS_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gpu_masked(static_cast<std::size_t>(doc_count));
    ZCGVS_CUDA_CHECK(cudaMemcpy(gpu_masked.data(), d_scores, sizeof(float) * doc_count, cudaMemcpyDeviceToHost));

    bool mask_ok = true;
    for (int i = 0; i < doc_count && mask_ok; ++i) {
        if (mask[static_cast<std::size_t>(i)] == 0) {
            if (gpu_masked[static_cast<std::size_t>(i)] != -FLT_MAX) {
                mask_ok = false;
            }
        } else if (gpu_masked[static_cast<std::size_t>(i)] == -FLT_MAX) {
            mask_ok = false;
        }
    }

    cudaFree(d_corpus);
    cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_mask);

    const float tolerance = 1e-3f;
    if (max_error > tolerance || !mask_ok) {
        std::printf("metrics_filter FAIL l2_error=%g mask_ok=%d\n", static_cast<double>(max_error),
                    mask_ok ? 1 : 0);
        return 1;
    }
    std::printf("metrics_filter PASS l2_error=%g\n", static_cast<double>(max_error));
    return 0;
}
