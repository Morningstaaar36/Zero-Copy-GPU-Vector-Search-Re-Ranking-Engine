#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_reference.hpp"
#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/topk.cuh"

int main() {
    const int doc_count = 5000;
    const int k = 100;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    std::vector<float> scores(static_cast<std::size_t>(doc_count));
    for (float& value : scores) {
        value = dist(rng);
    }

    const int scratch = zcgvs::cuda::topKScratchElements(doc_count, k);

    float* d_scores = nullptr;
    int* d_indices = nullptr;
    float* d_values = nullptr;
    float* d_scratch_values_a = nullptr;
    int* d_scratch_indices_a = nullptr;
    float* d_scratch_values_b = nullptr;
    int* d_scratch_indices_b = nullptr;
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * doc_count));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_indices, sizeof(int) * k));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * k));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scratch_values_a, sizeof(float) * scratch));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scratch_indices_a, sizeof(int) * scratch));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scratch_values_b, sizeof(float) * scratch));
    ZCGVS_CUDA_CHECK(cudaMalloc(&d_scratch_indices_b, sizeof(int) * scratch));
    ZCGVS_CUDA_CHECK(cudaMemcpy(d_scores, scores.data(), sizeof(float) * doc_count,
                                cudaMemcpyHostToDevice));

    zcgvs::cuda::launchTopK(d_scores, doc_count, k, 1, d_indices, d_values, d_scratch_values_a,
                            d_scratch_indices_a, d_scratch_values_b, d_scratch_indices_b, 0);
    ZCGVS_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> indices(static_cast<std::size_t>(k));
    std::vector<float> values(static_cast<std::size_t>(k));
    ZCGVS_CUDA_CHECK(cudaMemcpy(indices.data(), d_indices, sizeof(int) * k, cudaMemcpyDeviceToHost));
    ZCGVS_CUDA_CHECK(cudaMemcpy(values.data(), d_values, sizeof(float) * k, cudaMemcpyDeviceToHost));

    const std::vector<int> reference = zcgvs::testing::cpuTopKIndices(scores, k);

    bool ok = true;
    for (int i = 0; i < k && ok; ++i) {
        const int index = indices[static_cast<std::size_t>(i)];
        if (index < 0 || index >= doc_count) {
            ok = false;
            break;
        }
        if (std::fabs(values[static_cast<std::size_t>(i)] - scores[static_cast<std::size_t>(index)]) > 1e-5f) {
            ok = false;
            break;
        }
        if (std::fabs(values[static_cast<std::size_t>(i)] -
                      scores[static_cast<std::size_t>(reference[static_cast<std::size_t>(i)])]) > 1e-4f) {
            ok = false;
            break;
        }
        if (i > 0 && values[static_cast<std::size_t>(i)] > values[static_cast<std::size_t>(i - 1)] + 1e-6f) {
            ok = false;
            break;
        }
    }

    cudaFree(d_scores);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_scratch_values_a);
    cudaFree(d_scratch_indices_a);
    cudaFree(d_scratch_values_b);
    cudaFree(d_scratch_indices_b);

    if (!ok) {
        std::printf("topk FAIL\n");
        return 1;
    }
    std::printf("topk PASS\n");
    return 0;
}
