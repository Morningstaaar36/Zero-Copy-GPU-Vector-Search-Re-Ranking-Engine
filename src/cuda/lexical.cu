#include "zcgvs/cuda/lexical.cuh"

#include <cstddef>

#include "zcgvs/core/cuda_check.hpp"

namespace zcgvs {
namespace cuda {

namespace {

constexpr int kPostingThreads = 256;
constexpr int kReduceThreads = 256;
constexpr int kFuseThreads = 256;
constexpr int kMaxReduceBlocks = 256;

__global__ void bm25Kernel(const std::int64_t* __restrict__ offsets,
                           const int* __restrict__ docs,
                           const float* __restrict__ weights,
                           const int* __restrict__ query_terms,
                           int term_count,
                           int doc_count,
                           float* __restrict__ scores) {
    const int slot = blockIdx.x;
    if (slot >= term_count) {
        return;
    }

    const int term = query_terms[slot];
    if (term < 0) {
        return;
    }

    const std::int64_t begin = offsets[term];
    const std::int64_t end = offsets[term + 1];
    for (std::int64_t posting = begin + threadIdx.x; posting < end; posting += blockDim.x) {
        const int doc = docs[posting];
        if (doc >= 0 && doc < doc_count) {
            atomicAdd(&scores[doc], weights[posting]);
        }
    }
}

__global__ void maxPositiveKernel(const float* __restrict__ values, int count,
                                  float* __restrict__ out_max) {
    __shared__ float shared[kReduceThreads];

    float best = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        best = fmaxf(best, values[i]);
    }
    shared[threadIdx.x] = best;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(out_max), __float_as_uint(shared[0]));
    }
}

__global__ void fuseScoresKernel(float* __restrict__ dense, const float* __restrict__ lexical,
                                 const float* __restrict__ lexical_max, float weight, int count) {
    const float peak = *lexical_max;
    const float inverse = (peak > 0.0f) ? (1.0f / peak) : 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        dense[i] = weight * dense[i] + (1.0f - weight) * lexical[i] * inverse;
    }
}

}  // namespace

void launchBm25(const std::int64_t* d_offsets,
                const int* d_docs,
                const float* d_weights,
                const int* d_query_terms,
                int term_count,
                int doc_count,
                float* d_scores,
                cudaStream_t stream) {
    if (term_count <= 0 || doc_count <= 0) {
        return;
    }
    bm25Kernel<<<static_cast<unsigned int>(term_count), kPostingThreads, 0, stream>>>(
        d_offsets, d_docs, d_weights, d_query_terms, term_count, doc_count, d_scores);
    ZCGVS_CUDA_CHECK_LAST();
}

void launchMaxPositive(const float* d_values, int count, float* d_max, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    int blocks = (count + kReduceThreads - 1) / kReduceThreads;
    if (blocks > kMaxReduceBlocks) {
        blocks = kMaxReduceBlocks;
    }
    maxPositiveKernel<<<static_cast<unsigned int>(blocks), kReduceThreads, 0, stream>>>(
        d_values, count, d_max);
    ZCGVS_CUDA_CHECK_LAST();
}

void launchFuseScores(float* d_dense,
                      const float* d_lexical,
                      const float* d_lexical_max,
                      float weight,
                      int count,
                      cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    int blocks = (count + kFuseThreads - 1) / kFuseThreads;
    if (blocks > kMaxReduceBlocks) {
        blocks = kMaxReduceBlocks;
    }
    fuseScoresKernel<<<static_cast<unsigned int>(blocks), kFuseThreads, 0, stream>>>(
        d_dense, d_lexical, d_lexical_max, weight, count);
    ZCGVS_CUDA_CHECK_LAST();
}

}  // namespace cuda
}  // namespace zcgvs
