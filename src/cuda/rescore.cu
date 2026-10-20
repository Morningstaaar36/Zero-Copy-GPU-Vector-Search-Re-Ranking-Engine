#include "zcgvs/cuda/rescore.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/decode.cuh"
#include "zcgvs/cuda/reduction.cuh"

namespace zcgvs {
namespace cuda {

namespace {

constexpr int kRemapThreads = 256;

template <typename Storage>
__global__ void gatherRescoreKernel(const Storage* __restrict__ corpus,
                                    int doc_count,
                                    int dim,
                                    float scale,
                                    const float* __restrict__ queries,
                                    const int* __restrict__ candidate_indices,
                                    int candidate_count,
                                    int metric,
                                    float* __restrict__ values) {
    extern __shared__ float zcgvs_rescore_shared[];
    __shared__ float warp_sums[kMaxWarpsPerBlock];

    const int query = blockIdx.y;
    for (int t = threadIdx.x; t < dim; t += blockDim.x) {
        zcgvs_rescore_shared[t] = queries[static_cast<std::size_t>(query) * dim + t];
    }
    __syncthreads();

    const int slot = blockIdx.x;
    if (slot >= candidate_count) {
        return;
    }

    const std::size_t offset = static_cast<std::size_t>(query) * candidate_count + slot;
    const int doc = candidate_indices[offset];
    if (doc < 0 || doc >= doc_count) {
        if (threadIdx.x == 0) {
            values[offset] = -FLT_MAX;
        }
        return;
    }

    const Storage* row = corpus + static_cast<std::size_t>(doc) * dim;
    float partial = 0.0f;
    for (int t = threadIdx.x; t < dim; t += blockDim.x) {
        accumulateMetric(metric, zcgvs_rescore_shared[t], decodeStorage(row[t], scale), partial);
    }

    const float reduced = blockReduceSum(partial, warp_sums);
    if (threadIdx.x == 0) {
        values[offset] = finalizeMetric(metric, reduced);
    }
}

__global__ void remapIndicesKernel(const int* __restrict__ candidate_indices,
                                   const int* __restrict__ local_indices,
                                   int candidate_count,
                                   int k,
                                   int* __restrict__ out_indices) {
    const int query = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const std::size_t out_offset = static_cast<std::size_t>(query) * k + i;
        const int local = local_indices[out_offset];
        out_indices[out_offset] =
            (local >= 0 && local < candidate_count)
                ? candidate_indices[static_cast<std::size_t>(query) * candidate_count + local]
                : -1;
    }
}

}  // namespace

std::size_t rescoreSharedBytes(int dim) {
    return static_cast<std::size_t>(dim) * sizeof(float);
}

void launchGatherRescore(const void* d_corpus,
                         ScalarType dtype,
                         float scale,
                         int doc_count,
                         int dim,
                         const float* d_queries,
                         int batch,
                         const int* d_candidate_indices,
                         int candidate_count,
                         Metric metric,
                         float* d_values,
                         cudaStream_t stream) {
    if (candidate_count <= 0 || batch <= 0 || dim <= 0) {
        return;
    }
    if (dtype == ScalarType::Binary) {
        throw std::invalid_argument(
            "zcgvs rescore: the rescore store must hold float32, float16 or int8 rows");
    }

    const dim3 grid(static_cast<unsigned int>(candidate_count), static_cast<unsigned int>(batch));
    const std::size_t shared = rescoreSharedBytes(dim);
    const int metric_code = static_cast<int>(metric);

    switch (dtype) {
        case ScalarType::Float16:
            gatherRescoreKernel<__half><<<grid, kScoreThreads, shared, stream>>>(
                static_cast<const __half*>(d_corpus), doc_count, dim, scale, d_queries,
                d_candidate_indices, candidate_count, metric_code, d_values);
            break;
        case ScalarType::Int8:
            gatherRescoreKernel<std::int8_t><<<grid, kScoreThreads, shared, stream>>>(
                static_cast<const std::int8_t*>(d_corpus), doc_count, dim, scale, d_queries,
                d_candidate_indices, candidate_count, metric_code, d_values);
            break;
        case ScalarType::Float32:
        default:
            gatherRescoreKernel<float><<<grid, kScoreThreads, shared, stream>>>(
                static_cast<const float*>(d_corpus), doc_count, dim, scale, d_queries,
                d_candidate_indices, candidate_count, metric_code, d_values);
            break;
    }
    ZCGVS_CUDA_CHECK_LAST();
}

void launchRemapIndices(const int* d_candidate_indices,
                        const int* d_local_indices,
                        int candidate_count,
                        int k,
                        int batch,
                        int* d_out_indices,
                        cudaStream_t stream) {
    if (k <= 0 || batch <= 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>((k + kRemapThreads - 1) / kRemapThreads);
    remapIndicesKernel<<<dim3(blocks, static_cast<unsigned int>(batch)), kRemapThreads, 0, stream>>>(
        d_candidate_indices, d_local_indices, candidate_count, k, d_out_indices);
    ZCGVS_CUDA_CHECK_LAST();
}

}  // namespace cuda
}  // namespace zcgvs
