#include "zcgvs/cuda/similarity.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/decode.cuh"
#include "zcgvs/cuda/reduction.cuh"

namespace zcgvs {
namespace cuda {

namespace {

// Dynamic shared memory the query tile may claim. Deliberately below the 48 KB per-block
// default so the static warp-sum array always fits alongside it without an opt-in.
constexpr std::size_t kMaxQuerySharedBytes = 40960;

template <typename Storage>
__global__ void similarityBatchKernel(const Storage* __restrict__ corpus,
                                      int doc_count,
                                      int dim,
                                      float scale,
                                      const float* __restrict__ queries,
                                      int batch,
                                      const std::uint8_t* __restrict__ mask,
                                      int metric,
                                      float* __restrict__ scores) {
    extern __shared__ unsigned int zcgvs_shared_raw[];
    float* shared_queries = reinterpret_cast<float*>(zcgvs_shared_raw);
    __shared__ float warp_sums[kQueryTile][kMaxWarpsPerBlock];

    const int query_base = blockIdx.y * kQueryTile;
    const int tile = min(kQueryTile, batch - query_base);

    for (int i = threadIdx.x; i < tile * dim; i += blockDim.x) {
        shared_queries[i] = queries[static_cast<std::size_t>(query_base) * dim + i];
    }
    __syncthreads();

    const int doc = blockIdx.x;
    if (doc >= doc_count) {
        return;
    }

    if (mask != nullptr && mask[doc] == 0) {
        if (threadIdx.x == 0) {
            for (int q = 0; q < tile; ++q) {
                scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] = -FLT_MAX;
            }
        }
        return;
    }

    const Storage* row = corpus + static_cast<std::size_t>(doc) * dim;

    float partial[kQueryTile];
#pragma unroll
    for (int q = 0; q < kQueryTile; ++q) {
        partial[q] = 0.0f;
    }

    for (int t = threadIdx.x; t < dim; t += blockDim.x) {
        const float value = decodeStorage(row[t], scale);
#pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {
            if (q < tile) {
                accumulateMetric(metric, shared_queries[q * dim + t], value, partial[q]);
            }
        }
    }

#pragma unroll
    for (int q = 0; q < kQueryTile; ++q) {
        if (q < tile) {
            const float reduced = blockReduceSum(partial[q], warp_sums[q]);
            if (threadIdx.x == 0) {
                scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] =
                    finalizeMetric(metric, reduced);
            }
        }
        __syncthreads();
    }
}

__global__ void hammingBatchKernel(const unsigned int* __restrict__ corpus,
                                   int doc_count,
                                   int dim,
                                   int words,
                                   const unsigned int* __restrict__ queries,
                                   int batch,
                                   const std::uint8_t* __restrict__ mask,
                                   float* __restrict__ scores) {
    extern __shared__ unsigned int zcgvs_shared_raw[];
    unsigned int* shared_bits = zcgvs_shared_raw;
    __shared__ float warp_sums[kQueryTile][kMaxWarpsPerBlock];

    const int query_base = blockIdx.y * kQueryTile;
    const int tile = min(kQueryTile, batch - query_base);

    for (int i = threadIdx.x; i < tile * words; i += blockDim.x) {
        shared_bits[i] = queries[static_cast<std::size_t>(query_base) * words + i];
    }
    __syncthreads();

    const int doc = blockIdx.x;
    if (doc >= doc_count) {
        return;
    }

    if (mask != nullptr && mask[doc] == 0) {
        if (threadIdx.x == 0) {
            for (int q = 0; q < tile; ++q) {
                scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] = -FLT_MAX;
            }
        }
        return;
    }

    const unsigned int* row = corpus + static_cast<std::size_t>(doc) * words;

    float partial[kQueryTile];
#pragma unroll
    for (int q = 0; q < kQueryTile; ++q) {
        partial[q] = 0.0f;
    }

    for (int w = threadIdx.x; w < words; w += blockDim.x) {
        const unsigned int doc_word = row[w];
#pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {
            if (q < tile) {
                partial[q] += static_cast<float>(__popc(doc_word ^ shared_bits[q * words + w]));
            }
        }
    }

#pragma unroll
    for (int q = 0; q < kQueryTile; ++q) {
        if (q < tile) {
            const float reduced = blockReduceSum(partial[q], warp_sums[q]);
            if (threadIdx.x == 0) {
                scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] =
                    hammingToScore(static_cast<int>(reduced), dim);
            }
        }
        __syncthreads();
    }
}

}  // namespace

int similarityQueryTile() { return kQueryTile; }

namespace {

// Bytes one staged query occupies in shared memory.
std::size_t queryStagingBytes(ScalarType dtype, int dim) {
    if (dtype == ScalarType::Binary) {
        return binaryWords(dim) * sizeof(unsigned int);
    }
    return static_cast<std::size_t>(dim) * sizeof(float);
}

// Queries one launch can co-reside, bounded by both the tile width and shared memory. High
// embedding dimensions shrink this, which is why the batch is processed in chunks rather
// than assuming kQueryTile always fits.
int feasibleQueryTile(ScalarType dtype, int dim, int batch) {
    const std::size_t per_query = queryStagingBytes(dtype, dim);
    if (per_query == 0 || per_query > kMaxQuerySharedBytes) {
        throw std::invalid_argument(
            "zcgvs similarity: embedding dim " + std::to_string(dim) +
            " needs more shared memory than a block can stage");
    }
    const int by_memory = static_cast<int>(kMaxQuerySharedBytes / per_query);
    return std::max(1, std::min(std::min(kQueryTile, by_memory), batch));
}

}  // namespace

std::size_t similaritySharedBytes(ScalarType dtype, int dim, int batch) {
    if (batch <= 0 || dim <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(feasibleQueryTile(dtype, dim, batch)) *
           queryStagingBytes(dtype, dim);
}

void launchSimilarity(const void* d_corpus,
                      ScalarType dtype,
                      float scale,
                      int doc_count,
                      int dim,
                      const void* d_queries,
                      int batch,
                      const std::uint8_t* d_mask,
                      Metric metric,
                      float* d_scores,
                      cudaStream_t stream) {
    if (doc_count <= 0 || batch <= 0 || dim <= 0) {
        return;
    }

    const int chunk_width = feasibleQueryTile(dtype, dim, batch);
    const std::size_t shared =
        static_cast<std::size_t>(chunk_width) * queryStagingBytes(dtype, dim);
    const int metric_code = static_cast<int>(metric);
    const int words = static_cast<int>(binaryWords(dim));

    for (int start = 0; start < batch; start += chunk_width) {
        const int chunk = std::min(chunk_width, batch - start);
        const unsigned int query_tiles =
            static_cast<unsigned int>((chunk + kQueryTile - 1) / kQueryTile);
        const dim3 grid(static_cast<unsigned int>(doc_count), query_tiles);
        float* scores = d_scores + static_cast<std::size_t>(start) * doc_count;

        switch (dtype) {
            case ScalarType::Float16:
                similarityBatchKernel<__half><<<grid, kScoreThreads, shared, stream>>>(
                    static_cast<const __half*>(d_corpus), doc_count, dim, scale,
                    static_cast<const float*>(d_queries) + static_cast<std::size_t>(start) * dim,
                    chunk, d_mask, metric_code, scores);
                break;
            case ScalarType::Int8:
                similarityBatchKernel<std::int8_t><<<grid, kScoreThreads, shared, stream>>>(
                    static_cast<const std::int8_t*>(d_corpus), doc_count, dim, scale,
                    static_cast<const float*>(d_queries) + static_cast<std::size_t>(start) * dim,
                    chunk, d_mask, metric_code, scores);
                break;
            case ScalarType::Binary:
                hammingBatchKernel<<<grid, kScoreThreads, shared, stream>>>(
                    static_cast<const unsigned int*>(d_corpus), doc_count, dim, words,
                    static_cast<const unsigned int*>(d_queries) +
                        static_cast<std::size_t>(start) * words,
                    chunk, d_mask, scores);
                break;
            case ScalarType::Float32:
            default:
                similarityBatchKernel<float><<<grid, kScoreThreads, shared, stream>>>(
                    static_cast<const float*>(d_corpus), doc_count, dim, scale,
                    static_cast<const float*>(d_queries) + static_cast<std::size_t>(start) * dim,
                    chunk, d_mask, metric_code, scores);
                break;
        }
        ZCGVS_CUDA_CHECK_LAST();
    }
}

}  // namespace cuda
}  // namespace zcgvs
