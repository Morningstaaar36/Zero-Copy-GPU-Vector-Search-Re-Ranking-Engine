#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "zcgvs/core/metric.hpp"

namespace zcgvs {
namespace cuda {

// Number of dimensions a single dot-product/L2 block accumulates per thread pass, and the
// shared-memory fan-in width used by the block reductions in every scoring kernel.
constexpr int kScoreThreads = 256;
constexpr int kMaxWarpsPerBlock = 32;

// Queries co-resident in one block of the batched scoring kernels. Each corpus row is read
// from global memory once and reused across the tile, so corpus traffic drops by this factor.
constexpr int kQueryTile = 8;

__device__ inline float decodeStorage(float value, float) { return value; }
__device__ inline float decodeStorage(__half value, float) { return __half2float(value); }
__device__ inline float decodeStorage(std::int8_t value, float scale) {
    return static_cast<float>(value) * scale;
}

// Accumulates one dimension of a (query, document) pair under the requested metric. L2 is
// accumulated as squared distance and negated once by the caller so larger stays better.
__device__ inline void accumulateMetric(int metric, float query_value, float doc_value,
                                        float& accumulator) {
    if (metric == static_cast<int>(Metric::L2)) {
        const float difference = query_value - doc_value;
        accumulator += difference * difference;
    } else {
        accumulator += query_value * doc_value;
    }
}

__device__ inline float finalizeMetric(int metric, float accumulator) {
    return (metric == static_cast<int>(Metric::L2)) ? -accumulator : accumulator;
}

// Maps a packed Hamming distance to a cosine-like score in [-1, 1].
__device__ inline float hammingToScore(int hamming, int dim) {
    return static_cast<float>(dim - 2 * hamming) / static_cast<float>(dim);
}

}  // namespace cuda
}  // namespace zcgvs
