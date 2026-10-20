#include "zcgvs/cuda/topk.cuh"

#include <cfloat>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/bitonic.cuh"

namespace zcgvs {
namespace cuda {

namespace {

constexpr int kTileSize = 1024;
constexpr int kMergeTile = 4096;
constexpr int kLocalThreads = 256;
constexpr int kMergeThreads = 256;

int nextPowerOfTwo(int value) {
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

__global__ void localTopKKernel(const float* __restrict__ scores,
                                int doc_count,
                                int k,
                                float* __restrict__ out_values,
                                int* __restrict__ out_indices,
                                int stride_out) {
    __shared__ float s_keys[kTileSize];
    __shared__ int s_values[kTileSize];

    const int query = blockIdx.y;
    const int tile_start = blockIdx.x * kTileSize;
    const float* source_scores = scores + static_cast<std::size_t>(query) * doc_count;

    for (int i = threadIdx.x; i < kTileSize; i += blockDim.x) {
        const int global_index = tile_start + i;
        if (global_index < doc_count) {
            s_keys[i] = source_scores[global_index];
            s_values[i] = global_index;
        } else {
            s_keys[i] = -FLT_MAX;
            s_values[i] = -1;
        }
    }
    __syncthreads();

    bitonicSortSharedAscending(s_keys, s_values, kTileSize);

    float* destination_values =
        out_values + static_cast<std::size_t>(query) * stride_out + static_cast<std::size_t>(blockIdx.x) * k;
    int* destination_indices =
        out_indices + static_cast<std::size_t>(query) * stride_out + static_cast<std::size_t>(blockIdx.x) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int source = kTileSize - 1 - i;
        destination_values[i] = s_keys[source];
        destination_indices[i] = s_values[source];
    }
}

__global__ void mergeTopKKernel(const float* __restrict__ in_values,
                                const int* __restrict__ in_indices,
                                int count,
                                int stride_in,
                                int tile,
                                int k,
                                float* __restrict__ out_values,
                                int* __restrict__ out_indices,
                                int stride_out) {
    __shared__ float s_keys[kMergeTile];
    __shared__ int s_values[kMergeTile];

    const int query = blockIdx.y;
    const int tile_start = blockIdx.x * tile;
    const float* source_values = in_values + static_cast<std::size_t>(query) * stride_in;
    const int* source_indices = in_indices + static_cast<std::size_t>(query) * stride_in;

    for (int i = threadIdx.x; i < tile; i += blockDim.x) {
        const int global_index = tile_start + i;
        if (global_index < count) {
            s_keys[i] = source_values[global_index];
            s_values[i] = source_indices[global_index];
        } else {
            s_keys[i] = -FLT_MAX;
            s_values[i] = -1;
        }
    }
    __syncthreads();

    bitonicSortSharedAscending(s_keys, s_values, tile);

    float* destination_values =
        out_values + static_cast<std::size_t>(query) * stride_out + static_cast<std::size_t>(blockIdx.x) * k;
    int* destination_indices =
        out_indices + static_cast<std::size_t>(query) * stride_out + static_cast<std::size_t>(blockIdx.x) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int source = tile - 1 - i;
        destination_values[i] = s_keys[source];
        destination_indices[i] = s_values[source];
    }
}

}  // namespace

int topKTileSize() { return kTileSize; }

int topKMergeTile() { return kMergeTile; }

int topKMaxK() { return kTileSize; }

int topKBlockCount(int doc_count) {
    if (doc_count <= 0) {
        return 0;
    }
    return (doc_count + kTileSize - 1) / kTileSize;
}

int topKScratchElements(int doc_count, int k) {
    if (doc_count <= 0 || k <= 0) {
        return 0;
    }
    return topKBlockCount(doc_count) * k;
}

void launchTopK(const float* d_scores,
                int doc_count,
                int k,
                int batch,
                int* d_out_indices,
                float* d_out_values,
                float* d_scratch_values_a,
                int* d_scratch_indices_a,
                float* d_scratch_values_b,
                int* d_scratch_indices_b,
                cudaStream_t stream) {
    if (k <= 0 || doc_count <= 0 || batch <= 0) {
        return;
    }
    if (k > kTileSize) {
        throw std::invalid_argument("zcgvs top-k: k=" + std::to_string(k) +
                                    " exceeds per-tile capacity " + std::to_string(kTileSize));
    }
    if (k > doc_count) {
        throw std::invalid_argument("zcgvs top-k: k=" + std::to_string(k) +
                                    " exceeds candidate count " + std::to_string(doc_count));
    }

    const int local_blocks = topKBlockCount(doc_count);
    const int stride = local_blocks * k;
    const unsigned int queries = static_cast<unsigned int>(batch);

    localTopKKernel<<<dim3(static_cast<unsigned int>(local_blocks), queries), kLocalThreads, 0,
                      stream>>>(d_scores, doc_count, k, d_scratch_values_a, d_scratch_indices_a,
                                stride);
    ZCGVS_CUDA_CHECK_LAST();

    float* source_values = d_scratch_values_a;
    int* source_indices = d_scratch_indices_a;
    float* destination_values = d_scratch_values_b;
    int* destination_indices = d_scratch_indices_b;

    int count = stride;
    while (true) {
        const int blocks = (count + kMergeTile - 1) / kMergeTile;
        if (blocks <= 1) {
            mergeTopKKernel<<<dim3(1, queries), kMergeThreads, 0, stream>>>(
                source_values, source_indices, count, stride, nextPowerOfTwo(count), k,
                d_out_values, d_out_indices, k);
            ZCGVS_CUDA_CHECK_LAST();
            return;
        }

        mergeTopKKernel<<<dim3(static_cast<unsigned int>(blocks), queries), kMergeThreads, 0,
                          stream>>>(source_values, source_indices, count, stride, kMergeTile, k,
                                    destination_values, destination_indices, stride);
        ZCGVS_CUDA_CHECK_LAST();

        count = blocks * k;
        std::swap(source_values, destination_values);
        std::swap(source_indices, destination_indices);
    }
}

}  // namespace cuda
}  // namespace zcgvs
