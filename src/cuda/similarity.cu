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
// default so the kernels never need an opt-in.
constexpr std::size_t kMaxQuerySharedBytes = 40960;

// Bytes per vectorized global load. Every storage type is read in 128-bit transactions when
// the row stride and base pointer allow it.
constexpr int kVectorBytes = 16;

__device__ inline float warpAllReduceSum(float value) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Loads kVec consecutive elements of a corpus row and widens them to float32 in registers.
// kVec == 1 is the scalar fallback for dimensions that do not divide into 16-byte vectors.
template <typename Storage, int kVec>
__device__ inline void loadRow(const Storage* __restrict__ source, float scale, float (&out)[kVec]) {
    if constexpr (kVec == 1) {
        out[0] = decodeStorage(source[0], scale);
    } else if constexpr (sizeof(Storage) == 4) {
        const float4 packed = __ldg(reinterpret_cast<const float4*>(source));
        out[0] = packed.x;
        out[1] = packed.y;
        out[2] = packed.z;
        out[3] = packed.w;
    } else if constexpr (sizeof(Storage) == 2) {
        const uint4 packed = __ldg(reinterpret_cast<const uint4*>(source));
        const __half2* pairs = reinterpret_cast<const __half2*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 widened = __half22float2(pairs[i]);
            out[2 * i] = widened.x;
            out[2 * i + 1] = widened.y;
        }
    } else {
        const int4 packed = __ldg(reinterpret_cast<const int4*>(source));
        const char4* quads = reinterpret_cast<const char4*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[4 * i] = static_cast<float>(quads[i].x) * scale;
            out[4 * i + 1] = static_cast<float>(quads[i].y) * scale;
            out[4 * i + 2] = static_cast<float>(quads[i].z) * scale;
            out[4 * i + 3] = static_cast<float>(quads[i].w) * scale;
        }
    }
}

// Persistent warp-per-document scan.
//
// Each resident block stages the query tile into shared memory exactly once, then its warps
// stride over the corpus one document at a time. Lanes cover a row with 128-bit loads, so a
// 384-dim row is three float4 sweeps (fp32), two uint4 sweeps (fp16) or one int4 sweep
// (int8), and every loaded element is reused across the whole query tile. Shared-memory query
// reads are float4 as well, which keeps them free of bank conflicts.
//
// The previous design launched one block per document and re-staged the query tile in every
// block, so at 1M documents it moved ~12 GB of redundant query traffic to scan a 0.4-1.5 GB
// corpus. That overhead was identical for every storage type, which is why int8 and fp16 used
// to be no faster than fp32.
template <typename Storage, int kVec>
__global__ void similarityScanKernel(const Storage* __restrict__ corpus,
                                     int doc_count,
                                     int dim,
                                     float scale,
                                     const float* __restrict__ queries,
                                     int batch,
                                     const std::uint8_t* __restrict__ mask,
                                     int metric,
                                     float* __restrict__ scores) {
    extern __shared__ float4 zcgvs_shared_vec[];
    float* shared_queries = reinterpret_cast<float*>(zcgvs_shared_vec);

    const int query_base = blockIdx.y * kQueryTile;
    const int tile = min(kQueryTile, batch - query_base);

    for (int i = threadIdx.x; i < tile * dim; i += blockDim.x) {
        shared_queries[i] = queries[static_cast<std::size_t>(query_base) * dim + i];
    }
    __syncthreads();

    const int lane = threadIdx.x % warpSize;
    const int warps = blockDim.x / warpSize;
    const int stride = gridDim.x * warps;

    for (int doc = blockIdx.x * warps + threadIdx.x / warpSize; doc < doc_count; doc += stride) {
        if (mask != nullptr && mask[doc] == 0) {
            if (lane < tile) {
                scores[static_cast<std::size_t>(query_base + lane) * doc_count + doc] = -FLT_MAX;
            }
            continue;
        }

        const Storage* row = corpus + static_cast<std::size_t>(doc) * dim;

        float partial[kQueryTile];
#pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {
            partial[q] = 0.0f;
        }

        for (int base = lane * kVec; base < dim; base += warpSize * kVec) {
            float values[kVec];
            loadRow<Storage, kVec>(row + base, scale, values);

#pragma unroll
            for (int q = 0; q < kQueryTile; ++q) {
                if (q < tile) {
                    const float* query = shared_queries + q * dim + base;
                    if constexpr (kVec % 4 == 0) {
#pragma unroll
                        for (int e = 0; e < kVec; e += 4) {
                            const float4 qv = *reinterpret_cast<const float4*>(query + e);
                            accumulateMetric(metric, qv.x, values[e], partial[q]);
                            accumulateMetric(metric, qv.y, values[e + 1], partial[q]);
                            accumulateMetric(metric, qv.z, values[e + 2], partial[q]);
                            accumulateMetric(metric, qv.w, values[e + 3], partial[q]);
                        }
                    } else {
#pragma unroll
                        for (int e = 0; e < kVec; ++e) {
                            accumulateMetric(metric, query[e], values[e], partial[q]);
                        }
                    }
                }
            }
        }

        // Butterfly reduction leaves the sum in every lane, so lane q stores query q and the
        // tile's writes go out as one instruction.
#pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {
            if (q < tile) {
                const float reduced = warpAllReduceSum(partial[q]);
                if (lane == q) {
                    scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] =
                        finalizeMetric(metric, reduced);
                }
            }
        }
    }
}

// Persistent thread-per-document Hamming scan. A 384-dim binary row is only 48 bytes, far too
// small to split across a warp, so each thread owns a document: three uint4 loads, a popcount
// per word against the shared (broadcast) query bits, and a coalesced score store.
template <bool kVectorized>
__global__ void hammingScanKernel(const unsigned int* __restrict__ corpus,
                                  int doc_count,
                                  int dim,
                                  int words,
                                  const unsigned int* __restrict__ queries,
                                  int batch,
                                  const std::uint8_t* __restrict__ mask,
                                  float* __restrict__ scores) {
    extern __shared__ float4 zcgvs_shared_vec[];
    unsigned int* shared_bits = reinterpret_cast<unsigned int*>(zcgvs_shared_vec);

    const int query_base = blockIdx.y * kQueryTile;
    const int tile = min(kQueryTile, batch - query_base);

    for (int i = threadIdx.x; i < tile * words; i += blockDim.x) {
        shared_bits[i] = queries[static_cast<std::size_t>(query_base) * words + i];
    }
    __syncthreads();

    const int stride = gridDim.x * blockDim.x;
    for (int doc = blockIdx.x * blockDim.x + threadIdx.x; doc < doc_count; doc += stride) {
        if (mask != nullptr && mask[doc] == 0) {
            for (int q = 0; q < tile; ++q) {
                scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] = -FLT_MAX;
            }
            continue;
        }

        const unsigned int* row = corpus + static_cast<std::size_t>(doc) * words;

        int hamming[kQueryTile];
#pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {
            hamming[q] = 0;
        }

        if constexpr (kVectorized) {
            for (int w = 0; w < words; w += 4) {
                const uint4 bits = __ldg(reinterpret_cast<const uint4*>(row + w));
#pragma unroll
                for (int q = 0; q < kQueryTile; ++q) {
                    if (q < tile) {
                        const unsigned int* query = shared_bits + q * words + w;
                        hamming[q] += __popc(bits.x ^ query[0]) + __popc(bits.y ^ query[1]) +
                                      __popc(bits.z ^ query[2]) + __popc(bits.w ^ query[3]);
                    }
                }
            }
        } else {
            for (int w = 0; w < words; ++w) {
                const unsigned int bits = __ldg(row + w);
#pragma unroll
                for (int q = 0; q < kQueryTile; ++q) {
                    if (q < tile) {
                        hamming[q] += __popc(bits ^ shared_bits[q * words + w]);
                    }
                }
            }
        }

#pragma unroll
        for (int q = 0; q < kQueryTile; ++q) {
            if (q < tile) {
                scores[static_cast<std::size_t>(query_base + q) * doc_count + doc] =
                    hammingToScore(hamming[q], dim);
            }
        }
    }
}

bool isAligned(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % kVectorBytes == 0;
}

// Grid for a persistent kernel: enough blocks to fill every SM to its occupancy limit, and no
// more than the work needs. Extra blocks beyond residency would only re-stage the queries.
template <typename Kernel>
unsigned int persistentGrid(Kernel kernel, std::size_t shared, long long blocks_needed) {
    int device = 0;
    int sm_count = 0;
    int per_sm = 0;
    ZCGVS_CUDA_CHECK(cudaGetDevice(&device));
    ZCGVS_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    ZCGVS_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kScoreThreads, shared));
    const long long resident = static_cast<long long>(sm_count) * std::max(per_sm, 1);
    return static_cast<unsigned int>(std::max(1LL, std::min(resident, blocks_needed)));
}

template <typename Storage, int kVec>
void launchScan(const Storage* corpus, int doc_count, int dim, float scale,
                const float* queries, int chunk, const std::uint8_t* mask, int metric,
                float* scores, std::size_t shared, unsigned int query_tiles,
                cudaStream_t stream) {
    const auto kernel = similarityScanKernel<Storage, kVec>;
    const int warps_per_block = kScoreThreads / 32;
    const long long needed = (static_cast<long long>(doc_count) + warps_per_block - 1) /
                             warps_per_block;
    const dim3 grid(persistentGrid(kernel, shared, needed), query_tiles);
    kernel<<<grid, kScoreThreads, shared, stream>>>(corpus, doc_count, dim, scale, queries,
                                                    chunk, mask, metric, scores);
}

template <typename Storage>
void dispatchScan(const void* d_corpus, int doc_count, int dim, float scale,
                  const float* queries, int chunk, const std::uint8_t* mask, int metric,
                  float* scores, std::size_t shared, unsigned int query_tiles,
                  cudaStream_t stream) {
    constexpr int kVec = kVectorBytes / static_cast<int>(sizeof(Storage));
    const auto* corpus = static_cast<const Storage*>(d_corpus);
    if (dim % kVec == 0 && isAligned(d_corpus)) {
        launchScan<Storage, kVec>(corpus, doc_count, dim, scale, queries, chunk, mask, metric,
                                  scores, shared, query_tiles, stream);
    } else {
        launchScan<Storage, 1>(corpus, doc_count, dim, scale, queries, chunk, mask, metric,
                               scores, shared, query_tiles, stream);
    }
}

void dispatchHamming(const void* d_corpus, int doc_count, int dim, int words,
                     const unsigned int* queries, int chunk, const std::uint8_t* mask,
                     float* scores, std::size_t shared, unsigned int query_tiles,
                     cudaStream_t stream) {
    const auto* corpus = static_cast<const unsigned int*>(d_corpus);
    const long long needed = (static_cast<long long>(doc_count) + kScoreThreads - 1) /
                             kScoreThreads;
    if (words % 4 == 0 && isAligned(d_corpus)) {
        const dim3 grid(persistentGrid(hammingScanKernel<true>, shared, needed), query_tiles);
        hammingScanKernel<true><<<grid, kScoreThreads, shared, stream>>>(
            corpus, doc_count, dim, words, queries, chunk, mask, scores);
    } else {
        const dim3 grid(persistentGrid(hammingScanKernel<false>, shared, needed), query_tiles);
        hammingScanKernel<false><<<grid, kScoreThreads, shared, stream>>>(
            corpus, doc_count, dim, words, queries, chunk, mask, scores);
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
        float* scores = d_scores + static_cast<std::size_t>(start) * doc_count;
        const float* float_queries =
            static_cast<const float*>(d_queries) + static_cast<std::size_t>(start) * dim;

        switch (dtype) {
            case ScalarType::Float16:
                dispatchScan<__half>(d_corpus, doc_count, dim, scale, float_queries, chunk,
                                     d_mask, metric_code, scores, shared, query_tiles, stream);
                break;
            case ScalarType::Int8:
                dispatchScan<std::int8_t>(d_corpus, doc_count, dim, scale, float_queries, chunk,
                                          d_mask, metric_code, scores, shared, query_tiles,
                                          stream);
                break;
            case ScalarType::Binary:
                dispatchHamming(d_corpus, doc_count, dim, words,
                                static_cast<const unsigned int*>(d_queries) +
                                    static_cast<std::size_t>(start) * words,
                                chunk, d_mask, scores, shared, query_tiles, stream);
                break;
            case ScalarType::Float32:
            default:
                dispatchScan<float>(d_corpus, doc_count, dim, scale, float_queries, chunk,
                                    d_mask, metric_code, scores, shared, query_tiles, stream);
                break;
        }
        ZCGVS_CUDA_CHECK_LAST();
    }
}

}  // namespace cuda
}  // namespace zcgvs
