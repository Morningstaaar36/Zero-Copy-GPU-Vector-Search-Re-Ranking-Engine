#include "zcgvs/cuda/convert.cuh"

#include <cuda_fp16.h>

#include <cstdint>

#include "zcgvs/core/cuda_check.hpp"

namespace zcgvs {
namespace cuda {

namespace {

constexpr int kConvertThreads = 256;
constexpr std::size_t kMaxConvertBlocks = 65535;

__global__ void convertF32ToF16Kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                      std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        dst[index] = __float2half(src[index]);
    }
}

__global__ void convertF32ToI8Kernel(const float* __restrict__ src, std::int8_t* __restrict__ dst,
                                     std::size_t count, float inverse_scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float scaled = rintf(src[index] * inverse_scale);
        dst[index] = static_cast<std::int8_t>(fminf(fmaxf(scaled, -127.0f), 127.0f));
    }
}

__global__ void packBinaryKernel(const float* __restrict__ src, unsigned int* __restrict__ dst,
                                 int rows, int dim, int words) {
    const std::size_t total = static_cast<std::size_t>(rows) * words;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t row = index / static_cast<std::size_t>(words);
        const int word = static_cast<int>(index % static_cast<std::size_t>(words));
        const float* row_data = src + row * static_cast<std::size_t>(dim);

        unsigned int bits = 0u;
        const int base = word * 32;
        for (int bit = 0; bit < 32; ++bit) {
            const int dimension = base + bit;
            if (dimension < dim && row_data[dimension] >= 0.0f) {
                bits |= (1u << bit);
            }
        }
        dst[index] = bits;
    }
}

}  // namespace

void launchConvertF32ToF16(const float* src, void* dst_half, std::size_t count,
                           cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const std::size_t blocks = (count + kConvertThreads - 1) / kConvertThreads;
    convertF32ToF16Kernel<<<static_cast<unsigned int>(blocks), kConvertThreads, 0, stream>>>(
        src, static_cast<__half*>(dst_half), count);
    ZCGVS_CUDA_CHECK_LAST();
}

void launchConvertF32ToI8(const float* src, void* dst_int8, std::size_t count, float scale,
                          cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    const float inverse_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    const std::size_t blocks = (count + kConvertThreads - 1) / kConvertThreads;
    convertF32ToI8Kernel<<<static_cast<unsigned int>(blocks), kConvertThreads, 0, stream>>>(
        src, static_cast<std::int8_t*>(dst_int8), count, inverse_scale);
    ZCGVS_CUDA_CHECK_LAST();
}

void launchPackBinary(const float* src, void* dst_words, int rows, int dim, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) {
        return;
    }
    const int words = (dim + 31) / 32;
    const std::size_t total = static_cast<std::size_t>(rows) * static_cast<std::size_t>(words);
    std::size_t blocks = (total + kConvertThreads - 1) / kConvertThreads;
    if (blocks > kMaxConvertBlocks) {
        blocks = kMaxConvertBlocks;
    }
    packBinaryKernel<<<static_cast<unsigned int>(blocks), kConvertThreads, 0, stream>>>(
        src, static_cast<unsigned int*>(dst_words), rows, dim, words);
    ZCGVS_CUDA_CHECK_LAST();
}

}  // namespace cuda
}  // namespace zcgvs
