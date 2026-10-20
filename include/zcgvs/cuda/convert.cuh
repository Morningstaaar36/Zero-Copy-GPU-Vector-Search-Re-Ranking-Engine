#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace zcgvs {
namespace cuda {

void launchConvertF32ToF16(const float* src, void* dst_half, std::size_t count,
                           cudaStream_t stream);

// Symmetric quantization to signed bytes: q = clamp(round(x / scale), -127, 127).
void launchConvertF32ToI8(const float* src, void* dst_int8, std::size_t count, float scale,
                          cudaStream_t stream);

// Packs the sign bits of a row-major [rows, dim] matrix into ceil(dim/32) uint32 words per
// row. Bit b of word w carries dimension w * 32 + b and is set when the value is >= 0;
// dimensions past `dim` are left zero so they cancel in any XOR.
void launchPackBinary(const float* src, void* dst_words, int rows, int dim, cudaStream_t stream);

}  // namespace cuda
}  // namespace zcgvs
