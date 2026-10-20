#pragma once

#include <cuda_runtime.h>

namespace zcgvs {
namespace cuda {

__inline__ __device__ float warpReduceSum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__inline__ __device__ float blockReduceSum(float value, float* warp_sums) {
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;

    value = warpReduceSum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    value = (threadIdx.x < warp_count) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    return value;
}

}  // namespace cuda
}  // namespace zcgvs
