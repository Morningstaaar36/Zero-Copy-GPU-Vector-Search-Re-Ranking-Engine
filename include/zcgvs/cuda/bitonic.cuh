#pragma once

#include <cuda_runtime.h>

namespace zcgvs {
namespace cuda {

__inline__ __device__ void bitonicSortSharedAscending(float* keys, int* values, int length) {
    for (int stage = 2; stage <= length; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            for (int i = threadIdx.x; i < length; i += blockDim.x) {
                const int partner = i ^ step;
                if (partner > i) {
                    const bool ascending_block = (i & stage) == 0;
                    const bool out_of_order = ascending_block ? (keys[i] > keys[partner])
                                                              : (keys[i] < keys[partner]);
                    if (out_of_order) {
                        const float tmp_key = keys[i];
                        keys[i] = keys[partner];
                        keys[partner] = tmp_key;

                        const int tmp_val = values[i];
                        values[i] = values[partner];
                        values[partner] = tmp_val;
                    }
                }
            }
            __syncthreads();
        }
    }
}

}  // namespace cuda
}  // namespace zcgvs
