// Exercises the recursive merge: one million scores with k=512 forces several merge passes,
// which is the case the old single-pass merge could not express at all.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_reference.hpp"
#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/cuda/topk.cuh"

namespace {

constexpr int kDocCount = 1000000;
constexpr int kK = 512;

}  // namespace

int main() {
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);

    std::vector<float> scores(static_cast<std::size_t>(kDocCount));
    for (float& value : scores) {
        value = distribution(rng);
    }

    const zcgvs::CudaStream stream;
    const std::size_t scratch =
        static_cast<std::size_t>(zcgvs::cuda::topKScratchElements(kDocCount, kK));

    zcgvs::DeviceBuffer<float> d_scores(scores.size());
    zcgvs::DeviceBuffer<float> values_a(scratch);
    zcgvs::DeviceBuffer<int> indices_a(scratch);
    zcgvs::DeviceBuffer<float> values_b(scratch);
    zcgvs::DeviceBuffer<int> indices_b(scratch);
    zcgvs::DeviceBuffer<int> out_indices(kK);
    zcgvs::DeviceBuffer<float> out_values(kK);

    d_scores.copyFromHost(scores.data(), scores.size(), stream.get());
    zcgvs::cuda::launchTopK(d_scores.data(), kDocCount, kK, 1, out_indices.data(),
                            out_values.data(), values_a.data(), indices_a.data(), values_b.data(),
                            indices_b.data(), stream.get());
    stream.synchronize();

    std::vector<int> indices(kK);
    std::vector<float> values(kK);
    out_indices.copyToHost(indices.data(), kK, stream.get());
    out_values.copyToHost(values.data(), kK, stream.get());
    stream.synchronize();

    const std::vector<int> reference = zcgvs::testing::cpuTopKIndices(scores, kK);

    bool ok = true;
    for (int i = 0; i < kK; ++i) {
        const int index = indices[static_cast<std::size_t>(i)];
        if (index < 0 || index >= kDocCount) {
            std::printf("index %d out of range at rank %d\n", index, i);
            ok = false;
            break;
        }
        if (std::fabs(values[static_cast<std::size_t>(i)] -
                      scores[static_cast<std::size_t>(index)]) > 1e-5f) {
            std::printf("value/index disagreement at rank %d\n", i);
            ok = false;
            break;
        }
        if (std::fabs(values[static_cast<std::size_t>(i)] -
                      scores[static_cast<std::size_t>(reference[static_cast<std::size_t>(i)])]) >
            1e-4f) {
            std::printf("rank %d differs from the CPU reference\n", i);
            ok = false;
            break;
        }
        if (i > 0 && values[static_cast<std::size_t>(i)] >
                         values[static_cast<std::size_t>(i - 1)] + 1e-6f) {
            std::printf("results are not descending at rank %d\n", i);
            ok = false;
            break;
        }
    }

    std::printf("topk_large %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
