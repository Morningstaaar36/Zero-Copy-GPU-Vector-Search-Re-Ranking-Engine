// Checks that the query-tiled batched kernels produce, for every query in the batch, exactly
// what the single-query path produces on its own.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_reference.hpp"
#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/cuda/similarity.cuh"
#include "zcgvs/cuda/topk.cuh"
#include "zcgvs/data/embedding_store.hpp"

namespace {

constexpr int kDocCount = 4096;
constexpr int kDim = 256;
constexpr int kBatch = 11;  // deliberately not a multiple of the query tile
constexpr int kK = 64;

std::vector<float> randomMatrix(int rows, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> matrix(static_cast<std::size_t>(rows) * dim);
    for (int row = 0; row < rows; ++row) {
        for (int j = 0; j < dim; ++j) {
            matrix[static_cast<std::size_t>(row) * dim + j] = distribution(rng);
        }
        zcgvs::testing::l2Normalize(matrix, row * dim, dim);
    }
    return matrix;
}

}  // namespace

int main() {
    const std::vector<float> corpus = randomMatrix(kDocCount, kDim, 21u);
    const std::vector<float> queries = randomMatrix(kBatch, kDim, 22u);

    const zcgvs::CudaStream stream;
    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Float32, 1.0f, true);

    const std::size_t score_elements = static_cast<std::size_t>(kBatch) * kDocCount;
    const std::size_t scratch =
        static_cast<std::size_t>(kBatch) *
        static_cast<std::size_t>(zcgvs::cuda::topKScratchElements(kDocCount, kK));

    zcgvs::DeviceBuffer<float> d_queries(static_cast<std::size_t>(kBatch) * kDim);
    zcgvs::DeviceBuffer<float> d_scores(score_elements);
    zcgvs::DeviceBuffer<float> values_a(scratch);
    zcgvs::DeviceBuffer<int> indices_a(scratch);
    zcgvs::DeviceBuffer<float> values_b(scratch);
    zcgvs::DeviceBuffer<int> indices_b(scratch);
    zcgvs::DeviceBuffer<int> out_indices(static_cast<std::size_t>(kBatch) * kK);
    zcgvs::DeviceBuffer<float> out_values(static_cast<std::size_t>(kBatch) * kK);

    d_queries.copyFromHost(queries.data(), static_cast<std::size_t>(kBatch) * kDim, stream.get());
    zcgvs::cuda::launchSimilarity(store.deviceData(), zcgvs::ScalarType::Float32, 1.0f, kDocCount,
                                  kDim, d_queries.data(), kBatch, nullptr, zcgvs::Metric::Cosine,
                                  d_scores.data(), stream.get());
    zcgvs::cuda::launchTopK(d_scores.data(), kDocCount, kK, kBatch, out_indices.data(),
                            out_values.data(), values_a.data(), indices_a.data(), values_b.data(),
                            indices_b.data(), stream.get());
    stream.synchronize();

    std::vector<float> scores(score_elements);
    std::vector<int> indices(static_cast<std::size_t>(kBatch) * kK);
    std::vector<float> values(static_cast<std::size_t>(kBatch) * kK);
    d_scores.copyToHost(scores.data(), score_elements, stream.get());
    out_indices.copyToHost(indices.data(), indices.size(), stream.get());
    out_values.copyToHost(values.data(), values.size(), stream.get());
    stream.synchronize();

    bool ok = true;
    for (int q = 0; q < kBatch && ok; ++q) {
        std::vector<float> query(queries.begin() + static_cast<std::ptrdiff_t>(q) * kDim,
                                 queries.begin() + static_cast<std::ptrdiff_t>(q + 1) * kDim);
        const std::vector<float> reference =
            zcgvs::testing::cpuCosineSimilarity(corpus, kDocCount, kDim, query);

        for (int i = 0; i < kDocCount; ++i) {
            const std::size_t offset = static_cast<std::size_t>(q) * kDocCount + i;
            if (std::fabs(scores[offset] - reference[static_cast<std::size_t>(i)]) > 1e-4f) {
                std::printf("score mismatch at query %d doc %d\n", q, i);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        const std::vector<int> expected = zcgvs::testing::cpuTopKIndices(reference, kK);
        for (int i = 0; i < kK; ++i) {
            const std::size_t offset = static_cast<std::size_t>(q) * kK + i;
            const int index = indices[offset];
            if (index < 0 || index >= kDocCount) {
                std::printf("index out of range at query %d rank %d\n", q, i);
                ok = false;
                break;
            }
            if (std::fabs(values[offset] -
                          reference[static_cast<std::size_t>(expected[static_cast<std::size_t>(i)])]) >
                1e-4f) {
                std::printf("rank %d of query %d differs from the reference\n", i, q);
                ok = false;
                break;
            }
        }
    }

    std::printf("batched %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
