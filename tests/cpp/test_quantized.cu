// Verifies the int8 and binary storage paths of the similarity kernel against CPU references
// that reproduce the exact quantization and bit-packing conventions.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_reference.hpp"
#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/cuda/convert.cuh"
#include "zcgvs/cuda/similarity.cuh"
#include "zcgvs/data/embedding_store.hpp"

namespace {

constexpr int kDocCount = 3000;
constexpr int kDim = 384;

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

bool checkInt8(const std::vector<float>& corpus, const std::vector<float>& query,
               const zcgvs::CudaStream& stream) {
    const float scale = 1.0f / 127.0f;
    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Int8, scale, true);

    zcgvs::DeviceBuffer<float> d_query(kDim);
    zcgvs::DeviceBuffer<float> d_scores(kDocCount);
    d_query.copyFromHost(query.data(), kDim, stream.get());

    zcgvs::cuda::launchSimilarity(store.deviceData(), zcgvs::ScalarType::Int8, store.scale(),
                                  kDocCount, kDim, d_query.data(), 1, nullptr,
                                  zcgvs::Metric::Cosine, d_scores.data(), stream.get());
    stream.synchronize();

    std::vector<float> scores(kDocCount);
    d_scores.copyToHost(scores.data(), kDocCount, stream.get());
    stream.synchronize();

    const std::vector<float> dequantized =
        zcgvs::testing::cpuQuantizeDequantizeInt8(corpus, scale);
    const std::vector<float> reference =
        zcgvs::testing::cpuCosineSimilarity(dequantized, kDocCount, kDim, query);

    for (int i = 0; i < kDocCount; ++i) {
        if (std::fabs(scores[static_cast<std::size_t>(i)] -
                      reference[static_cast<std::size_t>(i)]) > 1e-3f) {
            std::printf("int8 mismatch at %d: gpu=%f cpu=%f\n", i, scores[i], reference[i]);
            return false;
        }
    }
    return true;
}

bool checkBinary(const std::vector<float>& corpus, const std::vector<float>& query,
                 const zcgvs::CudaStream& stream) {
    const int words = (kDim + 31) / 32;
    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Binary, 1.0f, true);

    zcgvs::DeviceBuffer<float> d_query(kDim);
    zcgvs::DeviceBuffer<unsigned int> d_query_bits(static_cast<std::size_t>(words));
    zcgvs::DeviceBuffer<float> d_scores(kDocCount);

    d_query.copyFromHost(query.data(), kDim, stream.get());
    zcgvs::cuda::launchPackBinary(d_query.data(), d_query_bits.data(), 1, kDim, stream.get());
    zcgvs::cuda::launchSimilarity(store.deviceData(), zcgvs::ScalarType::Binary, 1.0f, kDocCount,
                                  kDim, d_query_bits.data(), 1, nullptr, zcgvs::Metric::Cosine,
                                  d_scores.data(), stream.get());
    stream.synchronize();

    std::vector<float> scores(kDocCount);
    d_scores.copyToHost(scores.data(), kDocCount, stream.get());
    stream.synchronize();

    const std::vector<std::uint32_t> packed_corpus =
        zcgvs::testing::cpuPackBinary(corpus, kDocCount, kDim);
    const std::vector<std::uint32_t> packed_query = zcgvs::testing::cpuPackBinary(query, 1, kDim);

    for (int i = 0; i < kDocCount; ++i) {
        const int hamming = zcgvs::testing::cpuHamming(
            packed_corpus.data() + static_cast<std::size_t>(i) * words, packed_query.data(), words);
        const float expected = static_cast<float>(kDim - 2 * hamming) / static_cast<float>(kDim);
        if (std::fabs(scores[static_cast<std::size_t>(i)] - expected) > 1e-5f) {
            std::printf("binary mismatch at %d: gpu=%f cpu=%f\n", i, scores[i], expected);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<float> corpus = randomMatrix(kDocCount, kDim, 5u);
    const std::vector<float> query = randomMatrix(1, kDim, 6u);

    const zcgvs::CudaStream stream;
    const bool ok = checkInt8(corpus, query, stream) && checkBinary(corpus, query, stream);

    std::printf("quantized %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
