// Exercises every code path of the persistent scan kernels against CPU references: the 128-bit
// vectorized loads and the scalar fallback for each storage type (chosen by whether the row
// stride divides into 16-byte vectors), partial query tiles, multi-chunk batches, the
// allow-mask, and enough documents that every warp strides over the corpus several times.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
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

constexpr int kDocCount = 20000;
constexpr int kBatch = 11;  // one full tile of 8 plus a partial tile of 3
constexpr float kInt8Scale = 1.0f / 127.0f;

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

// The corpus exactly as the kernel will see it after the store's on-device conversion.
std::vector<float> storedCorpus(const std::vector<float>& corpus, zcgvs::ScalarType dtype) {
    if (dtype == zcgvs::ScalarType::Int8) {
        return zcgvs::testing::cpuQuantizeDequantizeInt8(corpus, kInt8Scale);
    }
    if (dtype == zcgvs::ScalarType::Float16) {
        std::vector<float> rounded(corpus.size());
        for (std::size_t i = 0; i < corpus.size(); ++i) {
            rounded[i] = __half2float(__float2half_rn(corpus[i]));
        }
        return rounded;
    }
    return corpus;
}

float referenceScore(const std::vector<float>& corpus, const std::vector<float>& queries,
                     int doc, int query, int dim, zcgvs::Metric metric) {
    float accumulator = 0.0f;
    for (int j = 0; j < dim; ++j) {
        const float d = corpus[static_cast<std::size_t>(doc) * dim + j];
        const float q = queries[static_cast<std::size_t>(query) * dim + j];
        accumulator += (metric == zcgvs::Metric::L2) ? (q - d) * (q - d) : q * d;
    }
    return (metric == zcgvs::Metric::L2) ? -accumulator : accumulator;
}

bool checkCase(zcgvs::ScalarType dtype, int dim, zcgvs::Metric metric, bool use_mask,
               const zcgvs::CudaStream& stream) {
    const std::vector<float> corpus = randomMatrix(kDocCount, dim, 11u + dim);
    const std::vector<float> queries = randomMatrix(kBatch, dim, 97u + dim);
    const bool binary = dtype == zcgvs::ScalarType::Binary;

    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, dim, dtype,
        dtype == zcgvs::ScalarType::Int8 ? kInt8Scale : 1.0f, true);

    std::vector<std::uint8_t> mask(kDocCount, 1);
    for (int i = 0; i < kDocCount; i += 7) {
        mask[static_cast<std::size_t>(i)] = 0;
    }

    const int words = (dim + 31) / 32;
    zcgvs::DeviceBuffer<float> d_queries(static_cast<std::size_t>(kBatch) * dim);
    zcgvs::DeviceBuffer<unsigned int> d_query_bits(static_cast<std::size_t>(kBatch) * words);
    zcgvs::DeviceBuffer<std::uint8_t> d_mask(kDocCount);
    zcgvs::DeviceBuffer<float> d_scores(static_cast<std::size_t>(kBatch) * kDocCount);

    d_queries.copyFromHost(queries.data(), queries.size(), stream.get());
    d_mask.copyFromHost(mask.data(), mask.size(), stream.get());
    const void* query_input = d_queries.data();
    if (binary) {
        zcgvs::cuda::launchPackBinary(d_queries.data(), d_query_bits.data(), kBatch, dim,
                                      stream.get());
        query_input = d_query_bits.data();
    }

    zcgvs::cuda::launchSimilarity(store.deviceData(), dtype, store.scale(), kDocCount, dim,
                                  query_input, kBatch, use_mask ? d_mask.data() : nullptr,
                                  metric, d_scores.data(), stream.get());
    std::vector<float> scores(static_cast<std::size_t>(kBatch) * kDocCount);
    d_scores.copyToHost(scores.data(), scores.size(), stream.get());
    stream.synchronize();

    const std::vector<float> stored = storedCorpus(corpus, dtype);
    const std::vector<std::uint32_t> packed_corpus =
        binary ? zcgvs::testing::cpuPackBinary(corpus, kDocCount, dim)
               : std::vector<std::uint32_t>{};
    const std::vector<std::uint32_t> packed_queries =
        binary ? zcgvs::testing::cpuPackBinary(queries, kBatch, dim)
               : std::vector<std::uint32_t>{};

    for (int q = 0; q < kBatch; ++q) {
        for (int doc = 0; doc < kDocCount; ++doc) {
            const float gpu = scores[static_cast<std::size_t>(q) * kDocCount + doc];
            float expected = 0.0f;
            if (use_mask && mask[static_cast<std::size_t>(doc)] == 0) {
                expected = -FLT_MAX;
            } else if (binary) {
                const int hamming = zcgvs::testing::cpuHamming(
                    packed_corpus.data() + static_cast<std::size_t>(doc) * words,
                    packed_queries.data() + static_cast<std::size_t>(q) * words, words);
                expected = static_cast<float>(dim - 2 * hamming) / static_cast<float>(dim);
            } else {
                expected = referenceScore(stored, queries, doc, q, dim, metric);
            }

            const float tolerance = (expected == -FLT_MAX) ? 0.0f : 1e-4f;
            if (std::fabs(gpu - expected) > tolerance) {
                std::printf("scan_paths mismatch dtype=%d dim=%d metric=%d mask=%d q=%d doc=%d "
                            "gpu=%f cpu=%f\n",
                            static_cast<int>(dtype), dim, static_cast<int>(metric),
                            use_mask ? 1 : 0, q, doc, gpu, expected);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    const zcgvs::CudaStream stream;

    // 384 and 768 take the vectorized path for every dtype; 100 is vectorized for float32 only;
    // 97 forces the scalar path everywhere, and gives binary a word count that is not a
    // multiple of four.
    const int dims[] = {384, 768, 100, 97};
    const zcgvs::ScalarType dtypes[] = {zcgvs::ScalarType::Float32, zcgvs::ScalarType::Float16,
                                        zcgvs::ScalarType::Int8, zcgvs::ScalarType::Binary};

    int cases = 0;
    bool ok = true;
    for (const int dim : dims) {
        for (const zcgvs::ScalarType dtype : dtypes) {
            const bool binary = dtype == zcgvs::ScalarType::Binary;
            ok = ok && checkCase(dtype, dim, zcgvs::Metric::DotProduct, false, stream);
            ok = ok && checkCase(dtype, dim, zcgvs::Metric::DotProduct, true, stream);
            cases += 2;
            if (!binary) {
                ok = ok && checkCase(dtype, dim, zcgvs::Metric::L2, true, stream);
                ++cases;
            }
        }
    }

    std::printf("scan_paths %s (%d cases)\n", ok ? "PASS" : "FAIL", cases);
    return ok ? 0 : 1;
}
