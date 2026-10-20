// Verifies the GPU BM25 scatter-add over a synthetic CSR inverted index, the max reduction
// used to normalize it, and the dense/lexical fusion, each against a CPU reference.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/cuda/lexical.cuh"

namespace {

constexpr int kDocCount = 20000;
constexpr int kVocabSize = 3000;
constexpr int kTermCount = 24;
constexpr float kDenseWeight = 0.6f;

struct HostIndex {
    std::vector<std::int64_t> offsets;
    std::vector<int> docs;
    std::vector<float> weights;
};

HostIndex buildIndex(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> postings_per_term(0, 400);
    std::uniform_int_distribution<int> doc_picker(0, kDocCount - 1);
    std::uniform_real_distribution<float> weight_picker(0.01f, 8.0f);

    HostIndex index;
    index.offsets.reserve(static_cast<std::size_t>(kVocabSize) + 1);
    index.offsets.push_back(0);
    for (int term = 0; term < kVocabSize; ++term) {
        const int count = postings_per_term(rng);
        for (int i = 0; i < count; ++i) {
            index.docs.push_back(doc_picker(rng));
            index.weights.push_back(weight_picker(rng));
        }
        index.offsets.push_back(static_cast<std::int64_t>(index.docs.size()));
    }
    return index;
}

}  // namespace

int main() {
    const HostIndex index = buildIndex(41u);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> term_picker(0, kVocabSize - 1);
    std::uniform_real_distribution<float> dense_picker(-1.0f, 1.0f);

    std::vector<int> query_terms(kTermCount);
    for (int& term : query_terms) {
        term = term_picker(rng);
    }
    query_terms[kTermCount - 1] = -1;  // padding slot must be ignored

    std::vector<float> dense(kDocCount);
    for (float& value : dense) {
        value = dense_picker(rng);
    }

    // CPU reference: scatter-add, then max-normalize, then the convex combination.
    std::vector<float> expected_lexical(kDocCount, 0.0f);
    for (const int term : query_terms) {
        if (term < 0) {
            continue;
        }
        for (std::int64_t p = index.offsets[static_cast<std::size_t>(term)];
             p < index.offsets[static_cast<std::size_t>(term) + 1]; ++p) {
            expected_lexical[static_cast<std::size_t>(index.docs[static_cast<std::size_t>(p)])] +=
                index.weights[static_cast<std::size_t>(p)];
        }
    }
    float expected_max = 0.0f;
    for (const float value : expected_lexical) {
        expected_max = std::max(expected_max, value);
    }
    const float inverse = expected_max > 0.0f ? 1.0f / expected_max : 0.0f;
    std::vector<float> expected_fused(kDocCount);
    for (int i = 0; i < kDocCount; ++i) {
        expected_fused[static_cast<std::size_t>(i)] =
            kDenseWeight * dense[static_cast<std::size_t>(i)] +
            (1.0f - kDenseWeight) * expected_lexical[static_cast<std::size_t>(i)] * inverse;
    }

    const zcgvs::CudaStream stream;
    zcgvs::DeviceBuffer<std::int64_t> d_offsets(index.offsets.size());
    zcgvs::DeviceBuffer<int> d_docs(index.docs.size());
    zcgvs::DeviceBuffer<float> d_weights(index.weights.size());
    zcgvs::DeviceBuffer<int> d_terms(query_terms.size());
    zcgvs::DeviceBuffer<float> d_lexical(kDocCount);
    zcgvs::DeviceBuffer<float> d_dense(kDocCount);
    zcgvs::DeviceBuffer<float> d_max(1);

    d_offsets.copyFromHost(index.offsets.data(), index.offsets.size(), stream.get());
    d_docs.copyFromHost(index.docs.data(), index.docs.size(), stream.get());
    d_weights.copyFromHost(index.weights.data(), index.weights.size(), stream.get());
    d_terms.copyFromHost(query_terms.data(), query_terms.size(), stream.get());
    d_dense.copyFromHost(dense.data(), dense.size(), stream.get());

    ZCGVS_CUDA_CHECK(cudaMemsetAsync(d_lexical.data(), 0, kDocCount * sizeof(float), stream.get()));
    ZCGVS_CUDA_CHECK(cudaMemsetAsync(d_max.data(), 0, sizeof(float), stream.get()));

    zcgvs::cuda::launchBm25(d_offsets.data(), d_docs.data(), d_weights.data(), d_terms.data(),
                            kTermCount, kDocCount, d_lexical.data(), stream.get());
    zcgvs::cuda::launchMaxPositive(d_lexical.data(), kDocCount, d_max.data(), stream.get());
    zcgvs::cuda::launchFuseScores(d_dense.data(), d_lexical.data(), d_max.data(), kDenseWeight,
                                  kDocCount, stream.get());
    stream.synchronize();

    std::vector<float> lexical(kDocCount);
    std::vector<float> fused(kDocCount);
    float maximum = 0.0f;
    d_lexical.copyToHost(lexical.data(), lexical.size(), stream.get());
    d_dense.copyToHost(fused.data(), fused.size(), stream.get());
    d_max.copyToHost(&maximum, 1, stream.get());
    stream.synchronize();

    bool ok = true;
    if (std::fabs(maximum - expected_max) > 1e-3f) {
        std::printf("max reduction: gpu=%f cpu=%f\n", maximum, expected_max);
        ok = false;
    }
    for (int i = 0; i < kDocCount && ok; ++i) {
        // Atomic accumulation reorders the additions, so compare with a relative tolerance.
        const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected_lexical[i]));
        if (std::fabs(lexical[static_cast<std::size_t>(i)] - expected_lexical[i]) > tolerance) {
            std::printf("bm25 mismatch at %d: gpu=%f cpu=%f\n", i, lexical[i],
                        expected_lexical[i]);
            ok = false;
            break;
        }
        if (std::fabs(fused[static_cast<std::size_t>(i)] - expected_fused[i]) > 1e-4f) {
            std::printf("fusion mismatch at %d: gpu=%f cpu=%f\n", i, fused[i], expected_fused[i]);
            ok = false;
            break;
        }
    }

    std::printf("lexical %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
