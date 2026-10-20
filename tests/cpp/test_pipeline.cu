// End-to-end device-pipeline checks through Searcher:
//   1. a lossless coarse pass plus on-device rescore reproduces the exact CPU top-k, which
//      pins down the gather, the second selection and the index remap
//   2. a lossy binary coarse pass still returns well-formed, exactly-scored, descending hits
//   3. a CUDA-graph replay produces bit-identical results to direct kernel launches

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_reference.hpp"
#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/data/embedding_store.hpp"
#include "zcgvs/search/searcher.hpp"

namespace {

constexpr int kDocCount = 8192;
constexpr int kDim = 256;
constexpr int kK = 10;
constexpr int kRescoreCandidates = 512;
constexpr int kBatch = 4;

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

zcgvs::HostTopK runRescore(const zcgvs::EmbeddingStore& coarse,
                           const zcgvs::EmbeddingStore& exact, const std::vector<float>& query,
                           const zcgvs::CudaStream& stream) {
    zcgvs::Searcher searcher(coarse);
    searcher.setRescoreStore(&exact);

    zcgvs::SearchParams params;
    params.k = kK;
    params.metric = zcgvs::Metric::Cosine;
    params.rescore_candidates = kRescoreCandidates;

    searcher.search(query.data(), kDim, 1, params, stream);
    return searcher.copyResultToHost(stream);
}

// Properties that must hold whatever the coarse pass throws away: every returned index is a
// real document, every returned score is the exact float32 similarity of that document, and
// the list is descending.
bool checkWellFormed(const char* label, const zcgvs::HostTopK& selection,
                     const std::vector<float>& reference) {
    for (int i = 0; i < kK; ++i) {
        const int index = selection.indices[static_cast<std::size_t>(i)];
        if (index < 0 || index >= kDocCount) {
            std::printf("%s: out-of-range index at rank %d\n", label, i);
            return false;
        }
        const float expected = reference[static_cast<std::size_t>(index)];
        if (std::fabs(selection.scores[static_cast<std::size_t>(i)] - expected) > 1e-4f) {
            std::printf("%s: rank %d is not scored with the exact similarity\n", label, i);
            return false;
        }
        if (i > 0 && selection.scores[static_cast<std::size_t>(i)] >
                         selection.scores[static_cast<std::size_t>(i - 1)] + 1e-6f) {
            std::printf("%s: results are not descending at rank %d\n", label, i);
            return false;
        }
    }
    return true;
}

bool checkLosslessRescore(const std::vector<float>& corpus, const std::vector<float>& query,
                          const zcgvs::CudaStream& stream) {
    const zcgvs::EmbeddingStore coarse = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Float16, 1.0f, true);
    const zcgvs::EmbeddingStore exact = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Float32, 1.0f, true);

    const zcgvs::HostTopK selection = runRescore(coarse, exact, query, stream);
    const std::vector<float> reference =
        zcgvs::testing::cpuCosineSimilarity(corpus, kDocCount, kDim, query);

    if (!checkWellFormed("float16 coarse", selection, reference)) {
        return false;
    }

    // A float16 coarse pass cannot lose the true top-10 out of a 512-wide candidate set, so
    // the final ranking must match the CPU exactly.
    const std::vector<int> expected = zcgvs::testing::cpuTopKIndices(reference, kK);
    for (int i = 0; i < kK; ++i) {
        if (selection.indices[static_cast<std::size_t>(i)] !=
            expected[static_cast<std::size_t>(i)]) {
            std::printf("float16 coarse: rank %d is %d, expected %d\n", i,
                        selection.indices[static_cast<std::size_t>(i)],
                        expected[static_cast<std::size_t>(i)]);
            return false;
        }
    }
    return true;
}

bool checkBinaryRescore(const std::vector<float>& corpus, const std::vector<float>& query,
                        const zcgvs::CudaStream& stream) {
    const zcgvs::EmbeddingStore coarse = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Binary, 1.0f, true);
    const zcgvs::EmbeddingStore exact = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Float32, 1.0f, true);

    const zcgvs::HostTopK selection = runRescore(coarse, exact, query, stream);
    const std::vector<float> reference =
        zcgvs::testing::cpuCosineSimilarity(corpus, kDocCount, kDim, query);

    return checkWellFormed("binary coarse", selection, reference);
}

bool checkGraphEquivalence(const std::vector<float>& corpus, const std::vector<float>& queries,
                           const zcgvs::CudaStream& stream) {
    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), kDocCount, kDim, zcgvs::ScalarType::Float16, 1.0f, true);
    zcgvs::Searcher searcher(store);

    zcgvs::SearchParams params;
    params.k = kK;
    params.metric = zcgvs::Metric::Cosine;

    params.use_graph = false;
    searcher.search(queries.data(), kDim, kBatch, params, stream);
    const zcgvs::HostTopK direct = searcher.copyResultToHost(stream);

    params.use_graph = true;
    searcher.search(queries.data(), kDim, kBatch, params, stream);
    const zcgvs::HostTopK replayed = searcher.copyResultToHost(stream);

    if (!searcher.lastSearchUsedGraph() || searcher.graphLaunchCount() == 0) {
        std::printf("graph: the replay path did not actually run\n");
        return false;
    }
    if (direct.indices.size() != replayed.indices.size()) {
        std::printf("graph: result sizes differ\n");
        return false;
    }
    for (std::size_t i = 0; i < direct.indices.size(); ++i) {
        if (direct.indices[i] != replayed.indices[i] || direct.scores[i] != replayed.scores[i]) {
            std::printf("graph: replay diverged at slot %zu\n", i);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::vector<float> corpus = randomMatrix(kDocCount, kDim, 31u);
    const std::vector<float> query = randomMatrix(1, kDim, 32u);
    const std::vector<float> queries = randomMatrix(kBatch, kDim, 33u);

    const zcgvs::CudaStream stream;
    const bool ok = checkLosslessRescore(corpus, query, stream) &&
                    checkBinaryRescore(corpus, query, stream) &&
                    checkGraphEquivalence(corpus, queries, stream);

    std::printf("pipeline %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
