#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/core/metric.hpp"
#include "zcgvs/core/pinned_buffer.hpp"
#include "zcgvs/data/embedding_store.hpp"
#include "zcgvs/data/lexical_index.hpp"
#include "zcgvs/search/search_params.hpp"
#include "zcgvs/search/search_result.hpp"

namespace zcgvs {

class CublasSimilarity;

// Drives the whole device-side pipeline and owns every scratch buffer it needs.
//
// Stages, in order, all chained through device memory:
//   1. query upload (and sign-bit packing when the corpus is binary)
//   2. brute-force score matrix   [batch, doc_count]
//   3. optional BM25 scatter-add over the lexical index, max-normalized and fused into (2)
//   4. multi-pass bitonic top-k   [batch, k] or [batch, rescore_candidates]
//   5. optional exact-precision gather rescore of (4), a second top-k, and an index remap
//
// Only the final [batch, k] indices and scores are ever copied to the host.
class Searcher {
public:
    explicit Searcher(const EmbeddingStore& store);
    ~Searcher();

    Searcher(const Searcher&) = delete;
    Searcher& operator=(const Searcher&) = delete;

    // Higher-precision copy of the same corpus, in the same document order, used by the
    // rescore stage. Passing nullptr disables rescoring.
    void setRescoreStore(const EmbeddingStore* store);

    // CSR inverted index enabling the hybrid lexical stage. Passing nullptr disables it.
    void setLexicalIndex(const LexicalIndex* index);

    void search(const float* host_queries,
                int dim,
                int batch,
                const SearchParams& params,
                const CudaStream& stream);

    HostTopK copyResultToHost(const CudaStream& stream);

    int resultK() const { return current_k_; }
    int resultBatch() const { return current_batch_; }
    double lastSearchGpuMs() const { return last_gpu_ms_; }
    bool lastSearchUsedGraph() const { return used_graph_; }
    std::uint64_t graphCaptureCount() const { return graph_captures_; }
    std::uint64_t graphLaunchCount() const { return graph_launches_; }

    bool hasRescoreStore() const { return rescore_store_ != nullptr; }
    bool hasLexicalIndex() const { return lexical_index_ != nullptr; }

private:
    struct GraphKey {
        const void* corpus = nullptr;
        const void* rescore_corpus = nullptr;
        const void* lexical_index = nullptr;
        const std::uint8_t* mask = nullptr;
        std::uint64_t generation = 0;
        std::int64_t doc_count = 0;
        int k = 0;
        int batch = 0;
        int rescore_candidates = 0;
        int lexical_stride = 0;
        int metric = 0;
        float dense_weight = 1.0f;

        bool matches(const GraphKey& other) const;
    };

    void validate(int dim, int batch, const SearchParams& params, std::int64_t doc_count) const;
    void ensureCapacity(std::int64_t doc_count, int batch, const SearchParams& params);
    void uploadInputs(const float* host_queries, int batch, const SearchParams& params,
                      const CudaStream& stream);
    void recordPipeline(std::int64_t doc_count, int batch, const SearchParams& params,
                        cudaStream_t stream);
    void launchViaGraph(std::int64_t doc_count, int batch, const SearchParams& params,
                        const CudaStream& stream);
    void captureGraph(std::int64_t doc_count, int batch, const SearchParams& params,
                      const CudaStream& stream);
    void destroyGraph() noexcept;
    GraphKey makeKey(std::int64_t doc_count, int batch, const SearchParams& params) const;
    bool lexicalEnabled(const SearchParams& params) const;
    bool rescoreEnabled(const SearchParams& params) const;

    const EmbeddingStore& store_;
    const EmbeddingStore* rescore_store_ = nullptr;
    const LexicalIndex* lexical_index_ = nullptr;
    int dim_ = 0;

    std::unique_ptr<CublasSimilarity> cublas_;

    DeviceBuffer<float> d_queries_;
    DeviceBuffer<unsigned int> d_query_bits_;
    DeviceBuffer<int> d_query_terms_;
    DeviceBuffer<float> d_scores_;
    DeviceBuffer<float> d_lexical_scores_;
    DeviceBuffer<float> d_lexical_max_;
    DeviceBuffer<float> d_scratch_values_a_;
    DeviceBuffer<int> d_scratch_indices_a_;
    DeviceBuffer<float> d_scratch_values_b_;
    DeviceBuffer<int> d_scratch_indices_b_;
    DeviceBuffer<int> d_coarse_indices_;
    DeviceBuffer<float> d_coarse_values_;
    DeviceBuffer<float> d_rescored_;
    DeviceBuffer<int> d_local_indices_;
    DeviceBuffer<int> d_out_indices_;
    DeviceBuffer<float> d_out_values_;

    PinnedBuffer<float> h_queries_;
    PinnedBuffer<int> h_query_terms_;
    PinnedBuffer<int> h_out_indices_;
    PinnedBuffer<float> h_out_values_;

    std::uint64_t generation_ = 0;
    int current_k_ = 0;
    int current_batch_ = 0;

    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_end_ = nullptr;
    double last_gpu_ms_ = 0.0;

    cudaGraphExec_t graph_exec_ = nullptr;
    GraphKey graph_key_;
    bool graph_valid_ = false;
    bool used_graph_ = false;
    std::uint64_t graph_captures_ = 0;
    std::uint64_t graph_launches_ = 0;
};

}  // namespace zcgvs
