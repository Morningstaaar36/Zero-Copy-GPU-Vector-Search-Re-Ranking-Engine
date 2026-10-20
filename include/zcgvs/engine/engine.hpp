#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/core/metric.hpp"
#include "zcgvs/data/corpus.hpp"
#include "zcgvs/data/lexical_index.hpp"
#include "zcgvs/engine/engine_config.hpp"
#include "zcgvs/engine/engine_stats.hpp"
#include "zcgvs/engine/search_hit.hpp"
#include "zcgvs/engine/search_options.hpp"
#include "zcgvs/engine/search_response.hpp"
#include "zcgvs/search/searcher.hpp"
#include "zcgvs/text/tokenizer.hpp"

namespace zcgvs {

class QueryEmbedder;
class Reranker;

// Composition root: owns the corpus, the optional rescore store and lexical index, the
// tokenizer, the bi-encoder, the searcher and the cross-encoder, and wires them into one
// request path.
class Engine {
public:
    explicit Engine(const EngineConfig& config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Options pre-filled from the engine configuration.
    SearchOptions defaultOptions() const;

    SearchResponse searchTimed(const std::string& query, const SearchOptions& options) const;

    BatchSearchResponse searchBatch(const std::vector<std::string>& queries,
                                    const SearchOptions& options) const;

    SearchResponse searchTimed(const std::string& query,
                               int top_k,
                               int candidates,
                               Metric metric,
                               const std::vector<int>& allow_ids) const;

    std::vector<SearchHit> search(const std::string& query, int top_k, int candidates) const;
    std::vector<SearchHit> search(const std::string& query) const;

    std::vector<Candidate> retrieve(const std::string& query,
                                    int k,
                                    Metric metric,
                                    const std::vector<int>& allow_ids) const;

    void addDocuments(const std::vector<std::string>& ids, const std::vector<std::string>& texts);

    EngineStats stats() const;

    std::int64_t corpusSize() const { return corpus_->size(); }
    int dim() const { return corpus_->dim(); }
    std::string storageDtype() const;
    bool hasLexicalIndex() const { return lexical_index_ != nullptr; }
    bool hasRescoreStore() const { return rescore_store_ != nullptr; }
    bool hasReranker() const { return reranker_ != nullptr; }
    const EngineConfig& config() const { return config_; }

private:
    struct ResolvedOptions {
        int top_k = 10;
        int candidates = 100;
        Metric metric = Metric::Cosine;
        int rescore_candidates = 0;
        float dense_weight = 1.0f;
        bool hybrid = false;
        bool rerank = true;
        SimilarityBackend backend = SimilarityBackend::Custom;
        bool use_graph = false;
    };

    ResolvedOptions resolve(const SearchOptions& options, std::int64_t corpus_size) const;
    const std::uint8_t* buildMask(const std::vector<int>& allow_ids, std::int64_t doc_count) const;
    std::vector<int> lexicalTerms(const std::string& query) const;
    std::vector<SearchHit> assemble(const std::string& query,
                                    const HostTopK& selection,
                                    int query_index,
                                    const ResolvedOptions& resolved,
                                    double& rerank_ms) const;

    EngineConfig config_;
    Tokenizer tokenizer_;
    std::unique_ptr<Corpus> corpus_;
    std::unique_ptr<EmbeddingStore> rescore_store_;
    std::unique_ptr<LexicalIndex> lexical_index_;
    std::unique_ptr<QueryEmbedder> embedder_;
    std::unique_ptr<Reranker> reranker_;
    std::unique_ptr<Searcher> searcher_;
    std::unique_ptr<CudaStream> stream_;

    mutable std::vector<std::uint8_t> host_mask_;
    mutable DeviceBuffer<std::uint8_t> d_mask_;
    mutable std::vector<int> host_terms_;
};

}  // namespace zcgvs
