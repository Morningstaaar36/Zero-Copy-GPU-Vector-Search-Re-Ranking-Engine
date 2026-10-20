#include "zcgvs/engine/engine.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_set>

#include "zcgvs/core/cuda_context.hpp"
#include "zcgvs/core/nvtx.hpp"
#include "zcgvs/core/scalar_type.hpp"
#include "zcgvs/embed/onnx_query_embedder.hpp"
#include "zcgvs/rerank/reranker.hpp"
#include "zcgvs/rerank/reranker_factory.hpp"
#include "zcgvs/search/search_result.hpp"

namespace zcgvs {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

Engine::Engine(const EngineConfig& config)
    : config_(config),
      tokenizer_(Tokenizer::fromVocabFile(config.vocab_path, config.lowercase)) {
    setDevice(config_.device_id);

    corpus_ =
        std::make_unique<Corpus>(Corpus::load(config_.embeddings_path, config_.metadata_path));
    stream_ = std::make_unique<CudaStream>();
    searcher_ = std::make_unique<Searcher>(corpus_->embeddings());

    if (!config_.rescore_embeddings_path.empty()) {
        rescore_store_ = std::make_unique<EmbeddingStore>(
            EmbeddingStore::loadFromFile(config_.rescore_embeddings_path));
        if (rescore_store_->docCount() < corpus_->size()) {
            throw std::runtime_error(
                "zcgvs: the rescore store holds fewer documents than the primary corpus");
        }
        searcher_->setRescoreStore(rescore_store_.get());
    }

    if (!config_.lexical_index_path.empty()) {
        lexical_index_ =
            std::make_unique<LexicalIndex>(LexicalIndex::loadFromFile(config_.lexical_index_path));
        if (lexical_index_->docCount() < corpus_->size()) {
            throw std::runtime_error(
                "zcgvs: the lexical index holds fewer documents than the corpus");
        }
        searcher_->setLexicalIndex(lexical_index_.get());
    }

    embedder_ = std::make_unique<OnnxQueryEmbedder>(
        config_.encoder_model_path, config_.device_id, tokenizer_, config_.encoder_max_seq_len,
        config_.encoder_output_name, corpus_->dim());

    if (!config_.reranker_model_path.empty()) {
        reranker_ = makeReranker(config_, tokenizer_);
    }
}

Engine::~Engine() = default;

std::string Engine::storageDtype() const {
    return scalarName(corpus_->embeddings().dtype());
}

SearchOptions Engine::defaultOptions() const {
    SearchOptions options;
    options.top_k = config_.default_top_k;
    options.candidates = config_.default_candidates;
    options.use_config_metric = true;
    options.rescore_candidates = -1;
    options.dense_weight = -1.0f;
    options.hybrid = config_.hybrid && lexical_index_ != nullptr;
    options.rerank = reranker_ != nullptr;
    options.use_config_backend = true;
    options.use_config_graphs = true;
    return options;
}

Engine::ResolvedOptions Engine::resolve(const SearchOptions& options,
                                        std::int64_t corpus_size) const {
    ResolvedOptions resolved;
    resolved.top_k = options.top_k > 0 ? options.top_k : config_.default_top_k;
    resolved.candidates = options.candidates > 0 ? options.candidates : config_.default_candidates;
    resolved.metric = options.use_config_metric ? config_.metric : options.metric;
    resolved.backend =
        options.use_config_backend ? config_.similarity_backend : options.backend;
    resolved.use_graph = options.use_config_graphs ? config_.use_cuda_graphs : options.use_graph;
    resolved.rerank = options.rerank && reranker_ != nullptr;
    resolved.hybrid = options.hybrid && lexical_index_ != nullptr;

    resolved.dense_weight =
        options.dense_weight >= 0.0f ? options.dense_weight : config_.dense_weight;
    resolved.dense_weight = std::min(1.0f, std::max(0.0f, resolved.dense_weight));

    if (resolved.top_k <= 0) {
        throw std::invalid_argument("zcgvs: top_k must be positive");
    }

    resolved.candidates = std::max(resolved.top_k, resolved.candidates);
    if (resolved.candidates > corpus_size) {
        resolved.candidates = static_cast<int>(corpus_size);
    }
    if (resolved.top_k > resolved.candidates) {
        resolved.top_k = resolved.candidates;
    }

    if (rescore_store_ != nullptr) {
        const int requested = options.rescore_candidates >= 0 ? options.rescore_candidates
                                                              : config_.rescore_candidates;
        resolved.rescore_candidates = std::max(requested, 0);
        if (resolved.rescore_candidates > 0) {
            resolved.rescore_candidates =
                std::max(resolved.rescore_candidates, resolved.candidates);
            if (resolved.rescore_candidates > corpus_size) {
                resolved.rescore_candidates = static_cast<int>(corpus_size);
            }
        }
    }
    return resolved;
}

const std::uint8_t* Engine::buildMask(const std::vector<int>& allow_ids,
                                      std::int64_t doc_count) const {
    if (allow_ids.empty()) {
        return nullptr;
    }
    host_mask_.assign(static_cast<std::size_t>(doc_count), 0);
    for (const int index : allow_ids) {
        if (index >= 0 && index < doc_count) {
            host_mask_[static_cast<std::size_t>(index)] = 1;
        }
    }
    if (d_mask_.size() < static_cast<std::size_t>(doc_count)) {
        d_mask_.allocate(static_cast<std::size_t>(doc_count));
    }
    d_mask_.copyFromHost(host_mask_.data(), static_cast<std::size_t>(doc_count), stream_->get());
    return d_mask_.data();
}

// WordPiece ids of the query, minus [CLS]/[SEP], deduplicated and capped. The offline index
// builder tokenizes the corpus with the same vocabulary, so ids line up on both sides.
std::vector<int> Engine::lexicalTerms(const std::string& query) const {
    const Encoding encoding = tokenizer_.encode(query, config_.encoder_max_seq_len);
    const std::size_t length = encoding.input_ids.size();

    std::vector<int> terms;
    if (length <= 2) {
        return terms;
    }

    std::unordered_set<int> seen;
    const int limit = std::max(1, config_.max_query_terms);
    for (std::size_t i = 1; i + 1 < length; ++i) {
        const int id = static_cast<int>(encoding.input_ids[i]);
        if (id < 0 || !seen.insert(id).second) {
            continue;
        }
        terms.push_back(id);
        if (static_cast<int>(terms.size()) >= limit) {
            break;
        }
    }
    return terms;
}

std::vector<SearchHit> Engine::assemble(const std::string& query, const HostTopK& selection,
                                        int query_index, const ResolvedOptions& resolved,
                                        double& rerank_ms) const {
    const std::int64_t corpus_size = corpus_->size();
    const std::size_t base = selection.offset(query_index);

    std::vector<int> corpus_indices;
    std::vector<float> similarities;
    std::vector<std::string> passages;
    corpus_indices.reserve(static_cast<std::size_t>(selection.k));
    similarities.reserve(static_cast<std::size_t>(selection.k));
    passages.reserve(static_cast<std::size_t>(selection.k));

    for (int i = 0; i < selection.k; ++i) {
        const int index = selection.indices[base + static_cast<std::size_t>(i)];
        if (index < 0 || index >= corpus_size) {
            continue;
        }
        corpus_indices.push_back(index);
        similarities.push_back(selection.scores[base + static_cast<std::size_t>(i)]);
        passages.push_back(corpus_->text(index));
    }

    std::vector<float> rerank_scores;
    if (resolved.rerank && !passages.empty()) {
        ZCGVS_NVTX_RANGE("zcgvs::rerank");
        const auto begin = Clock::now();
        rerank_scores = reranker_->score(query, passages);
        rerank_ms += elapsedMs(begin, Clock::now());
    } else {
        rerank_scores = similarities;
    }

    std::vector<SearchHit> hits;
    hits.reserve(corpus_indices.size());
    for (std::size_t i = 0; i < corpus_indices.size(); ++i) {
        SearchHit hit;
        hit.corpus_index = corpus_indices[i];
        hit.doc_id = corpus_->docId(corpus_indices[i]);
        hit.text = passages[i];
        hit.similarity = similarities[i];
        hit.rerank_score = rerank_scores[i];
        hits.push_back(std::move(hit));
    }

    std::sort(hits.begin(), hits.end(), [](const SearchHit& a, const SearchHit& b) {
        return a.rerank_score > b.rerank_score;
    });
    if (static_cast<int>(hits.size()) > resolved.top_k) {
        hits.resize(static_cast<std::size_t>(resolved.top_k));
    }
    return hits;
}

BatchSearchResponse Engine::searchBatch(const std::vector<std::string>& queries,
                                        const SearchOptions& options) const {
    ZCGVS_NVTX_RANGE("zcgvs::searchBatch");

    if (queries.empty()) {
        throw std::invalid_argument("zcgvs: searchBatch requires at least one query");
    }

    const auto t_start = Clock::now();
    const std::int64_t corpus_size = corpus_->size();
    const ResolvedOptions resolved = resolve(options, corpus_size);
    const int batch = static_cast<int>(queries.size());
    const int dimension = corpus_->dim();

    const auto t_embed_begin = Clock::now();
    std::vector<float> embeddings;
    embeddings.reserve(static_cast<std::size_t>(batch) * dimension);
    for (const std::string& query : queries) {
        const std::vector<float> vector = embedder_->embed(query);
        if (static_cast<int>(vector.size()) != dimension) {
            throw std::runtime_error("zcgvs: query embedding dim does not match the corpus");
        }
        embeddings.insert(embeddings.end(), vector.begin(), vector.end());
    }
    const auto t_embed_end = Clock::now();

    double tokenize_ms = 0.0;
    int lexical_stride = 0;
    if (resolved.hybrid) {
        const auto t_tokenize_begin = Clock::now();
        std::vector<std::vector<int>> per_query;
        per_query.reserve(queries.size());
        for (const std::string& query : queries) {
            per_query.push_back(lexicalTerms(query));
            lexical_stride = std::max(lexical_stride, static_cast<int>(per_query.back().size()));
        }
        if (lexical_stride > 0) {
            host_terms_.assign(static_cast<std::size_t>(batch) * lexical_stride, -1);
            for (std::size_t q = 0; q < per_query.size(); ++q) {
                std::copy(per_query[q].begin(), per_query[q].end(),
                          host_terms_.begin() + static_cast<std::ptrdiff_t>(q * lexical_stride));
            }
        }
        tokenize_ms = elapsedMs(t_tokenize_begin, Clock::now());
    }

    SearchParams params;
    params.k = resolved.candidates;
    params.metric = resolved.metric;
    params.d_mask = buildMask(options.allow, corpus_size);
    params.rescore_candidates = resolved.rescore_candidates;
    params.backend = resolved.backend;
    params.use_graph = resolved.use_graph;
    if (resolved.hybrid && lexical_stride > 0) {
        params.host_query_terms = host_terms_.data();
        params.lexical_stride = lexical_stride;
        params.dense_weight = resolved.dense_weight;
    }

    searcher_->search(embeddings.data(), dimension, batch, params, *stream_);

    const auto t_copy_begin = Clock::now();
    const HostTopK selection = searcher_->copyResultToHost(*stream_);
    const auto t_copy_end = Clock::now();

    double rerank_ms = 0.0;
    BatchSearchResponse response;
    response.results.reserve(queries.size());
    for (int q = 0; q < batch; ++q) {
        response.results.push_back(assemble(queries[static_cast<std::size_t>(q)], selection, q,
                                            resolved, rerank_ms));
    }

    response.timing.tokenize_ms = tokenize_ms;
    response.timing.embed_ms = elapsedMs(t_embed_begin, t_embed_end);
    response.timing.search_gpu_ms = searcher_->lastSearchGpuMs();
    response.timing.copy_ms = elapsedMs(t_copy_begin, t_copy_end);
    response.timing.rerank_ms = rerank_ms;
    response.timing.total_ms = elapsedMs(t_start, Clock::now());
    response.timing.batch = batch;
    response.timing.graph_replay = searcher_->lastSearchUsedGraph();
    return response;
}

SearchResponse Engine::searchTimed(const std::string& query, const SearchOptions& options) const {
    BatchSearchResponse batch = searchBatch({query}, options);
    SearchResponse response;
    response.hits = std::move(batch.results.front());
    response.timing = batch.timing;
    return response;
}

SearchResponse Engine::searchTimed(const std::string& query, int top_k, int candidates,
                                   Metric metric, const std::vector<int>& allow_ids) const {
    SearchOptions options = defaultOptions();
    options.top_k = top_k;
    options.candidates = candidates;
    options.use_config_metric = false;
    options.metric = metric;
    options.allow = allow_ids;
    return searchTimed(query, options);
}

std::vector<SearchHit> Engine::search(const std::string& query, int top_k, int candidates) const {
    SearchOptions options = defaultOptions();
    options.top_k = top_k;
    options.candidates = candidates;
    return searchTimed(query, options).hits;
}

std::vector<SearchHit> Engine::search(const std::string& query) const {
    return searchTimed(query, defaultOptions()).hits;
}

std::vector<Candidate> Engine::retrieve(const std::string& query, int k, Metric metric,
                                        const std::vector<int>& allow_ids) const {
    ZCGVS_NVTX_RANGE("zcgvs::retrieve");

    if (k <= 0) {
        throw std::invalid_argument("zcgvs: k must be positive");
    }

    SearchOptions options = defaultOptions();
    options.top_k = k;
    options.candidates = k;
    options.use_config_metric = false;
    options.metric = metric;
    options.allow = allow_ids;

    const std::int64_t corpus_size = corpus_->size();
    const ResolvedOptions resolved = resolve(options, corpus_size);
    const int dimension = corpus_->dim();

    const std::vector<float> embedding = embedder_->embed(query);

    SearchParams params;
    params.k = resolved.candidates;
    params.metric = resolved.metric;
    params.d_mask = buildMask(allow_ids, corpus_size);
    params.rescore_candidates = resolved.rescore_candidates;
    params.backend = resolved.backend;
    params.use_graph = resolved.use_graph;

    searcher_->search(embedding.data(), dimension, 1, params, *stream_);
    const HostTopK selection = searcher_->copyResultToHost(*stream_);

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(selection.k));
    for (int i = 0; i < selection.k; ++i) {
        const int index = selection.indices[static_cast<std::size_t>(i)];
        if (index < 0 || index >= corpus_size) {
            continue;
        }
        Candidate candidate;
        candidate.corpus_index = index;
        candidate.doc_id = corpus_->docId(index);
        candidate.score = selection.scores[static_cast<std::size_t>(i)];
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

void Engine::addDocuments(const std::vector<std::string>& ids,
                          const std::vector<std::string>& texts) {
    if (ids.size() != texts.size()) {
        throw std::invalid_argument("zcgvs: ids and texts must have equal length");
    }
    if (ids.empty()) {
        return;
    }
    if (rescore_store_ != nullptr || lexical_index_ != nullptr) {
        throw std::runtime_error(
            "zcgvs: incremental indexing is unavailable while a rescore store or lexical index "
            "is attached; rebuild those artifacts offline instead");
    }

    const int dimension = corpus_->dim();
    std::vector<float> vectors;
    vectors.reserve(ids.size() * static_cast<std::size_t>(dimension));
    for (const std::string& text : texts) {
        const std::vector<float> embedding = embedder_->embed(text);
        if (static_cast<int>(embedding.size()) != dimension) {
            throw std::runtime_error("zcgvs: embedding dim mismatch while adding documents");
        }
        vectors.insert(vectors.end(), embedding.begin(), embedding.end());
    }

    corpus_->addDocuments(ids, texts, vectors.data(), static_cast<std::int64_t>(ids.size()),
                          stream_->get());
}

EngineStats Engine::stats() const {
    EngineStats stats;
    stats.corpus_size = corpus_->size();
    stats.dim = corpus_->dim();
    stats.storage_dtype = scalarName(corpus_->embeddings().dtype());
    stats.corpus_bytes = corpus_->embeddings().deviceBytes();

    if (rescore_store_ != nullptr) {
        stats.has_rescore_store = true;
        stats.rescore_dtype = scalarName(rescore_store_->dtype());
        stats.rescore_bytes = rescore_store_->deviceBytes();
    }
    if (lexical_index_ != nullptr) {
        stats.has_lexical_index = true;
        stats.lexical_postings = lexical_index_->postingCount();
        stats.lexical_bytes = lexical_index_->deviceBytes();
    }
    stats.has_reranker = reranker_ != nullptr;
    stats.graph_captures = searcher_->graphCaptureCount();
    stats.graph_launches = searcher_->graphLaunchCount();

    const DeviceInfo info = queryDevice(config_.device_id);
    stats.device_name = info.name;
    stats.device_total_memory = info.total_memory;
    stats.compute_major = info.compute_major;
    stats.compute_minor = info.compute_minor;
    return stats;
}

}  // namespace zcgvs
