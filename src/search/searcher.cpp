#include "zcgvs/search/searcher.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/core/nvtx.hpp"
#include "zcgvs/cuda/convert.cuh"
#include "zcgvs/cuda/cublas_similarity.hpp"
#include "zcgvs/cuda/lexical.cuh"
#include "zcgvs/cuda/rescore.cuh"
#include "zcgvs/cuda/similarity.cuh"
#include "zcgvs/cuda/topk.cuh"

namespace zcgvs {

namespace {

template <typename Buffer>
bool grow(Buffer& buffer, std::size_t required) {
    if (buffer.size() >= required) {
        return false;
    }
    buffer.allocate(required);
    return true;
}

}  // namespace

bool Searcher::GraphKey::matches(const GraphKey& other) const {
    return corpus == other.corpus && rescore_corpus == other.rescore_corpus &&
           lexical_index == other.lexical_index && mask == other.mask &&
           generation == other.generation && doc_count == other.doc_count && k == other.k &&
           batch == other.batch && rescore_candidates == other.rescore_candidates &&
           lexical_stride == other.lexical_stride && metric == other.metric &&
           dense_weight == other.dense_weight;
}

Searcher::Searcher(const EmbeddingStore& store) : store_(store), dim_(store.dim()) {
    ZCGVS_CUDA_CHECK(cudaEventCreate(&ev_start_));
    ZCGVS_CUDA_CHECK(cudaEventCreate(&ev_end_));
}

Searcher::~Searcher() {
    destroyGraph();
    if (ev_start_ != nullptr) {
        cudaEventDestroy(ev_start_);
    }
    if (ev_end_ != nullptr) {
        cudaEventDestroy(ev_end_);
    }
}

void Searcher::destroyGraph() noexcept {
    if (graph_exec_ != nullptr) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    graph_valid_ = false;
}

void Searcher::setRescoreStore(const EmbeddingStore* store) {
    if (store != nullptr) {
        if (store->dim() != dim_) {
            throw std::invalid_argument("zcgvs: rescore store dim does not match the corpus");
        }
        if (store->dtype() == ScalarType::Binary) {
            throw std::invalid_argument("zcgvs: rescore store must not be binary");
        }
    }
    rescore_store_ = store;
    destroyGraph();
}

void Searcher::setLexicalIndex(const LexicalIndex* index) {
    lexical_index_ = index;
    destroyGraph();
}

bool Searcher::lexicalEnabled(const SearchParams& params) const {
    return lexical_index_ != nullptr && params.lexical_stride > 0 &&
           params.host_query_terms != nullptr;
}

bool Searcher::rescoreEnabled(const SearchParams& params) const {
    return rescore_store_ != nullptr && params.rescore_candidates > 0;
}

void Searcher::validate(int dim, int batch, const SearchParams& params,
                        std::int64_t doc_count) const {
    if (dim != dim_) {
        throw std::invalid_argument("zcgvs: query dim " + std::to_string(dim) +
                                    " does not match corpus dim " + std::to_string(dim_));
    }
    if (batch <= 0) {
        throw std::invalid_argument("zcgvs: batch must be positive");
    }
    if (params.k <= 0) {
        throw std::invalid_argument("zcgvs: k must be positive");
    }
    if (doc_count > static_cast<std::int64_t>(INT32_MAX)) {
        throw std::runtime_error("zcgvs: corpus exceeds the supported document count");
    }
    if (params.k > doc_count) {
        throw std::invalid_argument("zcgvs: k=" + std::to_string(params.k) +
                                    " exceeds corpus size " + std::to_string(doc_count));
    }
    if (params.k > cuda::topKMaxK()) {
        throw std::invalid_argument("zcgvs: k=" + std::to_string(params.k) +
                                    " exceeds kernel capacity " +
                                    std::to_string(cuda::topKMaxK()));
    }
    if (rescoreEnabled(params)) {
        if (params.rescore_candidates < params.k) {
            throw std::invalid_argument("zcgvs: rescore_candidates must be at least k");
        }
        if (params.rescore_candidates > doc_count) {
            throw std::invalid_argument("zcgvs: rescore_candidates exceeds corpus size");
        }
        if (params.rescore_candidates > cuda::topKMaxK()) {
            throw std::invalid_argument("zcgvs: rescore_candidates=" +
                                        std::to_string(params.rescore_candidates) +
                                        " exceeds kernel capacity " +
                                        std::to_string(cuda::topKMaxK()));
        }
        if (rescore_store_->docCount() < doc_count) {
            throw std::invalid_argument("zcgvs: rescore store holds fewer documents than the corpus");
        }
    }
    if (lexicalEnabled(params) && lexical_index_->docCount() < doc_count) {
        throw std::invalid_argument("zcgvs: lexical index holds fewer documents than the corpus");
    }
}

void Searcher::ensureCapacity(std::int64_t doc_count, int batch, const SearchParams& params) {
    const std::size_t queries = static_cast<std::size_t>(batch);
    const std::size_t docs = static_cast<std::size_t>(doc_count);
    const int selection_width =
        rescoreEnabled(params) ? params.rescore_candidates : params.k;

    bool reallocated = false;
    reallocated |= grow(d_queries_, queries * static_cast<std::size_t>(dim_));
    reallocated |= grow(d_scores_, queries * docs);

    const std::size_t scratch =
        queries * static_cast<std::size_t>(
                      cuda::topKScratchElements(static_cast<int>(doc_count), selection_width));
    reallocated |= grow(d_scratch_values_a_, scratch);
    reallocated |= grow(d_scratch_indices_a_, scratch);
    reallocated |= grow(d_scratch_values_b_, scratch);
    reallocated |= grow(d_scratch_indices_b_, scratch);

    reallocated |= grow(d_out_indices_, queries * static_cast<std::size_t>(params.k));
    reallocated |= grow(d_out_values_, queries * static_cast<std::size_t>(params.k));

    if (store_.dtype() == ScalarType::Binary) {
        reallocated |= grow(d_query_bits_, queries * binaryWords(dim_));
    }

    if (rescoreEnabled(params)) {
        const std::size_t candidates =
            queries * static_cast<std::size_t>(params.rescore_candidates);
        reallocated |= grow(d_coarse_indices_, candidates);
        reallocated |= grow(d_coarse_values_, candidates);
        reallocated |= grow(d_rescored_, candidates);
        reallocated |= grow(d_local_indices_, queries * static_cast<std::size_t>(params.k));
    }

    if (lexicalEnabled(params)) {
        reallocated |= grow(d_lexical_scores_, queries * docs);
        reallocated |= grow(d_lexical_max_, queries);
        reallocated |= grow(d_query_terms_,
                            queries * static_cast<std::size_t>(params.lexical_stride));
        if (h_query_terms_.size() < queries * static_cast<std::size_t>(params.lexical_stride)) {
            h_query_terms_.allocate(queries * static_cast<std::size_t>(params.lexical_stride));
        }
    }

    if (h_queries_.size() < queries * static_cast<std::size_t>(dim_)) {
        h_queries_.allocate(queries * static_cast<std::size_t>(dim_));
    }
    if (h_out_indices_.size() < queries * static_cast<std::size_t>(params.k)) {
        h_out_indices_.allocate(queries * static_cast<std::size_t>(params.k));
        h_out_values_.allocate(queries * static_cast<std::size_t>(params.k));
    }

    if (reallocated) {
        ++generation_;
        destroyGraph();
    }
}

void Searcher::uploadInputs(const float* host_queries, int batch, const SearchParams& params,
                            const CudaStream& stream) {
    const std::size_t elements = static_cast<std::size_t>(batch) * static_cast<std::size_t>(dim_);
    std::memcpy(h_queries_.data(), host_queries, elements * sizeof(float));
    d_queries_.copyFromHost(h_queries_.data(), elements, stream.get());

    if (lexicalEnabled(params)) {
        const std::size_t terms =
            static_cast<std::size_t>(batch) * static_cast<std::size_t>(params.lexical_stride);
        std::memcpy(h_query_terms_.data(), params.host_query_terms, terms * sizeof(int));
        d_query_terms_.copyFromHost(h_query_terms_.data(), terms, stream.get());
    }
}

void Searcher::recordPipeline(std::int64_t doc_count, int batch, const SearchParams& params,
                              cudaStream_t stream) {
    const int n = static_cast<int>(doc_count);
    const std::size_t score_elements = static_cast<std::size_t>(batch) * static_cast<std::size_t>(n);

    const void* query_data = d_queries_.data();
    if (store_.dtype() == ScalarType::Binary) {
        cuda::launchPackBinary(d_queries_.data(), d_query_bits_.data(), batch, dim_, stream);
        query_data = d_query_bits_.data();
    }

    const bool cublas_path =
        params.backend == SimilarityBackend::Cublas &&
        CublasSimilarity::supports(store_.dtype(), params.metric, params.d_mask != nullptr);
    if (cublas_path) {
        if (!cublas_) {
            cublas_ = std::make_unique<CublasSimilarity>();
        }
        cublas_->run(static_cast<const float*>(store_.deviceData()), n, dim_, d_queries_.data(),
                     batch, d_scores_.data(), stream);
    } else {
        cuda::launchSimilarity(store_.deviceData(), store_.dtype(), store_.scale(), n, dim_,
                               query_data, batch, params.d_mask, params.metric, d_scores_.data(),
                               stream);
    }

    if (lexicalEnabled(params)) {
        ZCGVS_CUDA_CHECK(cudaMemsetAsync(d_lexical_scores_.data(), 0,
                                         score_elements * sizeof(float), stream));
        ZCGVS_CUDA_CHECK(cudaMemsetAsync(d_lexical_max_.data(), 0,
                                         static_cast<std::size_t>(batch) * sizeof(float), stream));

        for (int query = 0; query < batch; ++query) {
            cuda::launchBm25(lexical_index_->deviceOffsets(), lexical_index_->devicePostingDocs(),
                             lexical_index_->devicePostingWeights(),
                             d_query_terms_.data() +
                                 static_cast<std::size_t>(query) * params.lexical_stride,
                             params.lexical_stride, n,
                             d_lexical_scores_.data() + static_cast<std::size_t>(query) * n,
                             stream);
        }
        for (int query = 0; query < batch; ++query) {
            cuda::launchMaxPositive(d_lexical_scores_.data() + static_cast<std::size_t>(query) * n,
                                    n, d_lexical_max_.data() + query, stream);
        }
        for (int query = 0; query < batch; ++query) {
            cuda::launchFuseScores(d_scores_.data() + static_cast<std::size_t>(query) * n,
                                   d_lexical_scores_.data() + static_cast<std::size_t>(query) * n,
                                   d_lexical_max_.data() + query, params.dense_weight, n, stream);
        }
    }

    if (rescoreEnabled(params)) {
        const int candidates = params.rescore_candidates;
        cuda::launchTopK(d_scores_.data(), n, candidates, batch, d_coarse_indices_.data(),
                         d_coarse_values_.data(), d_scratch_values_a_.data(),
                         d_scratch_indices_a_.data(), d_scratch_values_b_.data(),
                         d_scratch_indices_b_.data(), stream);
        cuda::launchGatherRescore(rescore_store_->deviceData(), rescore_store_->dtype(),
                                  rescore_store_->scale(),
                                  static_cast<int>(rescore_store_->docCount()), dim_,
                                  d_queries_.data(), batch, d_coarse_indices_.data(), candidates,
                                  params.metric, d_rescored_.data(), stream);
        cuda::launchTopK(d_rescored_.data(), candidates, params.k, batch, d_local_indices_.data(),
                         d_out_values_.data(), d_scratch_values_a_.data(),
                         d_scratch_indices_a_.data(), d_scratch_values_b_.data(),
                         d_scratch_indices_b_.data(), stream);
        cuda::launchRemapIndices(d_coarse_indices_.data(), d_local_indices_.data(), candidates,
                                 params.k, batch, d_out_indices_.data(), stream);
        return;
    }

    cuda::launchTopK(d_scores_.data(), n, params.k, batch, d_out_indices_.data(),
                     d_out_values_.data(), d_scratch_values_a_.data(),
                     d_scratch_indices_a_.data(), d_scratch_values_b_.data(),
                     d_scratch_indices_b_.data(), stream);
}

Searcher::GraphKey Searcher::makeKey(std::int64_t doc_count, int batch,
                                     const SearchParams& params) const {
    GraphKey key;
    key.corpus = store_.deviceData();
    key.rescore_corpus = rescoreEnabled(params) ? rescore_store_->deviceData() : nullptr;
    key.lexical_index = lexicalEnabled(params) ? static_cast<const void*>(lexical_index_) : nullptr;
    key.mask = params.d_mask;
    key.generation = generation_;
    key.doc_count = doc_count;
    key.k = params.k;
    key.batch = batch;
    key.rescore_candidates = rescoreEnabled(params) ? params.rescore_candidates : 0;
    key.lexical_stride = lexicalEnabled(params) ? params.lexical_stride : 0;
    key.metric = static_cast<int>(params.metric);
    key.dense_weight = lexicalEnabled(params) ? params.dense_weight : 1.0f;
    return key;
}

void Searcher::captureGraph(std::int64_t doc_count, int batch, const SearchParams& params,
                            const CudaStream& stream) {
    cudaGraph_t graph = nullptr;

    // Drain the query uploads already queued for this request before switching the stream into
    // capture mode, so the graph contains the pipeline and nothing else.
    stream.synchronize();

    ZCGVS_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    try {
        recordPipeline(doc_count, batch, params, stream.get());
    } catch (...) {
        cudaStreamEndCapture(stream.get(), &graph);
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        throw;
    }
    ZCGVS_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));

    destroyGraph();
    const cudaError_t status = cudaGraphInstantiateWithFlags(&graph_exec_, graph, 0);
    cudaGraphDestroy(graph);
    ZCGVS_CUDA_CHECK(status);

    graph_valid_ = true;
    ++graph_captures_;
}

// Capture happens outside the timed region so the reported GPU time is pure replay.
void Searcher::launchViaGraph(std::int64_t doc_count, int batch, const SearchParams& params,
                              const CudaStream& stream) {
    const GraphKey key = makeKey(doc_count, batch, params);
    if (!graph_valid_ || !graph_key_.matches(key)) {
        captureGraph(doc_count, batch, params, stream);
        graph_key_ = key;
    }
}

void Searcher::search(const float* host_queries, int dim, int batch, const SearchParams& params,
                      const CudaStream& stream) {
    ZCGVS_NVTX_RANGE("zcgvs::search");

    const std::int64_t doc_count = store_.docCount();
    validate(dim, batch, params, doc_count);
    ensureCapacity(doc_count, batch, params);
    uploadInputs(host_queries, batch, params, stream);

    const bool graph_path = params.use_graph && params.backend == SimilarityBackend::Custom;
    if (graph_path) {
        launchViaGraph(doc_count, batch, params, stream);
    }

    ZCGVS_CUDA_CHECK(cudaEventRecord(ev_start_, stream.get()));
    if (graph_path) {
        ZCGVS_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream.get()));
        ++graph_launches_;
    } else {
        recordPipeline(doc_count, batch, params, stream.get());
    }
    ZCGVS_CUDA_CHECK(cudaEventRecord(ev_end_, stream.get()));

    used_graph_ = graph_path;
    current_k_ = params.k;
    current_batch_ = batch;
}

HostTopK Searcher::copyResultToHost(const CudaStream& stream) {
    ZCGVS_NVTX_RANGE("zcgvs::copyResultToHost");

    if (current_k_ == 0 || current_batch_ == 0) {
        throw std::runtime_error("zcgvs: copyResultToHost called before search");
    }

    const std::size_t elements =
        static_cast<std::size_t>(current_batch_) * static_cast<std::size_t>(current_k_);

    d_out_indices_.copyToHost(h_out_indices_.data(), elements, stream.get());
    d_out_values_.copyToHost(h_out_values_.data(), elements, stream.get());
    stream.synchronize();

    HostTopK result;
    result.batch = current_batch_;
    result.k = current_k_;
    result.indices.assign(h_out_indices_.data(), h_out_indices_.data() + elements);
    result.scores.assign(h_out_values_.data(), h_out_values_.data() + elements);

    float elapsed_ms = 0.0f;
    ZCGVS_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start_, ev_end_));
    last_gpu_ms_ = static_cast<double>(elapsed_ms);

    return result;
}

}  // namespace zcgvs
