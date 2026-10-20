#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "zcgvs/core/metric.hpp"
#include "zcgvs/engine/engine.hpp"
#include "zcgvs/engine/engine_config.hpp"
#include "zcgvs/engine/engine_stats.hpp"
#include "zcgvs/engine/search_hit.hpp"
#include "zcgvs/engine/search_options.hpp"
#include "zcgvs/engine/search_response.hpp"
#include "zcgvs/search/search_params.hpp"

namespace py = pybind11;

PYBIND11_MODULE(zcgvs_engine, module) {
    module.doc() = "Zero-Copy GPU Vector Search & Re-Ranking Engine";

    py::enum_<zcgvs::Metric>(module, "Metric")
        .value("Cosine", zcgvs::Metric::Cosine)
        .value("DotProduct", zcgvs::Metric::DotProduct)
        .value("L2", zcgvs::Metric::L2);

    py::enum_<zcgvs::RerankerBackend>(module, "RerankerBackend")
        .value("Onnx", zcgvs::RerankerBackend::Onnx);

    py::enum_<zcgvs::SimilarityBackend>(module, "SimilarityBackend")
        .value("Custom", zcgvs::SimilarityBackend::Custom)
        .value("Cublas", zcgvs::SimilarityBackend::Cublas);

    py::class_<zcgvs::EngineConfig>(module, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("embeddings_path", &zcgvs::EngineConfig::embeddings_path)
        .def_readwrite("metadata_path", &zcgvs::EngineConfig::metadata_path)
        .def_readwrite("vocab_path", &zcgvs::EngineConfig::vocab_path)
        .def_readwrite("rescore_embeddings_path", &zcgvs::EngineConfig::rescore_embeddings_path)
        .def_readwrite("lexical_index_path", &zcgvs::EngineConfig::lexical_index_path)
        .def_readwrite("encoder_model_path", &zcgvs::EngineConfig::encoder_model_path)
        .def_readwrite("encoder_output_name", &zcgvs::EngineConfig::encoder_output_name)
        .def_readwrite("encoder_max_seq_len", &zcgvs::EngineConfig::encoder_max_seq_len)
        .def_readwrite("reranker_backend", &zcgvs::EngineConfig::reranker_backend)
        .def_readwrite("reranker_model_path", &zcgvs::EngineConfig::reranker_model_path)
        .def_readwrite("reranker_output_name", &zcgvs::EngineConfig::reranker_output_name)
        .def_readwrite("reranker_max_seq_len", &zcgvs::EngineConfig::reranker_max_seq_len)
        .def_readwrite("reranker_batch_size", &zcgvs::EngineConfig::reranker_batch_size)
        .def_readwrite("device_id", &zcgvs::EngineConfig::device_id)
        .def_readwrite("lowercase", &zcgvs::EngineConfig::lowercase)
        .def_readwrite("metric", &zcgvs::EngineConfig::metric)
        .def_readwrite("similarity_backend", &zcgvs::EngineConfig::similarity_backend)
        .def_readwrite("default_top_k", &zcgvs::EngineConfig::default_top_k)
        .def_readwrite("default_candidates", &zcgvs::EngineConfig::default_candidates)
        .def_readwrite("rescore_candidates", &zcgvs::EngineConfig::rescore_candidates)
        .def_readwrite("hybrid", &zcgvs::EngineConfig::hybrid)
        .def_readwrite("dense_weight", &zcgvs::EngineConfig::dense_weight)
        .def_readwrite("max_query_terms", &zcgvs::EngineConfig::max_query_terms)
        .def_readwrite("use_cuda_graphs", &zcgvs::EngineConfig::use_cuda_graphs);

    py::class_<zcgvs::SearchOptions>(module, "SearchOptions")
        .def(py::init<>())
        .def_readwrite("top_k", &zcgvs::SearchOptions::top_k)
        .def_readwrite("candidates", &zcgvs::SearchOptions::candidates)
        .def_readwrite("use_config_metric", &zcgvs::SearchOptions::use_config_metric)
        .def_readwrite("metric", &zcgvs::SearchOptions::metric)
        .def_readwrite("allow", &zcgvs::SearchOptions::allow)
        .def_readwrite("rescore_candidates", &zcgvs::SearchOptions::rescore_candidates)
        .def_readwrite("dense_weight", &zcgvs::SearchOptions::dense_weight)
        .def_readwrite("hybrid", &zcgvs::SearchOptions::hybrid)
        .def_readwrite("rerank", &zcgvs::SearchOptions::rerank)
        .def_readwrite("use_config_backend", &zcgvs::SearchOptions::use_config_backend)
        .def_readwrite("backend", &zcgvs::SearchOptions::backend)
        .def_readwrite("use_config_graphs", &zcgvs::SearchOptions::use_config_graphs)
        .def_readwrite("use_graph", &zcgvs::SearchOptions::use_graph);

    py::class_<zcgvs::SearchHit>(module, "SearchHit")
        .def_readonly("doc_id", &zcgvs::SearchHit::doc_id)
        .def_readonly("text", &zcgvs::SearchHit::text)
        .def_readonly("similarity", &zcgvs::SearchHit::similarity)
        .def_readonly("rerank_score", &zcgvs::SearchHit::rerank_score)
        .def_readonly("corpus_index", &zcgvs::SearchHit::corpus_index)
        .def("__repr__", [](const zcgvs::SearchHit& hit) {
            return "<SearchHit doc_id='" + hit.doc_id +
                   "' rerank_score=" + std::to_string(hit.rerank_score) + ">";
        });

    py::class_<zcgvs::Candidate>(module, "Candidate")
        .def_readonly("doc_id", &zcgvs::Candidate::doc_id)
        .def_readonly("score", &zcgvs::Candidate::score)
        .def_readonly("corpus_index", &zcgvs::Candidate::corpus_index)
        .def("__repr__", [](const zcgvs::Candidate& candidate) {
            return "<Candidate doc_id='" + candidate.doc_id +
                   "' score=" + std::to_string(candidate.score) + ">";
        });

    py::class_<zcgvs::SearchTiming>(module, "SearchTiming")
        .def_readonly("tokenize_ms", &zcgvs::SearchTiming::tokenize_ms)
        .def_readonly("embed_ms", &zcgvs::SearchTiming::embed_ms)
        .def_readonly("search_gpu_ms", &zcgvs::SearchTiming::search_gpu_ms)
        .def_readonly("copy_ms", &zcgvs::SearchTiming::copy_ms)
        .def_readonly("rerank_ms", &zcgvs::SearchTiming::rerank_ms)
        .def_readonly("total_ms", &zcgvs::SearchTiming::total_ms)
        .def_readonly("batch", &zcgvs::SearchTiming::batch)
        .def_readonly("graph_replay", &zcgvs::SearchTiming::graph_replay)
        .def("__repr__", [](const zcgvs::SearchTiming& timing) {
            return "<SearchTiming total_ms=" + std::to_string(timing.total_ms) +
                   " gpu_ms=" + std::to_string(timing.search_gpu_ms) + ">";
        });

    py::class_<zcgvs::SearchResponse>(module, "SearchResponse")
        .def_readonly("hits", &zcgvs::SearchResponse::hits)
        .def_readonly("timing", &zcgvs::SearchResponse::timing);

    py::class_<zcgvs::BatchSearchResponse>(module, "BatchSearchResponse")
        .def_readonly("results", &zcgvs::BatchSearchResponse::results)
        .def_readonly("timing", &zcgvs::BatchSearchResponse::timing);

    py::class_<zcgvs::EngineStats>(module, "EngineStats")
        .def_readonly("corpus_size", &zcgvs::EngineStats::corpus_size)
        .def_readonly("dim", &zcgvs::EngineStats::dim)
        .def_readonly("storage_dtype", &zcgvs::EngineStats::storage_dtype)
        .def_readonly("corpus_bytes", &zcgvs::EngineStats::corpus_bytes)
        .def_readonly("has_rescore_store", &zcgvs::EngineStats::has_rescore_store)
        .def_readonly("rescore_dtype", &zcgvs::EngineStats::rescore_dtype)
        .def_readonly("rescore_bytes", &zcgvs::EngineStats::rescore_bytes)
        .def_readonly("has_lexical_index", &zcgvs::EngineStats::has_lexical_index)
        .def_readonly("lexical_postings", &zcgvs::EngineStats::lexical_postings)
        .def_readonly("lexical_bytes", &zcgvs::EngineStats::lexical_bytes)
        .def_readonly("has_reranker", &zcgvs::EngineStats::has_reranker)
        .def_readonly("graph_captures", &zcgvs::EngineStats::graph_captures)
        .def_readonly("graph_launches", &zcgvs::EngineStats::graph_launches)
        .def_readonly("device_name", &zcgvs::EngineStats::device_name)
        .def_readonly("device_total_memory", &zcgvs::EngineStats::device_total_memory)
        .def_readonly("compute_major", &zcgvs::EngineStats::compute_major)
        .def_readonly("compute_minor", &zcgvs::EngineStats::compute_minor);

    py::class_<zcgvs::Engine>(module, "Engine")
        .def(py::init<const zcgvs::EngineConfig&>(), py::arg("config"),
             py::call_guard<py::gil_scoped_release>())
        .def(
            "search",
            [](const zcgvs::Engine& engine, const std::string& query, int top_k, int candidates) {
                return engine.search(query, top_k, candidates);
            },
            py::arg("query"), py::arg("top_k"), py::arg("candidates"),
            py::call_guard<py::gil_scoped_release>())
        .def(
            "search",
            [](const zcgvs::Engine& engine, const std::string& query) {
                return engine.search(query);
            },
            py::arg("query"), py::call_guard<py::gil_scoped_release>())
        .def(
            "search_timed",
            [](const zcgvs::Engine& engine, const std::string& query, int top_k, int candidates,
               zcgvs::Metric metric, const std::vector<int>& allow) {
                return engine.searchTimed(query, top_k, candidates, metric, allow);
            },
            py::arg("query"), py::arg("top_k"), py::arg("candidates"),
            py::arg("metric") = zcgvs::Metric::Cosine, py::arg("allow") = std::vector<int>{},
            py::call_guard<py::gil_scoped_release>())
        .def(
            "search_with",
            [](const zcgvs::Engine& engine, const std::string& query,
               const zcgvs::SearchOptions& options) { return engine.searchTimed(query, options); },
            py::arg("query"), py::arg("options"), py::call_guard<py::gil_scoped_release>())
        .def(
            "search_batch",
            [](const zcgvs::Engine& engine, const std::vector<std::string>& queries,
               const zcgvs::SearchOptions& options) {
                return engine.searchBatch(queries, options);
            },
            py::arg("queries"), py::arg("options"), py::call_guard<py::gil_scoped_release>())
        .def(
            "retrieve",
            [](const zcgvs::Engine& engine, const std::string& query, int k, zcgvs::Metric metric,
               const std::vector<int>& allow) { return engine.retrieve(query, k, metric, allow); },
            py::arg("query"), py::arg("k"), py::arg("metric") = zcgvs::Metric::Cosine,
            py::arg("allow") = std::vector<int>{}, py::call_guard<py::gil_scoped_release>())
        .def(
            "add_documents",
            [](zcgvs::Engine& engine, const std::vector<std::string>& ids,
               const std::vector<std::string>& texts) { engine.addDocuments(ids, texts); },
            py::arg("ids"), py::arg("texts"), py::call_guard<py::gil_scoped_release>())
        .def("default_options", &zcgvs::Engine::defaultOptions)
        .def("stats", &zcgvs::Engine::stats, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("corpus_size", &zcgvs::Engine::corpusSize)
        .def_property_readonly("dim", &zcgvs::Engine::dim)
        .def_property_readonly("storage_dtype", &zcgvs::Engine::storageDtype)
        .def_property_readonly("has_lexical_index", &zcgvs::Engine::hasLexicalIndex)
        .def_property_readonly("has_rescore_store", &zcgvs::Engine::hasRescoreStore)
        .def_property_readonly("has_reranker", &zcgvs::Engine::hasReranker);
}
