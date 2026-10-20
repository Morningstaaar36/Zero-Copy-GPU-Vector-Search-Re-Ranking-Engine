#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "zcgvs/engine/engine.hpp"

namespace {

std::unordered_map<std::string, std::string> parseFlags(int argc, char** argv) {
    std::unordered_map<std::string, std::string> flags;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        key = key.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            flags[key] = argv[i + 1];
            ++i;
        } else {
            flags[key] = "1";
        }
    }
    return flags;
}

std::string require(const std::unordered_map<std::string, std::string>& flags,
                    const std::string& key) {
    const auto it = flags.find(key);
    if (it == flags.end()) {
        throw std::runtime_error("missing required flag --" + key);
    }
    return it->second;
}

std::string optional(const std::unordered_map<std::string, std::string>& flags,
                     const std::string& key, const std::string& fallback) {
    const auto it = flags.find(key);
    return it == flags.end() ? fallback : it->second;
}

bool flagSet(const std::unordered_map<std::string, std::string>& flags, const std::string& key) {
    return flags.find(key) != flags.end();
}

std::string snippet(const std::string& text, std::size_t width) {
    if (text.size() <= width) {
        return text;
    }
    return text.substr(0, width) + "...";
}

std::string megabytes(std::size_t bytes) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buffer;
}

void printUsage() {
    std::cout << R"(zcgvs_search_cli — query the zero-copy GPU vector search engine

Required:
  --embeddings PATH        primary corpus matrix (.bin)
  --metadata PATH          corpus metadata (.meta)
  --vocab PATH             WordPiece vocabulary
  --encoder PATH           bi-encoder ONNX model
  --query TEXT             query string

Optional:
  --reranker PATH          cross-encoder ONNX model (omit for retrieval-only)
  --rescore-embeddings P   exact-precision corpus used by the on-device rescore stage
  --rescore-candidates N   coarse candidate width for the rescore stage
  --lexical PATH           CSR inverted index (.lex) enabling hybrid retrieval
  --hybrid                 fuse BM25 with the dense score
  --dense-weight F         convex weight on the dense score (default 0.7)
  --metric NAME            cosine | dot | l2   (ignored for binary storage)
  --backend NAME           custom | cublas
  --graphs                 replay the device pipeline from a CUDA graph
  --top-k N                results to return (default 10)
  --candidates N           candidates fed to the reranker (default 100)
  --repeat N               re-run the query N times and report the last timing
  --device N               CUDA device id
  --stats                  print engine and device statistics
)";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto flags = parseFlags(argc, argv);
        if (argc <= 1 || flagSet(flags, "help")) {
            printUsage();
            return EXIT_SUCCESS;
        }

        zcgvs::EngineConfig config;
        config.embeddings_path = require(flags, "embeddings");
        config.metadata_path = require(flags, "metadata");
        config.vocab_path = require(flags, "vocab");
        config.encoder_model_path = require(flags, "encoder");
        config.reranker_model_path = optional(flags, "reranker", "");
        config.rescore_embeddings_path = optional(flags, "rescore-embeddings", "");
        config.lexical_index_path = optional(flags, "lexical", "");
        config.encoder_output_name = optional(flags, "encoder-output", config.encoder_output_name);
        config.reranker_output_name =
            optional(flags, "reranker-output", config.reranker_output_name);
        config.device_id = std::stoi(optional(flags, "device", "0"));
        config.metric = zcgvs::metricFromName(optional(flags, "metric", "cosine"));
        config.similarity_backend = optional(flags, "backend", "custom") == "cublas"
                                        ? zcgvs::SimilarityBackend::Cublas
                                        : zcgvs::SimilarityBackend::Custom;
        config.use_cuda_graphs = flagSet(flags, "graphs");
        config.rescore_candidates = std::stoi(optional(flags, "rescore-candidates", "0"));
        config.dense_weight = std::stof(optional(flags, "dense-weight", "0.7"));
        config.default_top_k = std::stoi(optional(flags, "top-k", "10"));
        config.default_candidates = std::stoi(optional(flags, "candidates", "100"));

        const std::string query = require(flags, "query");
        const int repeat = std::stoi(optional(flags, "repeat", "1"));

        zcgvs::Engine engine(config);
        const zcgvs::EngineStats stats = engine.stats();

        std::cout << "corpus  : " << stats.corpus_size << " docs, dim " << stats.dim
                  << ", storage " << stats.storage_dtype << " (" << megabytes(stats.corpus_bytes)
                  << ")\n";
        if (stats.has_rescore_store) {
            std::cout << "rescore : " << stats.rescore_dtype << " ("
                      << megabytes(stats.rescore_bytes) << "), width "
                      << config.rescore_candidates << "\n";
        }
        if (stats.has_lexical_index) {
            std::cout << "lexical : " << stats.lexical_postings << " postings ("
                      << megabytes(stats.lexical_bytes) << ")\n";
        }
        std::cout << "engine  : metric " << zcgvs::metricName(config.metric) << ", backend "
                  << zcgvs::similarityBackendName(config.similarity_backend) << ", graphs "
                  << (config.use_cuda_graphs ? "on" : "off") << ", reranker "
                  << (stats.has_reranker ? "on" : "off") << "\n";

        if (flagSet(flags, "stats")) {
            std::cout << "device  : " << stats.device_name << " (sm_" << stats.compute_major
                      << stats.compute_minor << ", " << megabytes(stats.device_total_memory)
                      << ")\n";
        }

        zcgvs::SearchOptions options = engine.defaultOptions();
        options.hybrid = flagSet(flags, "hybrid") && stats.has_lexical_index;

        zcgvs::SearchResponse response = engine.searchTimed(query, options);
        for (int i = 1; i < repeat; ++i) {
            response = engine.searchTimed(query, options);
        }

        std::cout << "\nquery: " << query << "\n";
        int rank = 1;
        for (const zcgvs::SearchHit& hit : response.hits) {
            std::cout << std::setw(3) << rank << ". [" << hit.doc_id
                      << "] rerank=" << hit.rerank_score << " sim=" << hit.similarity << "\n     "
                      << snippet(hit.text, 120) << "\n";
            ++rank;
        }

        const zcgvs::SearchTiming& timing = response.timing;
        std::cout << "\ntiming (ms): tokenize=" << timing.tokenize_ms
                  << " embed=" << timing.embed_ms << " gpu_search=" << timing.search_gpu_ms
                  << " copy=" << timing.copy_ms << " rerank=" << timing.rerank_ms
                  << " total=" << timing.total_ms
                  << (timing.graph_replay ? "  [graph replay]" : "") << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
