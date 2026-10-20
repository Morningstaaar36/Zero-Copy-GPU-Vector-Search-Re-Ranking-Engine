#pragma once

#include <vector>

#include "zcgvs/core/metric.hpp"
#include "zcgvs/search/search_params.hpp"

namespace zcgvs {

// Per-request knobs. Fields left at their sentinel value fall back to EngineConfig; build one
// with Engine::defaultOptions() to start from the configured defaults instead.
struct SearchOptions {
    int top_k = 0;         // <= 0 : EngineConfig::default_top_k
    int candidates = 0;    // <= 0 : EngineConfig::default_candidates
    bool use_config_metric = true;
    Metric metric = Metric::Cosine;

    // Corpus indices the search is restricted to. Empty means no restriction.
    std::vector<int> allow;

    int rescore_candidates = -1;  // < 0 : EngineConfig::rescore_candidates
    float dense_weight = -1.0f;   // < 0 : EngineConfig::dense_weight

    bool hybrid = false;
    bool rerank = true;

    bool use_config_backend = true;
    SimilarityBackend backend = SimilarityBackend::Custom;

    bool use_config_graphs = true;
    bool use_graph = false;
};

}  // namespace zcgvs
