#pragma once

#include <vector>

#include "zcgvs/engine/search_hit.hpp"

namespace zcgvs {

// Host stages are wall clock; search_gpu_ms comes from CUDA events bracketing the device
// pipeline, so it excludes launch setup and graph capture.
struct SearchTiming {
    double tokenize_ms = 0.0;
    double embed_ms = 0.0;
    double search_gpu_ms = 0.0;
    double copy_ms = 0.0;
    double rerank_ms = 0.0;
    double total_ms = 0.0;
    int batch = 1;
    bool graph_replay = false;
};

struct SearchResponse {
    std::vector<SearchHit> hits;
    SearchTiming timing;
};

struct BatchSearchResponse {
    std::vector<std::vector<SearchHit>> results;
    SearchTiming timing;
};

}  // namespace zcgvs
