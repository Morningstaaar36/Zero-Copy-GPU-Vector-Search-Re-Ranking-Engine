#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace zcgvs {

// A snapshot of what the engine is holding in VRAM and which optional stages are wired up.
struct EngineStats {
    std::int64_t corpus_size = 0;
    int dim = 0;

    std::string storage_dtype;
    std::size_t corpus_bytes = 0;

    bool has_rescore_store = false;
    std::string rescore_dtype;
    std::size_t rescore_bytes = 0;

    bool has_lexical_index = false;
    std::int64_t lexical_postings = 0;
    std::size_t lexical_bytes = 0;

    bool has_reranker = false;

    std::uint64_t graph_captures = 0;
    std::uint64_t graph_launches = 0;

    std::string device_name;
    std::size_t device_total_memory = 0;
    int compute_major = 0;
    int compute_minor = 0;
};

}  // namespace zcgvs
