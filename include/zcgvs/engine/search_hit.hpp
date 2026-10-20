#pragma once

#include <string>

namespace zcgvs {

struct SearchHit {
    std::string doc_id;
    std::string text;
    float similarity = 0.0f;
    float rerank_score = 0.0f;
    int corpus_index = -1;
};

struct Candidate {
    std::string doc_id;
    float score = 0.0f;
    int corpus_index = -1;
};

}  // namespace zcgvs
