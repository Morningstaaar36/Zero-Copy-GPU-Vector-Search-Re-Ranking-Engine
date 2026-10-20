#pragma once

#include <memory>

namespace zcgvs {

struct EngineConfig;
class Reranker;
class Tokenizer;

std::unique_ptr<Reranker> makeReranker(const EngineConfig& config, const Tokenizer& tokenizer);

}  // namespace zcgvs
