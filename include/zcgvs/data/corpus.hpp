#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "zcgvs/data/embedding_store.hpp"

namespace zcgvs {

class Corpus {
public:
    static Corpus load(const std::string& embeddings_path, const std::string& metadata_path);

    const EmbeddingStore& embeddings() const { return embeddings_; }

    std::int64_t size() const { return static_cast<std::int64_t>(doc_ids_.size()); }
    int dim() const { return embeddings_.dim(); }

    const std::string& docId(int index) const { return doc_ids_.at(static_cast<std::size_t>(index)); }
    const std::string& text(int index) const { return texts_.at(static_cast<std::size_t>(index)); }

    void addDocuments(const std::vector<std::string>& ids,
                      const std::vector<std::string>& texts,
                      const float* host_float32,
                      std::int64_t add_count,
                      cudaStream_t stream);

private:
    EmbeddingStore embeddings_;
    std::vector<std::string> doc_ids_;
    std::vector<std::string> texts_;
};

}  // namespace zcgvs
