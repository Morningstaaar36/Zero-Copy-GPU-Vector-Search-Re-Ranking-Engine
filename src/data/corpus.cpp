#include "zcgvs/data/corpus.hpp"

#include <fstream>
#include <stdexcept>

#include "zcgvs/data/binary_format.hpp"

namespace zcgvs {

namespace {

void loadMetadata(const std::string& path,
                  std::int64_t expected_count,
                  std::vector<std::string>& doc_ids,
                  std::vector<std::string>& texts) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("zcgvs: cannot open metadata file '" + path + "'");
    }

    BinaryReader reader(stream);
    reader.expectMagic(kMetadataMagic);

    const std::uint32_t version = reader.readScalar<std::uint32_t>();
    if (version != kMetadataVersion) {
        throw std::runtime_error("zcgvs: unsupported metadata version " + std::to_string(version));
    }

    const std::int64_t doc_count = reader.readScalar<std::int64_t>();
    if (doc_count != expected_count) {
        throw std::runtime_error("zcgvs: metadata doc count does not match embeddings");
    }

    doc_ids.reserve(static_cast<std::size_t>(doc_count));
    texts.reserve(static_cast<std::size_t>(doc_count));
    for (std::int64_t i = 0; i < doc_count; ++i) {
        const std::uint32_t id_len = reader.readScalar<std::uint32_t>();
        doc_ids.push_back(reader.readString(id_len));
        const std::uint32_t text_len = reader.readScalar<std::uint32_t>();
        texts.push_back(reader.readString(text_len));
    }
}

}  // namespace

Corpus Corpus::load(const std::string& embeddings_path, const std::string& metadata_path) {
    Corpus corpus;
    corpus.embeddings_ = EmbeddingStore::loadFromFile(embeddings_path);
    loadMetadata(metadata_path, corpus.embeddings_.docCount(), corpus.doc_ids_, corpus.texts_);
    return corpus;
}

void Corpus::addDocuments(const std::vector<std::string>& ids,
                          const std::vector<std::string>& texts,
                          const float* host_float32,
                          std::int64_t add_count,
                          cudaStream_t stream) {
    if (ids.size() != texts.size() ||
        static_cast<std::int64_t>(ids.size()) != add_count) {
        throw std::runtime_error("zcgvs: addDocuments received mismatched id/text/vector counts");
    }
    if (add_count <= 0) {
        return;
    }

    embeddings_.append(host_float32, add_count, stream);
    doc_ids_.insert(doc_ids_.end(), ids.begin(), ids.end());
    texts_.insert(texts_.end(), texts.begin(), texts.end());
}

}  // namespace zcgvs
