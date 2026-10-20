#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace zcgvs {

struct Encoding {
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> attention_mask;
    std::vector<std::int64_t> token_type_ids;

    std::size_t length() const { return input_ids.size(); }
};

class Tokenizer {
public:
    static Tokenizer fromVocabFile(const std::string& path, bool lowercase = true);

    Encoding encode(const std::string& text, int max_length) const;

    Encoding encodePair(const std::string& text_a, const std::string& text_b, int max_length) const;

    int clsId() const { return cls_id_; }
    int sepId() const { return sep_id_; }
    int unkId() const { return unk_id_; }
    int padId() const { return pad_id_; }
    std::size_t vocabSize() const { return vocab_.size(); }

private:
    Tokenizer() = default;

    std::vector<std::string> basicTokenize(const std::string& text) const;
    void wordpieceInto(const std::string& token, std::vector<std::int64_t>& out) const;
    std::vector<std::int64_t> tokenizeToIds(const std::string& text) const;
    int lookup(const std::string& token) const;

    std::unordered_map<std::string, int> vocab_;
    bool lowercase_ = true;
    int cls_id_ = 0;
    int sep_id_ = 0;
    int unk_id_ = 0;
    int pad_id_ = 0;
    int max_chars_per_word_ = 200;
};

}  // namespace zcgvs
