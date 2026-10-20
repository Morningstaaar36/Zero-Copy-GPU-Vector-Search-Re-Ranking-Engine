#include "zcgvs/text/tokenizer.hpp"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace zcgvs {

namespace {

constexpr const char* kClsToken = "[CLS]";
constexpr const char* kSepToken = "[SEP]";
constexpr const char* kUnkToken = "[UNK]";
constexpr const char* kPadToken = "[PAD]";

bool isWhitespace(unsigned char c) { return std::isspace(c) != 0; }

bool isControl(unsigned char c) { return c != 0 && std::iscntrl(c) != 0 && std::isspace(c) == 0; }

bool isPunctuation(unsigned char c) { return c < 128 && std::ispunct(c) != 0; }

}  // namespace

Tokenizer Tokenizer::fromVocabFile(const std::string& path, bool lowercase) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("zcgvs: cannot open vocab file '" + path + "'");
    }

    Tokenizer tokenizer;
    tokenizer.lowercase_ = lowercase;

    std::string line;
    int index = 0;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        tokenizer.vocab_.emplace(line, index);
        ++index;
    }

    if (tokenizer.vocab_.empty()) {
        throw std::runtime_error("zcgvs: vocab file '" + path + "' is empty");
    }

    tokenizer.cls_id_ = tokenizer.lookup(kClsToken);
    tokenizer.sep_id_ = tokenizer.lookup(kSepToken);
    tokenizer.unk_id_ = tokenizer.lookup(kUnkToken);
    tokenizer.pad_id_ = tokenizer.lookup(kPadToken);
    if (tokenizer.cls_id_ < 0 || tokenizer.sep_id_ < 0 || tokenizer.unk_id_ < 0 ||
        tokenizer.pad_id_ < 0) {
        throw std::runtime_error("zcgvs: vocab file is missing required special tokens");
    }
    return tokenizer;
}

int Tokenizer::lookup(const std::string& token) const {
    const auto it = vocab_.find(token);
    return it == vocab_.end() ? -1 : it->second;
}

std::vector<std::string> Tokenizer::basicTokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    auto flush = [&]() {
        if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    };

    for (const char raw : text) {
        const unsigned char c = static_cast<unsigned char>(raw);
        if (isControl(c)) {
            continue;
        }
        if (isWhitespace(c)) {
            flush();
        } else if (isPunctuation(c)) {
            flush();
            tokens.emplace_back(1, static_cast<char>(c));
        } else {
            unsigned char folded = c;
            if (lowercase_ && c < 128) {
                folded = static_cast<unsigned char>(std::tolower(c));
            }
            current.push_back(static_cast<char>(folded));
        }
    }
    flush();
    return tokens;
}

void Tokenizer::wordpieceInto(const std::string& token, std::vector<std::int64_t>& out) const {
    if (static_cast<int>(token.size()) > max_chars_per_word_) {
        out.push_back(unk_id_);
        return;
    }

    std::size_t start = 0;
    std::vector<std::int64_t> pieces;
    while (start < token.size()) {
        std::size_t end = token.size();
        int matched = -1;
        std::size_t matched_end = start;
        while (start < end) {
            std::string piece = token.substr(start, end - start);
            if (start > 0) {
                piece = "##" + piece;
            }
            const int id = lookup(piece);
            if (id >= 0) {
                matched = id;
                matched_end = end;
                break;
            }
            --end;
        }
        if (matched < 0) {
            out.push_back(unk_id_);
            return;
        }
        pieces.push_back(matched);
        start = matched_end;
    }

    out.insert(out.end(), pieces.begin(), pieces.end());
}

std::vector<std::int64_t> Tokenizer::tokenizeToIds(const std::string& text) const {
    std::vector<std::int64_t> ids;
    for (const std::string& token : basicTokenize(text)) {
        wordpieceInto(token, ids);
    }
    return ids;
}

Encoding Tokenizer::encode(const std::string& text, int max_length) const {
    std::vector<std::int64_t> content = tokenizeToIds(text);

    const int budget = max_length - 2;
    if (budget > 0 && static_cast<int>(content.size()) > budget) {
        content.resize(static_cast<std::size_t>(budget));
    }

    Encoding encoding;
    encoding.input_ids.reserve(content.size() + 2);
    encoding.input_ids.push_back(cls_id_);
    encoding.input_ids.insert(encoding.input_ids.end(), content.begin(), content.end());
    encoding.input_ids.push_back(sep_id_);

    encoding.attention_mask.assign(encoding.input_ids.size(), 1);
    encoding.token_type_ids.assign(encoding.input_ids.size(), 0);
    return encoding;
}

Encoding Tokenizer::encodePair(const std::string& text_a, const std::string& text_b,
                               int max_length) const {
    std::vector<std::int64_t> a = tokenizeToIds(text_a);
    std::vector<std::int64_t> b = tokenizeToIds(text_b);

    int budget = max_length - 3;
    if (budget < 0) {
        budget = 0;
    }
    while (static_cast<int>(a.size() + b.size()) > budget) {
        if (a.size() > b.size()) {
            a.pop_back();
        } else {
            b.pop_back();
        }
    }

    Encoding encoding;
    const std::size_t total = a.size() + b.size() + 3;
    encoding.input_ids.reserve(total);
    encoding.token_type_ids.reserve(total);

    encoding.input_ids.push_back(cls_id_);
    encoding.token_type_ids.push_back(0);
    for (const std::int64_t id : a) {
        encoding.input_ids.push_back(id);
        encoding.token_type_ids.push_back(0);
    }
    encoding.input_ids.push_back(sep_id_);
    encoding.token_type_ids.push_back(0);
    for (const std::int64_t id : b) {
        encoding.input_ids.push_back(id);
        encoding.token_type_ids.push_back(1);
    }
    encoding.input_ids.push_back(sep_id_);
    encoding.token_type_ids.push_back(1);

    encoding.attention_mask.assign(encoding.input_ids.size(), 1);
    return encoding;
}

}  // namespace zcgvs
