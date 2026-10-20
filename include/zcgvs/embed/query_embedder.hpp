#pragma once

#include <string>
#include <vector>

namespace zcgvs {

class QueryEmbedder {
public:
    virtual ~QueryEmbedder() = default;

    virtual int dim() const = 0;

    virtual std::vector<float> embed(const std::string& text) const = 0;
};

}  // namespace zcgvs
