#pragma once

#include <string>
#include <vector>

namespace zcgvs {

class Reranker {
public:
    virtual ~Reranker() = default;

    virtual std::vector<float> score(const std::string& query,
                                     const std::vector<std::string>& passages) const = 0;
};

}  // namespace zcgvs
