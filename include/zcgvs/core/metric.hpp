#pragma once

#include <cstdint>
#include <string>

namespace zcgvs {

enum class Metric : std::int32_t {
    Cosine = 0,
    DotProduct = 1,
    L2 = 2,
};

inline const char* metricName(Metric metric) {
    switch (metric) {
        case Metric::DotProduct:
            return "dot";
        case Metric::L2:
            return "l2";
        case Metric::Cosine:
        default:
            return "cosine";
    }
}

inline Metric metricFromName(const std::string& name) {
    if (name == "dot" || name == "dotproduct" || name == "ip") {
        return Metric::DotProduct;
    }
    if (name == "l2" || name == "euclidean") {
        return Metric::L2;
    }
    return Metric::Cosine;
}

}  // namespace zcgvs
