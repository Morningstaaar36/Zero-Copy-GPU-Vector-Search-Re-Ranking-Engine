#pragma once

#include <cstddef>
#include <vector>

namespace zcgvs {

// Flattened [batch, k] selection copied back from the device. An index of -1 marks a padding
// slot, which occurs when the corpus holds fewer than k usable documents.
struct HostTopK {
    std::vector<int> indices;
    std::vector<float> scores;
    int batch = 0;
    int k = 0;

    std::size_t offset(int query) const {
        return static_cast<std::size_t>(query) * static_cast<std::size_t>(k);
    }
};

}  // namespace zcgvs
