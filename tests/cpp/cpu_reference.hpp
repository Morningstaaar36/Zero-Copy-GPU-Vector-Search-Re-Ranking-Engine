#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace zcgvs {
namespace testing {

inline std::vector<float> cpuCosineSimilarity(const std::vector<float>& corpus, int doc_count,
                                              int dim, const std::vector<float>& query) {
    std::vector<float> scores(static_cast<std::size_t>(doc_count), 0.0f);
    for (int i = 0; i < doc_count; ++i) {
        float dot = 0.0f;
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(dim);
        for (int j = 0; j < dim; ++j) {
            dot += corpus[base + static_cast<std::size_t>(j)] * query[static_cast<std::size_t>(j)];
        }
        scores[static_cast<std::size_t>(i)] = dot;
    }
    return scores;
}

inline std::vector<int> cpuTopKIndices(const std::vector<float>& scores, int k) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
    });
    order.resize(static_cast<std::size_t>(k));
    return order;
}

inline void l2Normalize(std::vector<float>& vector, int offset, int dim) {
    float norm = 0.0f;
    for (int j = 0; j < dim; ++j) {
        const float value = vector[static_cast<std::size_t>(offset + j)];
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm <= 0.0f) {
        return;
    }
    for (int j = 0; j < dim; ++j) {
        vector[static_cast<std::size_t>(offset + j)] /= norm;
    }
}

// Mirrors cuda::launchConvertF32ToI8 exactly: the same reciprocal, rounding mode and clamp,
// so the reference and the kernel agree bit for bit rather than merely closely.
inline std::vector<float> cpuQuantizeDequantizeInt8(const std::vector<float>& values, float scale) {
    const float inverse_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    std::vector<float> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        float quantized = std::nearbyint(values[i] * inverse_scale);
        quantized = std::min(127.0f, std::max(-127.0f, quantized));
        result[i] = quantized * scale;
    }
    return result;
}

// Mirrors cuda::launchPackBinary: bit b of word w carries dimension w * 32 + b.
inline std::vector<std::uint32_t> cpuPackBinary(const std::vector<float>& matrix, int rows,
                                                int dim) {
    const int words = (dim + 31) / 32;
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(rows) * words, 0u);
    for (int row = 0; row < rows; ++row) {
        for (int d = 0; d < dim; ++d) {
            if (matrix[static_cast<std::size_t>(row) * dim + d] >= 0.0f) {
                const std::size_t index = static_cast<std::size_t>(row) * words + (d / 32);
                packed[index] |= (1u << (d % 32));
            }
        }
    }
    return packed;
}

inline int cpuHamming(const std::uint32_t* a, const std::uint32_t* b, int words) {
    int distance = 0;
    for (int w = 0; w < words; ++w) {
        std::uint32_t x = a[w] ^ b[w];
        while (x != 0u) {
            x &= (x - 1u);
            ++distance;
        }
    }
    return distance;
}

}  // namespace testing
}  // namespace zcgvs
