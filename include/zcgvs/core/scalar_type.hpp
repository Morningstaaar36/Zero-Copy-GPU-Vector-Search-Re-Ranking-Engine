#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace zcgvs {

// Storage precision of the VRAM-resident corpus matrix. Float32 and Float16 keep one
// element per dimension; Int8 stores one quantized byte per dimension plus a matrix-wide
// dequantization scale; Binary packs one sign bit per dimension into uint32 words.
enum class ScalarType : std::int32_t {
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    Binary = 3,
};

inline bool isQuantized(ScalarType type) {
    return type == ScalarType::Int8 || type == ScalarType::Binary;
}

inline std::size_t binaryWords(int dim) {
    return static_cast<std::size_t>((dim + 31) / 32);
}

// Bytes occupied by a single document row. Binary rows are padded up to a whole word.
inline std::size_t rowBytes(ScalarType type, int dim) {
    switch (type) {
        case ScalarType::Float16:
            return static_cast<std::size_t>(dim) * 2;
        case ScalarType::Int8:
            return static_cast<std::size_t>(dim);
        case ScalarType::Binary:
            return binaryWords(dim) * sizeof(std::uint32_t);
        case ScalarType::Float32:
        default:
            return static_cast<std::size_t>(dim) * 4;
    }
}

inline const char* scalarName(ScalarType type) {
    switch (type) {
        case ScalarType::Float16:
            return "float16";
        case ScalarType::Int8:
            return "int8";
        case ScalarType::Binary:
            return "binary";
        case ScalarType::Float32:
        default:
            return "float32";
    }
}

inline ScalarType scalarTypeFromName(const std::string& name) {
    if (name == "float16" || name == "fp16" || name == "half") {
        return ScalarType::Float16;
    }
    if (name == "int8" || name == "i8") {
        return ScalarType::Int8;
    }
    if (name == "binary" || name == "bin" || name == "b1") {
        return ScalarType::Binary;
    }
    return ScalarType::Float32;
}

}  // namespace zcgvs
