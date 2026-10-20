#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/core/scalar_type.hpp"

namespace zcgvs {

// The VRAM-resident corpus matrix. Rows are stored at the file's precision (float32,
// float16, int8 or packed binary) and are never re-uploaded per query; `append` grows the
// device buffer geometrically and quantizes new rows on the device.
class EmbeddingStore {
public:
    EmbeddingStore() = default;

    static EmbeddingStore loadFromFile(const std::string& path);

    // Builds a store directly from a host float32 matrix, quantizing on the device. Used by
    // the kernel benchmarks and the correctness harnesses, which have no artifact files.
    static EmbeddingStore fromHostMatrix(const float* host_float32,
                                         std::int64_t doc_count,
                                         int dim,
                                         ScalarType dtype,
                                         float scale,
                                         bool normalized);

    const void* deviceData() const { return device_.data(); }
    std::int64_t docCount() const { return doc_count_; }
    int dim() const { return dim_; }
    bool normalized() const { return normalized_; }
    ScalarType dtype() const { return dtype_; }
    float scale() const { return scale_; }
    std::int64_t capacity() const { return capacity_; }
    std::size_t deviceBytes() const { return device_.bytes(); }
    std::size_t rowStride() const { return rowBytes(dtype_, dim_); }

    void append(const float* host_float32, std::int64_t add_count, cudaStream_t stream);

private:
    void ensureCapacity(std::int64_t required_docs, cudaStream_t stream);

    DeviceBuffer<std::uint8_t> device_;
    std::int64_t doc_count_ = 0;
    std::int64_t capacity_ = 0;
    int dim_ = 0;
    bool normalized_ = false;
    ScalarType dtype_ = ScalarType::Float32;
    float scale_ = 1.0f;
};

}  // namespace zcgvs
