#include "zcgvs/data/embedding_store.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/cuda/convert.cuh"
#include "zcgvs/data/binary_format.hpp"

namespace zcgvs {

namespace {

ScalarType scalarTypeFromCode(std::int32_t code) {
    switch (code) {
        case kDtypeFloat32:
            return ScalarType::Float32;
        case kDtypeFloat16:
            return ScalarType::Float16;
        case kDtypeInt8:
            return ScalarType::Int8;
        case kDtypeBinary:
            return ScalarType::Binary;
        default:
            throw std::runtime_error("zcgvs: unsupported embedding dtype code " +
                                     std::to_string(code));
    }
}

}  // namespace

EmbeddingStore EmbeddingStore::loadFromFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("zcgvs: cannot open embedding file '" + path + "'");
    }

    BinaryReader reader(stream);
    reader.expectMagic(kEmbeddingMagic);

    const std::uint32_t version = reader.readScalar<std::uint32_t>();
    if (version == 0 || version > kEmbeddingVersion) {
        throw std::runtime_error("zcgvs: unsupported embedding file version " +
                                 std::to_string(version));
    }

    const std::uint32_t flags = reader.readScalar<std::uint32_t>();
    const std::int64_t doc_count = reader.readScalar<std::int64_t>();
    const std::int32_t dim = reader.readScalar<std::int32_t>();
    const std::int32_t dtype_code = reader.readScalar<std::int32_t>();

    if (doc_count <= 0 || dim <= 0) {
        throw std::runtime_error("zcgvs: embedding file reports an empty matrix");
    }

    EmbeddingStore store;
    store.dim_ = static_cast<int>(dim);
    store.dtype_ = scalarTypeFromCode(dtype_code);
    store.normalized_ = (flags & kFlagNormalized) != 0;

    std::size_t stride = rowBytes(store.dtype_, store.dim_);
    if (version >= 2) {
        store.scale_ = reader.readScalar<float>();
        const std::int32_t recorded_stride = reader.readScalar<std::int32_t>();
        if (recorded_stride <= 0 || static_cast<std::size_t>(recorded_stride) != stride) {
            throw std::runtime_error("zcgvs: embedding row stride does not match dtype and dim");
        }
        stride = static_cast<std::size_t>(recorded_stride);
    } else if (isQuantized(store.dtype_)) {
        throw std::runtime_error("zcgvs: quantized embeddings require file version 2");
    }

    const std::size_t total_bytes = static_cast<std::size_t>(doc_count) * stride;
    std::vector<std::uint8_t> host(total_bytes);
    reader.readBytes(reinterpret_cast<char*>(host.data()), total_bytes);

    store.device_.allocate(total_bytes);
    ZCGVS_CUDA_CHECK(
        cudaMemcpy(store.device_.data(), host.data(), total_bytes, cudaMemcpyHostToDevice));
    store.doc_count_ = doc_count;
    store.capacity_ = doc_count;
    return store;
}

EmbeddingStore EmbeddingStore::fromHostMatrix(const float* host_float32, std::int64_t doc_count,
                                              int dim, ScalarType dtype, float scale,
                                              bool normalized) {
    if (host_float32 == nullptr || doc_count <= 0 || dim <= 0) {
        throw std::invalid_argument("zcgvs: fromHostMatrix requires a non-empty matrix");
    }

    EmbeddingStore store;
    store.dim_ = dim;
    store.dtype_ = dtype;
    store.normalized_ = normalized;
    store.scale_ = (dtype == ScalarType::Int8 && scale > 0.0f) ? scale : 1.0f;
    store.doc_count_ = doc_count;
    store.capacity_ = doc_count;
    store.device_.allocate(static_cast<std::size_t>(doc_count) * store.rowStride());

    const std::size_t elements =
        static_cast<std::size_t>(doc_count) * static_cast<std::size_t>(dim);

    if (dtype == ScalarType::Float32) {
        ZCGVS_CUDA_CHECK(cudaMemcpy(store.device_.data(), host_float32, elements * sizeof(float),
                                    cudaMemcpyHostToDevice));
        return store;
    }

    DeviceBuffer<float> staging(elements);
    ZCGVS_CUDA_CHECK(cudaMemcpy(staging.data(), host_float32, elements * sizeof(float),
                                cudaMemcpyHostToDevice));
    switch (dtype) {
        case ScalarType::Float16:
            cuda::launchConvertF32ToF16(staging.data(), store.device_.data(), elements, nullptr);
            break;
        case ScalarType::Int8:
            cuda::launchConvertF32ToI8(staging.data(), store.device_.data(), elements,
                                       store.scale_, nullptr);
            break;
        case ScalarType::Binary:
            cuda::launchPackBinary(staging.data(), store.device_.data(),
                                   static_cast<int>(doc_count), dim, nullptr);
            break;
        default:
            throw std::invalid_argument("zcgvs: unsupported storage dtype");
    }
    ZCGVS_CUDA_CHECK(cudaDeviceSynchronize());
    return store;
}

void EmbeddingStore::ensureCapacity(std::int64_t required_docs, cudaStream_t stream) {
    if (capacity_ >= required_docs) {
        return;
    }
    const std::int64_t grown = std::max(required_docs, capacity_ * 2);
    DeviceBuffer<std::uint8_t> resized(static_cast<std::size_t>(grown) * rowStride());
    if (doc_count_ > 0) {
        ZCGVS_CUDA_CHECK(cudaMemcpyAsync(resized.data(), device_.data(),
                                         static_cast<std::size_t>(doc_count_) * rowStride(),
                                         cudaMemcpyDeviceToDevice, stream));
    }
    ZCGVS_CUDA_CHECK(cudaStreamSynchronize(stream));
    device_ = std::move(resized);
    capacity_ = grown;
}

void EmbeddingStore::append(const float* host_float32, std::int64_t add_count,
                            cudaStream_t stream) {
    if (add_count <= 0) {
        return;
    }
    if (dim_ <= 0) {
        throw std::runtime_error("zcgvs: append requires an initialized store");
    }

    ensureCapacity(doc_count_ + add_count, stream);

    const std::size_t element_count =
        static_cast<std::size_t>(add_count) * static_cast<std::size_t>(dim_);
    std::uint8_t* destination = device_.data() + static_cast<std::size_t>(doc_count_) * rowStride();

    if (dtype_ == ScalarType::Float32) {
        ZCGVS_CUDA_CHECK(cudaMemcpyAsync(destination, host_float32, element_count * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
    } else {
        DeviceBuffer<float> staging(element_count);
        ZCGVS_CUDA_CHECK(cudaMemcpyAsync(staging.data(), host_float32,
                                         element_count * sizeof(float), cudaMemcpyHostToDevice,
                                         stream));
        switch (dtype_) {
            case ScalarType::Float16:
                cuda::launchConvertF32ToF16(staging.data(), destination, element_count, stream);
                break;
            case ScalarType::Int8:
                cuda::launchConvertF32ToI8(staging.data(), destination, element_count, scale_,
                                           stream);
                break;
            case ScalarType::Binary:
                cuda::launchPackBinary(staging.data(), destination, static_cast<int>(add_count),
                                       dim_, stream);
                break;
            default:
                throw std::runtime_error("zcgvs: append does not support this storage dtype");
        }
        ZCGVS_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ZCGVS_CUDA_CHECK(cudaStreamSynchronize(stream));
    doc_count_ += add_count;
}

}  // namespace zcgvs
