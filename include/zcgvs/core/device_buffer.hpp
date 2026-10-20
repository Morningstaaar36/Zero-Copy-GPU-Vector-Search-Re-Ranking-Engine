#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "zcgvs/core/cuda_check.hpp"

namespace zcgvs {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        if (count > 0) {
            void* raw = nullptr;
            ZCGVS_CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
            ptr_ = static_cast<T*>(raw);
        }
        count_ = count;
    }

    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    void copyFromHost(const T* host_src, std::size_t count, cudaStream_t stream = nullptr) {
        ZCGVS_CUDA_CHECK(cudaMemcpyAsync(ptr_, host_src, count * sizeof(T),
                                         cudaMemcpyHostToDevice, stream));
    }

    void copyToHost(T* host_dst, std::size_t count, cudaStream_t stream = nullptr) const {
        ZCGVS_CUDA_CHECK(cudaMemcpyAsync(host_dst, ptr_, count * sizeof(T),
                                         cudaMemcpyDeviceToHost, stream));
    }

    void fillZero(cudaStream_t stream = nullptr) {
        ZCGVS_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), stream));
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    bool empty() const noexcept { return count_ == 0; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace zcgvs
