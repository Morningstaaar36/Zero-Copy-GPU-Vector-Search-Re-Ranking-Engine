#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "zcgvs/core/cuda_check.hpp"

namespace zcgvs {

// Page-locked host staging. Device-to-host copies out of pageable memory are staged through
// a driver-owned bounce buffer and serialize against the stream; pinned memory lets the same
// copy run asynchronously and at full PCIe rate.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t count) { allocate(count); }

    ~PinnedBuffer() { release(); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
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
            ZCGVS_CUDA_CHECK(cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocDefault));
            ptr_ = static_cast<T*>(raw);
        }
        count_ = count;
    }

    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace zcgvs
