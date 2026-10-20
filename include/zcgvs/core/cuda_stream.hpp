#pragma once

#include <cuda_runtime.h>

#include "zcgvs/core/cuda_check.hpp"

namespace zcgvs {

class CudaStream {
public:
    CudaStream() { ZCGVS_CUDA_CHECK(cudaStreamCreate(&stream_)); }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }

    void synchronize() const { ZCGVS_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace zcgvs
