#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace zcgvs {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char* expr, const char* file, int line)
        : std::runtime_error(format(code, expr, file, line)), code_(code) {}

    cudaError_t code() const noexcept { return code_; }

private:
    static std::string format(cudaError_t code, const char* expr, const char* file, int line) {
        return std::string("CUDA error '") + cudaGetErrorName(code) + "' (" +
               cudaGetErrorString(code) + ") at " + file + ":" + std::to_string(line) +
               " while evaluating `" + expr + "`";
    }

    cudaError_t code_;
};

}  // namespace zcgvs

#define ZCGVS_CUDA_CHECK(expr)                                                       \
    do {                                                                             \
        cudaError_t zcgvs_cuda_status__ = (expr);                                    \
        if (zcgvs_cuda_status__ != cudaSuccess) {                                    \
            throw ::zcgvs::CudaError(zcgvs_cuda_status__, #expr, __FILE__, __LINE__); \
        }                                                                            \
    } while (0)

#define ZCGVS_CUDA_CHECK_LAST() ZCGVS_CUDA_CHECK(cudaGetLastError())
