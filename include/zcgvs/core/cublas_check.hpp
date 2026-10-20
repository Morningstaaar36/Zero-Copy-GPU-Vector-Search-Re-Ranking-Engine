#pragma once

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace zcgvs {

class CublasError : public std::runtime_error {
public:
    CublasError(cublasStatus_t status, const char* expr, const char* file, int line)
        : std::runtime_error(format(status, expr, file, line)), status_(status) {}

    cublasStatus_t status() const noexcept { return status_; }

private:
    static const char* statusName(cublasStatus_t status) {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            default:
                return "CUBLAS_STATUS_UNKNOWN";
        }
    }

    static std::string format(cublasStatus_t status, const char* expr, const char* file, int line) {
        return std::string("cuBLAS error '") + statusName(status) + "' at " + file + ":" +
               std::to_string(line) + " while evaluating `" + expr + "`";
    }

    cublasStatus_t status_;
};

}  // namespace zcgvs

#define ZCGVS_CUBLAS_CHECK(expr)                                                          \
    do {                                                                                  \
        cublasStatus_t zcgvs_cublas_status__ = (expr);                                    \
        if (zcgvs_cublas_status__ != CUBLAS_STATUS_SUCCESS) {                             \
            throw ::zcgvs::CublasError(zcgvs_cublas_status__, #expr, __FILE__, __LINE__); \
        }                                                                                 \
    } while (0)
