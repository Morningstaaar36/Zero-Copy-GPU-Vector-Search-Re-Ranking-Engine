#pragma once

// Nsight Systems range annotations. Compiled out unless ZCGVS_WITH_NVTX is defined
// (CMake option ZCGVS_ENABLE_NVTX), so the default build carries no profiler dependency.

#ifdef ZCGVS_WITH_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace zcgvs {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
#ifdef ZCGVS_WITH_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange() {
#ifdef ZCGVS_WITH_NVTX
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace zcgvs

#define ZCGVS_NVTX_CONCAT_INNER(a, b) a##b
#define ZCGVS_NVTX_CONCAT(a, b) ZCGVS_NVTX_CONCAT_INNER(a, b)
#define ZCGVS_NVTX_RANGE(name) \
    ::zcgvs::NvtxRange ZCGVS_NVTX_CONCAT(zcgvs_nvtx_range_, __LINE__)(name)
