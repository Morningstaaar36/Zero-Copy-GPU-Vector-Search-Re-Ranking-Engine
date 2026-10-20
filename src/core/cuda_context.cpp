#include "zcgvs/core/cuda_context.hpp"

#include <cuda_runtime.h>

#include "zcgvs/core/cuda_check.hpp"

namespace zcgvs {

void setDevice(int device_id) { ZCGVS_CUDA_CHECK(cudaSetDevice(device_id)); }

int currentDevice() {
    int device = 0;
    ZCGVS_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

int deviceCount() {
    int count = 0;
    ZCGVS_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceInfo queryDevice(int device_id) {
    cudaDeviceProp props{};
    ZCGVS_CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    DeviceInfo info;
    info.device_id = device_id;
    info.name = props.name;
    info.total_memory = props.totalGlobalMem;
    info.compute_major = props.major;
    info.compute_minor = props.minor;
    info.multiprocessor_count = props.multiProcessorCount;
    // CUDA 13 dropped cudaDeviceProp::memoryClockRate; the device attribute survives.
#if CUDART_VERSION >= 13000
    int memory_clock_khz = 0;
    ZCGVS_CUDA_CHECK(
        cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device_id));
    info.memory_clock_khz = memory_clock_khz;
#else
    info.memory_clock_khz = props.memoryClockRate;
#endif
    info.memory_bus_width_bits = props.memoryBusWidth;
    return info;
}

}  // namespace zcgvs
