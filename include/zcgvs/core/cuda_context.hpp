#pragma once

#include <cstddef>
#include <string>

namespace zcgvs {

struct DeviceInfo {
    int device_id = 0;
    std::string name;
    std::size_t total_memory = 0;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessor_count = 0;
    int memory_clock_khz = 0;
    int memory_bus_width_bits = 0;

    // Theoretical peak DRAM bandwidth, the denominator of the roofline the brute-force scan
    // is measured against.
    double peakBandwidthGBs() const {
        return 2.0 * static_cast<double>(memory_clock_khz) * 1e3 *
               (static_cast<double>(memory_bus_width_bits) / 8.0) / 1e9;
    }
};

void setDevice(int device_id);

int currentDevice();

int deviceCount();

DeviceInfo queryDevice(int device_id);

}  // namespace zcgvs
