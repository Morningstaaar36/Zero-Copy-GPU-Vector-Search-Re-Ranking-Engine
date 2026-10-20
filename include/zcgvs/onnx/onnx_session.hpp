#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace zcgvs {

struct OnnxInt64Input {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> data;
};

struct OnnxTensorF {
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

class OnnxSession {
public:
    OnnxSession(const std::string& model_path, int device_id);

    bool hasInput(const std::string& name) const;

    const std::vector<std::string>& inputNames() const { return input_names_; }
    const std::vector<std::string>& outputNames() const { return output_names_; }

    OnnxTensorF run(const std::vector<OnnxInt64Input>& inputs, const std::string& output_name);

private:
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace zcgvs
