#include "zcgvs/onnx/onnx_session.hpp"

#include <algorithm>
#include <stdexcept>

namespace zcgvs {

namespace {

Ort::Env& sharedEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zcgvs");
    return env;
}

Ort::SessionOptions makeSessionOptions(int device_id) {
    // The order in which the Session constructor's arguments are evaluated is unspecified, so
    // the Env has to be forced into existence here: ONNX Runtime registers its default logger
    // when the Env is created, and appending a provider below logs through it.
    sharedEnv();

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id;
    options.AppendExecutionProvider_CUDA(cuda_options);
    return options;
}

}  // namespace

OnnxSession::OnnxSession(const std::string& model_path, int device_id)
    : session_(sharedEnv(), model_path.c_str(), makeSessionOptions(device_id)) {
    const std::size_t input_count = session_.GetInputCount();
    input_names_.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr name = session_.GetInputNameAllocated(i, allocator_);
        input_names_.emplace_back(name.get());
    }

    const std::size_t output_count = session_.GetOutputCount();
    output_names_.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(name.get());
    }
}

bool OnnxSession::hasInput(const std::string& name) const {
    return std::find(input_names_.begin(), input_names_.end(), name) != input_names_.end();
}

OnnxTensorF OnnxSession::run(const std::vector<OnnxInt64Input>& inputs,
                             const std::string& output_name) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_values;
    std::vector<const char*> input_name_ptrs;
    input_values.reserve(inputs.size());
    input_name_ptrs.reserve(inputs.size());

    for (const OnnxInt64Input& input : inputs) {
        if (!hasInput(input.name)) {
            continue;
        }
        input_values.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info, const_cast<std::int64_t*>(input.data.data()), input.data.size(),
            input.shape.data(), input.shape.size()));
        input_name_ptrs.push_back(input.name.c_str());
    }

    if (input_values.empty()) {
        throw std::runtime_error("zcgvs onnx: no supplied input matched the model inputs");
    }

    const char* output_name_ptr = output_name.c_str();
    std::vector<Ort::Value> outputs =
        session_.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_values.data(),
                     input_values.size(), &output_name_ptr, 1);

    Ort::Value& output = outputs.front();
    Ort::TensorTypeAndShapeInfo info = output.GetTensorTypeAndShapeInfo();

    OnnxTensorF result;
    result.shape = info.GetShape();
    const std::size_t count = info.GetElementCount();
    const float* raw = output.GetTensorData<float>();
    result.data.assign(raw, raw + count);
    return result;
}

}  // namespace zcgvs
