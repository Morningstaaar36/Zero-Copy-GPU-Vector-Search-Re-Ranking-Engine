// Kernel-level benchmark harness. Runs entirely on synthetic data, so it needs no models,
// no corpus and no network access — it is the fastest way to prove the CUDA side works and
// to produce the numbers quoted in docs/BENCHMARKS.md.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "zcgvs/core/cuda_check.hpp"
#include "zcgvs/core/cuda_context.hpp"
#include "zcgvs/core/cuda_stream.hpp"
#include "zcgvs/core/device_buffer.hpp"
#include "zcgvs/cuda/convert.cuh"
#include "zcgvs/cuda/cublas_similarity.hpp"
#include "zcgvs/cuda/similarity.cuh"
#include "zcgvs/cuda/topk.cuh"
#include "zcgvs/data/embedding_store.hpp"
#include "zcgvs/search/searcher.hpp"

namespace {

// The synthetic corpus is materialized in host memory first, so the default is sized to be
// comfortably larger than any L2 (and therefore a real bandwidth measurement) without
// demanding gigabytes. Raise --docs to push the scan harder.
struct Args {
    int doc_count = 200000;
    int dim = 384;
    int k = 100;
    int warmup = 5;
    int iterations = 50;
    int device = 0;
    bool markdown = false;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--docs") {
            args.doc_count = std::stoi(value);
        } else if (key == "--dim") {
            args.dim = std::stoi(value);
        } else if (key == "--k") {
            args.k = std::stoi(value);
        } else if (key == "--warmup") {
            args.warmup = std::stoi(value);
        } else if (key == "--iterations") {
            args.iterations = std::stoi(value);
        } else if (key == "--device") {
            args.device = std::stoi(value);
        } else if (key == "--markdown") {
            args.markdown = value != "0";
        }
    }
    return args;
}

std::vector<float> randomNormalizedMatrix(std::int64_t rows, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    std::vector<float> matrix(static_cast<std::size_t>(rows) * dim);
    for (std::int64_t row = 0; row < rows; ++row) {
        float norm = 0.0f;
        const std::size_t base = static_cast<std::size_t>(row) * dim;
        for (int j = 0; j < dim; ++j) {
            const float value = distribution(rng);
            matrix[base + j] = value;
            norm += value * value;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (int j = 0; j < dim; ++j) {
                matrix[base + j] /= norm;
            }
        }
    }
    return matrix;
}

// Median-of-iterations wall time of a device operation, measured with CUDA events.
template <typename Fn>
double timeMs(const zcgvs::CudaStream& stream, int warmup, int iterations, Fn&& body) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ZCGVS_CUDA_CHECK(cudaEventCreate(&start));
    ZCGVS_CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        body();
    }
    stream.synchronize();

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        ZCGVS_CUDA_CHECK(cudaEventRecord(start, stream.get()));
        body();
        ZCGVS_CUDA_CHECK(cudaEventRecord(stop, stream.get()));
        ZCGVS_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        ZCGVS_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(static_cast<double>(elapsed));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void row(bool markdown, const char* label, const std::string& detail, double ms,
         const std::string& derived) {
    if (markdown) {
        std::printf("| %s | %s | %.3f | %s |\n", label, detail.c_str(), ms, derived.c_str());
    } else {
        std::printf("  %-22s %-22s %9.3f ms   %s\n", label, detail.c_str(), ms, derived.c_str());
    }
}

std::string gigabytesPerSecond(std::size_t bytes, double ms, double peak_gbs) {
    const double gbs = (static_cast<double>(bytes) / 1e9) / (ms / 1e3);
    char buffer[96];
    if (peak_gbs > 0.0) {
        std::snprintf(buffer, sizeof(buffer), "%.1f GB/s (%.0f%% of peak)", gbs,
                      100.0 * gbs / peak_gbs);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.1f GB/s", gbs);
    }
    return buffer;
}

std::string queriesPerSecond(int batch, double ms) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.0f queries/s", static_cast<double>(batch) / (ms / 1e3));
    return buffer;
}

void benchmarkStorage(const Args& args, const zcgvs::CudaStream& stream,
                      const std::vector<float>& corpus, const std::vector<float>& query,
                      double peak_gbs, bool markdown) {
    const zcgvs::ScalarType dtypes[] = {zcgvs::ScalarType::Float32, zcgvs::ScalarType::Float16,
                                        zcgvs::ScalarType::Int8, zcgvs::ScalarType::Binary};

    zcgvs::DeviceBuffer<float> d_query(static_cast<std::size_t>(args.dim));
    d_query.copyFromHost(query.data(), static_cast<std::size_t>(args.dim), stream.get());
    stream.synchronize();

    zcgvs::DeviceBuffer<float> d_scores(static_cast<std::size_t>(args.doc_count));
    zcgvs::DeviceBuffer<unsigned int> d_query_bits(zcgvs::binaryWords(args.dim));

    for (const zcgvs::ScalarType dtype : dtypes) {
        const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
            corpus.data(), args.doc_count, args.dim, dtype, 1.0f, true);

        const void* query_data = d_query.data();
        if (dtype == zcgvs::ScalarType::Binary) {
            zcgvs::cuda::launchPackBinary(d_query.data(), d_query_bits.data(), 1, args.dim,
                                          stream.get());
            stream.synchronize();
            query_data = d_query_bits.data();
        }

        const double ms = timeMs(stream, args.warmup, args.iterations, [&]() {
            zcgvs::cuda::launchSimilarity(store.deviceData(), dtype, store.scale(), args.doc_count,
                                          args.dim, query_data, 1, nullptr,
                                          zcgvs::Metric::Cosine, d_scores.data(), stream.get());
        });

        const std::size_t bytes = static_cast<std::size_t>(args.doc_count) * store.rowStride();
        row(markdown, "similarity scan", zcgvs::scalarName(dtype), ms,
            gigabytesPerSecond(bytes, ms, peak_gbs));
    }
}

void benchmarkCublas(const Args& args, const zcgvs::CudaStream& stream,
                     const std::vector<float>& corpus, const std::vector<float>& query,
                     double peak_gbs, bool markdown) {
    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), args.doc_count, args.dim, zcgvs::ScalarType::Float32, 1.0f, true);

    zcgvs::DeviceBuffer<float> d_query(static_cast<std::size_t>(args.dim));
    d_query.copyFromHost(query.data(), static_cast<std::size_t>(args.dim), stream.get());
    stream.synchronize();

    zcgvs::DeviceBuffer<float> d_scores(static_cast<std::size_t>(args.doc_count));
    zcgvs::CublasSimilarity cublas;

    const double ms = timeMs(stream, args.warmup, args.iterations, [&]() {
        cublas.run(static_cast<const float*>(store.deviceData()), args.doc_count, args.dim,
                   d_query.data(), 1, d_scores.data(), stream.get());
    });

    const std::size_t bytes = static_cast<std::size_t>(args.doc_count) * store.rowStride();
    row(markdown, "cublas sgemv", "float32", ms, gigabytesPerSecond(bytes, ms, peak_gbs));
}

void benchmarkTopK(const Args& args, const zcgvs::CudaStream& stream, bool markdown) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> scores(static_cast<std::size_t>(args.doc_count));
    for (float& value : scores) {
        value = distribution(rng);
    }

    zcgvs::DeviceBuffer<float> d_scores(scores.size());
    d_scores.copyFromHost(scores.data(), scores.size(), stream.get());
    stream.synchronize();

    const int widths[] = {10, 100, 1000};
    for (const int k : widths) {
        if (k > args.doc_count || k > zcgvs::cuda::topKMaxK()) {
            continue;
        }
        const std::size_t scratch =
            static_cast<std::size_t>(zcgvs::cuda::topKScratchElements(args.doc_count, k));
        zcgvs::DeviceBuffer<float> values_a(scratch);
        zcgvs::DeviceBuffer<int> indices_a(scratch);
        zcgvs::DeviceBuffer<float> values_b(scratch);
        zcgvs::DeviceBuffer<int> indices_b(scratch);
        zcgvs::DeviceBuffer<int> out_indices(static_cast<std::size_t>(k));
        zcgvs::DeviceBuffer<float> out_values(static_cast<std::size_t>(k));

        const double ms = timeMs(stream, args.warmup, args.iterations, [&]() {
            zcgvs::cuda::launchTopK(d_scores.data(), args.doc_count, k, 1, out_indices.data(),
                                    out_values.data(), values_a.data(), indices_a.data(),
                                    values_b.data(), indices_b.data(), stream.get());
        });

        const std::size_t bytes = scores.size() * sizeof(float);
        row(markdown, "top-k selection", "k=" + std::to_string(k), ms,
            gigabytesPerSecond(bytes, ms, 0.0));
    }
}

void benchmarkBatchAndGraphs(const Args& args, const zcgvs::CudaStream& stream,
                             const std::vector<float>& corpus, bool markdown) {
    const zcgvs::EmbeddingStore store = zcgvs::EmbeddingStore::fromHostMatrix(
        corpus.data(), args.doc_count, args.dim, zcgvs::ScalarType::Float16, 1.0f, true);

    zcgvs::Searcher searcher(store);

    const int batches[] = {1, 8, 32};
    for (const int batch : batches) {
        const std::vector<float> queries = randomNormalizedMatrix(batch, args.dim, 99u);

        for (const bool graphs : {false, true}) {
            zcgvs::SearchParams params;
            params.k = args.k;
            params.metric = zcgvs::Metric::Cosine;
            params.use_graph = graphs;

            const double ms = timeMs(stream, args.warmup, args.iterations, [&]() {
                searcher.search(queries.data(), args.dim, batch, params, stream);
            });

            row(markdown, graphs ? "pipeline (graph)" : "pipeline (direct)",
                "batch=" + std::to_string(batch), ms, queriesPerSecond(batch, ms));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        zcgvs::setDevice(args.device);

        const zcgvs::DeviceInfo info = zcgvs::queryDevice(args.device);
        const double peak_gbs = info.peakBandwidthGBs();

        std::printf("device      : %s (sm_%d%d, %d SMs)\n", info.name.c_str(), info.compute_major,
                    info.compute_minor, info.multiprocessor_count);
        std::printf("peak DRAM   : %.0f GB/s (theoretical)\n", peak_gbs);
        std::printf("workload    : %d docs x %d dims, k=%d, %d iterations\n\n", args.doc_count,
                    args.dim, args.k, args.iterations);

        if (args.markdown) {
            std::printf("| stage | variant | median | throughput |\n");
            std::printf("| --- | --- | ---: | ---: |\n");
        }

        const zcgvs::CudaStream stream;
        const std::vector<float> corpus =
            randomNormalizedMatrix(args.doc_count, args.dim, 1234u);
        const std::vector<float> query = randomNormalizedMatrix(1, args.dim, 4321u);

        benchmarkStorage(args, stream, corpus, query, peak_gbs, args.markdown);
        benchmarkCublas(args, stream, corpus, query, peak_gbs, args.markdown);
        benchmarkTopK(args, stream, args.markdown);
        benchmarkBatchAndGraphs(args, stream, corpus, args.markdown);

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
