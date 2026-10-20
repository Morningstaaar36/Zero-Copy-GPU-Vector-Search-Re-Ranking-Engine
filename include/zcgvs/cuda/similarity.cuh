#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "zcgvs/core/metric.hpp"
#include "zcgvs/core/scalar_type.hpp"

namespace zcgvs {
namespace cuda {

// Scores `batch` queries against every document in the VRAM-resident corpus.
//
//   d_queries : [batch, dim] float32, row-major. Binary corpora expect packed query bits
//               instead (see launchSimilarityBinary), which launchSimilarity dispatches to.
//   d_scores  : [batch, doc_count] float32, row-major.
//   d_mask    : optional [doc_count] uint8 allow-mask; masked documents score -FLT_MAX.
//   scale     : dequantization scale for Int8 storage, ignored otherwise.
void launchSimilarity(const void* d_corpus,
                      ScalarType dtype,
                      float scale,
                      int doc_count,
                      int dim,
                      const void* d_queries,
                      int batch,
                      const std::uint8_t* d_mask,
                      Metric metric,
                      float* d_scores,
                      cudaStream_t stream);

// Number of queries a single block of the batched kernels co-resides.
int similarityQueryTile();

// Dynamic shared-memory bytes a launch of the batched kernel requires.
std::size_t similaritySharedBytes(ScalarType dtype, int dim, int batch);

}  // namespace cuda
}  // namespace zcgvs
