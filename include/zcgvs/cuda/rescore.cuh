#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "zcgvs/core/metric.hpp"
#include "zcgvs/core/scalar_type.hpp"

namespace zcgvs {
namespace cuda {

// Re-scores a coarse candidate set against a higher-precision copy of the corpus.
//
// The candidate indices are read straight out of the device buffer the top-K pass wrote, so
// the coarse-to-exact hand-off never touches the host.
//
//   d_candidate_indices : [batch, candidate_count] corpus indices, -1 for padding slots.
//   d_values            : [batch, candidate_count] exact scores under `metric`.
void launchGatherRescore(const void* d_corpus,
                         ScalarType dtype,
                         float scale,
                         int doc_count,
                         int dim,
                         const float* d_queries,
                         int batch,
                         const int* d_candidate_indices,
                         int candidate_count,
                         Metric metric,
                         float* d_values,
                         cudaStream_t stream);

// Translates positions within a candidate set back into corpus indices.
//
//   d_local_indices : [batch, k] positions in [0, candidate_count).
//   d_out_indices   : [batch, k] corpus indices, -1 where the position was padding.
void launchRemapIndices(const int* d_candidate_indices,
                        const int* d_local_indices,
                        int candidate_count,
                        int k,
                        int batch,
                        int* d_out_indices,
                        cudaStream_t stream);

std::size_t rescoreSharedBytes(int dim);

}  // namespace cuda
}  // namespace zcgvs
