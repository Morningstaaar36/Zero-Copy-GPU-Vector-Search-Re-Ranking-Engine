#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace zcgvs {
namespace cuda {

// Accumulates BM25 scores for one query over a CSR inverted index.
//
// Posting weights are fully precomputed offline, so the device work is a scatter-add over the
// postings of the query's terms and nothing else.
//
//   d_offsets      : [vocab_size + 1] CSR row pointers over term ids.
//   d_docs         : [nnz] document index of each posting.
//   d_weights      : [nnz] BM25 contribution of each posting.
//   d_query_terms  : [term_count] deduplicated term ids.
//   d_scores       : [doc_count] accumulator; the caller must zero it first.
void launchBm25(const std::int64_t* d_offsets,
                const int* d_docs,
                const float* d_weights,
                const int* d_query_terms,
                int term_count,
                int doc_count,
                float* d_scores,
                cudaStream_t stream);

// Reduces the largest non-negative value of `d_values` into `d_max`, which the caller must
// zero first. Valid because BM25 posting weights are non-negative.
void launchMaxPositive(const float* d_values, int count, float* d_max, cudaStream_t stream);

// In-place convex combination of a dense score vector and a max-normalized lexical one:
//   dense[i] = weight * dense[i] + (1 - weight) * lexical[i] / max(lexical)
void launchFuseScores(float* d_dense,
                      const float* d_lexical,
                      const float* d_lexical_max,
                      float weight,
                      int count,
                      cudaStream_t stream);

}  // namespace cuda
}  // namespace zcgvs
