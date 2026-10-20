#pragma once

#include <cuda_runtime.h>

namespace zcgvs {
namespace cuda {

// Elements one block of the first (score-scanning) pass sorts in shared memory.
int topKTileSize();

// Elements one block of a merge pass sorts in shared memory. Every merge pass shrinks the
// candidate set by at least kMergeTile / k, which is what makes the recursion terminate.
int topKMergeTile();

// Largest k the shared-memory bitonic network supports.
int topKMaxK();

int topKBlockCount(int doc_count);

// Per-query element count of one ping-pong scratch buffer. Two buffers of this size (values
// and indices each) are required by launchTopK.
int topKScratchElements(int doc_count, int k);

// Selects the k highest scores for each of `batch` queries.
//
//   d_scores      : [batch, doc_count] float32, row-major.
//   d_out_*       : [batch, k], descending by score. Index -1 marks a padding slot.
//   d_scratch_*_a : [batch, topKScratchElements(doc_count, k)] ping-pong buffers.
//   d_scratch_*_b : same size as the _a buffers.
//
// The scores array and every intermediate candidate set stay in device memory; nothing is
// copied to the host.
void launchTopK(const float* d_scores,
                int doc_count,
                int k,
                int batch,
                int* d_out_indices,
                float* d_out_values,
                float* d_scratch_values_a,
                int* d_scratch_indices_a,
                float* d_scratch_values_b,
                int* d_scratch_indices_b,
                cudaStream_t stream);

}  // namespace cuda
}  // namespace zcgvs
