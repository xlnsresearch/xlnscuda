#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA error codes match cudaError_t. The pointers below refer to CUDA device
// memory. stream is a cudaStream_t cast to void *, or NULL for the default
// stream. This API has no dependency on ggml.
enum xlnscuda_binary_op {
    XLNSCUDA_ADD = 0,
    XLNSCUDA_SUB = 1,
    XLNSCUDA_MUL = 2,
    XLNSCUDA_DIV = 3,
};

int xlnscuda_f32_binary_async(int mode, enum xlnscuda_binary_op op,
                               const float * a, const float * b, float * dst,
                               size_t n, void * stream);

// ggml 2D layout: a is [K,M], b is [K,N], dst is [M,N], with dimension 0
// contiguous. K, M and N must be positive. This is a correctness baseline;
// each output thread reduces K terms sequentially.
int xlnscuda_f32_mul_mat_async(int mode, const float * a, const float * b,
                                float * dst, size_t k, size_t m, size_t n,
                                void * stream);

#ifdef __cplusplus
}
#endif
