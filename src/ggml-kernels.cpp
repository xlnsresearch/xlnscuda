#include "xlnscuda/ggml-kernels.h"

#include <cuda_runtime.h>

extern "C" int xlnscuda_f32_binary16_async(enum xlnscuda_binary_op, const float *, const float *, float *, size_t, void *);
extern "C" int xlnscuda_f32_binary32_async(enum xlnscuda_binary_op, const float *, const float *, float *, size_t, void *);
extern "C" int xlnscuda_f32_mul_mat16_async(const float *, const float *, float *, size_t, size_t, size_t, void *);
extern "C" int xlnscuda_f32_mul_mat32_async(const float *, const float *, float *, size_t, size_t, size_t, void *);

extern "C" int xlnscuda_f32_binary_async(int mode, enum xlnscuda_binary_op op,
                                          const float * a, const float * b, float * dst,
                                          size_t n, void * stream) {
    if (mode == 16) return xlnscuda_f32_binary16_async(op, a, b, dst, n, stream);
    if (mode == 32) return xlnscuda_f32_binary32_async(op, a, b, dst, n, stream);
    return cudaErrorInvalidValue;
}

extern "C" int xlnscuda_f32_mul_mat_async(int mode, const float * a, const float * b,
                                           float * dst, size_t k, size_t m, size_t n,
                                           void * stream) {
    if (mode == 16) return xlnscuda_f32_mul_mat16_async(a, b, dst, k, m, n, stream);
    if (mode == 32) return xlnscuda_f32_mul_mat32_async(a, b, dst, k, m, n, stream);
    return cudaErrorInvalidValue;
}
