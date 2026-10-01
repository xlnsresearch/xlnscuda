#include "xlnscuda/ggml-kernels.h"

#include <cuda_runtime.h>

#define xlns16_alt
#include "xlns16.cpp"
#include "xlns16d.cu"

static __global__ void binary16(const float * a, const float * b, float * dst, size_t n, int op) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const xlns16 x = xlns16d_from_float(a[i]);
        const xlns16 y = xlns16d_from_float(b[i]);
        xlns16 z = xlns16_zero;
        switch (op) {
            case XLNSCUDA_ADD: z = xlns16d_add(x, y); break;
            case XLNSCUDA_SUB: z = xlns16d_sub(x, y); break;
            case XLNSCUDA_MUL: z = xlns16d_mul(x, y); break;
            case XLNSCUDA_DIV: z = xlns16d_div(x, y); break;
        }
        dst[i] = xlns16d_to_float(z);
    }
}

static __global__ void mul_mat16(const float * a, const float * b, float * dst,
                                 size_t k, size_t m, size_t n) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < m * n; index += gridDim.x * blockDim.x) {
        const size_t row = index % m;
        const size_t col = index / m;
        xlns16 acc = xlns16_zero;
        for (size_t j = 0; j < k; ++j) {
            const xlns16 product = xlns16d_mul(xlns16d_from_float(a[row * k + j]),
                                                xlns16d_from_float(b[col * k + j]));
            acc = j == 0 ? product : xlns16d_add(acc, product);
        }
        dst[index] = xlns16d_to_float(acc);
    }
}

extern "C" int xlnscuda_f32_binary16_async(enum xlnscuda_binary_op op,
                                             const float * a, const float * b,
                                             float * dst, size_t n, void * stream) {
    if (op < XLNSCUDA_ADD || op > XLNSCUDA_DIV || (!a && n) || (!b && n) || (!dst && n))
        return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256 < 65535 ? (n + 255) / 256 : 65535);
    binary16<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a, b, dst, n, op);
    return cudaGetLastError();
}

extern "C" int xlnscuda_f32_mul_mat16_async(const float * a, const float * b,
                                              float * dst, size_t k, size_t m,
                                              size_t n, void * stream) {
    if (!a || !b || !dst || !k || !m || !n ||
        m > SIZE_MAX / n || k > SIZE_MAX / m || k > SIZE_MAX / n) return cudaErrorInvalidValue;
    const size_t total = m * n;
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256 < 65535 ? (total + 255) / 256 : 65535);
    mul_mat16<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a, b, dst, k, m, n);
    return cudaGetLastError();
}
