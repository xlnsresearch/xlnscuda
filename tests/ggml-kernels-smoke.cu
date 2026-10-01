#include "xlnscuda/ggml-kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

static bool check(cudaError_t error, const char * operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

static bool run_mode(int mode) {
    const float a[] = {1.0f, 2.0f, 4.0f, 8.0f}; // [K=2,M=2]
    const float b[] = {1.0f, 1.0f, 2.0f, 2.0f}; // [K=2,N=2]
    const float expected_mat[] = {3.0f, 12.0f, 6.0f, 24.0f};
    const float expected_add[] = {2.0f, 3.0f, 6.0f, 10.0f};
    float * da = nullptr;
    float * db = nullptr;
    float * dout = nullptr;
    bool ok = check(cudaMalloc(&da, sizeof(a)), "cudaMalloc(a)") &&
              check(cudaMalloc(&db, sizeof(b)), "cudaMalloc(b)") &&
              check(cudaMalloc(&dout, sizeof(a)), "cudaMalloc(out)");
    if (ok) ok = check(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice), "copy(a)") &&
                 check(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice), "copy(b)");
    if (ok) {
        ok = check(static_cast<cudaError_t>(xlnscuda_f32_binary_async(mode, XLNSCUDA_ADD,
                    da, db, dout, 4, nullptr)), "binary add") &&
             check(cudaDeviceSynchronize(), "sync(add)");
    }
    float output[4] = {};
    if (ok) ok = check(cudaMemcpy(output, dout, sizeof(output), cudaMemcpyDeviceToHost), "copy(add)");
    for (int i = 0; ok && i < 4; ++i) {
        if (!std::isfinite(output[i]) || std::fabs(output[i] - expected_add[i]) > expected_add[i] * 0.08f) {
            std::fprintf(stderr, "mode %d add[%d]: expected %g got %g\n", mode, i, expected_add[i], output[i]);
            ok = false;
        }
    }
    if (ok) {
        ok = check(static_cast<cudaError_t>(xlnscuda_f32_mul_mat_async(mode, da, db, dout,
                    2, 2, 2, nullptr)), "mul_mat") &&
             check(cudaDeviceSynchronize(), "sync(mul_mat)") &&
             check(cudaMemcpy(output, dout, sizeof(output), cudaMemcpyDeviceToHost), "copy(mul_mat)");
    }
    for (int i = 0; ok && i < 4; ++i) {
        if (!std::isfinite(output[i]) || std::fabs(output[i] - expected_mat[i]) > expected_mat[i] * 0.08f) {
            std::fprintf(stderr, "mode %d mat[%d]: expected %g got %g\n", mode, i, expected_mat[i], output[i]);
            ok = false;
        }
    }
    cudaFree(dout);
    cudaFree(db);
    cudaFree(da);
    return ok;
}

int main() {
    int count = 0;
    const cudaError_t available = cudaGetDeviceCount(&count);
    if (available != cudaSuccess || count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA GPU/driver (%s)\n", cudaGetErrorString(available));
        return 77;
    }
    return run_mode(16) && run_mode(32) ? 0 : 1;
}
