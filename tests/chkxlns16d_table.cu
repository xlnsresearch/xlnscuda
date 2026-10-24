// Bit-exactness of the xlns16d_table configuration against xlnscpp.
//
// With xlns16_table + xlns16_alt on the host and xlns16d_table on the device,
// every device routine must return exactly the bits the host returns. That is
// what lets a GPU backend reproduce a CPU backend's results instead of merely
// approximating them, so this test demands zero mismatches everywhere:
//   - xlns16d2fp for all 65536 values
//   - fp2xlns16d for every lookup bucket (both bucket edges)
//   - add / sub / mul / div for every x against a strided subset of y
//   - sigmoid / exp / log / silu / tanh / gelu for all 65536 values
//   - sum / vec_dot / rms_norm / softmax on long random vectors

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define xlns16_table
#define xlns16_alt
#define xlns16d_table
#include "../xlnscpp/xlns16.cpp"
#include "../xlns16d.cu"

#define CHECK_CUDA(call) \
	do { \
		cudaError_t err = (call); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA error %s:%d: %s\n", \
				__FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(1); \
		} \
	} while (0)

#define Y_STRIDE 61
#define NUM_Y ((65536 + Y_STRIDE - 1) / Y_STRIDE)
#define VEC_N 1536
#define NUM_VECS 64

enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, NUM_BINARY };
static const char *binary_names[NUM_BINARY] = { "add", "sub", "mul", "div" };

enum { FN_SIGMOID, FN_EXP, FN_LOG, FN_SILU, FN_TANH, FN_GELU, NUM_UNARY };
static const char *unary_names[NUM_UNARY] = { "sigmoid", "exp", "log", "silu", "tanh", "gelu" };

__global__ void to_float_kernel(float *out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < 65536) out[x] = xlns16d2fp((xlns16)x);
}

__global__ void from_float_kernel(const float *in, xlns16 *out, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = fp2xlns16d(in[i]);
}

__global__ void binary_kernel(int op, xlns16 *out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 65536 * NUM_Y) return;
	xlns16 x = (xlns16)(idx % 65536);
	xlns16 y = (xlns16)((idx / 65536) * Y_STRIDE);
	switch (op) {
	case OP_ADD: out[idx] = xlns16d_add(x, y); break;
	case OP_SUB: out[idx] = xlns16d_sub(x, y); break;
	case OP_MUL: out[idx] = xlns16d_mul(x, y); break;
	default:     out[idx] = xlns16d_div(x, y); break;
	}
}

__global__ void unary_kernel(int fn, xlns16 *out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= 65536) return;
	switch (fn) {
	case FN_SIGMOID: out[x] = xlns16d_sigmoid((xlns16)x); break;
	case FN_EXP:     out[x] = xlns16d_exp((xlns16)x); break;
	case FN_LOG:     out[x] = xlns16d_log((xlns16)x); break;
	case FN_SILU:    out[x] = xlns16d_silu((xlns16)x); break;
	case FN_TANH:    out[x] = xlns16d_tanh((xlns16)x); break;
	default:         out[x] = xlns16d_gelu((xlns16)x); break;
	}
}

// One thread per vector: the reductions are sequential by definition.
__global__ void vector_kernel(const xlns16 *a, const xlns16 *b, xlns16 eps, xlns16 scale,
			      xlns16 *sums, xlns16 *dots, xlns16 *rms, xlns16 *soft)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if (v >= NUM_VECS) return;
	const xlns16 *va = a + (size_t)v * VEC_N;
	const xlns16 *vb = b + (size_t)v * VEC_N;
	sums[v] = xlns16d_sum(va, VEC_N);
	dots[v] = xlns16d_vec_dot(va, vb, VEC_N);
	xlns16d_rms_norm(va, rms + (size_t)v * VEC_N, VEC_N, eps);
	xlns16d_softmax(va, soft + (size_t)v * VEC_N, VEC_N, scale);
}

static int report(const char *name, size_t wrong, size_t total)
{
	printf("  %-10s %s (%zu / %zu mismatches)\n", name, wrong ? "FAIL" : "ok", wrong, total);
	return wrong != 0;
}

static unsigned lcg_state = 12345u;
static float lcg_float(void)
{
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return ((float)(lcg_state >> 8) / 16777216.0f) * 8.0f - 4.0f;
}

int main(void)
{
	int failures = 0;
	CHECK_CUDA(xlns16d_table_init());

	// xlns16d2fp: every representable value
	{
		float *d_out, *h_out = (float *)malloc(65536 * sizeof(float));
		CHECK_CUDA(cudaMalloc(&d_out, 65536 * sizeof(float)));
		to_float_kernel<<<256, 256>>>(d_out);
		CHECK_CUDA(cudaMemcpy(h_out, d_out, 65536 * sizeof(float), cudaMemcpyDeviceToHost));
		size_t wrong = 0;
		for (int x = 0; x < 65536; x++) {
			float ref = xlns162fp((xlns16)x);
			if (memcmp(&ref, &h_out[x], sizeof(float)) != 0) wrong++;
		}
		failures += report("to_float", wrong, 65536);
		cudaFree(d_out);
		free(h_out);
	}

	// fp2xlns16d: both edges of every one of the 2^17 table buckets
	{
		const int n = 2 * 131072;
		float *h_in = (float *)malloc(n * sizeof(float));
		xlns16 *h_out = (xlns16 *)malloc(n * sizeof(xlns16));
		for (int b = 0; b < 131072; b++) {
			unsigned lo = (unsigned)b << 15, hi = lo | 0x7fffu;
			memcpy(&h_in[2 * b], &lo, sizeof(float));
			memcpy(&h_in[2 * b + 1], &hi, sizeof(float));
		}
		float *d_in;
		xlns16 *d_out;
		CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(xlns16)));
		CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
		from_float_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
		CHECK_CUDA(cudaMemcpy(h_out, d_out, n * sizeof(xlns16), cudaMemcpyDeviceToHost));
		size_t wrong = 0;
		for (int i = 0; i < n; i++)
			if (fp2xlns16(h_in[i]) != h_out[i]) wrong++;
		failures += report("from_float", wrong, n);
		cudaFree(d_in);
		cudaFree(d_out);
		free(h_in);
		free(h_out);
	}

	// binary ops: every x against every Y_STRIDE-th y
	{
		const size_t n = (size_t)65536 * NUM_Y;
		xlns16 *d_out, *h_out = (xlns16 *)malloc(n * sizeof(xlns16));
		CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(xlns16)));
		for (int op = 0; op < NUM_BINARY; op++) {
			binary_kernel<<<(unsigned)((n + 255) / 256), 256>>>(op, d_out);
			CHECK_CUDA(cudaMemcpy(h_out, d_out, n * sizeof(xlns16), cudaMemcpyDeviceToHost));
			size_t wrong = 0;
			for (size_t idx = 0; idx < n; idx++) {
				xlns16 x = (xlns16)(idx % 65536);
				xlns16 y = (xlns16)((idx / 65536) * Y_STRIDE);
				xlns16 ref = op == OP_ADD ? xlns16_add(x, y) :
					     op == OP_SUB ? xlns16_sub(x, y) :
					     op == OP_MUL ? xlns16_mul(x, y) : xlns16_div(x, y);
				if (ref != h_out[idx]) wrong++;
			}
			failures += report(binary_names[op], wrong, n);
		}
		cudaFree(d_out);
		free(h_out);
	}

	// unary functions: every representable value
	{
		xlns16 *d_out, *h_out = (xlns16 *)malloc(65536 * sizeof(xlns16));
		CHECK_CUDA(cudaMalloc(&d_out, 65536 * sizeof(xlns16)));
		for (int fn = 0; fn < NUM_UNARY; fn++) {
			unary_kernel<<<256, 256>>>(fn, d_out);
			CHECK_CUDA(cudaMemcpy(h_out, d_out, 65536 * sizeof(xlns16), cudaMemcpyDeviceToHost));
			size_t wrong = 0;
			for (int x = 0; x < 65536; x++) {
				xlns16 ref = fn == FN_SIGMOID ? xlns16_sigmoid((xlns16)x) :
					     fn == FN_EXP     ? xlns16_exp((xlns16)x) :
					     fn == FN_LOG     ? xlns16_log((xlns16)x) :
					     fn == FN_SILU    ? xlns16_silu((xlns16)x) :
					     fn == FN_TANH    ? xlns16_tanh((xlns16)x) : xlns16_gelu((xlns16)x);
				if (ref != h_out[x]) wrong++;
			}
			failures += report(unary_names[fn], wrong, 65536);
		}
		cudaFree(d_out);
		free(h_out);
	}

	// long reductions
	{
		const size_t n = (size_t)NUM_VECS * VEC_N;
		xlns16 *h_a = (xlns16 *)malloc(n * sizeof(xlns16));
		xlns16 *h_b = (xlns16 *)malloc(n * sizeof(xlns16));
		for (size_t i = 0; i < n; i++) {
			h_a[i] = fp2xlns16(lcg_float());
			h_b[i] = fp2xlns16(lcg_float());
		}
		const xlns16 eps = fp2xlns16(1e-6f), scale = fp2xlns16(0.125f);

		xlns16 *d_a, *d_b, *d_sums, *d_dots, *d_rms, *d_soft;
		CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(xlns16)));
		CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(xlns16)));
		CHECK_CUDA(cudaMalloc(&d_sums, NUM_VECS * sizeof(xlns16)));
		CHECK_CUDA(cudaMalloc(&d_dots, NUM_VECS * sizeof(xlns16)));
		CHECK_CUDA(cudaMalloc(&d_rms, n * sizeof(xlns16)));
		CHECK_CUDA(cudaMalloc(&d_soft, n * sizeof(xlns16)));
		CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(xlns16), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(xlns16), cudaMemcpyHostToDevice));
		vector_kernel<<<1, NUM_VECS>>>(d_a, d_b, eps, scale, d_sums, d_dots, d_rms, d_soft);

		xlns16 sums[NUM_VECS], dots[NUM_VECS];
		xlns16 *rms = (xlns16 *)malloc(n * sizeof(xlns16));
		xlns16 *soft = (xlns16 *)malloc(n * sizeof(xlns16));
		xlns16 *ref = (xlns16 *)malloc(VEC_N * sizeof(xlns16));
		CHECK_CUDA(cudaMemcpy(sums, d_sums, sizeof(sums), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(dots, d_dots, sizeof(dots), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(rms, d_rms, n * sizeof(xlns16), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(soft, d_soft, n * sizeof(xlns16), cudaMemcpyDeviceToHost));

		size_t wrong_sum = 0, wrong_dot = 0, wrong_rms = 0, wrong_soft = 0;
		for (int v = 0; v < NUM_VECS; v++) {
			const xlns16 *va = h_a + (size_t)v * VEC_N, *vb = h_b + (size_t)v * VEC_N;
			if (xlns16_sum(va, VEC_N) != sums[v]) wrong_sum++;
			if (xlns16_vec_dot(va, vb, VEC_N) != dots[v]) wrong_dot++;
			xlns16_rms_norm(va, ref, VEC_N, eps);
			for (int i = 0; i < VEC_N; i++) if (ref[i] != rms[(size_t)v * VEC_N + i]) wrong_rms++;
			xlns16_softmax(va, ref, VEC_N, scale);
			for (int i = 0; i < VEC_N; i++) if (ref[i] != soft[(size_t)v * VEC_N + i]) wrong_soft++;
		}
		failures += report("sum", wrong_sum, NUM_VECS);
		failures += report("vec_dot", wrong_dot, NUM_VECS);
		failures += report("rms_norm", wrong_rms, n);
		failures += report("softmax", wrong_soft, n);

		cudaFree(d_a); cudaFree(d_b); cudaFree(d_sums); cudaFree(d_dots);
		cudaFree(d_rms); cudaFree(d_soft);
		free(h_a); free(h_b); free(rms); free(soft); free(ref);
	}

	CHECK_CUDA(cudaGetLastError());
	printf("\nchkxlns16d_table %s (%d failing checks)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
