// Bit-exactness of the non-ideal xlns32d configuration against xlnscpp.
//
// Without xlns32_ideal, xlns32.cpp adds through an interpolated sb table and
// a cotransformed db; xlns32d.cu now ports both. Given identical xlns32
// inputs, the integer-only routines (add / sub / mul / div and the sequential
// sum / vec_dot) must match the host exactly. Routines that go through libm
// (conversions, and rms_norm via fp2xlns32d(1/n)) can differ from glibc by an
// ulp of the underlying double/float math; they are reported as a maximum
// distance in xlns32 units and must stay within MAX_LIBM_ULPS.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define xlns32_alt
#include "../xlnscpp/xlns32.cpp"
#include "../xlns32d.cu"

#define CHECK_CUDA(call) \
	do { \
		cudaError_t err = (call); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA error %s:%d: %s\n", \
				__FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(1); \
		} \
	} while (0)

#define NUM_PAIRS (1 << 23)
#define VEC_N 1536
#define NUM_VECS 64
#define MAX_LIBM_ULPS 1

enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, NUM_BINARY };
static const char *binary_names[NUM_BINARY] = { "add", "sub", "mul", "div" };

__global__ void binary_kernel(int op, const xlns32 *x, const xlns32 *y, xlns32 *out, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	switch (op) {
	case OP_ADD: out[i] = xlns32d_add(x[i], y[i]); break;
	case OP_SUB: out[i] = xlns32d_sub(x[i], y[i]); break;
	case OP_MUL: out[i] = xlns32d_mul(x[i], y[i]); break;
	default:     out[i] = xlns32d_div(x[i], y[i]); break;
	}
}

__global__ void convert_kernel(const float *in, xlns32 *lns, float *back, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	lns[i] = fp2xlns32d(in[i]);
	back[i] = xlns32d2fp(lns[i]);
}

__global__ void vector_kernel(const xlns32 *a, const xlns32 *b, xlns32 eps,
			      xlns32 *sums, xlns32 *dots, xlns32 *rms)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if (v >= NUM_VECS) return;
	const xlns32 *va = a + (size_t)v * VEC_N;
	const xlns32 *vb = b + (size_t)v * VEC_N;
	sums[v] = xlns32d_sum(va, VEC_N);
	dots[v] = xlns32d_vec_dot(va, vb, VEC_N);
	xlns32d_rms_norm(va, rms + (size_t)v * VEC_N, VEC_N, eps);
}

static unsigned lcg_state = 2463534242u;
static unsigned lcg_next(void)
{
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return lcg_state;
}

static float lcg_float(void)
{
	return ((float)(lcg_next() >> 8) / 16777216.0f) * 8.0f - 4.0f;
}

// Distance between two xlns32 values of the same sign, in units of the last
// log bit; a sign disagreement counts as infinitely far.
static long long lns_ulps(xlns32 a, xlns32 b)
{
	if (xlns32_sign(a) != xlns32_sign(b)) return 1LL << 40;
	long long d = (long long)xlns32_abs(a) - (long long)xlns32_abs(b);
	return d < 0 ? -d : d;
}

static int report_exact(const char *name, size_t wrong, size_t total)
{
	printf("  %-10s %s (%zu / %zu mismatches)\n", name, wrong ? "FAIL" : "ok", wrong, total);
	return wrong != 0;
}

static int report_ulps(const char *name, size_t differ, size_t total, long long max_ulps)
{
	const int fail = max_ulps > MAX_LIBM_ULPS;
	printf("  %-10s %s (%zu / %zu differ, max %lld ulp)\n",
	       name, fail ? "FAIL" : "ok", differ, total, max_ulps);
	return fail;
}

int main(void)
{
	int failures = 0;
	CHECK_CUDA(xlns32d_table_init());

	// binary ops on random pairs; half the pairs are close in magnitude so the
	// db cotransformation (subtraction of nearly equal values) is exercised
	{
		xlns32 *h_x = (xlns32 *)malloc(NUM_PAIRS * sizeof(xlns32));
		xlns32 *h_y = (xlns32 *)malloc(NUM_PAIRS * sizeof(xlns32));
		xlns32 *h_out = (xlns32 *)malloc(NUM_PAIRS * sizeof(xlns32));
		for (int i = 0; i < NUM_PAIRS; i++) {
			h_x[i] = (lcg_next() & 0x8fffffffu) | 0x30000000u;
			h_y[i] = (i & 1) ? ((h_x[i] + (lcg_next() & 0x00ffffffu)) ^ (lcg_next() & xlns32_signmask))
					 : ((lcg_next() & 0x8fffffffu) | 0x30000000u);
		}
		xlns32 *d_x, *d_y, *d_out;
		CHECK_CUDA(cudaMalloc(&d_x, NUM_PAIRS * sizeof(xlns32)));
		CHECK_CUDA(cudaMalloc(&d_y, NUM_PAIRS * sizeof(xlns32)));
		CHECK_CUDA(cudaMalloc(&d_out, NUM_PAIRS * sizeof(xlns32)));
		CHECK_CUDA(cudaMemcpy(d_x, h_x, NUM_PAIRS * sizeof(xlns32), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_y, h_y, NUM_PAIRS * sizeof(xlns32), cudaMemcpyHostToDevice));
		for (int op = 0; op < NUM_BINARY; op++) {
			binary_kernel<<<NUM_PAIRS / 256, 256>>>(op, d_x, d_y, d_out, NUM_PAIRS);
			CHECK_CUDA(cudaMemcpy(h_out, d_out, NUM_PAIRS * sizeof(xlns32), cudaMemcpyDeviceToHost));
			size_t wrong = 0;
			for (int i = 0; i < NUM_PAIRS; i++) {
				xlns32 ref = op == OP_ADD ? xlns32_add(h_x[i], h_y[i]) :
					     op == OP_SUB ? xlns32_sub(h_x[i], h_y[i]) :
					     op == OP_MUL ? xlns32_mul(h_x[i], h_y[i]) : xlns32_div(h_x[i], h_y[i]);
				if (ref != h_out[i]) wrong++;
			}
			failures += report_exact(binary_names[op], wrong, NUM_PAIRS);
		}
		cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
		free(h_x); free(h_y); free(h_out);
	}

	// conversions go through libm on both sides
	{
		const int n = 1 << 20;
		float *h_in = (float *)malloc(n * sizeof(float));
		float *h_back = (float *)malloc(n * sizeof(float));
		xlns32 *h_lns = (xlns32 *)malloc(n * sizeof(xlns32));
		for (int i = 0; i < n; i++) h_in[i] = lcg_float() * powf(2.0f, (float)(i % 61) - 30.0f);
		float *d_in, *d_back;
		xlns32 *d_lns;
		CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_back, n * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_lns, n * sizeof(xlns32)));
		CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
		convert_kernel<<<n / 256, 256>>>(d_in, d_lns, d_back, n);
		CHECK_CUDA(cudaMemcpy(h_lns, d_lns, n * sizeof(xlns32), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(h_back, d_back, n * sizeof(float), cudaMemcpyDeviceToHost));
		size_t differ = 0, back_differ = 0;
		long long max_ulps = 0;
		for (int i = 0; i < n; i++) {
			long long u = lns_ulps(fp2xlns32(h_in[i]), h_lns[i]);
			if (u) differ++;
			if (u > max_ulps) max_ulps = u;
			float ref = xlns322fp(h_lns[i]);
			if (memcmp(&ref, &h_back[i], sizeof(float)) != 0) back_differ++;
		}
		failures += report_ulps("from_float", differ, n, max_ulps);
		printf("  %-10s %s (%zu / %zu differ, compared from identical xlns32 inputs)\n",
		       "to_float", back_differ ? "note" : "ok", back_differ, (size_t)n);
		cudaFree(d_in); cudaFree(d_back); cudaFree(d_lns);
		free(h_in); free(h_back); free(h_lns);
	}

	// long reductions
	{
		const size_t n = (size_t)NUM_VECS * VEC_N;
		xlns32 *h_a = (xlns32 *)malloc(n * sizeof(xlns32));
		xlns32 *h_b = (xlns32 *)malloc(n * sizeof(xlns32));
		for (size_t i = 0; i < n; i++) {
			h_a[i] = fp2xlns32(lcg_float());
			h_b[i] = fp2xlns32(lcg_float());
		}
		const xlns32 eps = fp2xlns32(1e-6f);
		xlns32 *d_a, *d_b, *d_sums, *d_dots, *d_rms;
		CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(xlns32)));
		CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(xlns32)));
		CHECK_CUDA(cudaMalloc(&d_sums, NUM_VECS * sizeof(xlns32)));
		CHECK_CUDA(cudaMalloc(&d_dots, NUM_VECS * sizeof(xlns32)));
		CHECK_CUDA(cudaMalloc(&d_rms, n * sizeof(xlns32)));
		CHECK_CUDA(cudaMemcpy(d_a, h_a, n * sizeof(xlns32), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(xlns32), cudaMemcpyHostToDevice));
		vector_kernel<<<1, NUM_VECS>>>(d_a, d_b, eps, d_sums, d_dots, d_rms);

		xlns32 sums[NUM_VECS], dots[NUM_VECS];
		xlns32 *rms = (xlns32 *)malloc(n * sizeof(xlns32));
		xlns32 *ref = (xlns32 *)malloc(VEC_N * sizeof(xlns32));
		CHECK_CUDA(cudaMemcpy(sums, d_sums, sizeof(sums), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(dots, d_dots, sizeof(dots), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(rms, d_rms, n * sizeof(xlns32), cudaMemcpyDeviceToHost));

		size_t wrong_sum = 0, wrong_dot = 0, rms_differ = 0;
		long long rms_max_ulps = 0;
		for (int v = 0; v < NUM_VECS; v++) {
			const xlns32 *va = h_a + (size_t)v * VEC_N, *vb = h_b + (size_t)v * VEC_N;
			if (xlns32_sum(va, VEC_N) != sums[v]) wrong_sum++;
			if (xlns32_vec_dot(va, vb, VEC_N) != dots[v]) wrong_dot++;
			xlns32_rms_norm(va, ref, VEC_N, eps);
			for (int i = 0; i < VEC_N; i++) {
				long long u = lns_ulps(ref[i], rms[(size_t)v * VEC_N + i]);
				if (u) rms_differ++;
				if (u > rms_max_ulps) rms_max_ulps = u;
			}
		}
		failures += report_exact("sum", wrong_sum, NUM_VECS);
		failures += report_exact("vec_dot", wrong_dot, NUM_VECS);
		failures += report_ulps("rms_norm", rms_differ, n, rms_max_ulps);

		cudaFree(d_a); cudaFree(d_b); cudaFree(d_sums); cudaFree(d_dots); cudaFree(d_rms);
		free(h_a); free(h_b); free(rms); free(ref);
	}

	CHECK_CUDA(cudaGetLastError());
	printf("\nchkxlns32d_table %s (%d failing checks)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
