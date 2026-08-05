#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define xlns16_alt
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

#define MAX_ERROR_PERCENT 1.0f
#define THREADS_PER_BLOCK 32

enum BatchOp {
	OP_MUL,
	OP_ADD,
	OP_SUB,
	OP_DIV,
	OP_SCALE,
	OP_NEG,
	OP_ABS
};

static float percent_error(float expected, float got)
{
	if (expected == got) return 0.0f;
	if (isinf(expected) || isinf(got)) return INFINITY;
	float denom = fmaxf(fabsf(expected), FLT_MIN);
	return fabsf(expected - got) / denom * 100.0f;
}

static const char *op_name(enum BatchOp op)
{
	switch (op) {
	case OP_MUL: return "mul";
	case OP_ADD: return "add";
	case OP_SUB: return "sub";
	case OP_DIV: return "div";
	case OP_SCALE: return "scale";
	case OP_NEG: return "neg";
	case OP_ABS: return "abs";
	}
	return "unknown";
}

static void fill_expected(enum BatchOp op, const xlns16 *a, const xlns16 *b,
			  xlns16 scalar, xlns16 *expected, size_t n)
{
	switch (op) {
	case OP_MUL: xlns16_batch_mul(a, b, expected, n); break;
	case OP_ADD: xlns16_batch_add(a, b, expected, n); break;
	case OP_SUB: xlns16_batch_sub(a, b, expected, n); break;
	case OP_DIV: xlns16_batch_div(a, b, expected, n); break;
	case OP_SCALE: xlns16_batch_scale(a, scalar, expected, n); break;
	case OP_NEG: xlns16_batch_neg(a, expected, n); break;
	case OP_ABS: xlns16_batch_abs(a, expected, n); break;
	}
}

static void launch_kernel(enum BatchOp op, const xlns16 *d_a, const xlns16 *d_b,
			  xlns16 scalar, xlns16 *d_c, size_t n)
{
	unsigned blocks = (unsigned)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	if (blocks == 0) blocks = 1;
	switch (op) {
	case OP_MUL: xlns16d_batch_mul_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n); break;
	case OP_ADD: xlns16d_batch_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n); break;
	case OP_SUB: xlns16d_batch_sub_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n); break;
	case OP_DIV: xlns16d_batch_div_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n); break;
	case OP_SCALE: xlns16d_batch_scale_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, scalar, d_c, n); break;
	case OP_NEG: xlns16d_batch_neg_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_c, n); break;
	case OP_ABS: xlns16d_batch_abs_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_c, n); break;
	}
}

static int check_op(const char *case_name, enum BatchOp op,
		    const xlns16 *a, const xlns16 *b, xlns16 scalar, size_t n,
		    int print_rows)
{
	xlns16 *expected = (xlns16 *)malloc(n * sizeof(expected[0]));
	xlns16 *got = (xlns16 *)malloc(n * sizeof(got[0]));
	xlns16 *d_a = 0;
	xlns16 *d_b = 0;
	xlns16 *d_c = 0;
	int failures = 0;
	int bit_diffs = 0;

	fill_expected(op, a, b, scalar, expected, n);

	CHECK_CUDA(cudaMalloc((void **)&d_a, n * sizeof(a[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_b, n * sizeof(b[0])));
	CHECK_CUDA(cudaMalloc((void **)&d_c, n * sizeof(got[0])));
	CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(a[0]), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(b[0]), cudaMemcpyHostToDevice));

	launch_kernel(op, d_a, d_b, scalar, d_c, n);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(got, d_c, n * sizeof(got[0]), cudaMemcpyDeviceToHost));

	printf("\n[%s %s]\n", case_name, op_name(op));
	printf("idx | a fp        | b fp        | CPU bits | GPU bits | CPU fp       | GPU fp       | err %%     | status\n");
	printf("----|-------------|-------------|----------|----------|--------------|--------------|-----------|--------\n");
	for (size_t i = 0; i < n; i++) {
		float expected_fp = xlns162fp(expected[i]);
		float got_fp = xlns162fp(got[i]);
		float err = percent_error(expected_fp, got_fp);
		int bit_same = expected[i] == got[i];
		int fail = err > MAX_ERROR_PERCENT;
		if (!bit_same) bit_diffs++;
		if (fail) failures++;
		if (print_rows || fail) {
			printf("%3zu | %+11.4e | %+11.4e | %04x     | %04x     | %+12.5e | %+12.5e | %9.6f | %s\n",
			       i, xlns162fp(a[i]), xlns162fp(b[i]), expected[i], got[i],
			       expected_fp, got_fp, err, fail ? "FAIL" : (bit_same ? "OK" : "ROUND"));
		}
	}
	if (!print_rows && failures == 0)
		printf("%zu values OK (%d tolerated bit differences)\n", n, bit_diffs);

	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_c));
	free(expected);
	free(got);
	return failures;
}

static void fill_lns_from_float(const float *src, xlns16 *dst, size_t n)
{
	for (size_t i = 0; i < n; i++)
		dst[i] = fp2xlns16((double)src[i]);
}

static unsigned next_lcg(unsigned *state)
{
	*state = *state * 1664525u + 1013904223u;
	return *state;
}

static float next_value(unsigned *state)
{
	unsigned v = next_lcg(state);
	int mag = (int)(v % 15u) + 1;
	float x = (float)mag / 4.0f;
	return (v & 0x80000000u) ? -x : x;
}

static void fill_random_lns(xlns16 *a, xlns16 *b, size_t n)
{
	unsigned state = 0x1234abcdu;
	for (size_t i = 0; i < n; i++) {
		float af = next_value(&state);
		float bf = next_value(&state);
		if (bf == 0.0f) bf = 1.0f;
		a[i] = fp2xlns16((double)af);
		b[i] = fp2xlns16((double)bf);
	}
}

static int check_empty(void)
{
	printf("\n[empty]\n");
	launch_kernel(OP_MUL, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_kernel(OP_ADD, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_kernel(OP_SUB, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_kernel(OP_DIV, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_kernel(OP_SCALE, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_kernel(OP_NEG, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_kernel(OP_ABS, 0, 0, xlns16_one, 0, 0);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
	printf("empty kernels completed\n");
	return 0;
}

static int check_case(const char *case_name, const xlns16 *a, const xlns16 *b,
		      xlns16 scalar, size_t n, int print_rows)
{
	int failures = 0;
	failures += check_op(case_name, OP_MUL, a, b, scalar, n, print_rows);
	failures += check_op(case_name, OP_ADD, a, b, scalar, n, print_rows);
	failures += check_op(case_name, OP_SUB, a, b, scalar, n, print_rows);
	failures += check_op(case_name, OP_DIV, a, b, scalar, n, print_rows);
	failures += check_op(case_name, OP_SCALE, a, b, scalar, n, print_rows);
	failures += check_op(case_name, OP_NEG, a, b, scalar, n, print_rows);
	failures += check_op(case_name, OP_ABS, a, b, scalar, n, print_rows);
	return failures;
}

int main(void)
{
	const float fixed_a_fp[] = {
		0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f,
		-3.0f, 0.5f, -0.5f, 4.0f, -4.0f, 8.0f
	};
	const float fixed_b_fp[] = {
		1.0f, -1.0f, 1.0f, 3.0f, 3.0f, -3.0f,
		3.0f, 0.5f, -0.5f, -4.0f, 4.0f, 2.0f
	};
	const size_t fixed_n = sizeof(fixed_a_fp) / sizeof(fixed_a_fp[0]);
	const size_t random_n = 19;
	xlns16 fixed_a[fixed_n];
	xlns16 fixed_b[fixed_n];
	xlns16 random_a[random_n];
	xlns16 random_b[random_n];
	xlns16 scalar = fp2xlns16(2.5f);
	int failures = 0;

	fill_lns_from_float(fixed_a_fp, fixed_a, fixed_n);
	fill_lns_from_float(fixed_b_fp, fixed_b, fixed_n);
	fill_random_lns(random_a, random_b, random_n);

	printf("=== xlns16d batch arithmetic CPU vs GPU ===\n");
	failures += check_empty();
	failures += check_case("fixed", fixed_a, fixed_b, scalar, fixed_n, 1);
	failures += check_case("deterministic_random", random_a, random_b, scalar, random_n, 0);

	printf("\nchkxlns16d_batch_ops %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
