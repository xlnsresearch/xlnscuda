#include <stdio.h>
#include <stdlib.h>

#define xlns16_alt
#include "xlns16.cpp"
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

#define NUM_CASES 8
#define NUM_BOOL_RESULTS 8
#define NUM_LNS_RESULTS 3

static const char *bool_names[NUM_BOOL_RESULTS] = {
	"is_zero", "is_negative", "is_positive", "gt", "lt", "eq", "ge", "le"
};

static const char *lns_names[NUM_LNS_RESULTS] = {
	"max", "min", "copysign"
};

__global__ void xlns16d_utils_kernel(const xlns16 *a, const xlns16 *b,
				     int *bool_results, xlns16 *lns_results,
				     int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	xlns16 av = a[i];
	xlns16 bv = b[i];
	int base_bool = i * NUM_BOOL_RESULTS;
	int base_lns = i * NUM_LNS_RESULTS;

	bool_results[base_bool + 0] = xlns16d_is_zero(av);
	bool_results[base_bool + 1] = xlns16d_is_negative(av);
	bool_results[base_bool + 2] = xlns16d_is_positive(av);
	bool_results[base_bool + 3] = xlns16d_gt(av, bv);
	bool_results[base_bool + 4] = xlns16d_lt(av, bv);
	bool_results[base_bool + 5] = xlns16d_eq(av, bv);
	bool_results[base_bool + 6] = xlns16d_ge(av, bv);
	bool_results[base_bool + 7] = xlns16d_le(av, bv);

	lns_results[base_lns + 0] = xlns16d_max(av, bv);
	lns_results[base_lns + 1] = xlns16d_min(av, bv);
	lns_results[base_lns + 2] = xlns16d_copysign(av, bv);
}

static int check_bool_result(int case_idx, int result_idx, int expected, int got)
{
	if (expected == got) return 0;
	printf("bool mismatch case=%d result=%d expected=%d got=%d\n",
	       case_idx, result_idx, expected, got);
	return 1;
}

static int check_lns_result(int case_idx, int result_idx, xlns16 expected, xlns16 got)
{
	if (expected == got) return 0;
	printf("lns mismatch case=%d result=%d expected=%04x got=%04x\n",
	       case_idx, result_idx, expected, got);
	return 1;
}

static void print_results_table(const xlns16 *a, const xlns16 *b,
				const int *expected_bool, const int *got_bool,
				const xlns16 *expected_lns, const xlns16 *got_lns)
{
	printf("\n=== xlns16d utility CPU vs GPU table (grouped by function) ===\n");
	for (int f = 0; f < NUM_BOOL_RESULTS; f++) {
		printf("\n[%s]\n", bool_names[f]);
		printf("case | a bits | b bits | a fp | b fp | CPU | GPU\n");
		printf("-----|--------|--------|------|------|-----|-----\n");
		for (int i = 0; i < NUM_CASES; i++) {
			int idx = i * NUM_BOOL_RESULTS + f;
			printf("%4d | %04x   | %04x   | %+7.4f | %+7.4f | %3d | %3d\n",
			       i, a[i], b[i], xlns162fp(a[i]), xlns162fp(b[i]),
			       expected_bool[idx], got_bool[idx]);
		}
	}
	for (int f = 0; f < NUM_LNS_RESULTS; f++) {
		printf("\n[%s]\n", lns_names[f]);
		printf("case | a bits | b bits | a fp | b fp | CPU | GPU\n");
		printf("-----|--------|--------|------|------|-----|-----\n");
		for (int i = 0; i < NUM_CASES; i++) {
			int idx = i * NUM_LNS_RESULTS + f;
			printf("%4d | %04x   | %04x   | %+7.4f | %+7.4f | %04x | %04x\n",
			       i, a[i], b[i], xlns162fp(a[i]), xlns162fp(b[i]),
			       expected_lns[idx], got_lns[idx]);
		}
	}
	printf("\n");
}

int main(void)
{
	xlns16 h_a[NUM_CASES] = {
		/* zero */
		xlns16_zero,
		/* positive */
		xlns16_one,
		/* negative */
		xlns16_neg_one,
		/* gt/lt */
		xlns16_two,
		/* eq/ge/le */
		xlns16_half,
		/* max/min */
		fp2xlns16(0.25f),
		/* copysign */
		fp2xlns16(-0.25f),
		/* mixed edge */
		fp2xlns16(3.5f)
	};
	xlns16 h_b[NUM_CASES] = {
		/* zero */
		xlns16_zero,
		/* positive */
		xlns16_two,
		/* negative */
		xlns16_neg_two,
		/* gt/lt */
		xlns16_one,
		/* eq/ge/le */
		xlns16_half,
		/* max/min */
		fp2xlns16(0.25f),
		/* copysign */
		fp2xlns16(0.25f),
		/* mixed edge */
		fp2xlns16(-4.0f)
	};

	int expected_bool[NUM_CASES * NUM_BOOL_RESULTS];
	xlns16 expected_lns[NUM_CASES * NUM_LNS_RESULTS];
	int got_bool[NUM_CASES * NUM_BOOL_RESULTS];
	xlns16 got_lns[NUM_CASES * NUM_LNS_RESULTS];

	for (int i = 0; i < NUM_CASES; i++) {
		int base_bool = i * NUM_BOOL_RESULTS;
		int base_lns = i * NUM_LNS_RESULTS;
		expected_bool[base_bool + 0] = xlns16_is_zero(h_a[i]);
		expected_bool[base_bool + 1] = xlns16_is_negative(h_a[i]);
		expected_bool[base_bool + 2] = xlns16_is_positive(h_a[i]);
		expected_bool[base_bool + 3] = xlns16_gt(h_a[i], h_b[i]);
		expected_bool[base_bool + 4] = xlns16_lt(h_a[i], h_b[i]);
		expected_bool[base_bool + 5] = xlns16_eq(h_a[i], h_b[i]);
		expected_bool[base_bool + 6] = xlns16_ge(h_a[i], h_b[i]);
		expected_bool[base_bool + 7] = xlns16_le(h_a[i], h_b[i]);

		expected_lns[base_lns + 0] = xlns16_max(h_a[i], h_b[i]);
		expected_lns[base_lns + 1] = xlns16_min(h_a[i], h_b[i]);
		expected_lns[base_lns + 2] = xlns16_copysign(h_a[i], h_b[i]);
	}

	xlns16 *d_a = 0;
	xlns16 *d_b = 0;
	int *d_bool = 0;
	xlns16 *d_lns = 0;

	CHECK_CUDA(cudaMalloc((void **)&d_a, sizeof(h_a)));
	CHECK_CUDA(cudaMalloc((void **)&d_b, sizeof(h_b)));
	CHECK_CUDA(cudaMalloc((void **)&d_bool, sizeof(got_bool)));
	CHECK_CUDA(cudaMalloc((void **)&d_lns, sizeof(got_lns)));
	CHECK_CUDA(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));

	xlns16d_utils_kernel<<<1, 32>>>(d_a, d_b, d_bool, d_lns, NUM_CASES);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaMemcpy(got_bool, d_bool, sizeof(got_bool), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(got_lns, d_lns, sizeof(got_lns), cudaMemcpyDeviceToHost));

	print_results_table(h_a, h_b, expected_bool, got_bool, expected_lns, got_lns);

	int wrong = 0;
	for (int i = 0; i < NUM_CASES; i++) {
		for (int j = 0; j < NUM_BOOL_RESULTS; j++) {
			int idx = i * NUM_BOOL_RESULTS + j;
			wrong += check_bool_result(i, j, expected_bool[idx], got_bool[idx]);
		}
		for (int j = 0; j < NUM_LNS_RESULTS; j++) {
			int idx = i * NUM_LNS_RESULTS + j;
			wrong += check_lns_result(i, j, expected_lns[idx], got_lns[idx]);
		}
	}

	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_b));
	CHECK_CUDA(cudaFree(d_bool));
	CHECK_CUDA(cudaFree(d_lns));

	printf("chkxlns16d_utils %s (%d wrong)\n", wrong ? "FAIL" : "PASS", wrong);
	return wrong ? 1 : 0;
}
