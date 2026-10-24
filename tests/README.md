# xlnscuda tests

These tests are additive CUDA regression tests. Do not modify `../chkxlns16d.cu`;
that file remains the existing golden primitive regression test.

Current repository shape:

- `xlns16d.cu` and `xlns32d.cu` rely on typedefs/macros from the matching CPU
  implementation file.
- CPU-vs-GPU parity tests therefore include the matching CPU implementation
  from the `xlnscpp` submodule. Initialize it first with
  `git submodule update --init`.

Example from the `xlnscuda` repository root:

```sh
nvcc -std=c++11 tests/chkxlns16d_utils.cu -o /tmp/chkxlns16d_utils
nvcc -std=c++11 tests/chkxlns32d_utils.cu -o /tmp/chkxlns32d_utils
nvcc -std=c++11 tests/chkxlns16d_batch_convert.cu -o /tmp/chkxlns16d_batch_convert
nvcc -std=c++11 tests/chkxlns32d_batch_convert.cu -o /tmp/chkxlns32d_batch_convert
nvcc -std=c++11 tests/chkxlns16d_helper_functions.cu -o /tmp/chkxlns16d_helper_functions
nvcc -std=c++11 tests/chkxlns32d_helper_functions.cu -o /tmp/chkxlns32d_helper_functions
nvcc -std=c++11 tests/chkxlns16d_softmax_layernorm.cu -o /tmp/chkxlns16d_softmax_layernorm
nvcc -std=c++11 tests/chkxlns32d_softmax_layernorm.cu -o /tmp/chkxlns32d_softmax_layernorm
nvcc -std=c++11 tests/chkxlns16d_table.cu -o /tmp/chkxlns16d_table
nvcc -std=c++11 tests/chkxlns32d_table.cu -o /tmp/chkxlns32d_table

/tmp/chkxlns16d_utils
/tmp/chkxlns32d_utils
/tmp/chkxlns16d_batch_convert
/tmp/chkxlns32d_batch_convert
/tmp/chkxlns16d_helper_functions
/tmp/chkxlns32d_helper_functions
/tmp/chkxlns16d_softmax_layernorm
/tmp/chkxlns32d_softmax_layernorm
/tmp/chkxlns16d_table
/tmp/chkxlns32d_table
```

`chkxlns16d_table` and `chkxlns32d_table` check the host-matching configurations
(`xlns16d_table`, non-ideal `xlns32d`) for bit-exactness against xlnscpp: all
65536 xlns16 values through every unary function, every x against a strided
set of y for the binary operations, and long sum / vec_dot / rms_norm / softmax
reductions. They demand zero mismatches, except that xlns32 conversions (libm
on both sides) may differ by one ulp.

If your nvcc rejects the default host compiler (CUDA 12.0 supports GCC up to
12), add `-ccbin g++-12`.
