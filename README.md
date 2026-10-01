# xlnscuda
XLNSCUDA: a CUDA package for Logarithmic Number System eXperimentation

## Compiled ggml kernel API

`include/xlnscuda/ggml-kernels.h` is a small C-compatible launch interface for
F32 tensors in CUDA device memory. It provides elementwise add/sub/mul/div
with equal contiguous shapes and 2D matrix multiplication in ggml layout.
The format mode (`16` or `32`) is selected at each launch. The matrix kernel
is a simple correctness baseline: one CUDA thread computes each output and
reduces K terms sequentially. LNS16 uses the device's base addition, not
xlnscpp's `sum32` reduction. Quantized weights and other ggml operators are
outside this library's current kernel API.

Build it against an initialized xlnscpp checkout:

```sh
cmake -S . -B build -DXLNSCPP_DIR=/path/to/xlnscpp -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`xlnscuda_ggml_smoke` checks both modes with actual device operations. It is
skipped when no CUDA driver or GPU is available. The xlnscpp include model is
contained in the library's CUDA translation units; ggml callers include only
the public header and link `xlnscuda-ggml`.

This repository provides `xlns16d.cu` and `xlns32d.cu` along with a few programs that illustrate their use. They are translated from similar routines in the `xlnscpp` repository. The appropriate C++ file from the `xlnscpp` subdirectory (`xlnscpp/xlns16.cpp` or `xlnscpp/xlns32.cpp`) must be included before including the associated file from this repository.  All these routines are based on similar math foundation (Gaussian logs, sb and db) as the Python xlns repository, but unlike the Python they use different internal storage format.

The `xlnscpp` directory is a git submodule, not a duplicated copy of the C++ repository. After cloning this repository, initialize it with:

```sh
git submodule update --init
```

Unlike the Python xlns, here internal representation is not twos complement; it is offset by a constant (`xlns16_logsignmask` or `xlns32_logsignmask`). With the 16-bit format of `xlns16d.cu`, this is roughly similar to `bfloat16` (1 sign bit, 8 `int(log2)` bits, 7 `frac(log2)` bits). With the 32-bit format of `xlns32d.cu`, this is roughly similar to `float` (1 sign bit, 8 `int(log2)` bits, 23 `frac(log2)` bits). There is an exact representation of 0.0, but no subnormals or NaNs.

Just as with `xlns16.cpp` and `xlns32.cpp`, there are two ways to use this library: function calls (like `xlns16d_add` or `xlns32d_add`) that operate on integer representations (`typedef` as `xlns_16` or `xlns_32`, the same as used in `xlns16.cpp` and `xlns32.cpp`) that represent the LNS value; or C++ overloaded operators that operate on an LNS class (either `xlns16d_float` or `xlns32d_float`--note different than the C++ version).  The functions are a bit faster but overloading is easier.  

This CUDA package does not support (ignores) the `xlns16_table` from the C++ version because it is not practical to share the large tables involved with the many threads in typical CUDA usage. It is allowed to have the host use `xlns16_table` (for conversions) while the device uses another method of addition. Here only `xlns32_ideal` is supported; however, non-ideal implementations of addition (both varients of LPVIP, with and without `xlns16_alt` and `xlns16_altopt`) are supported in `xlns16d.cu`.
