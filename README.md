# xlnscuda
XLNSCUDA: a CUDA package for Logarithmic Number System eXperimentation

This repository provides `xlns16d.cu` and `xlns32d.cu` along with a few programs that illustrate their use. They are translated from similar routines in the `xlnscpp` repository. The appropriate C++ file from the `xlnscpp` subdirectory (`xlnscpp/xlns16.cpp` or `xlnscpp/xlns32.cpp`) must be included before including the associated file from this repository.  All these routines are based on similar math foundation (Gaussian logs, sb and db) as the Python xlns repository, but unlike the Python they use different internal storage format.

The `xlnscpp` directory is a git submodule, not a duplicated copy of the C++ repository. After cloning this repository, initialize it with:

```sh
git submodule update --init
```

Unlike the Python xlns, here internal representation is not twos complement; it is offset by a constant (`xlns16_logsignmask` or `xlns32_logsignmask`). With the 16-bit format of `xlns16d.cu`, this is roughly similar to `bfloat16` (1 sign bit, 8 `int(log2)` bits, 7 `frac(log2)` bits). With the 32-bit format of `xlns32d.cu`, this is roughly similar to `float` (1 sign bit, 8 `int(log2)` bits, 23 `frac(log2)` bits). There is an exact representation of 0.0, but no subnormals or NaNs.

Just as with `xlns16.cpp` and `xlns32.cpp`, there are two ways to use this library: function calls (like `xlns16d_add` or `xlns32d_add`) that operate on integer representations (`typedef` as `xlns_16` or `xlns_32`, the same as used in `xlns16.cpp` and `xlns32.cpp`) that represent the LNS value; or C++ overloaded operators that operate on an LNS class (either `xlns16d_float` or `xlns32d_float`--note different than the C++ version).  The functions are a bit faster but overloading is easier.  

By default this CUDA package ignores the `xlns16_table` from the C++ version, so the host may use `xlns16_table` (for conversions) while the device uses another method of addition (both variants of LPVIP, with and without `xlns16_alt` and `xlns16_altopt`, are supported in `xlns16d.cu`).

## Matching the host bit for bit

Code that must reproduce host results exactly (for example a GPU port of a CPU computation) can mirror the host configuration on the device:

- **`xlns16d_table`** (requires `xlns16_table`): the device uses copies of the host lookup tables for conversion (`fp2xlns16d`, `xlns16d2fp`), `xlns16d_sigmoid`, `xlns16d_exp`, `xlns16d_log` and, with `xlns16_alt`, for the sb/db of `xlns16d_add`. The tables take about 900 KB of device memory and are uploaded by calling `xlns16d_table_init()` once per device before any kernel runs.
- **non-ideal `xlns32d.cu`** (`xlns32_ideal` not defined): addition uses ports of `xlns32.cpp`'s interpolated sb table and cotransformed db. The tables (about 75 KB) are uploaded with `xlns32d_table_init()`. Previously only `xlns32_ideal` compiled.

```c++
#define xlns16_table
#define xlns16_alt
#define xlns16d_table
#include "xlnscpp/xlns16.cpp"
#include "xlns16d.cu"
...
xlns16d_table_init();   // before the first kernel
```

With these, every xlns16 routine and the integer-only xlns32 routines return exactly the host's bits; xlns32 conversions and activations go through libm on both sides and can differ in the last ulp. `tests/chkxlns16d_table.cu` and `tests/chkxlns32d_table.cu` check this.

`xlns16d_rms_norm` and `xlns32d_rms_norm` port `xlns16_rms_norm` / `xlns32_rms_norm` (RMSNorm as used by LLaMA-family models).
