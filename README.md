# xlnscuda
XLNSCPP: a CUDA package for Logarithmic Number System eXperimentation

This repository provides `xlns16d.cu` and `xlns32d.cu` along with a few programs that illustrate their use. They are translated from similar routines in the xlnscpp repository. The appropriate file (`xlns16.cpp` or `xlns32.cpp`) must be included before including the associated file from this repository.  All these routines are based on similar math foundation (Gaussian logs, sb and db) as the Python xlns repository, but unlike the Python they use different internal storage format.

Unlike the Python xlns, here internal representation is not twos complement; it is offset by a constant (`xlns16_logsignmask` or `xlns32_logsignmask`). With the 16-bit format of `xlns16.cpp`, this is roughly similar to `bfloat16` (1 sign bit, 8 `int(log2)` bits, 7 `frac(log2)` bits). With the 32-bit format of `xlns32.cpp`, this is roughly similar to `float` (1 sign bit, 8 `int(log2)` bits, 23 `frac(log2)` bits). There is an exact representation of 0.0, but no subnormals or NaNs.

Just as with `xlns16.cpp` and `xlns32.cpp`, there are two ways to use this library: function calls (like `xlns16d_add` or `xlns32d_add`) that operate on integer representations (`typedef` as `xlns_16` or `xlns_32`, the same as used in `xlns16.cpp` and `xlns32.cpp`) that represent the LNS value; or C++ overloaded operators that operate on an LNS class (either `xlns16d_float` or `xlns32d_float`--note different than the C++ version).  The functions are a bit faster but overloading is easier.  

This CUDA package does not support (ignores) the `xlns16_table` from the C++ version because it is not practical to share the large tables involved with the many threads in typical CUDA usage. It is allowed to have the host use `xlns16_table` (for conversions) while the device uses another method of addition. Only `xlns32_ideal` is supported; non-ideal implementations of addition are supported in `xlns16d.cu`.
