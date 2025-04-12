# xlnscuda
XLNSCPP: a CUDA package for Logarithmic Number System eXperimentation

This repository provides `xlns16d.cu` and `xlns32d.cu` along with a few programs that illustrate their use. They are translated from similar routines in the xlnscpp repository (the appropriate file (`xlns16.cpp` or `xlns32.cpp`) before including the library in this repository.  All these routines are based on similar math foundation (Gaussian logs, sb and db) as the Python xlns repository, but unlike the Python they use different internal storage format.

Unlike the Python xlns, here internal representation is not twos complement; it is offset by a constant (`xlns16_logsignmask` or `xlns32_logsignmask`). With the 16-bit format of `xlns16.cpp`, this is roughly similar to `bfloat16` (1 sign bit, 8 `int(log2)` bits, 7 `frac(log2)` bits). With the 32-bit format of `xlns32.cpp`, this is roughly similar to `float` (1 sign bit, 8 `int(log2)` bits, 23 `frac(log2)` bits). There is an exact representation of 0.0, but no subnormals or NaNs.
