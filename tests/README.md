# xlnscuda tests

These tests are additive CUDA regression tests. Do not modify `../chkxlns16d.cu`;
that file remains the existing golden primitive regression test.

Current repository shape:

- `xlns16d.cu` and `xlns32d.cu` rely on typedefs/macros from the matching CPU
  implementation file.
- CPU-vs-GPU parity tests therefore need the `xlnscpp` sources on the include
  path.
- The tests do not hardcode a sibling repository path; pass it with `-I`.

Example from the `xlnscuda` repository root:

```sh
nvcc -std=c++11 -I/path/to/xlnscpp tests/chkxlns16d_utils.cu -o /tmp/chkxlns16d_utils
nvcc -std=c++11 -I/path/to/xlnscpp tests/chkxlns32d_utils.cu -o /tmp/chkxlns32d_utils

/tmp/chkxlns16d_utils
/tmp/chkxlns32d_utils
```

For the side-by-side workspace used during this audit:

```sh
nvcc -std=c++11 -I../xlnscpp tests/chkxlns16d_utils.cu -o /tmp/chkxlns16d_utils
nvcc -std=c++11 -I../xlnscpp tests/chkxlns32d_utils.cu -o /tmp/chkxlns32d_utils
```

