cmake_minimum_required(VERSION 3.18)
project(xlnscuda LANGUAGES CXX CUDA)

if(NOT DEFINED XLNSCPP_DIR)
    set(XLNSCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/xlnscpp")
endif()
if(NOT EXISTS "${XLNSCPP_DIR}/xlns16.cpp" OR NOT EXISTS "${XLNSCPP_DIR}/xlns32.cpp")
    message(FATAL_ERROR "Set XLNSCPP_DIR to an initialized xlnscpp checkout")
endif()

find_package(CUDAToolkit REQUIRED)
add_library(xlnscuda-ggml STATIC
    src/ggml-kernels.cpp
    src/ggml-kernels16.cu
    src/ggml-kernels32.cu)
target_include_directories(xlnscuda-ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(xlnscuda-ggml PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${XLNSCPP_DIR}")
target_link_libraries(xlnscuda-ggml PUBLIC CUDA::cudart)
target_compile_features(xlnscuda-ggml PUBLIC cxx_std_17)
set_target_properties(xlnscuda-ggml PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(xlnscuda-ggml-smoke tests/ggml-kernels-smoke.cu)
        target_link_libraries(xlnscuda-ggml-smoke PRIVATE xlnscuda-ggml CUDA::cudart)
        set_target_properties(xlnscuda-ggml-smoke PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME xlnscuda_ggml_smoke COMMAND xlnscuda-ggml-smoke)
        set_tests_properties(xlnscuda_ggml_smoke PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
