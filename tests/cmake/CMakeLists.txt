# Minimal consumer project: checks cudagfx_add_kernels end to end, without a GPU or bgfx
# (shaderc replaced by tests/tools/fake_shaderc.sh, runtime compiled with CUDAGFX_NO_BGFX).
cmake_minimum_required(VERSION 3.20)
project(CudaGFXConsumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include("${CUDAGFX_ROOT}/cmake/CudaGFX.cmake")

add_executable(consumer main.cpp)
target_compile_definitions(consumer PRIVATE CUDAGFX_NO_BGFX)
cudagfx_add_kernels(
    TARGET consumer
    SOURCES "${CUDAGFX_ROOT}/tests/golden/saxpy/saxpy.cu"
    TARGETS linux:spirv linux:430
    CUDAGFX_EXECUTABLE "${CUDAGFX_EXE_PATH}"
    SHADERC_EXECUTABLE "${FAKE_SHADERC_PATH}")
