# GPU validation: runs kernels on the GPU through bgfx and compares them with a host reference.
#
#   cmake -S examples/gpu_validation -B build-gv -DCMAKE_PREFIX_PATH=<CudaGFX install;bgfx install> \
#         -DSHADERC=<path to shaderc>
#   cmake --build build-gv && ./build-gv/gpu_validation --renderer vulkan
#
# The target list below is for Linux; add windows:s_5_0 (Direct3D 11), windows:spirv, osx:metal... for the others.
cmake_minimum_required(VERSION 3.20)
project(CudaGFXGpuValidation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CudaGFX CONFIG REQUIRED)
find_package(bgfx CONFIG REQUIRED)
find_package(Threads REQUIRED)
set(SHADERC "" CACHE FILEPATH "shaderc executable (bgfx tools)")

add_executable(gpu_validation main.cpp)
target_link_libraries(gpu_validation PRIVATE CudaGFX::rt bgfx::bgfx Threads::Threads)
target_include_directories(gpu_validation PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
cudagfx_add_kernels(
    TARGET gpu_validation
    SOURCES gpu_checks.cu
    TARGETS linux:spirv linux:430
    SHADERC_EXECUTABLE "${SHADERC}"
    EMIT_CPU)
