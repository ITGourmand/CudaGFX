# Consumer project OUTSIDE the CudaGFX tree: it only knows the installed package (find_package(CudaGFX CONFIG)).
# Without a GPU or bgfx: shaderc is replaced by tests/tools/fake_shaderc.sh, the runtime is compiled with CUDAGFX_NO_BGFX,
# and the program (a .cu that provides main()) runs the kernels on the CPU backend.
cmake_minimum_required(VERSION 3.20)
project(CudaGFXInstalledConsumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CudaGFX 0.1 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(app)
target_compile_definitions(app PRIVATE CUDAGFX_NO_BGFX)
target_link_libraries(app PRIVATE CudaGFX::rt Threads::Threads)
cudagfx_add_kernels(
    TARGET app
    SOURCES "${CPU_SOURCE}"
    EMIT_CPU
    SHADERC_EXECUTABLE "${FAKE_SHADERC_PATH}")
