# CMakeLists.txt: builds the cudagfx tool, the tests (CTest) and the install rules.
# (The Makefile remains the reference build path of this repository; this file serves consumer projects.)
cmake_minimum_required(VERSION 3.20)
project(CudaGFX VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CUDAGFX_BUILD_TESTS "Build and register the tests (CTest)" ON)

# The transpiler library: every include/CudaGFX/*.cpp (the runtime headers are header-only).
file(GLOB CUDAGFX_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/CudaGFX/*.cpp")

add_library(cudagfx_core STATIC ${CUDAGFX_CORE_SOURCES})
target_include_directories(cudagfx_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(NOT MSVC)
    target_compile_options(cudagfx_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cudagfx src/cudagfx.cpp)
target_link_libraries(cudagfx PRIVATE cudagfx_core)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    # Keeps the executable free of libstdc++/libgcc_s dynamic dependencies (checked by the test suite).
    target_link_options(cudagfx PRIVATE -static-libgcc -static-libstdc++)
endif()

# Runtime (header-only): link it to the applications that use the generated code.
add_library(cudagfx_rt INTERFACE)
add_library(CudaGFX::rt ALIAS cudagfx_rt)
target_include_directories(cudagfx_rt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(cudagfx_rt INTERFACE cxx_std_17)

include(cmake/CudaGFX.cmake)

install(TARGETS cudagfx RUNTIME DESTINATION bin)
install(FILES include/CudaGFX/cudagfx_rt.h include/CudaGFX/cudagfx_abi.h include/CudaGFX/cudagfx_compat.h
        DESTINATION include/CudaGFX)
install(FILES cmake/CudaGFX.cmake DESTINATION lib/cmake/CudaGFX)

if(CUDAGFX_BUILD_TESTS)
    enable_testing()

    # Front-end unit tests
    file(GLOB CUDAGFX_UNIT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/*.cpp")
    add_executable(cudagfx_unit_tests ${CUDAGFX_UNIT_SOURCES})
    target_link_libraries(cudagfx_unit_tests PRIVATE cudagfx_core)
    add_test(NAME unit COMMAND cudagfx_unit_tests)

    # Runtime test without a GPU (compiled without exceptions or RTTI, like the runtime contract requires)
    add_executable(cudagfx_runtime_cpu_test tests/runtime/test_runtime_cpu.cpp)
    target_compile_definitions(cudagfx_runtime_cpu_test PRIVATE CUDAGFX_NO_BGFX)
    target_link_libraries(cudagfx_runtime_cpu_test PRIVATE cudagfx_rt)
    if(NOT MSVC)
        target_compile_options(cudagfx_runtime_cpu_test PRIVATE -fno-exceptions -fno-rtti -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME runtime_cpu COMMAND cudagfx_runtime_cpu_test)

    find_program(CUDAGFX_BASH bash)
    if(CUDAGFX_BASH)
        # The end-to-end suite, run against the freshly built executable
        add_test(NAME suite COMMAND "${CUDAGFX_BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.sh")
        set_tests_properties(suite PROPERTIES ENVIRONMENT "CUDAGFX_BIN=$<TARGET_FILE:cudagfx>" TIMEOUT 600)
        # cudagfx_add_kernels, end to end, in a throw-away consumer project (see tests/cmake)
        add_test(NAME cmake_integration COMMAND "${CMAKE_COMMAND}"
                 -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake
                 -DBIN=${CMAKE_CURRENT_BINARY_DIR}/cmake_integration
                 -DCUDAGFX=$<TARGET_FILE:cudagfx>
                 -DFAKE_SHADERC=${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/fake_shaderc.sh
                 -DCUDAGFX_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/run_consumer.cmake)
    endif()
endif()
