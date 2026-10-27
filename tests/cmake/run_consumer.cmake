# Configures, builds, then checks the consumer project (called by CTest: cmake -P).
file(REMOVE_RECURSE "${BIN}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${SRC}" -B "${BIN}" -DCUDAGFX_ROOT=${CUDAGFX_ROOT}
            -DCUDAGFX_EXE_PATH=${CUDAGFX} -DFAKE_SHADERC_PATH=${FAKE_SHADERC}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "consumer configuration failed :\n${out}\n${err}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BIN}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "consumer build failed :\n${out}\n${err}")
endif()
# The generated host code (saxpy.cu) provides main(): we check generation and compilation, without running it.
file(GLOB_RECURSE generated "${BIN}/cudagfx_gen_consumer/saxpy_gen_kernels.h")
if(NOT generated)
    message(FATAL_ERROR "stubs header not generated")
endif()
file(READ "${generated}" hdr)
if(NOT hdr MATCHES "cs_saxpy_256_spv")
    message(FATAL_ERROR "the generated header does not reference the embedded shader")
endif()
message(STATUS "cmake_integration : OK")
