# Installs CudaGFX in a temporary prefix, then configures, builds and RUNS a consumer project that only uses
# find_package(CudaGFX CONFIG) (called by CTest: cmake -P).
set(prefix "${BIN}/prefix")
file(REMOVE_RECURSE "${BIN}")
file(MAKE_DIRECTORY "${BIN}")
if(NOT CONFIG)
    set(CONFIG Release)
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config ${CONFIG}
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "install failed:\n${out}\n${err}")
endif()
foreach(f include/CudaGFX/cudagfx_rt.h include/CudaGFX/cudagfx_cpu.h lib/cmake/CudaGFX/CudaGFXConfig.cmake
          lib/cmake/CudaGFX/CudaGFXConfigVersion.cmake lib/cmake/CudaGFX/CudaGFXTargets.cmake lib/cmake/CudaGFX/CudaGFX.cmake)
    if(NOT EXISTS "${prefix}/${f}")
        message(FATAL_ERROR "the install tree lacks ${f}")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${SRC}" -B "${BIN}/build" -DCMAKE_PREFIX_PATH=${prefix}
            -DCPU_SOURCE=${CPU_SOURCE} -DFAKE_SHADERC_PATH=${FAKE_SHADERC} -DCMAKE_BUILD_TYPE=${CONFIG}
    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "consumer configuration failed:\n${out}\n${err}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BIN}/build" --config ${CONFIG} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "consumer build failed:\n${out}\n${err}")
endif()

# The program compares the result of its kernel (CPU backend) with a host reference and prints OK.
file(GLOB_RECURSE app_exe "${BIN}/build/app" "${BIN}/build/app.exe")
if(NOT app_exe)
    message(FATAL_ERROR "the consumer executable was not built")
endif()
list(GET app_exe 0 app_exe)
execute_process(COMMAND "${app_exe}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0 OR NOT out MATCHES "OK")
    message(FATAL_ERROR "the consumer program failed (exit ${rc}):\n${out}\n${err}")
endif()
message(STATUS "cmake_install_consumer : OK")
