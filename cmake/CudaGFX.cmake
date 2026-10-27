# cmake/CudaGFX.cmake: CMake integration of CudaGFX.
#
# Transpiles CUDA kernels (.cu) into C++17 host code + embedded bgfx compute shaders, and adds the result to
# an existing target:
#
#   include(cmake/CudaGFX.cmake)
#   add_executable(app main.cpp)
#   cudagfx_add_kernels(
#       TARGET   app
#       SOURCES  kernels.cu                       # relative to the current directory
#       TARGETS  linux:spirv linux:430            # optional (default: linux:spirv linux:430)
#       INCLUDES ${BGFX_DIR}/src                  # directory of bgfx_compute.sh (shaderc -i AND preprocessor -I)
#       DEFINES  USE_FAST_PATH=1)                 # optional: -D...
#   target_link_libraries(app PRIVATE bgfx)
#
# Executables: the `cudagfx` CMake target if it exists, otherwise CUDAGFX_EXECUTABLE, otherwise find_program.
# shaderc: SHADERC_TARGET, otherwise SHADERC_EXECUTABLE, otherwise a `shaderc` / `bgfx::shaderc` target, otherwise find_program.
# The headers included by a .cu (#include "x.cuh") trigger regeneration (--depends -> DEPFILE).
cmake_minimum_required(VERSION 3.20)

# Repository root (or install prefix): the runtime headers live next to this file.
set(_cudagfx_here "${CMAKE_CURRENT_LIST_DIR}")
foreach(_cand "${_cudagfx_here}/../include" "${_cudagfx_here}/../../../include")
    if(EXISTS "${_cand}/CudaGFX/cudagfx_rt.h")
        get_filename_component(CUDAGFX_INCLUDE_DIR "${_cand}" ABSOLUTE)
        break()
    endif()
endforeach()
unset(_cudagfx_here)

function(cudagfx_add_kernels)
    set(oneValueArgs TARGET RUNTIME_INCLUDE SHADERC_TARGET SHADERC_EXECUTABLE CUDAGFX_EXECUTABLE)
    set(multiValueArgs SOURCES TARGETS INCLUDES SHADER_INCLUDES DEFINES)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "cudagfx_add_kernels: TARGET is required")
    endif()
    if(NOT TARGET ${ARG_TARGET})
        message(FATAL_ERROR "cudagfx_add_kernels: target '${ARG_TARGET}' does not exist (create it before the call)")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "cudagfx_add_kernels: SOURCES is required")
    endif()

    # 1. The cudagfx tool. A real build dependency is declared when it is a target of this project, so that the
    #    tool is rebuilt (and the kernels regenerated) when its sources change.
    set(cudagfx_dep "")
    if(ARG_CUDAGFX_EXECUTABLE)
        set(cudagfx_exe "${ARG_CUDAGFX_EXECUTABLE}")
    elseif(TARGET cudagfx)
        set(cudagfx_exe $<TARGET_FILE:cudagfx>)
        set(cudagfx_dep cudagfx)
    else()
        find_program(CUDAGFX_EXE NAMES cudagfx HINTS "${CMAKE_SOURCE_DIR}/.build/linux/bin" REQUIRED)
        set(cudagfx_exe "${CUDAGFX_EXE}")
    endif()

    # 2. shaderc (same logic: a real dependency when it is a target)
    set(shaderc_dep "")
    if(ARG_SHADERC_EXECUTABLE)
        set(shaderc_exe "${ARG_SHADERC_EXECUTABLE}")
    elseif(ARG_SHADERC_TARGET)
        set(shaderc_exe $<TARGET_FILE:${ARG_SHADERC_TARGET}>)
        set(shaderc_dep ${ARG_SHADERC_TARGET})
    elseif(TARGET shaderc)
        set(shaderc_exe $<TARGET_FILE:shaderc>)
        set(shaderc_dep shaderc)
    elseif(TARGET bgfx::shaderc)
        set(shaderc_exe $<TARGET_FILE:bgfx::shaderc>)
        set(shaderc_dep bgfx::shaderc)
    else()
        find_program(CUDAGFX_SHADERC_EXE NAMES shaderc HINTS "${CMAKE_SOURCE_DIR}/.deps/bgfx/bin" REQUIRED)
        set(shaderc_exe "${CUDAGFX_SHADERC_EXE}")
    endif()

    # 3. Options shared by every source file
    set(gen_dir "${CMAKE_CURRENT_BINARY_DIR}/cudagfx_gen_${ARG_TARGET}")
    file(MAKE_DIRECTORY "${gen_dir}")

    if(NOT ARG_TARGETS)
        set(ARG_TARGETS "linux:spirv" "linux:430")
    endif()
    set(common_flags "")
    foreach(tgt IN LISTS ARG_TARGETS)
        list(APPEND common_flags "--target" "${tgt}")
    endforeach()
    # INCLUDES feeds both shaderc (-i: bgfx_compute.sh...) and the preprocessor (-I: #include "x.cuh")
    foreach(inc IN LISTS ARG_INCLUDES)
        list(APPEND common_flags "-i" "${inc}" "-I" "${inc}")
    endforeach()
    foreach(inc IN LISTS ARG_SHADER_INCLUDES)
        list(APPEND common_flags "-i" "${inc}")
    endforeach()
    foreach(def IN LISTS ARG_DEFINES)
        list(APPEND common_flags "-D${def}")
    endforeach()
    if(ARG_RUNTIME_INCLUDE)
        list(APPEND common_flags "--runtime-include" "${ARG_RUNTIME_INCLUDE}")
    endif()

    # 4. One custom command per source file
    set(generated "")
    foreach(cu_file IN LISTS ARG_SOURCES)
        get_filename_component(cu_stem "${cu_file}" NAME_WE)
        get_filename_component(cu_abs "${cu_file}" ABSOLUTE)

        set(out_cpp "${gen_dir}/${cu_stem}_gen.cpp")
        set(out_hdr "${gen_dir}/${cu_stem}_gen_kernels.h")
        set(out_dep "${gen_dir}/${cu_stem}.d")

        add_custom_command(
            OUTPUT "${out_cpp}" "${out_hdr}"
            COMMAND "${cudagfx_exe}" "${cu_abs}"
                    -o "${out_cpp}" --kernels-header "${out_hdr}" --shader-dir "${gen_dir}"
                    --shaderc "${shaderc_exe}" --embed --depends "${out_dep}" ${common_flags}
            DEPENDS "${cu_abs}" ${cudagfx_dep} ${shaderc_dep}
            DEPFILE "${out_dep}"
            COMMENT "CudaGFX: ${cu_stem}.cu -> host code, stubs and shaders"
            VERBATIM
        )
        list(APPEND generated "${out_cpp}")
    endforeach()

    # 5. Integration into the target
    target_sources(${ARG_TARGET} PRIVATE ${generated})
    target_include_directories(${ARG_TARGET} PRIVATE "${gen_dir}")
    if(CUDAGFX_INCLUDE_DIR)
        target_include_directories(${ARG_TARGET} PRIVATE "${CUDAGFX_INCLUDE_DIR}")
    endif()
endfunction()
