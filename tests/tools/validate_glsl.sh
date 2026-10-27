#!/usr/bin/env bash
# Validates a .sc shader (GLSL 4.30 backend) without shaderc: C preprocessor with the real bgfx headers
# (bgfx_compute.sh, bgfx_shader.sh), then glslangValidator.
#
# This emulates what shaderc does for the GLSL backend: bgfx_compute.sh turns BUFFER_RO/BUFFER_RW/NUM_THREADS/...
# into plain GLSL, so a successful glslangValidator run proves the generated types and syntax are valid.
# It does NOT replace shaderc (SPIR-V, HLSL, Metal and WGSL are not exercised).
#
# Usage: BGFX_SH_DIR=<directory containing bgfx_compute.sh> validate_glsl.sh <file.sc>
# Exit codes: 0 valid, 1 invalid, 3 preprocessor error, 77 tools missing (test skipped).
set -u
sc="$1"
inc="${BGFX_SH_DIR:-}"
if [ -z "$inc" ] || [ ! -f "$inc/bgfx_compute.sh" ] || ! command -v glslangValidator >/dev/null || ! command -v cpp >/dev/null; then
    exit 77
fi
pre=$(mktemp --suffix=.comp)
trap 'rm -f "$pre" "$pre.err"' EXIT
{
    echo "#version 430"
    # shaderc defines every BGFX_SHADER_LANGUAGE_* macro (0 or the version): do the same for GLSL.
    cpp -P -x c -undef -nostdinc -I"$inc" \
        -DBGFX_SHADER_LANGUAGE_HLSL=0 -DBGFX_SHADER_LANGUAGE_METAL=0 -DBGFX_SHADER_LANGUAGE_PSSL=0 \
        -DBGFX_SHADER_LANGUAGE_SPIRV=0 -DBGFX_SHADER_LANGUAGE_WGSL=0 -DBGFX_SHADER_LANGUAGE_GLSL=430 \
        -DBGFX_SHADER_TYPE_COMPUTE=1 "$sc"
} > "$pre" 2> "$pre.err" || { cat "$pre.err"; exit 3; }
glslangValidator -S comp "$pre" 2>&1 | sed "s#$pre#GLSL#"
exit "${PIPESTATUS[0]}"
