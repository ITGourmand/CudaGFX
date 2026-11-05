#!/usr/bin/env bash
# Builds bgfx, bx, bimg and shaderc from the sources of bgfx.cmake and installs them in a prefix laid out as
# tests/run_tests.sh expects: <prefix>/bin/shaderc, <prefix>/include/{bgfx,bx,bimg}, <prefix>/lib/lib{bgfx,bimg,bx}.a
#
#   build-bgfx.sh <prefix> [<bgfx.cmake commit | tag | branch>]      (default: the latest commit of bgfx.cmake)
#
# CONFIGURE_ONLY=1 stops after the configuration (a quick check that the revision and the options are fine).
# The first build is long (shaderc includes the WebGPU compiler of Dawn): the workflows cache the prefix.
set -euo pipefail

prefix=${1:?usage: build-bgfx.sh <prefix> [revision]}
ref=${2:-}
work=${RUNNER_TEMP:-/tmp}/bgfx.cmake

mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

rm -rf "$work" "$work-build"
git init -q "$work"
if [ -z "$ref" ]; then ref=HEAD; fi
git -C "$work" fetch -q --depth 1 https://github.com/bkaradzic/bgfx.cmake.git "$ref"
git -C "$work" checkout -q FETCH_HEAD
echo "bgfx.cmake $(git -C "$work" rev-parse HEAD) ($(git -C "$work" log -1 --format=%cd))"

cmake -S "$work" -B "$work-build" -DCMAKE_BUILD_TYPE=Release \
  -DBGFX_BUILD_TOOLS=ON -DBGFX_BUILD_TOOLS_SHADER=ON -DBGFX_BUILD_TOOLS_TEXTURE=OFF -DBGFX_BUILD_TOOLS_GEOMETRY=OFF \
  -DBGFX_BUILD_EXAMPLES=OFF -DBGFX_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib

if [ "${CONFIGURE_ONLY:-0}" = 1 ]; then
  echo "configured only (CONFIGURE_ONLY=1)"
  exit 0
fi

jobs=$(nproc 2>/dev/null || echo 2)
cmake --build "$work-build" --target install -j"$jobs"

test -x "$prefix/bin/shaderc"
test -f "$prefix/lib/libbgfx.a"
test -f "$prefix/include/bgfx/bgfx_compute.sh"
echo "installed in $prefix:"
ls "$prefix/bin" "$prefix/lib" | head -n 20
