#!/usr/bin/env bash
# CudaGFX end-to-end test suite.
#
# What it covers (one section per topic below):
#   - unit tests of the front-end (built and run through `make unit`)
#   - diagnostics: every tests/negative/*.cu must produce the expected code AT the expected position
#   - valid kernels (tests/positive + examples): transpilation, generated host code compiles, shaders validate,
#     shaders match the reference files in tests/golden/shaders
#   - full golden outputs for saxpy (shader, host code, stubs header, stubs header with --embed)
#   - host line-number preservation, #include handling and --depends
#   - command line, exit codes, shaderc plumbing (with a fake shaderc), unusual paths, idempotent writes
#   - runtime without a GPU (CUDAGFX_NO_BGFX), including sanitizers and multi-compiler portability
#   - CPU backend (--emit-cpu): every valid kernel compiles, the self-checking programs of tests/cpu run (also under
#     ThreadSanitizer), host layout checks
#   - GPU (examples/gpu_validation: Conway, particle systems, algorithms): built and run where a GPU is available, skipped
#     otherwise (CUDAGFX_GPU=0 to disable, CUDAGFX_GPU_RENDERERS to choose the renderers)
#   - executable dependencies, fuzzing, scalability
#
# How to add a test:
#   - a rejected construct   -> tests/negative/test_<name>.cu, ending with a line
#                               `// EXPECT <error|warning> <CGFXnnnn> <line>:<column>` (only the first diagnostic matters;
#                               the line is appended at the END of the file so it never shifts positions);
#   - a supported construct  -> tests/positive/<name>.cu, then run once with UPDATE_GOLDEN=1, REVIEW the new
#                               tests/golden/shaders/*.sc, and commit them.
#
# Environment variables (all optional):
#   CUDAGFX_BIN      executable to test (default: .build/linux/bin/cudagfx, built on demand)
#   SHADERC          real shaderc: validates the shaders for real (default: .deps/bgfx/bin/shaderc if it exists)
#   BGFX_SH_DIR      directory containing bgfx_compute.sh: enables GLSL validation through glslangValidator
#   BGFX_INC_DIR     directory containing bgfx/bgfx.h: type-checks the runtime against the real bgfx headers
#   UPDATE_GOLDEN=1  rewrite the reference files instead of comparing against them
#   VERBOSE=1        also print the tests that pass
#   FUZZ_RUNS=n      number of fuzzing iterations (default: 150)
#
# A test whose required tool is missing is reported as "skipped" (never counted as passed).
# Exit code: 0 if there is no failure, 1 otherwise (2 if the tool could not be built).
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 2
BIN="${CUDAGFX_BIN:-$ROOT/.build/linux/bin/cudagfx}"
CXX="${CXX:-g++}"
# Every test works in a private temporary directory (removed on exit): nothing is ever written in the repository,
# except the reference files when UPDATE_GOLDEN is set.
TMP=$(mktemp -d "${TMPDIR:-/tmp}/cudagfx-tests.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
SKIP=0
ok() { PASS=$((PASS + 1)); [ -n "${VERBOSE:-}" ] && echo "  ok       $1"; return 0; }
# ko <name> [details]: records a failure and shows the first lines of the details.
ko() {
    FAIL=$((FAIL + 1))
    echo "  FAILED   $1"
    [ -n "${2:-}" ] && printf '%s\n' "$2" | head -12 | sed 's/^/             /'
    return 0
}
skip() { SKIP=$((SKIP + 1)); echo "  skipped  $1 ($2)"; return 0; }
section() { echo "== $1"; }
# check <name> <command...>: passes if the command exits with 0 (its output is shown on failure).
check() {
    local name="$1"; shift
    local out
    if out=$("$@" 2>&1); then ok "$name"; else ko "$name" "$out"; fi
}

# --- optional tools ----------------------------------------------------------------------------------------------
SHADERC="${SHADERC:-}"
[ -z "$SHADERC" ] && [ -x "$ROOT/.deps/bgfx/bin/shaderc" ] && SHADERC="$ROOT/.deps/bgfx/bin/shaderc"
BGFX_SH_DIR="${BGFX_SH_DIR:-}"
[ -z "$BGFX_SH_DIR" ] && [ -f "$ROOT/.deps/bgfx/include/bgfx/bgfx_compute.sh" ] && BGFX_SH_DIR="$ROOT/.deps/bgfx/include/bgfx"
export BGFX_SH_DIR # read by tests/tools/validate_glsl.sh
BGFX_INC_DIR="${BGFX_INC_DIR:-}"
[ -z "$BGFX_INC_DIR" ] && [ -f "$ROOT/.deps/bgfx/include/bgfx/bgfx.h" ] && BGFX_INC_DIR="$ROOT/.deps/bgfx/include"

# Flags used to compile generated code: strict, so that the tool never emits warnings of its own.
# -Wunused-const-variable matters for clang users (see the [[maybe_unused]] rewrite in HostEmitter.cpp).
CXXFLAGS_STRICT=(-std=c++17 -Wall -Wextra -Wpedantic -Wunused-const-variable -Werror)

if [ ! -x "$BIN" ]; then
    echo "Building the tool..."
    make -s linux >"$TMP/build.log" 2>&1 || { cat "$TMP/build.log"; echo "build failed"; exit 2; }
fi
[ -x "$BIN" ] || { echo "executable not found: $BIN"; exit 2; }

# validate_shader <file.sc>: real shaderc if available, otherwise glslang.
# Returns 0 if valid, 1 if invalid (details on stdout), 77 if no validator is available.
validate_shader() {
    local sc="$1" out
    if [ -n "$SHADERC" ]; then
        local inc=()
        [ -n "$BGFX_SH_DIR" ] && inc=(-i "$BGFX_SH_DIR")
        for target in linux:spirv linux:430; do
            if ! out=$("$SHADERC" -f "$sc" -o "$TMP/shaderc.out" --type compute --platform "${target%%:*}" --profile "${target##*:}" "${inc[@]+"${inc[@]}"}" 2>&1); then
                echo "$out"
                return 1
            fi
        done
        return 0
    fi
    out=$(tests/tools/validate_glsl.sh "$sc" 2>&1)
    local rc=$?
    [ "$rc" -ne 0 ] && [ "$rc" -ne 77 ] && echo "$out"
    return "$rc"
}

# ===================================================================================================================
section "front-end unit tests"
if command -v make >/dev/null; then
    if out=$(make -s TARGET_OS=linux unit 2>&1); then ok "unit_tests ($(echo "$out" | tail -1))"; else ko "unit_tests" "$(echo "$out" | tail -15)"; fi
else
    skip "unit_tests" "make not found"
fi

# ===================================================================================================================
section "diagnostics: code AND position (tests/negative)"
for f in tests/negative/*.cu; do
    name=$(basename "$f" .cu)
    expects=$(grep -E '^// EXPECT ' "$f" || true)
    if [ -z "$expects" ]; then ko "$name" "no line '// EXPECT <error|warning> <CODE> <line:column>'"; continue; fi
    rm -f "$TMP/neg.sc" "$TMP/neg.cpp"
    out=$("$BIN" "$f" -o "$TMP/neg.cpp" --shader-out "$TMP/neg.sc" 2>&1)
    rc=$?
    good=1
    wantErr=0
    msg=""
    while read -r _ _ level code pos; do
        [ "$level" = error ] && wantErr=1
        if ! grep -qE ":${pos}: ${level}: ${code}:" <<<"$out"; then
            good=0
            msg="expected: ${level} ${code} at ${pos}"$'\n'"got:"$'\n'"$(grep -E '^[^ ].*: (error|warning):' <<<"$out" | head -4)"
        fi
    done <<<"$expects"
    # An internal error (CGFX0100) or a sanitizer report must never pass for a normal diagnostic.
    if grep -qE "CGFX0100|Sanitizer|runtime error:" <<<"$out"; then good=0; msg="internal error or sanitizer report"$'\n'"$out"; fi
    if [ "$wantErr" -eq 1 ] && [ "$rc" -ne 1 ]; then good=0; msg="exit code $rc instead of 1"; fi
    if [ "$wantErr" -eq 0 ] && [ "$rc" -ne 0 ]; then good=0; msg="exit code $rc instead of 0 (warning only)"; fi
    if [ "$wantErr" -eq 1 ] && { [ -e "$TMP/neg.sc" ] || [ -e "$TMP/neg.cpp" ]; }; then good=0; msg="files were written despite the error"; fi
    if [ "$good" -eq 1 ]; then ok "$name"; else ko "$name" "$msg"; fi
done

# ===================================================================================================================
section "valid kernels: transpilation, compilable host code, valid shaders (tests/positive and examples)"
mkdir -p tests/golden/shaders "$TMP/pos"
shader_skip_reason=""
for f in tests/positive/*.cu examples/*/*.cu; do
    name=$(basename "$f" .cu)
    dir="$TMP/pos/$name"
    mkdir -p "$dir"
    if ! out=$("$BIN" "$f" -o "$dir/$name.cpp" --shader-dir "$dir" 2>&1); then ko "$name: transpilation" "$out"; continue; fi
    if grep -qE "CGFX0100|Sanitizer|runtime error:" <<<"$out"; then ko "$name: internal error or sanitizer report" "$out"; continue; fi
    if out=$("$CXX" "${CXXFLAGS_STRICT[@]}" -DCUDAGFX_NO_BGFX -Iinclude -Itests/include -I"$(dirname "$f")" -c "$dir/$name.cpp" -o "$dir/host.o" 2>&1); then
        ok "$name: generated host code compiles (strict C++17, -Werror)"
    else
        ko "$name: generated host code" "$out"
    fi
    # The CPU implementation (--emit-cpu) of every valid kernel must compile too, warning-free.
    if out=$("$BIN" "$f" -o "$dir/${name}_cpu_host.cpp" --emit-cpu --shader-dir "$dir" 2>&1) &&
       out=$("$CXX" "${CXXFLAGS_STRICT[@]}" -DCUDAGFX_NO_BGFX -Iinclude -Itests/include -I"$(dirname "$f")" -fsyntax-only "$dir/${name}_cpu_host.cpp" 2>&1); then
        ok "$name: --emit-cpu output compiles (strict C++17, -Werror)"
    else
        ko "$name: --emit-cpu output" "$out"
    fi
    for sc in "$dir"/cs_*.sc; do
        base=$(basename "$sc")
        validate_shader "$sc" >"$TMP/vout" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then ok "$name: $base is valid"
        elif [ "$rc" -eq 77 ]; then shader_skip_reason="neither shaderc nor (glslangValidator + BGFX_SH_DIR)"
        else ko "$name: $base is invalid" "$(cat "$TMP/vout")"; fi
        # Reference files: tests/golden/shaders/<shader>.sc (one per generated shader).
        if [ -n "${UPDATE_GOLDEN:-}" ]; then
            cp "$sc" "tests/golden/shaders/$base"
        elif [ -f "tests/golden/shaders/$base" ]; then
            if diff -u "tests/golden/shaders/$base" "$sc" >"$TMP/gdiff" 2>&1; then ok "$name: $base matches the reference"; else ko "$name: $base differs from the reference (UPDATE_GOLDEN=1 to accept)" "$(cat "$TMP/gdiff")"; fi
        else
            ko "$name: $base has no reference" "run with UPDATE_GOLDEN=1 after reviewing the output"
        fi
    done
done
[ -n "$shader_skip_reason" ] && skip "shader validation" "$shader_skip_reason"

# ===================================================================================================================
section "golden: complete saxpy (shader, host code, stubs)"
mkdir -p "$TMP/g" "$TMP/g_embed"
"$BIN" tests/golden/saxpy/saxpy.cu -o "$TMP/g/saxpy_gen.cpp" --shader-dir "$TMP/g" >"$TMP/g.log" 2>&1 || ko "golden: transpilation" "$(cat "$TMP/g.log")"
# Same program through the whole shaderc/--embed pipeline, with a fake shaderc (see tests/tools/fake_shaderc.sh).
"$BIN" tests/golden/saxpy/saxpy.cu -o "$TMP/g_embed/saxpy_gen.cpp" --shader-dir "$TMP/g_embed" \
    --shaderc "$ROOT/tests/tools/fake_shaderc.sh" --embed --target linux:spirv --target windows:s_5_0 --target windows:s_6_0 >"$TMP/g.log" 2>&1 \
    || ko "golden: transpilation with --embed" "$(cat "$TMP/g.log")"
compare_golden() { # <expected> <actual> <name>
    if [ -n "${UPDATE_GOLDEN:-}" ]; then cp "$2" "$1"; return; fi
    if diff -u "$1" "$2" >"$TMP/gdiff" 2>&1; then ok "$3"; else ko "$3 (UPDATE_GOLDEN=1 to accept)" "$(cat "$TMP/gdiff")"; fi
}
compare_golden tests/golden/saxpy/cs_saxpy_256.sc "$TMP/g/cs_saxpy_256.sc" "golden: cs_saxpy_256.sc"
compare_golden tests/golden/saxpy/saxpy_gen.cpp "$TMP/g/saxpy_gen.cpp" "golden: saxpy_gen.cpp"
compare_golden tests/golden/saxpy/saxpy_gen_kernels.h "$TMP/g/saxpy_gen_kernels.h" "golden: saxpy_gen_kernels.h"
compare_golden tests/golden/saxpy/saxpy_gen_kernels_embed.h "$TMP/g_embed/saxpy_gen_kernels.h" "golden: saxpy_gen_kernels.h (--embed, 3 targets)"
for backend in spv dxbc dxil; do
    [ -f "$TMP/g_embed/cs_saxpy_256_$backend.h" ] && ok "--embed: cs_saxpy_256_$backend.h produced" || ko "--embed: cs_saxpy_256_$backend.h missing"
done

# ===================================================================================================================
section "host line-number preservation (C7)"
mkdir -p "$TMP/lines"
if "$BIN" tests/preserve/lines.cu -o "$TMP/lines/lines.cpp" --shader-dir "$TMP/lines" >"$TMP/l.log" 2>&1; then
    # tests/preserve/lines.cu ends some lines with a `// L<n>` marker, where <n> is that line's number:
    # after the rewrite every surviving marker must still sit on its own line number.
    bad=$(awk '{ if (match($0, /\/\/ L[0-9]+$/)) { n = substr($0, RSTART + 4) + 0; if (n != NR) printf "marker L%d on line %d\n", n, NR } }' "$TMP/lines/lines.cpp")
    found=$(grep -cE '// L[0-9]+$' "$TMP/lines/lines.cpp")
    [ -z "$bad" ] && ok "unmodified lines keep their line number" || ko "line numbering shifted" "$bad"
    [ "$found" -ge 14 ] && ok "markers found ($found)" || ko "markers lost" "$found found, at least 14 expected"
    [ "$(wc -l <tests/preserve/lines.cu)" -eq "$(wc -l <"$TMP/lines/lines.cpp")" ] && ok "same number of lines" || ko "different number of lines"
    check "lines.cpp compiles" "$CXX" "${CXXFLAGS_STRICT[@]}" -DCUDAGFX_NO_BGFX -Iinclude -c "$TMP/lines/lines.cpp" -o "$TMP/lines/lines.o"
    # Without a CUDA #include in the original: the include is added, followed by `#line 1` so that the host compiler's
    # line numbers still match the original file.
    printf '__global__ void k(float* y) { y[0] = 1.f; }\nint main() { return 0; } // end\n' >"$TMP/lines/noinc.cu"
    "$BIN" "$TMP/lines/noinc.cu" -o "$TMP/lines/noinc.cpp" --shader-dir "$TMP/lines" >/dev/null 2>&1
    if head -2 "$TMP/lines/noinc.cpp" | grep -q '^#line 1 "noinc.cu"$'; then ok "include added together with #line 1"; else ko "include missing: no #line 1" "$(head -3 "$TMP/lines/noinc.cpp")"; fi
else
    ko "transpilation of lines.cu" "$(cat "$TMP/l.log")"
fi

# ===================================================================================================================
section "#include \"...\" and the dependency file"
mkdir -p "$TMP/inc"
if "$BIN" tests/include/main.cu -I tests/include -o "$TMP/inc/main.cpp" --shader-dir "$TMP/inc" --depends "$TMP/inc/main.d" >"$TMP/i.log" 2>&1; then
    grep -q "scaled" "$TMP/inc/cs_offset_scale_256.sc" && grep -q "3.0" "$TMP/inc/cs_offset_scale_256.sc" && ok "function and macro of a user header are carried into the shader" || ko "header content missing from the shader" "$(cat "$TMP/inc/cs_offset_scale_256.sc")"
    grep -q "helper.cuh" "$TMP/inc/main.d" && ok "--depends lists the followed header" || ko "--depends incomplete" "$(cat "$TMP/inc/main.d")"
    grep -q '#include "helper.cuh"' "$TMP/inc/main.cpp" && ok "the user include is kept in the host code" || ko "user include lost"
    check "host code with a header compiles" "$CXX" "${CXXFLAGS_STRICT[@]}" -DCUDAGFX_NO_BGFX -Iinclude -Itests/include -c "$TMP/inc/main.cpp" -o "$TMP/inc/main.o"
else
    ko "transpilation with -I" "$(cat "$TMP/i.log")"
fi
# Without access to the header (file copied elsewhere, no -I): the functions it defines are unknown.
mkdir -p "$TMP/inc2"
cp tests/include/main.cu "$TMP/inc2/main.cu"
out=$("$BIN" "$TMP/inc2/main.cu" -o "$TMP/inc2/x.cpp" --shader-dir "$TMP/inc2" 2>&1)
if [ $? -eq 1 ] && grep -q "CGFX0010" <<<"$out"; then ok "header not found: unknown identifiers reported (CGFX0010)"; else ko "missing header not reported" "$out"; fi

# ===================================================================================================================
section "command line, shaderc, exit codes"
FAKE="$ROOT/tests/tools/fake_shaderc.sh"
if [ "$("$BIN" --version 2>&1)" = "cudagfx $(cat VERSION)" ]; then ok "--version prints the VERSION file"; else ko "--version must print 'cudagfx $(cat VERSION)'" "$("$BIN" --version 2>&1)"; fi
"$BIN" >/dev/null 2>&1; [ $? -eq 2 ] && ok "no argument: exit code 2" || ko "no argument: exit code 2 expected"
"$BIN" /nonexistent.cu >/dev/null 2>&1; [ $? -eq 2 ] && ok "missing file: exit code 2" || ko "missing file: exit code 2 expected"
"$BIN" tests/golden/saxpy/saxpy.cu --bogus >/dev/null 2>&1; [ $? -eq 2 ] && ok "unknown option: exit code 2" || ko "unknown option: exit code 2 expected"
"$BIN" tests/golden/saxpy/saxpy.cu --embed --shader-dir "$TMP/cli" >/dev/null 2>&1; [ $? -eq 2 ] && ok "--embed without --shaderc: exit code 2" || ko "--embed without --shaderc: exit code 2 expected"
"$BIN" tests/golden/saxpy/saxpy.cu --target linux:pssl --shader-dir "$TMP/cli" >/dev/null 2>&1; [ $? -eq 2 ] && ok "unknown profile: exit code 2" || ko "unknown profile: exit code 2 expected"
FAKE_SHADERC_FAIL=1 "$BIN" tests/golden/saxpy/saxpy.cu --shader-dir "$TMP/cli" --shaderc "$FAKE" >"$TMP/c.log" 2>&1
[ $? -eq 3 ] && grep -q "shaderc failed" "$TMP/c.log" && ok "shaderc failure: exit code 3 and message" || ko "shaderc failure: exit code 3 expected" "$(cat "$TMP/c.log")"
echo '#define' >"$TMP/cli_bad.cu"
"$BIN" "$TMP/cli_bad.cu" --shader-dir "$TMP/cli" >/dev/null 2>&1; [ $? -eq 1 ] && ok "source error: exit code 1" || ko "source error: exit code 1 expected"

# Paths with spaces and an apostrophe: passed unchanged to shaderc; directories are created on demand.
weird="$TMP/directory with spaces and an apostrophe's/gen"
if "$BIN" tests/golden/saxpy/saxpy.cu -o "$TMP/cli/o.cpp" --shader-dir "$weird" --shaderc "$FAKE" -i "$TMP/inc dir" --embed >"$TMP/c.log" 2>&1; then
    [ -f "$weird/cs_saxpy_256_spv.h" ] && ok "output directory created, path with spaces/apostrophe" || ko "output missing" "$(ls "$weird" 2>&1)"
    grep -q -- "-i $TMP/inc dir" "$weird/cs_saxpy_256_spv.h.args" 2>/dev/null && ok "-i passed to shaderc unaltered" || ko "-i badly passed" "$(cat "$weird/cs_saxpy_256_spv.h.args" 2>&1)"
else
    ko "unusual paths" "$(cat "$TMP/c.log")"
fi
# Idempotent writes: a second identical run must not touch the files (keeps incremental builds fast).
"$BIN" tests/golden/saxpy/saxpy.cu -o "$TMP/idem/a.cpp" --shader-dir "$TMP/idem" >/dev/null 2>&1
m1=$(stat -c %Y.%y "$TMP/idem/a.cpp" 2>/dev/null)
sleep 1.1
"$BIN" tests/golden/saxpy/saxpy.cu -o "$TMP/idem/a.cpp" --shader-dir "$TMP/idem" >/dev/null 2>&1
m2=$(stat -c %Y.%y "$TMP/idem/a.cpp" 2>/dev/null)
[ -n "$m1" ] && [ "$m1" = "$m2" ] && ok "identical regeneration: files not rewritten" || ko "file rewritten although identical"
# --no-line-preserve: the host code is no longer padded with newlines
"$BIN" tests/preserve/lines.cu -o "$TMP/cli/nolp.cpp" --shader-dir "$TMP/cli" --no-line-preserve >/dev/null 2>&1
if [ -f "$TMP/cli/nolp.cpp" ] && [ "$(wc -l <"$TMP/cli/nolp.cpp")" -lt "$(wc -l <tests/preserve/lines.cu)" ]; then ok "--no-line-preserve removes the line padding"; else ko "--no-line-preserve has no effect"; fi
# --dump-ir
if "$BIN" tests/golden/saxpy/saxpy.cu --dump-ir --shader-dir "$TMP/cli" 2>/dev/null | grep -q '"launch_sites"'; then ok "--dump-ir produces a JSON summary"; else ko "--dump-ir"; fi
# Several kernels: '%' pattern and block-size variants
mkdir -p "$TMP/multi"
cat >"$TMP/multi/m.cu" <<'CUEOF'
__global__ void a(float* y) { y[threadIdx.x] = 1.f; }
__global__ void b(float* y) { y[threadIdx.x] = 2.f; }
void h(float* d) { a<<<1, 64>>>(d); a<<<1, 128>>>(d); b<<<1, 32>>>(d); }
CUEOF
"$BIN" "$TMP/multi/m.cu" -o "$TMP/multi/m.cpp" --shader-out "$TMP/multi/cs_%.sc" >"$TMP/c.log" 2>&1
if [ -f "$TMP/multi/cs_a_64.sc" ] && [ -f "$TMP/multi/cs_a_128.sc" ] && [ -f "$TMP/multi/cs_b.sc" ]; then ok "--shader-out with % and block-size variants"; else ko "% pattern or variants" "$(ls "$TMP/multi")"$'\n'"$(cat "$TMP/c.log")"; fi

# ===================================================================================================================
section "runtime (no GPU)"
# test_runtime_cpu.cpp exercises memory, copies, bounds and launch validation without bgfx.
if "$CXX" "${CXXFLAGS_STRICT[@]}" -fno-exceptions -fno-rtti -DCUDAGFX_NO_BGFX -Iinclude tests/runtime/test_runtime_cpu.cpp -o "$TMP/rt_cpu" 2>"$TMP/rt.log"; then
    ok "runtime compiles without exceptions or RTTI (C12)"
    if out=$("$TMP/rt_cpu" 2>&1); then ok "test_runtime_cpu: $out"; else ko "test_runtime_cpu" "$out"; fi
else
    ko "runtime compilation" "$(cat "$TMP/rt.log")"
fi
# The CPU executor exists in two forms (fibers on Linux x86-64, OS threads elsewhere): test both everywhere.
if "$CXX" "${CXXFLAGS_STRICT[@]}" -fno-exceptions -fno-rtti -DCUDAGFX_NO_BGFX -DCUDAGFX_CPU_THREADS_ONLY -Iinclude tests/runtime/test_runtime_cpu.cpp -o "$TMP/rt_thr" -pthread 2>"$TMP/rt.log"; then
    if out=$("$TMP/rt_thr" 2>&1); then ok "test_runtime_cpu with the OS-thread executor: $out"; else ko "test_runtime_cpu (OS-thread executor)" "$out"; fi
else
    ko "runtime compilation (OS-thread executor)" "$(cat "$TMP/rt.log")"
fi
if "$CXX" -std=c++17 -g -fsanitize=address,undefined -fno-sanitize-recover=all -DCUDAGFX_NO_BGFX -Iinclude tests/runtime/test_runtime_cpu.cpp -o "$TMP/rt_san" 2>"$TMP/rt.log"; then
    if out=$("$TMP/rt_san" 2>&1); then ok "test_runtime_cpu under ASan/UBSan"; else ko "test_runtime_cpu under ASan/UBSan" "$out"; fi
else
    skip "runtime under ASan/UBSan" "sanitizers unavailable"
fi
if "$CXX" -std=c++17 -fsyntax-only -DCUDAGFX_NO_BGFX -Iinclude tests/runtime/compile_fail_double_arg.cpp >/dev/null 2>&1; then ko "a double argument must be rejected at compile time"; else ok "Launch::arg(double) rejected at compile time"; fi
if out=$("$CXX" -std=c++17 -fsyntax-only -Iinclude -Itests/runtime/old_bgfx tests/runtime/test_runtime_cpu.cpp 2>&1); then
    ko "a bgfx older than BGFX_API_VERSION 161 must be rejected at compile time"
elif grep -q "BGFX_API_VERSION >= 161" <<<"$out"; then
    ok "a bgfx older than BGFX_API_VERSION 161 is rejected with an explicit message"
else
    ko "an old bgfx is rejected, but without the explicit BGFX_API_VERSION message" "$out"
fi
# Runtime portability: every available compiler (gcc, clang, mingw) must accept it under -Werror.
for cxx in g++ clang++ x86_64-w64-mingw32-g++; do
    command -v "$cxx" >/dev/null || continue
    check "runtime accepted by $cxx (without bgfx, -Werror)" "$cxx" "${CXXFLAGS_STRICT[@]}" -fno-exceptions -fno-rtti -fsyntax-only -DCUDAGFX_NO_BGFX -Iinclude tests/runtime/test_runtime_cpu.cpp
    [ -n "$BGFX_INC_DIR" ] && check "runtime accepted by $cxx (against bgfx, -Werror)" "$cxx" "${CXXFLAGS_STRICT[@]}" -fno-exceptions -fno-rtti -fsyntax-only -Iinclude -I"$BGFX_INC_DIR" tests/runtime/test_runtime_cpu.cpp
done
if [ -n "$BGFX_INC_DIR" ]; then
    check "runtime against the real bgfx headers (types and signatures)" "$CXX" "${CXXFLAGS_STRICT[@]}" -fno-exceptions -fno-rtti -fsyntax-only -Iinclude -I"$BGFX_INC_DIR" tests/runtime/test_runtime_cpu.cpp
    check "handwritten example against bgfx" "$CXX" "${CXXFLAGS_STRICT[@]}" -fsyntax-only -Iinclude -I"$BGFX_INC_DIR" -Iexamples/handwritten examples/handwritten/main_handwritten.cpp
    if [ -f "$TMP/pos/param_packing/param_packing.cpp" ]; then
        check "generated stub against bgfx" "$CXX" "${CXXFLAGS_STRICT[@]}" -fsyntax-only -Iinclude -I"$BGFX_INC_DIR" "$TMP/pos/param_packing/param_packing.cpp"
    fi
else
    skip "runtime against bgfx" "BGFX_INC_DIR not set and .deps/bgfx/include missing"
fi

# E2E CPU backend execution (--emit-cpu)
mkdir -p "$TMP/e2e_cpu"
cat << 'CUEOF' > "$TMP/e2e_cpu/saxpy.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void saxpy_cpu(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

int main() {
    cudagfx::setExecutionMode(cudagfx::ExecutionMode::CPU);
    const int N = 64;
    std::vector<float> h_x(N, 2.0f), h_y(N, 1.0f);
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    saxpy_cpu<<<1, 64>>>(d_y, d_x, 3.0f, N);

    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);

    for (int i = 0; i < N; ++i) {
        if (h_y[i] != 7.0f) {
            std::printf("MISMATCH at %d: expected 7.0, got %f\n", i, h_y[i]);
            return 1;
        }
    }
    std::printf("OK: saxpy CPU execution passed (result = 7.0)\n");
    return 0;
}
CUEOF
if "$BIN" "$TMP/e2e_cpu/saxpy.cu" -o "$TMP/e2e_cpu/saxpy.cpp" --emit-cpu --shader-dir "$TMP/e2e_cpu" >"$TMP/e2e_cpu/transpile.log" 2>&1; then
    if "$CXX" "${CXXFLAGS_STRICT[@]}" -DCUDAGFX_NO_BGFX -Iinclude -lpthread "$TMP/e2e_cpu/saxpy.cpp" -o "$TMP/e2e_cpu/saxpy_bin" 2>"$TMP/e2e_cpu/comp.log"; then
        if out=$("$TMP/e2e_cpu/saxpy_bin" 2>&1); then
            ok "--emit-cpu end-to-end execution: $out"
        else
            ko "--emit-cpu execution failed" "$out"
        fi
    else
        ko "--emit-cpu host compilation failed" "$(cat "$TMP/e2e_cpu/comp.log")"
    fi
else
    ko "--emit-cpu transpilation failed" "$(cat "$TMP/e2e_cpu/transpile.log")"
fi

# ===================================================================================================================
section "CPU backend (--emit-cpu): execution, layout checks, command line"
mkdir -p "$TMP/cpu"
# tests/cpu/*.cu are complete programs that compare the result of their kernels with a host reference.
for f in tests/cpu/*.cu; do
    name=$(basename "$f" .cu)
    dir="$TMP/cpu/$name"
    mkdir -p "$dir"
    if ! out=$("$BIN" "$f" -o "$dir/$name.cpp" --emit-cpu --shader-dir "$dir" 2>&1); then ko "$name: transpilation" "$out"; continue; fi
    if out=$("$CXX" "${CXXFLAGS_STRICT[@]}" -O1 -DCUDAGFX_NO_BGFX -Iinclude "$dir/$name.cpp" -o "$dir/$name" -pthread 2>&1); then
        ok "$name: CPU code compiles (strict C++17, -Werror)"
    else
        ko "$name: CPU code does not compile" "$out"
        continue
    fi
    if out=$(timeout 120 "$dir/$name" 2>&1); then ok "$name: result matches the host reference"; else ko "$name: execution" "$out"; fi
    if "$CXX" "${CXXFLAGS_STRICT[@]}" -O1 -DCUDAGFX_NO_BGFX -DCUDAGFX_CPU_THREADS_ONLY -Iinclude "$dir/$name.cpp" -o "$dir/${name}_thr" -pthread 2>"$dir/thr.log"; then
        if out=$(timeout 300 "$dir/${name}_thr" 2>&1); then ok "$name: same result with the OS-thread executor"; else ko "$name: OS-thread executor" "$out"; fi
    else
        ko "$name: OS-thread executor does not compile" "$(cat "$dir/thr.log")"
    fi
    # Data races and lost wake-ups in the executor (barriers, shared memory reuse between blocks) show up here.
    if "$CXX" -std=c++17 -g -O1 -fsanitize=thread -DCUDAGFX_NO_BGFX -Iinclude "$dir/$name.cpp" -o "$dir/${name}_tsan" -pthread 2>"$dir/tsan.log"; then
        if out=$(timeout 300 "$dir/${name}_tsan" 2>&1); then ok "$name under ThreadSanitizer"; else ko "$name under ThreadSanitizer" "$out"; fi
    else
        skip "$name under ThreadSanitizer" "ThreadSanitizer unavailable"
    fi
done

# --emit-cpu needs -o (the host code includes the CPU header at its end); the help documents both options.
"$BIN" tests/cpu/cpu_reduce.cu --emit-cpu >/dev/null 2>&1; rc=$?
if [ "$rc" -eq 2 ]; then ok "--emit-cpu without -o is a usage error (exit 2)"; else ko "--emit-cpu without -o" "exit code $rc"; fi
if out=$("$BIN" --help 2>&1) && grep -q -- "--emit-cpu" <<<"$out" && grep -q -- "--cpu-header" <<<"$out"; then ok "--help documents --emit-cpu and --cpu-header"; else ko "--help must document --emit-cpu and --cpu-header" "$out"; fi

# --cpu-header chooses the path of the CPU header; the host code includes it (relative to its own directory).
mkdir -p "$TMP/cpu/custom/sub"
if "$BIN" tests/cpu/cpu_reduce.cu -o "$TMP/cpu/custom/host.cpp" --emit-cpu --cpu-header "$TMP/cpu/custom/sub/my_cpu.h" --shader-dir "$TMP/cpu/custom" >/dev/null 2>&1 &&
   [ -f "$TMP/cpu/custom/sub/my_cpu.h" ] && grep -q '#include "sub/my_cpu.h"' "$TMP/cpu/custom/host.cpp"; then
    ok "--cpu-header: header written where requested and included by the host code"
else
    ko "--cpu-header"
fi
# Without --emit-cpu nothing CPU-related is generated or included.
mkdir -p "$TMP/cpu/none"
"$BIN" tests/cpu/cpu_reduce.cu -o "$TMP/cpu/none/host.cpp" --shader-dir "$TMP/cpu/none" >/dev/null 2>&1
if [ ! -e "$TMP/cpu/none/host_cpu.h" ] && ! grep -q "_cpu.h" "$TMP/cpu/none/host.cpp"; then ok "without --emit-cpu: no CPU header"; else ko "without --emit-cpu: a CPU header was generated"; fi

# --suppress silences a warning (never an error) and rejects an unknown code.
mkdir -p "$TMP/cpu/suppress"
cat >"$TMP/cpu/suppress/div.cu" <<'DVEOF'
__global__ void k(int* y, int d) { y[0] = y[0] / d; }
DVEOF
out=$("$BIN" "$TMP/cpu/suppress/div.cu" --shader-dir "$TMP/cpu/suppress" 2>&1)
if grep -q "CGFX0017" <<<"$out"; then ok "CGFX0017 is reported by default"; else ko "CGFX0017 must be reported by default" "$out"; fi
out=$("$BIN" "$TMP/cpu/suppress/div.cu" --shader-dir "$TMP/cpu/suppress" --suppress cgfx0017 2>&1)
if ! grep -q "CGFX0017" <<<"$out"; then ok "--suppress CGFX0017 silences the warning"; else ko "--suppress CGFX0017" "$out"; fi
"$BIN" "$TMP/cpu/suppress/div.cu" --shader-dir "$TMP/cpu/suppress" --suppress NOPE >/dev/null 2>&1; rc=$?
if [ "$rc" -eq 2 ]; then ok "--suppress with an unknown code is a usage error (exit 2)"; else ko "--suppress NOPE" "exit code $rc"; fi
out=$("$BIN" tests/negative/test_cgfx0021_offset.cu --suppress CGFX0021 --shader-dir "$TMP/cpu/suppress" 2>&1); rc=$?
if [ "$rc" -eq 1 ] && grep -q "CGFX0021" <<<"$out"; then ok "an error cannot be suppressed"; else ko "--suppress must not hide an error" "exit $rc: $out"; fi

# A division of two literals (`BLOCK / 2`, macros are expanded) cannot be negative: no CGFX0017.
cat >"$TMP/cpu/suppress/const_div.cu" <<'CDEOF'
#define BLOCK 256
__global__ void k(int* y, int d) { y[0] = BLOCK / 2 + BLOCK % 16 + y[1] / 2; }
CDEOF
out=$("$BIN" "$TMP/cpu/suppress/const_div.cu" --shader-dir "$TMP/cpu/suppress" 2>&1)
if [ "$(grep -c "CGFX0017" <<<"$out")" -eq 1 ]; then ok "CGFX0017 is not reported for a division of literals (only for y[1] / 2)"; else ko "CGFX0017 on constants" "$out"; fi

# examples/gpu_validation (Conway, particle systems, algorithms...): the CPU-only build of every suite is executed; the GPU
# half is type-checked against the real bgfx headers when they are available, and EXECUTED in the GPU section below.
mkdir -p "$TMP/cpu/gv"
gv_cpu_srcs=()
gv_ready=1
for f in examples/gpu_validation/*.cu; do
    n=$(basename "$f" .cu)
    if "$BIN" "$f" -o "$TMP/cpu/gv/$n.cpp" --emit-cpu --shader-dir "$TMP/cpu/gv" >"$TMP/cpu/gv/$n.log" 2>&1; then
        gv_cpu_srcs+=("$TMP/cpu/gv/$n.cpp")
    else
        ko "gpu_validation: $n transpilation" "$(cat "$TMP/cpu/gv/$n.log")"
        gv_ready=0
    fi
done
if [ "$gv_ready" -eq 1 ]; then
    if out=$("$CXX" "${CXXFLAGS_STRICT[@]}" -O2 -DCUDAGFX_NO_BGFX -DCUDAGFX_CPU_ONLY -Iinclude -Iexamples/gpu_validation \
            "${gv_cpu_srcs[@]}" examples/gpu_validation/main.cpp -o "$TMP/cpu/gv/gv_cpu" -pthread 2>&1); then
        if out=$(timeout 300 "$TMP/cpu/gv/gv_cpu" 2>&1); then
            ok "gpu_validation: every check passes on the CPU backend ($(grep -c '^PASS' <<<"$out") checks)"
        else
            ko "gpu_validation (CPU backend)" "$out"
        fi
    else
        ko "gpu_validation: CPU build" "$out"
    fi
    if [ -n "$BGFX_INC_DIR" ]; then
        for src in "${gv_cpu_srcs[@]}"; do
            check "gpu_validation: $(basename "$src") GPU path against the real bgfx headers" "$CXX" "${CXXFLAGS_STRICT[@]}" -fsyntax-only -Iinclude -I"$BGFX_INC_DIR" -Iexamples/gpu_validation "$src"
        done
        check "gpu_validation: main.cpp against the real bgfx headers" "$CXX" "${CXXFLAGS_STRICT[@]}" -fsyntax-only -Iinclude -I"$BGFX_INC_DIR" -Iexamples/gpu_validation examples/gpu_validation/main.cpp
    else
        skip "gpu_validation against bgfx" "BGFX_INC_DIR not set"
    fi
fi

# The generated host code static_assert's the layout the shaders were generated for: a struct that the C++ compiler lays
# out differently (here #pragma pack) must stop the build instead of silently feeding wrong data to the GPU.
mkdir -p "$TMP/cpu/pack"
cat >"$TMP/cpu/pack/pack.cu" <<'PKEOF'
#pragma pack(push, 1)
struct Packed {
    float4 a;
    float4 b;
};
#pragma pack(pop)
__global__ void k(Packed* p, int n) {
    int i = threadIdx.x;
    if (i < n) p[i].a.x = 1.0f;
}
PKEOF
if "$BIN" "$TMP/cpu/pack/pack.cu" -o "$TMP/cpu/pack/pack.cpp" --shader-dir "$TMP/cpu/pack" >"$TMP/cpu/pack/t.log" 2>&1; then
    if out=$("$CXX" -std=c++17 -fsyntax-only -DCUDAGFX_NO_BGFX -Iinclude "$TMP/cpu/pack/pack.cpp" 2>&1); then
        ko "a packed struct must be rejected by the generated layout checks"
    elif grep -q "differs from the layout of the generated shaders" <<<"$out"; then
        ok "host layout checks: a packed struct stops the build"
    else
        ko "host layout checks: unexpected failure" "$out"
    fi
else
    ko "layout check test: transpilation" "$(cat "$TMP/cpu/pack/t.log")"
fi

# ===================================================================================================================
section "GPU (executed where bgfx and a GPU are available, otherwise skipped)"
# Needs: a working shaderc (SHADERC=, or .deps/bgfx/bin/shaderc), the bgfx headers and static libraries
# (BGFX_LIB_DIR=, or .deps/bgfx/lib64 | lib), and a renderer able to run compute shaders without a window.
#   CUDAGFX_GPU=0                           never run the GPU tests
#   CUDAGFX_GPU_RENDERERS="vulkan opengl"   the renderers to try (default: vulkan opengl)
# gpu_validation exits with 2 when bgfx cannot be initialised on a renderer: that renderer is then skipped.
gpu_reason=""
BGFX_LIB_DIR="${BGFX_LIB_DIR:-}"
if [ -z "$BGFX_LIB_DIR" ]; then
    for d in "$ROOT/.deps/bgfx/lib64" "$ROOT/.deps/bgfx/lib"; do
        [ -f "$d/libbgfx.a" ] && BGFX_LIB_DIR="$d" && break
    done
fi
if [ "${CUDAGFX_GPU:-auto}" = "0" ]; then gpu_reason="CUDAGFX_GPU=0"
elif [ -z "$SHADERC" ]; then gpu_reason="no shaderc (set SHADERC or provide .deps/bgfx/bin/shaderc)"
elif [ -z "$BGFX_INC_DIR" ]; then gpu_reason="no bgfx headers (set BGFX_INC_DIR or provide .deps/bgfx/include)"
elif [ -z "$BGFX_LIB_DIR" ] || [ ! -f "$BGFX_LIB_DIR/libbgfx.a" ]; then gpu_reason="no bgfx libraries (set BGFX_LIB_DIR or provide .deps/bgfx/lib64)"
fi

gpu_exe="$TMP/gpu/gpu_validation"
if [ -z "$gpu_reason" ]; then
    mkdir -p "$TMP/gpu"
    gpu_srcs=()
    for f in examples/gpu_validation/*.cu; do
        n=$(basename "$f" .cu)
        shader_inc=()
        [ -n "$BGFX_SH_DIR" ] && shader_inc=(-i "$BGFX_SH_DIR")
        if "$BIN" "$f" -o "$TMP/gpu/$n.cpp" --shader-dir "$TMP/gpu" --shaderc "$SHADERC" --embed --emit-cpu --target linux:spirv --target linux:430 \
               "${shader_inc[@]+"${shader_inc[@]}"}" >"$TMP/gpu/$n.log" 2>&1; then
            gpu_srcs+=("$TMP/gpu/$n.cpp")
        else
            gpu_reason="shaderc / transpilation of $n failed: $(grep -m1 -E 'error' "$TMP/gpu/$n.log")"
            break
        fi
    done
fi
if [ -z "$gpu_reason" ]; then
    # libbgfx loads OpenGL, X11 and Vulkan with dlopen: it needs no -lGL / -lX11 (which a Nix shell often cannot find).
    # The system libraries differ between systems (-lrt / -ldl are stubs or absent on a recent glibc): try a few lists,
    # in the C locale so that the errors that are reported are the ones of the linker, whatever the user's language.
    gpu_linked=0
    gpu_first_error=""
    for syslibs in "-lpthread -ldl -lrt" "-lpthread -ldl" "-lpthread" "" "-lpthread -ldl -lrt -lGL -lX11"; do
        # shellcheck disable=SC2086
        if out=$(LC_ALL=C "$CXX" -std=c++17 -O1 -Iinclude -I"$BGFX_INC_DIR" -Iexamples/gpu_validation -I"$TMP/gpu" "${gpu_srcs[@]}" examples/gpu_validation/main.cpp \
                -L"$BGFX_LIB_DIR" -lbgfx -lbimg -lbx $syslibs -o "$gpu_exe" 2>&1); then
            gpu_linked=1
            break
        fi
        [ -z "$gpu_first_error" ] && gpu_first_error="$out"
    done
    if [ "$gpu_linked" -eq 0 ]; then
        gpu_reason="cannot link with bgfx in $BGFX_LIB_DIR: $(grep -v '^collect2' <<<"$gpu_first_error" | head -4 | cut -c1-200 | tr '\n' ' ')"
    fi
fi
if [ -n "$gpu_reason" ]; then
    skip "GPU tests" "$gpu_reason"
else
    for renderer in ${CUDAGFX_GPU_RENDERERS:-vulkan opengl}; do
        out=$(timeout 600 "$gpu_exe" --renderer "$renderer" 2>&1)
        rc=$?
        case "$rc" in
            0) ok "GPU [$renderer]: every check passes ($(head -1 <<<"$out"); $(grep -c '^PASS.*\[GPU\]' <<<"$out") GPU checks)" ;;
            2) skip "GPU [$renderer]" "bgfx could not be initialised on this renderer: $(head -14 <<<"$out" | cut -c1-160 | tr '\n' '|')" ;;
            *) ko "GPU [$renderer]: exit code $rc" "$out" ;;
        esac
    done
fi

# ===================================================================================================================
section "executable dependencies (C13)"
if command -v ldd >/dev/null; then
    deps=$(ldd "$BIN" 2>&1 || true)
    if grep -qE 'libstdc\+\+|libgcc_s|libpython' <<<"$deps"; then ko "unexpected dynamic dependencies" "$deps"; else ok "no libstdc++/libgcc_s/python dependency"; fi
else
    skip "ldd" "ldd not found"
fi

# ===================================================================================================================
section "robustness: fuzzing (never a crash or a hang)"
# Two generators: random token soup, and mutations (truncation, deleted/duplicated line, swapped punctuation) of the
# real test programs. The only acceptable exit codes are 0 and 1, and no output may contain an internal error.
FUZZ_RUNS="${FUZZ_RUNS:-150}"
mkdir -p "$TMP/fuzz"
fuzz_fail=0
fuzz_run() { # <file> <name>
    local rc
    timeout 10 "$BIN" "$1" -o "$TMP/fuzz/out.cpp" --shader-dir "$TMP/fuzz" >"$TMP/fuzz/log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
        fuzz_fail=$((fuzz_fail + 1))
        if [ "$fuzz_fail" -le 2 ]; then ko "fuzz $2: exit code $rc" "$(head -c 400 "$1")"; fi
    elif grep -qE "CGFX0100|Sanitizer|runtime error:" "$TMP/fuzz/log"; then
        fuzz_fail=$((fuzz_fail + 1))
        if [ "$fuzz_fail" -le 2 ]; then ko "fuzz $2: internal error" "$(head -c 400 "$1")"; fi
    fi
}
vocab=('__global__' '__device__' '__shared__' 'void' 'float' 'int' 'unsigned' 'float4' 'bool' '*' '(' ')' '{' '}' '[' ']' ';' ',' ':' '?'
       '<<<' '>>>' '<' '>' 'k' 'x' 'y' 'n' '1' '2.f' '0x' '+' '-' '=' '==' '&' '&&' '|' '%' '/' '++' '#define A' '#if 1' '#else' '#endif'
       '#include "a.h"' '#' '##' '\' '/*' '*/' '//' '"' "'" 'return' 'if' 'else' 'for' 'while' 'do' 'switch' 'case' 'default' 'break'
       'threadIdx.x' 'blockIdx' 'gridDim' 'atomicAdd' 'sizeof' 'struct' 'template' 'const' '__launch_bounds__' '@' '.' '->' 'dim3' 'make_float4')
RANDOM=20260 # fixed seed: a failure is reproducible
for i in $(seq 1 "$FUZZ_RUNS"); do
    n=$((RANDOM % 50 + 5))
    : >"$TMP/fuzz/in.cu"
    for _ in $(seq 1 "$n"); do
        printf '%s' "${vocab[$((RANDOM % ${#vocab[@]}))]}" >>"$TMP/fuzz/in.cu"
        if [ $((RANDOM % 4)) -eq 0 ]; then printf '\n' >>"$TMP/fuzz/in.cu"; else printf ' ' >>"$TMP/fuzz/in.cu"; fi
    done
    fuzz_run "$TMP/fuzz/in.cu" "tokens #$i"
done
for f in tests/positive/*.cu examples/*/*.cu tests/include/main.cu tests/preserve/lines.cu; do
    size=$(wc -c <"$f")
    for m in 1 2 3 4; do
        case "$m" in
            1) head -c $((RANDOM % size)) "$f" >"$TMP/fuzz/m.cu" ;;                    # truncation
            2) sed "$((RANDOM % $(wc -l <"$f") + 1))d" "$f" >"$TMP/fuzz/m.cu" ;;      # deleted line
            3) sed "$((RANDOM % $(wc -l <"$f") + 1))p" "$f" >"$TMP/fuzz/m.cu" ;;      # duplicated line
            4) tr '{;' ';}' <"$f" >"$TMP/fuzz/m.cu" ;;                                # swapped punctuation
        esac
        fuzz_run "$TMP/fuzz/m.cu" "mutation $m of $(basename "$f")"
    done
done
[ "$fuzz_fail" -eq 0 ] && ok "fuzzing: $FUZZ_RUNS random inputs + mutations, no crash and no hang" || ko "fuzzing: $fuzz_fail problematic cases"

# ===================================================================================================================
section "scalability"
# 600 device functions + 600 kernels + 600 launches: guards against accidentally quadratic code in the tool.
{
    echo '#include <cuda_runtime.h>'
    for i in $(seq 1 600); do
        printf '__device__ float f%d(float v) { return v * %d.0f + 1.0f; }\n' "$i" "$i"
        printf '__global__ void k%d(float* y, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) y[i] = f%d(y[i]) + sinf(y[i]) * 0.5f;\n}\n' "$i" "$i"
    done
    echo 'void h(float* d) {'
    for i in $(seq 1 600); do printf '    k%d<<<4, 256>>>(d, 1024);\n' "$i"; done
    echo '}'
} >"$TMP/big.cu"
start=$(date +%s.%N)
if timeout 60 "$BIN" "$TMP/big.cu" -o "$TMP/big.cpp" --shader-dir "$TMP/bigout" >"$TMP/big.log" 2>&1; then
    elapsed=$(awk -v s="$start" -v e="$(date +%s.%N)" 'BEGIN { printf "%.2f", e - s }')
    ok "1200 functions, 600 shaders, 600 launches: ${elapsed}s ($(wc -l <"$TMP/big.cu") lines)"
    if awk -v t="$elapsed" 'BEGIN { exit !(t < 10) }'; then ok "transpilation time < 10 s"; else ko "transpilation too slow" "${elapsed}s"; fi
else
    ko "large file" "$(tail -3 "$TMP/big.log")"
fi

# ===================================================================================================================
echo
echo "Result: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
