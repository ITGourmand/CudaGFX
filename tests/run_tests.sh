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
    if out=$("$CXX" "${CXXFLAGS_STRICT[@]}" -DCUDAGFX_NO_BGFX -Iinclude -Itests/include -c "$dir/$name.cpp" -o "$dir/host.o" 2>&1); then
        ok "$name: generated host code compiles (strict C++17, -Werror)"
    else
        ko "$name: generated host code" "$out"
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
if "$CXX" -std=c++17 -g -fsanitize=address,undefined -fno-sanitize-recover=all -DCUDAGFX_NO_BGFX -Iinclude tests/runtime/test_runtime_cpu.cpp -o "$TMP/rt_san" 2>"$TMP/rt.log"; then
    if out=$("$TMP/rt_san" 2>&1); then ok "test_runtime_cpu under ASan/UBSan"; else ko "test_runtime_cpu under ASan/UBSan" "$out"; fi
else
    skip "runtime under ASan/UBSan" "sanitizers unavailable"
fi
if "$CXX" -std=c++17 -fsyntax-only -DCUDAGFX_NO_BGFX -Iinclude tests/runtime/compile_fail_double_arg.cpp >/dev/null 2>&1; then ko "a double argument must be rejected at compile time"; else ok "Launch::arg(double) rejected at compile time"; fi
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
