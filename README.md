# CudaGFX

A **CUDA → bgfx compute shader transpiler**. From a `.cu` file, `cudagfx` produces:

* one **compute shader (`.sc`)** per kernel and per block size, ready to be compiled by `shaderc` for Vulkan, OpenGL,
  OpenGL ES, Direct3D 11/12, Metal and WebGPU;
* the **host code as standard C++17** (the `.cu` file without its kernels, with every `k<<<g, b>>>(…)` rewritten as a
  stub call) — lines that are not modified keep their line number;
* a **stubs header** that binds the arguments and the embedded shader binaries to the runtime `cudagfx_rt.h`, which
  emulates `cudaMalloc` / `cudaMemcpy` / `cudaMemset` / `cudaFree` on top of bgfx.

```
 prog.cu ──► cudagfx ──┬─► cs_<kernel>_<block>.sc ──► shaderc ──► cs_<kernel>_<block>_<backend>.h
                       ├─► prog_gen.cpp          (host code, C++17)
                       └─► prog_gen_kernels.h    (stubs + shader tables)   + cudagfx_rt.h (runtime)
```

> **Guiding principle: never emit a wrong shader silently.** Any construct outside the supported subset produces a
> precise `CGFXnnnn` diagnostic (file, line, column); every implicit conversion becomes an explicit cast in the
> shader; runtime errors carry a `CGFXRnnn` code.

## Build and test

```sh
make              # tool in .build/linux/bin/cudagfx (and Windows through mingw: make windows)
make test         # unit tests + full suite (tests/run_tests.sh)
nix develop       # (NixOS) shell with make, cmake, gdb, mingw, glslang, shellcheck
```

The `tests/run_tests.sh` suite only needs `bash`, `make` and a C++17 compiler. It additionally uses, when present: a real
`shaderc` (`SHADERC=…` or `.deps/bgfx/bin/shaderc`), `glslangValidator` + the bgfx shader headers (`BGFX_SH_DIR=…` or
`.deps/bgfx/include/bgfx`), and the bgfx headers (`BGFX_INC_DIR=…`). A test whose tool is missing is shown as
"skipped", never counted as passed. See the header of the script for the variables (`UPDATE_GOLDEN=1`, `VERBOSE=1`,
`FUZZ_RUNS=n`).

**Status (27 October 2026).** With the real `shaderc` from `.deps/bgfx`: 123 unit checks and 159 end-to-end checks pass,
0 failed, 0 skipped. Every generated shader of the suite (22) is compiled by the real `shaderc` for `linux:spirv` and
`linux:430`, and the runtime and the generated stubs are type-checked against the real bgfx headers. **Not validated yet:**
execution on a GPU (see *Known limitations*).

With CMake: `cmake -S . -B build && cmake --build build && ctest --test-dir build`.

## Usage

```sh
cudagfx saxpy.cu -o gen/saxpy_gen.cpp --shader-dir gen \
        --shaderc .deps/bgfx/bin/shaderc --embed -i .deps/bgfx/include/bgfx
```

| Option | Purpose |
|---|---|
| `-o <file>` | generated host code; the stubs header is `<file without extension>_kernels.h` |
| `--kernels-header <file>` | explicit path of the stubs header |
| `--shader-out <path>` | `.sc` shader; a pattern containing `%` is replaced by the kernel name |
| `--shader-dir <dir>` | shader directory (created if needed), names `cs_<kernel>_<block>.sc` |
| `--shaderc <path>` | compile every shader with shaderc; `--target <platform:profile>` (repeatable) selects the targets (default: `linux:spirv`, `linux:430`) |
| `--embed` | embed the binaries (`--bin2c`) in `cs_<kernel>_<block>_<backend>.h` headers referenced by the stubs header |
| `-i <dir>` / `-I <dir>` | include directory for **shaderc** / for the **preprocessor** (`#include "x.cuh"`) |
| `-D<name>[=<value>]` | define a macro |
| `--depends <file>` | write the dependencies (followed headers) in Makefile format |
| `--no-line-preserve` | do not preserve the line numbering of the host code |
| `--dump-ir` | JSON summary (kernels, parameters, blocks, call closures) on standard output |

Exit codes: `0` success · `1` error in the CUDA source · `2` usage / unreadable file · `3` shaderc failure ·
`4` internal error or unable to write. Files are only written after a successful analysis, and only if their content
changes (incremental builds).

Block sizes and backends: one shader variant is produced for every distinct block size seen at the launch sites. shaderc
profiles map to bgfx backends following `embedded_shader.h`: `spirv`→`_spv` (Vulkan), `120…450`→`_glsl` (OpenGL),
`*_es`→`_essl`, `s_4_x/s_5_x`→`_dxbc` (Direct3D 11), `s_6_x`→`_dxil` (Direct3D 12), `metal`→`_mtl`, `wgsl`→`_wgsl`.

### CMake

```cmake
include(cmake/CudaGFX.cmake)
add_executable(app main.cpp)
cudagfx_add_kernels(TARGET app SOURCES kernels.cu
                    TARGETS linux:spirv linux:430
                    INCLUDES ${BGFX_DIR}/include/bgfx)
target_link_libraries(app PRIVATE bgfx)
```

Regeneration is triggered by the `.cu`, by the headers it includes (dependency file) and by the `cudagfx` / `shaderc`
targets when they are part of the project.

## Supported CUDA subset

| Area | Supported |
|---|---|
| Types | `bool`, `int`, `unsigned`/`uint`, `float`; `float2/3/4`, `int2/3/4`, `uint2/3/4`; arrays (including multidimensional); POD structs; global `const`/`constexpr` constants |
| Kernel parameters | scalars and vectors by value (at most 64 32-bit words); buffer pointers `float*`, `int*`, `unsigned*`, `float2/4*`, `int2/4*`, `uint2/4*` (`const` ⇒ read-only; a buffer that is never read is declared write-only) |
| Built-in variables | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`.x .y .z`, unsigned as in CUDA) |
| Shared memory | `__shared__ T t[N]` (constant N, up to 32 KB; warning above 16 KB); `__syncthreads()` → `barrier()` |
| Statements | `if/else`, `for`, `while`, `do`, `switch/case/default`, `break`, `continue`, `return`, multiple declarators, `#pragma unroll` ignored |
| Operators | every C operator on scalars; component-wise `+ - * /` on vectors (scalar broadcast); swizzles `.xy`, `.zw`…; compound assignments with C-faithful implicit conversion |
| Functions | `__device__` and `__host__ __device__` (return types honoured, prototypes, sorted call chains); `__forceinline__`, `inline`, `__restrict__` |
| Natives | `sinf cosf tanf asinf acosf atanf sinhf coshf tanhf expf exp2f logf log2f sqrtf rsqrtf fabsf floorf ceilf truncf roundf fminf fmaxf powf atan2f fmodf`, `min max abs`, `make_{float,int,uint}{2,3,4}`, `__float_as_int` & co, `__popc`, `isnan`, `isinf` |
| Atomics | `atomicAdd/Sub/Min/Max/And/Or/Xor/Exch/CAS` on `int`/`unsigned`, **in statement form**: `atomicAdd(&a[i], v);`, `old = atomicAdd(…);`, `int old = atomicAdd(…);` |
| Preprocessor | object-like/function-like/variadic `#define`, `#if/#ifdef/#elif/#else/#endif` (full C precedence), `#include "x.cuh"` followed (`-I`), `#pragma once` |
| Launch | `k<<<grid, block>>>(args)`; the block size must be known at compile time: literals, macros, `const`/`constexpr`, `dim3(…)` or a `dim3` variable initialized with constants and never modified |

### Diagnostics

| Code | Meaning |
|---|---|
| `CGFX0001` | pointer: alias, local pointer variable, `*p`, `&x`, pointer passed to a `__device__` function |
| `CGFX0002` | unsupported type (`double`, 64-bit integers, `char`, `short`, `size_t`…) or 64-bit literal |
| `CGFX0003` | warp intrinsic (`__shfl*`, `__ballot*`…) |
| `CGFX0004` | dynamic shared memory (`extern __shared__`, 3rd argument of `<<<>>>`) or non-constant size |
| `CGFX0005` | unsupported function (`printf`, `malloc`, `cuda*`…) or **recursion** |
| `CGFX0006` | block size is not a compile-time constant at the launch |
| `CGFX0007` | *(warning)* block size inconsistent with `__launch_bounds__` |
| `CGFX0008` | block or shared memory beyond the limits (block ≤ 1024 threads, z ≤ 64; 32 KB shared) |
| `CGFX0009` | unsupported C++ construct (template, lambda, `new`, `auto`, dynamic parallelism…) |
| `CGFX0010` | unknown identifier, function or kernel; `__device__` function without a visible definition |
| `CGFX0011` | `--shader-out` without a `%` pattern while there are several kernels |
| `CGFX0012` | unsupported buffer element type (`float3*`, `double*`, `bool*`, `struct*`) |
| `CGFX0013` | malformed `<<<>>>` launch, inside a macro, or inside an included file |
| `CGFX0014` | unsupported or malformed preprocessor directive (`#`/`##`, `#error`…) |
| `CGFX0015` | invalid atomic, or its result is used in the middle of an expression |
| `CGFX0016` | reserved identifier (`gl_`, `bgfx_`, `cgfx_`, `__`) or collision (`<buffer>Buffer`) |
| `CGFX0017` | *(warning)* `/` or `%` on signed integers |
| `CGFX0018` | lexical or syntax error |
| `CGFX0019` | type error: invalid operands, write to a `const`, impossible conversion… |
| `CGFX0020` | *(warning)* different semantics (`fmodf` → `mod`) |
| `CGFX0100` | internal error (please report it) |

Identifiers that are reserved words of a target language (`input`, `output`, `sample`, `filter`, `half`, `mod`…) are
**renamed automatically** (`cgfx_input`…): the CUDA source does not have to care.

### Accepted differences in semantics

* **Unsuffixed literals**: `0.5` is a `double` in CUDA; here it is treated as `float` (32-bit precision).
* **Signed integer `/` and `%`**: the result differs between backends for negative operands (warning `CGFX0017`).
* **`fmodf`**: translated to `mod` (different sign of the remainder) — warning `CGFX0020`.
* **`__syncthreads()`**: `barrier()`. Guarantees visibility of **shared** memory; visibility of global memory between
  threads of the same block is not guaranteed as in CUDA.
* **`threadIdx`, `blockIdx`, `blockDim`, `gridDim`** are unsigned: `threadIdx.x - 1` is `4294967295u` for thread 0, as in CUDA.

## Runtime (`cudagfx_rt.h`)

Header-only, C++17, no exceptions, no RTTI, no file I/O.

```cpp
bgfx::init(...);
cudagfx::init();                       // compute view (0 by default)
// ... cudaMalloc / cudaMemcpy / generated launches ...
cudagfx::shutdown();                   // BEFORE bgfx::shutdown()
```

**Execution model.** A launch is queued in the current bgfx frame and executes during `bgfx::frame()`. To respect
program order, any operation that touches a GPU buffer (`cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`) pushes
the frame first; `cudaMemcpyDeviceToHost` also waits for the data to come back. `cudaDeviceSynchronize()` pushes the
frame but does not guarantee that GPU execution has finished (use a readback). The runtime is not thread-safe (like bgfx).

**Errors.** CUDA convention (`cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorString`) **plus** a detailed code and
message: `cudagfx::lastErrorCode()` / `cudagfx::lastErrorMessage()`. A launch with an invalid argument is not executed.

| Code | Cause |
|---|---|
| `CGFXR001` | buffer size is not a multiple of the element size |
| `CGFXR002` | buffer already used with another element type |
| `CGFXR003` | same buffer bound twice in a launch, with at least one write |
| `CGFXR004` | no shader for the current backend (or compute / bgfx unavailable) |
| `CGFXR005` | unknown or null pointer, or misaligned offset (multiple of the element **and** of 256 bytes, a `bgfx::setBuffer` requirement) |
| `CGFXR006` | block or grid size inconsistent with the compiled shader |
| `CGFXR007` | too many by-value arguments (maximum 16 vec4) |
| `CGFXR008` | copy or fill outside the bounds of a buffer |

Without bgfx (`CUDAGFX_NO_BGFX`, or `<bgfx/bgfx.h>` not found), memory remains usable but every launch fails
explicitly (`CGFXR004`): there is no CPU backend.

## Repository layout

```
include/CudaGFX/   transpiler (Lexer, Preprocessor, Parser, Sema, ShaderEmitter, HostEmitter, Compiler…)
                   and runtime (cudagfx_rt.h, cudagfx_abi.h, cudagfx_compat.h)
src/cudagfx.cpp   command line
cmake/             CMake integration (cudagfx_add_kernels)
tests/             unit/ negative/ positive/ golden/ preserve/ include/ runtime/ cmake/ tools/ + run_tests.sh
docs/              verifications.md (technical points checked against bgfx), REVUE.md (code review report, in French)
examples/          saxpy, reduce, transpose; handwritten = validation of the runtime without the transpiler
CudaGFX_spec_prompt.md   the original specification the implementation follows
```

Each pipeline stage has a single responsibility: `SourceManager` (file contents), `Lexer`, `Preprocessor` (conditionals,
includes, macros), `ConstEvaluator` (one constant-expression grammar, shared by `#if`, array sizes and block sizes),
`Parser` (+ `LaunchScanner` for the host code), `Sema` (typing and validation), `ShaderEmitter` (pure translation of the
typed AST), `KernelAbi` (parameter binding, a single definition shared with the runtime through `abi::ArgLayout`),
`HostEmitter`, `ShadercDriver`, `Compiler` (orchestration).

## Contributing

* Build with `make` (the build uses `-std=c++17 -Wall -Wextra -Wpedantic -Werror`: keep the code warning-free) and run
  `make test` before sending a change. `tests/run_tests.sh` explains how to add a negative test (`// EXPECT …` line), a
  positive test (+ `UPDATE_GOLDEN=1` and review of the generated shaders) and how to enable the optional tools.
* Every user-facing error must go through the diagnostics engine (`Diagnostics.hpp`) with a `CGFXnnnn` code and a precise
  location — never a bare `std::cerr` — and a new code must be documented in the table above.
* Typing decisions belong in `Sema`; the emitters only translate. If the emitter needs to know a type, the AST is missing
  something.
* The parameter ABI (`cudagfx_abi.h`) is shared between the tool and the runtime: change it in one place only, and update
  both sides' tests.

## Known limitations

* No CPU backend (fallback C++ emission); the runtime only runs on bgfx.
* A kernel defined in a header is not supported (define it in the main file); `__device__` functions and constants from
  `#include "…"` headers are.
* No `typedef` / `using` of types (use the native type or `#define`), streams, events, managed memory or textures.
* A kernel cannot take a struct by value (pass its fields) or a `float3*` pointer (use `float4*`).
* Execution on a GPU has not been validated: no program of the repository initializes bgfx and runs a kernel yet
  (`examples/handwritten` does not call `bgfx::init` itself). The order of `bgfx::update` / `dispatch`, the `bgfx::read`
  readback and the uniforms therefore remain to be checked on a real device.
* The automated suite compiles the shaders with the real `shaderc` for `linux:spirv` and `linux:430` only; the other
  profiles (Metal, WGSL, DXBC, DXIL, ESSL) were only checked by hand on a few kernels (see `docs/verifications.md`, V24).
  With the bgfx/Dawn version used, `--profile wgsl` crashes in Tint on kernels using `isnan` / `isinf`.
