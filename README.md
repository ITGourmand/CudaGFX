# CudaGFX

A **CUDA → bgfx compute shader transpiler**. From a `.cu` file, `cudagfx` produces:

* one **compute shader (`.sc`)** per kernel and per block size, ready to be compiled by `shaderc` for Vulkan, OpenGL,
  OpenGL ES, Direct3D 11/12, Metal and WebGPU;
* the **host code as standard C++17** (the `.cu` file without its kernels, with every `k<<<g, b>>>(…)` rewritten as a
  stub call) — lines that are not modified keep their line number;
* a **stubs header** that binds the arguments and the embedded shader binaries to the runtime `cudagfx_rt.h`, which
  emulates `cudaMalloc` / `cudaMemcpy` / `cudaMemset` / `cudaFree` on top of bgfx;
* optionally (`--emit-cpu`), a **CPU implementation of the same kernels** (C++17), to validate a kernel without a GPU
  or to run the program on a machine that has none.

```
 prog.cu ──► cudagfx ──┬─► cs_<kernel>_<block>.sc ──► shaderc ──► cs_<kernel>_<block>_<backend>.h
                       ├─► prog_gen.cpp          (host code, C++17)
                       ├─► prog_gen_kernels.h    (stubs + shader tables)   + cudagfx_rt.h (runtime)
                       └─► prog_gen_cpu.h        (--emit-cpu: kernels on the CPU)  + cudagfx_cpu.h
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

**Status (28 October 2026).** `make test` (unit tests + `tests/run_tests.sh`) passes with 0 failure and 0 skipped test when
`glslangValidator` (GLSL validation of every generated shader) and the bgfx headers are available. The CPU backend is
exercised end to end (compilation of every valid kernel, execution of the programs of `tests/cpu/` against a host
reference, under both executors and ThreadSanitizer). **Not validated:** anything that needs a GPU, and the other shader
profiles (see [Known limitations](#known-limitations)). The test counts are printed by the suite itself.

With CMake: `cmake -S . -B build && cmake --build build && ctest --test-dir build` (includes the install + `find_package` test).

Building needs a C++17 compiler and the **static** `libstdc++` (the tool is linked with `-static-libstdc++`): it is part of
`libstdc++-dev` on Debian / Ubuntu and `gcc` on Arch, but a separate package on Fedora (`sudo dnf install libstdc++-static`).
For a binary with no dependency at all (glibc included), which is what the releases ship: `make linux LDFLAGS="-static -lpthread"`
(on Fedora this also needs `glibc-static`).


**GPU tests.** `make test` also builds `examples/gpu_validation` against the bgfx libraries and runs it on each renderer where
bgfx can be initialised; it reports the test as *skipped*, with the reason, when it cannot (no GPU, no working `shaderc`, no
bgfx libraries). The program runs 19 checks twice, on the GPU and on the CPU backend, each against a host reference:
by-value arguments, a 32-byte struct buffer, shared memory with barriers, atomics, vector swizzles, `__constant__`
updated between launches, Conway's Game of Life (global and shared-memory tile, 100 chained generations, bit-exact),
Particle Life, the Primordial Particle System, Gray-Scott, bitonic sort, prefix scan (three dependent kernels), tiled matmul and
Mandelbrot. Every check prints `PASS` / `FAIL` with its time on each side.

```sh
make test                                          # GPU tests run when possible
CUDAGFX_GPU_RENDERERS="vulkan" make test            # choose the renderers (default: vulkan opengl)
CUDAGFX_GPU=0 make test                             # never run them
BGFX_LIB_DIR=/path/to/lib make test                 # libbgfx.a, libbx.a, libbimg.a (default: .deps/bgfx/lib64 or lib)
./path/to/gpu_validation --renderer vulkan --only life   # run one suite by hand (basics|life|particles|algorithms)
```

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
| `--emit-cpu` | also generate the CPU implementation of the kernels (requires `-o`): header `<file without extension>_cpu.h`, included at the end of the host code (see [CPU backend](#cpu-backend)) |
| `--cpu-header <file>` | explicit path of that CPU header |
| `--suppress <CGFXnnnn>` | silence a warning (repeatable); an error cannot be suppressed |
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
find_package(CudaGFX CONFIG REQUIRED)      # installed package: CudaGFX::rt, CudaGFX::cudagfx, cudagfx_add_kernels()
# (from a source tree instead: include(cmake/CudaGFX.cmake))
add_executable(app main.cpp)
cudagfx_add_kernels(TARGET app SOURCES kernels.cu
                    TARGETS linux:spirv linux:430
                    INCLUDES ${BGFX_DIR}/include/bgfx   # shaderc -i and preprocessor -I
                    SUPPRESS CGFX0017                   # optional: silence warnings
                    EMIT_CPU)                           # optional: also generate the CPU implementation
target_link_libraries(app PRIVATE bgfx CudaGFX::rt)
```

Regeneration is triggered by the `.cu`, by the headers it includes (dependency file) and by the `cudagfx` / `shaderc`
targets when they are part of the project. `cmake --install` installs the tool, the runtime headers and the package
files (`CudaGFXConfig.cmake`, targets, version: same minor version while the project is 0.x).

## Continuous integration and releases

`.github/workflows/ci.yml` runs on every push to `main` and every pull request: the whole test suite with GCC and Clang (the
shaders validated as GLSL, the runtime type-checked against the real bgfx headers, the CPU backend executed, sanitizers,
fuzzing), the suite again on Ubuntu 22.04 (GCC 11), the CMake build with the install + `find_package` test, the CPU backend with
the OS-thread executor, the Windows build (mingw) and a run of `cudagfx.exe` on a Windows runner. The GPU tests are skipped there
(a runner has no GPU).

`.github/workflows/shaderc.yml` builds bgfx and `shaderc` from the sources of the latest `bgfx.cmake` on Ubuntu 22.04 and 24.04,
Debian 12 and 13, Arch Linux and Fedora, and runs the whole suite with that real `shaderc` (every shader compiled for Vulkan and
OpenGL, the GPU test program built and linked, then skipped for lack of a GPU). It runs on every push to `main`, every Monday and
on demand (*Actions* > *Real shaderc* > *Run workflow*, with an optional `bgfx.cmake` commit); the first build per distribution takes
a while, then the result is cached until `bgfx.cmake` changes, so a regression of bgfx shows up here first.

`.github/workflows/release.yml` publishes a release when a tag `v...` is pushed, after the tests pass on the tagged commit:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

It attaches `cudagfx-<tag>-linux-x86_64.tar.gz` (fully static: no glibc or libstdc++ needed, it runs on any x86-64 Linux),
`cudagfx-<tag>-windows-x86_64.zip` and
`SHA256SUMS.txt`; each archive holds the binary, the runtime headers and `cmake/CudaGFX.cmake`.

## Supported CUDA subset

| Area | Supported |
|---|---|
| Types | `bool`, `int`, `unsigned`/`uint`, `float`; `float2/3/4`, `int2/3/4`, `uint2/3/4`; arrays (including multidimensional); POD structs; global `const`/`constexpr` constants; `__constant__` memory: `int`/`uint`/`float` and their 2-4 component vectors, alone or as a one-dimensional array of at most 16 KiB. It is **writable at run time** (`cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, with an offset) as in CUDA, see [Constant memory](#constant-memory) |
| Kernel parameters | scalars and vectors by value (at most 64 32-bit words); buffer pointers `float*`, `int*`, `unsigned*`, `float2/4*`, `int2/4*`, `uint2/4*`, and POD struct pointers `Particle*` (≤ 256 bytes) whose layout is identical on the host, in std430 and in HLSL (see [Struct buffers](#struct-buffers)); `const` ⇒ read-only; unread buffers are declared write-only |
| Kernels & Templates | `__global__` kernels and instantiated kernel templates (`template <...> __global__ void k(...)`), specialized for launch sites observed in host code |
| Built-in variables | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`.x .y .z`, unsigned as in CUDA) |
| Shared memory | `__shared__ T t[N]` (constant N, up to 32 KB; warning above 16 KB); `__syncthreads()` → `barrier()` |
| Statements | `if/else`, `for`, `while`, `do`, `switch/case/default`, `break`, `continue`, `return`, multiple declarators, `#pragma unroll` ignored |
| Operators | every C operator on scalars; component-wise `+ - * /` on vectors (scalar broadcast); swizzles `.xy`, `.zw`…; compound assignments with C-faithful implicit conversion |
| Functions | `__device__` and `__host__ __device__` (return types honoured, prototypes, sorted call chains); `__forceinline__`, `inline`, `__restrict__` |
| Natives | `sinf cosf tanf asinf acosf atanf sinhf coshf tanhf expf exp2f logf log2f sqrtf rsqrtf fabsf floorf ceilf truncf roundf fminf fmaxf powf atan2f fmodf`, `min max abs`, `make_{float,int,uint}{2,3,4}`, `__float_as_int` & co, `__popc`, `isnan`, `isinf` |
| Atomics | `atomicAdd/Sub/Min/Max/And/Or/Xor/Exch/CAS` on `int`/`unsigned` (not on `float`), **in statement form**: `atomicAdd(&a[i], v);`, `old = atomicAdd(…);`, `int old = atomicAdd(…);` |
| Preprocessor | object-like/function-like/variadic `#define`, `#if/#ifdef/#elif/#else/#endif` (full C precedence), `#include "x.cuh"` followed (`-I`), `#pragma once` |
| Launch | `k<<<grid, block>>>(args)` (and `k<args><<<grid, block>>>(args)`); the block size must be known at compile time: literals, macros, `const`/`constexpr`, `dim3(…)` or a `dim3` variable initialized with constants and never modified |

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
| `CGFX0009` | unsupported C++ construct (uninstantiated template, lambda, `new`, `auto`, dynamic parallelism…) |
| `CGFX0010` | unknown identifier, function or kernel; `__device__` function without a visible definition |
| `CGFX0011` | `--shader-out` without a `%` pattern while there are several kernels |
| `CGFX0012` | unsupported buffer element type (`float3*`, `double*`, `bool*`, non-POD struct) |
| `CGFX0013` | malformed `<<<>>>` launch, inside a macro, or inside an included file |
| `CGFX0014` | unsupported or malformed preprocessor directive (`#`/`##`, `#error`…) |
| `CGFX0015` | invalid atomic, or its result is used in the middle of an expression |
| `CGFX0016` | reserved identifier (`gl_`, `bgfx_`, `cgfx_`, `__`) or collision (`<buffer>Buffer`) |
| `CGFX0017` | *(warning)* `/` or `%` on signed integers |
| `CGFX0018` | lexical or syntax error |
| `CGFX0019` | type error: invalid operands, write to a `const`, impossible conversion… |
| `CGFX0020` | *(warning)* different semantics (`fmodf` → `mod`) |
| `CGFX0021` | struct buffer element whose layout differs between the host and the shaders (misaligned vector, `float3`/`bool` field, trailing padding, > 256 bytes) |
| `CGFX0022` | `cudaMemcpyToSymbol` on a `__constant__` that is baked into the shaders (a `bool` array, or more than 16 KiB, always with an initializer): it is read-only there |
| `CGFX0023` | `__constant__` of a type that cannot live in constant memory (a struct...) and that has no initializer to bake |
| `CGFX0100` | internal error (please report it) |

Identifiers that are reserved words of a target language (`input`, `output`, `sample`, `filter`, `half`, `mod`…) are
**renamed automatically** (`cgfx_input`…): the CUDA source does not have to care.

### Accepted differences in semantics

* **Unsuffixed literals**: `0.5` is a `double` in CUDA; here it is treated as `float` (32-bit precision).
* **Signed integer `/` and `%`**: the result differs between backends for negative operands (warning `CGFX0017`; use a shift or
  unsigned operands when the value cannot be negative, or `--suppress CGFX0017`).
* **`fmodf`**: translated to `mod` (different sign of the remainder) — warning `CGFX0020`.
* **`__syncthreads()`**: `barrier()`. Guarantees visibility of **shared** memory; visibility of global memory between
  threads of the same block is not guaranteed as in CUDA.
* **`threadIdx`, `blockIdx`, `blockDim`, `gridDim`** are unsigned: `threadIdx.x - 1` is `4294967295u` for thread 0, as in CUDA.

### Struct buffers

A struct travels through a buffer as raw 32-bit words, and three compilers lay it out: the host C++ compiler, GLSL
`std430` (OpenGL, Vulkan, WebGPU) and HLSL structured buffers (Direct3D, which pack tightly). The shader reads the
right bytes at the wrong offsets, without any error, as soon as they disagree. `cudagfx` therefore computes the three
layouts and **refuses the struct (`CGFX0021`)** unless they are identical, and the generated host code ends with
`static_assert`s (`sizeof`, `alignof`, `offsetof`) that make the C++ compiler confirm it.

In practice: 2-component vectors at an offset multiple of 8, 4-component vectors at a multiple of 16, no `float3` or
`bool` field (use `float4` / `int`), explicit padding fields, and a total size that is a multiple of the largest
alignment. The CUDA vector types of `cudagfx_compat.h` carry the CUDA alignments (`float2`: 8, `float4`: 16).

```cpp
struct Particle { float4 pos; float4 vel; };                   // ok: 32 bytes
struct Item     { int id; float weight; float2 uv; float4 color; };  // ok: offsets 0, 4, 8, 16
struct Bad      { float a; float4 b; };                        // CGFX0021: b is at 16 (host, std430) but at 4 (HLSL)
```

### Constant memory

A `__constant__` object reaches the shaders as a **uniform array** (`uniform vec4 u_cmem_<name>[N]`) holding the 32-bit
words of the host object. The generated host code registers each such object with the runtime at start-up, and the
runtime copies its **current** content into the uniform at every launch of a kernel that reads it. So the CUDA idiom works:

```cpp
__constant__ float c_lut[256];
...
cudaMemcpyToSymbol(c_lut, table, sizeof(table));   // writes the host object...
kernel<<<grid, block>>>(...);                        // ...which this launch uploads
```

The shader reads an element through a small generated accessor (`cgfx_cm_c_lut(i)`), so any index expression, vectors
(`float3` arrays have a 12-byte stride, as on the host) and integers (transported as their bit pattern, like the by-value
kernel arguments) work. Limits: numeric scalars and vectors, one dimension, 16 KiB. Anything else with an initializer (a
`bool` array, a bigger table) is baked into the shader as a read-only constant and `cudaMemcpyToSymbol` on it is rejected
(`CGFX0022`); without an initializer it is an error (`CGFX0023`).

Buffers of structs reach the shaders as **raw 32-bit words** (`BUFFER_RAW_*`, `rawLoad*` / `rawStore*` of bgfx), not as
`StructuredBuffer<Struct>`: the generated shader reads and writes each field at the offset computed by `StructLayout` (the
same on every backend, that is what `CGFX0021` guarantees), and the fields of a struct are loaded / stored by small
generated functions (`cgfx_ld_<buffer>_<Struct>`, `cgfx_st_<buffer>_<Struct>`). The newest `shaderc` (bgfx, October 2026)
rejects any `StructuredBuffer` of a struct ("Global 'buf.@data.field' is implicitly a uniform"); raw words do not depend on it.
A field is assigned (`=`, `op=`, `++`, `--`) in a statement of its own. Atomics on a struct field
(`atomicAdd(&cells[i].count, 1u)`) are not supported (`CGFX0015`): the target of an atomic must be `&a[i]`, with `a` a
buffer of 32-bit integers or a `__shared__` array.

## CPU backend

`--emit-cpu` generates `<host>_cpu.h`: the kernels as C++17, included at the very end of the generated host code (where
every struct, constant and `__constant__` array is complete). It is meant to **validate kernels without a GPU** and to
run on machines that have none, not to be fast.

```cpp
cudagfx::setExecutionMode(cudagfx::ExecutionMode::CPU);   // at run time (the GPU path stays available), or
// -DCUDAGFX_CPU_ONLY at build time: no bgfx, no shader needed
```

* Kernels without `__syncthreads()` run their threads one after the other. A kernel with barriers runs the threads of a
  block as **fibers** on Linux x86-64 (a user-space context switch, no system call) and as one OS thread per CUDA
  thread elsewhere, under the sanitizers, or with `-DCUDAGFX_CPU_THREADS_ONLY`. A thread that leaves the kernel before a
  barrier does not hold the others up, as on a GPU. Blocks run one after the other.
* `__constant__` objects are read directly from the host objects, so `cudaMemcpyToSymbol` works (the GPU path re-uploads them
  at each launch: see [Constant memory](#constant-memory)).
* Vector arithmetic, swizzles (also as lvalues), the math natives and all atomics are provided by `cudagfx_cpu.h`.
* Buffers that already live on the GPU are brought back to the host when a CPU kernel touches them, so mixing both
  modes in one program stays coherent.
* Measured on a 262 144-element block reduction (1024 blocks of 256 threads, 11 barriers per block): 0.12 s with fibers,
  11 s with OS threads.

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
| `CGFXR004` | no shader for the current backend |
| `CGFXR005` | unknown or null pointer, or misaligned offset (multiple of the element **and** of 256 bytes, a `bgfx::setBuffer` requirement) |
| `CGFXR006` | block or grid size inconsistent with the compiled shader |
| `CGFXR007` | too many by-value arguments (maximum 16 vec4) |
| `CGFXR008` | copy or fill outside the bounds of a buffer, or of a `cudaMemcpyToSymbol` / `FromSymbol` |
| `CGFXR009` | no usable compute backend: bgfx absent, `Noop` renderer, compute unsupported (the kernel is **not** executed) |
| `CGFXR010` | bgfx failed to create a buffer, uniform or program, or to read data back |
| `CGFXR011` | element type that cannot be described to bgfx with a matching stride |
| `CGFXR012` | a `__constant__` read by the kernel is not registered with the runtime, or smaller than the shader expects |

Without bgfx (`CUDAGFX_NO_BGFX`, or `<bgfx/bgfx.h>` not found), GPU launches fail explicitly (`CGFXR009`); the
[CPU backend](#cpu-backend) still runs the kernels. All the checks of a launch happen before the first `bgfx::set*` call,
so a refused launch leaves no binding behind for the next one.

## Repository layout

```
include/CudaGFX/   the libraries: the transpiler (Lexer, Preprocessor, Parser, Sema, StructLayout, ShaderEmitter,
                   CpuEmitter, HostEmitter, Compiler…) and the header-only runtime that applications include:
                   cudagfx_rt.h (runtime), cudagfx_cpu.h (CPU backend), cudagfx_abi.h (parameter ABI shared
                   with the tool), cudagfx_compat.h (CUDA types and qualifiers)
src/cudagfx.cpp   the command line (main)
cmake/             CMake integration (cudagfx_add_kernels) and package configuration
tests/             unit/ negative/ positive/ golden/ preserve/ include/ runtime/ cpu/ cmake/ cmake_install/ tools/ + run_tests.sh
docs/              REVIEW.md (review history), verifications.md (technical points checked against bgfx),
                   CudaGFX_spec_prompt.md (the original specification, in French)
examples/          saxpy, reduce, transpose; handwritten = the runtime without the transpiler; gpu_validation = GPU check
```

Each pipeline stage has a single responsibility: `SourceManager` (file contents), `Lexer`, `Preprocessor` (conditionals,
includes, macros), `ConstEvaluator` (one constant-expression grammar, shared by `#if`, array sizes and block sizes),
`Parser` (+ `LaunchScanner` for the host code), `Sema` (typing and validation), `ShaderEmitter` (pure translation of the
typed AST), `KernelAbi` (parameter binding, a single definition shared with the runtime through `abi::ArgLayout`),
`StructLayout` (host / std430 / HLSL layouts of buffer structs), `AstPrinter` (the statement / expression printer shared by
the two code generators), `ShaderEmitter` (shader dialect), `CpuEmitter` (C++ dialect, the CPU backend), `HostEmitter` (host code
and stubs), `ShadercDriver`, `Compiler` (orchestration).

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

* The CPU backend is not fast: a barrier-free kernel is a plain loop; a kernel with barriers needs fibers (Linux x86-64) or
  OS threads. Blocks are not run in parallel.
* A kernel defined in a header is not supported (define it in the main file); `__device__` functions and constants from
  `#include "…"` headers are. The structs used as buffer elements must be declared at global scope.
* No `typedef` / `using` of types (use the native type or `#define`), streams, events, managed memory or textures. The
  preprocessor does not support `#` / `##` in macro bodies (`CGFX0014`), even in host code.
* A kernel cannot take a struct by value (pass its fields) or a `float3*` pointer (use `float4*`).
* The automated suite validates the shaders for GLSL (`glslangValidator`) and, when a working `shaderc` is available, for
  `linux:spirv` and `linux:430`; the other profiles (Metal, WGSL, DXBC, DXIL, ESSL) were only checked by hand on a few
  kernels (see `docs/verifications.md`, V24). With the bgfx/Dawn version used, `--profile wgsl` crashes in Tint on kernels
  using `isnan` / `isinf`. The `shaderc` binary shipped in `install.zip` was built against glibc 2.44 (Nix) and does not
  run on older systems: build `shaderc` from the bgfx sources for CI (see `.github/workflows/ci.yml`).
