// cudagfx_rt.h: the CudaGFX runtime (header-only, C++17, no exceptions, no RTTI, no file I/O).
//
// Emulates the CUDA host API on top of bgfx: cudaMalloc/cudaMemcpy/cudaMemset/cudaFree, and launches of
// kernels compiled to compute shaders. Errors follow the CUDA convention (cudaGetLastError); in addition,
// every error carries a CGFXR00n code and a precise message, readable through cudagfx::lastErrorCode() and
// cudagfx::lastErrorMessage().
//
// Execution model (C8): launches are queued in the current bgfx frame and execute during
// bgfx::frame(). Any operation that touches the memory of a GPU buffer (cudaMemcpy, cudaMemset,
// cudaDeviceSynchronize) first pushes the frame (flush) to respect program order; a read
// (cudaMemcpyDeviceToHost) additionally waits for the data to come back. The runtime is not thread-safe
// (like bgfx: everything from the thread that initialized it). Call cudagfx::shutdown() BEFORE bgfx::shutdown().
#pragma once

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "cudagfx_abi.h"
#include "cudagfx_compat.h"

#if __has_include(<bgfx/bgfx.h>) && !defined(CUDAGFX_NO_BGFX)
#   define CUDAGFX_HAS_BGFX 1
#   include <bgfx/bgfx.h>
#else
#   define CUDAGFX_HAS_BGFX 0
#endif

// ============================================================================
// Fundamental CUDA types
// ============================================================================

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

enum cudaError_t {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorLaunchFailure         = 4,
    cudaErrorPriorLaunchFailure    = 5,
    cudaErrorLaunchTimeout         = 6,
    cudaErrorLaunchOutOfResources  = 7,
    cudaErrorInvalidDeviceFunction = 8,
    cudaErrorInvalidConfiguration  = 9,
    cudaErrorInvalidDevice         = 10,
    cudaErrorInvalidValueType      = 11,
    cudaErrorUnknown               = 999
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};

namespace cudagfx {

enum class Access { Read = 0, Write = 1, ReadWrite = 2 };

// Error codes specific to the runtime (documented in the README).
enum class ErrorCode {
    None = 0,
    ElementSize = 1,     // CGFXR001: buffer size is not a multiple of the element size
    TypeReinterpret = 2, // CGFXR002: buffer already used with another element type
    PointerAlias = 3,    // CGFXR003: same buffer bound twice in a launch (at least one write)
    MissingShader = 4,   // CGFXR004: no shader for the current backend (or compute not supported)
    BadPointer = 5,      // CGFXR005: unknown or null pointer, or misaligned offset
    BadBlock = 6,        // CGFXR006: block or grid size inconsistent with the compiled shader
    ArgOverflow = 7,     // CGFXR007: too many by-value arguments (maximum 16 vec4)
    OutOfBounds = 8      // CGFXR008: copy or fill outside the bounds of the buffer
};

inline const char* errorCodeName(ErrorCode c) {
    switch (c) {
        case ErrorCode::None: return "OK";
        case ErrorCode::ElementSize: return "CGFXR001";
        case ErrorCode::TypeReinterpret: return "CGFXR002";
        case ErrorCode::PointerAlias: return "CGFXR003";
        case ErrorCode::MissingShader: return "CGFXR004";
        case ErrorCode::BadPointer: return "CGFXR005";
        case ErrorCode::BadBlock: return "CGFXR006";
        case ErrorCode::ArgOverflow: return "CGFXR007";
        case ErrorCode::OutOfBounds: return "CGFXR008";
    }
    return "CGFXR???";
}

struct EmbeddedShaderRef {
    const uint8_t* data = nullptr;
    uint32_t size = 0;
};

// One field per bgfx backend (embedded_shader.h suffixes: _spv, _glsl, _essl, _dxbc, _dxil, _mtl, _wgsl).
struct ShaderBinaryTable {
    const char* kernel = "";
    const char* argsUniform = "";   // u_<kernel>_args
    const char* gridUniform = "";   // u_<kernel>_grid
    unsigned block[3] = {1, 1, 1};  // block size compiled into the shader
    EmbeddedShaderRef spv;    // Vulkan
    EmbeddedShaderRef glsl;   // OpenGL
    EmbeddedShaderRef essl;   // OpenGL ES
    EmbeddedShaderRef dxbc;   // Direct3D 11
    EmbeddedShaderRef dxil;   // Direct3D 12
    EmbeddedShaderRef mtl;    // Metal
    EmbeddedShaderRef wgsl;   // WebGPU
};

namespace detail {

// Alignment required by bgfx::setBuffer for the offset of a buffer view.
constexpr size_t kBufferViewAlign = 256;
// bgfx describes buffer sizes on 32 bits.
constexpr size_t kMaxAllocationBytes = 0x7FFFFFFFu;

struct Allocation {
    uintptr_t baseAddress = 0;
    size_t sizeInBytes = 0;
    std::vector<uint8_t> shadowBuffer;  // content as long as the buffer has not moved to the GPU
    bool isGpuAllocated = false;
    uint32_t elementTypeCode = 0;
    size_t elementSize = 0;             // element size once typed (0 = not typed yet)
#if CUDAGFX_HAS_BGFX
    bgfx::DynamicVertexBufferHandle gpuBuffer = BGFX_INVALID_HANDLE;
#endif
};

class RuntimeState {
public:
    static RuntimeState& instance() {
        static RuntimeState s_instance;
        return s_instance;
    }

    uint16_t computeViewId = 0;
    cudaError_t lastError = cudaSuccess;
    ErrorCode detailCode = ErrorCode::None;
    char detailMessage[224] = {0};
    std::map<uintptr_t, Allocation> allocations;
    uintptr_t nextFakeAddress = 0x10000000;
    unsigned pendingDispatches = 0;  // launches queued since the last bgfx::frame()

#if CUDAGFX_HAS_BGFX
    struct UniformEntry {
        bgfx::UniformHandle handle = BGFX_INVALID_HANDLE;
        uint16_t slots = 0;
    };
    std::map<const uint8_t*, bgfx::ProgramHandle> programs;  // one program per shader binary
    std::map<std::string, UniformEntry> uniforms;            // one uniform per name
#endif

    Allocation* findAllocation(uintptr_t addr) {
        if (allocations.empty()) return nullptr;
        auto it = allocations.upper_bound(addr);
        if (it == allocations.begin()) return nullptr;
        --it;
        Allocation& a = it->second;
        return (addr >= a.baseAddress && addr < a.baseAddress + a.sizeInBytes) ? &a : nullptr;
    }
};

inline RuntimeState& state() { return RuntimeState::instance(); }

#if defined(__GNUC__)
#   define CUDAGFX_PRINTF_FORMAT(a, b) __attribute__((format(printf, a, b)))
#else
#   define CUDAGFX_PRINTF_FORMAT(a, b)
#endif

// Records the error (CUDA code + CGFXR code + message) and returns the CUDA code.
CUDAGFX_PRINTF_FORMAT(3, 4)
inline cudaError_t fail(cudaError_t cuda, ErrorCode code, const char* fmt, ...) {
    RuntimeState& rt = state();
    rt.lastError = cuda;
    rt.detailCode = code;
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(rt.detailMessage, sizeof(rt.detailMessage), fmt, args);
    va_end(args);
    return cuda;
}

// Launch configuration error: called by the generated stubs when no variant matches.
inline void badBlockSize(const char* kernel, dim3 blk) {
    fail(cudaErrorInvalidConfiguration, ErrorCode::BadBlock,
         "kernel '%s': no variant compiled for a block of %ux%ux%u (the block size must be one of "
         "celles vues aux lancements <<<>>> du source)",
         kernel, blk.x, blk.y, blk.z);
}

template <typename T> struct TypeHelper;
template <> struct TypeHelper<float>    { static constexpr uint32_t code = 1;  static constexpr unsigned components = 1; };
template <> struct TypeHelper<float2>   { static constexpr uint32_t code = 2;  static constexpr unsigned components = 2; };
template <> struct TypeHelper<float4>   { static constexpr uint32_t code = 4;  static constexpr unsigned components = 4; };
template <> struct TypeHelper<int32_t>  { static constexpr uint32_t code = 11; static constexpr unsigned components = 1; };
template <> struct TypeHelper<int2>     { static constexpr uint32_t code = 12; static constexpr unsigned components = 2; };
template <> struct TypeHelper<int4>     { static constexpr uint32_t code = 14; static constexpr unsigned components = 4; };
template <> struct TypeHelper<uint32_t> { static constexpr uint32_t code = 21; static constexpr unsigned components = 1; };
template <> struct TypeHelper<uint2>    { static constexpr uint32_t code = 22; static constexpr unsigned components = 2; };
template <> struct TypeHelper<uint4>    { static constexpr uint32_t code = 24; static constexpr unsigned components = 4; };

inline size_t roundUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// ----------------------------------------------------------------------------
// Synchronization with bgfx
// ----------------------------------------------------------------------------

// Pushes the frame if launches are pending: the memory operations that follow will execute
// after them, as in an ordered CUDA stream.
inline void flushPending() {
#if CUDAGFX_HAS_BGFX
    RuntimeState& rt = state();
    if (rt.pendingDispatches > 0) {
        bgfx::frame();
        rt.pendingDispatches = 0;
    }
#endif
}

// ----------------------------------------------------------------------------
// Access to the content of an allocation (shadow or GPU)
// ----------------------------------------------------------------------------

inline cudaError_t checkRange(const Allocation& a, size_t offset, size_t count) {
    if (offset > a.sizeInBytes || count > a.sizeInBytes - offset) {
        return fail(cudaErrorInvalidValue, ErrorCode::OutOfBounds,
                    "access of %.0f byte(s) at offset %.0f of a buffer of %.0f byte(s)", static_cast<double>(count),
                    static_cast<double>(offset), static_cast<double>(a.sizeInBytes));
    }
    return cudaSuccess;
}

inline cudaError_t deviceRead(Allocation& a, size_t offset, size_t count, void* dst) {
    if (cudaError_t e = checkRange(a, offset, count)) return e;
    if (count == 0) return cudaSuccess;
    if (!a.isGpuAllocated) {
        std::memcpy(dst, a.shadowBuffer.data() + offset, count);
        return cudaSuccess;
    }
#if CUDAGFX_HAS_BGFX
    RuntimeState& rt = state();
    // Range aligned on the element: bgfx describes typed buffers in whole elements.
    const size_t es = a.elementSize ? a.elementSize : 1;
    const size_t begin = offset - offset % es;
    const size_t end = std::min(roundUp(offset + count, es), a.sizeInBytes);
    std::vector<uint8_t> tmp;
    void* target = dst;
    const bool aligned = begin == offset && end == offset + count;
    if (!aligned) {
        tmp.resize(end - begin);
        target = tmp.data();
    }
    bgfx::BufferRegion region;
    region.init(a.gpuBuffer, static_cast<uint32_t>(begin), static_cast<uint32_t>(end - begin));
    const uint32_t ready = bgfx::read(region, target);
    unsigned guard = 0;
    while (bgfx::frame() < ready) {
        if (++guard > 32) {
            return fail(cudaErrorLaunchTimeout, ErrorCode::MissingShader, "GPU buffer read: data not available after %u frames", guard);
        }
    }
    rt.pendingDispatches = 0;
    if (!aligned) std::memcpy(dst, tmp.data() + (offset - begin), count);
    return cudaSuccess;
#else
    return fail(cudaErrorInitializationError, ErrorCode::MissingShader, "GPU buffer without bgfx");
#endif
}

inline cudaError_t deviceWrite(Allocation& a, size_t offset, const void* src, size_t count) {
    if (cudaError_t e = checkRange(a, offset, count)) return e;
    if (count == 0) return cudaSuccess;
    if (!a.isGpuAllocated) {
        std::memmove(a.shadowBuffer.data() + offset, src, count);
        return cudaSuccess;
    }
#if CUDAGFX_HAS_BGFX
    flushPending(); // launches already submitted must see the old content
    const size_t es = a.elementSize ? a.elementSize : 1;
    const size_t begin = offset - offset % es;
    const size_t end = std::min(roundUp(offset + count, es), a.sizeInBytes);
    if (begin == offset && end == offset + count) {
        const bgfx::Memory* mem = bgfx::copy(src, static_cast<uint32_t>(count));
        bgfx::update(a.gpuBuffer, static_cast<uint32_t>(offset / es), mem);
        return cudaSuccess;
    }
    // Unaligned range: read-modify-write on the touched elements.
    std::vector<uint8_t> tmp(end - begin);
    if (cudaError_t e = deviceRead(a, begin, end - begin, tmp.data())) return e;
    std::memcpy(tmp.data() + (offset - begin), src, count);
    const bgfx::Memory* mem = bgfx::copy(tmp.data(), static_cast<uint32_t>(tmp.size()));
    bgfx::update(a.gpuBuffer, static_cast<uint32_t>(begin / es), mem);
    return cudaSuccess;
#else
    return fail(cudaErrorInitializationError, ErrorCode::MissingShader, "GPU buffer without bgfx");
#endif
}

inline void* allocate(size_t bytes) {
    RuntimeState& rt = state();
    if (bytes == 0) return nullptr; // as in CUDA: success, null pointer
    const size_t span = roundUp(bytes, 256) + 256; // 256 guard bytes: base + size belongs to no neighbouring allocation
    if (bytes > kMaxAllocationBytes ||
        rt.nextFakeAddress + span < rt.nextFakeAddress /* overflow */) {
        fail(cudaErrorMemoryAllocation, ErrorCode::OutOfBounds, "allocation of %.0f byte(s) refused (maximum %.0f)", static_cast<double>(bytes),
             static_cast<double>(kMaxAllocationBytes));
        return nullptr;
    }
    const uintptr_t addr = rt.nextFakeAddress;
    rt.nextFakeAddress += span;

    Allocation a;
    a.baseAddress = addr;
    a.sizeInBytes = bytes;
    a.shadowBuffer.assign(bytes, 0);
    rt.allocations.emplace(addr, std::move(a));
    return reinterpret_cast<void*>(addr);
}

inline void destroyGpu(Allocation& a) {
#if CUDAGFX_HAS_BGFX
    if (a.isGpuAllocated && bgfx::isValid(a.gpuBuffer)) {
        bgfx::destroy(a.gpuBuffer);
        a.gpuBuffer = BGFX_INVALID_HANDLE;
    }
#else
    (void)a;
#endif
}

inline cudaError_t release(void* ptr) {
    if (!ptr) return cudaSuccess;
    RuntimeState& rt = state();
    auto it = rt.allocations.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == rt.allocations.end()) {
        return fail(cudaErrorInvalidValue, ErrorCode::BadPointer, "cudaFree: unknown pointer or not the start of an allocation");
    }
    destroyGpu(it->second);
    rt.allocations.erase(it);
    return cudaSuccess;
}

} // namespace detail

// ============================================================================
// CudaGFX API
// ============================================================================

inline void init(uint16_t computeViewId = 0) {
    detail::state().computeViewId = computeViewId;
#if CUDAGFX_HAS_BGFX
    bgfx::setViewMode(computeViewId, bgfx::ViewMode::Sequential);
    const bgfx::Caps* caps = bgfx::getCaps();
    if (caps && !(caps->supported & BGFX_CAPS_COMPUTE)) {
        detail::fail(cudaErrorInitializationError, ErrorCode::MissingShader,
                     "the bgfx renderer '%s' does not support compute shaders", bgfx::getRendererName(bgfx::getRendererType()));
    }
#endif
}

// Frees buffers, programs and uniforms. To be called before bgfx::shutdown().
inline void shutdown() {
    detail::RuntimeState& rt = detail::state();
    for (auto& pair : rt.allocations) detail::destroyGpu(pair.second);
    rt.allocations.clear();
#if CUDAGFX_HAS_BGFX
    for (auto& p : rt.programs) {
        if (bgfx::isValid(p.second)) bgfx::destroy(p.second);
    }
    rt.programs.clear();
    for (auto& u : rt.uniforms) {
        if (bgfx::isValid(u.second.handle)) bgfx::destroy(u.second.handle);
    }
    rt.uniforms.clear();
#endif
    rt.lastError = cudaSuccess;
    rt.detailCode = ErrorCode::None;
    rt.detailMessage[0] = '\0';
    rt.pendingDispatches = 0;
}

inline void flush() { detail::flushPending(); }

inline ErrorCode lastErrorCode() { return detail::state().detailCode; }
inline const char* lastErrorMessage() { return detail::state().detailMessage; }

// ============================================================================
// Launching a kernel
// ============================================================================

class Launch {
public:
    explicit Launch(const ShaderBinaryTable& table) : table_(table) {}

    bool arg(float v) { return push(packer_.pushFloat(v)); }
    bool arg(int32_t v) { return push(packer_.pushInt(v)); }
    bool arg(uint32_t v) { return push(packer_.pushUInt(v)); }
    bool arg(bool v) { return push(packer_.pushBool(v)); }
    bool arg(const float2& v) { return pushVec(&v, 2); }
    bool arg(const float3& v) { return pushVec(&v, 3); }
    bool arg(const float4& v) { return pushVec(&v, 4); }
    bool arg(const int2& v) { return pushVec(&v, 2); }
    bool arg(const int3& v) { return pushVec(&v, 3); }
    bool arg(const int4& v) { return pushVec(&v, 4); }
    bool arg(const uint2& v) { return pushVec(&v, 2); }
    bool arg(const uint3& v) { return pushVec(&v, 3); }
    bool arg(const uint4& v) { return pushVec(&v, 4); }
    // Any other type (double, 64-bit integers...) is rejected at compile time rather than silently converted.
    template <typename T> bool arg(T) = delete;

    template <typename T>
    bool buffer(uint8_t slot, const T* devPtr, Access access) {
        if (failed_) return false;
        detail::RuntimeState& rt = detail::state();
        if (!devPtr) {
            return fatal(detail::fail(cudaErrorInvalidValue, ErrorCode::BadPointer, "kernel '%s': null pointer for the buffer of slot %u", table_.kernel, slot));
        }
        const uintptr_t addr = reinterpret_cast<uintptr_t>(devPtr);
        detail::Allocation* a = rt.findAllocation(addr);
        if (!a) {
            return fatal(detail::fail(cudaErrorInvalidValue, ErrorCode::BadPointer,
                                      "kernel '%s': the pointer of slot %u was not obtained from cudaMalloc (or has been freed)", table_.kernel, slot));
        }
        const size_t offset = static_cast<size_t>(addr - a->baseAddress);
        if (offset % sizeof(T) != 0) {
            return fatal(detail::fail(cudaErrorInvalidValue, ErrorCode::BadPointer,
                                      "kernel '%s': pointer of slot %u is not aligned to the element size (%u bytes)", table_.kernel, slot,
                                      static_cast<unsigned>(sizeof(T))));
        }
        if (offset % detail::kBufferViewAlign != 0) {
            return fatal(detail::fail(cudaErrorInvalidValue, ErrorCode::BadPointer,
                                      "kernel '%s': pointer of slot %u is offset by %.0f bytes inside its buffer; bgfx requires an offset that is a multiple of %u "
                                      "(avoid pointer arithmetic on the buffers passed to a kernel)",
                                      table_.kernel, slot, static_cast<double>(offset), static_cast<unsigned>(detail::kBufferViewAlign)));
        }
        if (a->sizeInBytes % sizeof(T) != 0) {
            return fatal(detail::fail(cudaErrorInvalidValue, ErrorCode::ElementSize,
                                      "kernel '%s': buffer of %.0f bytes, not a multiple of the element size (%u)", table_.kernel,
                                      static_cast<double>(a->sizeInBytes), static_cast<unsigned>(sizeof(T))));
        }
        constexpr uint32_t typeCode = detail::TypeHelper<T>::code;
        if (a->elementSize != 0 && a->elementTypeCode != typeCode) {
            return fatal(detail::fail(cudaErrorInvalidValueType, ErrorCode::TypeReinterpret,
                                      "kernel '%s': this buffer was already used with another element type (reinterpretation is not allowed)", table_.kernel));
        }
        for (const auto& b : bound_) {
            if (b.alloc == a && !(b.access == Access::Read && access == Access::Read)) {
                return fatal(detail::fail(cudaErrorInvalidValue, ErrorCode::PointerAlias,
                                          "kernel '%s': the same buffer is passed twice (slots %u and %u) with at least one write: aliasing is not allowed",
                                          table_.kernel, b.slot, slot));
            }
        }
#if CUDAGFX_HAS_BGFX
        const bgfx::Caps* caps = bgfx::getCaps();
        if (caps && slot >= caps->limits.maxComputeBindings) {
            return fatal(detail::fail(cudaErrorLaunchOutOfResources, ErrorCode::MissingShader,
                                      "kernel '%s': buffer slot %u, the renderer only accepts %u compute bindings", table_.kernel, slot,
                                      static_cast<unsigned>(caps->limits.maxComputeBindings)));
        }
        if (!a->isGpuAllocated) {
            constexpr unsigned comp = detail::TypeHelper<T>::components;
            bgfx::VertexLayout layout;
            layout.begin();
            if constexpr (typeCode >= 20) layout.add(bgfx::Attrib::TexCoord0, comp, bgfx::AttribType::Uint32);
            else if constexpr (typeCode >= 10) layout.add(bgfx::Attrib::TexCoord0, comp, bgfx::AttribType::Int32);
            else layout.add(bgfx::Attrib::TexCoord0, comp, bgfx::AttribType::Float);
            layout.end();
            const bgfx::Memory* mem = bgfx::copy(a->shadowBuffer.data(), static_cast<uint32_t>(a->sizeInBytes));
            a->gpuBuffer = bgfx::createDynamicVertexBuffer(mem, layout, BGFX_BUFFER_COMPUTE_READ_WRITE);
            a->isGpuAllocated = true;
            a->shadowBuffer.clear();
            a->shadowBuffer.shrink_to_fit();
        }
#endif
        a->elementTypeCode = typeCode;
        a->elementSize = sizeof(T);

        BoundBuffer bb;
        bb.slot = slot;
        bb.alloc = a;
        bb.offset = static_cast<uint32_t>(offset);
        bb.access = access;
        bound_.push_back(bb);
        return true;
    }

    // `blk` must match the block compiled into the shader (CGFXR006).
    void dispatch(dim3 grid, dim3 blk) {
        detail::RuntimeState& rt = detail::state();
        if (failed_) return; // the error has already been recorded; an incomplete launch is not executed
        if (blk.x != table_.block[0] || blk.y != table_.block[1] || blk.z != table_.block[2]) {
            detail::fail(cudaErrorInvalidConfiguration, ErrorCode::BadBlock,
                         "kernel '%s': block %ux%ux%u requested, the shader was compiled for %ux%ux%u", table_.kernel, blk.x, blk.y, blk.z,
                         table_.block[0], table_.block[1], table_.block[2]);
            return;
        }
        if (grid.x == 0 || grid.y == 0 || grid.z == 0 || grid.x > 65535 || grid.y > 65535 || grid.z > 65535) {
            detail::fail(cudaErrorInvalidConfiguration, ErrorCode::BadBlock,
                         "kernel '%s': invalid grid %ux%ux%u (each dimension must be between 1 and 65535)", table_.kernel, grid.x, grid.y, grid.z);
            return;
        }
        if (packer_.failed()) {
            detail::fail(cudaErrorInvalidValue, ErrorCode::ArgOverflow, "kernel '%s': too many by-value arguments", table_.kernel);
            return;
        }
#if CUDAGFX_HAS_BGFX
        const bgfx::RendererType::Enum renderer = bgfx::getRendererType();
        if (renderer == bgfx::RendererType::Noop) return; // nothing to execute
        const EmbeddedShaderRef* ref = nullptr;
        switch (renderer) {
            case bgfx::RendererType::Vulkan: ref = &table_.spv; break;
            case bgfx::RendererType::OpenGL: ref = &table_.glsl; break;
            case bgfx::RendererType::OpenGLES: ref = &table_.essl; break;
            case bgfx::RendererType::Direct3D11: ref = &table_.dxbc; break;
            case bgfx::RendererType::Direct3D12: ref = &table_.dxil; break;
            case bgfx::RendererType::Metal: ref = &table_.mtl; break;
            case bgfx::RendererType::WebGPU: ref = &table_.wgsl; break;
            default: break;
        }
        if (!ref || !ref->data || ref->size == 0) {
            detail::fail(cudaErrorInvalidDeviceFunction, ErrorCode::MissingShader,
                         "kernel '%s': no shader compiled for the bgfx backend '%s' (add the matching target to --target)", table_.kernel,
                         bgfx::getRendererName(renderer));
            return;
        }

        bgfx::ProgramHandle program = BGFX_INVALID_HANDLE;
        const auto cached = rt.programs.find(ref->data);
        if (cached != rt.programs.end()) {
            program = cached->second;
        } else {
            bgfx::ShaderHandle sh = bgfx::createShader(bgfx::makeRef(ref->data, ref->size));
            if (bgfx::isValid(sh)) program = bgfx::createProgram(sh, true);
            if (!bgfx::isValid(program)) {
                detail::fail(cudaErrorInvalidDeviceFunction, ErrorCode::MissingShader,
                             "kernel '%s': bgfx rejected the shader (backend '%s')", table_.kernel, bgfx::getRendererName(renderer));
                return;
            }
            rt.programs.emplace(ref->data, program);
        }

        for (const auto& b : bound_) {
            bgfx::Access::Enum acc = bgfx::Access::ReadWrite;
            if (b.access == Access::Read) acc = bgfx::Access::Read;
            else if (b.access == Access::Write) acc = bgfx::Access::Write;
            bgfx::setBuffer(b.slot, b.alloc->gpuBuffer, acc, b.offset, static_cast<uint32_t>(b.alloc->sizeInBytes - b.offset));
        }

        if (packer_.slotCount() > 0 && table_.argsUniform[0] != '\0') {
            if (bgfx::isValid(uniformFor(table_.argsUniform, static_cast<uint16_t>(packer_.slotCount())))) {
                bgfx::setUniform(rt.uniforms[table_.argsUniform].handle, packer_.data(), static_cast<uint16_t>(packer_.slotCount()));
            } else {
                return;
            }
        }
        if (table_.gridUniform[0] != '\0') {
            if (bgfx::isValid(uniformFor(table_.gridUniform, 1))) {
                uint32_t bits[4] = {grid.x, grid.y, grid.z, 0};
                float lanes[4];
                std::memcpy(lanes, bits, sizeof(lanes)); // gridDim travels as a bit-cast uint, like the other arguments
                bgfx::setUniform(rt.uniforms[table_.gridUniform].handle, lanes, 1);
            }
        }

        bgfx::dispatch(rt.computeViewId, program, grid.x, grid.y, grid.z);
        ++rt.pendingDispatches;
#else
        detail::fail(cudaErrorInitializationError, ErrorCode::MissingShader,
                     "kernel '%s': runtime compiled without bgfx (<bgfx/bgfx.h> not found or CUDAGFX_NO_BGFX defined): the kernel was not executed",
                     table_.kernel);
        (void)rt;
#endif
    }

private:
    struct BoundBuffer {
        uint8_t slot;
        detail::Allocation* alloc;
        uint32_t offset;
        Access access;
    };

    bool push(bool ok) {
        if (!ok) {
            failed_ = true;
            detail::fail(cudaErrorInvalidValue, ErrorCode::ArgOverflow,
                         "kernel '%s': too many by-value arguments (maximum %u 32-bit words)", table_.kernel, abi::kMaxArgLanes);
        }
        return ok;
    }
    bool pushVec(const void* v, unsigned comps) { return push(packer_.pushWords(v, comps)); }
    bool fatal(cudaError_t) {
        failed_ = true;
        return false;
    }

#if CUDAGFX_HAS_BGFX
    static bgfx::UniformHandle uniformFor(const char* name, uint16_t slots) {
        detail::RuntimeState& rt = detail::state();
        auto it = rt.uniforms.find(name);
        if (it != rt.uniforms.end()) {
            if (it->second.slots != slots) {
                detail::fail(cudaErrorInvalidValue, ErrorCode::ArgOverflow,
                             "uniform '%s': %u slot(s) provided, %u expected by the shader", name, static_cast<unsigned>(slots),
                             static_cast<unsigned>(it->second.slots));
                return BGFX_INVALID_HANDLE;
            }
            return it->second.handle;
        }
        detail::RuntimeState::UniformEntry e;
        e.handle = bgfx::createUniform(name, bgfx::UniformType::Vec4, slots);
        e.slots = slots;
        rt.uniforms.emplace(name, e);
        return e.handle;
    }
#endif

    ShaderBinaryTable table_;
    abi::ArgPacker packer_;
    std::vector<BoundBuffer> bound_;
    bool failed_ = false;
};

} // namespace cudagfx

// ============================================================================
// Emulated CUDA host API
// ============================================================================

inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!devPtr) return cudagfx::detail::fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMalloc: null devPtr");
    *devPtr = cudagfx::detail::allocate(size);
    if (size != 0 && !*devPtr) return cudagfx::detail::state().lastError;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* devPtr) { return cudagfx::detail::release(devPtr); }

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    using namespace cudagfx::detail;
    RuntimeState& rt = state();
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMemcpy: null pointer");

    Allocation* d = rt.findAllocation(reinterpret_cast<uintptr_t>(dst));
    Allocation* s = rt.findAllocation(reinterpret_cast<uintptr_t>(src));
    if (kind == cudaMemcpyDefault) {
        kind = d ? (s ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : (s ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    }
    auto offsetIn = [](const Allocation* a, const void* p) { return static_cast<size_t>(reinterpret_cast<uintptr_t>(p) - a->baseAddress); };

    switch (kind) {
        case cudaMemcpyHostToHost:
            std::memmove(dst, src, count);
            return cudaSuccess;
        case cudaMemcpyHostToDevice:
            if (!d) return fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMemcpy(HostToDevice): destination unknown to cudaMalloc");
            return deviceWrite(*d, offsetIn(d, dst), src, count);
        case cudaMemcpyDeviceToHost:
            if (!s) return fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMemcpy(DeviceToHost): source unknown to cudaMalloc");
            return deviceRead(*s, offsetIn(s, src), count, dst);
        case cudaMemcpyDeviceToDevice: {
            if (!d || !s) return fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMemcpy(DeviceToDevice): pointer unknown to cudaMalloc");
            const size_t so = offsetIn(s, src), dof = offsetIn(d, dst);
            if (cudaError_t e = checkRange(*s, so, count)) return e;
            if (cudaError_t e = checkRange(*d, dof, count)) return e;
#if CUDAGFX_HAS_BGFX
            if (s->isGpuAllocated && d->isGpuAllocated && s != d) {
                flushPending();
                bgfx::BufferRegion sr, dr;
                sr.init(s->gpuBuffer, static_cast<uint32_t>(so), static_cast<uint32_t>(count));
                dr.init(d->gpuBuffer, static_cast<uint32_t>(dof), static_cast<uint32_t>(count));
                bgfx::blit(rt.computeViewId, dr, sr);
                ++rt.pendingDispatches;
                return cudaSuccess;
            }
#endif
            std::vector<uint8_t> tmp(count);
            if (cudaError_t e = deviceRead(*s, so, count, tmp.data())) return e;
            return deviceWrite(*d, dof, tmp.data(), count);
        }
        default:
            return fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMemcpy: invalid copy kind");
    }
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    using namespace cudagfx::detail;
    RuntimeState& rt = state();
    Allocation* a = devPtr ? rt.findAllocation(reinterpret_cast<uintptr_t>(devPtr)) : nullptr;
    if (!a) return fail(cudaErrorInvalidValue, cudagfx::ErrorCode::BadPointer, "cudaMemset: pointer unknown to cudaMalloc");
    if (count == 0) return cudaSuccess;
    const size_t offset = static_cast<size_t>(reinterpret_cast<uintptr_t>(devPtr) - a->baseAddress);
    if (cudaError_t e = checkRange(*a, offset, count)) return e;
    if (!a->isGpuAllocated) {
        std::memset(a->shadowBuffer.data() + offset, value, count);
        return cudaSuccess;
    }
    std::vector<uint8_t> tmp(count, static_cast<uint8_t>(value));
    return deviceWrite(*a, offset, tmp.data(), count);
}

inline cudaError_t cudaDeviceSynchronize() {
    cudagfx::detail::flushPending();
    return cudaSuccess;
}

inline cudaError_t cudaGetLastError() {
    cudagfx::detail::RuntimeState& rt = cudagfx::detail::state();
    const cudaError_t err = rt.lastError;
    rt.lastError = cudaSuccess;
    rt.detailCode = cudagfx::ErrorCode::None;
    rt.detailMessage[0] = '\0';
    return err;
}

inline cudaError_t cudaPeekAtLastError() { return cudagfx::detail::state().lastError; }

inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorInvalidValueType: return "cudaErrorInvalidValueType";
        default: return "cudaErrorUnknown";
    }
}
