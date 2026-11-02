// GPU validation: initialises bgfx without a window (headless: the swap chain's window handle is NULL, which bgfx
// accepts "provided the rendering API supports it"), runs the checks of gpu_checks.cu on the GPU and on the CPU
// backend, and exits with the number of failures.
//
//   gpu_validation [--renderer vulkan|opengl|d3d11|d3d12|metal] [--only basics|life|particles|algorithms]
//
// Exit code: 0 all checks passed, 1 a check failed, 2 the GPU could not be initialised (the test suite reports
// that as "skipped": there is nothing to run on).
//
// If bgfx::init fails for the renderer you asked for, that renderer cannot run headless here: give bgfx a window
// (init.swapChain.nwh / ndt, see bgfx::Init in bgfx.h) in this file.
//
// Built with -DCUDAGFX_CPU_ONLY (no bgfx) it only runs the CPU checks: that is how the automated suite executes it.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_checks.h"

#ifndef CUDAGFX_CPU_ONLY

#include <bgfx/bgfx.h>

#include "CudaGFX/cudagfx_rt.h"

#if defined(__linux__)
#include <dlfcn.h>
#endif

// What bgfx says while it initialises (it reports why a renderer cannot start through this callback): kept, and shown
// when bgfx::init fails.
class MessageLog : public bgfx::CallbackI {
public:
    ~MessageLog() override {}
    void fatal(const char*, uint16_t, bgfx::Fatal::Enum, const char* text) override { add(std::string("fatal: ") + text); }
    void traceVargs(const char*, uint16_t, const char* format, va_list args) override {
        char buffer[512];
        std::vsnprintf(buffer, sizeof(buffer), format, args);
        add(buffer);
    }
    void profilerBegin(const char*, uint32_t, const char*, uint16_t) override {}
    void profilerBeginLiteral(const char*, uint32_t, const char*, uint16_t) override {}
    void profilerEnd() override {}
    uint32_t cacheReadSize(uint64_t) override { return 0; }
    bool cacheRead(uint64_t, void*, uint32_t) override { return false; }
    void cacheWrite(uint64_t, const void*, uint32_t) override {}
    void screenShot(const char*, uint32_t, uint32_t, uint32_t, bgfx::TextureFormat::Enum, const void*, uint32_t, bool) override {}
    void captureBegin(uint32_t, uint32_t, uint32_t, bgfx::TextureFormat::Enum, bool) override {}
    void captureEnd() override {}
    void captureFrame(const void*, uint32_t) override {}

    void print(const char* indent) const {
        for (const std::string& line : lines_) std::fprintf(stderr, "%sbgfx: %s", indent, line.c_str());
    }
    bool empty() const { return lines_.empty(); }

private:
    void add(std::string line) {
        if (line.empty() || line.back() != '\n') line += '\n';
        if (lines_.size() < 24) lines_.push_back(std::move(line));
    }
    std::vector<std::string> lines_;
};

// bgfx loads the graphics libraries at run time (dlopen): when it cannot start, the first thing to know is whether
// the system can find them. On NixOS they are only found through LD_LIBRARY_PATH (/run/opengl-driver/lib, the
// vulkan-loader and libGL packages...).
static void explainInitFailure() {
    const char* display = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    const char* libraryPath = std::getenv("LD_LIBRARY_PATH");
    std::fprintf(stderr, "  DISPLAY=%s WAYLAND_DISPLAY=%s LD_LIBRARY_PATH=%s\n", display ? display : "(unset)", wayland ? wayland : "(unset)",
                 libraryPath && *libraryPath ? libraryPath : "(unset)");
#if defined(__linux__)
    const char* libraries[] = {"libvulkan.so.1", "libGL.so.1", "libEGL.so.1", "libX11.so.6"};
    for (const char* name : libraries) {
        void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
        if (handle) {
            std::fprintf(stderr, "  %-16s found\n", name);
            dlclose(handle);
        } else {
            const char* why = dlerror();
            std::fprintf(stderr, "  %-16s NOT FOUND (%s)\n", name, why ? why : "?");
        }
    }
#endif
}

static int runSuites(bool useGpu, const char* only);

static bgfx::RendererType::Enum parseRenderer(const char* name) {
    if (!std::strcmp(name, "vulkan")) return bgfx::RendererType::Vulkan;
    if (!std::strcmp(name, "opengl")) return bgfx::RendererType::OpenGL;
    if (!std::strcmp(name, "d3d11")) return bgfx::RendererType::Direct3D11;
    if (!std::strcmp(name, "d3d12")) return bgfx::RendererType::Direct3D12;
    if (!std::strcmp(name, "metal")) return bgfx::RendererType::Metal;
    return bgfx::RendererType::Count; // let bgfx choose
}

int main(int argc, char** argv) {
    bgfx::RendererType::Enum renderer = bgfx::RendererType::Count;
    const char* only = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--renderer")) renderer = parseRenderer(argv[i + 1]);
        if (!std::strcmp(argv[i], "--only")) only = argv[i + 1];
    }

    bgfx::renderFrame(); // called before init: bgfx runs single-threaded, this thread is the render thread
    MessageLog messages;
    bgfx::Init init;
    init.callback = &messages;
    init.type = renderer;
    // Headless: no window handle, and then bgfx REQUIRES a 0x0 back buffer (bgfx::init returns false at once, without a
    // message, if the swap chain has a size and no window).
    init.swapChain.nwh = nullptr;
    init.swapChain.ndt = nullptr;
    init.swapChain.width = 0;
    init.swapChain.height = 0;
    if (!bgfx::init(init)) {
        std::fprintf(stderr, "bgfx::init failed (renderer: %s)\n", renderer == bgfx::RendererType::Count ? "default" : bgfx::getRendererName(renderer));
        messages.print("  ");
        explainInitFailure();
        return 2;
    }
    if (bgfx::getRendererType() == bgfx::RendererType::Noop) {
        // bgfx falls back to the Noop renderer when the one that was asked for cannot start: nothing can run on it.
        std::fprintf(stderr, "bgfx::init fell back to the Noop renderer (renderer asked: %s)\n",
                     renderer == bgfx::RendererType::Count ? "default" : bgfx::getRendererName(renderer));
        messages.print("  ");
        explainInitFailure();
        bgfx::shutdown();
        return 2;
    }
    const bgfx::Caps* caps = bgfx::getCaps();
    std::printf("renderer: %s, vendor 0x%04x, device 0x%04x, compute %s, %u compute bindings\n", bgfx::getRendererName(bgfx::getRendererType()),
                caps ? caps->vendorId : 0u, caps ? caps->deviceId : 0u, caps && (caps->supported & BGFX_CAPS_COMPUTE) ? "yes" : "NO",
                caps ? caps->limits.maxComputeBindings : 0u);

    cudagfx::init(); // records an error at once if the renderer cannot run compute shaders
    const int failures = runSuites(true, only);

    cudagfx::shutdown();
    bgfx::shutdown();
    std::printf(failures ? "%d check(s) FAILED\n" : "all checks passed\n", failures);
    return failures ? 1 : 0;
}

#else // CUDAGFX_CPU_ONLY

static int runSuites(bool useGpu, const char* only);

int main() {
    const int failures = runSuites(false, nullptr);
    std::printf(failures ? "%d check(s) FAILED\n" : "all CPU checks passed\n", failures);
    return failures ? 1 : 0;
}

#endif

// The suites, in the order of increasing complexity. `only` selects one of them.
static int runSuites(bool useGpu, const char* only) {
    int failures = 0;
    if (!only || !std::strcmp(only, "basics")) failures += runGpuChecks(useGpu);
    if (!only || !std::strcmp(only, "life")) failures += runLifeChecks(useGpu);
    if (!only || !std::strcmp(only, "particles")) failures += runParticleChecks(useGpu);
    if (!only || !std::strcmp(only, "algorithms")) failures += runAlgorithmChecks(useGpu);
    return failures;
}
