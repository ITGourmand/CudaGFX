// Must FAIL to compile: a double argument is never silently converted (Launch::arg is deleted).
#include "CudaGFX/cudagfx_rt.h"
int main() {
    cudagfx::Launch l(cudagfx::ShaderBinaryTable{});
    l.arg(2.0);
    return 0;
}
