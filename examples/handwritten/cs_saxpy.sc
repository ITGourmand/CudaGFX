#include <bgfx_compute.sh>

// CudaGFX Phase 0: handwritten SAXPY compute shader used to validate bgfx / shaderc
uniform vec4 u_saxpy_args[1];

BUFFER_RO(x, float, 0);
BUFFER_RW(y, float, 1);

NUM_THREADS(256, 1, 1)
void main()
{
    float a = u_saxpy_args[0].x;
    int   n = floatBitsToInt(u_saxpy_args[0].y);

    int i = int(gl_WorkGroupID.x) * 256 + int(gl_LocalInvocationID.x);
    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}
