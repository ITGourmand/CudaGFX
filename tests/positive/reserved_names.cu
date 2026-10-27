// User identifiers that are GLSL/HLSL reserved words: renamed by the tool.
__global__ void reserved_names(float* output, const float* input, int common) {
    float sample = input[0];
    float filter = sample * 2.f;
    int half = common + 1;
    float mod = float(half);
    output[0] = filter + mod;
}
