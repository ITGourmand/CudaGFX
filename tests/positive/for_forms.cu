// for: initialization by expression, several declarators, several increments.
__global__ void for_forms(float* y, int n) {
    int i = threadIdx.x;
    int j;
    for (j = 0; j < 4; j++) y[i] += 1.0f;
    for (int a = 0, b = 10; a < b; a++, b--) y[i] += (float)b;
    int w = 0;
    while (w < 3) { w++; if (w == 2) continue; y[i] += 1.f; }
    do { w--; } while (w > 0);
}
