// true/false, #pragma unroll, empty statement, functional cast.
__global__ void bool_pragma(float* y, int n) {
    bool b = true;
    int i = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (b && i < n) y[i] += float(j);
    }
    ;
    b = false;
}
