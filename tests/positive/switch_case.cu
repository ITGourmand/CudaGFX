// switch / case / default, fallthrough and break.
__global__ void switch_case(int* y, int n) {
    int i = threadIdx.x;
    switch (y[i]) {
        case 0:
        case 1:
            y[i] = 10;
            break;
        case 2: y[i] = 20; break;
        default:
            y[i] = -1;
    }
}
