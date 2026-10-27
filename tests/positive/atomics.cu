// Atomics in statement form, with the old value, on a buffer and on shared memory.
__global__ void atomics(int* hist, const int* in, unsigned* counter, int n) {
    __shared__ int local[16];
    int i = threadIdx.x;
    if (i < 16) local[i] = 0;
    __syncthreads();
    int old = atomicAdd(&hist[in[i] & 15], 1);
    atomicMax(&hist[0], old);
    atomicAdd(&local[i % 16], 1);
    unsigned prev;
    prev = atomicExch(&counter[0], 7u);
    atomicSub(&hist[1], 2);
    atomicCAS(&hist[2], 0, 1);
    __syncthreads();
    if (i < 16) atomicAdd(&hist[i], local[i]);
}
