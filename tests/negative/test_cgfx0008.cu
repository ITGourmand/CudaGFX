__global__ void k(float* a) {}
int main() {
    float* a = nullptr;
    k<<<1, 2048>>>(a);
}
// EXPECT error CGFX0008 4:5
