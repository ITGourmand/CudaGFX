__global__ void k(const unsigned char* img, float* y) {
    y[0] = img[0] / 255.0f;
}
// EXPECT error CGFX0012 1:40
