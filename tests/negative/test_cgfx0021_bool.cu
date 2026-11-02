struct Bad {
    bool flag;
    int n;
};
__global__ void k(Bad* p) { p[0].n = 1; }
// EXPECT error CGFX0021 5:24
