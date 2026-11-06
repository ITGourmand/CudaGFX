// A parenthesised variable whose name is also a type name stays an expression, not a cast.
__global__ void cast_type_names(float* y) {
    float half = 0.5f;
    y[0] = (half) * y[1] - (half);
}
