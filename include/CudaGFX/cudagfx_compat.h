// cudagfx_compat.h: minimal CUDA types and qualifiers, for the IDE and for the generated host code.
//
// The qualifiers are empty: a `__host__ __device__` function kept on the host side compiles as standard C++17.
// (No threadIdx/blockIdx built-ins here: they only exist inside kernels.)
#pragma once

#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __noinline__
#define __noinline__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif

// Vector types (same names, fields and alignments as CUDA: 2-component vectors are 8-byte aligned, 4-component
// vectors 16-byte aligned, 3-component vectors 4-byte aligned). The layout matters: a struct that contains them
// is read by the shaders at the offsets computed by cudagfx (see StructLayout), and the generated host code
// static_assert's that the C++ compiler agrees.
struct alignas(8) float2 { float x, y; };
struct float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };

struct alignas(8) int2 { int x, y; };
struct int3 { int x, y, z; };
struct alignas(16) int4 { int x, y, z, w; };

struct alignas(8) uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct alignas(16) uint4 { unsigned int x, y, z, w; };

static_assert(sizeof(float2) == 8 && sizeof(float3) == 12 && sizeof(float4) == 16, "float vector sizes");
static_assert(sizeof(int2) == 8 && sizeof(int3) == 12 && sizeof(int4) == 16, "int vector sizes");
static_assert(sizeof(uint2) == 8 && sizeof(uint3) == 12 && sizeof(uint4) == 16, "uint vector sizes");

inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

inline int2 make_int2(int x, int y) { return int2{x, y}; }
inline int3 make_int3(int x, int y, int z) { return int3{x, y, z}; }
inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }

inline uint2 make_uint2(unsigned int x, unsigned int y) { return uint2{x, y}; }
inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return uint3{x, y, z}; }
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return uint4{x, y, z, w}; }
