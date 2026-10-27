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

// Vector types (same names and fields as CUDA).
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };

struct uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct uint4 { unsigned int x, y, z, w; };

inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

inline int2 make_int2(int x, int y) { return int2{x, y}; }
inline int3 make_int3(int x, int y, int z) { return int3{x, y, z}; }
inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }

inline uint2 make_uint2(unsigned int x, unsigned int y) { return uint2{x, y}; }
inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return uint3{x, y, z}; }
inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return uint4{x, y, z, w}; }
