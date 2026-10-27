// Intrinsics.cpp
#include "Intrinsics.hpp"

#include <unordered_map>

namespace cudagfx {

namespace {

using K = Type::Kind;
using S = Intrinsic::Shape;

std::vector<Intrinsic> buildTable() {
    std::vector<Intrinsic> t;
    auto unary = [&](const std::string& cuda, const std::string& sh) {
        t.push_back({cuda, sh, S::FloatUnary, 1, K::Float, K::Float, 1, false});
    };
    auto binary = [&](const std::string& cuda, const std::string& sh, bool warn) {
        t.push_back({cuda, sh, S::FloatBinary, 2, K::Float, K::Float, 1, warn});
    };

    // Floating-point functions: suffixed form (sinf) and unsuffixed form applied to floats (sin).
    auto unaryBoth = [&](const std::string& base, const std::string& shader) {
        unary(base, shader);
        unary(base + "f", shader);
    };
    const char* const sameName[] = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                                    "exp", "exp2", "log", "log2", "sqrt", "floor", "ceil", "trunc", "round"};
    for (const char* n : sameName) unaryBoth(n, n);
    unaryBoth("fabs", "abs");
    unaryBoth("rsqrt", "inversesqrt");
    unary("__sinf", "sin");
    unary("__cosf", "cos");
    unary("__tanf", "tan");
    unary("__expf", "exp");
    unary("__logf", "log");
    unary("__log2f", "log2");
    unary("__exp2f", "exp2");

    auto binaryBoth = [&](const std::string& base, const std::string& shader, bool warn) {
        binary(base, shader, warn);
        binary(base + "f", shader, warn);
    };
    binaryBoth("fmin", "min", false);
    binaryBoth("fmax", "max", false);
    binaryBoth("pow", "pow", false);
    binaryBoth("atan2", "atan2", false);
    binaryBoth("fmod", "mod", true);
    binary("__powf", "pow", false);

    t.push_back({"min", "min", S::Generic, 2, K::Float, K::Float, 1, false});
    t.push_back({"max", "max", S::Generic, 2, K::Float, K::Float, 1, false});
    t.push_back({"abs", "abs", S::Generic, 1, K::Float, K::Float, 1, false});

    const char* const makeNames[] = {"float", "int", "uint"};
    const K makeKinds[] = {K::Float, K::Int, K::UInt};
    for (int k = 0; k < 3; ++k) {
        for (unsigned n = 2; n <= 4; ++n) {
            t.push_back({std::string("make_") + makeNames[k] + std::to_string(n), "", S::VectorMake, n,
                         makeKinds[k], makeKinds[k], n, false});
        }
    }

    t.push_back({"__float_as_int", "floatBitsToInt", S::Fixed, 1, K::Float, K::Int, 1, false});
    t.push_back({"__float_as_uint", "floatBitsToUint", S::Fixed, 1, K::Float, K::UInt, 1, false});
    t.push_back({"__int_as_float", "intBitsToFloat", S::Fixed, 1, K::Int, K::Float, 1, false});
    t.push_back({"__uint_as_float", "uintBitsToFloat", S::Fixed, 1, K::UInt, K::Float, 1, false});
    t.push_back({"__popc", kPopcountShaderName, S::Fixed, 1, K::UInt, K::Int, 1, false});
    t.push_back({"isnan", "isnan", S::Fixed, 1, K::Float, K::Bool, 1, false});
    t.push_back({"isinf", "isinf", S::Fixed, 1, K::Float, K::Bool, 1, false});
    return t;
}

} // namespace

const Intrinsic* findIntrinsic(const std::string& name) {
    static const std::vector<Intrinsic> table = buildTable();
    static const std::unordered_map<std::string, const Intrinsic*> index = [] {
        std::unordered_map<std::string, const Intrinsic*> m;
        for (const auto& i : table) m.emplace(i.cudaName, &i);
        return m;
    }();
    const auto it = index.find(name);
    return it == index.end() ? nullptr : it->second;
}

bool isWarpIntrinsic(const std::string& name) {
    static const char* const prefixes[] = {"__shfl", "__ballot", "__any_sync", "__all_sync", "__activemask",
                                           "__match_", "__reduce_", "__syncwarp", "__vote", "__lane", "warpSize"};
    for (const char* p : prefixes) {
        if (name.rfind(p, 0) == 0) return true;
    }
    return false;
}

bool isUnsupportedDeviceFunction(const std::string& name) {
    static const char* const exact[] = {"printf", "malloc", "free", "memcpy", "memset", "memmove", "assert",
                                        "__trap", "__brkpt", "abort", "exit", "calloc", "realloc", "puts",
                                        "__threadfence", "__threadfence_block", "__threadfence_system"};
    for (const char* e : exact) {
        if (name == e) return true;
    }
    return name.rfind("cuda", 0) == 0 || name.rfind("curand", 0) == 0 || name.rfind("cublas", 0) == 0;
}

bool parseAtomicName(const std::string& name, AtomicOp& op) {
    if (name == "atomicAdd") { op = AtomicOp::Add; return true; }
    if (name == "atomicSub") { op = AtomicOp::Sub; return true; }
    if (name == "atomicMin") { op = AtomicOp::Min; return true; }
    if (name == "atomicMax") { op = AtomicOp::Max; return true; }
    if (name == "atomicAnd") { op = AtomicOp::And; return true; }
    if (name == "atomicOr") { op = AtomicOp::Or; return true; }
    if (name == "atomicXor") { op = AtomicOp::Xor; return true; }
    if (name == "atomicExch") { op = AtomicOp::Exch; return true; }
    if (name == "atomicCAS") { op = AtomicOp::Cas; return true; }
    return false;
}

} // namespace cudagfx
