// Unit tests of the CudaGFX front-end: no framework, one CHECK per assertion, non-zero exit code on failure.
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "CudaGFX/ConstEval.hpp"
#include "CudaGFX/Diagnostics.hpp"
#include "CudaGFX/Lexer.hpp"
#include "CudaGFX/Literal.hpp"
#include "CudaGFX/Parser.hpp"
#include "CudaGFX/Preprocessor.hpp"
#include "CudaGFX/Sema.hpp"
#include "CudaGFX/ShaderEmitter.hpp"
#include "CudaGFX/ShaderNames.hpp"
#include "CudaGFX/ShaderTargets.hpp"
#include "CudaGFX/ShadercDriver.hpp"
#include "CudaGFX/SourceManager.hpp"
#include "CudaGFX/Variants.hpp"
#include "CudaGFX/cudagfx_abi.h"

using namespace cudagfx;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond)                                                                           \
    do {                                                                                      \
        ++g_checks;                                                                           \
        if (!(cond)) {                                                                        \
            std::cout << "FAILED " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";         \
            ++g_failures;                                                                     \
        }                                                                                     \
    } while (0)

namespace {

std::vector<Token> lex(const std::string& src, DiagnosticsEngine& d, SourceManager& sm) {
    d.setSourceManager(&sm);
    const FileId f = sm.add("t.cu", src);
    Lexer l(f, sm.content(f), d);
    return l.tokenize();
}

std::string preprocess(const std::string& src, DiagnosticsEngine& d, SourceManager& sm, bool expandAll = true,
                       std::vector<std::string> includes = {}, const char* defName = nullptr, const char* defValue = nullptr) {
    d.setSourceManager(&sm);
    Preprocessor pp(d, sm, std::move(includes));
    if (defName) pp.defineMacro(defName, defValue);
    const FileId f = sm.add("/tmp/cgfx_unit_main.cu", src);
    auto toks = pp.process(f);
    if (expandAll) toks = pp.expand(toks);
    std::string out;
    for (const auto& t : toks) {
        if (t.type == TokenType::EndOfFile) break;
        if (!out.empty()) out += ' ';
        out += t.text;
    }
    return out;
}

struct Frontend {
    SourceManager sm;
    DiagnosticsEngine diags{&sm};
    std::unique_ptr<Preprocessor> pp;
    ParsedModule module;
    bool ok = false;
};

// Runs the preprocessor + parser + Sema on `src`.
void runFrontend(const std::string& src, Frontend& f) {
    const FileId id = f.sm.add("unit.cu", src);
    f.pp = std::make_unique<Preprocessor>(f.diags, f.sm);
    auto toks = f.pp->process(id);
    Parser parser(*f.pp, f.sm, std::move(toks), id, f.diags);
    f.module = parser.parse();
    if (!f.diags.hasErrors()) Sema(f.diags).analyze(f.module);
    f.ok = !f.diags.hasErrors();
}

bool hasCode(const Frontend& f, DiagCode c) {
    for (const auto& d : f.diags.diagnostics()) {
        if (d.code == c) return true;
    }
    return false;
}

void testLexer() {
    {
        SourceManager sm; DiagnosticsEngine d;
        auto t = lex("x = 1.f + 2. * 1e-3f + 0x1Fu + 1'000;", d, sm);
        CHECK(!d.hasErrors());
        CHECK(t[2].type == TokenType::FloatLiteral && t[2].text == "1.f");
        CHECK(t[4].type == TokenType::FloatLiteral && t[4].text == "2.");
        CHECK(t[6].type == TokenType::FloatLiteral && t[6].text == "1e-3f");
        CHECK(t[8].type == TokenType::IntLiteral && t[8].text == "0x1Fu");
        CHECK(t[10].type == TokenType::IntLiteral && t[10].text == "1'000");
    }
    {   // >>> only closes a launch if one is open
        SourceManager sm; DiagnosticsEngine d;
        auto t = lex("std::vector<std::vector<std::vector<int>>> v; k<<<g, b>>>(x);", d, sm);
        int triple = 0, shift = 0;
        for (const auto& k : t) {
            if (k.type == TokenType::TripleGreater) ++triple;
            if (k.type == TokenType::GreaterGreater) ++shift;
        }
        CHECK(triple == 1);
        CHECK(shift == 1);
    }
    {   // line continuation: the directive covers the whole definition
        SourceManager sm; DiagnosticsEngine d;
        auto t = lex("#define M(a) \\\n   a + 1\nint x;\n#include <cuda_runtime.h>\n  # pragma unroll\n", d, sm);
        CHECK(t[0].type == TokenType::PpDefine && t[0].atLineStart);
        std::size_t i = 1;
        while (!t[i].atLineStart) ++i;
        CHECK(t[i].type == TokenType::KwInt);
        bool header = false, pragma = false;
        for (const auto& k : t) {
            if (k.type == TokenType::HeaderName && k.text == "<cuda_runtime.h>") header = true;
            if (k.type == TokenType::PpPragma) pragma = true;
        }
        CHECK(header);
        CHECK(pragma);
    }
    {   SourceManager sm; DiagnosticsEngine d; lex("int x; /* unterminated", d, sm); CHECK(d.errorCount() == 1); }
    {   SourceManager sm; DiagnosticsEngine d; lex("const char* s = \"abc\n int y; @", d, sm); CHECK(d.errorCount() == 2); }
    {   // a // comment continues on the next line after a backslash
        SourceManager sm; DiagnosticsEngine d;
        auto t = lex("// comment \\\n still a comment\nint z;", d, sm);
        CHECK(t[0].type == TokenType::KwInt);
        CHECK(!d.hasErrors());
    }
}

void testLiteralsAndConstEval() {
    CHECK(parseNumericLiteral("0xFFFFFFFF").kind == NumericLiteral::Kind::UInt);
    CHECK(parseNumericLiteral("4294967296").kind == NumericLiteral::Kind::Int64);
    CHECK(parseNumericLiteral("5ULL").kind == NumericLiteral::Kind::Int64);
    CHECK(parseNumericLiteral("1.f").kind == NumericLiteral::Kind::Float);
    CHECK(parseNumericLiteral("08").kind == NumericLiteral::Kind::Invalid);
    CHECK(parseNumericLiteral("0b101u").intValue == 5);
    CHECK(formatFloatLiteral(0.1) == "0.1");
    CHECK(formatFloatLiteral(1.0) == "1.0");
    CHECK(formatFloatLiteral(1e-3) == "0.001");
    CHECK(formatFloatLiteral(3.14159265358979) == "3.1415927");

    SourceManager sm; DiagnosticsEngine d;
    auto eval = [&](const std::string& src, std::int64_t& v, ConstEvaluator& ev) {
        auto t = lex(src, d, sm);
        return ev.evaluate(t.data(), t.data() + t.size() - 1, v);
    };
    std::int64_t v = 0;
    ConstEvaluator plain;
    CHECK(eval("1 + 2 * 3 == 7 && (1 << 4) - 1 > 14 ? 5 : 6", v, plain) && v == 5);
    CHECK(eval("10 - 2 - 3", v, plain) && v == 5);          // left associativity
    CHECK(eval("1 == 1 + 1", v, plain) && v == 0);          // precedence: + binds tighter than ==
    CHECK(eval("-5 / 2 + (7 % 4)", v, plain) && v == 1);
    CHECK(!eval("1 / 0", v, plain));
    CHECK(!eval("n * 2", v, plain) && !plain.error().empty());
    ConstEvaluator withN([](const std::string& n, std::int64_t& o) { if (n == "N") { o = 6; return true; } return false; });
    CHECK(eval("(int)(2 + N) * 16", v, withN) && v == 128);
}

void testPreprocessor() {
    {
        SourceManager sm; DiagnosticsEngine d;
        CHECK(preprocess("#define BS 128\n#define SQ(x) ((x)*(x))\n#define ADD(a,b) a + b\nint t[BS]; int y = SQ(BS+1) + ADD(1, SQ(2));", d, sm) ==
              "int t [ 128 ] ; int y = ( ( 128 + 1 ) * ( 128 + 1 ) ) + 1 + ( ( 2 ) * ( 2 ) ) ;");
        CHECK(!d.hasErrors());
    }
    {   // precedence in #if, defined(), #elif, #else
        SourceManager sm; DiagnosticsEngine d;
        CHECK(preprocess("#if 1 + 2 * 3 == 7 && defined(X) || A\nint a;\n#elif 1\nint b;\n#else\nint c;\n#endif\n", d, sm, true, {}, "X", "1") == "int a ;");
    }
    {   // lexical errors inside a skipped group: ignored
        SourceManager sm; DiagnosticsEngine d;
        CHECK(preprocess("#if 0\ndon't parse this ' \n#else\nint ok;\n#endif\n", d, sm) == "int ok ;");
        CHECK(!d.hasErrors());
    }
    {
        SourceManager sm; DiagnosticsEngine d;
        CHECK(preprocess("#pragma unroll\nint a;\n#pragma once\n#include <cuda_runtime.h>\n", d, sm) == "int a ;");
        CHECK(!d.hasErrors());
    }
    {   // variadic macros, bounded recursion
        SourceManager sm; DiagnosticsEngine d;
        CHECK(preprocess("#define V(x, ...) f(x, __VA_ARGS__)\nint z = V(1, 2, 3);\n#define REC REC + 1\nint r = REC;", d, sm) ==
              "int z = f ( 1 , 2 , 3 ) ; int r = REC + 1 ;");
        CHECK(!d.hasErrors());
    }
    {   // # and ## rejected, launch inside a macro rejected, unknown directive
        SourceManager sm; DiagnosticsEngine d;
        preprocess("#define CAT(a,b) a ## b\n#define LAUNCH k<<<1,2>>>()\n#if\n#endif\n#bogus\n", d, sm);
        CHECK(d.errorCount() == 4);
    }
    {   SourceManager sm; DiagnosticsEngine d; preprocess("#if 1\nint x;\n", d, sm); CHECK(d.errorCount() == 1); }
    {   // #include "..." followed, #pragma once honoured
        const std::string dir = "/tmp/cgfx_unit_inc";
        std::string cmd = "mkdir -p " + dir;
        CHECK(std::system(cmd.c_str()) == 0);
        { std::ofstream h(dir + "/h.h"); h << "#pragma once\n#define FROM_HEADER 7\n__device__ int fh(int a) { return a + FROM_HEADER; }\n"; }
        SourceManager sm; DiagnosticsEngine d;
        const std::string s = preprocess("#include \"h.h\"\n#include \"h.h\"\nint v = FROM_HEADER;", d, sm, true, {dir});
        CHECK(s.find("__device__ int fh") == 0);
        CHECK(s.find("int v = 7 ;") != std::string::npos);
        CHECK(!d.hasErrors());
    }
}

void testLaunchSites() {
    Frontend f;
    runFrontend(
        "__global__ void k(float* a, int n) { a[threadIdx.x] += 1.0f; }\n"
        "std::vector<std::vector<std::vector<int>>> nested;\n"
        "#define GRID 4\n"
        "const int BLOCK = 128;\n"
        "void h(float* d) {\n"
        "    dim3 blk(16, 4);\n"
        "    k<<<GRID,\n"
        "        BLOCK>>>(d,\n"
        "                 1024);\n"
        "    k<<<dim3(2), dim3(BLOCK / 2)>>>(d, 10);\n"
        "    k<<<1, blk>>>(d, 5);\n"
        "}\n", f);
    CHECK(f.ok);
    CHECK(f.module.launchSites.size() == 3);
    if (f.module.launchSites.size() == 3) {
        const auto& s0 = f.module.launchSites[0];
        CHECK(s0.gridExprText == "GRID");
        CHECK(s0.blockExprText == "BLOCK");
        CHECK(s0.evaluatedBlock[0] == 128 && s0.evaluatedBlock[1] == 1);
        CHECK(s0.argExprTexts.size() == 2);
        CHECK(f.module.launchSites[1].evaluatedBlock[0] == 64);
        CHECK(f.module.launchSites[2].evaluatedBlock[0] == 16 && f.module.launchSites[2].evaluatedBlock[1] == 4);
    }
    // Variants: one per distinct size
    const FunctionDecl* k = f.module.findFunction("k");
    CHECK(k != nullptr);
    if (k) CHECK(variantsOf(f.module, *k).size() == 3);

    // Non-constant size, modified variable: CGFX0006
    Frontend g;
    runFrontend("__global__ void k(float* a) { a[0] = 1.f; }\nvoid h(float* d, int n) { int b = 64; k<<<1, b>>>(d); k<<<1, n>>>(d); }\n", g);
    CHECK(hasCode(g, DiagCode::CGFX0006));
    Frontend m;
    runFrontend("__global__ void k(float* a) { a[0] = 1.f; }\nvoid h(float* d) { dim3 b(32); b.x = 64; k<<<1, b>>>(d); }\n", m);
    CHECK(hasCode(m, DiagCode::CGFX0006));
}

void testSemaTyping() {
    {   // vectors: never converted to integers
        Frontend f;
        runFrontend("__global__ void k(float4* y, float4 a, float s) { y[0] = y[0] + a * s; }\n", f);
        CHECK(f.ok);
        const FunctionDecl* k = f.module.findFunction("k");
        const std::string text = ShaderEmitter().emit(f.module, *k, BlockVariant{}, "unit.cu");
        CHECK(text.find("y[0] = y[0] + a * s;") != std::string::npos);
        CHECK(text.find("int(") == std::string::npos);
    }
    {   // return type of a device function honoured, closure sorted
        Frontend f;
        runFrontend("__device__ int g(int a) { return a; }\n__device__ int h(int a) { return g(a) + 1; }\n"
                    "__global__ void k(int* y) { y[0] = h(2); }\n", f);
        CHECK(f.ok);
        const FunctionDecl* k = f.module.findFunction("k");
        const auto& clo = f.module.helperClosure[k];
        CHECK(clo.size() == 2);
        if (clo.size() == 2) CHECK(clo[0]->name == "g" && clo[1]->name == "h");
    }
    {   // recursion, pointer alias, write to const, floating-point atomic
        Frontend f;
        runFrontend("__device__ int r(int n) { return n ? r(n - 1) : 0; }\n__global__ void k(const float* c, float* y) { float* q = y; y[0] = r(2); }\n", f);
        CHECK(hasCode(f, DiagCode::CGFX0005));
        CHECK(hasCode(f, DiagCode::CGFX0001));
        Frontend g;
        runFrontend("__global__ void k(const float* c, float* y) { c[0] = 1.f; atomicAdd(&y[0], 1.f); }\n", g);
        CHECK(hasCode(g, DiagCode::CGFX0019));
        CHECK(hasCode(g, DiagCode::CGFX0015));
    }
    {   // reserved words renamed
        Frontend f;
        runFrontend("__global__ void k(float* output, const float* input) { float sample = input[0]; output[0] = sample; }\n", f);
        CHECK(f.ok);
        const std::string text = ShaderEmitter().emit(f.module, *f.module.findFunction("k"), BlockVariant{}, "unit.cu");
        CHECK(text.find("cgfx_output") != std::string::npos && text.find("cgfx_sample") != std::string::npos);
    }
    {   // unsigned built-ins, gridDim declared
        Frontend f;
        runFrontend("__global__ void k(int* y) { int i = blockIdx.x * blockDim.x + threadIdx.x; y[i] = gridDim.x; }\n", f);
        CHECK(f.ok);
        const std::string text = ShaderEmitter().emit(f.module, *f.module.findFunction("k"), BlockVariant{}, "unit.cu");
        CHECK(text.find("int i = int(gl_WorkGroupID.x * 256u + gl_LocalInvocationID.x);") != std::string::npos);
        CHECK(text.find("uniform vec4 u_k_grid;") != std::string::npos);
        CHECK(text.find("floatBitsToUint(u_k_grid.x)") != std::string::npos);
    }
    {   // __popc: bitCount is GLSL-only (HLSL says countbits), so the shader declares a macro, and only when needed
        const char* const guard = "#define cgfx_popcount(x) countbits(x)";
        Frontend f;
        runFrontend("__global__ void k(int* z) { z[threadIdx.x] = (int)__popc(7u); }\n", f);
        CHECK(f.ok);
        const std::string text = ShaderEmitter().emit(f.module, *f.module.findFunction("k"), BlockVariant{}, "unit.cu");
        CHECK(text.find("#if BGFX_SHADER_LANGUAGE_GLSL\n#define cgfx_popcount(x) bitCount(x)\n#else\n") != std::string::npos);
        CHECK(text.find(guard) != std::string::npos);
        CHECK(text.find("int(cgfx_popcount(7u))") != std::string::npos);
        // the only mention of bitCount is inside the macro definition
        CHECK(text.find("bitCount(") == text.rfind("bitCount("));
        Frontend g;
        runFrontend("__global__ void k(int* z) { z[threadIdx.x] = 1; }\n", g);
        CHECK(g.ok);
        CHECK(ShaderEmitter().emit(g.module, *g.module.findFunction("k"), BlockVariant{}, "unit.cu").find("cgfx_popcount") == std::string::npos);
        // through a __device__ helper: the kernel that calls it needs the macro, the one that does not call it must not
        Frontend h;
        runFrontend("__device__ int bits(unsigned v) { return (int)__popc(v); }\n"
                    "__global__ void a(int* z) { z[0] = bits(3u); }\n"
                    "__global__ void b(int* z) { z[0] = 2; }\n", h);
        CHECK(h.ok);
        CHECK(ShaderEmitter().emit(h.module, *h.module.findFunction("a"), BlockVariant{}, "unit.cu").find(guard) != std::string::npos);
        CHECK(ShaderEmitter().emit(h.module, *h.module.findFunction("b"), BlockVariant{}, "unit.cu").find("cgfx_popcount") == std::string::npos);
    }
}

void testAbiAndTargets() {
    abi::ArgLayout l;
    unsigned lane = 0;
    CHECK(l.place(1, lane) && lane == 0);
    CHECK(l.place(2, lane) && lane == 2);   // float2 on an even lane
    CHECK(l.place(3, lane) && lane == 4);   // float3 at the start of a slot
    CHECK(l.place(1, lane) && lane == 7);
    CHECK(!l.place(5, lane) && l.failed());
    abi::ArgLayout big;
    for (int i = 0; i < 16; ++i) CHECK(big.place(4, lane));
    CHECK(!big.place(1, lane));
    CHECK(std::string(abi::laneSwizzle(2, 2)) == ".zw");
    CHECK(abi::laneSwizzle(1, 2) == nullptr);

    ShaderTarget t;
    std::string err;
    CHECK(parseShaderTarget("linux:spirv", t, err) && t.backend == "spv");
    CHECK(parseShaderTarget("windows:s_5_0", t, err) && t.backend == "dxbc");
    CHECK(parseShaderTarget("windows:s_6_0", t, err) && t.backend == "dxil");
    CHECK(parseShaderTarget("linux:430", t, err) && t.backend == "glsl");
    CHECK(parseShaderTarget("android:310_es", t, err) && t.backend == "essl");
    CHECK(parseShaderTarget("osx:metal", t, err) && t.backend == "mtl");
    CHECK(!parseShaderTarget("linux:pssl", t, err));
    CHECK(!parseShaderTarget("spirv", t, err));
    CHECK(shaderSymbolName("saxpy", BlockVariant{16, 16, 1}, "spv") == "cs_saxpy_16x16_spv");

    CHECK(ShadercDriver::quoteArgument("a b").find("a b") != std::string::npos);
#ifndef _WIN32
    CHECK(ShadercDriver::quoteArgument("it's") == "'it'\\''s'");
#endif
    CHECK(safeIdentifier("input") == "cgfx_input" && safeIdentifier("value") == "value");
    CHECK(isForbiddenIdentifier("gl_Foo") && isForbiddenIdentifier("a__b") && !isForbiddenIdentifier("foo_bar"));
}

} // namespace

int main() {
    testLexer();
    testLiteralsAndConstEval();
    testPreprocessor();
    testLaunchSites();
    testSemaTyping();
    testAbiAndTargets();
    std::cout << (g_failures ? "FAILURES: " : "OK: ") << g_checks << " checks, " << g_failures << " failure(s)\n";
    return g_failures == 0 ? 0 : 1;
}
