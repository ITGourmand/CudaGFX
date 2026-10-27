// Parser.hpp: syntax analysis of the supported CUDA subset + analysis of the launch sites.
//
// Two deliberately distinct views of the same file:
//   * the ACTIVE stream (macros not expanded) is used to find the <<<>>> launches of the host code,
//     whose exact text is copied unchanged into the generated code;
//   * the EXPANDED stream is used to analyze kernels and __device__ functions (macros resolved).
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "AST.hpp"
#include "ConstEval.hpp"
#include "Diagnostics.hpp"
#include "Preprocessor.hpp"
#include "SourceManager.hpp"
#include "Token.hpp"

namespace cudagfx {

class Parser {
public:
    Parser(const Preprocessor& pp, const SourceManager& sources, std::vector<Token> activeTokens,
           FileId mainFile, DiagnosticsEngine& diags);

    // Parses the file. Errors are reported to the diagnostics engine; the returned module is always
    // structurally valid (never a null pointer in the AST).
    ParsedModule parse();

private:
    // --- cursor over the expanded stream
    const Token& peek(std::size_t ahead = 0) const;
    const Token& get();
    bool check(TokenType type) const { return peek().type == type; }
    bool match(TokenType type);
    bool expect(TokenType type, const char* what);
    bool atEnd() const { return peek().type == TokenType::EndOfFile; }
    std::size_t findMatching(std::size_t openIndex, TokenType open, TokenType close) const;

    // --- errors and recovery
    void errorAt(const SourceLocation& loc, const std::string& message);
    void unsupported(const SourceLocation& loc, const std::string& what); // CGFX0009
    void synchronize();

    // --- file level
    void parseTopLevel();
    bool isTransparentBrace(std::size_t braceIndex) const;
    void parseDeviceDeclaration(std::size_t declStart);
    bool tryParseStruct();
    bool tryParseGlobalConst();
    std::unique_ptr<FunctionDecl> parseFunction(std::size_t declStart);
    bool parseParameter(Parameter& out);
    unsigned parseLaunchBounds();
    void skipBalanced(TokenType open, TokenType close);

    // --- types
    Type parseType(bool allowPointers = true);
    Type parseBaseType();
    Type applyPointers(Type base);
    bool isTypeKeyword(TokenType t) const;
    bool isCastStart() const;
    bool isDeclStart() const;

    // --- statements
    std::unique_ptr<CompoundStmt> parseCompound();
    StmtPtr parseStatement();
    StmtPtr parseDeclStmt(bool consumeSemicolon = true);
    StmtPtr parseIf();
    StmtPtr parseFor();
    StmtPtr parseWhile();
    StmtPtr parseDoWhile();
    StmtPtr parseSwitch();
    StmtPtr parseReturn();

    // --- expressions
    ExprPtr parseCommaExpression();
    ExprPtr parseExpression(int minPrec = 1);
    ExprPtr parseUnary();
    ExprPtr parsePostfix(ExprPtr base);
    ExprPtr parsePrimary();
    ExprPtr makeErrorExpr(const SourceLocation& loc);

    // --- constants
    bool evalConst(std::size_t first, std::size_t last, std::int64_t& out);
    bool resolveConst(const std::string& name, std::int64_t& value) const;

    // --- launch sites of the host code (LaunchScanner.cpp)
    struct HostEntry {
        bool isConst = false;
        bool known = false;
        bool isDim3 = false;
        bool modified = false;
        std::int64_t value = 0;
        std::int64_t dims[3] = {1, 1, 1};
    };
    void scanLaunchSites();
    bool parseHostDeclaration(std::size_t& i);
    bool handleLaunch(std::size_t& i);
    bool evaluateBlock(std::size_t first, std::size_t last, const SourceLocation& at, std::string& text,
                       unsigned dims[3]);
    bool evalHostRange(const std::vector<Token>& expanded, std::int64_t& out, std::string& why);
    const HostEntry* findHostEntry(const std::string& name) const;
    std::string sliceText(std::size_t firstTok, std::size_t lastTok) const; // [first, last] inclusive

    const Preprocessor& pp_;
    const SourceManager& sources_;
    std::vector<Token> raw_;       // active stream (macros not expanded)
    std::vector<Token> toks_;      // expanded stream
    std::size_t pos_ = 0;
    FileId mainFile_;
    DiagnosticsEngine& diags_;
    ParsedModule module_;

    bool panic_ = false;
    int opaqueDepth_ = 0;
    std::unordered_map<std::string, std::int64_t> globalInts_;
    std::vector<std::unordered_map<std::string, std::int64_t>> localConsts_;
    std::unordered_set<std::string> structNames_;
    std::vector<std::unordered_map<std::string, HostEntry>> hostScopes_;
    mutable std::string lastUnresolved_;
};

} // namespace cudagfx
