// Token.hpp: tokens produced by the Lexer
#pragma once

#include <string>
#include "SourceLocation.hpp"

namespace cudagfx {

enum class TokenType {
    EndOfFile,
    Unknown,

    Identifier,
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,
    HeaderName,     // <...> right after #include

    // CUDA qualifiers and keywords
    KwGlobal, KwDevice, KwHost, KwShared, KwConstant, KwLaunchBounds, KwSyncthreads,

    // Types
    KwVoid, KwBool, KwChar, KwShort, KwInt, KwUnsigned, KwSigned, KwLong, KwFloat, KwDouble,
    KwUint,      // `uint` (CUDA typedef)
    KwVecType,   // float2/3/4, int2/3/4, uint2/3/4 (the text holds the exact name)

    // Control flow
    KwIf, KwElse, KwFor, KwWhile, KwDo, KwSwitch, KwCase, KwDefault, KwBreak, KwContinue, KwReturn,

    // C++ qualifiers
    KwConst, KwConstexpr, KwStatic, KwInline, KwVolatile, KwRestrict, KwExtern,
    KwStruct, KwSizeof, KwTrue, KwFalse, KwNamespace,

    // Constructs rejected inside a kernel (CGFX0009)
    KwTemplate, KwVirtual, KwTry, KwCatch, KwThrow, KwNew, KwDelete, KwGoto, KwClass,
    KwAuto, KwUsing, KwUnion, KwTypedef, KwEnum, KwOperator,

    // Punctuation and operators
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Semi, Comma, Colon, ColonColon, Question, Dot, Arrow, Ellipsis,
    Plus, Minus, Star, Slash, Percent, Amp, Pipe, Caret, Tilde, Bang,
    Assign, Less, Greater, Equal, NotEqual, LessEqual, GreaterEqual,
    AmpAmp, PipePipe, LessLess, GreaterGreater, PlusPlus, MinusMinus,
    PlusAssign, MinusAssign, StarAssign, SlashAssign, PercentAssign,
    AmpAssign, PipeAssign, CaretAssign, LessLessAssign, GreaterGreaterAssign,
    TripleLess,     // <<<
    TripleGreater,  // >>>
    Hash,           // # not at the start of a line (stringification)
    HashHash,       // ##

    // Preprocessor directives (a single token at the start of a line; the rest of the line is made of
    // the following tokens, up to the next token with atLineStart set)
    PpInclude, PpDefine, PpUndef, PpIf, PpIfdef, PpIfndef, PpElif, PpElse, PpEndif,
    PpPragma, PpLine, PpOther
};

struct Token {
    TokenType type = TokenType::Unknown;
    std::string text;
    SourceLocation location;
    unsigned length = 0;        // length in bytes in the source (line continuations included)
    bool atLineStart = false;   // first token of its logical line
    bool spaceBefore = false;   // preceded by a space, a newline or a comment

    bool is(TokenType t) const { return type == t; }
    std::size_t endOffset() const { return location.offset + length; }
};

const char* tokenTypeName(TokenType type);

// True for preprocessor directives (PpInclude ... PpOther).
inline bool isDirective(TokenType t) { return t >= TokenType::PpInclude && t <= TokenType::PpOther; }

// True for the C++ keywords that CudaGFX rejects in device code (CGFX0009).
inline bool isUnsupportedKeyword(TokenType t) {
    return t >= TokenType::KwTemplate && t <= TokenType::KwOperator;
}

} // namespace cudagfx
