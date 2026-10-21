// board.cpp
#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace promethius {

// Kept identical in movegen.cpp. No header files are used by this project.
struct Board {
    std::array<int8_t, 64> squares{}; // a1 = 0; White pieces are positive.
    int sideToMove = 0;              // 0 = White, 1 = Black
    uint8_t castlingRights = 0;      // K=1, Q=2, k=4, q=8
    int enPassantSquare = -1;
    uint32_t halfmoveClock = 0;
    uint32_t fullmoveNumber = 1;
    uint8_t kingSquare[2] = {4, 60};
    uint64_t pieceBB[2][7]{}; // pieceBB[color][kind], kinds 1..6
    uint64_t occupancy[2] = {0, 0};
    uint64_t allOccupancy = 0;
};

using Move = uint16_t;
struct FastUndo {
    int8_t capturedPiece;
    uint8_t castlingRights;
    uint8_t capturedSquare;
    uint8_t padding;
    int enPassantSquare;
};
struct Undo {
    FastUndo state;
    uint32_t halfmoveClock;
    uint32_t fullmoveNumber;
};

bool board_set_fen(Board* board, const std::string& fen, std::string* error);
void initialize_movegen();

namespace {
constexpr int WHITE = 0;
constexpr int BLACK = 1;
constexpr int PAWN = 1;
constexpr int KNIGHT = 2;
constexpr int BISHOP = 3;
constexpr int ROOK = 4;
constexpr int QUEEN = 5;
constexpr int KING = 6;
constexpr unsigned WHITE_KINGSIDE = 1U;
constexpr unsigned WHITE_QUEENSIDE = 2U;
constexpr unsigned BLACK_KINGSIDE = 4U;
constexpr unsigned BLACK_QUEENSIDE = 8U;
constexpr std::string_view START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

inline int absPiece(int piece) { return piece < 0 ? -piece : piece; }

int pieceFromFen(char c) {
    int piece = 0;
    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(c)))) {
        case 'p': piece = PAWN; break;
        case 'n': piece = KNIGHT; break;
        case 'b': piece = BISHOP; break;
        case 'r': piece = ROOK; break;
        case 'q': piece = QUEEN; break;
        case 'k': piece = KING; break;
        default: return 0;
    }
    return std::isupper(static_cast<unsigned char>(c)) ? piece : -piece;
}

char pieceToFen(int piece) {
    char c = '?';
    switch (absPiece(piece)) {
        case PAWN: c = 'p'; break;
        case KNIGHT: c = 'n'; break;
        case BISHOP: c = 'b'; break;
        case ROOK: c = 'r'; break;
        case QUEEN: c = 'q'; break;
        case KING: c = 'k'; break;
        default: return '?';
    }
    return piece > 0 ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c;
}

inline bool parseSquare(std::string_view text, size_t offset, int& square) {
    if (offset + 1 >= text.size()) return false;
    const char file = static_cast<char>(std::tolower(static_cast<unsigned char>(text[offset])));
    const char rank = text[offset + 1];
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8') return false;
    square = (rank - '1') * 8 + (file - 'a');
    return true;
}

std::string squareToString(int square) {
    if (square < 0 || square >= 64) return "-";
    std::string result(2, ' ');
    result[0] = static_cast<char>('a' + (square & 7));
    result[1] = static_cast<char>('1' + (square >> 3));
    return result;
}

bool parseUnsigned(std::string_view text, uint32_t& result, bool allowZero) {
    if (text.empty()) return false;
    uint64_t value = 0;
    for (unsigned char c : text) {
        if (c < '0' || c > '9') return false;
        value = value * 10 + (c - '0');
        if (value > UINT32_MAX) return false;
    }
    if (!allowZero && value == 0) return false;
    result = static_cast<uint32_t>(value);
    return true;
}

constexpr uint8_t castlingMaskForSquare(int square) {
    return square == 0 ? static_cast<uint8_t>(~WHITE_QUEENSIDE) :
           square == 7 ? static_cast<uint8_t>(~WHITE_KINGSIDE) :
           square == 56 ? static_cast<uint8_t>(~BLACK_QUEENSIDE) :
           square == 63 ? static_cast<uint8_t>(~BLACK_KINGSIDE) : uint8_t{0x0F};
}
constexpr auto ROOK_RIGHT_MASKS = [] {
    std::array<uint8_t, 64> masks{};
    for (int square = 0; square < 64; ++square) masks[square] = castlingMaskForSquare(square);
    return masks;
}();
inline void clearRookRight(Board& b, int square) {
    b.castlingRights &= ROOK_RIGHT_MASKS[square];
}

inline int moveFrom(Move move) { return move & 63; }
inline int moveTo(Move move) { return (move >> 6) & 63; }
inline int movePromotion(Move move) { return (move >> 12) & 7; }

} // namespace

Board* board_create() {
    initialize_movegen();
    Board* board = new Board{};
    std::string error;
    if (!board_set_fen(board, std::string(START_FEN), &error)) {
        delete board;
        return nullptr;
    }
    return board;
}

Board* board_clone(const Board* board) {
    return board ? new Board(*board) : nullptr;
}

void board_destroy(Board* board) { delete board; }

bool board_set_fen(Board* board, const std::string& fen, std::string* error) {
    initialize_movegen();
    auto fail = [error](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (!board) return fail("board is null");

    // Tokenize six FEN fields without stream extraction overhead.
    std::string_view fields[7];
    size_t fieldCount = 0, i = 0;
    const std::string_view input(fen);
    while (i < input.size()) {
        while (i < input.size() && (input[i] == ' ' || input[i] == '\t' || input[i] == '\r' || input[i] == '\n')) ++i;
        if (i == input.size()) break;
        const size_t begin = i;
        while (i < input.size() && input[i] != ' ' && input[i] != '\t' && input[i] != '\r' && input[i] != '\n') ++i;
        if (fieldCount >= 7) return fail("FEN must contain exactly six fields");
        fields[fieldCount++] = input.substr(begin, i - begin);
    }
    if (fieldCount != 6) return fail("FEN must contain exactly six fields");

    Board candidate{};
    candidate.squares.fill(0);
    candidate.kingSquare[WHITE] = 255;
    candidate.kingSquare[BLACK] = 255;

    int rank = 7, file = 0, whiteKings = 0, blackKings = 0;
    for (char c : fields[0]) {
        if (c == '/') {
            if (file != 8 || rank <= 0) return fail("invalid FEN board ranks");
            --rank;
            file = 0;
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) return fail("FEN rank contains more than eight squares");
            continue;
        }
        const int piece = pieceFromFen(c);
        if (piece == 0 || file >= 8 || rank < 0) return fail("invalid FEN board layout or piece character");
        const int square = rank * 8 + file++;
        candidate.squares[square] = static_cast<int8_t>(piece);
        const uint64_t squareBit = uint64_t{1} << square;
        const int pieceColor = piece > 0 ? WHITE : BLACK;
        candidate.pieceBB[pieceColor][absPiece(piece)] |= squareBit;
        candidate.occupancy[pieceColor] |= squareBit;
        candidate.allOccupancy |= squareBit;
        if (piece == KING) { ++whiteKings; candidate.kingSquare[WHITE] = static_cast<uint8_t>(square); }
        if (piece == -KING) { ++blackKings; candidate.kingSquare[BLACK] = static_cast<uint8_t>(square); }
    }
    if (rank != 0 || file != 8) return fail("FEN board must describe eight complete ranks");
    if (whiteKings != 1 || blackKings != 1) return fail("FEN must contain exactly one king of each color");

    if (fields[1] == "w") candidate.sideToMove = WHITE;
    else if (fields[1] == "b") candidate.sideToMove = BLACK;
    else return fail("invalid FEN side-to-move field");

    candidate.castlingRights = 0;
    if (fields[2] != "-") {
        for (char c : fields[2]) {
            unsigned right = 0;
            switch (c) {
                case 'K': right = WHITE_KINGSIDE; break;
                case 'Q': right = WHITE_QUEENSIDE; break;
                case 'k': right = BLACK_KINGSIDE; break;
                case 'q': right = BLACK_QUEENSIDE; break;
                default: return fail("invalid FEN castling-rights field");
            }
            if (candidate.castlingRights & right) return fail("duplicate castling right in FEN");
            candidate.castlingRights |= static_cast<uint8_t>(right);
        }
    }

    candidate.enPassantSquare = -1;
    if (fields[3] != "-") {
        int square = -1;
        if (!parseSquare(fields[3], 0, square) || fields[3].size() != 2) return fail("invalid FEN en-passant square");
        const int expectedRank = candidate.sideToMove == WHITE ? 5 : 2;
        if ((square >> 3) != expectedRank) return fail("FEN en-passant square is on the wrong rank for the side to move");
        candidate.enPassantSquare = square;
    }

    if (!parseUnsigned(fields[4], candidate.halfmoveClock, true)) return fail("invalid FEN halfmove clock");
    if (!parseUnsigned(fields[5], candidate.fullmoveNumber, false)) return fail("invalid FEN fullmove number");

    *board = candidate;
    if (error) error->clear();
    return true;
}

std::string board_get_fen(const Board* board) {
    if (!board) return {};
    std::string out;
    out.reserve(96);
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const int piece = board->squares[rank * 8 + file];
            if (!piece) {
                ++empty;
            } else {
                if (empty) { out.push_back(static_cast<char>('0' + empty)); empty = 0; }
                out.push_back(pieceToFen(piece));
            }
        }
        if (empty) out.push_back(static_cast<char>('0' + empty));
        if (rank) out.push_back('/');
    }
    out += board->sideToMove == WHITE ? " w " : " b ";
    if (!board->castlingRights) out.push_back('-');
    else {
        if (board->castlingRights & WHITE_KINGSIDE) out.push_back('K');
        if (board->castlingRights & WHITE_QUEENSIDE) out.push_back('Q');
        if (board->castlingRights & BLACK_KINGSIDE) out.push_back('k');
        if (board->castlingRights & BLACK_QUEENSIDE) out.push_back('q');
    }
    out.push_back(' ');
    out += board->enPassantSquare < 0 ? "-" : squareToString(board->enPassantSquare);
    out.push_back(' ');
    out += std::to_string(board->halfmoveClock);
    out.push_back(' ');
    out += std::to_string(board->fullmoveNumber);
    return out;
}

int board_piece_at(const Board* board, int square) {
    return board && static_cast<unsigned>(square) < 64U ? board->squares[square] : 0;
}
int board_side_to_move(const Board* board) { return board ? board->sideToMove : WHITE; }
unsigned board_castling_rights(const Board* board) { return board ? board->castlingRights : 0U; }
int board_en_passant_square(const Board* board) { return board ? board->enPassantSquare : -1; }
int board_halfmove_clock(const Board* board) { return board ? static_cast<int>(board->halfmoveClock) : 0; }
int board_fullmove_number(const Board* board) { return board ? static_cast<int>(board->fullmoveNumber) : 1; }
int board_king_square(const Board* board, int color) {
    return board && (color == WHITE || color == BLACK) ? board->kingSquare[color] : -1;
}

// Move-generation make/unmake stores only rule-critical state. Perft skips
// fifty-move counters, which do not affect legal move generation.
static void boardMakeMoveFastInternal(Board* board, Move move, FastUndo* undo) {
    const int from = moveFrom(move), to = moveTo(move), promo = movePromotion(move);
    const int movingPiece = board->squares[from];
    const int movingKind = absPiece(movingPiece);
    int capturedSquare = to;
    if (movingKind == PAWN && to == board->enPassantSquare && board->squares[to] == 0 && ((from & 7) != (to & 7)))
        capturedSquare = to + (board->sideToMove == WHITE ? -8 : 8);

    undo->capturedPiece = board->squares[capturedSquare];
    undo->capturedSquare = static_cast<uint8_t>(capturedSquare);
    undo->castlingRights = board->castlingRights;
    undo->enPassantSquare = board->enPassantSquare;

    const int moverColor = board->sideToMove;
    const uint64_t fromBit = uint64_t{1} << from;
    const uint64_t toBit = uint64_t{1} << to;

    // Most perft edges are quiet moves. Keep them on a short path that avoids
    // the capture, en-passant, promotion and castling-rook bookkeeping below.
    const bool kingCastle = movingKind == KING &&
        (to - from == 2 || from - to == 2);
    if (undo->capturedPiece == 0 && promo == 0 && !kingCastle) {
        board->squares[from] = 0;
        board->squares[to] = static_cast<int8_t>(movingPiece);
        board->occupancy[moverColor] = (board->occupancy[moverColor] & ~fromBit) | toBit;
        board->allOccupancy = (board->allOccupancy & ~fromBit) | toBit;
        board->pieceBB[moverColor][movingKind] =
            (board->pieceBB[moverColor][movingKind] & ~fromBit) | toBit;

        if (movingKind == KING) {
            board->kingSquare[moverColor] = static_cast<uint8_t>(to);
            board->castlingRights &= static_cast<uint8_t>(moverColor == WHITE
                ? ~(WHITE_KINGSIDE | WHITE_QUEENSIDE)
                : ~(BLACK_KINGSIDE | BLACK_QUEENSIDE));
        }
        board->castlingRights &= static_cast<uint8_t>(ROOK_RIGHT_MASKS[from] & ROOK_RIGHT_MASKS[to]);
        board->enPassantSquare = movingKind == PAWN &&
            (to - from == 16 || from - to == 16) ? (from + to) / 2 : -1;
        board->sideToMove ^= 1;
        return;
    }

    board->occupancy[moverColor] &= ~fromBit;
    board->allOccupancy &= ~fromBit;
    board->pieceBB[moverColor][movingKind] &= ~fromBit;
    if (undo->capturedPiece != 0) {
        const uint64_t capturedBit = uint64_t{1} << capturedSquare;
        const int capturedKind = absPiece(undo->capturedPiece);
        board->occupancy[moverColor ^ 1] &= ~capturedBit;
        board->allOccupancy &= ~capturedBit;
        board->pieceBB[moverColor ^ 1][capturedKind] &= ~capturedBit;
    }

    board->squares[from] = 0;
    if (capturedSquare != to) board->squares[capturedSquare] = 0;

    if (movingKind == KING && (to - from == 2 || from - to == 2)) {
        if (to > from) {
            board->squares[from + 1] = board->squares[from + 3];
            board->squares[from + 3] = 0;
            board->occupancy[moverColor] &= ~(uint64_t{1} << (from + 3));
            board->occupancy[moverColor] |= uint64_t{1} << (from + 1);
            board->allOccupancy &= ~(uint64_t{1} << (from + 3));
            board->allOccupancy |= uint64_t{1} << (from + 1);
            board->pieceBB[moverColor][ROOK] &= ~(uint64_t{1} << (from + 3));
            board->pieceBB[moverColor][ROOK] |= uint64_t{1} << (from + 1);
        } else {
            board->squares[from - 1] = board->squares[from - 4];
            board->squares[from - 4] = 0;
            board->occupancy[moverColor] &= ~(uint64_t{1} << (from - 4));
            board->occupancy[moverColor] |= uint64_t{1} << (from - 1);
            board->allOccupancy &= ~(uint64_t{1} << (from - 4));
            board->allOccupancy |= uint64_t{1} << (from - 1);
            board->pieceBB[moverColor][ROOK] &= ~(uint64_t{1} << (from - 4));
            board->pieceBB[moverColor][ROOK] |= uint64_t{1} << (from - 1);
        }
    }

    int placedPiece = movingPiece;
    if (promo) {
        const int promotedKind = promo == 1 ? KNIGHT : promo == 2 ? BISHOP : promo == 3 ? ROOK : QUEEN;
        placedPiece = moverColor == WHITE ? promotedKind : -promotedKind;
    }
    board->squares[to] = static_cast<int8_t>(placedPiece);
    board->occupancy[moverColor] |= toBit;
    board->allOccupancy |= toBit;
    board->pieceBB[moverColor][absPiece(placedPiece)] |= toBit;
    if (movingKind == KING) board->kingSquare[moverColor] = static_cast<uint8_t>(to);

    if (movingKind == KING) {
        board->castlingRights &= static_cast<uint8_t>(moverColor == WHITE
            ? ~(WHITE_KINGSIDE | WHITE_QUEENSIDE) : ~(BLACK_KINGSIDE | BLACK_QUEENSIDE));
    }
    clearRookRight(*board, from);
    clearRookRight(*board, to);

    board->enPassantSquare = -1;
    if (movingKind == PAWN && (to - from == 16 || from - to == 16))
        board->enPassantSquare = (from + to) / 2;
    board->sideToMove ^= 1;
}

static void boardUnmakeMoveFastInternal(Board* board, Move move, const FastUndo* undo) {
    const int from = moveFrom(move), to = moveTo(move), promo = movePromotion(move);
    const int moverColor = board->sideToMove ^ 1;
    const int currentPiece = board->squares[to];
    const int originalPiece = promo ? (moverColor == WHITE ? PAWN : -PAWN) : currentPiece;
    const bool kingCastle = absPiece(originalPiece) == KING &&
        (to - from == 2 || from - to == 2);

    if (undo->capturedPiece == 0 && promo == 0 && !kingCastle) {
        const uint64_t fromBit = uint64_t{1} << from;
        const uint64_t toBit = uint64_t{1} << to;
        board->squares[from] = static_cast<int8_t>(currentPiece);
        board->squares[to] = 0;
        board->pieceBB[moverColor][absPiece(currentPiece)] =
            (board->pieceBB[moverColor][absPiece(currentPiece)] & ~toBit) | fromBit;
        board->occupancy[moverColor] = (board->occupancy[moverColor] & ~toBit) | fromBit;
        board->allOccupancy = (board->allOccupancy & ~toBit) | fromBit;
        if (absPiece(currentPiece) == KING) board->kingSquare[moverColor] = static_cast<uint8_t>(from);
        board->sideToMove = moverColor;
        board->castlingRights = undo->castlingRights;
        board->enPassantSquare = undo->enPassantSquare;
        return;
    }

    board->squares[from] = static_cast<int8_t>(originalPiece);
    board->squares[to] = 0;
    board->pieceBB[moverColor][absPiece(currentPiece)] &= ~(uint64_t{1} << to);
    board->pieceBB[moverColor][absPiece(originalPiece)] |= uint64_t{1} << from;

    if (absPiece(originalPiece) == KING) board->kingSquare[moverColor] = static_cast<uint8_t>(from);
    if (absPiece(originalPiece) == KING && (to - from == 2 || from - to == 2)) {
        if (to > from) {
            board->squares[from + 3] = board->squares[from + 1];
            board->squares[from + 1] = 0;
            board->pieceBB[moverColor][ROOK] &= ~(uint64_t{1} << (from + 1));
            board->pieceBB[moverColor][ROOK] |= uint64_t{1} << (from + 3);
        } else {
            board->squares[from - 4] = board->squares[from - 1];
            board->squares[from - 1] = 0;
            board->pieceBB[moverColor][ROOK] &= ~(uint64_t{1} << (from - 1));
            board->pieceBB[moverColor][ROOK] |= uint64_t{1} << (from - 4);
        }
    }
    board->squares[undo->capturedSquare] = undo->capturedPiece;
    if (undo->capturedPiece != 0)
        board->pieceBB[moverColor ^ 1][absPiece(undo->capturedPiece)] |= uint64_t{1} << undo->capturedSquare;

    const uint64_t fromBit = uint64_t{1} << from;
    const uint64_t toBit = uint64_t{1} << to;
    board->occupancy[moverColor] &= ~toBit;
    board->occupancy[moverColor] |= fromBit;
    if (undo->capturedPiece != 0) {
        const uint64_t capturedBit = uint64_t{1} << undo->capturedSquare;
        board->occupancy[moverColor ^ 1] |= capturedBit;
        board->allOccupancy |= capturedBit;
    }
    if (undo->capturedPiece == 0 || undo->capturedSquare != to)
        board->allOccupancy &= ~toBit;
    board->allOccupancy |= fromBit;
    if (absPiece(originalPiece) == KING && (to - from == 2 || from - to == 2)) {
        const int rookFrom = to > from ? from + 3 : from - 4;
        const int rookTo = to > from ? from + 1 : from - 1;
        board->occupancy[moverColor] &= ~(uint64_t{1} << rookTo);
        board->occupancy[moverColor] |= uint64_t{1} << rookFrom;
        board->allOccupancy &= ~(uint64_t{1} << rookTo);
        board->allOccupancy |= uint64_t{1} << rookFrom;
    }
    board->sideToMove = moverColor;
    board->castlingRights = undo->castlingRights;
    board->enPassantSquare = undo->enPassantSquare;
}

void board_make_move_fast(Board* board, Move move, FastUndo* undo) {
    boardMakeMoveFastInternal(board, move, undo);
}
void board_unmake_move_fast(Board* board, Move move, const FastUndo* undo) {
    boardUnmakeMoveFastInternal(board, move, undo);
}
void board_make_move(Board* board, Move move, Undo* undo) {
    const int oldHalfmove = static_cast<int>(board->halfmoveClock);
    const uint32_t oldFullmove = board->fullmoveNumber;
    const int side = board->sideToMove;
    const int movingKind = absPiece(board->squares[moveFrom(move)]);
    boardMakeMoveFastInternal(board, move, &undo->state);
    undo->halfmoveClock = static_cast<uint32_t>(oldHalfmove);
    undo->fullmoveNumber = oldFullmove;
    if (movingKind == PAWN || undo->state.capturedPiece != 0) board->halfmoveClock = 0;
    else ++board->halfmoveClock;
    if (side == BLACK) ++board->fullmoveNumber;
}
void board_unmake_move(Board* board, Move move, const Undo* undo) {
    boardUnmakeMoveFastInternal(board, move, &undo->state);
    board->halfmoveClock = undo->halfmoveClock;
    board->fullmoveNumber = undo->fullmoveNumber;
}

// Slow-path UCI helper; only used while parsing an externally supplied move list.
bool board_apply_move_unchecked(Board* board, const std::string& uciMove, std::string* error) {
    auto fail = [error](const char* message) { if (error) *error = message; return false; };
    if (!board) return fail("board is null");
    if (uciMove.size() != 4 && uciMove.size() != 5) return fail("invalid UCI move length");
    int from = -1, to = -1;
    if (!parseSquare(uciMove, 0, from) || !parseSquare(uciMove, 2, to) || from == to) return fail("invalid UCI move squares");
    const int piece = board->squares[from];
    if (!piece || ((piece > 0) != (board->sideToMove == WHITE))) return fail("move starts from an empty square or opponent piece");

    int promo = 0;
    if (uciMove.size() == 5) {
        switch (static_cast<char>(std::tolower(static_cast<unsigned char>(uciMove[4])))) {
            case 'n': promo = 1; break;
            case 'b': promo = 2; break;
            case 'r': promo = 3; break;
            case 'q': promo = 4; break;
            default: return fail("invalid promotion piece");
        }
    }
    const Move move = static_cast<Move>(from | (to << 6) | (promo << 12));
    Undo undo{};
    board_make_move(board, move, &undo);
    if (error) error->clear();
    return true;
}

// Declarations of movegen helpers are repeated here because this project has no headers.
std::vector<std::string> generate_legal_moves(const Board* board);
std::string move_to_uci(Move move);

bool board_play_uci_move(Board* board, const std::string& uciMove, std::string* error) {
    auto fail = [error](const char* message) { if (error) *error = message; return false; };
    if (!board) return fail("board is null");
    std::string normalized = uciMove;
    for (char& c : normalized) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const std::vector<std::string> legal = generate_legal_moves(board);
    for (const std::string& candidate : legal) {
        if (candidate == normalized) {
            // Parse and apply the already-validated legal move.
            if (!board_apply_move_unchecked(board, normalized, error)) return false;
            return true;
        }
    }
    return fail("move is not legal in the current position");
}

} // namespace promethius
