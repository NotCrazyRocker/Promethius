// movegen.cpp
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace promethius {

// These definitions must match board.cpp exactly. No header files are used.
struct Board {
    std::array<int8_t, 64> squares{};
    int sideToMove = 0;
    uint8_t castlingRights = 0;
    int enPassantSquare = -1;
    uint32_t halfmoveClock = 0;
    uint32_t fullmoveNumber = 1;
    uint8_t kingSquare[2] = {4, 60};
    uint64_t pieceBB[2][7]{};
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

void board_make_move_fast(Board* board, Move move, FastUndo* undo);
void board_unmake_move_fast(Board* board, Move move, const FastUndo* undo);
bool is_square_attacked(const Board* board, int square, int byColor);
bool is_in_check(const Board* board, int color);
void initialize_movegen();

namespace {
constexpr int WHITE = 0, BLACK = 1, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6;
constexpr unsigned WK = 1U, WQ = 2U, BK = 4U, BQ = 8U;
constexpr int KNIGHT_DELTAS[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
constexpr int KING_DELTAS[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
constexpr int RAY_DIRS[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
constexpr char PROMO_CHARS[5] = {'\0','n','b','r','q'};
constexpr uint32_t ROOK_TABLE_SIZE = 102400;
constexpr uint32_t BISHOP_TABLE_SIZE = 5248;

constexpr uint64_t ROOK_MAGICS[64] = {
    0x0080021620804001ULL, 0x0040001000200041ULL, 0x0200102200088040ULL, 0x4080040800821000ULL, 0x2200020004200810ULL, 0x4b00020c000d0008ULL, 0x01000c4183000600ULL, 0x2080010000402c80ULL, 0x8002800826864000ULL, 0x0410802000884000ULL, 0x0c01004010200100ULL, 0x020300100100203cULL, 0x0450800801040080ULL, 0x4010800200040080ULL, 0x8804000208048110ULL, 0x0c40800080004100ULL, 0xa2018880024004a0ULL, 0x0080848020004004ULL, 0x1010410010200101ULL, 0x2010008008008010ULL, 0x0a08010004110008ULL, 0x0802080104209040ULL, 0x0080040090010802ULL, 0x0280020000841069ULL, 0x080c400080248000ULL, 0x2048850100224008ULL, 0x00200800c0300040ULL, 0x11400d0100201000ULL, 0x0041001100080204ULL, 0x4802000200040810ULL, 0x0100080c00103601ULL, 0x0020084200043085ULL, 0x0100804000800022ULL, 0x0460401000402002ULL, 0x8309002001001044ULL, 0x0000800800801000ULL, 0x0000800800800400ULL, 0xb542040080800200ULL, 0x1041000401000200ULL, 0x000318b04a000401ULL, 0x0280082000484000ULL, 0x0080400081010030ULL, 0x0010002000108080ULL, 0x012010002101000aULL, 0x0801000408010012ULL, 0x0004008002008004ULL, 0x0ad1005200110014ULL, 0x4000004110820004ULL, 0x9400400080003080ULL, 0x0000802200490200ULL, 0x1521100080200280ULL, 0x9021000824100100ULL, 0x0081080080840280ULL, 0x0002000904100200ULL, 0x0130024801302400ULL, 0x0102008100442200ULL, 0x0080984063800101ULL, 0x0016810201412812ULL, 0x40200101603008c1ULL, 0x2851100004082101ULL, 0x1049001002880005ULL, 0x0081000804000201ULL, 0x100020901208410cULL, 0x0101064400813102ULL
};
constexpr uint64_t BISHOP_MAGICS[64] = {
    0x24e0440c00802202ULL, 0x00881808841a4500ULL, 0x29c1021085004190ULL, 0x18c4041080042020ULL, 0x0841104000008108ULL, 0x890828080880c088ULL, 0x0006021024062018ULL, 0x2000404044104040ULL, 0x09000504104a0210ULL, 0x0088390204040820ULL, 0x4001420082008402ULL, 0x028108048b001142ULL, 0x1c00140421001008ULL, 0x0008021212200400ULL, 0x080000581a082004ULL, 0x3000048208027204ULL, 0x0120004044148482ULL, 0x4021000808108090ULL, 0x0084011808009452ULL, 0x11c802242020e000ULL, 0x0124002210140002ULL, 0x4009008200420200ULL, 0x0000830202100202ULL, 0x9002042500420200ULL, 0x0a60200004480210ULL, 0x0402481020480080ULL, 0x8001100101004200ULL, 0x6240104004004080ULL, 0x1124848014002000ULL, 0x00180200204100a0ULL, 0x8020890844880800ULL, 0x0000802009040204ULL, 0x0410042041100280ULL, 0x0804022000020440ULL, 0x2418280400480024ULL, 0x0801080800420a00ULL, 0x4002248400020020ULL, 0x3020004102038084ULL, 0x84280110601c0200ULL, 0x2004004208088080ULL, 0x0008022220041210ULL, 0x00820e0120000440ULL, 0x0002002201020822ULL, 0x0000002019000804ULL, 0x0211204c10101100ULL, 0x0604808081001200ULL, 0x1010029204030041ULL, 0x1008090102110621ULL, 0x0002015002100c00ULL, 0x06002c040404400aULL, 0xc030002201100011ULL, 0x4040008020884000ULL, 0x0248000903040100ULL, 0xc010092008008040ULL, 0x6008084108020494ULL, 0x28102182008e0042ULL, 0x0010210820842002ULL, 0x4080020111491002ULL, 0x0108100084008800ULL, 0x0022242100420221ULL, 0x10a8008110020210ULL, 0x400019122a900102ULL, 0x00800a1051080300ULL, 0x0420222088008080ULL
};

struct AttackTables {
    uint64_t knight[64]{};
    uint64_t king[64]{};
    uint64_t pawn[2][64]{};
    uint64_t rookMask[64]{};
    uint64_t bishopMask[64]{};
    uint32_t rookOffset[64]{};
    uint32_t bishopOffset[64]{};
    uint8_t rookBits[64]{};
    uint8_t bishopBits[64]{};
    uint64_t between[64][64]{};
    uint64_t rook[ROOK_TABLE_SIZE]{};
    uint64_t bishop[BISHOP_TABLE_SIZE]{};
};

inline int fileOf(int square) { return square & 7; }
inline int rankOf(int square) { return square >> 3; }
inline bool inside(int file, int rank) { return static_cast<unsigned>(file) < 8U && static_cast<unsigned>(rank) < 8U; }
inline int squareOf(int file, int rank) { return rank * 8 + file; }
inline int absPiece(int p) { return p < 0 ? -p : p; }
inline Move makeMove(int from, int to, int promo = 0) { return static_cast<Move>(from | (to << 6) | (promo << 12)); }
inline int fromOf(Move m) { return m & 63; }
inline int toOf(Move m) { return (m >> 6) & 63; }
inline int promoOf(Move m) { return (m >> 12) & 7; }

inline int popLeastBit(uint64_t& bits) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, bits);
    bits &= bits - 1;
    return static_cast<int>(index);
#else
    const int index = __builtin_ctzll(bits);
    bits &= bits - 1;
    return index;
#endif
}

inline int popCount(uint64_t bits) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(bits));
#else
    return __builtin_popcountll(bits);
#endif
}


uint64_t rayAttacks(int square, uint64_t occupancy, const int dirs[][2], int directionCount) {
    uint64_t attacks = 0;
    const int file = fileOf(square), rank = rankOf(square);
    for (int d = 0; d < directionCount; ++d) {
        int f = file + dirs[d][0], r = rank + dirs[d][1];
        while (inside(f, r)) {
            const int to = squareOf(f, r);
            const uint64_t bit = uint64_t{1} << to;
            attacks |= bit;
            if (occupancy & bit) break;
            f += dirs[d][0]; r += dirs[d][1];
        }
    }
    return attacks;
}

uint64_t relevantSliderMask(int square, const int dirs[][2], int directionCount) {
    uint64_t mask = 0;
    const int file = fileOf(square), rank = rankOf(square);
    for (int d = 0; d < directionCount; ++d) {
        int f = file + dirs[d][0], r = rank + dirs[d][1];
        // The terminal edge square is not needed in the occupancy index.
        while (inside(f, r)) {
            const int nf = f + dirs[d][0], nr = r + dirs[d][1];
            if (!inside(nf, nr)) break;
            mask |= uint64_t{1} << squareOf(f, r);
            f = nf; r = nr;
        }
    }
    return mask;
}

uint64_t occupancyFromIndex(uint32_t index, uint64_t mask) {
    uint64_t occupancy = 0;
    uint32_t bitIndex = 0;
    while (mask) {
        const uint64_t bit = mask & (~mask + 1);
        if (index & (uint32_t{1} << bitIndex)) occupancy |= bit;
        mask &= mask - 1;
        ++bitIndex;
    }
    return occupancy;
}

inline uint32_t sliderIndex(const AttackTables&, uint64_t occupancy, uint64_t mask,
                            uint64_t magic, int relevantBits) {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__)) && !defined(__znver1__) && !defined(__znver2__)
    (void)magic;
    (void)relevantBits;
    return static_cast<uint32_t>(_pext_u64(occupancy, mask));
#else
    return static_cast<uint32_t>(((occupancy & mask) * magic) >> (64 - relevantBits));
#endif
}

AttackTables g_attackTables{};
bool g_attackTablesReady = false;

void buildAttackTables(AttackTables& t) {
    for (int square = 0; square < 64; ++square) {
            const int file = fileOf(square), rank = rankOf(square);
            for (int i = 0; i < 8; ++i) {
                const int nf = file + KNIGHT_DELTAS[i][0], nr = rank + KNIGHT_DELTAS[i][1];
                if (inside(nf, nr)) t.knight[square] |= uint64_t{1} << squareOf(nf, nr);
                const int kf = file + KING_DELTAS[i][0], kr = rank + KING_DELTAS[i][1];
                if (inside(kf, kr)) t.king[square] |= uint64_t{1} << squareOf(kf, kr);
            }
            if (rank < 7) {
                if (file > 0) t.pawn[WHITE][square] |= uint64_t{1} << (square + 7);
                if (file < 7) t.pawn[WHITE][square] |= uint64_t{1} << (square + 9);
            }
            if (rank > 0) {
                if (file > 0) t.pawn[BLACK][square] |= uint64_t{1} << (square - 9);
                if (file < 7) t.pawn[BLACK][square] |= uint64_t{1} << (square - 7);
            }
        }

        for (int from = 0; from < 64; ++from) {
            const int ff = fileOf(from), fr = rankOf(from);
            for (int to = 0; to < 64; ++to) {
                if (from == to) continue;
                const int tf = fileOf(to), tr = rankOf(to);
                const int dfRaw = tf - ff, drRaw = tr - fr;
                if (dfRaw != 0 && drRaw != 0 && (dfRaw < 0 ? -dfRaw : dfRaw) != (drRaw < 0 ? -drRaw : drRaw)) continue;
                const int df = (dfRaw > 0) - (dfRaw < 0), dr = (drRaw > 0) - (drRaw < 0);
                if (dfRaw == 0 && drRaw == 0) continue;
                int f = ff + df, r = fr + dr;
                uint64_t between = 0;
                while (f != tf || r != tr) {
                    between |= uint64_t{1} << squareOf(f, r);
                    f += df; r += dr;
                }
                t.between[from][to] = between & ~(uint64_t{1} << to);
            }
        }

        uint32_t rookOffset = 0, bishopOffset = 0;
        for (int square = 0; square < 64; ++square) {
            t.rookMask[square] = relevantSliderMask(square, RAY_DIRS, 4);
            t.bishopMask[square] = relevantSliderMask(square, RAY_DIRS + 4, 4);
            t.rookOffset[square] = rookOffset;
            t.bishopOffset[square] = bishopOffset;
            t.rookBits[square] = static_cast<uint8_t>(popCount(t.rookMask[square]));
            t.bishopBits[square] = static_cast<uint8_t>(popCount(t.bishopMask[square]));
            const uint32_t rookCount = uint32_t{1} << t.rookBits[square];
            const uint32_t bishopCount = uint32_t{1} << t.bishopBits[square];
            for (uint32_t i = 0; i < rookCount; ++i) {
                const uint64_t occ = occupancyFromIndex(i, t.rookMask[square]);
                const uint32_t index = sliderIndex(t, occ, t.rookMask[square], ROOK_MAGICS[square], t.rookBits[square]);
                const uint64_t attack = rayAttacks(square, occ, RAY_DIRS, 4);
                t.rook[rookOffset + index] = attack;
            }
            for (uint32_t i = 0; i < bishopCount; ++i) {
                const uint64_t occ = occupancyFromIndex(i, t.bishopMask[square]);
                const uint32_t index = sliderIndex(t, occ, t.bishopMask[square], BISHOP_MAGICS[square], t.bishopBits[square]);
                const uint64_t attack = rayAttacks(square, occ, RAY_DIRS + 4, 4);
                t.bishop[bishopOffset + index] = attack;
            }
            rookOffset += rookCount;
            bishopOffset += bishopCount;
        }

}

inline const AttackTables& attackTables() { return g_attackTables; }

inline uint64_t rookAttacks(int square, uint64_t occupancy) {
    const AttackTables& t = attackTables();
    const uint32_t index = sliderIndex(t, occupancy, t.rookMask[square], ROOK_MAGICS[square], t.rookBits[square]);
    return t.rook[t.rookOffset[square] + index];
}

inline uint64_t bishopAttacks(int square, uint64_t occupancy) {
    const AttackTables& t = attackTables();
    const uint32_t index = sliderIndex(t, occupancy, t.bishopMask[square], BISHOP_MAGICS[square], t.bishopBits[square]);
    return t.bishop[t.bishopOffset[square] + index];
}

struct MoveList {
    Move moves[256];
    int count = 0;
    inline void push(Move move) { if (count < 256) moves[count++] = move; }
};

struct CheckInfo;
template<bool CountOnly>
struct LegalCollector {
    Board* board;
    MoveList* list;
    const CheckInfo& check;
    int count = 0;
    inline void emit(Move move) {
        if constexpr (CountOnly) ++count;
        else list->push(move);
    }
    void pushSquares(int from, int to, int promotion, int movingKind, bool enPassant = false);
};
CheckInfo computeCheckInfo(const Board* board);

template<bool CountOnly>
inline void addTargets(LegalCollector<CountOnly>& out, int from, uint64_t targets, int movingKind) {
    while (targets) out.pushSquares(from, popLeastBit(targets), 0, movingKind);
}

template<bool CountOnly>
inline void addPawnMove(LegalCollector<CountOnly>& out, int from, int to, bool promotion) {
    if (!promotion) { out.pushSquares(from, to, 0, PAWN); return; }
    out.pushSquares(from, to, 4, PAWN);
    out.pushSquares(from, to, 3, PAWN);
    out.pushSquares(from, to, 2, PAWN);
    out.pushSquares(from, to, 1, PAWN);
}

inline bool kingDestinationSafeNoMake(const Board* b, int from, int to, int side);

template<bool CountOnly>
void generateKingMoves(Board* b, LegalCollector<CountOnly>& out) {
    const int side = b->sideToMove;
    const int enemy = side ^ 1;
    const int from = b->kingSquare[side];
    const AttackTables& t = attackTables();
    const uint64_t targetsMask = ~(b->occupancy[side] | b->pieceBB[enemy][KING]);
    addTargets(out, from, t.king[from] & targetsMask, KING);

    if (side == WHITE && from == 4 && (b->castlingRights & (WK | WQ)) &&
        !is_square_attacked(b, 4, BLACK)) {
        if ((b->castlingRights & WK) && b->squares[7] == ROOK &&
            b->squares[5] == 0 && b->squares[6] == 0 &&
            kingDestinationSafeNoMake(b, 4, 5, WHITE) &&
            kingDestinationSafeNoMake(b, 4, 6, WHITE))
            out.pushSquares(4, 6, 0, KING);
        if ((b->castlingRights & WQ) && b->squares[0] == ROOK &&
            b->squares[1] == 0 && b->squares[2] == 0 && b->squares[3] == 0 &&
            kingDestinationSafeNoMake(b, 4, 3, WHITE) &&
            kingDestinationSafeNoMake(b, 4, 2, WHITE))
            out.pushSquares(4, 2, 0, KING);
    } else if (side == BLACK && from == 60 && (b->castlingRights & (BK | BQ)) &&
               !is_square_attacked(b, 60, WHITE)) {
        if ((b->castlingRights & BK) && b->squares[63] == -ROOK &&
            b->squares[61] == 0 && b->squares[62] == 0 &&
            kingDestinationSafeNoMake(b, 60, 61, BLACK) &&
            kingDestinationSafeNoMake(b, 60, 62, BLACK))
            out.pushSquares(60, 62, 0, KING);
        if ((b->castlingRights & BQ) && b->squares[56] == -ROOK &&
            b->squares[57] == 0 && b->squares[58] == 0 && b->squares[59] == 0 &&
            kingDestinationSafeNoMake(b, 60, 59, BLACK) &&
            kingDestinationSafeNoMake(b, 60, 58, BLACK))
            out.pushSquares(60, 58, 0, KING);
    }
}

template<bool CountOnly>
void generatePseudo(Board* b, LegalCollector<CountOnly>& out) {
    const int side = b->sideToMove;
    const int pawnStep = side == WHITE ? 8 : -8;
    const int startRank = side == WHITE ? 1 : 6;
    const int promotionRank = side == WHITE ? 7 : 0;
    const uint64_t own = b->occupancy[side];
    const uint64_t enemyKing = b->pieceBB[side ^ 1][KING];
    const uint64_t targetsMask = ~(own | enemyKing);
    const AttackTables& t = attackTables();

    // Handle pawns in their own loop to avoid per-piece type dispatch.
    uint64_t pieces = b->pieceBB[side][PAWN];
    while (pieces) {
        const int from = popLeastBit(pieces);
        const int rank = rankOf(from);
        const int one = from + pawnStep;
        if (static_cast<unsigned>(one) < 64U && b->squares[one] == 0) {
            addPawnMove(out, from, one, rankOf(one) == promotionRank);
            const int two = one + pawnStep;
            if (rank == startRank && b->squares[two] == 0) out.pushSquares(from, two, 0, PAWN);
        }
        uint64_t captures = t.pawn[side][from] & b->occupancy[side ^ 1] & ~enemyKing;
        while (captures) {
            const int to = popLeastBit(captures);
            addPawnMove(out, from, to, rankOf(to) == promotionRank);
        }
        if (b->enPassantSquare >= 0 &&
            (t.pawn[side][from] & (uint64_t{1} << b->enPassantSquare)))
            out.pushSquares(from, b->enPassantSquare, 0, PAWN, true);
    }

    pieces = b->pieceBB[side][KNIGHT];
    while (pieces) {
        const int from = popLeastBit(pieces);
        addTargets(out, from, t.knight[from] & targetsMask, KNIGHT);
    }

    pieces = b->pieceBB[side][BISHOP];
    while (pieces) {
        const int from = popLeastBit(pieces);
        addTargets(out, from, bishopAttacks(from, b->allOccupancy) & targetsMask, BISHOP);
    }

    pieces = b->pieceBB[side][ROOK];
    while (pieces) {
        const int from = popLeastBit(pieces);
        addTargets(out, from, rookAttacks(from, b->allOccupancy) & targetsMask, ROOK);
    }

    pieces = b->pieceBB[side][QUEEN];
    while (pieces) {
        const int from = popLeastBit(pieces);
        addTargets(out, from, (rookAttacks(from, b->allOccupancy) |
                                bishopAttacks(from, b->allOccupancy)) & targetsMask, QUEEN);
    }

    generateKingMoves(b, out);
}

struct CheckInfo {
    uint64_t checkers;
    uint64_t pinned;
    uint64_t evasionMask;
    int checkerCount;
};

inline bool remainsOnKingRay(int king, int from, int to) {
    const int kf = fileOf(king), kr = rankOf(king);
    const int ff = fileOf(from) - kf, fr = rankOf(from) - kr;
    const int tf = fileOf(to) - kf, tr = rankOf(to) - kr;
    return ff * tr == fr * tf;
}

CheckInfo computeCheckInfo(const Board* b) {
    CheckInfo info;
    info.checkers = 0;
    info.pinned = 0;
    info.evasionMask = ~uint64_t{0};
    info.checkerCount = 0;

    const int side = b->sideToMove;
    const int enemy = side ^ 1;
    const int king = b->kingSquare[side];
    const AttackTables& t = attackTables();
    const uint64_t enemyRooksQueens = b->pieceBB[enemy][ROOK] | b->pieceBB[enemy][QUEEN];
    const uint64_t enemyBishopsQueens = b->pieceBB[enemy][BISHOP] | b->pieceBB[enemy][QUEEN];

    info.checkers |= t.pawn[side][king] & b->pieceBB[enemy][PAWN];
    info.checkers |= t.knight[king] & b->pieceBB[enemy][KNIGHT];
    info.checkers |= t.king[king] & b->pieceBB[enemy][KING];

    // The x-rays reveal both direct slider checks (zero intervening pieces)
    // and absolute pins (exactly one friendly blocker), requiring only two lookups.
    const uint64_t occupancyWithoutOwn = b->allOccupancy & ~b->occupancy[side];
    uint64_t pinners = 0;
    if (enemyRooksQueens) {
        pinners = rookAttacks(king, occupancyWithoutOwn) & enemyRooksQueens;
        while (pinners) {
            const int pinner = popLeastBit(pinners);
            const uint64_t between = t.between[king][pinner];
            const uint64_t blockers = between & b->allOccupancy;
            if (!blockers) {
                info.checkers |= uint64_t{1} << pinner;
            } else if (!(blockers & (blockers - 1)) && (blockers & b->occupancy[side])) {
                info.pinned |= blockers;
            }
        }
    }

    if (enemyBishopsQueens) {
        pinners = bishopAttacks(king, occupancyWithoutOwn) & enemyBishopsQueens;
        while (pinners) {
            const int pinner = popLeastBit(pinners);
            const uint64_t between = t.between[king][pinner];
            const uint64_t blockers = between & b->allOccupancy;
            if (!blockers) {
                info.checkers |= uint64_t{1} << pinner;
            } else if (!(blockers & (blockers - 1)) && (blockers & b->occupancy[side])) {
                info.pinned |= blockers;
            }
        }
    }

    info.checkerCount = popCount(info.checkers);
    if (info.checkerCount == 1) {
        uint64_t checkerBits = info.checkers;
        const int checker = popLeastBit(checkerBits);
        info.evasionMask = uint64_t{1} << checker;
        const int piece = absPiece(b->squares[checker]);
        if (piece == BISHOP || piece == ROOK || piece == QUEEN)
            info.evasionMask |= t.between[king][checker];
    } else if (info.checkerCount > 1) {
        info.evasionMask = 0;
    }
    return info;
}


// Hot-path predicate for perft horizon counting. Most positions are neither
// in check nor constrained by absolute pins. Return false as soon as a check
// or pin is found; only then pay for constructing the complete CheckInfo used
// by the general legal-move generator.
inline bool hasNoChecksOrPins(const Board* b) {
    const int side = b->sideToMove;
    const int enemy = side ^ 1;
    const int king = b->kingSquare[side];
    const AttackTables& t = attackTables();

    if ((t.pawn[side][king] & b->pieceBB[enemy][PAWN]) ||
        (t.knight[king] & b->pieceBB[enemy][KNIGHT]) ||
        (t.king[king] & b->pieceBB[enemy][KING]))
        return false;

    const uint64_t own = b->occupancy[side];
    const uint64_t occupancyWithoutOwn = b->allOccupancy & ~own;
    const uint64_t enemyRooksQueens = b->pieceBB[enemy][ROOK] | b->pieceBB[enemy][QUEEN];
    uint64_t pinners = 0;
    if (enemyRooksQueens) {
        pinners = rookAttacks(king, occupancyWithoutOwn) & enemyRooksQueens;
        while (pinners) {
            const int pinner = popLeastBit(pinners);
            const uint64_t blockers = t.between[king][pinner] & b->allOccupancy;
            if (!blockers || (!(blockers & (blockers - 1)) && (blockers & own)))
                return false;
        }
    }

    const uint64_t enemyBishopsQueens = b->pieceBB[enemy][BISHOP] | b->pieceBB[enemy][QUEEN];
    if (enemyBishopsQueens) {
        pinners = bishopAttacks(king, occupancyWithoutOwn) & enemyBishopsQueens;
        while (pinners) {
            const int pinner = popLeastBit(pinners);
            const uint64_t blockers = t.between[king][pinner] & b->allOccupancy;
            if (!blockers || (!(blockers & (blockers - 1)) && (blockers & own)))
                return false;
        }
    }
    return true;
}

template<bool CountOnly>
#if defined(__GNUC__)
__attribute__((always_inline)) inline
#elif defined(_MSC_VER)
__forceinline
#endif
void LegalCollector<CountOnly>::pushSquares(int from, int to, int promotion, int movingKind, bool enPassant) {
    const int side = board->sideToMove;

    // Construct packed moves only when they are needed by the make/unmake
    // path or when collecting moves for divide/UCI. Horizon counting avoids
    // the pack-then-unpack work that the former Move-based interface did.
    if (movingKind == KING || enPassant) {
        const Move move = makeMove(from, to, promotion);
        FastUndo undo;
        board_make_move_fast(board, move, &undo);
        const bool legalMove = !is_in_check(board, side);
        board_unmake_move_fast(board, move, &undo);
        if (legalMove) {
            if constexpr (CountOnly) ++count;
            else list->push(move);
        }
        return;
    }

    if (check.checkerCount > 1) return;
    const uint64_t fromBit = uint64_t{1} << from;
    const uint64_t toBit = uint64_t{1} << to;
    if ((check.pinned & fromBit) &&
        !remainsOnKingRay(board->kingSquare[side], from, to)) return;
    if (check.checkerCount == 1 && !(check.evasionMask & toBit)) return;
    if constexpr (CountOnly) ++count;
    else list->push(makeMove(from, to, promotion));
}

inline bool kingDestinationSafeNoMake(const Board* b, int from, int to, int side) {
    const int enemy = side ^ 1;
    const AttackTables& t = attackTables();
    const uint64_t toBit = uint64_t{1} << to;
    // The king vacates its source and the captured piece disappears. The
    // target square itself is the attack-map origin and is not a ray blocker.
    const uint64_t occupancy = b->allOccupancy & ~(uint64_t{1} << from) & ~toBit;
    if (t.pawn[side][to] & b->pieceBB[enemy][PAWN]) return false;
    if (t.knight[to] & b->pieceBB[enemy][KNIGHT]) return false;
    if (t.king[to] & b->pieceBB[enemy][KING]) return false;
    const uint64_t enemyRooksQueens = b->pieceBB[enemy][ROOK] | b->pieceBB[enemy][QUEEN];
    if (enemyRooksQueens && (rookAttacks(to, occupancy) & enemyRooksQueens)) return false;
    const uint64_t enemyBishopsQueens = b->pieceBB[enemy][BISHOP] | b->pieceBB[enemy][QUEEN];
    if (enemyBishopsQueens && (bishopAttacks(to, occupancy) & enemyBishopsQueens)) return false;
    return true;
}

uint32_t countKingMovesNoCheckNoPins(Board* b) {
    const int side = b->sideToMove;
    const int enemy = side ^ 1;
    const AttackTables& t = attackTables();
    const int from = b->kingSquare[side];
    const uint64_t targetsMask = ~(b->occupancy[side] | b->pieceBB[enemy][KING]);
    uint64_t targets = t.king[from] & targetsMask;
    uint32_t count = 0;
    while (targets) {
        const int to = popLeastBit(targets);
        if (kingDestinationSafeNoMake(b, from, to, side)) ++count;
    }

    // Retain the exact same castling conditions as generateKingMoves.
    if (side == WHITE && from == 4 && (b->castlingRights & (WK | WQ))) {
        if ((b->castlingRights & WK) && b->squares[7] == ROOK &&
            b->squares[5] == 0 && b->squares[6] == 0 &&
            kingDestinationSafeNoMake(b, 4, 5, WHITE) &&
            kingDestinationSafeNoMake(b, 4, 6, WHITE)) ++count;
        if ((b->castlingRights & WQ) && b->squares[0] == ROOK &&
            b->squares[1] == 0 && b->squares[2] == 0 && b->squares[3] == 0 &&
            kingDestinationSafeNoMake(b, 4, 3, WHITE) &&
            kingDestinationSafeNoMake(b, 4, 2, WHITE)) ++count;
    } else if (side == BLACK && from == 60 && (b->castlingRights & (BK | BQ))) {
        if ((b->castlingRights & BK) && b->squares[63] == -ROOK &&
            b->squares[61] == 0 && b->squares[62] == 0 &&
            kingDestinationSafeNoMake(b, 60, 61, BLACK) &&
            kingDestinationSafeNoMake(b, 60, 62, BLACK)) ++count;
        if ((b->castlingRights & BQ) && b->squares[56] == -ROOK &&
            b->squares[57] == 0 && b->squares[58] == 0 && b->squares[59] == 0 &&
            kingDestinationSafeNoMake(b, 60, 59, BLACK) &&
            kingDestinationSafeNoMake(b, 60, 58, BLACK)) ++count;
    }
    return count;
}

uint64_t countLegalMovesNoCheckNoPins(Board* b) {
    const int side = b->sideToMove;
    const int enemy = side ^ 1;
    const uint64_t enemyKing = b->pieceBB[enemy][KING];
    const uint64_t enemyOccupancy = b->occupancy[enemy] & ~enemyKing;
    const uint64_t targetsMask = ~(b->occupancy[side] | enemyKing);
    const AttackTables& t = attackTables();
    uint32_t total = 0;

    // With neither check nor a pinned piece, every non-king move except
    // en-passant is legal. Count pawn moves using whole-bitboard shifts:
    // this avoids iterating each pawn at every horizon node.
    constexpr uint64_t FILE_A = 0x0101010101010101ULL;
    constexpr uint64_t FILE_H = 0x8080808080808080ULL;
    constexpr uint64_t RANK_1 = 0x00000000000000FFULL;
    constexpr uint64_t RANK_2 = 0x000000000000FF00ULL;
    constexpr uint64_t RANK_7 = 0x00FF000000000000ULL;
    constexpr uint64_t RANK_8 = 0xFF00000000000000ULL;

    const uint64_t pawns = b->pieceBB[side][PAWN];
    const uint64_t empty = ~b->allOccupancy;
    const uint64_t promotionRank = side == WHITE ? RANK_8 : RANK_1;
    const uint64_t promotionSourceRank = side == WHITE ? RANK_7 : RANK_2;
    const bool hasPromotionPawn = (pawns & promotionSourceRank) != 0;
    uint64_t pawnPushes = side == WHITE ? ((pawns << 8) & empty) : ((pawns >> 8) & empty);
    // The extra POPCNTs for promotions are skipped in the overwhelmingly
    // common case where there is no pawn on the penultimate rank.
    total += static_cast<uint64_t>(popCount(pawnPushes));
    if (hasPromotionPawn)
        total += 3ULL * static_cast<uint64_t>(popCount(pawnPushes & promotionRank));

    // Double pushes are generated only from the home rank, with both
    // intermediate and destination squares empty.
    const uint64_t homePawns = pawns & (side == WHITE ? RANK_2 : RANK_7);
    const uint64_t oneStep = side == WHITE ? ((homePawns << 8) & empty) : ((homePawns >> 8) & empty);
    const uint64_t twoStep = side == WHITE ? ((oneStep << 8) & empty) : ((oneStep >> 8) & empty);
    total += static_cast<uint64_t>(popCount(twoStep));

    // Pawn captures, including promotions, are also two shifts and POPCNTs.
    uint64_t capturesA, capturesH;
    if (side == WHITE) {
        capturesA = (((pawns & ~FILE_A) << 7) & enemyOccupancy);
        capturesH = (((pawns & ~FILE_H) << 9) & enemyOccupancy);
    } else {
        capturesA = (((pawns & ~FILE_A) >> 9) & enemyOccupancy);
        capturesH = (((pawns & ~FILE_H) >> 7) & enemyOccupancy);
    }
    // Do not union the two attack sets: two adjacent pawns can capture the
    // same target, and those are two distinct moves.
    total += static_cast<uint64_t>(popCount(capturesA));
    total += static_cast<uint64_t>(popCount(capturesH));
    if (hasPromotionPawn) {
        total += 3ULL * static_cast<uint64_t>(popCount(capturesA & promotionRank));
        total += 3ULL * static_cast<uint64_t>(popCount(capturesH & promotionRank));
    }

    // Non-pawn pieces still need their per-origin attack maps. Their move
    // counts are then reduced to hardware POPCNT operations.
    uint64_t pieces = b->pieceBB[side][KNIGHT];
    while (pieces) {
        const int from = popLeastBit(pieces);
        total += static_cast<uint64_t>(popCount(t.knight[from] & targetsMask));
    }
    pieces = b->pieceBB[side][BISHOP];
    while (pieces) {
        const int from = popLeastBit(pieces);
        total += static_cast<uint64_t>(popCount(bishopAttacks(from, b->allOccupancy) & targetsMask));
    }
    pieces = b->pieceBB[side][ROOK];
    while (pieces) {
        const int from = popLeastBit(pieces);
        total += static_cast<uint64_t>(popCount(rookAttacks(from, b->allOccupancy) & targetsMask));
    }
    pieces = b->pieceBB[side][QUEEN];
    while (pieces) {
        const int from = popLeastBit(pieces);
        const uint64_t attacks = rookAttacks(from, b->allOccupancy) |
                                 bishopAttacks(from, b->allOccupancy);
        total += static_cast<uint64_t>(popCount(attacks & targetsMask));
    }

    // En-passant can uncover a rook/bishop/queen attack on the king, so it
    // still needs the ordinary make/check/unmake legality test.
    if (b->enPassantSquare >= 0) {
        const uint64_t epBit = uint64_t{1} << b->enPassantSquare;
        uint64_t epPawns = b->pieceBB[side][PAWN];
        const CheckInfo noCheck{0, 0, ~uint64_t{0}, 0};
        LegalCollector<true> collector{b, nullptr, noCheck};
        while (epPawns) {
            const int from = popLeastBit(epPawns);
            if (t.pawn[side][from] & epBit)
                collector.pushSquares(from, b->enPassantSquare, 0, PAWN, true);
        }
        total += static_cast<uint64_t>(collector.count);
    }

    // King moves and castling are counted without mutating the board; all
    // destinations are checked against attacks under the post-move occupancy.
    total += countKingMovesNoCheckNoPins(b);
    return total;
}

MoveList generateLegalFast(Board* board) {
    MoveList legal;
    const CheckInfo check = computeCheckInfo(board);
    LegalCollector<false> out{board, &legal, check};
    generatePseudo<false>(board, out);
    return legal;
}

uint64_t countLegalMovesFast(Board* board) {
    if (hasNoChecksOrPins(board))
        return countLegalMovesNoCheckNoPins(board);
    const CheckInfo check = computeCheckInfo(board);
    LegalCollector<true> out{board, nullptr, check};
    generatePseudo<true>(board, out);
    return static_cast<uint64_t>(out.count);
}

uint64_t perftRec(Board* board, int depth) {
    if (depth <= 0) return 1;
    if (depth == 1) return countLegalMovesFast(board);
    const MoveList moves = generateLegalFast(board);
    uint64_t nodes = 0;
    for (int i = 0; i < moves.count; ++i) {
        const Move move = moves.moves[i];
        FastUndo undo;
        board_make_move_fast(board, move, &undo);
        nodes += perftRec(board, depth - 1);
        board_unmake_move_fast(board, move, &undo);
    }
    return nodes;
}

// Compile the common benchmark depths as distinct recursion levels. This
// removes the repeated run-time depth test/decrement from interior perft nodes.
template<int Depth>
uint64_t perftRecFixed(Board* board) {
    if constexpr (Depth == 0) {
        return 1;
    } else if constexpr (Depth == 1) {
        return countLegalMovesFast(board);
    } else {
        const MoveList moves = generateLegalFast(board);
        uint64_t nodes = 0;
        for (int i = 0; i < moves.count; ++i) {
            const Move move = moves.moves[i];
            FastUndo undo;
            board_make_move_fast(board, move, &undo);
            nodes += perftRecFixed<Depth - 1>(board);
            board_unmake_move_fast(board, move, &undo);
        }
        return nodes;
    }
}

uint64_t perftDispatch(Board* board, int depth) {
    switch (depth) {
        case 0: return 1;
        case 1: return perftRecFixed<1>(board);
        case 2: return perftRecFixed<2>(board);
        case 3: return perftRecFixed<3>(board);
        case 4: return perftRecFixed<4>(board);
        case 5: return perftRecFixed<5>(board);
        case 6: return perftRecFixed<6>(board);
        case 7: return perftRecFixed<7>(board);
        case 8: return perftRecFixed<8>(board);
        case 9: return perftRecFixed<9>(board);
        case 10: return perftRecFixed<10>(board);
        default: return perftRec(board, depth);
    }
}

} // namespace

void initialize_movegen() { if (!g_attackTablesReady) { buildAttackTables(g_attackTables); g_attackTablesReady = true; } }

bool is_square_attacked(const Board* b, int square, int byColor) {
    if (!b || static_cast<unsigned>(square) >= 64U || (byColor != WHITE && byColor != BLACK)) return false;
    const AttackTables& t = attackTables();
    if (t.pawn[byColor ^ 1][square] & b->pieceBB[byColor][PAWN]) return true;
    if (t.knight[square] & b->pieceBB[byColor][KNIGHT]) return true;
    if (t.king[square] & b->pieceBB[byColor][KING]) return true;
    const uint64_t rooksQueens = b->pieceBB[byColor][ROOK] | b->pieceBB[byColor][QUEEN];
    if (rooksQueens && (rookAttacks(square, b->allOccupancy) & rooksQueens)) return true;
    const uint64_t bishopsQueens = b->pieceBB[byColor][BISHOP] | b->pieceBB[byColor][QUEEN];
    if (bishopsQueens && (bishopAttacks(square, b->allOccupancy) & bishopsQueens)) return true;
    return false;
}

bool is_in_check(const Board* b, int color) {
    if (!b || (color != WHITE && color != BLACK)) return false;
    return is_square_attacked(b, b->kingSquare[color], color ^ 1);
}

std::string move_to_uci(Move move) {
    const int from = fromOf(move), to = toOf(move), promo = promoOf(move);
    std::string result;
    result.reserve(promo ? 5 : 4);
    result.push_back(static_cast<char>('a' + fileOf(from)));
    result.push_back(static_cast<char>('1' + rankOf(from)));
    result.push_back(static_cast<char>('a' + fileOf(to)));
    result.push_back(static_cast<char>('1' + rankOf(to)));
    if (promo > 0 && promo < 5) result.push_back(PROMO_CHARS[promo]);
    return result;
}

std::vector<std::string> generate_legal_moves(const Board* board) {
    std::vector<std::string> out;
    if (!board) return out;
    Board copy = *board;
    const MoveList legal = generateLegalFast(&copy);
    out.reserve(static_cast<size_t>(legal.count));
    for (int i = 0; i < legal.count; ++i) out.push_back(move_to_uci(legal.moves[i]));
    return out;
}

void generate_legal_moves_fast(Board* board, Move* moves, int* count) {
    if (!count) return;
    *count = 0;
    if (!board || !moves) return;
    const MoveList legal = generateLegalFast(board);
    *count = legal.count;
    for (int i = 0; i < legal.count; ++i) moves[i] = legal.moves[i];
}

uint64_t board_perft(Board* board, int depth) {
    if (!board || depth < 0) return 0;
    return perftDispatch(board, depth);
}

std::vector<std::pair<std::string, uint64_t>> board_perft_divide(Board* board, int depth) {
    std::vector<std::pair<std::string, uint64_t>> out;
    if (!board || depth < 1) return out;
    const MoveList moves = generateLegalFast(board);
    out.reserve(static_cast<size_t>(moves.count));
    for (int i = 0; i < moves.count; ++i) {
        const Move move = moves.moves[i];
        FastUndo undo;
        board_make_move_fast(board, move, &undo);
        const uint64_t nodes = depth == 1 ? 1 : perftDispatch(board, depth - 1);
        board_unmake_move_fast(board, move, &undo);
        out.emplace_back(move_to_uci(move), nodes);
    }
    return out;
}

} // namespace promethius
