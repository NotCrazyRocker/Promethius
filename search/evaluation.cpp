#include "evaluation.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace promethius {

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

namespace {

constexpr int WHITE = 0, BLACK = 1;
constexpr int PAWN = 1, KNIGHT = 2, BISHOP = 3;
constexpr int ROOK = 4, QUEEN = 5, KING = 6;

constexpr int MG_VALUE[7] = {
    0, 100, 325, 335, 500, 975, 0
};

constexpr int EG_VALUE[7] = {
    0, 120, 310, 330, 520, 950, 0
};

constexpr int PHASE_WEIGHT[7] = {
    0, 0, 1, 1, 2, 4, 0
};

constexpr int MAX_PHASE = 24;

constexpr int KNIGHT_DELTAS[8][2] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
    { 1, -2}, { 1, 2}, { 2, -1}, { 2, 1}
};

constexpr int RAY_DIRS[8][2] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
};

inline int absPiece(int p) {
    return p < 0 ? -p : p;
}

inline int fileOf(int sq) {
    return sq & 7;
}

inline int rankOf(int sq) {
    return sq >> 3;
}

inline int relativeRank(int sq, int color) {
    return color == WHITE ? rankOf(sq) : 7 - rankOf(sq);
}

inline int centerScore(int file, int rank) {
    return 7 - (std::abs(file * 2 - 7) +
                std::abs(rank * 2 - 7)) / 2;
}

inline int popLsb(uint64_t& bits) {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, bits);
    bits &= bits - 1;
    return static_cast<int>(index);
#else
    int index = __builtin_ctzll(bits);
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

bool isPassedPawn(const Board* b, int sq, int color) {
    int file = fileOf(sq);
    int rank = relativeRank(sq, color);

    uint64_t enemyPawns = b->pieceBB[color ^ 1][PAWN];

    while (enemyPawns) {
        int other = popLsb(enemyPawns);

        if (std::abs(fileOf(other) - file) > 1)
            continue;

        if (relativeRank(other, color) > rank)
            return false;
    }

    return true;
}

bool isSupportedKnightOutpost(const Board* b, int sq, int color) {
    int rank = relativeRank(sq, color);

    if (rank < 3 || rank > 5)
        return false;

    // Require support from a friendly pawn behind the knight.
    int supportRank = color == WHITE ? rankOf(sq) - 1
                                     : rankOf(sq) + 1;

    if (supportRank < 0 || supportRank > 7)
        return false;

    bool supported = false;

    for (int df : {-1, 1}) {
        int file = fileOf(sq) + df;

        if (file < 0 || file > 7)
            continue;

        int supportSq = supportRank * 8 + file;
        int expectedPawn = color == WHITE ? PAWN : -PAWN;

        if (b->squares[supportSq] == expectedPawn)
            supported = true;
    }

    if (!supported)
        return false;

    // An enemy pawn farther forward on an adjacent file can challenge it.
    uint64_t enemyPawns = b->pieceBB[color ^ 1][PAWN];

    while (enemyPawns) {
        int other = popLsb(enemyPawns);

        if (std::abs(fileOf(other) - fileOf(sq)) <= 1 &&
            relativeRank(other, color) > rank) {
            return false;
        }
    }

    return true;
}

int pieceMobility(const Board* b, int sq, int kind, int color) {
    int file = fileOf(sq);
    int rank = rankOf(sq);
    int mobility = 0;

    if (kind == KNIGHT) {
        for (const auto& d : KNIGHT_DELTAS) {
            int f = file + d[0];
            int r = rank + d[1];

            if (f < 0 || f > 7 || r < 0 || r > 7)
                continue;

            int target = r * 8 + f;

            if (!(b->occupancy[color] & (uint64_t{1} << target)))
                ++mobility;
        }

        return mobility;
    }

    int firstDir = kind == BISHOP ? 4 : 0;
    int lastDir = kind == ROOK ? 4 : 8;

    for (int dir = firstDir; dir < lastDir; ++dir) {
        int f = file + RAY_DIRS[dir][0];
        int r = rank + RAY_DIRS[dir][1];

        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int target = r * 8 + f;
            uint64_t bit = uint64_t{1} << target;

            if (b->occupancy[color] & bit)
                break;

            ++mobility;

            if (b->allOccupancy & bit)
                break;

            f += RAY_DIRS[dir][0];
            r += RAY_DIRS[dir][1];
        }
    }

    return mobility;
}

int evaluateForWhite(const Board* b) {
    int mg = 0;
    int eg = 0;
    int phase = 0;

    int pawnCount[2][8]{};

    // Count pawns on each file.
    for (int color = WHITE; color <= BLACK; ++color) {
        uint64_t pawns = b->pieceBB[color][PAWN];

        while (pawns) {
            int sq = popLsb(pawns);
            ++pawnCount[color][fileOf(sq)];
        }
    }

    // Material, piece-square bonuses and piece activity.
    constexpr int mobilityMG[7] = {
        0, 0, 4, 3, 2, 1, 0
    };

    constexpr int mobilityEG[7] = {
        0, 0, 3, 3, 2, 1, 0
    };

    for (int sq = 0; sq < 64; ++sq) {
        int piece = b->squares[sq];

        if (!piece)
            continue;

        int color = piece > 0 ? WHITE : BLACK;
        int kind = absPiece(piece);
        int sign = color == WHITE ? 1 : -1;

        int file = fileOf(sq);
        int rank = relativeRank(sq, color);
        int center = centerScore(file, rank);

        int mgPst = 0;
        int egPst = 0;

        switch (kind) {
        case PAWN:
            mgPst = rank * 8 + center * 2;
            egPst = rank * 13 + center;
            break;

        case KNIGHT:
            mgPst = center * 8 - (rank == 0 ? 5 : 0);
            egPst = center * 5;

            if (isSupportedKnightOutpost(b, sq, color)) {
                mgPst += 14;
                egPst += 7;
            }
            break;

        case BISHOP:
            mgPst = center * 4 + rank * 2;
            egPst = center * 4 + rank;
            break;

        case ROOK:
            mgPst = rank * 2 + (rank == 6 ? 12 : 0);
            egPst = rank * 3;
            break;

        case QUEEN:
            mgPst = center * 2;
            egPst = center * 4;
            break;

        case KING:
            mgPst = -center * 7 + (rank == 0 ? 10 : 0);
            egPst = center * 10;

            if (rank <= 1 &&
                (file == 1 || file == 2 ||
                 file == 5 || file == 6)) {
                mgPst += 22;
            }
            break;
        }

        int mobility = 0;

        if (kind >= KNIGHT && kind <= QUEEN)
            mobility = pieceMobility(b, sq, kind, color);

        mg += sign * (MG_VALUE[kind] + mgPst +
                      mobility * mobilityMG[kind]);

        eg += sign * (EG_VALUE[kind] + egPst +
                      mobility * mobilityEG[kind]);

        phase += PHASE_WEIGHT[kind];
    }

    // Pawn structure and passed pawns.
    constexpr int passerMG[8] = {
        0, 0, 8, 14, 24, 42, 72, 0
    };

    constexpr int passerEG[8] = {
        0, 0, 12, 22, 38, 68, 120, 0
    };

    for (int color = WHITE; color <= BLACK; ++color) {
        int sign = color == WHITE ? 1 : -1;

        // Doubled pawns.
        for (int file = 0; file < 8; ++file) {
            if (pawnCount[color][file] > 1) {
                int extras = pawnCount[color][file] - 1;
                mg -= sign * extras * 14;
                eg -= sign * extras * 20;
            }
        }

        uint64_t pawns = b->pieceBB[color][PAWN];

        while (pawns) {
            int sq = popLsb(pawns);

            int file = fileOf(sq);
            int rank = relativeRank(sq, color);

            bool leftFriendly =
                file > 0 && pawnCount[color][file - 1] > 0;

            bool rightFriendly =
                file < 7 && pawnCount[color][file + 1] > 0;

            // Isolated pawns.
            if (!leftFriendly && !rightFriendly) {
                mg -= sign * 12;
                eg -= sign * 10;
            }

            // Connected pawns supported from behind.
            int supportRank = color == WHITE ? rankOf(sq) - 1
                                             : rankOf(sq) + 1;

            if (supportRank >= 0 && supportRank <= 7) {
                for (int df : {-1, 1}) {
                    int f = file + df;

                    if (f < 0 || f > 7)
                        continue;

                    int supportSq = supportRank * 8 + f;
                    int ownPawn = color == WHITE ? PAWN : -PAWN;

                    if (b->squares[supportSq] == ownPawn) {
                        mg += sign * 8;
                        eg += sign * 12;
                        break;
                    }
                }
            }

            if (rank < 2 || rank > 6 ||
                !isPassedPawn(b, sq, color))
                continue;

            mg += sign * passerMG[rank];
            eg += sign * passerEG[rank];

            // In the endgame, a passed pawn is more dangerous when the
            // enemy king is far away and its own king can support it.
            int enemyKing = b->kingSquare[color ^ 1];
            int ownKing = b->kingSquare[color];

            int enemyDistance = std::max(
                std::abs(fileOf(enemyKing) - fileOf(sq)),
                std::abs(rankOf(enemyKing) - rankOf(sq)));

            int ownDistance = std::max(
                std::abs(fileOf(ownKing) - fileOf(sq)),
                std::abs(rankOf(ownKing) - rankOf(sq)));

            eg += sign * (enemyDistance - ownDistance) * 3;
        }
    }

    // Bishop pair, open files, and king shelter.
    for (int color = WHITE; color <= BLACK; ++color) {
        int sign = color == WHITE ? 1 : -1;

        if (popCount(b->pieceBB[color][BISHOP]) >= 2) {
            mg += sign * 28;
            eg += sign * 38;
        }

        uint64_t rooks = b->pieceBB[color][ROOK];

        while (rooks) {
            int sq = popLsb(rooks);
            int file = fileOf(sq);
            int rank = relativeRank(sq, color);

            if (pawnCount[color][file] == 0) {
                mg += sign *
                    (pawnCount[color ^ 1][file] == 0 ? 16 : 8);
                eg += sign * 10;
            }

            // An active rook on the seventh rank.
            if (rank == 6) {
                mg += sign * 12;
                eg += sign * 18;
            }
        }

        int kingSq = b->kingSquare[color];
        int kingFile = fileOf(kingSq);
        int kingRank = relativeRank(kingSq, color);

        if (kingRank <= 1) {
            int shield = 0;

            for (int df = -1; df <= 1; ++df) {
                int file = kingFile + df;

                if (file < 0 || file > 7)
                    continue;

                for (int dr = 1; dr <= 2; ++dr) {
                    int rank = kingRank + dr;

                    if (rank > 7)
                        continue;

                    int boardRank = color == WHITE ? rank : 7 - rank;
                    int pawnSq = boardRank * 8 + file;
                    int ownPawn = color == WHITE ? PAWN : -PAWN;

                    if (b->squares[pawnSq] == ownPawn) {
                        shield += dr == 1 ? 9 : 5;
                        break;
                    }
                }
            }

            mg += sign * shield;
        }
    }

    // Promoted material can exceed the normal starting phase.
    phase = std::clamp(phase, 0, MAX_PHASE);

    return (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
}

} // namespace

int evaluate_position(const Board* board) {
    if (!board)
        return 0;

    int score = evaluateForWhite(board);
    score += board->sideToMove == WHITE ? 8 : -8;

    return board->sideToMove == WHITE ? score : -score;
}

} // namespace promethius
