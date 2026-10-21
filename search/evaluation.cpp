#include "evaluation.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace promethius {

// Must match Board in board.cpp and movegen.cpp.
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

constexpr int MG_VALUE[7] = {0, 100, 325, 335, 500, 975, 0};
constexpr int EG_VALUE[7] = {0, 120, 310, 330, 520, 950, 0};
constexpr int PHASE_WEIGHT[7] = {0, 0, 1, 1, 2, 4, 0};
constexpr int MAX_PHASE = 24;

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
    int rank = rankOf(sq);
    return color == WHITE ? rank : 7 - rank;
}

inline int centerScore(int file, int rank) {
    return 7 - (std::abs(file * 2 - 7) +
                std::abs(rank * 2 - 7)) / 2;
}

inline int popCount(uint64_t b) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}

bool isPassedPawn(const Board* b, int sq, int color) {
    int file = fileOf(sq);
    int rank = relativeRank(sq, color);

    uint64_t enemyPawns = b->pieceBB[color ^ 1][PAWN];

    while (enemyPawns) {
#if defined(_MSC_VER)
        unsigned long index = 0;
        _BitScanForward64(&index, enemyPawns);
        int other = static_cast<int>(index);
#else
        int other = __builtin_ctzll(enemyPawns);
#endif
        enemyPawns &= enemyPawns - 1;

        int enemyFile = fileOf(other);

        if (std::abs(enemyFile - file) > 1)
            continue;

        if (relativeRank(other, color) > rank)
            return false;
    }

    return true;
}

int evaluateForWhite(const Board* b) {
    int mg = 0, eg = 0, phase = 0;
    int pawnCount[2][8]{};

    // Count pawns on every file.
    for (int color = WHITE; color <= BLACK; ++color) {
        uint64_t pawns = b->pieceBB[color][PAWN];

        while (pawns) {
#if defined(_MSC_VER)
            unsigned long index = 0;
            _BitScanForward64(&index, pawns);
            int sq = static_cast<int>(index);
#else
            int sq = __builtin_ctzll(pawns);
#endif
            pawns &= pawns - 1;
            ++pawnCount[color][fileOf(sq)];
        }
    }

    // Material and piece-square evaluation.
    for (int sq = 0; sq < 64; ++sq) {
        int piece = b->squares[sq];
        if (!piece) continue;

        int color = piece > 0 ? WHITE : BLACK;
        int kind = absPiece(piece);
        int sign = color == WHITE ? 1 : -1;

        int file = fileOf(sq);
        int rank = relativeRank(sq, color);
        int center = centerScore(file, rank);

        int mgPst = 0, egPst = 0;

        switch (kind) {
        case PAWN:
            mgPst = rank * 8 + center * 2;
            egPst = rank * 13 + center;
            break;

        case KNIGHT:
            mgPst = center * 8 - (rank == 0 ? 5 : 0);
            egPst = center * 5;
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

            // Encourage castled or otherwise sheltered kings.
            if (rank <= 1 &&
                (file == 1 || file == 2 || file == 5 || file == 6))
                mgPst += 22;
            break;
        }

        mg += sign * (MG_VALUE[kind] + mgPst);
        eg += sign * (EG_VALUE[kind] + egPst);
        phase += PHASE_WEIGHT[kind];
    }

    // Doubled, isolated and passed pawns.
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
#if defined(_MSC_VER)
            unsigned long index = 0;
            _BitScanForward64(&index, pawns);
            int sq = static_cast<int>(index);
#else
            int sq = __builtin_ctzll(pawns);
#endif
            pawns &= pawns - 1;

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

            // Passed pawns become increasingly valuable as they advance.
            if (rank >= 2 && rank <= 6 &&
                isPassedPawn(b, sq, color)) {
                mg += sign * passerMG[rank];
                eg += sign * passerEG[rank];
            }
        }
    }

    // Bishop pair, open files and king pawn shields.
    for (int color = WHITE; color <= BLACK; ++color) {
        int sign = color == WHITE ? 1 : -1;

        if (popCount(b->pieceBB[color][BISHOP]) >= 2) {
            mg += sign * 28;
            eg += sign * 38;
        }

        uint64_t rooks = b->pieceBB[color][ROOK];

        while (rooks) {
#if defined(_MSC_VER)
            unsigned long index = 0;
            _BitScanForward64(&index, rooks);
            int sq = static_cast<int>(index);
#else
            int sq = __builtin_ctzll(rooks);
#endif
            rooks &= rooks - 1;

            int file = fileOf(sq);

            // An open or semi-open file helps rooks.
            if (pawnCount[color][file] == 0) {
                mg += sign *
                    (pawnCount[color ^ 1][file] == 0 ? 16 : 8);
                eg += sign * 10;
            }
        }

        // Reward pawns protecting a king near its home rank.
        int kingSq = b->kingSquare[color];
        int kingFile = fileOf(kingSq);
        int kingRank = relativeRank(kingSq, color);

        if (kingRank <= 1) {
            int shield = 0;

            for (int df = -1; df <= 1; ++df) {
                int file = kingFile + df;
                if (file < 0 || file > 7) continue;

                for (int dr = 1; dr <= 2; ++dr) {
                    int rank = kingRank + dr;
                    if (rank > 7) continue;

                    int boardRank =
                        color == WHITE ? rank : 7 - rank;

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

    // Clamp phase because promoted material can exceed the normal starting phase.
    phase = std::clamp(phase, 0, MAX_PHASE);

    return (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
}

} // namespace

int evaluate_position(const Board* board) {
    if (!board) return 0;

    int score = evaluateForWhite(board);

    // Small tempo bonus.
    score += board->sideToMove == WHITE ? 8 : -8;

    return board->sideToMove == WHITE ? score : -score;
}

} // namespace promethius