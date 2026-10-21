#include "search.h"
#include "evaluation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

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

void board_make_move(Board*, Move, Undo*);
void board_unmake_move(Board*, Move, const Undo*);
void generate_legal_moves_fast(Board*, Move*, int*);
std::string move_to_uci(Move);
bool is_in_check(const Board*, int);

namespace {

using Clock = std::chrono::steady_clock;

constexpr int WHITE = 0, BLACK = 1;
constexpr int PAWN = 1, KNIGHT = 2, BISHOP = 3;
constexpr int ROOK = 4, QUEEN = 5, KING = 6;

constexpr int INF = 32000;
constexpr int MATE = 30000;
constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;
constexpr size_t TT_SIZE = size_t{1} << 20;
constexpr size_t TT_MASK = TT_SIZE - 1;

constexpr int VALUE[7] = {
    0, 100, 320, 335, 500, 950, 20000
};

enum Bound : uint8_t {
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER
};

inline int moveFrom(Move m) {
    return m & 63;
}

inline int moveTo(Move m) {
    return (m >> 6) & 63;
}

inline int movePromotion(Move m) {
    return (m >> 12) & 7;
}

inline int absPiece(int p) {
    return p < 0 ? -p : p;
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

struct TTEntry {
    uint64_t key = 0;
    Move move = 0;
    int16_t score = 0;
    int16_t depth = -1;
    uint8_t bound = BOUND_EXACT;
    uint8_t age = 0;
};

std::vector<TTEntry>& ttTable() {
    static std::vector<TTEntry> table(TT_SIZE);
    return table;
}

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct ZobristKeys {
    uint64_t piece[2][7][64]{};
    uint64_t side = 0;
    uint64_t castle[16]{};
    uint64_t ep[64]{};
    uint64_t halfmove[101]{};

    ZobristKeys() {
        uint64_t seed = 0x50524f4d45544849ULL;

        for (int c = 0; c < 2; ++c)
            for (int p = 1; p <= 6; ++p)
                for (int s = 0; s < 64; ++s)
                    piece[c][p][s] = splitmix64(seed);

        side = splitmix64(seed);

        for (auto& x : castle) x = splitmix64(seed);
        for (auto& x : ep) x = splitmix64(seed);
        for (auto& x : halfmove) x = splitmix64(seed);
    }
};

const ZobristKeys& zobrist() {
    static const ZobristKeys z;
    return z;
}

uint64_t positionKey(const Board* b) {
    const ZobristKeys& z = zobrist();
    uint64_t key = 0;

    // Iterate over occupied piece bitboards instead of all 64 squares.
    for (int color = WHITE; color <= BLACK; ++color) {
        for (int piece = PAWN; piece <= KING; ++piece) {
            uint64_t bits = b->pieceBB[color][piece];

            while (bits) {
                int sq = popLsb(bits);
                key ^= z.piece[color][piece][sq];
            }
        }
    }

    if (b->sideToMove == BLACK)
        key ^= z.side;

    key ^= z.castle[b->castlingRights & 15];

    if (b->enPassantSquare >= 0 && b->enPassantSquare < 64)
        key ^= z.ep[b->enPassantSquare];

    return key;
}

inline uint64_t ttKey(const Board* b, uint64_t key) {
    return key ^ zobrist().halfmove[
        std::min<uint32_t>(100, b->halfmoveClock)];
}

inline int scoreToTT(int score, int ply) {
    if (score > MATE - MAX_PLY) return score + ply;
    if (score < -MATE + MAX_PLY) return score - ply;
    return score;
}

inline int scoreFromTT(int score, int ply) {
    if (score > MATE - MAX_PLY) return score - ply;
    if (score < -MATE + MAX_PLY) return score + ply;
    return score;
}

bool isCapture(const Board* b, Move m) {
    if (b->squares[moveTo(m)] != 0)
        return true;

    return absPiece(b->squares[moveFrom(m)]) == PAWN &&
           moveTo(m) == b->enPassantSquare &&
           ((moveFrom(m) & 7) != (moveTo(m) & 7));
}

int capturedPieceValue(const Board* b, Move m) {
    int victim = absPiece(b->squares[moveTo(m)]);

    if (victim)
        return VALUE[victim];

    if (isCapture(b, m))
        return VALUE[PAWN];

    return 0;
}

int promotionValue(int promo) {
    return promo >= 1 && promo <= 4 ? VALUE[promo + 1] : 0;
}

// Used by SEE. Pin legality is intentionally approximated here;
// SEE is for ordering captures, not deciding whether a move is legal.
bool attacksSquare(const std::array<int8_t, 64>& board,
                   int from, int target) {
    int piece = board[from];
    if (!piece || from == target)
        return false;

    int kind = absPiece(piece);
    int side = piece > 0 ? WHITE : BLACK;

    int ff = from & 7, fr = from >> 3;
    int tf = target & 7, tr = target >> 3;
    int df = tf - ff, dr = tr - fr;
    int adf = std::abs(df), adr = std::abs(dr);

    if (kind == PAWN)
        return adf == 1 && dr == (side == WHITE ? 1 : -1);

    if (kind == KNIGHT)
        return (adf == 1 && adr == 2) ||
               (adf == 2 && adr == 1);

    if (kind == KING)
        return adf <= 1 && adr <= 1;

    bool diagonal = adf == adr && adf != 0;
    bool straight = (df == 0) != (dr == 0);

    if (kind == BISHOP && !diagonal) return false;
    if (kind == ROOK && !straight) return false;
    if (kind == QUEEN && !diagonal && !straight) return false;

    int stepFile = (df > 0) - (df < 0);
    int stepRank = (dr > 0) - (dr < 0);

    int f = ff + stepFile;
    int r = fr + stepRank;

    while (f != tf || r != tr) {
        if (board[r * 8 + f] != 0)
            return false;

        f += stepFile;
        r += stepRank;
    }

    return true;
}

int staticExchange(const Board* b, Move move) {
    if (!isCapture(b, move) && !movePromotion(move))
        return 0;

    int from = moveFrom(move);
    int to = moveTo(move);
    int promo = movePromotion(move);

    int gain[32]{};
    gain[0] = capturedPieceValue(b, move);

    if (promo)
        gain[0] += promotionValue(promo) - VALUE[PAWN];

    Board temp = *b;
    Undo undo{};
    board_make_move(&temp, move, &undo);

    std::array<int8_t, 64> squares = temp.squares;
    int occupantValue = VALUE[absPiece(squares[to])];
    int side = temp.sideToMove;
    int depth = 0;

    while (depth < 30) {
        int attackerSq = -1;
        int attackerValue = INF;

        for (int sq = 0; sq < 64; ++sq) {
            int p = squares[sq];

            if (!p || (p > 0 ? WHITE : BLACK) != side)
                continue;

            if (!attacksSquare(squares, sq, to))
                continue;

            int v = VALUE[absPiece(p)];

            if (v < attackerValue) {
                attackerValue = v;
                attackerSq = sq;
            }
        }

        if (attackerSq < 0)
            break;

        ++depth;
        gain[depth] = occupantValue - gain[depth - 1];

        int attacker = squares[attackerSq];
        squares[attackerSq] = 0;
        squares[to] = static_cast<int8_t>(attacker);
        occupantValue = VALUE[absPiece(attacker)];

        // Account for an exchange pawn promoting on the target square.
        int targetRank = to >> 3;

        if (absPiece(attacker) == PAWN &&
            ((side == WHITE && targetRank == 7) ||
             (side == BLACK && targetRank == 0))) {
            gain[depth] += VALUE[QUEEN] - VALUE[PAWN];
            squares[to] = static_cast<int8_t>(
                side == WHITE ? QUEEN : -QUEEN);
            occupantValue = VALUE[QUEEN];
        }

        side ^= 1;
    }

    while (depth > 0) {
        gain[depth - 1] =
            -std::max(-gain[depth - 1], gain[depth]);
        --depth;
    }

    return gain[0];
}

bool hasNullMoveMaterial(const Board* b, int side) {
    if (b->pieceBB[side][ROOK] || b->pieceBB[side][QUEEN])
        return true;

    return popCount(b->pieceBB[side][KNIGHT] |
                    b->pieceBB[side][BISHOP]) >= 2;
}

bool insufficientMaterial(const Board* b) {
    if (b->pieceBB[WHITE][PAWN] || b->pieceBB[BLACK][PAWN] ||
        b->pieceBB[WHITE][ROOK] || b->pieceBB[BLACK][ROOK] ||
        b->pieceBB[WHITE][QUEEN] || b->pieceBB[BLACK][QUEEN])
        return false;

    int knights = popCount(b->pieceBB[WHITE][KNIGHT] |
                           b->pieceBB[BLACK][KNIGHT]);

    uint64_t bishops = b->pieceBB[WHITE][BISHOP] |
                       b->pieceBB[BLACK][BISHOP];

    if (knights + popCount(bishops) <= 1)
        return true;

    if (knights != 0)
        return false;

    int squareColor = -1;

    while (bishops) {
        int sq = popLsb(bishops);
        int current = ((sq & 7) + (sq >> 3)) & 1;

        if (squareColor < 0)
            squareColor = current;
        else if (squareColor != current)
            return false;
    }

    return true;
}

struct OrderedMove {
    Move move = 0;
    int score = 0;
};

struct SearchContext {
    SearchControl* control = nullptr;
    uint64_t nodeLimit = 0;
    uint64_t nodes = 0;
    bool stopped = false;

    Clock::time_point start = Clock::now();
    std::vector<TTEntry>& tt;

    uint8_t age = 0;

    Move killers[MAX_PLY][2]{};
    int history[2][64][64]{};

    uint64_t path[MAX_PLY]{};
    uint32_t pathHalfmove[MAX_PLY]{};
    int evalStack[MAX_PLY]{};

    explicit SearchContext(SearchControl* c, uint64_t limit)
        : control(c), nodeLimit(limit), tt(ttTable()) {
        static uint8_t nextAge = 0;
        age = ++nextAge;
    }

    bool visit() {
        if (stopped)
            return false;

        if (control && control->stop.load(std::memory_order_relaxed)) {
            stopped = true;
            return false;
        }

        if (nodeLimit && nodes >= nodeLimit) {
            stopped = true;
            return false;
        }

        ++nodes;

        if ((nodes & 255ULL) == 0 && control) {
            int64_t deadline =
                control->deadlineNs.load(std::memory_order_relaxed);

            if (deadline > 0) {
                int64_t now =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now().time_since_epoch()).count();

                if (now >= deadline) {
                    stopped = true;
                    return false;
                }
            }
        }

        return true;
    }

    bool isRepetition(uint64_t key, int ply) const {
        if (ply < 2)
            return false;

        int earliest = std::max(
            0, ply - static_cast<int>(
                std::min<uint32_t>(100, pathHalfmove[ply])));

        for (int i = ply - 2; i >= earliest; i -= 2) {
            if (path[i] == key)
                return true;
        }

        return false;
    }

    TTEntry* probe(uint64_t key) {
        TTEntry& entry = tt[static_cast<size_t>(key) & TT_MASK];
        return entry.key == key ? &entry : nullptr;
    }

    void store(uint64_t key, Move move, int score,
               int depth, Bound bound, int ply) {
        TTEntry& entry = tt[static_cast<size_t>(key) & TT_MASK];

        // Keep a deeper entry for the same position. This prevents a
        // shallow iteration from needlessly replacing a deeper result.
        if (entry.key != key || depth >= entry.depth) {
            entry.key = key;
            entry.move = move;
            entry.score = static_cast<int16_t>(scoreToTT(score, ply));
            entry.depth = static_cast<int16_t>(depth);
            entry.bound = bound;
            entry.age = age;
        }
    }

    int moveOrderScore(const Board* b, Move move,
                       Move ttMove, int ply) const {
        if (move == ttMove && move != 0)
            return 2'000'000;

        int promo = movePromotion(move);
        int from = moveFrom(move);
        int to = moveTo(move);

        if (promo) {
            return 1'500'000 + promotionValue(promo) * 8
                   + capturedPieceValue(b, move) * 2;
        }

        if (isCapture(b, move)) {
            int see = staticExchange(b, move);

            int attacker = VALUE[absPiece(b->squares[from])];
            int victim = capturedPieceValue(b, move);

            // Winning/equal exchanges precede killers; losing captures
            // remain searchable but are ordered lower.
            int base = see >= 0 ? 1'100'000 : 500'000;

            return base + see * 16 + victim * 2 - attacker;
        }

        if (ply < MAX_PLY) {
            if (killers[ply][0] == move) return 900'000;
            if (killers[ply][1] == move) return 850'000;
        }

        return history[b->sideToMove][from][to];
    }

    void orderMoves(Board* b, Move* moves, int count,
                    Move ttMove, int ply) {
        std::array<OrderedMove, MAX_MOVES> ordered{};

        for (int i = 0; i < count; ++i) {
            ordered[i].move = moves[i];
            ordered[i].score = moveOrderScore(b, moves[i], ttMove, ply);
        }

        std::sort(ordered.begin(), ordered.begin() + count,
            [](const OrderedMove& a, const OrderedMove& c) {
                return a.score > c.score;
            });

        for (int i = 0; i < count; ++i)
            moves[i] = ordered[i].move;
    }

    void orderRoot(Board* b, std::vector<Move>& moves, Move ttMove) {
        std::array<OrderedMove, MAX_MOVES> ordered{};

        int count = static_cast<int>(moves.size());

        for (int i = 0; i < count; ++i) {
            ordered[i].move = moves[i];
            ordered[i].score = moveOrderScore(b, moves[i], ttMove, 0);
        }

        std::sort(ordered.begin(), ordered.begin() + count,
            [](const OrderedMove& a, const OrderedMove& c) {
                return a.score > c.score;
            });

        for (int i = 0; i < count; ++i)
            moves[i] = ordered[i].move;
    }

    void recordQuietCutoff(const Board* b, Move move,
                           int depth, int ply) {
        if (isCapture(b, move) || movePromotion(move))
            return;

        if (ply < MAX_PLY && killers[ply][0] != move) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = move;
        }

        int& h = history[b->sideToMove]
                        [moveFrom(move)][moveTo(move)];

        h = std::clamp(h + depth * depth * 16, -30000, 30000);
    }

    int qsearch(Board* b, int alpha, int beta,
                int ply, bool allowRepetition) {
        if (!visit())
            return evaluate_position(b);

        if (ply >= MAX_PLY - 1)
            return evaluate_position(b);

        uint64_t key = positionKey(b);
        path[ply] = key;
        pathHalfmove[ply] = b->halfmoveClock;

        bool inCheck = is_in_check(b, b->sideToMove);
        bool draw = b->halfmoveClock >= 100 ||
                    insufficientMaterial(b) ||
                    (allowRepetition && isRepetition(key, ply));

        Move moves[MAX_MOVES];
        int count = -1;

        // A checked king can still be checkmated on a draw-clock boundary.
        if (draw && inCheck) {
            generate_legal_moves_fast(b, moves, &count);

            if (count == 0)
                return -MATE + ply;

            return 0;
        }

        if (draw)
            return 0;

        int standPat = -INF;

        if (!inCheck) {
            standPat = evaluate_position(b);

            if (standPat >= beta) {
                // A stand-pat cutoff must not hide stalemate.
                generate_legal_moves_fast(b, moves, &count);

                if (count == 0)
                    return 0;

                return standPat;
            }

            if (standPat > alpha)
                alpha = standPat;
        }

        if (count < 0)
            generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return inCheck ? -MATE + ply : 0;

        orderMoves(b, moves, count, 0, ply);

        for (int i = 0; i < count; ++i) {
            Move move = moves[i];

            // In check, all evasions must be examined.
            if (!inCheck && !isCapture(b, move) &&
                !movePromotion(move))
                continue;

            Undo undo{};
            board_make_move(b, move, &undo);

            int score = -qsearch(
                b, -beta, -alpha, ply + 1, allowRepetition);

            board_unmake_move(b, move, &undo);

            if (stopped)
                return alpha;

            if (score >= beta)
                return score;

            if (score > alpha)
                alpha = score;
        }

        return alpha;
    }

    int negamax(Board* b, int depth, int alpha, int beta,
                int ply, bool allowRepetition = true,
                bool allowNull = true) {
        if (depth <= 0)
            return qsearch(b, alpha, beta, ply, allowRepetition);

        if (!visit())
            return evaluate_position(b);

        if (ply >= MAX_PLY - 1)
            return evaluate_position(b);

        uint64_t repKey = positionKey(b);
        path[ply] = repKey;
        pathHalfmove[ply] = b->halfmoveClock;

        bool inCheck = is_in_check(b, b->sideToMove);

        bool draw = b->halfmoveClock >= 100 ||
                    insufficientMaterial(b) ||
                    (allowRepetition && isRepetition(repKey, ply));

        // At a draw boundary, verify checkmate first.
        if (draw) {
            if (!inCheck)
                return 0;

            Move drawMoves[MAX_MOVES];
            int drawCount = 0;
            generate_legal_moves_fast(b, drawMoves, &drawCount);

            return drawCount == 0 ? -MATE + ply : 0;
        }

        if (inCheck && depth < MAX_PLY - ply - 1)
            ++depth;

        uint64_t key = ttKey(b, repKey);
        bool pvNode = beta - alpha > 1;
        int originalAlpha = alpha;
        Move ttMove = 0;

        if (TTEntry* entry = probe(key)) {
            ttMove = entry->move;

            if (entry->depth >= depth) {
                int score = scoreFromTT(entry->score, ply);

                if (entry->bound == BOUND_EXACT)
                    return score;

                if (!pvNode && entry->bound == BOUND_LOWER &&
                    score >= beta)
                    return score;

                if (!pvNode && entry->bound == BOUND_UPPER &&
                    score <= alpha)
                    return score;
            }
        }

        Move moves[MAX_MOVES];
        int count = 0;
        generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return inCheck ? -MATE + ply : 0;

        int staticEval = evaluate_position(b);
        evalStack[ply] = staticEval;

        bool improving = ply >= 2 &&
                         staticEval > evalStack[ply - 2];

        // Reverse futility pruning: conservative, shallow, non-PV only.
        if (!pvNode && !inCheck && depth <= 3) {
            int margin = 90 * depth + (improving ? 0 : 35);

            if (staticEval - margin >= beta)
                return staticEval - margin;
        }

        // Null-move pruning, with verification in deeper positions.
        if (allowNull && !pvNode && !inCheck &&
            depth >= 3 && staticEval >= beta - 90 &&
            beta > -MATE + MAX_PLY && beta < MATE - MAX_PLY &&
            hasNullMoveMaterial(b, b->sideToMove)) {

            int oldSide = b->sideToMove;
            int oldEp = b->enPassantSquare;
            uint32_t oldFullmove = b->fullmoveNumber;

            b->sideToMove ^= 1;
            b->enPassantSquare = -1;

            if (oldSide == BLACK)
                ++b->fullmoveNumber;

            int reduction = depth >= 6 ? 3 : 2;

            int score = -negamax(
                b, depth - 1 - reduction, -beta, -beta + 1,
                ply + 1, false, false);

            b->sideToMove = oldSide;
            b->enPassantSquare = oldEp;
            b->fullmoveNumber = oldFullmove;

            if (stopped)
                return alpha;

            if (score >= beta) {
                // Verification reduces false cutoffs in zugzwang-like
                // positions, especially when the search is deep.
                if (depth >= 7) {
                    int verify = negamax(
                        b, depth - reduction, beta - 1, beta,
                        ply, allowRepetition, false);

                    if (stopped)
                        return alpha;

                    if (verify >= beta)
                        return verify;
                } else {
                    return score;
                }
            }
        }

        orderMoves(b, moves, count, ttMove, ply);

        int bestScore = -INF;
        Move bestMove = 0;
        int searched = 0;

        for (int i = 0; i < count; ++i) {
            Move move = moves[i];

            bool quiet = !isCapture(b, move) &&
                         movePromotion(move) == 0;

            int movingSide = b->sideToMove;
            Undo undo{};
            board_make_move(b, move, &undo);

            bool givesCheck = is_in_check(b, b->sideToMove);

            // Prune only late, quiet moves that do not give check.
            if (searched > 0 && !pvNode && !inCheck && depth <= 3 &&
                quiet && !givesCheck && move != ttMove &&
                move != killers[ply][0] && move != killers[ply][1]) {

                int margin = 110 * depth;

                if (staticEval + margin <= alpha) {
                    board_unmake_move(b, move, &undo);
                    continue;
                }
            }

            int score;

            if (searched == 0) {
                score = -negamax(
                    b, depth - 1, -beta, -alpha, ply + 1,
                    allowRepetition, true);
            } else {
                int reduction = 0;

                if (depth >= 3 && searched >= 3 && quiet &&
                    !inCheck && !givesCheck &&
                    move != killers[ply][0] &&
                    move != killers[ply][1]) {

                    reduction = 1;

                    if (depth >= 6 && searched >= 8)
                        ++reduction;

                    if (history[movingSide][moveFrom(move)][moveTo(move)] < -100)
                        ++reduction;
                }

                reduction = std::clamp(
                    reduction, 0, std::max(0, depth - 2));

                score = -negamax(
                    b, depth - 1 - reduction, -alpha - 1, -alpha,
                    ply + 1, allowRepetition, true);

                if (!stopped && reduction > 0 && score > alpha) {
                    score = -negamax(
                        b, depth - 1, -alpha - 1, -alpha,
                        ply + 1, allowRepetition, true);
                }

                if (!stopped && score > alpha && score < beta) {
                    score = -negamax(
                        b, depth - 1, -beta, -alpha, ply + 1,
                        allowRepetition, true);
                }
            }

            board_unmake_move(b, move, &undo);

            if (stopped)
                return alpha;

            ++searched;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha)
                alpha = score;

            if (alpha >= beta) {
                if (quiet)
                    recordQuietCutoff(b, move, depth, ply);

                break;
            }
        }

        if (searched == 0)
            return staticEval;

        Bound bound = bestScore <= originalAlpha ? BOUND_UPPER :
                      bestScore >= beta ? BOUND_LOWER : BOUND_EXACT;

        store(key, bestMove, bestScore, depth, bound, ply);
        return bestScore;
    }

    bool searchRoot(Board* b, std::vector<Move>& rootMoves,
                    int depth, int alpha, int beta,
                    int& outScore, Move& outMove) {
        if (!visit())
            return false;

        uint64_t repKey = positionKey(b);
        uint64_t key = ttKey(b, repKey);

        path[0] = repKey;
        pathHalfmove[0] = b->halfmoveClock;
        evalStack[0] = evaluate_position(b);

        Move ttMove = 0;
        if (TTEntry* entry = probe(key))
            ttMove = entry->move;

        orderRoot(b, rootMoves, ttMove);

        int originalAlpha = alpha;
        int bestScore = -INF;
        Move bestMove = rootMoves.front();
        int searched = 0;

        for (Move move : rootMoves) {
            Undo undo{};
            board_make_move(b, move, &undo);

            int score;

            if (searched == 0) {
                score = -negamax(
                    b, depth - 1, -beta, -alpha, 1, true, true);
            } else {
                score = -negamax(
                    b, depth - 1, -alpha - 1, -alpha, 1, true, true);

                if (!stopped && score > alpha && score < beta) {
                    score = -negamax(
                        b, depth - 1, -beta, -alpha, 1, true, true);
                }
            }

            board_unmake_move(b, move, &undo);

            if (stopped)
                return false;

            ++searched;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha)
                alpha = score;

            if (alpha >= beta)
                break;
        }

        Bound bound = bestScore <= originalAlpha ? BOUND_UPPER :
                      bestScore >= beta ? BOUND_LOWER : BOUND_EXACT;

        store(key, bestMove, bestScore, depth, bound, 0);

        outScore = bestScore;
        outMove = bestMove;
        return true;
    }
};

} // namespace

SearchResult search_best_move(
    Board* board,
    int maxDepth,
    uint64_t nodeLimit,
    const std::vector<std::string>& searchMoves,
    std::function<void(const SearchIteration&)> onIteration,
    SearchControl* control) {

    SearchResult result;

    if (!board)
        return result;

    SearchContext ctx(control, nodeLimit);
    maxDepth = std::clamp(maxDepth, 1, MAX_PLY - 2);

    Move legalMoves[MAX_MOVES];
    int legalCount = 0;
    generate_legal_moves_fast(board, legalMoves, &legalCount);

    if (legalCount == 0) {
        result.score = is_in_check(board, board->sideToMove) ? -MATE : 0;
        return result;
    }

    std::vector<Move> roots;
    roots.reserve(static_cast<size_t>(legalCount));

    if (searchMoves.empty()) {
        for (int i = 0; i < legalCount; ++i)
            roots.push_back(legalMoves[i]);
    } else {
        for (int i = 0; i < legalCount; ++i) {
            std::string uci = move_to_uci(legalMoves[i]);

            if (std::find(searchMoves.begin(), searchMoves.end(), uci)
                != searchMoves.end()) {
                roots.push_back(legalMoves[i]);
            }
        }
    }

    if (roots.empty())
        return result;

    // Always have a legal fallback if the search is stopped mid-iteration.
    result.bestMove = move_to_uci(roots.front());
    result.score = evaluate_position(board);
    result.depth = 0;

    int previousScore = result.score;

    for (int depth = 1; depth <= maxDepth && !ctx.stopped; ++depth) {
        int window = depth >= 4 ? 40 : INF;
        int alpha = std::max(-INF, previousScore - window);
        int beta = std::min(INF, previousScore + window);

        int score = previousScore;
        Move bestMove = roots.front();
        bool completed = false;

        while (!ctx.stopped) {
            if (!ctx.searchRoot(board, roots, depth, alpha, beta,
                                score, bestMove))
                break;

            // Re-search with wider aspiration windows on failure.
            if (window < INF && (score <= alpha || score >= beta)) {
                window = std::min(INF, window * 2);
                alpha = std::max(-INF, score - window);
                beta = std::min(INF, score + window);
                continue;
            }

            completed = true;
            break;
        }

        if (!completed)
            break;

        result.bestMove = move_to_uci(bestMove);
        result.score = score;
        result.depth = depth;
        result.nodes = ctx.nodes;
        previousScore = score;

        if (onIteration) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - ctx.start).count();

            uint64_t ms = static_cast<uint64_t>(
                std::max<int64_t>(0, elapsed));

            SearchIteration iteration;
            iteration.depth = depth;
            iteration.score = score;
            iteration.nodes = ctx.nodes;
            iteration.timeMs = ms;
            iteration.nps = static_cast<uint64_t>(
                static_cast<long double>(ctx.nodes) * 1000.0L /
                std::max<uint64_t>(1, ms));

            onIteration(iteration);
        }

        if (std::abs(score) >= MATE - MAX_PLY)
            break;
    }

    result.nodes = ctx.nodes;
    return result;
}

} // namespace promethius
