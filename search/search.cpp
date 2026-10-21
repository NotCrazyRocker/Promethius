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
    0, 100, 320, 335, 500, 900, 20000
};

enum Bound : uint8_t {
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER
};

inline int from(Move m) {
    return m & 63;
}

inline int to(Move m) {
    return (m >> 6) & 63;
}

inline int promotion(Move m) {
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
    Bound bound = BOUND_EXACT;
    uint8_t age = 0;
};

std::vector<TTEntry>& transpositionTable() {
    static std::vector<TTEntry> table(TT_SIZE);
    return table;
}

uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Zobrist {
    uint64_t piece[2][7][64]{};
    uint64_t side = 0;
    uint64_t castle[16]{};
    uint64_t ep[64]{};
    uint64_t halfmove[101]{};

    Zobrist() {
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

const Zobrist& zobrist() {
    static const Zobrist z;
    return z;
}

uint64_t positionKey(const Board* b) {
    const Zobrist& z = zobrist();
    uint64_t key = 0;

    // Bitboard iteration avoids scanning all 64 squares for hashing.
    for (int color = WHITE; color <= BLACK; ++color) {
        for (int kind = PAWN; kind <= KING; ++kind) {
            uint64_t pieces = b->pieceBB[color][kind];

            while (pieces) {
                int sq = popLsb(pieces);
                key ^= z.piece[color][kind][sq];
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

uint64_t tableKey(const Board* b, uint64_t key) {
    return key ^ zobrist().halfmove[
        std::min<uint32_t>(100, b->halfmoveClock)];
}

inline int mateToTable(int score, int ply) {
    if (score > MATE - MAX_PLY) return score + ply;
    if (score < -MATE + MAX_PLY) return score - ply;
    return score;
}

inline int mateFromTable(int score, int ply) {
    if (score > MATE - MAX_PLY) return score - ply;
    if (score < -MATE + MAX_PLY) return score + ply;
    return score;
}

int capturedValue(const Board* b, Move m) {
    int victim = absPiece(b->squares[to(m)]);

    if (victim)
        return VALUE[victim];

    if (absPiece(b->squares[from(m)]) == PAWN &&
        to(m) == b->enPassantSquare &&
        ((from(m) & 7) != (to(m) & 7)))
        return VALUE[PAWN];

    return 0;
}

bool isCapture(const Board* b, Move m) {
    return b->squares[to(m)] != 0 ||
           (absPiece(b->squares[from(m)]) == PAWN &&
            to(m) == b->enPassantSquare &&
            ((from(m) & 7) != (to(m) & 7)));
}

int promotionValue(int p) {
    return p >= 1 && p <= 4 ? VALUE[p + 1] : 0;
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

    int whiteKnights = popCount(b->pieceBB[WHITE][KNIGHT]);
    int blackKnights = popCount(b->pieceBB[BLACK][KNIGHT]);

    uint64_t bishops = b->pieceBB[WHITE][BISHOP] |
                       b->pieceBB[BLACK][BISHOP];

    int bishopCount = popCount(bishops);
    int knightCount = whiteKnights + blackKnights;

    if (knightCount + bishopCount <= 1)
        return true;

    // K+NN versus a bare king is a dead position.
    if (bishopCount == 0 &&
        ((whiteKnights == 2 && blackKnights == 0) ||
         (blackKnights == 2 && whiteKnights == 0)))
        return true;

    if (knightCount != 0)
        return false;

    // Bishops alone on one square colour cannot force mate.
    int squareColor = -1;

    while (bishops) {
        int sq = popLsb(bishops);
        int color = ((sq & 7) + (sq >> 3)) & 1;

        if (squareColor < 0)
            squareColor = color;
        else if (squareColor != color)
            return false;
    }

    return true;
}

struct PersistentHeuristics {
    int history[2][64][64]{};
    Move counter[2][64][64]{};
};

PersistentHeuristics& persistentHeuristics() {
    static PersistentHeuristics h;
    return h;
}

struct OrderedMove {
    Move move = 0;
    int score = 0;
};

struct SearchContext {
    SearchControl* control = nullptr;
    uint64_t nodeLimit = 0;
    uint64_t nodes = 0;
    uint64_t repetitionHits = 0;
    bool stopped = false;

    Clock::time_point start = Clock::now();
    std::vector<TTEntry>& tt;
    PersistentHeuristics& heuristics;

    uint8_t age = 0;

    Move killers[MAX_PLY][2]{};
    Move previousMove[MAX_PLY]{};
    uint64_t path[MAX_PLY]{};
    uint32_t pathHalfmove[MAX_PLY]{};
    int evalStack[MAX_PLY]{};

    SearchContext(SearchControl* c, uint64_t limit)
        : control(c), nodeLimit(limit),
          tt(transpositionTable()),
          heuristics(persistentHeuristics()) {
        static uint8_t nextAge = 0;
        age = ++nextAge;

        // Preserve useful quiet-move statistics between moves, but prevent
        // old history from dominating indefinitely.
        for (auto& side : heuristics.history)
            for (auto& source : side)
                for (int& value : source)
                    value = value * 7 / 8;
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

        int reversiblePlies = static_cast<int>(
            std::min<uint32_t>(100, pathHalfmove[ply]));

        int earliest = std::max(0, ply - reversiblePlies);

        for (int i = ply - 2; i >= earliest; i -= 2) {
            if (path[i] == key)
                return true;
        }

        return false;
    }

    TTEntry* probe(uint64_t key) {
        TTEntry& e = tt[static_cast<size_t>(key) & TT_MASK];
        return e.key == key ? &e : nullptr;
    }

    void store(uint64_t key, Move move, int score,
               int depth, Bound bound, int ply) {
        TTEntry& e = tt[static_cast<size_t>(key) & TT_MASK];

        if (e.key != key || e.age != age ||
            depth >= e.depth ||
            (bound == BOUND_EXACT && depth + 2 >= e.depth)) {
            e.key = key;
            e.move = move;
            e.score = static_cast<int16_t>(mateToTable(score, ply));
            e.depth = static_cast<int16_t>(depth);
            e.bound = bound;
            e.age = age;
        }
    }

    Move counterMove(int ply, int side) const {
        if (ply <= 0)
            return 0;

        Move previous = previousMove[ply];
        if (!previous)
            return 0;

        return heuristics.counter[side][from(previous)][to(previous)];
    }

    int moveScore(const Board* b, Move m, Move ttMove,
                  Move counter, int ply) const {
        if (m == ttMove && m != 0)
            return 2'000'000;

        int promo = promotion(m);
        int victim = capturedValue(b, m);

        if (promo)
            return 1'500'000 + promotionValue(promo) * 8 +
                   victim * 2;

        if (victim) {
            int attacker = VALUE[absPiece(b->squares[from(m)])];

            // MVV-LVA ordering, avoiding a costly exchange simulation
            // for every capture at every searched node.
            return 1'100'000 + victim * 16 - attacker;
        }

        if (ply < MAX_PLY) {
            if (m == killers[ply][0]) return 900'000;
            if (m == killers[ply][1]) return 850'000;
        }

        if (m == counter && m != 0)
            return 800'000;

        return heuristics.history[b->sideToMove][from(m)][to(m)];
    }

    void orderMoves(Board* b, Move* moves, int count,
                    Move ttMove, int ply) {
        std::array<OrderedMove, MAX_MOVES> ordered{};
        Move counter = counterMove(ply, b->sideToMove);

        for (int i = 0; i < count; ++i) {
            ordered[i].move = moves[i];
            ordered[i].score = moveScore(
                b, moves[i], ttMove, counter, ply);
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
            ordered[i].score = moveScore(b, moves[i], ttMove, 0, 0);
        }

        std::sort(ordered.begin(), ordered.begin() + count,
            [](const OrderedMove& a, const OrderedMove& c) {
                return a.score > c.score;
            });

        for (int i = 0; i < count; ++i)
            moves[i] = ordered[i].move;
    }

    void updateHistory(int side, Move move, int bonus) {
        int& h = heuristics.history[side][from(move)][to(move)];
        h = std::clamp(h + bonus, -30000, 30000);
    }

    void recordQuietCutoff(int side, Move move, int depth, int ply,
                           const Move* tried, int triedCount) {
        int bonus = depth * depth * 16;

        // Penalize earlier quiet moves that failed to produce a cutoff.
        for (int i = 0; i < triedCount; ++i) {
            if (tried[i] != move)
                updateHistory(side, tried[i], -bonus / 2);
        }

        updateHistory(side, move, bonus);

        if (ply < MAX_PLY && killers[ply][0] != move) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = move;
        }

        if (ply > 0) {
            Move previous = previousMove[ply];

            if (previous) {
                heuristics.counter[side][from(previous)][to(previous)] =
                    move;
            }
        }
    }

    int quiescence(Board* b, int alpha, int beta,
                   int ply, bool allowRepetition) {
        if (!visit())
            return evaluate_position(b);

        if (ply >= MAX_PLY - 1)
            return evaluate_position(b);

        uint64_t key = positionKey(b);
        path[ply] = key;
        pathHalfmove[ply] = b->halfmoveClock;

        bool inCheck = is_in_check(b, b->sideToMove);
        bool repetition = allowRepetition && isRepetition(key, ply);

        if (repetition)
            ++repetitionHits;

        if (b->halfmoveClock >= 100 || insufficientMaterial(b) || repetition) {
            if (!inCheck)
                return 0;

            Move legal[MAX_MOVES];
            int count = 0;
            generate_legal_moves_fast(b, legal, &count);

            return count == 0 ? -MATE + ply : 0;
        }

        Move moves[MAX_MOVES];
        int count = 0;
        generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return inCheck ? -MATE + ply : 0;

        if (!inCheck) {
            int standPat = evaluate_position(b);

            if (standPat >= beta)
                return standPat;

            if (standPat > alpha)
                alpha = standPat;
        }

        orderMoves(b, moves, count, 0, ply);

        for (int i = 0; i < count; ++i) {
            Move move = moves[i];

            if (!inCheck && !isCapture(b, move) && !promotion(move))
                continue;

            previousMove[ply + 1] = move;

            Undo undo{};
            board_make_move(b, move, &undo);

            int score = -quiescence(
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
            return quiescence(b, alpha, beta, ply, allowRepetition);

        if (!visit())
            return evaluate_position(b);

        if (ply >= MAX_PLY - 1)
            return evaluate_position(b);

        uint64_t repKey = positionKey(b);
        path[ply] = repKey;
        pathHalfmove[ply] = b->halfmoveClock;

        bool inCheck = is_in_check(b, b->sideToMove);
        bool repetition = allowRepetition && isRepetition(repKey, ply);

        if (repetition)
            ++repetitionHits;

        if (b->halfmoveClock >= 100 ||
            insufficientMaterial(b) || repetition) {
            if (!inCheck)
                return 0;

            Move legal[MAX_MOVES];
            int count = 0;
            generate_legal_moves_fast(b, legal, &count);

            return count == 0 ? -MATE + ply : 0;
        }

        if (inCheck && depth < MAX_PLY - ply - 1)
            ++depth;

        // Mate-distance pruning.
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);

        if (alpha >= beta)
            return alpha;

        uint64_t key = tableKey(b, repKey);
        int originalAlpha = alpha;
        bool pvNode = beta - alpha > 1;
        Move ttMove = 0;

        // Skip TT reuse during null-move searches. Such positions are
        // artificial and should not pollute results for legal positions.
        if (allowRepetition) {
            if (TTEntry* e = probe(key)) {
                ttMove = e->move;

                if (e->depth >= depth) {
                    int score = mateFromTable(e->score, ply);

                    if (e->bound == BOUND_EXACT)
                        return score;

                    if (!pvNode && e->bound == BOUND_LOWER &&
                        score >= beta)
                        return score;

                    if (!pvNode && e->bound == BOUND_UPPER &&
                        score <= alpha)
                        return score;
                }
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

        // Conservative reverse futility pruning at shallow depth.
        if (!pvNode && !inCheck && depth <= 2) {
            int margin = 100 * depth + (improving ? 0 : 30);

            if (staticEval - margin >= beta)
                return staticEval - margin;
        }

        // Null-move pruning with verification in deeper positions.
        if (allowNull && !pvNode && !inCheck && depth >= 3 &&
            staticEval >= beta - 80 &&
            beta > -MATE + MAX_PLY && beta < MATE - MAX_PLY &&
            hasNullMoveMaterial(b, b->sideToMove)) {

            int oldSide = b->sideToMove;
            int oldEp = b->enPassantSquare;
            uint32_t oldFullmove = b->fullmoveNumber;

            b->sideToMove ^= 1;
            b->enPassantSquare = -1;

            if (oldSide == BLACK)
                ++b->fullmoveNumber;

            previousMove[ply + 1] = 0;

            int reduction = 2 + (depth >= 7 ? 1 : 0);

            int score = -negamax(
                b, depth - 1 - reduction, -beta, -beta + 1,
                ply + 1, false, false);

            b->sideToMove = oldSide;
            b->enPassantSquare = oldEp;
            b->fullmoveNumber = oldFullmove;

            if (stopped)
                return alpha;

            if (score >= beta) {
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
        int quietTriedCount = 0;
        Move quietTried[MAX_MOVES]{};

        // If any repetition affected this node, don't save its result:
        // a draw caused by path history is not a position-only score.
        uint64_t repetitionCountBefore = repetitionHits;

        for (int i = 0; i < count; ++i) {
            Move move = moves[i];

            bool quiet = !isCapture(b, move) && !promotion(move);
            int movingSide = b->sideToMove;

            Undo undo{};
            board_make_move(b, move, &undo);

            bool givesCheck = is_in_check(b, b->sideToMove);

            // Shallow quiet-move futility pruning.
            if (searched > 0 && !pvNode && !inCheck &&
                depth <= 2 && quiet && !givesCheck &&
                move != ttMove && move != killers[ply][0] &&
                move != killers[ply][1] &&
                move != counterMove(ply, movingSide)) {

                int margin = 100 * depth;

                if (staticEval + margin <= alpha) {
                    board_unmake_move(b, move, &undo);
                    continue;
                }
            }

            previousMove[ply + 1] = move;

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
                    move != killers[ply][1] &&
                    move != counterMove(ply, movingSide)) {

                    reduction = 1;

                    if (depth >= 6 && searched >= 8)
                        reduction = 2;

                    int h = heuristics.history[movingSide]
                                               [from(move)][to(move)];

                    if (h > 3000)
                        reduction = std::max(0, reduction - 1);
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
                if (quiet) {
                    recordQuietCutoff(movingSide, move, depth, ply,
                                      quietTried, quietTriedCount);
                }

                break;
            }

            if (quiet && quietTriedCount < MAX_MOVES)
                quietTried[quietTriedCount++] = move;
        }

        if (searched == 0)
            return staticEval;

        // Do not cache path-dependent search results.
        if (allowRepetition && repetitionHits == repetitionCountBefore) {
            Bound bound = bestScore <= originalAlpha ? BOUND_UPPER :
                          bestScore >= beta ? BOUND_LOWER : BOUND_EXACT;

            store(key, bestMove, bestScore, depth, bound, ply);
        }

        return bestScore;
    }

    bool searchRoot(Board* b, std::vector<Move>& moves,
                    int depth, int alpha, int beta,
                    int& outScore, Move& outMove) {
        if (!visit())
            return false;

        uint64_t repKey = positionKey(b);
        uint64_t key = tableKey(b, repKey);

        path[0] = repKey;
        pathHalfmove[0] = b->halfmoveClock;
        evalStack[0] = evaluate_position(b);
        previousMove[0] = 0;

        Move ttMove = 0;

        if (TTEntry* e = probe(key))
            ttMove = e->move;

        orderRoot(b, moves, ttMove);

        int originalAlpha = alpha;
        int bestScore = -INF;
        Move bestMove = moves.front();
        int searched = 0;
        uint64_t repBefore = repetitionHits;

        for (Move move : moves) {
            previousMove[1] = move;

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

        if (repetitionHits == repBefore) {
            Bound bound = bestScore <= originalAlpha ? BOUND_UPPER :
                          bestScore >= beta ? BOUND_LOWER : BOUND_EXACT;

            store(key, bestMove, bestScore, depth, bound, 0);
        }

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
    ctx.start = Clock::now();

    maxDepth = std::clamp(maxDepth, 1, MAX_PLY - 2);

    Move legal[MAX_MOVES];
    int legalCount = 0;
    generate_legal_moves_fast(board, legal, &legalCount);

    if (legalCount == 0) {
        result.score = is_in_check(board, board->sideToMove) ? -MATE : 0;
        return result;
    }

    std::vector<Move> roots;
    roots.reserve(static_cast<size_t>(legalCount));

    if (searchMoves.empty()) {
        for (int i = 0; i < legalCount; ++i)
            roots.push_back(legal[i]);
    } else {
        for (int i = 0; i < legalCount; ++i) {
            std::string uci = move_to_uci(legal[i]);

            if (std::find(searchMoves.begin(), searchMoves.end(), uci)
                != searchMoves.end())
                roots.push_back(legal[i]);
        }
    }

    if (roots.empty())
        return result;

    result.bestMove = move_to_uci(roots.front());
    result.score = evaluate_position(board);

    int previousScore = result.score;

    for (int depth = 1; depth <= maxDepth && !ctx.stopped; ++depth) {
        int window = depth >= 4 ? 40 : INF;

        int alpha = window == INF ? -INF :
                    std::max(-INF, previousScore - window);

        int beta = window == INF ? INF :
                   std::min(INF, previousScore + window);

        int score = previousScore;
        Move bestMove = roots.front();
        bool completed = false;

        while (!ctx.stopped) {
            if (!ctx.searchRoot(board, roots, depth, alpha, beta,
                                score, bestMove))
                break;

            if (score <= alpha && alpha > -INF) {
                window = std::min(INF, window * 2);

                if (window == INF) {
                    alpha = -INF;
                    beta = INF;
                } else {
                    alpha = std::max(-INF, score - window);
                }

                continue;
            }

            if (score >= beta && beta < INF) {
                window = std::min(INF, window * 2);

                if (window == INF) {
                    alpha = -INF;
                    beta = INF;
                } else {
                    beta = std::min(INF, score + window);
                }

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
