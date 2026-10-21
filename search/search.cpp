#include "search.h"
#include "evaluation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace promethius {

// These definitions must match board.cpp and movegen.cpp.
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

void board_make_move(Board* board, Move move, Undo* undo);
void board_unmake_move(Board* board, Move move, const Undo* undo);
void generate_legal_moves_fast(Board* board, Move* moves, int* count);
std::string move_to_uci(Move move);
bool is_in_check(const Board* board, int color);

namespace {

using Clock = std::chrono::steady_clock;

constexpr int WHITE = 0, BLACK = 1;
constexpr int PAWN = 1, KNIGHT = 2, BISHOP = 3;
constexpr int ROOK = 4, QUEEN = 5, KING = 6;

constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;

constexpr int INF = 32000;
constexpr int MATE_SCORE = 30000;
constexpr int MATE_BOUND = 29000;

constexpr int VALUE[7] = {
    0, 100, 320, 330, 500, 900, 20000
};

constexpr int TT_BITS = 20;
constexpr size_t TT_SIZE = size_t{1} << TT_BITS;
constexpr size_t TT_MASK = TT_SIZE - 1;

enum Bound : uint8_t {
    BOUND_EXACT = 0,
    BOUND_LOWER = 1,
    BOUND_UPPER = 2
};

inline int moveFrom(Move move) {
    return move & 63;
}

inline int moveTo(Move move) {
    return (move >> 6) & 63;
}

inline int movePromotion(Move move) {
    return (move >> 12) & 7;
}

inline int absPiece(int piece) {
    return piece < 0 ? -piece : piece;
}

inline int popCount(uint64_t b) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
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

// The table persists between searches so later moves can reuse it.
std::vector<TTEntry>& ttTable() {
    static std::vector<TTEntry> table(TT_SIZE);
    return table;
}

uint8_t nextAge() {
    static uint8_t age = 0;
    return ++age;
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
    static const ZobristKeys keys;
    return keys;
}

// Position key excludes the halfmove clock, so repetitions can match.
uint64_t positionKey(const Board* b) {
    const ZobristKeys& z = zobrist();
    uint64_t key = 0;

    for (int sq = 0; sq < 64; ++sq) {
        int piece = b->squares[sq];

        if (piece) {
            key ^= z.piece[piece > 0 ? WHITE : BLACK]
                          [absPiece(piece)][sq];
        }
    }

    if (b->sideToMove == BLACK)
        key ^= z.side;

    key ^= z.castle[b->castlingRights & 15];

    if (b->enPassantSquare >= 0 && b->enPassantSquare < 64)
        key ^= z.ep[b->enPassantSquare];

    return key;
}

inline int scoreToTT(int score, int ply) {
    if (score > MATE_BOUND) return score + ply;
    if (score < -MATE_BOUND) return score - ply;
    return score;
}

inline int scoreFromTT(int score, int ply) {
    if (score > MATE_BOUND) return score - ply;
    if (score < -MATE_BOUND) return score + ply;
    return score;
}

bool isCapture(const Board* b, Move move) {
    int from = moveFrom(move);
    int to = moveTo(move);

    if (b->squares[to] != 0)
        return true;

    // En passant captures an empty target square.
    return absPiece(b->squares[from]) == PAWN &&
           to == b->enPassantSquare &&
           ((from & 7) != (to & 7));
}

int capturedPieceValue(const Board* b, Move move) {
    int victim = absPiece(b->squares[moveTo(move)]);

    if (victim)
        return VALUE[victim];

    int from = moveFrom(move);
    int to = moveTo(move);

    if (absPiece(b->squares[from]) == PAWN &&
        to == b->enPassantSquare &&
        ((from & 7) != (to & 7))) {
        return VALUE[PAWN];
    }

    return 0;
}

int promotionValue(int promo) {
    if (promo <= 0 || promo > 4)
        return 0;

    // Promotion codes: knight=1, bishop=2, rook=3, queen=4.
    return VALUE[promo + 1];
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
        b->pieceBB[WHITE][QUEEN] || b->pieceBB[BLACK][QUEEN]) {
        return false;
    }

    int knights = popCount(b->pieceBB[WHITE][KNIGHT] |
                           b->pieceBB[BLACK][KNIGHT]);

    uint64_t bishops = b->pieceBB[WHITE][BISHOP] |
                       b->pieceBB[BLACK][BISHOP];

    int bishopCount = popCount(bishops);

    if (knights + bishopCount <= 1)
        return true;

    if (knights != 0)
        return false;

    // Bishops alone on the same square colour cannot force mate.
    int squareColor = -1;

    while (bishops) {
#if defined(_MSC_VER)
        unsigned long index = 0;
        _BitScanForward64(&index, bishops);
        int sq = static_cast<int>(index);
#else
        int sq = __builtin_ctzll(bishops);
#endif
        bishops &= bishops - 1;

        int current = ((sq & 7) + (sq >> 3)) & 1;

        if (squareColor < 0)
            squareColor = current;
        else if (squareColor != current)
            return false;
    }

    return true;
}

struct SearchContext {
    SearchControl* control = nullptr;
    uint64_t nodeLimit = 0;
    uint64_t nodes = 0;
    bool stopped = false;

    Clock::time_point start = Clock::now();
    uint8_t age = 0;

    std::array<std::array<Move, 2>, MAX_PLY> killers{};
    int history[2][64][64]{};

    uint64_t path[MAX_PLY]{};
    uint32_t bHalfmoveAtPath[MAX_PLY]{};

    std::vector<TTEntry>& tt;

    explicit SearchContext(SearchControl* c, uint64_t limit)
        : control(c), nodeLimit(limit), tt(ttTable()) {
        age = nextAge();
    }

    bool visitNode() {
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

        // Poll the clock periodically instead of calling it at every node.
        if ((nodes & 127ULL) == 0 || nodes == 1) {
            if (control) {
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

                if (control->stop.load(std::memory_order_relaxed)) {
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
                std::min<uint32_t>(100, bHalfmoveAtPath[ply])));

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

        if (entry.key != key || entry.age != age ||
            depth >= entry.depth - 2 || bound == BOUND_EXACT) {
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
        int side = b->sideToMove;
        int from = moveFrom(move);
        int to = moveTo(move);

        if (promo)
            return 1'000'000 + promotionValue(promo);

        if (isCapture(b, move)) {
            int victim = capturedPieceValue(b, move);
            int attacker = VALUE[absPiece(b->squares[from])];

            return 900'000 + victim * 16 - attacker;
        }

        if (ply < MAX_PLY) {
            if (killers[ply][0] == move)
                return 800'000;

            if (killers[ply][1] == move)
                return 700'000;
        }

        return history[side][from][to];
    }

    void orderMoves(const Board* b, Move* moves, int count,
                    Move ttMove, int ply) const {
        std::sort(moves, moves + count,
            [&](Move a, Move other) {
                return moveOrderScore(b, a, ttMove, ply) >
                       moveOrderScore(b, other, ttMove, ply);
            });
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

    int quiescence(Board* b, int alpha, int beta,
                   int ply, bool allowRepetition) {
        if (!visitNode())
            return evaluate_position(b);

        if (ply >= MAX_PLY - 1)
            return evaluate_position(b);

        uint64_t repKey = positionKey(b);

        path[ply] = repKey;
        bHalfmoveAtPath[ply] = b->halfmoveClock;

        if (b->halfmoveClock >= 100 || insufficientMaterial(b))
            return 0;

        if (allowRepetition && isRepetition(repKey, ply))
            return 0;

        bool inCheck = is_in_check(b, b->sideToMove);

        int standPat = -INF;

        if (!inCheck) {
            standPat = evaluate_position(b);

            if (standPat >= beta)
                return standPat;

            if (standPat > alpha)
                alpha = standPat;
        }

        Move moves[MAX_MOVES];
        int count = 0;

        generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return inCheck ? -MATE_SCORE + ply : 0;

        orderMoves(b, moves, count, 0, ply);

        for (int i = 0; i < count; ++i) {
            Move move = moves[i];

            // Outside check, only search captures and promotions.
            if (!inCheck && !isCapture(b, move) &&
                !movePromotion(move)) {
                continue;
            }

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
                int ply, bool allowRepetition = true) {
        if (depth <= 0)
            return quiescence(b, alpha, beta, ply, allowRepetition);

        if (!visitNode())
            return evaluate_position(b);

        if (ply >= MAX_PLY - 1)
            return evaluate_position(b);

        uint64_t repKey = positionKey(b);

        // TT key additionally includes the halfmove clock.
        uint64_t key = repKey ^
            zobrist().halfmove[std::min<uint32_t>(100, b->halfmoveClock)];

        path[ply] = repKey;
        bHalfmoveAtPath[ply] = b->halfmoveClock;

        if (b->halfmoveClock >= 100 || insufficientMaterial(b))
            return 0;

        if (allowRepetition && isRepetition(repKey, ply))
            return 0;

        bool inCheck = is_in_check(b, b->sideToMove);
        int originalAlpha = alpha;
        bool pvNode = beta - alpha > 1;

        Move ttMove = 0;

        if (TTEntry* entry = probe(key)) {
            ttMove = entry->move;

            if (entry->depth >= depth && !pvNode) {
                int ttScore = scoreFromTT(entry->score, ply);

                if (entry->bound == BOUND_EXACT)
                    return ttScore;

                if (entry->bound == BOUND_LOWER && ttScore >= beta)
                    return ttScore;

                if (entry->bound == BOUND_UPPER && ttScore <= alpha)
                    return ttScore;
            } else if (entry->depth >= depth &&
                       entry->bound == BOUND_EXACT) {
                return scoreFromTT(entry->score, ply);
            }
        }

        // Null-move pruning, disabled in check and low-material positions.
        if (depth >= 3 && !inCheck && !pvNode &&
            beta < MATE_BOUND && beta > -MATE_BOUND &&
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
                ply + 1, false);

            b->sideToMove = oldSide;
            b->enPassantSquare = oldEp;
            b->fullmoveNumber = oldFullmove;

            if (stopped)
                return alpha;

            if (score >= beta)
                return score;
        }

        Move moves[MAX_MOVES];
        int count = 0;

        generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return inCheck ? -MATE_SCORE + ply : 0;

        orderMoves(b, moves, count, ttMove, ply);

        int bestScore = -INF;
        Move bestMove = 0;
        int searched = 0;

        for (int i = 0; i < count; ++i) {
            Move move = moves[i];

            bool quiet =
                !isCapture(b, move) && movePromotion(move) == 0;

            Undo undo{};
            board_make_move(b, move, &undo);

            int score;

            if (searched == 0) {
                score = -negamax(
                    b, depth - 1, -beta, -alpha, ply + 1,
                    allowRepetition);
            } else {
                int reduction = 0;

                // Late-move reduction for quiet, late-ordered moves.
                if (depth >= 3 && searched >= 3 && quiet && !inCheck) {
                    reduction = 1 +
                        (searched >= 8 && depth >= 5 ? 1 : 0);
                }

                // Principal-variation search with a narrow window.
                score = -negamax(
                    b, depth - 1 - reduction, -alpha - 1, -alpha,
                    ply + 1, allowRepetition);

                if (!stopped && score > alpha && reduction > 0) {
                    score = -negamax(
                        b, depth - 1, -alpha - 1, -alpha,
                        ply + 1, allowRepetition);
                }

                if (!stopped && score > alpha && score < beta) {
                    score = -negamax(
                        b, depth - 1, -beta, -alpha, ply + 1,
                        allowRepetition);
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

        Bound bound =
            bestScore <= originalAlpha ? BOUND_UPPER :
            bestScore >= beta ? BOUND_LOWER : BOUND_EXACT;

        store(key, bestMove, bestScore, depth, bound, ply);

        return bestScore;
    }

    bool searchRoot(Board* b, std::vector<Move>& rootMoves,
                    int depth, int& outScore, Move& outMove) {
        if (!visitNode())
            return false;

        uint64_t repKey = positionKey(b);

        uint64_t key = repKey ^
            zobrist().halfmove[std::min<uint32_t>(100, b->halfmoveClock)];

        path[0] = repKey;
        bHalfmoveAtPath[0] = b->halfmoveClock;

        Move ttMove = 0;

        if (TTEntry* entry = probe(key))
            ttMove = entry->move;

        std::sort(rootMoves.begin(), rootMoves.end(),
            [&](Move a, Move c) {
                return moveOrderScore(b, a, ttMove, 0) >
                       moveOrderScore(b, c, ttMove, 0);
            });

        int alpha = -INF;
        const int beta = INF;

        int bestScore = -INF;
        Move bestMove = rootMoves.front();
        int searched = 0;

        for (Move move : rootMoves) {
            Undo undo{};
            board_make_move(b, move, &undo);

            int score;

            if (searched == 0) {
                score = -negamax(
                    b, depth - 1, -beta, -alpha, 1, true);
            } else {
                score = -negamax(
                    b, depth - 1, -alpha - 1, -alpha, 1, true);

                if (!stopped && score > alpha && score < beta) {
                    score = -negamax(
                        b, depth - 1, -beta, -alpha, 1, true);
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
        }

        store(key, bestMove, bestScore, depth, BOUND_EXACT, 0);

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

    Move legalMoves[MAX_MOVES];
    int legalCount = 0;

    generate_legal_moves_fast(board, legalMoves, &legalCount);

    if (legalCount == 0) {
        result.score = is_in_check(board, board->sideToMove)
            ? -MATE_SCORE : 0;

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

    // Always retain a legal fallback move if the search is stopped early.
    result.bestMove = move_to_uci(roots.front());
    result.score = evaluate_position(board);
    result.depth = 0;

    for (int depth = 1;
         depth <= maxDepth && !ctx.stopped;
         ++depth) {

        int score = result.score;
        Move bestMove = roots.front();

        if (!ctx.searchRoot(board, roots, depth, score, bestMove))
            break;

        // Only completed iterations replace the previous best move.
        result.bestMove = move_to_uci(bestMove);
        result.score = score;
        result.depth = depth;
        result.nodes = ctx.nodes;

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
                static_cast<long double>(std::max<uint64_t>(1, ms)));

            onIteration(iteration);
        }

        if (std::abs(score) >= MATE_SCORE - MAX_PLY)
            break;
    }

    result.nodes = ctx.nodes;
    return result;
}

} // namespace promethius