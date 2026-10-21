
#include "search.h"
#include "evaluation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

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
    uint64_t occupancy[2]{};
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

constexpr int INF = 32000;
constexpr int MATE = 30000;
constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;
constexpr int WHITE = 0;
constexpr int BLACK = 1;

constexpr int VALUE[7] = {
    0, 100, 320, 335, 500, 950, 20000
};

constexpr size_t TT_SIZE = 1u << 20;

enum Bound : uint8_t {
    EXACT, LOWER, UPPER
};

struct Entry {
    uint64_t key = 0;
    Move move = 0;
    int score = 0;
    int depth = -1;
    Bound bound = EXACT;
    uint8_t age = 0;
};

std::vector<Entry>& table() {
    static std::vector<Entry> t(TT_SIZE);
    return t;
}

uint64_t random64(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Keys {
    uint64_t piece[2][7][64]{};
    uint64_t side = 0;
    uint64_t castle[16]{};
    uint64_t ep[64]{};

    Keys() {
        uint64_t seed = 0x50524f4d45544849ULL;

        for (auto& c : piece)
            for (auto& p : c)
                for (auto& s : p)
                    s = random64(seed);

        side = random64(seed);

        for (auto& x : castle) x = random64(seed);
        for (auto& x : ep) x = random64(seed);
    }
};

const Keys& keys() {
    static const Keys k;
    return k;
}

uint64_t hash(const Board* b) {
    const Keys& k = keys();
    uint64_t h = 0;

    for (int sq = 0; sq < 64; ++sq) {
        int p = b->squares[sq];

        if (p)
            h ^= k.piece[p > 0 ? WHITE : BLACK]
                       [std::abs(p)][sq];
    }

    if (b->sideToMove == BLACK)
        h ^= k.side;

    h ^= k.castle[b->castlingRights & 15];

    if (b->enPassantSquare >= 0 && b->enPassantSquare < 64)
        h ^= k.ep[b->enPassantSquare];

    return h;
}

int from(Move m) { return m & 63; }
int to(Move m) { return (m >> 6) & 63; }
int promo(Move m) { return (m >> 12) & 7; }

int capturedValue(const Board* b, Move m) {
    int victim = std::abs(b->squares[to(m)]);

    if (victim) return VALUE[victim];

    if (std::abs(b->squares[from(m)]) == 1 &&
        to(m) == b->enPassantSquare &&
        (from(m) & 7) != (to(m) & 7))
        return VALUE[1];

    return 0;
}

bool capture(const Board* b, Move m) {
    return capturedValue(b, m) != 0;
}

int promotionValue(Move m) {
    int p = promo(m);
    return p >= 1 && p <= 4 ? VALUE[p + 1] : 0;
}

int mateToTT(int s, int ply) {
    if (s > MATE - MAX_PLY) return s + ply;
    if (s < -MATE + MAX_PLY) return s - ply;
    return s;
}

int mateFromTT(int s, int ply) {
    if (s > MATE - MAX_PLY) return s - ply;
    if (s < -MATE + MAX_PLY) return s + ply;
    return s;
}

struct Context {
    SearchControl* control;
    uint64_t nodeLimit;
    uint64_t nodes = 0;
    bool stopped = false;

    Clock::time_point start = Clock::now();
    std::vector<Entry>& tt = table();
    uint8_t age = 0;

    Move killers[MAX_PLY][2]{};
    int history[2][64][64]{};
    uint64_t path[MAX_PLY]{};
    int pathPly = 0;

    explicit Context(SearchControl* c, uint64_t limit)
        : control(c), nodeLimit(limit) {
        static uint8_t nextAge = 0;
        age = ++nextAge;
    }

    bool visit() {
        if (stopped) return false;

        if (control && control->stop.load(std::memory_order_relaxed)) {
            stopped = true;
            return false;
        }

        if (nodeLimit && nodes >= nodeLimit) {
            stopped = true;
            return false;
        }

        ++nodes;

        if ((nodes & 255) == 0 && control) {
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

    bool repetition(uint64_t h, int ply) const {
        for (int i = ply - 2; i >= 0; i -= 2) {
            if (path[i] == h) return true;
        }
        return false;
    }

    Entry* probe(uint64_t h) {
        Entry& e = tt[h & (TT_SIZE - 1)];
        return e.key == h ? &e : nullptr;
    }

    void store(uint64_t h, Move m, int s, int d,
               Bound b, int ply) {
        Entry& e = tt[h & (TT_SIZE - 1)];

        if (e.key != h || e.age != age ||
            d >= e.depth || b == EXACT) {
            e.key = h;
            e.move = m;
            e.score = mateToTT(s, ply);
            e.depth = d;
            e.bound = b;
            e.age = age;
        }
    }

    int orderScore(const Board* b, Move m, Move ttMove,
                   int ply) const {
        if (m == ttMove && m != 0) return 2000000;

        int p = promo(m);
        if (p) return 1000000 + promotionValue(m);

        if (capture(b, m)) {
            int attacker = std::abs(b->squares[from(m)]);
            return 900000 + capturedValue(b, m) * 16
                          - VALUE[attacker];
        }

        if (ply < MAX_PLY) {
            if (killers[ply][0] == m) return 800000;
            if (killers[ply][1] == m) return 700000;
        }

        return history[b->sideToMove][from(m)][to(m)];
    }

    void order(Board* b, Move* moves, int count,
               Move ttMove, int ply) {
        std::sort(moves, moves + count, [&](Move a, Move c) {
            return orderScore(b, a, ttMove, ply) >
                   orderScore(b, c, ttMove, ply);
        });
    }

    void quietCutoff(const Board* b, Move m, int depth, int ply) {
        if (capture(b, m) || promo(m)) return;

        if (killers[ply][0] != m) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = m;
        }

        int& h = history[b->sideToMove][from(m)][to(m)];
        h = std::clamp(h + depth * depth * 12, -30000, 30000);
    }

    int qsearch(Board* b, int alpha, int beta, int ply) {
        if (!visit()) return evaluate_position(b);
        if (ply >= MAX_PLY - 1) return evaluate_position(b);

        uint64_t h = hash(b);
        path[ply] = h;

        if (b->halfmoveClock >= 100 || repetition(h, ply))
            return 0;

        bool check = is_in_check(b, b->sideToMove);

        // Generate moves before stand-pat cutoffs so stalemate is detected.
        Move moves[MAX_MOVES];
        int count = 0;
        generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return check ? -MATE + ply : 0;

        int stand = -INF;

        if (!check) {
            stand = evaluate_position(b);

            if (stand >= beta) return stand;
            if (stand > alpha) alpha = stand;
        }

        order(b, moves, count, 0, ply);

        for (int i = 0; i < count; ++i) {
            Move m = moves[i];

            if (!check && !capture(b, m) && !promo(m))
                continue;

            Undo u{};
            board_make_move(b, m, &u);
            int score = -qsearch(b, -beta, -alpha, ply + 1);
            board_unmake_move(b, m, &u);

            if (stopped) return alpha;
            if (score >= beta) return score;
            if (score > alpha) alpha = score;
        }

        return alpha;
    }

    int search(Board* b, int depth, int alpha, int beta, int ply) {
        if (depth <= 0) return qsearch(b, alpha, beta, ply);
        if (!visit()) return evaluate_position(b);
        if (ply >= MAX_PLY - 1) return evaluate_position(b);

        uint64_t h = hash(b);
        path[ply] = h;

        if (b->halfmoveClock >= 100 || repetition(h, ply))
            return 0;

        bool check = is_in_check(b, b->sideToMove);

        // Check extensions improve forcing-line visibility.
        if (check && depth < MAX_PLY - ply - 1)
            ++depth;

        int originalAlpha = alpha;
        Move ttMove = 0;

        if (Entry* e = probe(h)) {
            ttMove = e->move;

            if (e->depth >= depth) {
                int s = mateFromTT(e->score, ply);

                if (e->bound == EXACT) return s;
                if (e->bound == LOWER && s >= beta) return s;
                if (e->bound == UPPER && s <= alpha) return s;
            }
        }

        // Null-move pruning, only outside check and with non-pawn material.
        if (depth >= 3 && !check && beta < MATE - MAX_PLY &&
            beta > -MATE + MAX_PLY) {
            int side = b->sideToMove;
            uint64_t nonPawns =
                b->pieceBB[side][2] | b->pieceBB[side][3] |
                b->pieceBB[side][4] | b->pieceBB[side][5];

            if (nonPawns) {
                int oldEp = b->enPassantSquare;
                uint32_t oldHalf = b->halfmoveClock;
                uint32_t oldFull = b->fullmoveNumber;

                b->sideToMove ^= 1;
                b->enPassantSquare = -1;
                ++b->halfmoveClock;
                if (side == BLACK) ++b->fullmoveNumber;

                int reduction = depth >= 6 ? 3 : 2;
                int score = -search(b, depth - 1 - reduction,
                                    -beta, -beta + 1, ply + 1);

                b->sideToMove = side;
                b->enPassantSquare = oldEp;
                b->halfmoveClock = oldHalf;
                b->fullmoveNumber = oldFull;

                if (stopped) return alpha;
                if (score >= beta) return score;
            }
        }

        Move moves[MAX_MOVES];
        int count = 0;
        generate_legal_moves_fast(b, moves, &count);

        if (count == 0)
            return check ? -MATE + ply : 0;

        order(b, moves, count, ttMove, ply);

        int best = -INF;
        Move bestMove = 0;
        int searched = 0;

        for (int i = 0; i < count; ++i) {
            Move m = moves[i];
            bool quiet = !capture(b, m) && !promo(m);

            Undo u{};
            board_make_move(b, m, &u);

            int score;

            if (searched == 0) {
                score = -search(b, depth - 1,
                                -beta, -alpha, ply + 1);
            } else {
                int reduction = 0;

                if (depth >= 3 && searched >= 4 && quiet && !check)
                    reduction = depth >= 6 && searched >= 10 ? 2 : 1;

                score = -search(b, depth - 1 - reduction,
                                -alpha - 1, -alpha, ply + 1);

                if (!stopped && reduction && score > alpha)
                    score = -search(b, depth - 1,
                                    -alpha - 1, -alpha, ply + 1);

                if (!stopped && score > alpha && score < beta)
                    score = -search(b, depth - 1,
                                    -beta, -alpha, ply + 1);
            }

            board_unmake_move(b, m, &u);

            if (stopped) return alpha;

            ++searched;

            if (score > best) {
                best = score;
                bestMove = m;
            }

            if (score > alpha) alpha = score;

            if (alpha >= beta) {
                if (quiet) quietCutoff(b, m, depth, ply);
                break;
            }
        }

        Bound bound = best <= originalAlpha ? UPPER :
                      best >= beta ? LOWER : EXACT;

        store(h, bestMove, best, depth, bound, ply);
        return best;
    }

    bool root(Board* b, std::vector<Move>& moves, int depth,
              int alpha, int beta, int& score, Move& bestMove) {
        if (!visit()) return false;

        uint64_t h = hash(b);
        path[0] = h;

        Move ttMove = 0;
        if (Entry* e = probe(h)) ttMove = e->move;

        std::sort(moves.begin(), moves.end(), [&](Move a, Move c) {
            return orderScore(b, a, ttMove, 0) >
                   orderScore(b, c, ttMove, 0);
        });

        int best = -INF;
        Move chosen = moves.front();
        int searched = 0;

        for (Move m : moves) {
            Undo u{};
            board_make_move(b, m, &u);

            int s;

            if (searched == 0) {
                s = -search(b, depth - 1, -beta, -alpha, 1);
            } else {
                s = -search(b, depth - 1, -alpha - 1, -alpha, 1);

                if (!stopped && s > alpha && s < beta)
                    s = -search(b, depth - 1, -beta, -alpha, 1);
            }

            board_unmake_move(b, m, &u);

            if (stopped) return false;

            ++searched;

            if (s > best) {
                best = s;
                chosen = m;
            }

            if (s > alpha) alpha = s;
        }

        store(h, chosen, best, depth, EXACT, 0);
        score = best;
        bestMove = chosen;
        return true;
    }
};

} // namespace

SearchResult search_best_move(
    Board* board, int maxDepth, uint64_t nodeLimit,
    const std::vector<std::string>& searchMoves,
    std::function<void(const SearchIteration&)> onIteration,
    SearchControl* control) {

    SearchResult result;
    if (!board) return result;

    Context ctx(control, nodeLimit);
    maxDepth = std::clamp(maxDepth, 1, MAX_PLY - 2);

    Move legal[MAX_MOVES];
    int count = 0;
    generate_legal_moves_fast(board, legal, &count);

    if (!count) {
        result.score = is_in_check(board, board->sideToMove) ? -MATE : 0;
        return result;
    }

    std::vector<Move> roots;
    for (int i = 0; i < count; ++i) {
        std::string uci = move_to_uci(legal[i]);

        if (searchMoves.empty() ||
            std::find(searchMoves.begin(), searchMoves.end(), uci)
                != searchMoves.end())
            roots.push_back(legal[i]);
    }

    if (roots.empty()) return result;

    result.bestMove = move_to_uci(roots.front());
    result.score = evaluate_position(board);

    int previousScore = 0;

    for (int depth = 1; depth <= maxDepth && !ctx.stopped; ++depth) {
        int window = depth >= 4 ? 35 : INF;
        int alpha = std::max(-INF, previousScore - window);
        int beta = std::min(INF, previousScore + window);
        int score = previousScore;
        Move best = roots.front();

        while (true) {
            if (!ctx.root(board, roots, depth, alpha, beta, score, best))
                break;

            if (score <= alpha && alpha > -INF) {
                window *= 2;
                alpha = std::max(-INF, score - window);
                continue;
            }

            if (score >= beta && beta < INF) {
                window *= 2;
                beta = std::min(INF, score + window);
                continue;
            }

            break;
        }

        if (ctx.stopped) break;

        result.bestMove = move_to_uci(best);
        result.score = score;
        result.depth = depth;
        result.nodes = ctx.nodes;
        previousScore = score;

        if (onIteration) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - ctx.start).count();

            SearchIteration it;
            it.depth = depth;
            it.score = score;
            it.nodes = ctx.nodes;
            it.timeMs = static_cast<uint64_t>(std::max<int64_t>(0, ms));
            it.nps = ctx.nodes * 1000 /
                std::max<uint64_t>(1, it.timeMs);

            onIteration(it);
        }

        if (std::abs(score) >= MATE - MAX_PLY)
            break;
    }

    result.nodes = ctx.nodes;
    return result;
}

} // namespace promethius
