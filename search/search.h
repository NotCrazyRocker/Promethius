#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace promethius {

struct Board;

struct SearchControl {
    std::atomic<bool> stop{false};
    // Absolute steady-clock time in nanoseconds. Zero means no deadline.
    std::atomic<int64_t> deadlineNs{0};
};

struct SearchIteration {
    int depth = 0;
    int score = 0;
    uint64_t nodes = 0;
    uint64_t timeMs = 0;
    uint64_t nps = 0;
};

struct SearchResult {
    std::string bestMove;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
};

SearchResult search_best_move(
    Board* board,
    int maxDepth,
    uint64_t nodeLimit,
    const std::vector<std::string>& searchMoves,
    std::function<void(const SearchIteration&)> onIteration = {},
    SearchControl* control = nullptr);

} // namespace promethius