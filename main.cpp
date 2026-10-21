// main.cpp
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "search/search.h"

namespace promethius {
struct Board;
Board* board_create();
Board* board_clone(const Board* board);
void board_destroy(Board* board);
bool board_set_fen(Board* board, const std::string& fen, std::string* error);
std::string board_get_fen(const Board* board);
bool board_play_uci_move(Board* board, const std::string& move, std::string* error);
uint64_t board_perft(Board* board, int depth);
std::vector<std::pair<std::string, uint64_t>> board_perft_divide(Board* board, int depth);
int board_side_to_move(const Board* board);
} // namespace promethius

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::string_view START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '\n')) ++i;
        if (i == line.size()) break;
        const size_t begin = i;
        while (i < line.size() && line[i] != ' ' && line[i] != '\t' && line[i] != '\r' && line[i] != '\n') ++i;
        tokens.emplace_back(line.substr(begin, i - begin));
    }
    return tokens;
}

std::string toLower(std::string text) {
    for (char& c : text) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    }
    return text;
}

bool parseInt64(const std::string& text, int64_t& value) {
    if (text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parseInt(const std::string& text, int& value) {
    if (text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

struct SearchLimits {
    int depth = 0;
    int64_t nodes = -1;
    int64_t movetime = -1;
    int64_t wtime = -1;
    int64_t btime = -1;
    int64_t winc = 0;
    int64_t binc = 0;
    int64_t movestogo = 30;
    int64_t mate = 0;
    bool movestogoSpecified = false;
    bool infinite = false;
    bool ponder = false;
    std::vector<std::string> searchMoves;
};

bool isGoKeyword(const std::string& token) {
    const std::string word = toLower(token);
    return word == "depth" || word == "nodes" || word == "movetime" ||
           word == "wtime" || word == "btime" || word == "winc" ||
           word == "binc" || word == "movestogo" || word == "mate" ||
           word == "infinite" || word == "ponder" || word == "searchmoves";
}

SearchLimits parseGo(const std::vector<std::string>& tokens) {
    SearchLimits limits;
    for (size_t i = 1; i < tokens.size(); ++i) {
        const std::string word = toLower(tokens[i]);
        auto nextNumber = [&]() -> int64_t {
            if (i + 1 >= tokens.size() || isGoKeyword(tokens[i + 1])) return 0;
            int64_t value = 0;
            const std::string& text = tokens[++i];
            return parseInt64(text, value) ? value : 0;
        };

        if (word == "depth") {
            limits.depth = static_cast<int>(std::clamp<int64_t>(nextNumber(), 0, 128));
        } else if (word == "nodes") {
            limits.nodes = std::max<int64_t>(0, nextNumber());
        } else if (word == "movetime") {
            limits.movetime = std::max<int64_t>(0, nextNumber());
        } else if (word == "wtime") {
            limits.wtime = std::max<int64_t>(0, nextNumber());
        } else if (word == "btime") {
            limits.btime = std::max<int64_t>(0, nextNumber());
        } else if (word == "winc") {
            limits.winc = std::max<int64_t>(0, nextNumber());
        } else if (word == "binc") {
            limits.binc = std::max<int64_t>(0, nextNumber());
        } else if (word == "movestogo") {
            limits.movestogoSpecified = true;
            limits.movestogo = std::max<int64_t>(1, nextNumber());
        } else if (word == "mate") {
            limits.mate = std::max<int64_t>(0, nextNumber());
        } else if (word == "infinite") {
            limits.infinite = true;
        } else if (word == "ponder") {
            limits.ponder = true;
        } else if (word == "searchmoves") {
            while (i + 1 < tokens.size() && !isGoKeyword(tokens[i + 1])) {
                limits.searchMoves.push_back(toLower(tokens[++i]));
            }
        }
    }
    return limits;
}

// The returned budget leaves time for UCI output and operating-system scheduling.
// movetime is handled as a hard cap: e.g. movetime 100 allocates 90 ms to search.
int64_t allocateSearchTimeMs(const SearchLimits& limits, int sideToMove) {
    if (limits.movetime >= 0) {
        if (limits.movetime == 0) return 0;
        const int64_t safety = std::max<int64_t>(2, std::min<int64_t>(20, limits.movetime / 8));
        return std::max<int64_t>(0, limits.movetime - safety);
    }

    if (limits.infinite) return -1;

    const int64_t remaining = sideToMove == 0 ? limits.wtime : limits.btime;
    if (remaining < 0) {
        // Explicit depth/node/mate limits do not need a default clock.
        if (limits.depth > 0 || limits.nodes >= 0 || limits.mate > 0) return -1;
        return 1000; // Bare "go" gets a finite, useful default.
    }
    if (remaining == 0) return 0;

    const int64_t increment = sideToMove == 0 ? limits.winc : limits.binc;
    const int64_t reserve = std::max<int64_t>(1, std::min<int64_t>(50, remaining / 20));
    const int64_t available = std::max<int64_t>(0, remaining - reserve);
    if (available == 0) return 0;

    const int64_t movesToGo = std::max<int64_t>(1, limits.movestogoSpecified ? limits.movestogo : 30);
    // Floating-point arithmetic here avoids signed overflow for malformed/extreme limits.
    const long double desired = static_cast<long double>(remaining) /
                                    static_cast<long double>(movesToGo + 1) +
                                static_cast<long double>(increment) * 0.75L;
    const int64_t planned = desired >= static_cast<long double>(available)
                                ? available
                                : std::max<int64_t>(1, static_cast<int64_t>(desired));
    return std::min(planned, available);
}

void setDeadlineAfterMs(promethius::SearchControl& control, int64_t budgetMs) {
    if (budgetMs < 0) {
        control.deadlineNs.store(0, std::memory_order_relaxed);
        return;
    }
    constexpr int64_t NS_PER_MS = 1'000'000;
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
    const int64_t maxValue = INT64_MAX;
    const int64_t delta = budgetMs > (maxValue - now) / NS_PER_MS
                              ? maxValue - now
                              : budgetMs * NS_PER_MS;
    control.deadlineNs.store(now + delta, std::memory_order_relaxed);
}

std::string scoreToUci(int score) {
    constexpr int MATE_SCORE = 30000;
    if (score > MATE_SCORE - 1000) {
        const int moves = std::max(1, (MATE_SCORE - score + 1) / 2);
        return "mate " + std::to_string(moves);
    }
    if (score < -MATE_SCORE + 1000) {
        const int moves = -std::max(1, (MATE_SCORE + score + 1) / 2);
        return "mate " + std::to_string(moves);
    }
    return "cp " + std::to_string(score);
}

class Engine {
public:
    Engine() : board_(promethius::board_create()) {}
    ~Engine() {
        stopSearchAndJoin();
        promethius::board_destroy(board_);
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto tokens = tokenize(line);
            if (tokens.empty()) continue;
            const std::string command = toLower(tokens[0]);
            if (command == "uci") handleUci();
            else if (command == "isready") respond("readyok\n");
            else if (command == "ucinewgame") handleNewGame();
            else if (command == "position") handlePosition(tokens);
            else if (command == "setoption") handleSetOption(tokens);
            else if (command == "perft") handlePerft(tokens, 1);
            else if (command == "go") {
                if (tokens.size() >= 2 && toLower(tokens[1]) == "perft") handlePerft(tokens, 2);
                else handleGo(tokens);
            } else if (command == "stop") {
                stopSearchAndJoin();
            } else if (command == "ponderhit") {
                handlePonderHit();
            } else if (command == "quit") {
                stopSearchAndJoin();
                break;
            } else if (command == "d") {
                respond("info string fen " + promethius::board_get_fen(board_) + "\n");
            }
        }
    }

private:
    promethius::Board* board_ = nullptr;
    std::thread searchThread_;
    std::shared_ptr<promethius::SearchControl> activeControl_;
    int64_t activeBudgetMs_ = -1;
    bool infoEnabled_ = true;

    static void respond(const std::string& text) {
        static std::mutex outputMutex;
        std::lock_guard<std::mutex> lock(outputMutex);
        std::cout << text << std::flush;
    }

    void stopSearchAndJoin() {
        if (activeControl_) activeControl_->stop.store(true, std::memory_order_relaxed);
        if (searchThread_.joinable()) searchThread_.join();
        activeControl_.reset();
        activeBudgetMs_ = -1;
    }

    void handlePonderHit() {
        if (activeControl_ && activeBudgetMs_ >= 0 &&
            !activeControl_->stop.load(std::memory_order_relaxed)) {
            setDeadlineAfterMs(*activeControl_, activeBudgetMs_);
        }
    }

    void handleUci() const {
        respond("id name Promethius\nid author grim\n"
                "option name Info type check default true\n"
                "uciok\n");
    }

    void handleNewGame() {
        stopSearchAndJoin();
        promethius::Board* fresh = promethius::board_create();
        if (!fresh) {
            respond("info string error: could not create board\n");
            return;
        }
        promethius::board_destroy(board_);
        board_ = fresh;
    }

    void handlePosition(const std::vector<std::string>& tokens) {
        stopSearchAndJoin();
        if (tokens.size() < 2) return;
        promethius::Board* candidate = promethius::board_create();
        if (!candidate) {
            respond("info string error: could not create temporary board\n");
            return;
        }
        size_t index = 2;
        std::string error;

        if (toLower(tokens[1]) == "startpos") {
            if (!promethius::board_set_fen(candidate, std::string(START_FEN), &error)) {
                respond("info string error: " + error + "\n");
                promethius::board_destroy(candidate);
                return;
            }
        } else if (toLower(tokens[1]) == "fen") {
            if (tokens.size() < 8) {
                respond("info string error: position fen requires six FEN fields\n");
                promethius::board_destroy(candidate);
                return;
            }
            std::string fen;
            fen.reserve(96);
            for (size_t i = 2; i < 8; ++i) {
                if (!fen.empty()) fen.push_back(' ');
                fen += tokens[i];
            }
            if (!promethius::board_set_fen(candidate, fen, &error)) {
                respond("info string error: invalid FEN: " + error + "\n");
                promethius::board_destroy(candidate);
                return;
            }
            index = 8;
        } else {
            respond("info string error: expected startpos or fen after position\n");
            promethius::board_destroy(candidate);
            return;
        }

        if (index < tokens.size()) {
            if (toLower(tokens[index]) != "moves") {
                respond("info string error: expected moves after position\n");
                promethius::board_destroy(candidate);
                return;
            }
            for (++index; index < tokens.size(); ++index) {
                if (!promethius::board_play_uci_move(candidate, tokens[index], &error)) {
                    respond("info string error: position move sequence rejected: " + error + "\n");
                    promethius::board_destroy(candidate);
                    return;
                }
            }
        }
        promethius::board_destroy(board_);
        board_ = candidate;
    }

    void handleSetOption(const std::vector<std::string>& tokens) {
        if (tokens.size() < 5 || toLower(tokens[1]) != "name") return;
        size_t valueIndex = tokens.size();
        for (size_t i = 2; i < tokens.size(); ++i) {
            if (toLower(tokens[i]) == "value") { valueIndex = i; break; }
        }
        if (valueIndex == tokens.size() || valueIndex <= 2 || valueIndex + 1 >= tokens.size()) return;
        std::string name, valueText;
        for (size_t i = 2; i < valueIndex; ++i) {
            if (!name.empty()) name.push_back(' ');
            name += tokens[i];
        }
        for (size_t i = valueIndex + 1; i < tokens.size(); ++i) {
            if (!valueText.empty()) valueText.push_back(' ');
            valueText += tokens[i];
        }
        if (toLower(name) != "info") return;
        const std::string value = toLower(valueText);
        if (value == "true") infoEnabled_ = true;
        else if (value == "false") infoEnabled_ = false;
    }

    void handlePerft(const std::vector<std::string>& tokens, size_t argIndex) {
        bool divide = false;
        if (argIndex < tokens.size() && toLower(tokens[argIndex]) == "divide") { divide = true; ++argIndex; }
        if (argIndex >= tokens.size()) {
            respond("info string usage: perft [divide] <depth>\n");
            return;
        }
        int depth = 0;
        if (!parseInt(tokens[argIndex], depth) || depth < 0 || depth > 20) {
            respond("info string error: perft depth must be an integer from 0 to 20\n");
            return;
        }
        if (divide && depth == 0) {
            respond("info string error: perft divide requires depth >= 1\n");
            return;
        }
        stopSearchAndJoin();

        const auto start = Clock::now();
        uint64_t nodes = 0;
        std::vector<std::pair<std::string, uint64_t>> divideRows;
        if (divide) {
            divideRows = promethius::board_perft_divide(board_, depth);
            for (const auto& row : divideRows) nodes += row.second;
        } else {
            nodes = promethius::board_perft(board_, depth);
        }
        const auto stop = Clock::now();
        auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        if (elapsedNs < 1) elapsedNs = 1;
        const uint64_t elapsedUs = static_cast<uint64_t>((elapsedNs + 500) / 1'000);
        const uint64_t nps = static_cast<uint64_t>((static_cast<long double>(nodes) * 1'000'000'000.0L) / elapsedNs);
        std::ostringstream report;
        for (const auto& row : divideRows) {
            respond("info string " + row.first + " " + std::to_string(row.second) + "\n");
        }
        report << "info string perft depth " << depth
               << " nodes " << nodes
               << " time_ms " << std::fixed << std::setprecision(3)
               << (static_cast<double>(elapsedNs) / 1'000'000.0)
               << " time_us " << elapsedUs
               << " nps " << nps << '\n';
        respond(report.str());
    }

    void handleGo(const std::vector<std::string>& tokens) {
        stopSearchAndJoin();
        if (!board_) {
            respond("bestmove 0000\n");
            return;
        }

        const SearchLimits limits = parseGo(tokens);
        const int sideToMove = promethius::board_side_to_move(board_);
        const int64_t budgetMs = allocateSearchTimeMs(limits, sideToMove);
        int maxDepth = limits.depth > 0 ? limits.depth : 128;
        if (limits.mate > 0) {
            const int64_t mateDepth = limits.mate >= 64 ? 128 : std::max<int64_t>(1, limits.mate * 2);
            maxDepth = std::min(maxDepth, static_cast<int>(mateDepth));
        }
        const uint64_t nodeLimit = limits.nodes >= 0
                                       ? static_cast<uint64_t>(std::max<int64_t>(1, limits.nodes))
                                       : 0ULL;
        std::vector<std::string> searchMoves = limits.searchMoves;
        promethius::Board* searchBoard = promethius::board_clone(board_);
        if (!searchBoard) {
            respond("info string error: could not clone search position\nbestmove 0000\n");
            return;
        }

        auto control = std::make_shared<promethius::SearchControl>();
        activeControl_ = control;
        activeBudgetMs_ = budgetMs;
        if (!limits.ponder) setDeadlineAfterMs(*control, budgetMs);
        const bool reportInfo = infoEnabled_;

        try {
            searchThread_ = std::thread([this, searchBoard, control, maxDepth, nodeLimit,
                                         searchMoves = std::move(searchMoves), reportInfo]() {
                const auto reportIteration = [reportInfo](const promethius::SearchIteration& iteration) {
                    if (!reportInfo) return;
                    std::ostringstream info;
                    info << "info depth " << iteration.depth
                         << " score " << scoreToUci(iteration.score)
                         << " nodes " << iteration.nodes
                         << " time " << iteration.timeMs
                         << " nps " << iteration.nps << '\n';
                    respond(info.str());
                };
                const promethius::SearchResult result = promethius::search_best_move(
                    searchBoard, maxDepth, nodeLimit, searchMoves,
                    reportInfo ? reportIteration : std::function<void(const promethius::SearchIteration&)>{},
                    control.get());
                respond("bestmove " + (result.bestMove.empty() ? std::string("0000") : result.bestMove) + "\n");
                promethius::board_destroy(searchBoard);
            });
        } catch (const std::exception&) {
            promethius::board_destroy(searchBoard);
            activeControl_.reset();
            activeBudgetMs_ = -1;
            respond("info string error: failed to start search thread\nbestmove 0000\n");
        }
    }
};

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    Engine engine;
    engine.run();
    return 0;
}
