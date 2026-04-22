#include <algorithm>
#include <cstdint>
#include <doctest/doctest.h>
#include <execution>
#include <fmt/format.h>
#include <nanobench.h>
#include <pbat/common/CountingSort.h>
#include <pbat/common/RadixSort.h>
#include <random>
#include <ranges>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

template <class T, class TCount, class FProject>
static void RunBenchmark(
    std::vector<T>& input,
    std::vector<T>& output,
    std::vector<T>& cpy,
    std::vector<TCount>& work,
    pbat::common::RadixSortWorkspace<TCount>& rwork,
    std::vector<pbat::common::RadixSortWorkspace<TCount>>& lwork,
    FProject fProject,
    std::size_t nCores,
    int range,
    int n,
    std::string_view test)
{
    ankerl::nanobench::Bench bench;
    bench.title(fmt::format("{} (range: {}, n: {})", test, range, n))
        .relative(true)
        .performanceCounters(true);
    output = input;
    bench.run("std::sort", [&]() { std::ranges::sort(output); });
    // Counting sort is not applicable to ranges of tuples
    if constexpr (not pbat::common::CTupleLike<T>)
    {
        output = input;
        bench.run("pbat::common::CountingSort", [&]() {
            std::int32_t min{0};
            std::int32_t max = range - 1;
            pbat::common::CountingSort(output, work, min, max, fProject);
        });
    }
    output = input;
    bench.run("pbat::common::StableCountingSort", [&]() {
        std::int32_t min{0};
        std::int32_t max = range - 1;
        if constexpr (pbat::common::CTupleLike<T>)
        {
            std::array<std::int32_t, std::tuple_size_v<T>> mins;
            mins.fill(min);
            std::array<std::int32_t, std::tuple_size_v<T>> maxs;
            maxs.fill(max);
            std::array<FProject, std::tuple_size_v<T>> fProjects;
            fProjects.fill(fProject);
            pbat::common::StableCountingSort(output, cpy, work, mins, maxs, fProjects);
        }
        else
        {
            pbat::common::StableCountingSort(output, cpy, work, min, max, fProject);
        }
    });
    output = input;
    bench.run("pbat::common::RadixSort", [&]() {
        std::int32_t max = range - 1;
        if constexpr (pbat::common::CTupleLike<T>)
        {
            std::array<std::int32_t, std::tuple_size_v<T>> maxs;
            maxs.fill(max);
            std::array<FProject, std::tuple_size_v<T>> fProjects;
            fProjects.fill(fProject);
            pbat::common::RadixSort(output, cpy, rwork, fProjects, maxs);
        }
        else
        {
            pbat::common::RadixSort(output, cpy, rwork, fProject, max);
        }
    });
    output = input;
    bench.run("std::sort (parallel)", [&]() {
        std::sort(std::execution::par, output.begin(), output.end());
    });
    for (auto nThreads = 2; nThreads < nCores; nThreads <<= 1)
    {
        lwork.resize(nThreads);
        bench.run(fmt::format("pbat::common::RadixSort ({} threads)", nThreads), [&]() {
            std::int32_t max = range - 1;
            if constexpr (pbat::common::CTupleLike<T>)
            {
                std::array<std::int32_t, std::tuple_size_v<T>> maxs;
                maxs.fill(max);
                std::array<FProject, std::tuple_size_v<T>> fProjects;
                fProjects.fill(fProject);
                pbat::common::RadixSort(output, cpy, lwork, fProjects, maxs);
            }
            else
            {
                pbat::common::RadixSort(output, cpy, lwork, fProject, max);
            }
        });
    }
}

} // namespace

TEST_CASE("Sorting algorithms")
{
    using TCount = std::uint32_t;
    std::vector<TCount> work{};
    pbat::common::RadixSortWorkspace<TCount> rwork{};
    std::vector<pbat::common::RadixSortWorkspace<TCount>> lwork{};
    auto const nCores = 2 * std::thread::hardware_concurrency();
    lwork.reserve(nCores);
    int ranges[] = {1 << 4, 1 << 8, 1 << 12, 1 << 16, 1 << 20};
    int ns[]     = {1 << 4, 1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 24};
    SUBCASE("Uniform integer distribution")
    {
        SUBCASE("int")
        {
            std::vector<std::int32_t> input{};
            std::vector<std::int32_t> output{};
            std::vector<std::int32_t> cpy{};
            auto const fSetupAndRunBenchmark = [&](std::string_view test, auto range, auto n) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int32_t> dis(0, range - 1);
                input.resize(n);
                for (size_t i = 0; i < n; ++i)
                    input[i] = dis(gen);
                output.resize(input.size());
                cpy = input;
                work.resize(range);
                RunBenchmark(
                    input,
                    output,
                    cpy,
                    work,
                    rwork,
                    lwork,
                    std::identity{},
                    nCores,
                    range,
                    n,
                    test);
            };
            for (auto range : ranges)
                for (auto n : ns)
                    fSetupAndRunBenchmark("Uniform integer distribution", range, n);
        }
        SUBCASE("int pair")
        {
            using T = std::pair<std::int32_t, std::int32_t>;
            std::vector<T> input{};
            std::vector<T> output{};
            std::vector<T> cpy{};
            auto const fSetupAndRunBenchmark = [&](std::string_view test, auto range, auto n) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int32_t> dis(0, range - 1);
                input.resize(n);
                for (size_t i = 0; i < n; ++i)
                    input[i] = {dis(gen), dis(gen)};
                output.resize(input.size());
                cpy = input;
                work.resize(range);
                RunBenchmark(
                    input,
                    output,
                    cpy,
                    work,
                    rwork,
                    lwork,
                    std::identity{},
                    nCores,
                    range,
                    n,
                    test);
            };
            for (auto range : ranges)
                for (auto n : ns)
                    fSetupAndRunBenchmark("Uniform integer pair distribution", range, n);
        }
        SUBCASE("int triplet") {}
    }
}