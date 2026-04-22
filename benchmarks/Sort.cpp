#include <algorithm>
#include <cstdint>
#include <doctest/doctest.h>
#include <execution>
#include <fmt/format.h>
#include <nanobench.h>
#include <numeric>
#include <pbat/common/CountingSort.h>
#include <pbat/common/RadixSort.h>
#include <random>
#include <ranges>
#include <string_view>
#include <tbb/parallel_sort.h>
#include <tuple>
#include <vector>

namespace {

template <int KeyTupleSize, class T, class TCount, class FProject>
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
    output = input;
    bench.run("std::stable_sort", [&]() { std::ranges::stable_sort(output); });
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
            std::array<std::int32_t, KeyTupleSize> mins;
            mins.fill(min);
            std::array<std::int32_t, KeyTupleSize> maxs;
            maxs.fill(max);
            std::array<FProject, KeyTupleSize> fProjects;
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
            std::array<std::int32_t, KeyTupleSize> maxs;
            maxs.fill(max);
            std::array<FProject, KeyTupleSize> fProjects;
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
        std::sort(std::execution::par_unseq, output.begin(), output.end());
    });
    for (auto nThreads = 2; nThreads < nCores; nThreads <<= 1)
    {
        output = input;
        bench.run(fmt::format("tbb::parallel_sort ({} threads)", nThreads), [&]() {
            tbb::global_control gc{
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(nThreads)};
            tbb::parallel_sort(output);
        });
        lwork.resize(nThreads);
        output = input;
        bench.run(fmt::format("pbat::common::RadixSort ({} threads)", nThreads), [&]() {
            std::int32_t max = range - 1;
            if constexpr (pbat::common::CTupleLike<T>)
            {
                std::array<std::int32_t, KeyTupleSize> maxs;
                maxs.fill(max);
                std::array<FProject, KeyTupleSize> fProjects;
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
    int ranges[] = {1 << 8, 1 << 12, 1 << 16, 1 << 20};
    int ns[]     = {1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 24};
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
                RunBenchmark<1>(
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
                RunBenchmark<2>(
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
        SUBCASE("int triplet sorted by first 2 elements")
        {
            using T = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
            std::vector<T> input{};
            std::vector<T> output{};
            std::vector<T> cpy{};
            auto const fSetupAndRunBenchmark = [&](std::string_view test, auto range, auto n) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int32_t> dis(0, range - 1);
                input.resize(n);
                for (size_t i = 0; i < n; ++i)
                    input[i] = {dis(gen), dis(gen), dis(gen)};
                output.resize(input.size());
                cpy = input;
                work.resize(range);
                RunBenchmark<2>(
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
                    fSetupAndRunBenchmark("Uniform integer triplet distribution", range, n);
        }
    }
    SUBCASE("Non-uniform integer distribution")
    {
        SUBCASE("int")
        {
            using T = std::int32_t;
            std::vector<T> input{};
            std::vector<T> output{};
            std::vector<T> cpy{};
            std::vector<std::int32_t> permutation{};
            auto const fSetupAndRunBenchmark =
                [&](std::string_view test, auto range, auto n, auto subrange) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    permutation.resize(range);
                    std::iota(permutation.begin(), permutation.end(), 0);
                    std::shuffle(permutation.begin(), permutation.end(), gen);
                    std::uniform_int_distribution<int32_t> dis(0, subrange - 1);
                    input.resize(n);
                    for (size_t i = 0; i < n; ++i)
                        input[i] = permutation[dis(gen)];
                    output.resize(input.size());
                    cpy = input;
                    work.resize(range);
                    RunBenchmark<1>(
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
                    fSetupAndRunBenchmark("Non-uniform integer distribution", range, n, range / 10);
        }
        SUBCASE("int pair")
        {
            using T = std::pair<std::int32_t, std::int32_t>;
            std::vector<T> input{};
            std::vector<T> output{};
            std::vector<T> cpy{};
            std::vector<std::int32_t> permutation{};
            auto const fSetupAndRunBenchmark =
                [&](std::string_view test, auto range, auto n, auto subrange) {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    permutation.resize(range);
                    std::iota(permutation.begin(), permutation.end(), 0);
                    std::shuffle(permutation.begin(), permutation.end(), gen);
                    std::uniform_int_distribution<int32_t> dis1(0, subrange - 1);
                    std::uniform_int_distribution<int32_t> dis2(range - subrange, range - 1);
                    input.resize(n);
                    for (size_t i = 0; i < n; ++i)
                        input[i] = {permutation[dis1(gen)], permutation[dis2(gen)]};
                    output.resize(input.size());
                    cpy = input;
                    work.resize(range);
                    RunBenchmark<2>(
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
                    fSetupAndRunBenchmark(
                        "Non-uniform integer pair distribution",
                        range,
                        n,
                        range / 20);
        }
    }
}