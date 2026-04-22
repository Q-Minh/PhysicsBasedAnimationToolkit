#include <algorithm>
#include <cstdint>
#include <doctest/doctest.h>
#include <fmt/format.h>
#include <nanobench.h>
#include <pbat/common/CountingSort.h>
#include <pbat/common/RadixSort.h>
#include <random>
#include <ranges>
#include <vector>

TEST_CASE("Sorting algorithms")
{
    std::vector<std::int32_t> input{};
    std::vector<int32_t> output{};
    std::vector<std::int32_t> cpy{};
    std::vector<std::uint32_t> work{};
    pbat::common::RadixSortCountArray rwork{};
    auto const fSetupBenchmark = [&](auto range, auto n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32_t> dis(0, range - 1);
        input.resize(n);
        for (size_t i = 0; i < n; ++i)
            input[i] = dis(gen);
        output.resize(input.size());
        cpy = input;
        work.resize(range);
    };
    for (auto range : {1 << 4, 1 << 8, 1 << 12, 1 << 16, 1 << 20})
    {
        for (auto n : {1 << 4, 1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 24})
        {
            ankerl::nanobench::Bench bench;
            bench.title(fmt::format("Sorting algorithms (range: {}, n: {})", range, n))
                .relative(true)
                .performanceCounters(true);
            fSetupBenchmark(range, n);
            output = input;
            bench.run("std::sort", [&]() { std::ranges::sort(output); });
            output = input;
            bench.run("pbat::common::CountingSort", [&]() {
                std::int32_t min{0};
                std::int32_t max = range - 1;
                pbat::common::CountingSort(output, work, min, max);
            });
            output = input;
            bench.run("pbat::common::RadixSort", [&]() {
                std::int32_t max = range - 1;
                pbat::common::RadixSort(output, cpy, rwork, std::identity{}, max);
            });
        }
    }
}