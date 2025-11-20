/**
 * @file GoldenSectionSearch.cpp
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Implementation and tests for Golden Section Search
 * @date 2025-11-18
 * @copyright Copyright (c) 2025
 */

#include "GoldenSectionSearch.h"

#include <doctest/doctest.h>

TEST_CASE("[math][optimization] Golden Section Search")
{
    using namespace pbat::math::optimization;

    SUBCASE("Quadratic function")
    {
        // Minimize f(x) = (x - 2)^2 on [0, 5]
        auto f = [](double x) {
            return (x - 2.0) * (x - 2.0);
        };

        auto result = GoldenSectionSearch(f, 0.0, 5.0);

        CHECK(result.converged);
        CHECK(result.xmin == doctest::Approx(2.0).epsilon(1e-6));
        CHECK(result.fmin == doctest::Approx(0.0).epsilon(1e-10));
    }
    SUBCASE("Cubic function")
    {
        // Minimize f(x) = x^3 - 6x^2 + 9x + 1 on [0, 4]
        // Has minimum at x = 3
        auto f = [](double x) {
            return x * x * x - 6.0 * x * x + 9.0 * x + 1.0;
        };

        auto result = GoldenSectionSearch(f, 0.0, 4.0);

        CHECK(result.converged);
        CHECK(result.xmin == doctest::Approx(3.0).epsilon(1e-6));
        CHECK(result.fmin == doctest::Approx(1.0).epsilon(1e-8));
    }
    SUBCASE("Cosine function")
    {
        // Minimize f(x) = cos(x) on [0, 2*pi]
        // Has minimum at x = pi
        auto f = [](double x) {
            return std::cos(x);
        };

        auto result = GoldenSectionSearch(f, 0.0, 2.0 * 3.14159265358979323846);

        CHECK(result.converged);
        CHECK(result.xmin == doctest::Approx(3.14159265358979323846).epsilon(1e-6));
        CHECK(result.fmin == doctest::Approx(-1.0).epsilon(1e-10));
    }
    SUBCASE("Custom tolerances")
    {
        auto f = [](double x) {
            return (x - 1.5) * (x - 1.5);
        };

        // Very tight tolerance
        auto result = GoldenSectionSearch(f, 0.0, 3.0, 1e-12, 1000);

        CHECK(result.converged);
        CHECK(result.xmin == doctest::Approx(1.5).epsilon(1e-10));
        CHECK(result.interval < 1e-11);
    }
    SUBCASE("Float precision")
    {
        auto f = [](float x) {
            return (x - 3.0f) * (x - 3.0f);
        };

        auto result = GoldenSectionSearch<decltype(f), float>(f, 0.0f, 6.0f);

        CHECK(result.converged);
        CHECK(result.xmin == doctest::Approx(3.0f).epsilon(1e-4f));
        CHECK(result.fmin == doctest::Approx(0.0f).epsilon(1e-6f));
    }
    SUBCASE("Swapped interval endpoints")
    {
        // Algorithm should handle a > b by swapping
        auto f = [](double x) {
            return (x - 2.0) * (x - 2.0);
        };

        auto result = GoldenSectionSearch(f, 5.0, 0.0);

        CHECK(result.converged);
        CHECK(result.xmin == doctest::Approx(2.0).epsilon(1e-6));
    }
    SUBCASE("Maximum iterations")
    {
        auto f = [](double x) {
            return (x - 2.0) * (x - 2.0);
        };

        // Very few iterations - should not converge
        auto result = GoldenSectionSearch(f, 0.0, 5.0, 1e-10, 5);

        CHECK_FALSE(result.converged);
        CHECK(result.niters == 5);
    }
}
