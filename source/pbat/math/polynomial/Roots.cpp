#include "Roots.h"

#include "pbat/common/ConstexprFor.h"

#include <cmath>
#include <doctest/doctest.h>
#include <random>

TEST_CASE("[math][polynomial] Roots")
{
    using namespace pbat;
    using namespace pbat::math;

    common::ForRange<2, 6>([]<auto N>() {
        // Arrange
        Scalar constexpr kMaxCoeff = 1e10;
        Scalar constexpr epsilon =
            1e-7; // Tests the residual P(root) \approx 0. Note that the residual is highly
                  // ill-conditioned for high degree polynomials. Thus, we only apply this test to
                  // polynomials of degree up to 5.
        bool bHasRoot{false};
        std::array<Scalar, N + 1> coeffs;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Scalar> dis(-kMaxCoeff, kMaxCoeff);
        do
        {
            for (auto& coeff : coeffs)
                coeff = dis(gen);
            bHasRoot = polynomial::HasRoot<N>(coeffs);
        } while (not bHasRoot);
        // Act
        auto roots = polynomial::Roots<N>(coeffs);
        for (Scalar root : roots)
        {
            if (std::isnan(root))
                break;
            polynomial::detail::cy::Polynomial<Scalar, N> poly{};
            std::copy(coeffs.begin(), coeffs.end(), poly.coef);
            Scalar P = poly.Eval(root);
            // Assert
            CHECK_EQ(P / (2 * kMaxCoeff), doctest::Approx(0).epsilon(epsilon));
        }
    });
}