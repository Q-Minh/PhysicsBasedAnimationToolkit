#include "ChebyshevIntegrator.h"

#include "Kernels.h"

#include <tbb/parallel_for.h>

namespace pbat::sim::vbd {

ChebyshevIntegrator::ChebyshevIntegrator(Data dataIn)
    : Integrator(std::move(dataIn)), xkm1(data.x), xkm2(data.x)
{
}

void ChebyshevIntegrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    Scalar rho2 = data.rho * data.rho;
    Scalar omega{};
    for (auto k = 0; k < iterations; ++k)
    {
        // Vanilla VBD iteration
        RunVbdIteration(sdt, sdt2);
        // Chebyshev Update
        omega    = kernels::ChebyshevOmega(k, rho2, omega);
        auto& xk = data.x;
        if (k > 1)
            xk = omega * (xk - xkm2) + xkm2;
        xkm2 = xkm1;
        xkm1 = xk;
    }
}

} // namespace pbat::sim::vbd

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd] ChebyshevIntegrator")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    MatrixX P(3, 8);
    IndexMatrixX V(1, 8);
    IndexMatrixX T(4, 5);
    IndexMatrixX F(3, 12);
    // clang-format off
    P << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on
    V.reshaped().setLinSpaced(0, static_cast<Index>(P.cols() - 1));
    // Problem parameters
    auto constexpr dt         = Scalar{1e-2};
    auto constexpr substeps   = 1;
    auto constexpr iterations = 10;
    auto constexpr rho        = Scalar{0.9};

    // Act
    using pbat::common::ToEigen;
    using pbat::sim::vbd::ChebyshevIntegrator;
    ChebyshevIntegrator cvbd{sim::vbd::Data()
                                 .WithVolumeMesh(P, T)
                                 .WithSurfaceMesh(V, F)
                                 .WithChebyshevAcceleration(rho)
                                 .Construct()};
    cvbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = cvbd.data.x - P;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}