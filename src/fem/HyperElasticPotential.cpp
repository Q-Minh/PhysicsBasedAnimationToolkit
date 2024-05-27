#include "pba/fem/HyperElasticPotential.h"

#include "pba/common/ConstexprFor.h"
#include "pba/fem/Mesh.h"
#include "pba/fem/Tetrahedron.h"
#include "pba/math/LinearOperator.h"
#include "pba/physics/HyperElasticity.h"
#include "pba/physics/StableNeoHookeanEnergy.h"

#include <Eigen/Eigenvalues>
#include <doctest/doctest.h>

TEST_CASE("[fem] HyperElasticPotential")
{
    using namespace pba;
    // Cube tetrahedral mesh
    MatrixX V(3, 8);
    IndexMatrixX C(4, 5);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 0., 1., 1., 0., 0., 1., 1.,
            0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
            1, 2, 4, 7, 5,
            3, 0, 6, 5, 3,
            5, 6, 0, 3, 6;
    // clang-format on
    Scalar constexpr zero = 1e-8;
    Scalar constexpr Y    = 1e6;
    Scalar constexpr nu   = 0.45;
    common::ForValues<1, 2, 3>([&]<auto kOrder>() {
        auto constexpr kDims       = 3;
        using ElasticEnergyType    = physics::StableNeoHookeanEnergy<3>;
        using ElementType          = fem::Tetrahedron<kOrder>;
        using MeshType             = fem::Mesh<ElementType, kDims>;
        using ElasticPotentialType = fem::HyperElasticPotential<MeshType, ElasticEnergyType>;
        using QuadratureType       = ElasticPotentialType::QuadratureRuleType;

        CHECK(math::CLinearOperator<ElasticPotentialType>);

        // For order-k basis functions, our quadrature should be order k-1, otherwise 1
        if constexpr (kOrder == 1)
        {
            CHECK_EQ(QuadratureType::kOrder, 1);
        }
        else
        {
            CHECK_EQ(QuadratureType::kOrder, kOrder - 1);
        }
        MeshType const M(V, C);
        VectorX const x = M.X.reshaped();
        ElasticPotentialType U(M, x, Y, nu);
        CSCMatrix const H      = U.ToMatrix();
        CSCMatrix const HT     = H.transpose();
        Scalar const Esymmetry = (HT - H).squaredNorm() / H.squaredNorm();
        CHECK_LE(Esymmetry, zero);
        Eigen::SelfAdjointEigenSolver<CSCMatrix> eigs(H);
        // The hessian is generally rank-deficient, due to its invariance to translations and
        // rotations. The zero eigenvalues can manifest as numerically negative, but close to zero.
        Scalar const minEigenValue = eigs.eigenvalues().minCoeff();
        bool const bIsPositiveSemiDefinite =
            (eigs.info() == Eigen::ComputationInfo::Success) and (minEigenValue > -zero);
        CHECK(bIsPositiveSemiDefinite);

        // Elastic energy is invariant to translations
        Scalar constexpr t = 2.;
        U.ComputeElementElasticity(VectorX{x.array() + t});
        CSCMatrix const Htranslated = U.ToMatrix();
        Scalar const hessianTranslationInvarianceError =
            (Htranslated - H).squaredNorm() / H.squaredNorm();
        CHECK_LE(hessianTranslationInvarianceError, zero);

        // NOTE: Also invariant to rotations. We can likewise verify that the energy itself, and its
        // gradient are also invariant to translations and rotations, and are not invariant to
        // scaling, stretching, shearing, etc...

        // Check linearity of matrix-free operator
        Scalar constexpr k = -3.;
        VectorX y          = VectorX::Zero(x.size());
        VectorX yExpected  = k * H * x + H * x;
        U.Apply(k * x + x, y);
        Scalar const linearityError = (y - yExpected).norm() / yExpected.norm();
        CHECK_LE(linearityError, zero);
    });
}
