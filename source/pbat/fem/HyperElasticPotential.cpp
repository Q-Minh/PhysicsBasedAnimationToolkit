#include "HyperElasticPotential.h"

#include "Mesh.h"
#include "MeshQuadrature.h"
#include "ShapeFunctions.h"
#include "Tetrahedron.h"

#include <Eigen/Eigenvalues>
#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/LinearOperator.h>
#include <pbat/physics/HyperElasticity.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>

TEST_CASE("[fem] HyperElasticPotential")
{
    using namespace pbat;
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
        auto constexpr kDims            = 3;
        auto constexpr kQuadratureOrder = [&]() {
            if constexpr (kOrder == 1)
                return 1;
            else
                return kOrder - 1;
        }();
        using ElasticEnergyType = physics::StableNeoHookeanEnergy<3>;
        using ElementType       = fem::Tetrahedron<kOrder>;
        using MeshType          = fem::Mesh<ElementType, kDims>;

        MeshType const M(V, C);
        VectorX const x           = M.X.reshaped();
        auto const wg             = fem::MeshQuadratureWeights<kQuadratureOrder>(M);
        auto const GNeg           = fem::ShapeFunctionGradients<kQuadratureOrder>(M);
        auto const eg             = fem::MeshQuadratureElements(M.E, wg);
        auto const [mug, lambdag] = physics::LameCoefficients(
            VectorX::Constant(wg.size(), Y),
            VectorX::Constant(wg.size(), nu));
        VectorX Ug;
        MatrixX Hg, Gg;
        fem::ToElementElasticity<ElasticEnergyType>(
            M,
            eg.reshaped(),
            wg.reshaped(),
            GNeg,
            mug,
            lambdag,
            x,
            Ug,
            Gg,
            Hg,
            fem::EElementElasticityComputationFlags::Potential |
                fem::EElementElasticityComputationFlags::Gradient |
                fem::EElementElasticityComputationFlags::Hessian,
            fem::EHyperElasticSpdCorrection::None);
        Scalar const UMaterial      = fem::HyperElasticPotential(Ug);
        VectorX const gradUMaterial = fem::HyperElasticGradient(M, eg.reshaped(), Gg);
        CSCMatrix const HMaterial = fem::HyperElasticHessian<Eigen::ColMajor>(M, eg.reshaped(), Hg);
        CSCMatrix const HMaterialT = HMaterial.transpose();
        Scalar const Esymmetry = (HMaterialT - HMaterial).squaredNorm() / HMaterial.squaredNorm();
        CHECK_LE(Esymmetry, zero);
        Eigen::SelfAdjointEigenSolver<CSCMatrix> eigs(HMaterial);
        // The hessian is generally rank-deficient, due to its invariance to translations and
        // rotations. The zero eigenvalues can manifest as numerically negative, but close to zero.
        Scalar const minEigenValue = eigs.eigenvalues().minCoeff();
        bool const bIsPositiveSemiDefinite =
            (eigs.info() == Eigen::ComputationInfo::Success) and (minEigenValue > -zero);
        CHECK(bIsPositiveSemiDefinite);

        // Elastic energy is invariant to translations
        Scalar constexpr t = 2.;
        fem::ToElementElasticity<ElasticEnergyType>(
            M,
            eg.reshaped(),
            wg.reshaped(),
            GNeg,
            mug,
            lambdag,
            (x.array() + t).matrix(),
            Ug,
            Gg,
            Hg,
            fem::EElementElasticityComputationFlags::Potential |
                fem::EElementElasticityComputationFlags::Gradient |
                fem::EElementElasticityComputationFlags::Hessian,
            fem::EHyperElasticSpdCorrection::None);
        Scalar const UTranslated       = fem::HyperElasticPotential(Ug);
        Scalar const UTranslationError = std::abs(UTranslated - UMaterial);
        CHECK_LE(UTranslationError, zero);
        VectorX const gradUTranslated      = fem::HyperElasticGradient(M, eg.reshaped(), Gg);
        Scalar const gradUTranslationError = (gradUTranslated - gradUMaterial).squaredNorm();
        CHECK_LE(gradUTranslationError, zero);
        CSCMatrix const Htranslated =
            fem::HyperElasticHessian<Eigen::ColMajor>(M, eg.reshaped(), Hg);
        Scalar const hessianTranslationInvarianceError =
            (Htranslated - HMaterial).squaredNorm() / HMaterial.squaredNorm();
        CHECK_LE(hessianTranslationInvarianceError, zero);

        // NOTE: Also invariant to rotations. We can likewise verify that the energy itself, and its
        // gradient are also invariant to translations and rotations, and are not invariant to
        // scaling, stretching, shearing, etc...

        // Check linearity of matrix-free operator
        Scalar constexpr k = -3.;
        VectorX y          = VectorX::Zero(x.size());
        VectorX yExpected  = k * HMaterial * x + HMaterial * x;
        fem::GemmHyperElastic(M, eg.reshaped(), Hg, k * x + x, y);
        Scalar const linearityError = (y - yExpected).norm() / yExpected.norm();
        CHECK_LE(linearityError, zero);
    });
}
