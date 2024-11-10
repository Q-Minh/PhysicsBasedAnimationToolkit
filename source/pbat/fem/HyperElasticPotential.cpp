#include "HyperElasticPotential.h"

#include "Jacobian.h"
#include "Mesh.h"
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
        using ElasticEnergyType    = physics::StableNeoHookeanEnergy<3>;
        using ElementType          = fem::Tetrahedron<kOrder>;
        using MeshType             = fem::Mesh<ElementType, kDims>;
        using ElasticPotentialType = fem::HyperElasticPotential<MeshType, ElasticEnergyType>;

        CHECK(math::CLinearOperator<ElasticPotentialType>);

        MeshType const M(V, C);
        VectorX const x       = M.X.reshaped();
        MatrixX const wg      = fem::InnerProductWeights<kQuadratureOrder>(M).reshaped();
        MatrixX const GNeg    = fem::ShapeFunctionGradients<kQuadratureOrder>(M);
        IndexVectorX const eg = IndexVectorX::LinSpaced(M.E.cols(), Index(0), M.E.cols() - 1)
                                    .replicate(1, wg.size() / M.E.cols())
                                    .transpose()
                                    .reshaped();
        ElasticPotentialType U(M, eg, wg, GNeg, x, Y, nu);
        Scalar const UMaterial      = U.Eval();
        VectorX const gradUMaterial = U.ToVector();
        CSCMatrix const HMaterial   = U.ToMatrix();
        CSCMatrix const HMaterialT  = HMaterial.transpose();
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
        U.ComputeElementElasticity(VectorX{x.array() + t});
        Scalar const UTranslated       = U.Eval();
        Scalar const UTranslationError = std::abs(UTranslated - UMaterial);
        CHECK_LE(UTranslationError, zero);
        VectorX const gradUTranslated      = U.ToVector();
        Scalar const gradUTranslationError = (gradUTranslated - gradUMaterial).squaredNorm();
        CHECK_LE(gradUTranslationError, zero);
        CSCMatrix const Htranslated = U.ToMatrix();
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
        U.Apply(k * x + x, y);
        Scalar const linearityError = (y - yExpected).norm() / yExpected.norm();
        CHECK_LE(linearityError, zero);
    });
}
