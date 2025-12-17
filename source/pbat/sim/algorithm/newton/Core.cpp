#include "Core.h"

namespace pbat::sim::algorithm::newton {

Params& Params::WithOptimizer(math::optimization::Newton<Scalar> optimizer)
{
    this->newton = std::move(optimizer);
    return *this;
}

Params& Params::WithSpdCorrection(fem::EHyperElasticSpdCorrection _eSpdCorrection)
{
    this->eSpdCorrection = _eSpdCorrection;
    return *this;
}

Params& Params::WithLinearSolver(ELinearSolver _eLinearSolver, Eigen::Index maxIters, Scalar tol)
{
    this->eLinearSolver = _eLinearSolver;
    switch (eLinearSolver)
    {
        case ELinearSolver::LLT: {
            this->Hinv.emplace<DecompositionType>();
            break;
        }
        case ELinearSolver::PCGJacobi: {
            this->Hinv.template emplace<Eigen::ConjugateGradient<
                decltype(hessian),
                Eigen::Lower | Eigen::Upper,
                Eigen::DiagonalPreconditioner<Scalar>>>();
            break;
        }
        case ELinearSolver::PCGIC: {
            this->Hinv.template emplace<Eigen::ConjugateGradient<
                decltype(hessian),
                Eigen::Lower | Eigen::Upper,
                IncompleteCholeskyType>>();
            break;
        }
        case ELinearSolver::PCGILUT: {
            this->Hinv.template emplace<Eigen::ConjugateGradient<
                decltype(hessian),
                Eigen::Lower | Eigen::Upper,
                IncompleteLUTType>>();
            break;
        }
        case ELinearSolver::PCGLaplacian: {
            // this->Hinv.template emplace<
            //     Eigen::ConjugateGradient<
            //         decltype(hessian),
            //         Eigen::Lower | Eigen::Upper,
            //         fem::LaplacianPreconditioner<Scalar>>>();
            throw std::invalid_argument("PCGLaplacian solver not yet implemented");
            break;
        }
        default: throw std::invalid_argument("Unknown linear solver type");
    }
    std::visit(
        [maxIters, tol](auto& solver) {
            if constexpr (not std::is_same_v<std::decay_t<decltype(solver)>, DecompositionType>)
            {
                solver.setMaxIterations(maxIters);
                solver.setTolerance(tol);
            }
        },
        this->Hinv);
    return *this;
}

Params& Params::Construct(bool bValidate)
{
    return *this;
}

void Params::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.newton.Params"];
    newton.Serialize(group);
    group.WriteMetaData("eSpdCorrection", static_cast<int>(eSpdCorrection));
    group.WriteMetaData("eLinearSolver", static_cast<int>(eLinearSolver));
}

void Params::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.newton.Params"];
    newton.Deserialize(group);
    eSpdCorrection =
        static_cast<fem::EHyperElasticSpdCorrection>(group.ReadMetaData<int>("eSpdCorrection"));
    eLinearSolver = static_cast<ELinearSolver>(group.ReadMetaData<int>("eLinearSolver"));
    this->WithLinearSolver(eLinearSolver);
}

} // namespace pbat::sim::algorithm::newton

#include "pbat/geometry/Device.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/sim/contact/MultiMesh.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][algorithm][newton] Core")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
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
    // Problem parameters
    using namespace pbat::sim::algorithm;
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = newton::FemElastoDynamics<ElasticEnergyType>;
    using MeshDynamics      = sim::algorithm::newton::MeshDynamics;
    FemElastoDynamics dynamics{};
    dynamics.Construct(V, C);
    MeshDynamics contact{};
    sim::contact::MultiMesh<Index> dynamicMesh{};
    dynamicMesh.ConstructFromTetrahedralMesh(
        dynamics.mesh.E,
        Eigen::Vector<Index, Eigen::Dynamic>::Zero(V.cols()),
        1);
    contact.SetDynamicGeometry(V, std::move(dynamicMesh));
    contact.Initialize(geometry::Device{});
    // Act
    dynamics.SetInitialConditions(dynamics.x, dynamics.v);
    newton::Params params{};
    params
        .WithOptimizer(
            math::optimization::Newton<Scalar>(
                /*nMaxIters=*/20,
                /*gtol=*/Scalar{1e-6},
                /*n=*/dynamics.x.size(),
                /*lineSearchIn=*/math::optimization::BackTrackingLineSearch<Scalar>{}))
        .Construct();
    dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = dynamics.Objective(dynamics.x);
    VectorX g0 = dynamics.Gradient(dynamics.x);
    newton::Solve(dynamics, contact, params);
    // Assert
    auto constexpr zero = Scalar{1e-4};
    auto xt    = dynamics.bdf.CurrentState(0).reshaped(dynamics.x.rows(), dynamics.x.cols());
    MatrixX dx = dynamics.x - xt;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    dynamics.ComputeElasticEnergy(
        dynamics.x,
        fem::EElementElasticityComputationFlags::Potential |
            fem::EElementElasticityComputationFlags::Gradient,
        fem::EHyperElasticSpdCorrection::None);
    Scalar f = dynamics.Objective(dynamics.x);
    CHECK_LT(f, f0);
    VectorX g     = dynamics.Gradient(dynamics.x);
    Scalar g0norm = g0.norm();
    Scalar gnorm  = g.norm();
    CHECK_LT(gnorm, g0norm);
}