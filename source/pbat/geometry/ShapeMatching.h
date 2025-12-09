#ifndef PBAT_GEOMETRY_SHAPEMATCHING_H
#define PBAT_GEOMETRY_SHAPEMATCHING_H

#include "pbat/Aliases.h"
#include "pbat/fem/HyperElasticPotential.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/MeshQuadrature.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/math/optimization/Newton.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include <functional>

namespace pbat::geometry {

void MatchTargetShape(
    fem::Mesh<fem::Tetrahedron<3>, 3> const& mesh,
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& x,
    Eigen::Vector<Index, Eigen::Dynamic> const& b,
    std::function<void(
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic>&,
        Eigen::Matrix<Scalar, 3, Eigen::Dynamic>&)> const fProjectOntoTargetShape,
    Scalar K = 1e5)
{
    using ScalarType = Scalar;
    using EnergyType = physics::StableNeoHookeanEnergy</*kDims*/ 3>;
    math::optimization::Newton<ScalarType> newton{
        5,
        ScalarType(1e-4),
        x.size(),
        math::optimization::BackTrackingLineSearch<ScalarType>(
            20,
            ScalarType(0.5),
            ScalarType(1e-4),
            ScalarType(1),
            x.size())};
    // Allocate storage and optimization problem parameters
    auto const nConstraints = b.size();
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xref(3, nConstraints);
    Eigen::Vector<Scalar, Eigen::Dynamic> g;
    g.resize(3, nConstraints);
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index> gradg;
    gradg.resize(x.size(), 3);
    Eigen::Vector<Scalar, Eigen::Dynamic> Ug;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> gradUg;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> hessUg;
    Eigen::Vector<Scalar, Eigen::Dynamic> gradU;
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index> hessU;
    hessU.resize(x.size(), x.size());
    Eigen::Vector<Scalar, Eigen::Dynamic> gradE;
    gradE.resize(x.size());

    auto const wgU   = fem::MeshQuadratureWeights<1>(mesh);
    auto const GNegU = fem::ShapeFunctionGradients<1>(mesh);
    auto const egU =
        Eigen::Vector<Index, Eigen::Dynamic>::LinSpaced(mesh.E.cols(), Index(0), mesh.E.cols() - 1);
    auto constexpr Y        = 1e6;
    auto constexpr nu       = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    auto const mug          = Eigen::Vector<ScalarType, Eigen::Dynamic>::Constant(wgU.size(), mu);
    auto const lambdag = Eigen::Vector<ScalarType, Eigen::Dynamic>::Constant(wgU.size(), lambda);
    int const eElasticComputationFlags = fem::EElementElasticityComputationFlags::Potential |
                                         fem::EElementElasticityComputationFlags::Gradient |
                                         fem::EElementElasticityComputationFlags::Hessian;
    // Derivative precomputation
    auto const fPrepareDerivatives = [&](Eigen::Matrix<Scalar, 3, Eigen::Dynamic> const& xk) {
        // Elastic energy + derivatives
        fem::ToElementElasticity<EnergyType>(
            mesh,
            egU,
            wgU,
            GNegU,
            mug,
            lambdag,
            xk.reshaped(),
            Ug,
            gradUg,
            hessUg,
            eElasticComputationFlags,
            fem::EHyperElasticSpdCorrection::Absolute);
        // Penalty constraints + derivatives
        fProjectOntoTargetShape(x, xref);
        for (auto c = 0; c < g.cols(); ++c)
            g.col(c) = x.col(b(c)) - xref.col(c);
        gradg.setZero();
        gradg.reserve(Eigen::Vector<Index, 3>::Constant(nConstraints));
        for (auto j = 0; j < 3; ++j)
            for (auto i : b)
                gradg.insert(i * 3 + j, j) = Scalar(1);
        // Return total energy
        Scalar const U = fem::HyperElasticPotential(Ug);
        Scalar const C = Scalar(0.5) * K * g.squaredNorm();
        return U + C;
    };
    // Objective function
    auto const fObjective = [&](Eigen::Vector<Scalar, Eigen::Dynamic> const& xk) {
        // Elastic energy
        fem::ToElementElasticity<EnergyType>(
            mesh,
            egU,
            wgU,
            GNegU,
            mug,
            lambdag,
            xk.reshaped(),
            Ug,
            gradUg,
            hessUg,
            fem::EElementElasticityComputationFlags::Potential,
            fem::EHyperElasticSpdCorrection::None);
        // Penalty constraints
        fProjectOntoTargetShape(x, xref);
        for (auto c = 0; c < g.cols(); ++c)
            g.col(c) = xk.reshaped(3, x.cols()).col(b(c)) - xref.col(c);
        // Return total energy
        Scalar const U = fem::HyperElasticPotential(Ug);
        Scalar const C = Scalar(0.5) * K * g.squaredNorm();
        return U + C;
    };
    // Objective gradient
    auto const fGradient = [&]([[maybe_unused]] auto const& xk,
                               Eigen::Vector<ScalarType, Eigen::Dynamic>& gk) {
        gk.setZero();
        // Elastic energy gradient
        fem::ToHyperElasticGradient(mesh, egU, gradUg, gradU);
        // Constraint penalty gradient
        gk += gradU;
        gk += K * (gradg.transpose() * g.reshaped());
    };
    // Hessian inverse product function
    auto const fHessInv = [&]([[maybe_unused]] auto const& xk,
                              Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
                              Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
        // Assemble elastic Hessian
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index> hessian =
            fem::HyperElasticHessian<Eigen::ColMajor>(mesh, egU, hessUg);
        // Assemble constraint penalty Hessian
        hessian += K * (gradg * gradg.transpose());
        // Solve
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>> solver;
        solver.compute(hessian);
        dxk = solver.solve(gk);
    };
    // Optimize
    auto xk = x.reshaped();
    newton.Solve(fPrepareDerivatives, fObjective, fGradient, fHessInv, xk);
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_SHAPEMATCHING_H
