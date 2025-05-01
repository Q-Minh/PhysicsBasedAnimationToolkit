/**
 * @file FemElastoDynamics.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the Finite Element Elasto-Dynamics module.
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
#define PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H

#include "pbat/fem/HyperElasticPotential.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/integration/Bdf.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string_view>
#include <type_traits>
#include <unsupported/Eigen/SparseExtra>

namespace pbat::sim::dynamics {

/**
 * @brief
 *
 * @tparam TMesh
 * @tparam THyperElasticEnergy
 */
template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
struct FemElastoDynamics
{
    using MeshType          = fem::Mesh<TElement, Dims>; ///< Mesh type
    using ElementType       = TElement;                  ///< Element type
    using ElasticEnergyType = THyperElasticEnergy;       ///< Elastic energy type
    using ElasticPotential =
        fem::HyperElasticPotential<MeshType, ElasticEnergyType>; ///< Elastic potential
    static auto constexpr kDims = Dims;                          ///< Dimensionality of the mesh

    MeshType mesh; ///< FEM mesh

    MatrixX x; ///< `kDims x |# nodes|` matrix of nodal positions
    MatrixX v; ///< `kDims x |# nodes|` matrix of nodal velocities

    IndexVectorX
        egM; ///< `|# quad.pts.| x 1` vector of element indices for quadrature points of mass matrix
    VectorX wgM;  ///< `|# quad.pts.| x 1` vector of quadrature weights for mass matrix
    VectorX rhog; ///< `|# quad.pts.| x 1` vector of mass density at quadrature points
    VectorX m;    ///< `|# nodes| x 1` lumped mass matrix

    IndexVectorX egU; ///< `|# quad.pts.| x 1` vector of element indices for quadrature points of
                      ///< elastic potential
    VectorX wgU;      ///< `|# quad.pts.| x 1` vector of quadrature weights for elastic potential
    MatrixX GNegU;    ///< `|ElementType::kNodes| x |kDims * # quad.pts.|` matrix of shape function
                      ///< gradients at quadrature points
    MatrixX lamegU;   ///< `2 x |# quad.pts.|` matrix of Lame coefficients at quadrature points
    std::unique_ptr<ElasticPotential> U; ///< Hyper elastic potential

    IndexVectorX
        egB; ///< `|# quad.pts.| x 1` vector of element indices for quadrature points of body forces
    VectorX wgB;  ///< `|# quad.pts.| x 1` vector of quadrature weights for body forces
    MatrixX bg;   ///< `kDims x |# quad.pts.|` matrix of body forces at quadrature points
    MatrixX fext; ///< `kDims x |# nodes|` matrix of external forces at nodes

    Index ndbc;       ///< Number of Dirichlet constrained nodes
    IndexVectorX dbc; ///< `|# nodes| x 1` concatenated vector of Dirichlet unconstrained and
                      ///< constrained nodes, partitioned as
                      ///< `[ dbc(0 : |#nodes|-ndbc), dbc(|# nodes|-ndbc : |# nodes|) ]`

    integration::Bdf bdf; ///< BDF time integration scheme

    FemElastoDynamics() = default;
    FemElastoDynamics(std::string_view dir);
    FemElastoDynamics(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    void Construct(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    template <class TDerivedX0, class TDerivedV0>
    void SetInitialConditions(
        Eigen::DenseBase<TDerivedX0> const& x0,
        Eigen::DenseBase<TDerivedV0> const& v0);

    template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedRhog>
    void SetMassMatrix(
        Eigen::DenseBase<TDerivedEg> const& eg,
        Eigen::DenseBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::DenseBase<TDerivedRhog> const& rhog);

    template <
        class TDerivedEg,
        class TDerivedWg,
        class TDerivedXg,
        class TDerivedMug,
        class TDerivedLambdag>
    void SetElasticEnergy(
        Eigen::DenseBase<TDerivedEg> const& eg,
        Eigen::DenseBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::DenseBase<TDerivedMug> const& mug,
        Eigen::DenseBase<TDerivedLambdag> const& lambdag,
        bool bWithElasticPotential = true);

    template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedBg>
    void SetExternalLoad(
        Eigen::DenseBase<TDerivedEg> const& eg,
        Eigen::DenseBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::DenseBase<TDerivedBg> const& bg);

    template <typename TDerivedDirichletMask>
    void Constrain(Eigen::DenseBase<TDerivedDirichletMask> const& D);

    void SetTimeIntegrationScheme(Scalar dt = Scalar(1e-2), int s = 1);

    auto DirichletNodes() const { return dbc.tail(ndbc); }
    auto DirichletCoordinates() const { return x(Eigen::placeholders::all, DirichletNodes()); }
    auto DirichletVelocities() const { return v(Eigen::placeholders::all, DirichletNodes()); }
    auto DirichletCoordinates() { return x(Eigen::placeholders::all, DirichletNodes()); }
    auto DirichletVelocities() { return v(Eigen::placeholders::all, DirichletNodes()); }
    auto FreeNodes() const { return dbc.head(dbc.size() - ndbc); }
    auto FreeCoordinates() const { return x(Eigen::placeholders::all, FreeNodes()); }
    auto FreeVelocities() const { return v(Eigen::placeholders::all, FreeNodes()); }
    auto FreeCoordinates() { return x(Eigen::placeholders::all, FreeNodes()); }
    auto FreeVelocities() { return v(Eigen::placeholders::all, FreeNodes()); }
};

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::FemElastoDynamics(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    Construct(V, C);
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::Construct(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    mesh.Construct(V, C);
    auto const nElements = mesh.E.cols();
    auto const nNodes    = mesh.X.cols();
    SetInitialConditions(mesh.X, MatrixX::Zero(kDims, nNodes));
    auto constexpr kOrder = TElement::kOrder;
    // Compute mesh quadrature points
    MatrixX const XgM              = mesh.QuadraturePoints<2 * kOrder>();
    MatrixX const XgU              = mesh.QuadraturePoints<kOrder>();
    MatrixX const XgB              = mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementM = XgM.cols() / nElements;
    auto const nQuadPtsPerElementU = XgU.cols() / nElements;
    auto const nQuadPtsPerElementB = XgB.cols() / nElements;
    // Mass
    SetMassMatrix(
        IndexVectorX::LinSpaced(nElements, Index(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementM)
            .transpose()
            .reshaped() /*eg*/,
        fem::InnerProductWeights<2 * kOrder>(mesh).reshaped() /*wg*/,
        XgM /*Xg*/,
        VectorX::Constant(XgM.cols(), Scalar(1e3)) /*rhog*/
    );
    // Elasticity
    Scalar constexpr Y      = 1e6;
    Scalar constexpr nu     = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    SetElasticEnergy(
        IndexVectorX::LinSpaced(nElements, Index(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementU)
            .transpose()
            .reshaped() /*eg*/,
        fem::InnerProductWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgU /*Xg*/,
        VectorX::Constant(XgU.cols(), mu) /*mug*/,
        VectorX::Constant(XgU.cols(), lambda) /*lambdag*/,
        true /*bWithElasticPotential*/
    );
    // External load
    Vector<kDims> a = Vector<kDims>::Zero();
    a(a.rows() - 1) = Scalar(-9.81);
    auto M          = m.replicate(1, kDims).transpose().reshaped().asDiagonal();
    auto ag         = a.replicate(1, egB.size());
    SetExternalLoad(
        IndexVectorX::LinSpaced(nElements, Index(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementB)
            .transpose()
            .reshaped() /*eg*/,
        fem::InnerProductWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgB /*Xg*/,
        ag * M /*bg*/
    );
    // Time integration scheme
    SetTimeIntegrationScheme(Scalar(1e-2), 1);
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetTimeIntegrationScheme(Scalar dt, int s)
{
    bdf.SetStep(s);
    if (bdf.Order() != 2)
        bdf.SetOrder(2);
    bdf.SetTimeStep(dt);
    if (x.size() > 0 and v.size() > 0)
        bdf.SetInitialConditions(x, v);
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedX0, class TDerivedV0>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetInitialConditions(
    Eigen::DenseBase<TDerivedX0> const& x0,
    Eigen::DenseBase<TDerivedV0> const& v0)
{
    x = x0;
    v = v0;
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedRhog>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetMassMatrix(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::DenseBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedRhog> const& rhog)
{
    egM         = eg;
    wgM         = wg;
    rhog        = rhog;
    CSRMatrix N = fem::ShapeFunctionMatrix(mesh, egM, Xg);
    CSCMatrix M = N.transpose() * wgM.asDiagonal() * N;
    m.resize(M.cols());
    for (auto j = 0; j < M.cols(); ++j)
        m(j) = M.col(j).sum();
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <
    class TDerivedEg,
    class TDerivedWg,
    class TDerivedXg,
    class TDerivedMug,
    class TDerivedLambdag>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetElasticEnergy(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::DenseBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedMug> const& mug,
    Eigen::DenseBase<TDerivedLambdag> const& lambdag,
    bool bWithElasticPotential)
{
    egU   = eg;
    wgU   = wg;
    GNegU = fem::ShapeFunctionGradientsAt(mesh, egU, Xg);
    lamegU.resize(2, egU.size());
    lamegU.row(0) = mug;
    lamegU.row(1) = lambdag;
    if (bWithElasticPotential)
        U = std::make_unique<ElasticPotential>(mesh, egU, wgU, GNegU, lamegU);
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedBg>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetExternalLoad(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::DenseBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedBg> const& bg)
{
    egB         = eg;
    wgB         = wg;
    bg          = bg;
    CSRMatrix N = fem::ShapeFunctionMatrix(mesh, egB, Xg);
    fext        = bg * wgB.asDiagonal() * N;
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <typename TDerivedDirichletMask>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::Constrain(
    Eigen::DenseBase<TDerivedDirichletMask> const& D)
{
    static_assert(
        std::is_same_v<typename TDerivedDirichletMask::Scalar, bool>,
        "Dirichlet mask must be of type bool");
    auto const nNodes = mesh.X.cols();
    assert(D.size() == nNodes);
    dbc.setLinSpaced(nNodes, Index(0), nNodes - 1);
    auto it = std::stable_partition(dbc.begin(), dbc.end(), [&D](Index i) { return not D(i); });
    ndbc    = std::distance(dbc.begin(), it);
}

} // namespace pbat::sim::dynamics

#endif // PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
