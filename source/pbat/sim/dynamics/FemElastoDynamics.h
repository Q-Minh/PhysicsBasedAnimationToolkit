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
#include "pbat/io/Concepts.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/integration/Bdf.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>

namespace pbat::sim::dynamics {

/**
 * @brief Finite Element Elasto-Dynamics initial value problem with Dirichlet boundary conditions.
 *
 * @tparam TElement Element type
 * @tparam Dims Dimensionality of the mesh
 * @tparam THyperElasticEnergy Hyper elastic energy type
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

    MatrixX x;    ///< `kDims x |# nodes|` matrix of nodal positions
    MatrixX v;    ///< `kDims x |# nodes|` matrix of nodal velocities
    MatrixX fext; ///< `kDims x |# nodes|` matrix of external forces at nodes
    VectorX m;    ///< `|# nodes| x 1` lumped mass matrix

    /**
     * @brief Quadrature for elastic potential
     */
    struct ElasticityQuadrature
    {
        IndexVectorX eg; ///< `|# quad.pts.| x 1` vector of element indices for quadrature points of
                         ///< elastic potential
        VectorX wg;      ///< `|# quad.pts.| x 1` vector of quadrature weights for elastic potential
        MatrixX GNeg;  ///< `|ElementType::kNodes| x |kDims * # quad.pts.|` matrix of shape function
                       ///< gradients at quadrature points
        MatrixX lameg; ///< `2 x |# quad.pts.|` matrix of Lame coefficients at quadrature points
    } QU;              ///< Quadrature for elastic potential
    std::unique_ptr<ElasticPotential> U; ///< Hyper elastic potential

    Index ndbc;       ///< Number of Dirichlet constrained nodes
    IndexVectorX dbc; ///< `|# nodes| x 1` concatenated vector of Dirichlet unconstrained and
                      ///< constrained nodes, partitioned as
                      ///< `[ dbc(0 : |#nodes|-ndbc), dbc(|# nodes|-ndbc : |# nodes|) ]`

    integration::Bdf bdf; ///< BDF time integration scheme

    /**
     * @brief Construct an empty FemElastoDynamics problem
     */
    FemElastoDynamics() = default;
    /**
     * @brief Construct an FemElastoDynamics problem on the mesh domain (V,C).
     *
     * All FemElastoDynamics quantities are initialized to sensible defaults, i.e. rest pose
     * positions, zero velocities, homogeneous rubber-like material properties, and gravitational
     * load.
     *
     * @param V `kDims x |# verts|` matrix of mesh vertex positions
     * @param C `|# cell nodes| x |# cells|` matrix of mesh cells
     */
    FemElastoDynamics(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);
    /**
     * @brief Construct an FemElastoDynamics problem on the mesh domain (V,C).
     *
     * All FemElastoDynamics quantities are initialized to sensible defaults, i.e. rest pose
     * positions, zero velocities, homogeneous rubber-like material properties, and gravitational
     * load.
     *
     * @param V `kDims x |# verts|` matrix of mesh vertex positions
     * @param C `|# cell nodes| x |# cells|` matrix of mesh cells
     */
    void Construct(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);
    /**
     * @brief Set the initial conditions for the initial value problem
     * @param x0 `kDims x |# nodes|` matrix of initial nodal coordinates
     * @param v0 `kDims x |# nodes|` matrix of initial nodal velocities
     */
    template <class TDerivedX0, class TDerivedV0>
    void SetInitialConditions(
        Eigen::DenseBase<TDerivedX0> const& x0,
        Eigen::DenseBase<TDerivedV0> const& v0);
    /**
     * @brief Compute, lump and set the mass matrix with homogeneous density \f$ \rho \f$.
     * @param rho Mass density of the material
     */
    void SetMassMatrix(Scalar rho);
    /**
     * @brief Set the elastic energy quadrature for a homogeneous material with Lame coefficients
     * \f$ \mu \f$ and
     * \lambda \f$.
     * @param mu 1st Lame coefficient \f$ \mu \f$
     * @param lambda 2nd Lame coefficient \f$ \lambda \f$
     * @param bWithElasticPotential Also construct the elastic potential `U`
     */
    void SetElasticEnergy(Scalar mu, Scalar lambda, bool bWithElasticPotential = true);
    /**
     * @brief Compute and set the external load vector given by fixed body forces \f$ b \f$.
     * @param b `kDims x 1` fixed body forces
     */
    void SetExternalLoad(Vector<kDims> const& b);
    /**
     * @brief Set the BDF (backward differentiation formula) time integration scheme
     * @param dt Time step size
     * @param s Step of the BDF scheme
     */
    void SetTimeIntegrationScheme(Scalar dt = Scalar(1e-2), int s = 1);
    /**
     * @brief Set the Dirichlet boundary conditions
     * @param D `|# nodes| x 1` mask of Dirichlet boundary conditions s.t. `D(i) == true` if
     * node `i` is constrained
     * @pre `D.size() == mesh.X.cols()`
     */
    template <typename TDerivedDirichletMask>
    void Constrain(Eigen::DenseBase<TDerivedDirichletMask> const& D);
    /**
     * @brief Compute, lump and set the mass matrix with variable density \f$ \rho(X) \f$
     * at quadrature points \f$ X_g \f$ of the given quadrature rule \f$ (w_g, X_g) \f$.
     *
     * @tparam TDerivedEg Eigen dense expression type for element indices
     * @tparam TDerivedWg Eigen dense expression type for quadrature weights
     * @tparam TDerivedXg Eigen dense expression type for quadrature points
     * @tparam TDerivedRhog Eigen dense expression type for mass density
     * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
     * @param wg `|# quad.pts.| x 1` vector of quadrature weights
     * @param Xg `|# dims| x |# quad.pts.|` matrix of quadrature points
     * @param rhog `|# quad.pts.| x 1` vector of mass density at quadrature points
     */
    template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedRhog>
    void SetMassMatrix(
        Eigen::DenseBase<TDerivedEg> const& eg,
        Eigen::DenseBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::DenseBase<TDerivedRhog> const& rhog);
    /**
     * @brief Compute and set the elastic energy quadrature for a heterogeneous material with
     * variable Lame coefficients \f$ \mu(X) \f$ and \f$ \lambda(X) \f$ at quadrature points \f$ X_g
     * \f$ of the given quadrature rule \f$ (w_g, X_g) \f$.
     *
     * @tparam TDerivedEg Eigen dense expression type for element indices
     * @tparam TDerivedWg Eigen dense expression type for quadrature weights
     * @tparam TDerivedXg Eigen dense expression type for quadrature points
     * @tparam TDerivedMug Eigen dense expression type for 1st Lame coefficients
     * @tparam TDerivedLambdag Eigen dense expression type for 2nd Lame coefficients
     * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
     * @param wg `|# quad.pts.| x 1` vector of quadrature weights
     * @param Xg `|# dims| x |# quad.pts.|` matrix of quadrature points
     * @param mug `|# quad.pts.| x 1` vector of 1st Lame coefficients at quadrature points
     * @param lambdag `|# quad.pts.| x 1` vector of 2nd Lame coefficients at quadrature points
     * @param bWithElasticPotential Also construct the elastic potential `U`
     */
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
    /**
     * @brief Compute and set the external load vector given by variable body forces \f$ b(X) \f$ at
     * quadrature points \f$ X_g \f$ of the given quadrature rule \f$ (w_g, X_g) \f$.
     *
     * @tparam TDerivedEg Eigen dense expression type for element indices
     * @tparam TDerivedWg Eigen dense expression type for quadrature weights
     * @tparam TDerivedXg Eigen dense expression type for quadrature points
     * @tparam TDerivedBg Eigen dense expression type for body forces
     * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
     * @param wg `|# quad.pts.| x 1` vector of quadrature weights
     * @param Xg `|# dims| x |# quad.pts.|` matrix of quadrature points
     * @param bg `kDims x |# quad.pts.|` matrix of body forces at quadrature points
     */
    template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedBg>
    void SetExternalLoad(
        Eigen::DenseBase<TDerivedEg> const& eg,
        Eigen::DenseBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::DenseBase<TDerivedBg> const& bg);
    /**
     * @brief Array of Dirichlet constrained nodes
     * @return `ndbc x 1` array of Dirichlet constrained nodes
     */
    auto DirichletNodes() const { return dbc.tail(ndbc); }
    /**
     * @brief Dirichlet nodal positions
     * @return `kDims x ndbc` matrix of Dirichlet constrained nodal positions
     */
    auto DirichletCoordinates() const { return x(Eigen::placeholders::all, DirichletNodes()); }
    /**
     * @brief Dirichlet nodal velocities
     * @return `kDims x ndbc` matrix of Dirichlet constrained nodal velocities
     */
    auto DirichletVelocities() const { return v(Eigen::placeholders::all, DirichletNodes()); }
    /**
     * @brief Dirichlet nodal positions
     * @return `kDims x ndbc` matrix of Dirichlet constrained nodal positions
     */
    auto DirichletCoordinates() { return x(Eigen::placeholders::all, DirichletNodes()); }
    /**
     * @brief Dirichlet nodal velocities
     * @return `kDims x ndbc` matrix of Dirichlet constrained nodal velocities
     */
    auto DirichletVelocities() { return v(Eigen::placeholders::all, DirichletNodes()); }
    /**
     * @brief Array of unconstrained nodes
     * @return `|# nodes - ndbc| x 1` array of unconstrained nodes
     */
    auto FreeNodes() const { return dbc.head(dbc.size() - ndbc); }
    /**
     * @brief Free nodal positions
     * @return `kDims x |# nodes - ndbc|` matrix of unconstrained nodal positions
     */
    auto FreeCoordinates() const { return x(Eigen::placeholders::all, FreeNodes()); }
    /**
     * @brief Free nodal velocities
     * @return `kDims x |# nodes - ndbc|` matrix of unconstrained nodal velocities
     */
    auto FreeVelocities() const { return v(Eigen::placeholders::all, FreeNodes()); }
    /**
     * @brief Free nodal positions
     * @return `kDims x |# nodes - ndbc|` matrix of unconstrained nodal positions
     */
    auto FreeCoordinates() { return x(Eigen::placeholders::all, FreeNodes()); }
    /**
     * @brief Free nodal velocities
     * @return `kDims x |# nodes - ndbc|` matrix of unconstrained nodal velocities
     */
    auto FreeVelocities() { return v(Eigen::placeholders::all, FreeNodes()); }
    /**
     * @brief Serialize to HDF5 group
     * @tparam TGroup Group type
     * @param parent Group to serialize to
     */
    template <io::CGroup TGroup>
    void Serialize(TGroup& parent) const;
    /**
     * @brief Deserialize from HDF5 group
     * @tparam TGroup Group type
     * @param parent Group to deserialize from
     */
    template <io::CGroup TGroup>
    void Deserialize(TGroup const& parent);
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
    // Mass
    Scalar constexpr rho{1e3};
    SetMassMatrix(rho);
    // Elasticity
    Scalar constexpr Y      = 1e6;
    Scalar constexpr nu     = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    SetElasticEnergy(mu, lambda, true /*bWithElasticPotential*/);
    // External load
    Vector<kDims> load    = Vector<kDims>::Zero();
    load(load.rows() - 1) = rho * Scalar(-9.81);
    SetExternalLoad(load);
    // Time integration scheme
    Scalar constexpr dt{1e-2};
    int constexpr bdfstep = 1;
    SetTimeIntegrationScheme(dt, bdfstep);
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedX0, class TDerivedV0>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetInitialConditions(
    Eigen::DenseBase<TDerivedX0> const& x0,
    Eigen::DenseBase<TDerivedV0> const& v0)
{
    x = x0;
    v = v0;
    bdf.SetOrder(2);
    bdf.SetInitialConditions(x0.reshaped(), v0.reshaped());
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetMassMatrix(Scalar rho)
{
    auto constexpr kOrder          = 2 * TElement::kOrder;
    auto const nElements           = mesh.E.cols();
    MatrixX const XgM              = mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementM = XgM.cols() / nElements;
    // Mass
    SetMassMatrix(
        IndexVectorX::LinSpaced(nElements, Index(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementM)
            .transpose()
            .reshaped() /*eg*/,
        fem::InnerProductWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgM /*Xg*/,
        VectorX::Constant(XgM.cols(), rho) /*rhog*/
    );
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetElasticEnergy(
    Scalar mu,
    Scalar lambda,
    bool bWithElasticPotential)
{
    auto constexpr kOrder = TElement::kOrder;
    auto const nElements  = mesh.E.cols();
    // Compute mesh quadrature points
    MatrixX const XgU              = mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementU = XgU.cols() / nElements;
    // Elasticity
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
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetExternalLoad(Vector<kDims> const& b)
{
    auto constexpr kOrder          = TElement::kOrder;
    auto const nElements           = mesh.E.cols();
    MatrixX const XgB              = mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementB = XgB.cols() / nElements;
    // External load
    SetExternalLoad(
        IndexVectorX::LinSpaced(nElements, Index(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementB)
            .transpose()
            .reshaped() /*eg*/,
        fem::InnerProductWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgB /*Xg*/,
        b.replicate(1, XgB.cols()) /*bg*/
    );
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetTimeIntegrationScheme(Scalar dt, int s)
{
    bdf.SetStep(s);
    bdf.SetTimeStep(dt);
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

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedRhog>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetMassMatrix(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::DenseBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedRhog> const& rhog)
{
    CSRMatrix N   = fem::ShapeFunctionMatrix(mesh, eg, Xg);
    CSRMatrix wgN = wg.asDiagonal() * N;
    CSCMatrix M   = N.transpose() * wgN;
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
    QU.eg   = eg;
    QU.wg   = wg;
    QU.GNeg = fem::ShapeFunctionGradientsAt(mesh, eg, Xg);
    QU.lameg.resize(2, eg.size());
    QU.lameg.row(0) = mug;
    QU.lameg.row(1) = lambdag;
    if (bWithElasticPotential)
    {
        U = std::make_unique<ElasticPotential>(mesh, QU.eg, QU.wg, QU.GNeg, QU.lameg);
        U->PrecomputeHessianSparsity();
    }
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedBg>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::SetExternalLoad(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::DenseBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedBg> const& bg)
{
    CSRMatrix N = fem::ShapeFunctionMatrix(mesh, eg, Xg);
    fext        = bg * wg.asDiagonal() * N;
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <io::CGroup TGroup>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::Serialize(TGroup& parent) const
{
    HighFive::Group group = parent.createGroup("pbat.sim.dynamics.FemElastoDynamics");
    auto meshGroup        = group.createGroup("mesh");
    meshGroup.createDataSet("X", mesh.X);
    meshGroup.createDataSet("E", mesh.E);
    group.createDataSet("x", x);
    group.createDataSet("v", v);
    group.createDataSet("fext", fext);
    group.createDataSet("m", m);
    auto elasticQuadratureGroup = group.createGroup("QU");
    elasticQuadratureGroup.createDataSet("eg", QU.eg);
    elasticQuadratureGroup.createDataSet("wg", QU.wg);
    elasticQuadratureGroup.createDataSet("GNeg", QU.GNeg);
    elasticQuadratureGroup.createDataSet("lameg", QU.lameg);
    if (U)
    {
        auto elasticPotentialGroup = group.createGroup("U");
        elasticPotentialGroup.createDataSet("Hg", U->Hg);
        elasticPotentialGroup.createDataSet("Gg", U->Gg);
        elasticPotentialGroup.createDataSet("Ug", U->Ug);
    }
    group.createAttribute("ndbc", ndbc);
    group.createDataSet("dbc", dbc);
    bdf.Serialize(group);
}

template <fem::CElement TElement, int Dims, physics::CHyperElasticEnergy THyperElasticEnergy>
template <io::CGroup TGroup>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy>::Deserialize(TGroup const& parent)
{
    HighFive::Group group       = parent.getGroup("pbat.sim.dynamics.FemElastoDynamics");
    auto meshGroup              = group.getGroup("mesh");
    mesh.X                      = meshGroup.getDataSet("X").read<MatrixX>();
    mesh.E                      = meshGroup.getDataSet("E").read<IndexMatrixX>();
    x                           = group.getDataSet("x").read<MatrixX>();
    v                           = group.getDataSet("v").read<MatrixX>();
    fext                        = group.getDataSet("fext").read<MatrixX>();
    m                           = group.getDataSet("m").read<VectorX>();
    auto elasticQuadratureGroup = group.getGroup("QU");
    QU.eg                       = elasticQuadratureGroup.getDataSet("eg").read<IndexVectorX>();
    QU.wg                       = elasticQuadratureGroup.getDataSet("wg").read<VectorX>();
    QU.GNeg                     = elasticQuadratureGroup.getDataSet("GNeg").read<MatrixX>();
    QU.lameg                    = elasticQuadratureGroup.getDataSet("lameg").read<MatrixX>();
    if (group.hasGroup("U"))
    {
        auto elasticPotentialGroup = group.getGroup("U");
        U     = std::make_unique<ElasticPotential>(mesh, QU.eg, QU.wg, QU.GNeg, QU.lameg);
        U->Hg = elasticPotentialGroup.getDataSet("Hg").read<CSCMatrix>();
        U->Gg = elasticPotentialGroup.getDataSet("Gg").read<MatrixX>();
        U->Ug = elasticPotentialGroup.getDataSet("Ug").read<VectorX>();
        U->PrecomputeHessianSparsity();
    }
    ndbc = group.getAttribute("ndbc").read<Index>();
    dbc  = group.getDataSet("dbc").read<IndexVectorX>();
    bdf.Deserialize(group);
}

} // namespace pbat::sim::dynamics

#endif // PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
