/**
 * @file FemElastoDynamics.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the Finite Element Elasto-Dynamics module.
 * @date 2025-04-30
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
#define PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H

#include "pbat/common/Concepts.h"
#include "pbat/fem/HyperElasticPotential.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/MeshQuadrature.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/io/Archive.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/integration/Bdf.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <type_traits>

namespace pbat::sim::dynamics {

/**
 * @brief Finite Element Elasto-Dynamics initial value problem with Dirichlet boundary conditions
 * using BDF (backward differentiation formula) as the time discretization.
 *
 * Represents the problem
 * \f[
 * \min_x \frac{1}{2} || x - \tilde{x} ||_M^2 + \tilde{\beta}_\text{bdf-s}^2 U(x) ,
 * \f]
 * where \f$ M \f$ is the mass matrix, \f$ \tilde{x} \f$ is the BDF inertial target,
 * \f$ \tilde{\beta}_\text{bdf-s} \f$ is the forcing term's coefficient, and \f$ U(x) \f$ is the
 * hyper-elastic potential.
 *
 * @tparam TElement Element type
 * @tparam Dims Dimensionality of the mesh
 * @tparam THyperElasticEnergy Hyper elastic energy type
 */
template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar = Scalar,
    common::CIndex TIndex          = Index>
struct FemElastoDynamics
{
    using MeshType              = fem::Mesh<TElement, Dims, TScalar, TIndex>; ///< Mesh type
    using ElementType           = TElement;                                   ///< Element type
    using ElasticEnergyType     = THyperElasticEnergy;               ///< Elastic energy type
    using BdfType               = integration::Bdf<TScalar, TIndex>; ///< BDF time integrator type
    using ScalarType            = TScalar;                           ///< Floating point scalar type
    using IndexType             = TIndex;                            ///< Integer index type
    static auto constexpr kDims = Dims;                              ///< Dimensionality of the mesh

    MeshType mesh; ///< FEM mesh
    BdfType bdf;   ///< BDF time integration scheme

    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic>
        fext; ///< `kDims x |# nodes|` matrix of external forces at nodes
    Eigen::Vector<ScalarType, Eigen::Dynamic> m; ///< `|# nodes| x 1` lumped mass matrix
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic>
        xtilde;                                         ///< `kDims x |# nodes|` inertial targets
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> x; ///< `kDims x |# nodes|` positions
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> v; ///< `kDims x |# nodes|` velocities

    Eigen::Vector<IndexType, Eigen::Dynamic> egU; ///< `|# quad.pts.| x 1` vector of element indices
                                                  ///< for quadrature points of elastic potential
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        wgU; ///< `|# quad.pts.| x 1` vector of quadrature weights for elastic potential
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>
        GNegU; ///< `|ElementType::kNodes| x |kDims * # quad.pts.|` matrix of shape function
               ///< gradients at quadrature points
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>
        lamegU; ///< `2 x |# quad.pts.|` matrix of Lame coefficients at quadrature points
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        UgU; ///< `|# quad.pts.| x 1` vector of elastic energy density at quadrature points
    Eigen::Matrix<ScalarType, kDims * ElementType::kNodes, Eigen::Dynamic>
        GgU; ///< `|# dims * # elem nodes| x |# quad.pts.|` matrix of element elastic gradient
             ///< vectors at quadrature points
    Eigen::Matrix<ScalarType, kDims * ElementType::kNodes, Eigen::Dynamic>
        HgU; ///< `|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.|`
             ///< matrix of element elastic hessian matrices at quadrature points

    IndexType ndbc; ///< Number of Dirichlet constrained nodes
    Eigen::Vector<IndexType, Eigen::Dynamic>
        dbc; ///< `|# nodes| x 1` concatenated vector of Dirichlet unconstrained and
             ///< constrained nodes, partitioned as
             ///< `[ dbc(0 : |#nodes|-ndbc), dbc(|# nodes|-ndbc : |# nodes|) ]`
    Eigen::Vector<bool, Eigen::Dynamic> dmask; ///< `|# nodes| x 1` mask of Dirichlet
                                               ///< boundary conditions s.t. `dmask(i) == true`
                                               ///< if node `i` is constrained

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
    FemElastoDynamics(
        Eigen::Ref<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> const& C);
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
    void Construct(
        Eigen::Ref<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> const& C);
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
    void SetMassMatrix(ScalarType rho);
    /**
     * @brief Set the elastic energy quadrature for a homogeneous material with Lame coefficients
     * \f$ \mu \f$ and
     * \lambda \f$.
     * @param mu 1st Lame coefficient \f$ \mu \f$
     * @param lambda 2nd Lame coefficient \f$ \lambda \f$
     */
    void SetElasticEnergy(ScalarType mu, ScalarType lambda);
    /**
     * @brief Compute and set the external load vector given by fixed body forces \f$ b \f$.
     * @param b `kDims x 1` fixed body forces
     */
    void SetExternalLoad(Eigen::Vector<ScalarType, kDims> const& b);
    /**
     * @brief Set the BDF (backward differentiation formula) time integration scheme
     * @param dt Time step size
     * @param s Step of the BDF scheme
     */
    void SetTimeIntegrationScheme(ScalarType dt = ScalarType(1e-2), int s = 1);
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
        Eigen::MatrixBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::MatrixBase<TDerivedRhog> const& rhog);
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
        Eigen::DenseBase<TDerivedLambdag> const& lambdag);
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
        Eigen::MatrixBase<TDerivedWg> const& wg,
        Eigen::MatrixBase<TDerivedXg> const& Xg,
        Eigen::MatrixBase<TDerivedBg> const& bg);
    /**
     * @brief Set the BDF inertial target for elasto dynamics
     */
    void SetupTimeIntegrationOptimization();
    /**
     * @brief Compute the quadrature point elastic energies of the current configuration into Ug,
     * Gg, Hg
     * @param eElasticComputationFlags Flags for computing elastic potential, gradient, and/or
     * hessian
     * @param eSpdCorrectionFlags Flags for SPD correction of element hessians
     */
    void ComputeElasticEnergy(
        int eElasticComputationFlags,
        fem::EHyperElasticSpdCorrection eSpdCorrectionFlags);
    /**
     * @brief k-dimensional mass matrix
     * @return `kDims * |#nodes| x 1` vector of the `kDims`-dimensional lumped mass matrix diagonal
     * coefficients
     */
    auto M() const { return m.replicate(1, kDims).transpose().reshaped(); };
    /**
     * @brief External acceleration
     * @return `kDims x |# nodes|` external acceleration
     */
    auto aext() const { return (fext * m.cwiseInverse().asDiagonal()); }
    /**
     * @brief Check if a node is Dirichlet constrained
     * @param node Node index
     * @return `true` if the node is Dirichlet constrained, `false` otherwise
     */
    bool IsDirichletNode(IndexType node) const { return dmask(node); }
    /**
     * @brief Check if a coordinate is Dirichlet constrained
     * @param i Coordinate index
     * @return `true` if the coordinate is Dirichlet constrained, `false` otherwise
     */
    bool IsDirichletDof(IndexType i) const { return dmask(i / kDims); }
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
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize from HDF5 group
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive const& archive);
};

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::FemElastoDynamics(
    Eigen::Ref<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> const& C)
{
    Construct(V, C);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::Construct(
    Eigen::Ref<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> const& C)
{
    mesh.Construct(V, C);
    auto const nNodes = mesh.X.cols();
    SetInitialConditions(
        mesh.X,
        Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>::Zero(kDims, nNodes));
    // Mass
    ScalarType constexpr rho{1e3};
    SetMassMatrix(rho);
    // Elasticity
    ScalarType constexpr Y  = 1e6;
    ScalarType constexpr nu = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    SetElasticEnergy(mu, lambda);
    // External load
    Eigen::Vector<ScalarType, kDims> load = Eigen::Vector<ScalarType, kDims>::Zero();
    load(load.rows() - 1)                 = rho * ScalarType(-9.81);
    SetExternalLoad(load);
    // Time integration scheme
    ScalarType constexpr dt{1e-2};
    int constexpr bdfstep = 1;
    SetTimeIntegrationScheme(dt, bdfstep);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
template <class TDerivedX0, class TDerivedV0>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetInitialConditions(
    Eigen::DenseBase<TDerivedX0> const& x0,
    Eigen::DenseBase<TDerivedV0> const& v0)
{
    x = x0.reshaped(kDims, x0.size() / kDims);
    v = v0.reshaped(kDims, v0.size() / kDims);
    bdf.SetOrder(2);
    bdf.SetInitialConditions(x0.reshaped(), v0.reshaped());
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetMassMatrix(
    ScalarType rho)
{
    auto constexpr kOrder     = 2 * TElement::kOrder;
    IndexType const nElements = static_cast<IndexType>(mesh.E.cols());
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const XgM =
        mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementM = XgM.cols() / nElements;
    // Mass
    SetMassMatrix(
        Eigen::Vector<IndexType, Eigen::Dynamic>::LinSpaced(nElements, IndexType(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementM)
            .transpose()
            .reshaped() /*eg*/,
        fem::MeshQuadratureWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgM /*Xg*/,
        Eigen::Vector<ScalarType, Eigen::Dynamic>::Constant(XgM.cols(), rho) /*rhog*/
    );
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetElasticEnergy(
    ScalarType mu,
    ScalarType lambda)
{
    auto constexpr kOrder     = TElement::kOrder;
    IndexType const nElements = static_cast<IndexType>(mesh.E.cols());
    // Compute mesh quadrature points
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const XgU =
        mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementU = XgU.cols() / nElements;
    // Elasticity
    SetElasticEnergy(
        Eigen::Vector<IndexType, Eigen::Dynamic>::LinSpaced(nElements, IndexType(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementU)
            .transpose()
            .reshaped() /*eg*/,
        fem::MeshQuadratureWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgU /*Xg*/,
        Eigen::Vector<ScalarType, Eigen::Dynamic>::Constant(XgU.cols(), mu) /*mug*/,
        Eigen::Vector<ScalarType, Eigen::Dynamic>::Constant(XgU.cols(), lambda) /*lambdag*/);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetExternalLoad(
    Eigen::Vector<ScalarType, kDims> const& b)
{
    auto constexpr kOrder     = TElement::kOrder;
    IndexType const nElements = static_cast<IndexType>(mesh.E.cols());
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const XgB =
        mesh.QuadraturePoints<kOrder>();
    auto const nQuadPtsPerElementB = XgB.cols() / nElements;
    // External load
    SetExternalLoad(
        Eigen::Vector<IndexType, Eigen::Dynamic>::LinSpaced(nElements, IndexType(0), nElements - 1)
            .replicate(1, nQuadPtsPerElementB)
            .transpose()
            .reshaped() /*eg*/,
        fem::MeshQuadratureWeights<kOrder>(mesh).reshaped() /*wg*/,
        XgB /*Xg*/,
        b.replicate(1, XgB.cols()) /*bg*/
    );
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetTimeIntegrationScheme(
    ScalarType dt,
    int s)
{
    bdf.SetStep(s);
    bdf.SetTimeStep(dt);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
template <typename TDerivedDirichletMask>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::Constrain(
    Eigen::DenseBase<TDerivedDirichletMask> const& D)
{
    static_assert(
        std::is_same_v<typename TDerivedDirichletMask::Scalar, bool>,
        "Dirichlet mask must be of type bool");
    IndexType const nNodes = static_cast<IndexType>(mesh.X.cols());
    assert(D.size() == nNodes);
    dmask = D.cast<bool>();
    dbc.setLinSpaced(nNodes, IndexType(0), nNodes - 1);
    auto it = std::stable_partition(dbc.begin(), dbc.end(), [&D](IndexType i) { return not D[i]; });
    ndbc    = nNodes - std::distance(dbc.begin(), it);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedRhog>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetMassMatrix(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::MatrixBase<TDerivedRhog> const& rhog)
{
    auto N = fem::ShapeFunctionMatrixAt(mesh, eg.derived(), Xg.derived());
    Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndexType> rhogwgN =
        rhog.cwiseProduct(wg).asDiagonal() * N;
    Eigen::SparseMatrix<ScalarType, Eigen::ColMajor, IndexType> M = N.transpose() * rhogwgN;
    m.resize(M.cols());
    for (auto j = 0; j < M.cols(); ++j)
        m(j) = M.col(j).sum();
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
template <
    class TDerivedEg,
    class TDerivedWg,
    class TDerivedXg,
    class TDerivedMug,
    class TDerivedLambdag>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetElasticEnergy(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::DenseBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::DenseBase<TDerivedMug> const& mug,
    Eigen::DenseBase<TDerivedLambdag> const& lambdag)
{
    egU   = eg;
    wgU   = wg;
    GNegU = fem::ShapeFunctionGradientsAt(mesh, eg, Xg);
    lamegU.resize(2, eg.size());
    lamegU.row(0) = mug;
    lamegU.row(1) = lambdag;
    UgU.resize(eg.size());
    GgU.resize(kDims * ElementType::kNodes, eg.size());
    HgU.resize(kDims * ElementType::kNodes, kDims * ElementType::kNodes * eg.size());
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
template <class TDerivedEg, class TDerivedWg, class TDerivedXg, class TDerivedBg>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::SetExternalLoad(
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedWg> const& wg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    Eigen::MatrixBase<TDerivedBg> const& bg)
{
    Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndexType> N =
        fem::ShapeFunctionMatrixAt(mesh, eg.derived(), Xg.derived());
    fext = bg * wg.asDiagonal() * N;
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::
    SetupTimeIntegrationOptimization()
{
    auto xtildeBdf = bdf.Inertia(0);
    auto vtildeBdf = bdf.Inertia(1);
    auto betaTilde = bdf.BetaTilde();
    xtilde.resize(kDims, xtildeBdf.size() / kDims);
    xtilde.reshaped() =
        -(xtildeBdf + betaTilde * vtildeBdf) + (betaTilde * betaTilde) * (aext().reshaped());
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void
FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::ComputeElasticEnergy(
    int eElasticComputationFlags,
    fem::EHyperElasticSpdCorrection eSpdCorrectionFlags)
{
    fem::ToElementElasticity<ElasticEnergyType>(
        mesh,
        egU,
        wgU,
        GNegU,
        lamegU.row(0),
        lamegU.row(1),
        x.reshaped(),
        UgU,
        GgU,
        HgU,
        eElasticComputationFlags,
        eSpdCorrectionFlags);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::Serialize(
    io::Archive& archive) const
{
    io::Archive femElastoDynamicsArchive = archive["pbat.sim.dynamics.FemElastoDynamics"];
    mesh.Serialize(femElastoDynamicsArchive);
    bdf.Serialize(femElastoDynamicsArchive);
    femElastoDynamicsArchive.WriteData("fext", fext);
    femElastoDynamicsArchive.WriteData("m", m);
    femElastoDynamicsArchive.WriteData("xtilde", xtilde);
    femElastoDynamicsArchive.WriteData("x", x);
    femElastoDynamicsArchive.WriteData("v", v);
    femElastoDynamicsArchive.WriteData("egU", egU);
    femElastoDynamicsArchive.WriteData("wgU", wgU);
    femElastoDynamicsArchive.WriteData("GNegU", GNegU);
    femElastoDynamicsArchive.WriteData("lamegU", lamegU);
    femElastoDynamicsArchive.WriteData("UgU", UgU);
    femElastoDynamicsArchive.WriteData("GgU", GgU);
    femElastoDynamicsArchive.WriteData("HgU", HgU);
    femElastoDynamicsArchive.WriteMetaData("ndbc", ndbc);
    femElastoDynamicsArchive.WriteData("dbc", dbc);
    femElastoDynamicsArchive.WriteData("dmask", dmask);
}

template <
    fem::CElement TElement,
    int Dims,
    physics::CHyperElasticEnergy THyperElasticEnergy,
    common::CFloatingPoint TScalar,
    common::CIndex TIndex>
inline void FemElastoDynamics<TElement, Dims, THyperElasticEnergy, TScalar, TIndex>::Deserialize(
    io::Archive const& archive)
{
    io::Archive const femElastoDynamicsArchive = archive["pbat.sim.dynamics.FemElastoDynamics"];
    mesh.Deserialize(femElastoDynamicsArchive);
    bdf.Deserialize(femElastoDynamicsArchive);
    if (bdf.Order() != 2)
    {
        throw std::runtime_error("FemElastoDynamics only supports BDF of order 2");
    }
    fext = femElastoDynamicsArchive
               .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("fext");
    m      = femElastoDynamicsArchive.ReadData<Eigen::Vector<ScalarType, Eigen::Dynamic>>("m");
    xtilde = femElastoDynamicsArchive
                 .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("xtilde");
    x = femElastoDynamicsArchive
            .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("x");
    v = femElastoDynamicsArchive
            .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("v");
    egU   = femElastoDynamicsArchive.ReadData<Eigen::Vector<IndexType, Eigen::Dynamic>>("egU");
    wgU   = femElastoDynamicsArchive.ReadData<Eigen::Vector<ScalarType, Eigen::Dynamic>>("wgU");
    GNegU = femElastoDynamicsArchive
                .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("GNegU");
    lamegU = femElastoDynamicsArchive
                 .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("lamegU");
    UgU = femElastoDynamicsArchive.ReadData<Eigen::Vector<ScalarType, Eigen::Dynamic>>("UgU");
    GgU = femElastoDynamicsArchive
              .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("GgU");
    HgU = femElastoDynamicsArchive
              .ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("HgU");
    ndbc  = femElastoDynamicsArchive.ReadMetaData<IndexType>("ndbc");
    dbc   = femElastoDynamicsArchive.ReadData<Eigen::Vector<IndexType, Eigen::Dynamic>>("dbc");
    dmask = femElastoDynamicsArchive.ReadData<Eigen::Vector<bool, Eigen::Dynamic>>("dmask");
}

} // namespace pbat::sim::dynamics

#endif // PBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
