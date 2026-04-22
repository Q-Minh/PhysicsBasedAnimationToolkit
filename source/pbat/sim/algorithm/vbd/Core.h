/**
 * @file Core.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Core VBD API.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_CORE_H
#define PBAT_SIM_ALGORITHM_VBD_CORE_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Enums.h"
#include "pbat/io/Archive.h"
#include "pbat/math/LogInterpolate.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/algorithm/common/Common.h"
#include "pbat/sim/algorithm/vbd/Kernels.h"
#include "pbat/sim/contact/MeshDynamics.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <limits>
#include <tbb/parallel_for.h>

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Construct vertex-element adjacency graph
 *
 * @param E Element connectivity `|# nodes per element| x |# elements|` array
 * @param nNodes Number of nodes in the mesh
 * @param GVGp `|# verts+1|` prefixes into GVGe
 * @param GVGe `|# of vertex-elems adjacencies|` element indices s.t. `GVGe[k] for GVGp[i] <= k <
 * GVGp[i+1]` gives the element `e` adjacent to vertex `i`
 * @param GVGilocal `|# of vertex-elems adjacencies|` local vertex indices s.t. `GVGilocal[k] for
 * GVGp[i] <= k < GVGp[i+1]` gives the element `e` adjacent to vertex `i` GVGp[i+1]` gives the local
 * vertex index of vertex `i` in element `e=GVGe[k]`
 */
PBAT_API void VertexElementAdjacencyGraph(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    IndexVectorX& GVGp,
    IndexVectorX& GVGe,
    IndexVectorX& GVGilocal);

/**
 * @brief Compute vertex colors using a greedy algorithm
 *
 * @param E `|# nodes per element| x |# elements|` element connectivity array
 * @param nNodes Number of nodes in the mesh
 * @param eOrdering Vertex color ordering strategy
 * @param eSelection Vertex color selection strategy
 * @param GVVp `|# verts+1|` prefixes into GVVadj
 * @param GVVadj `|# vertex-vertex adjacencies|` adjacent vertex indices
 * @param colors `|# verts| x 1` Vertex colors
 */
PBAT_API void VertexColors(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    graph::EGreedyColorOrderingStrategy eOrdering,
    graph::EGreedyColorSelectionStrategy eSelection,
    IndexVectorX& GVVp,
    IndexVectorX& GVVadj,
    IndexVectorX& colors);

/**
 * @brief VBD simulation configuration
 * @details See \cite anka2024vbd
 */
struct Params
{
  public:
    /**
     * @brief Vertex-element adjacency graph
     * @param _GVGp `|# verts+1|` prefixes into GVGe
     * @param _GVGe `|# of vertex-elems adjacencies|` element indices s.t. `GVGe[k] for GVGp[i] <= k
     * < GVGp[i+1]` gives the element `e` adjacent to vertex `i`
     * @param _GVGilocal `|# of vertex-elems adjacencies|` local vertex indices s.t. `GVGilocal[k]
     * for GVGp[i] <= k < GVGp[i+1]` gives the local vertex index of vertex `i` in element
     * `e=GVGe[k]`
     * @return Reference to this
     */
    PBAT_API Params& WithVertexElementAdjacencyGraph(
        Eigen::Ref<IndexVectorX const> const& _GVGp,
        Eigen::Ref<IndexVectorX const> const& _GVGe,
        Eigen::Ref<IndexVectorX const> const& _GVGilocal);
    /**
     * @brief Vertex colors
     * @param _GVVp `|# verts+1|` prefixes into GVVadj
     * @param _GVVadj `|# vertex-vertex adjacencies|` adjacent vertex indices
     * @param _colors Vertex colors
     * @return Reference to this
     */
    PBAT_API Params& WithVertexColors(
        Eigen::Ref<IndexVectorX const> const& _GVVp,
        Eigen::Ref<IndexVectorX const> const& _GVVadj,
        Eigen::Ref<IndexVectorX const> const& _colors);
    /**
     * @brief Rayleigh damping coefficient
     * @param _betaR Rayleigh damping coefficient
     * @return Reference to this
     */
    PBAT_API Params& WithDamping(Scalar _betaR);
    /**
     * @brief Maximum number of VBD iterations
     * @param nIters Maximum number of iterations
     * @return Reference to this
     */
    PBAT_API Params& WithMaximumIterations(Index nIters);
    /**
     * @brief Homogenization strategy
     * @param strategy Homogenization strategy
     * @param betac Contact homogenization conditioning factor for stiffness matching strategy
     * @return Reference to this
     */
    PBAT_API Params& WithHomogenization(EHomogenizationStrategy strategy, Scalar betac = 0.5);
    /**
     * @brief Stencil gradient acceleration parameters
     * @param betaG0 Initial augmentation coefficient `0 < betaG0 < 1`
     * @param rhohat Lipschitz-normalized threshold above which steps are considered small (i.e.
     * solver progress is slow)
     * @param gammadown Beta reduction factor
     * @param gammaup Beta increase factor
     * @return Reference to this
     */
    PBAT_API Params&
    WithStencilGradientAcceleration(Scalar betaG0, Scalar rhohat, Scalar gammadown, Scalar gammaup);
    /**
     * @brief Numerical zero for hessian pseudo-singularity check
     * @param zero Numerical zero
     * @return Reference to this
     */
    PBAT_API Params& WithHessianDeterminantZeroUnder(Scalar zero);
    /**
     * @brief Construct the simulation data
     * @param bValidate Throw on detected ill-formed inputs
     * @return Reference to this
     */
    PBAT_API Params& Construct(bool bValidate = true);
    /**
     * @brief Serialize this to archive
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize this from archive
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);

  public:
    // Vertex-element adjacency graph
    IndexVectorX GVGp;      ///< `|# verts+1|` prefixes into GVGe
    IndexVectorX GVGe;      ///< `|# of vertex-elems adjacencies|` element indices s.t.
                            ///< `GVGe[k] for GVGp[i] <= k < GVGp[i+1]` gives the element `e`
                            ///< adjacent to vertex `i`
    IndexVectorX GVGilocal; ///< `|# of vertex-elems adjacencies|` local vertex indices s.t.
                            ///< `GVGilocal[k] for GVGp[i] <= k < GVGp[i+1]` gives the local vertex
                            ///< index of vertex `i` in element `e=GVGe[k]`
    // Parallelization
    IndexVectorX colors; ///< `|# vertices|` map of vertex colors
    IndexVectorX GVVp;   ///< `|# verts+1|` prefixes into GVVadj
    IndexVectorX GVVadj; ///< `|# vertex-vertex adjacencies|` adjacent vertex indices
    IndexVectorX Pptr;   ///< `|# partitions+1|` partition pointers, s.t. the range `[Pptr[p],
                         ///< Pptr[p+1])` indexes into Padj from partition `p`
    IndexVectorX Padj;   ///< `|# verts|` partition vertices
    Scalar betaR{0};     ///< Rayleigh damping coefficient
    Index nMaxIters{25}; ///< Maximum number of VBD iterations
    Scalar detHZero{0};  ///< Numerical zero for hessian pseudo-singularity check

    // Homogenization
    EHomogenizationStrategy eHomogenizationStrategy{
        EHomogenizationStrategy::None}; ///< Homogenization strategy
    Scalar betac{
        10}; ///< Contact homogenization conditioning factor for stiffness matching strategy

    // Stencil gradient acceleration
    Scalar betaG0{0.5};   ///< Initial stencil gradient augmentation coefficient `0 < betaG0 < 1`
    Scalar rhohat{0.005}; ///< Lipschitz-normalized threshold above which steps are considered small
                          ///< (i.e. solver progress is slow)
    Scalar gammadown{0.95}; ///< Beta reduction factor
    Scalar gammaup{0.5};    ///< Beta increase factor

    /**
     * @brief Read-write
     */
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xb; ///< `3 x |# nodes|` buffer positions
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        gk; ///< `3 x |# nodes|` approximate gradient at iteration k
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xk; ///< `3 x |# nodes|` past iteration
    Eigen::Vector<Scalar, Eigen::Dynamic> Hnk;   ///< Hessian norms at iteration k
    Eigen::Vector<Scalar, Eigen::Dynamic>
        betaG; ///< Per-vertex stencil gradient augmentation scale coefficient `0 < betaG < 1`
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic>
        log10lame; ///< `2 x |# vertex-element adj.|` matrix of \f$ \log_{10}(\min \mu_{g'} /
                   ///< \mu_{g}) \f$
    Index k;       ///< Current iteration index
};

/**
 * @brief Initialize VBD solve by preparing contact displacement bounds
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

/**
 * @brief One VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

/**
 * @brief Integrate FEM elasto dynamics one step using VBD as the non-linear solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

namespace detail {

/**
 * @brief Accumulate elastic energy derivatives for vertex i
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @tparam FOnEnergyDerivativesComputed Callable with signature `void(Scalar lambda, Scalar mu,
 * mini::SVector<Scalar,3> const& gi, mini::SMatrix<Scalar,3,3> const& Hi)`
 * @param i Vertex index
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 * @param fOnEnergyDerivativesComputed Callback invoked when energy derivatives are computed
 */
template <physics::CHyperElasticEnergy TElasticEnergy, class FOnEnergyDerivativesComputed>
void AccumulateElasticEnergy(
    Index i,
    common::FemElastoDynamics<TElasticEnergy>& fem,
    Params& params,
    FOnEnergyDerivativesComputed&& fOnEnergyDerivativesComputed)
{
    using namespace math::linalg;
    using mini::FromEigen;
    Scalar kbudget = (1 - Scalar(params.k) / (params.nMaxIters - 1));
    auto begin     = params.GVGp(i);
    auto end       = params.GVGp(i + 1);
    for (auto n = begin; n < end; ++n)
    {
        auto ilocal                     = params.GVGilocal(n);
        auto e                          = params.GVGe(n);
        auto lamee                      = fem.lamegU.col(e);
        auto wg                         = fem.wgU(e);
        auto ti                         = fem.mesh.E.col(e);
        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(fem.GNegU.template block<4, 3>(0, e * 3));
        mini::SMatrix<Scalar, 3, 4> xe =
            FromEigen(fem.x(Eigen::placeholders::all, ti).template block<3, 4>(0, 0));
        mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
        TElasticEnergy Psi{};
        mini::SVector<Scalar, 9> gF;
        mini::SMatrix<Scalar, 9, 9> HF;
        if (params.eHomogenizationStrategy ==
            EHomogenizationStrategy::HomogeneousElasticityWithDynamicsMatchingContactStiffness)
        {
            auto gammaMu     = std::pow(Scalar(10), kbudget * params.log10lame(0, n));
            auto gammaLambda = std::pow(Scalar(10), kbudget * params.log10lame(1, n));
            Psi.GradAndHessian(Fe, gammaMu * lamee(0), gammaLambda * lamee(1), gF, HF);
        }
        else
        {
            Psi.GradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
        }
        mini::SMatrix<Scalar, 3, 3> Hie = mini::Zeros<Scalar, 3, 3>();
        mini::SVector<Scalar, 3> gie    = mini::Zeros<Scalar, 3, 1>();
        kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hie);
        kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gie);
        fOnEnergyDerivativesComputed(gie, Hie);
    }
}

/**
 * @brief Accumulate contact energy derivatives for vertex i
 * @tparam FOnEnergyDerivativesComputed Callable with signature `void(mini::SVector<Scalar,3> const&
 * gi, mini::SMatrix<Scalar,3,3> const& Hi)`
 * @tparam TDerivedx Type of position matrix
 * @tparam TDerivedxt Type of position matrix at time t
 * @param i Vertex index
 * @param xi Position of vertex i
 * @param xti Position of vertex i at time t
 * @param x Position matrix
 * @param xt Previous position matrix
 * @param contact Mesh contact dynamics
 * @param params Solver parameters
 * @param rB Contact radius
 * @param kcB Contact stiffness
 * @param kcpB Contact stiffness for penalty
 * @param bB Contact barrier parameter
 * @param mu Contact friction coefficient
 * @param epsvh Relative velocity threshold for static to dynamic friction transition
 * @param h2inv Inverse time step size squared
 * @param fOnEnergyDerivativesComputed Callback invoked when energy derivatives are computed
 */
template <class FOnEnergyDerivativesComputed, class TDerivedx, class TDerivedxt>
inline void AccumulateContactEnergy(
    Index i,
    math::linalg::mini::SVector<Scalar, 3> const& xi,
    math::linalg::mini::SVector<Scalar, 3> const& xti,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedxt> const& xt,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params,
    Scalar rB,
    Scalar kcB,
    Scalar kcpB,
    Scalar bB,
    Scalar mu,
    Scalar epsvh,
    Scalar h2inv,
    FOnEnergyDerivativesComputed&& fOnEnergyDerivativesComputed)
{
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;
    // auto const& Xenv = contact.StaticPointPositions();
    // contact.ForEachPointDynamicMeshContact(
    //     i,
    //     // Vertex-vertex contact
    //     [&](Index j) {
    //         mini::SVector<Scalar, 3> xcp    = FromEigen(params.xb.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> xtcp   = FromEigen(xt.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateVertexClosestPointContactDerivatives(
    //             xi,
    //             xti,
    //             xcp,
    //             xtcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     },
    //     // Vertex-edge contact
    //     [&](Eigen::Vector<Index, 2> const& einds) {
    //         mini::SVector<Scalar, 3> xe1 = FromEigen(params.xb.col(einds(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xe2 = FromEigen(params.xb.col(einds(1)).template head<3>());
    //         mini::SVector<Scalar, 2> uv =
    //             geometry::ClosestPointQueries::UvPointOnLineSegment(xi, xe1, xe2);
    //         mini::SVector<Scalar, 3> xcp    = uv(0) * xe1 + uv(1) * xe2;
    //         mini::SVector<Scalar, 3> xte1   = FromEigen(xt.col(einds(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xte2   = FromEigen(xt.col(einds(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xtcp   = uv(0) * xte1 + uv(1) * xte2;
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateVertexClosestPointContactDerivatives(
    //             xi,
    //             xti,
    //             xcp,
    //             xtcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     },
    //     // Vertex-triangle contact
    //     [&](Eigen::Vector<Index, 3> const& finds) {
    //         mini::SVector<Scalar, 3> xa = FromEigen(params.xb.col(finds(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xb = FromEigen(params.xb.col(finds(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xc = FromEigen(params.xb.col(finds(2)).template head<3>());
    //         mini::SVector<Scalar, 3> uvw =
    //             geometry::ClosestPointQueries::UvwPointInTriangle(xi, xa, xb, xc);
    //         mini::SVector<Scalar, 3> xcp    = uvw(0) * xa + uvw(1) * xb + uvw(2) * xc;
    //         mini::SVector<Scalar, 3> xta    = FromEigen(xt.col(finds(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xtb    = FromEigen(xt.col(finds(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xtc    = FromEigen(xt.col(finds(2)).template head<3>());
    //         mini::SVector<Scalar, 3> xtcp   = uvw(0) * xta + uvw(1) * xtb + uvw(2) * xtc;
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateVertexClosestPointContactDerivatives(
    //             xi,
    //             xti,
    //             xcp,
    //             xtcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     });
    // contact.ForEachPointStaticMeshContact(
    //     i,
    //     // Vertex-vertex contact
    //     [&](Index j) {
    //         mini::SVector<Scalar, 3> xcp    = FromEigen(Xenv.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateVertexClosestPointContactDerivatives(
    //             xi,
    //             xti,
    //             xcp,
    //             xcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     },
    //     // Vertex-edge contact
    //     [&](Eigen::Vector<Index, 2> const& einds) {
    //         mini::SVector<Scalar, 3> xe1 = FromEigen(Xenv.col(einds(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xe2 = FromEigen(Xenv.col(einds(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xcp =
    //             geometry::ClosestPointQueries::PointOnLineSegment(xi, xe1, xe2);
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateVertexClosestPointContactDerivatives(
    //             xi,
    //             xti,
    //             xcp,
    //             xcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     },
    //     // Vertex-triangle contact
    //     [&](Eigen::Vector<Index, 3> const& finds) {
    //         mini::SVector<Scalar, 3> xf1 = FromEigen(Xenv.col(finds(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xf2 = FromEigen(Xenv.col(finds(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xf3 = FromEigen(Xenv.col(finds(2)).template head<3>());
    //         mini::SVector<Scalar, 3> xcp =
    //             geometry::ClosestPointQueries::PointInTriangle(xi, xf1, xf2, xf3);
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateVertexClosestPointContactDerivatives(
    //             xi,
    //             xti,
    //             xcp,
    //             xcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     });
    // contact.ForEachHalfEdgeDynamicMeshContactIncidentOnPoint(
    //     i,
    //     // Edge-vertex contact
    //     [&](Eigen::Vector<Index, 2> const& eindsi, Index j) {
    //         // NOTE: xi1 should be xi
    //         // mini::SVector<Scalar, 3> xi1 =
    //         //     FromEigen(fem.x.col(eindsi(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xi2  = FromEigen(params.xb.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xcp  = FromEigen(params.xb.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> xtcp = FromEigen(xt.col(j).template head<3>());
    //         mini::SVector<Scalar, 2> uv =
    //             geometry::ClosestPointQueries::UvPointOnLineSegment(xcp, xi, xi2);
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
    //             xi,
    //             xi2,
    //             xti,
    //             xti2,
    //             uv,
    //             0 /*ilocal == 0 because i == eindsi(0)*/,
    //             xcp,
    //             xtcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     },
    //     // Edge-edge contact
    //     [&](Eigen::Vector<Index, 2> const& eindsi, Eigen::Vector<Index, 2> const& eindsj) {
    //         mini::SVector<Scalar, 3> xi2 = FromEigen(params.xb.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xj1 = FromEigen(params.xb.col(eindsj(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xj2 = FromEigen(params.xb.col(eindsj(1)).template head<3>());
    //         mini::SVector<Scalar, 2> st =
    //             geometry::ClosestPointQueries::LineSegments(xi, xi2, xj1, xj2);
    //         mini::SVector<Scalar, 3> xcp  = (1 - st(1)) * xj1 + st(1) * xj2;
    //         mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xtj1 = FromEigen(xt.col(eindsj(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xtj2 = FromEigen(xt.col(eindsj(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xtcp = (1 - st(1)) * xtj1 + st(1) * xtj2;
    //         mini::SVector<Scalar, 2> uv1{1 - st(0), st(0)};
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
    //             xi,
    //             xi2,
    //             xti,
    //             xti2,
    //             uv1,
    //             0 /*ilocal == 0 because i == eindsi(0)*/,
    //             xcp,
    //             xtcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     });
    // contact.ForEachHalfEdgeStaticMeshContactIncidentOnPoint(
    //     i,
    //     // Edge-vertex contact
    //     [&](Eigen::Vector<Index, 2> const& eindsi, Index j) {
    //         mini::SVector<Scalar, 3> xi2  = FromEigen(params.xb.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xcp  = FromEigen(Xenv.col(j).template head<3>());
    //         mini::SVector<Scalar, 2> uv =
    //             geometry::ClosestPointQueries::UvPointOnLineSegment(xcp, xi, xi2);
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
    //             xi,
    //             xi2,
    //             xti,
    //             xti2,
    //             uv,
    //             0 /*ilocal == 0 because i == eindsi(0)*/,
    //             xcp,
    //             xcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     },
    //     // Edge-edge contact
    //     [&](Eigen::Vector<Index, 2> const& eindsi, Eigen::Vector<Index, 2> const& eindsj) {
    //         mini::SVector<Scalar, 3> xi2  = FromEigen(params.xb.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
    //         mini::SVector<Scalar, 3> xj1  = FromEigen(Xenv.col(eindsj(0)).template head<3>());
    //         mini::SVector<Scalar, 3> xj2  = FromEigen(Xenv.col(eindsj(1)).template head<3>());
    //         mini::SVector<Scalar, 2> st =
    //             geometry::ClosestPointQueries::LineSegments(xi, xi2, xj1, xj2);
    //         mini::SVector<Scalar, 3> xcp = (1 - st(1)) * xj1 + st(1) * xj2;
    //         mini::SVector<Scalar, 2> uv1{1 - st(0), st(0)};
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
    //             xi,
    //             xi2,
    //             xti,
    //             xti2,
    //             uv1,
    //             0 /*ilocal == 0 because i == eindsi(0)*/,
    //             xcp,
    //             xcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     });
    // contact.ForEachDynamicPointContactOnTrianglesIncidentOnPoint(
    //     i,
    //     // Triangle-vertex contact
    //     [&](Eigen::Vector<Index, 3> const& finds, Index j) {
    //         int ilocal =
    //             /*(finds(0) == i) * 0 + */ (finds(1) == i) * 1 + (finds(2) == i) * 2;
    //         int jlocal = (ilocal + 1) % 3;
    //         int klocal = (ilocal + 2) % 3;
    //         mini::SVector<Scalar, 3> xb =
    //             FromEigen(params.xb.col(finds(jlocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xc =
    //             FromEigen(params.xb.col(finds(klocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xtb  = FromEigen(xt.col(finds(jlocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xtc  = FromEigen(xt.col(finds(klocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xcp  = FromEigen(params.xb.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> xtcp = FromEigen(xt.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> uvw =
    //             geometry::ClosestPointQueries::UvwPointInTriangle(xcp, xi, xb, xc);
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateTriangleVertexToClosestPointContactDerivatives(
    //             xi,
    //             xb,
    //             xc,
    //             xti,
    //             xtb,
    //             xtc,
    //             uvw,
    //             0 /*ilocal == 0, because finds(ilocal) == i*/,
    //             xcp,
    //             xtcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     });
    // contact.ForEachStaticPointContactOnTrianglesIncidentOnPoint(
    //     i,
    //     // Triangle-vertex contact
    //     [&](Eigen::Vector<Index, 3> const& finds, Index j) {
    //         int ilocal =
    //             /*(finds(0) == i) * 0 + */ (finds(1) == i) * 1 + (finds(2) == i) * 2;
    //         int jlocal = (ilocal + 1) % 3;
    //         int klocal = (ilocal + 2) % 3;
    //         mini::SVector<Scalar, 3> xb =
    //             FromEigen(params.xb.col(finds(jlocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xc =
    //             FromEigen(params.xb.col(finds(klocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xtb = FromEigen(xt.col(finds(jlocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xtc = FromEigen(xt.col(finds(klocal)).template head<3>());
    //         mini::SVector<Scalar, 3> xcp = FromEigen(Xenv.col(j).template head<3>());
    //         mini::SVector<Scalar, 3> uvw =
    //             geometry::ClosestPointQueries::UvwPointInTriangle(xcp, xi, xb, xc);
    //         mini::SVector<Scalar, 3> gic    = mini::Zeros<Scalar, 3, 1>();
    //         mini::SMatrix<Scalar, 3, 3> Hic = mini::Zeros<Scalar, 3, 3>();
    //         Scalar dc = kernels::AccumulateTriangleVertexToClosestPointContactDerivatives(
    //             xi,
    //             xb,
    //             xc,
    //             xti,
    //             xtb,
    //             xtc,
    //             uvw,
    //             0 /*ilocal == 0, because finds(ilocal) == i*/,
    //             xcp,
    //             xcp,
    //             rB,
    //             kcB,
    //             kcpB,
    //             bB,
    //             mu,
    //             epsvh,
    //             h2inv,
    //             gic,
    //             Hic);
    //         assert(
    //             not ToEigen(gic).hasNaN() and not ToEigen(Hic).hasNaN() and
    //             ToEigen(gic).allFinite() and ToEigen(Hic).allFinite());
    //         fOnEnergyDerivativesComputed(dc, gic, Hic);
    //     });
}

/**
 * @brief Build the local vertex equation (gradient and Hessian) for vertex i
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @tparam TDerivedx Type of position matrix
 * @tparam TDerivedxt Type of position matrix at time t
 * @param i Vertex index
 * @param xi Position of vertex i
 * @param xti Position of vertex i at time t
 * @param xtildei Inertial target position of vertex i
 * @param m Mass of vertex i
 * @param h Time step size
 * @param h2 Time step size squared
 * @param x Position matrix
 * @param xt Previous position matrix
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact dynamics
 * @param params Solver parameters
 * @return (Hi, gi) where Hi is the Hessian and gi is the gradient for vertex i
 */
template <physics::CHyperElasticEnergy TElasticEnergy, class TDerivedx, class TDerivedxt>
auto BuildVertexEquation(
    Index i,
    math::linalg::mini::SVector<Scalar, 3> const& xi,
    math::linalg::mini::SVector<Scalar, 3> const& xti,
    math::linalg::mini::SVector<Scalar, 3> const& xtildei,
    Scalar m,
    Scalar h,
    Scalar h2,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedxt> const& xt,
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
    -> std::pair<math::linalg::mini::SMatrix<Scalar, 3, 3>, math::linalg::mini::SVector<Scalar, 3>>
{
    using namespace math::linalg;
    using namespace std;
    auto const& contactParams      = contact.GetParams();
    Scalar h2inv                   = 1 / h2;
    Scalar rB                      = contactParams.mOgcParams.r;
    Scalar kcB                     = contactParams.kc;
    Scalar kcpB                    = contactParams.kcp;
    Scalar bB                      = contactParams.b;
    Scalar epsvh                   = contactParams.epsv * h;
    Scalar mu                      = contactParams.mu;
    mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
    mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
    switch (params.eHomogenizationStrategy)
    {
        case EHomogenizationStrategy::None: {
            // Elastic energy
            AccumulateElasticEnergy<TElasticEnergy>(
                i,
                fem,
                params,
                [&](mini::SVector<Scalar, 3> const& gie, mini::SMatrix<Scalar, 3, 3> const& Hie) {
                    gi += gie;
                    Hi += Hie;
                });
            gi *= h2;
            Hi *= h2;
            // Contact energy
            AccumulateContactEnergy(
                i,
                xi,
                xti,
                x,
                xt,
                contact,
                params,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                Scalar(1) /*h2inv*/,
                [&]([[maybe_unused]] Scalar dc,
                    mini::SVector<Scalar, 3> const& gic,
                    mini::SMatrix<Scalar, 3, 3> const& Hic) {
                    gi += gic;
                    Hi += Hic;
                });
            // Kinetic energy
            kernels::AddInertiaDerivatives(Scalar(1) /*h2*/, m, xtildei, xi, gi, Hi);
            // Damping
            kernels::AddDamping(Scalar(1) / h, xti, xi, params.betaR, gi, Hi);
            break;
        }
        case EHomogenizationStrategy::HomogeneousElasticityWithDynamicsMatchingContactStiffness: {
            // Approximate sub-stepping
            auto K = static_cast<Scalar>(params.nMaxIters);
            auto k = static_cast<Scalar>(params.k);
            // Linear interpolation:
            // h *= ((k + 1) < 0.8 * K) ? (k + 1) / K / 0.8 : 1;
            // Exponential interpolation:
            // h     = h * exp(-log(K) * (1 - k / (K - 1)));
            // Log interpolation
            Scalar regime{0.2};
            h     = ((k + 1) < regime * K) ?
                        math::LogInterpolate(h / K, h, Scalar(1), K * regime, k + 1) :
                        h;
            h2    = h * h;
            h2inv = 1 / h2;
            // Elastic energy
            mini::SMatrix<Scalar, 3, 3> HiU = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> giU    = mini::Zeros<Scalar, 3, 1>();
            AccumulateElasticEnergy<TElasticEnergy>(
                i,
                fem,
                params,
                [&](mini::SVector<Scalar, 3> const& gie, mini::SMatrix<Scalar, 3, 3> const& Hie) {
                    giU += gie;
                    HiU += Hie;
                });
            giU *= h2;
            HiU *= h2;
            // Contact energy with dynamics-matching stiffness
            Scalar gammacK                  = m * sqrt(3);
            Scalar gammacU                  = Norm(HiU);
            mini::SVector<Scalar, 3> giC    = mini::Zeros<Scalar, 3, 1>();
            mini::SMatrix<Scalar, 3, 3> HiC = mini::Zeros<Scalar, 3, 3>();
            AccumulateContactEnergy(
                i,
                xi,
                xti,
                x,
                xt,
                contact,
                params,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                Scalar(1) /*h2inv*/,
                [&]([[maybe_unused]] Scalar dc,
                    mini::SVector<Scalar, 3> const& gic,
                    mini::SMatrix<Scalar, 3, 3> const& Hic) {
                    giC += (gammacK / dc + gammacU) * gic;
                    HiC += (gammacK / dc + gammacU) * Hic;
                });
            giC *= params.betac;
            HiC *= params.betac;
            // Kinetic energy
            mini::SVector<Scalar, 3> giK    = mini::Zeros<Scalar, 3, 1>();
            mini::SMatrix<Scalar, 3, 3> HiK = mini::Zeros<Scalar, 3, 3>();
            kernels::AddInertiaDerivatives(Scalar(1) /*h2*/, m, xtildei, xi, giK, HiK);
            // Assemble
            gi = giU + giC + giK;
            Hi = HiU + HiC + HiK;
            kernels::AddDamping(Scalar(1) / h, xti, xi, params.betaR, gi, Hi);
            break;
        }
        case EHomogenizationStrategy::Conditioning: {
            // Elastic energy
            mini::SMatrix<Scalar, 3, 3> HiU = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> giU    = mini::Zeros<Scalar, 3, 1>();
            AccumulateElasticEnergy<TElasticEnergy>(
                i,
                fem,
                params,
                [&](mini::SVector<Scalar, 3> const& gie, mini::SMatrix<Scalar, 3, 3> const& Hie) {
                    giU += gie;
                    HiU += Hie;
                });
            giU *= h2;
            HiU *= h2;
            // Contact energy
            mini::SMatrix<Scalar, 3, 3> HiC = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> giC    = mini::Zeros<Scalar, 3, 1>();
            AccumulateContactEnergy(
                i,
                xi,
                xti,
                x,
                xt,
                contact,
                params,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                Scalar(1) /*h2inv*/,
                [&]([[maybe_unused]] Scalar dc,
                    mini::SVector<Scalar, 3> const& gic,
                    mini::SMatrix<Scalar, 3, 3> const& Hic) {
                    giC += gic;
                    HiC += Hic;
                });
            // Kinetic energy
            mini::SMatrix<Scalar, 3, 3> HiK = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> giK    = mini::Zeros<Scalar, 3, 1>();
            kernels::AddInertiaDerivatives(Scalar(1) /*h2*/, m, xtildei, xi, giK, HiK);
            // Assemble
            Scalar gammaK = math::LogInterpolate(
                Scalar(1) / max(SquaredNorm(HiK), numeric_limits<Scalar>::epsilon()),
                Scalar(1),
                Scalar(1),
                static_cast<Scalar>(params.nMaxIters),
                static_cast<Scalar>(params.k + 1));
            Scalar gammaU = math::LogInterpolate(
                Scalar(1) / max(Norm(HiU), numeric_limits<Scalar>::epsilon()),
                Scalar(1),
                Scalar(1),
                static_cast<Scalar>(params.nMaxIters),
                static_cast<Scalar>(params.k + 1));
            Scalar gammaC = math::LogInterpolate(
                Scalar(1) / max(Norm(HiC), numeric_limits<Scalar>::epsilon()),
                Scalar(1),
                Scalar(1),
                static_cast<Scalar>(params.nMaxIters),
                static_cast<Scalar>(params.k + 1));
            gi = gammaU * giU + gammaC * giC + gammaK * giK;
            Hi = gammaU * HiU + gammaC * HiC + gammaK * HiK;
            // Damping
            kernels::AddDamping(Scalar(1) / h, xti, xi, params.betaR, gi, Hi);
            break;
        }
    }
    return {Hi, gi};
}

/**
 * @brief Adapt stencil gradient acceleration parameter for vertex i
 * @param i Vertex index
 * @param xi Current position of vertex i
 * @param gi Gradient at vertex i
 * @param Hi Hessian at vertex i
 * @param params Solver parameters (in/out: betaG, gk, xk, Hnk are updated)
 */
inline void AdaptStencilGradientAccelerationParameter(
    Index i,
    math::linalg::mini::SVector<Scalar, 3> const& xi,
    math::linalg::mini::SVector<Scalar, 3> const& gi,
    math::linalg::mini::SMatrix<Scalar, 3, 3> const& Hi,
    Params& params)
{
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::Norm;
    using mini::ToEigen;
    if (params.k > 0)
    {
        Scalar ngk       = Norm(gi);
        auto gkm1        = params.gk.col(i).template head<3>();
        Scalar ngkm1     = Norm(FromEigen(gkm1));
        Scalar ndgkm1    = Norm(gi - FromEigen(gkm1));
        params.gk.col(i) = ToEigen(gi);
        auto xk          = params.xk.col(i).template head<3>();
        Scalar ndxkm1    = Norm(xi - FromEigen(xk));
        Scalar L         = params.Hnk(i) + ngk / ndxkm1;
        Scalar rho       = ndgkm1 / (L * ndxkm1);
        if (ngk > ngkm1)
            params.betaG(i) *= params.gammadown;
        else if (rho > params.rhohat)
            params.betaG(i) += (1 - params.betaG(i)) * params.gammaup;
    }
    params.gk.col(i) = ToEigen(gi);
    params.xk.col(i) = ToEigen(xi);
    params.Hnk(i)    = Norm(Hi);
}

/**
 * @brief Compute stencil gradient augmentation for vertex i
 * @param i Vertex index
 * @param gi Gradient at vertex i
 * @param params Solver parameters
 * @return Augmentation vector to be added to the gradient
 */
inline math::linalg::mini::SVector<Scalar, 3> ComputeStencilGradientAugmentation(
    Index i,
    math::linalg::mini::SVector<Scalar, 3> const& gi,
    Params const& params)
{
    using namespace math::linalg;
    using mini::Dot;
    auto nbegin                 = params.GVVp(i);
    auto nend                   = params.GVVp(i + 1);
    mini::SVector<Scalar, 3> gp = mini::Zeros<Scalar, 3, 1>();
    for (auto n = nbegin; n < nend; ++n)
    {
        auto j = params.GVVadj(n);
        gp += mini::FromEigen(params.gk.col(j).template head<3>());
    }
    Scalar lambda = params.betaG(i) * Dot(gi, gp) / Dot(gp, gp);
    return std::max(lambda, Scalar(0)) * gp;
}

} // namespace detail

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Iterate");
    auto h                 = fem.bdf.BetaTilde();
    auto h2                = h * h;
    auto xtildeBdf         = fem.bdf.Inertia(0).reshaped(fem.x.rows(), fem.x.cols());
    auto xt                = -xtildeBdf;
    params.xb              = fem.x; // Copy current positions to buffer
    auto const nPartitions = params.Pptr.size() - 1;
    for (Index p = 0; p < nPartitions; ++p)
    {
        auto const pBegin = params.Pptr(p);
        auto const pEnd   = params.Pptr(p + 1);
        tbb::parallel_for(pBegin, pEnd, [&](Index k) {
            // Solve vertex i
            using namespace math::linalg;
            using mini::FromEigen;
            using mini::ToEigen;
            auto i = params.Padj(k);
            if (fem.IsDirichletNode(i))
                return;
            // Fetch vertex data
            mini::SVector<Scalar, 3> xi      = FromEigen(fem.x.col(i).template head<3>());
            mini::SVector<Scalar, 3> xti     = FromEigen(xt.col(i).template head<3>());
            Scalar m                         = fem.m(i);
            mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).template head<3>());
            // Build and solve the vertex equation
            auto [Hi, gi] = detail::BuildVertexEquation<
                TElasticEnergy>(i, xi, xti, xtildei, m, h, h2, fem.x, xt, fem, contact, params);
            // Adapt stencil gradient acceleration parameter
            detail::AdaptStencilGradientAccelerationParameter(i, xi, gi, Hi, params);
            // Augment gradient
            gi += detail::ComputeStencilGradientAugmentation(i, gi, params);
            // Solve
            kernels::IntegratePositions(gi, Hi, xi, params.detHZero);
            fem.x.col(i) = ToEigen(xi);
        });
    }
    ++params.k;
}

/**
 * @brief Initialize homogenization for VBD solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact dynamics
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeHomogenization(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    [[maybe_unused]] contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeHomogenization");
    switch (params.eHomogenizationStrategy)
    {
        case EHomogenizationStrategy::None: break;
        case EHomogenizationStrategy::HomogeneousElasticityWithDynamicsMatchingContactStiffness: {
            // Elasticity
            tbb::parallel_for(Index(0), fem.x.cols(), [&](Index i) {
                auto begin             = params.GVGp(i);
                auto end               = params.GVGp(i + 1);
                auto nAdjacentElements = end - begin;
                auto e                 = params.GVGe(Eigen::seqN(begin, nAdjacentElements));
                auto lamee             = fem.lamegU(Eigen::placeholders::all, e);
                auto minMu             = lamee.row(0).minCoeff();
                auto minLambda         = lamee.row(1).minCoeff();
                for (auto k = 0; k < nAdjacentElements; ++k)
                {
                    params.log10lame(0, begin + k) = std::log10(minMu / lamee(0, k));
                    params.log10lame(1, begin + k) = std::log10(minLambda / lamee(1, k));
                }
            });
            break;
        }
        case EHomogenizationStrategy::Conditioning: {
            break;
        }
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeSolve");
    auto const xt = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
    contact.GetParams().ComputeQueryRadius((fem.xtilde - xt).colwise().norm().maxCoeff());
    contact.UpdateConstraintSet(xt);
    contact.RestoreFeasibility(fem.x, fem.dmask);
    InitializeHomogenization<TElasticEnergy>(fem, contact, params);
    params.k = 0;
    params.gk.setZero();
    params.xk = fem.x;
    params.betaG.setConstant(params.betaG0);
    params.Hnk.setZero();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Solve");
    while (params.k < params.nMaxIters)
    {
        if (contact.RequiresConstraintSetUpdate())
            contact.UpdateConstraintSet(fem.x);
        Iterate<TElasticEnergy>(fem, contact, params);
        contact.RestoreFeasibility(fem.x, fem.dmask);
    }
    fem.BackSubstituteIntegratedPositionsIntoVelocities();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Integrate");
    Solve<TElasticEnergy>(fem, contact, params);
    fem.Step();
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CORE_H
