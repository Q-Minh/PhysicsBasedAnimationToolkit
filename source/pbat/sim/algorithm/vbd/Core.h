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
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/algorithm/common/Common.h"
#include "pbat/sim/algorithm/vbd/Kernels.h"
#include "pbat/sim/contact/MeshDynamics.h"

#include <Eigen/Core>
#include <cassert>
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
    Eigen::Ref<IndexVectorX> GVGp,
    Eigen::Ref<IndexVectorX> GVGe,
    Eigen::Ref<IndexVectorX> GVGilocal);

/**
 * @brief Compute vertex colors using a greedy algorithm
 *
 * @param E `|# nodes per element| x |# elements|` element connectivity array
 * @param nNodes Number of nodes in the mesh
 * @param eOrdering Vertex color ordering strategy
 * @param eSelection Vertex color selection strategy
 * @param colors `|# verts| x 1` Vertex colors
 */
PBAT_API void VertexColors(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    graph::EGreedyColorOrderingStrategy eOrdering,
    graph::EGreedyColorSelectionStrategy eSelection,
    Eigen::Ref<IndexVectorX> colors);

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
     * @param _colors Vertex colors
     * @return Reference to this
     */
    PBAT_API Params& WithVertexColors(Eigen::Ref<IndexVectorX const> const& _colors);
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
    IndexVectorX Pptr;   ///< `|# partitions+1|` partition pointers, s.t. the range `[Pptr[p],
                         ///< Pptr[p+1])` indexes into Padj from partition `p`
    IndexVectorX Padj;   ///< `|# verts|` partition vertices
    Scalar betaR{0};     ///< Rayleigh damping coefficient
    Index nMaxIters{25}; ///< Maximum number of VBD iterations
    Scalar detHZero{0};  ///< Numerical zero for hessian pseudo-singularity check

    /**
     * @brief Read-write
     */
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xb; ///< `3 x |# nodes|` buffer positions
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
    [[maybe_unused]] Params& params);

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
 * @param i Vertex index
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 * @param gi Gradient accumulator (in/out)
 * @param Hi Hessian accumulator (in/out)
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AccumulateElasticEnergy(
    Index i,
    common::FemElastoDynamics<TElasticEnergy>& fem,
    Params& params,
    math::linalg::mini::SVector<Scalar, 3>& gi,
    math::linalg::mini::SMatrix<Scalar, 3, 3>& Hi)
{
    using namespace math::linalg;
    using mini::FromEigen;
    auto begin = params.GVGp(i);
    auto end   = params.GVGp(i + 1);
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
        Psi.GradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
        kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
        kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
    }
}

/**
 * @brief Accumulate contact energy derivatives for vertex i
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
 * @param gi Gradient accumulator (in/out)
 * @param Hi Hessian accumulator (in/out)
 */
template <class TDerivedx, class TDerivedxt>
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
    math::linalg::mini::SVector<Scalar, 3>& gi,
    math::linalg::mini::SMatrix<Scalar, 3, 3>& Hi)
{
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;
    auto const& Xenv = contact.StaticPointPositions();
    contact.ForEachPointDynamicMeshContact(
        i,
        // Vertex-vertex contact
        [&](Index j) {
            mini::SVector<Scalar, 3> xcp  = FromEigen(params.xb.col(j).template head<3>());
            mini::SVector<Scalar, 3> xtcp = FromEigen(xt.col(j).template head<3>());
            kernels::AccumulateVertexClosestPointContactDerivatives(
                xi,
                xti,
                xcp,
                xtcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        },
        // Vertex-edge contact
        [&](Eigen::Vector<Index, 2> const& einds) {
            mini::SVector<Scalar, 3> xe1 = FromEigen(params.xb.col(einds(0)).template head<3>());
            mini::SVector<Scalar, 3> xe2 = FromEigen(params.xb.col(einds(1)).template head<3>());
            mini::SVector<Scalar, 2> uv =
                geometry::ClosestPointQueries::UvPointOnLineSegment(xi, xe1, xe2);
            mini::SVector<Scalar, 3> xcp  = uv(0) * xe1 + uv(1) * xe2;
            mini::SVector<Scalar, 3> xte1 = FromEigen(xt.col(einds(0)).template head<3>());
            mini::SVector<Scalar, 3> xte2 = FromEigen(xt.col(einds(1)).template head<3>());
            mini::SVector<Scalar, 3> xtcp = uv(0) * xte1 + uv(1) * xte2;
            kernels::AccumulateVertexClosestPointContactDerivatives(
                xi,
                xti,
                xcp,
                xtcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        },
        // Vertex-triangle contact
        [&](Eigen::Vector<Index, 3> const& finds) {
            mini::SVector<Scalar, 3> xa = FromEigen(params.xb.col(finds(0)).template head<3>());
            mini::SVector<Scalar, 3> xb = FromEigen(params.xb.col(finds(1)).template head<3>());
            mini::SVector<Scalar, 3> xc = FromEigen(params.xb.col(finds(2)).template head<3>());
            mini::SVector<Scalar, 3> uvw =
                geometry::ClosestPointQueries::UvwPointInTriangle(xi, xa, xb, xc);
            mini::SVector<Scalar, 3> xcp  = uvw(0) * xa + uvw(1) * xb + uvw(2) * xc;
            mini::SVector<Scalar, 3> xta  = FromEigen(xt.col(finds(0)).template head<3>());
            mini::SVector<Scalar, 3> xtb  = FromEigen(xt.col(finds(1)).template head<3>());
            mini::SVector<Scalar, 3> xtc  = FromEigen(xt.col(finds(2)).template head<3>());
            mini::SVector<Scalar, 3> xtcp = uvw(0) * xta + uvw(1) * xtb + uvw(2) * xtc;
            kernels::AccumulateVertexClosestPointContactDerivatives(
                xi,
                xti,
                xcp,
                xtcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        });
    contact.ForEachPointStaticMeshContact(
        i,
        // Vertex-vertex contact
        [&](Index j) {
            mini::SVector<Scalar, 3> xcp = FromEigen(Xenv.col(j).template head<3>());
            kernels::AccumulateVertexClosestPointContactDerivatives(
                xi,
                xti,
                xcp,
                xcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        },
        // Vertex-edge contact
        [&](Eigen::Vector<Index, 2> const& einds) {
            mini::SVector<Scalar, 3> xe1 = FromEigen(Xenv.col(einds(0)).template head<3>());
            mini::SVector<Scalar, 3> xe2 = FromEigen(Xenv.col(einds(1)).template head<3>());
            mini::SVector<Scalar, 3> xcp =
                geometry::ClosestPointQueries::PointOnLineSegment(xi, xe1, xe2);
            kernels::AccumulateVertexClosestPointContactDerivatives(
                xi,
                xti,
                xcp,
                xcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        },
        // Vertex-triangle contact
        [&](Eigen::Vector<Index, 3> const& finds) {
            mini::SVector<Scalar, 3> xf1 = FromEigen(Xenv.col(finds(0)).template head<3>());
            mini::SVector<Scalar, 3> xf2 = FromEigen(Xenv.col(finds(1)).template head<3>());
            mini::SVector<Scalar, 3> xf3 = FromEigen(Xenv.col(finds(2)).template head<3>());
            mini::SVector<Scalar, 3> xcp =
                geometry::ClosestPointQueries::PointInTriangle(xi, xf1, xf2, xf3);
            kernels::AccumulateVertexClosestPointContactDerivatives(
                xi,
                xti,
                xcp,
                xcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        });
    contact.ForEachHalfEdgeDynamicMeshContactIncidentOnPoint(
        i,
        // Edge-vertex contact
        [&](Eigen::Vector<Index, 2> const& eindsi, Index j) {
            // NOTE: xi1 should be xi
            // mini::SVector<Scalar, 3> xi1 =
            //     FromEigen(fem.x.col(eindsi(0)).template head<3>());
            mini::SVector<Scalar, 3> xi2  = FromEigen(params.xb.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xcp  = FromEigen(params.xb.col(j).template head<3>());
            mini::SVector<Scalar, 3> xtcp = FromEigen(xt.col(j).template head<3>());
            mini::SVector<Scalar, 2> uv =
                geometry::ClosestPointQueries::UvPointOnLineSegment(xcp, xi, xi2);
            kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
                xi,
                xi2,
                xti,
                xti2,
                uv,
                0 /*ilocal == 0 because i == eindsi(0)*/,
                xcp,
                xtcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        },
        // Edge-edge contact
        [&](Eigen::Vector<Index, 2> const& eindsi, Eigen::Vector<Index, 2> const& eindsj) {
            mini::SVector<Scalar, 3> xi2 = FromEigen(params.xb.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xj1 = FromEigen(params.xb.col(eindsj(0)).template head<3>());
            mini::SVector<Scalar, 3> xj2 = FromEigen(params.xb.col(eindsj(1)).template head<3>());
            mini::SVector<Scalar, 2> st =
                geometry::ClosestPointQueries::LineSegments(xi, xi2, xj1, xj2);
            mini::SVector<Scalar, 3> xcp  = (1 - st(1)) * xj1 + st(1) * xj2;
            mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xtj1 = FromEigen(xt.col(eindsj(0)).template head<3>());
            mini::SVector<Scalar, 3> xtj2 = FromEigen(xt.col(eindsj(1)).template head<3>());
            mini::SVector<Scalar, 3> xtcp = (1 - st(1)) * xtj1 + st(1) * xtj2;
            mini::SVector<Scalar, 2> uv1{1 - st(0), st(0)};
            kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
                xi,
                xi2,
                xti,
                xti2,
                uv1,
                0 /*ilocal == 0 because i == eindsi(0)*/,
                xcp,
                xtcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        });
    contact.ForEachHalfEdgeStaticMeshContactIncidentOnPoint(
        i,
        // Edge-vertex contact
        [&](Eigen::Vector<Index, 2> const& eindsi, Index j) {
            mini::SVector<Scalar, 3> xi2  = FromEigen(params.xb.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xcp  = FromEigen(Xenv.col(j).template head<3>());
            mini::SVector<Scalar, 2> uv =
                geometry::ClosestPointQueries::UvPointOnLineSegment(xcp, xi, xi2);
            kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
                xi,
                xi2,
                xti,
                xti2,
                uv,
                0 /*ilocal == 0 because i == eindsi(0)*/,
                xcp,
                xcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        },
        // Edge-edge contact
        [&](Eigen::Vector<Index, 2> const& eindsi, Eigen::Vector<Index, 2> const& eindsj) {
            mini::SVector<Scalar, 3> xi2  = FromEigen(params.xb.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xti2 = FromEigen(xt.col(eindsi(1)).template head<3>());
            mini::SVector<Scalar, 3> xj1  = FromEigen(Xenv.col(eindsj(0)).template head<3>());
            mini::SVector<Scalar, 3> xj2  = FromEigen(Xenv.col(eindsj(1)).template head<3>());
            mini::SVector<Scalar, 2> st =
                geometry::ClosestPointQueries::LineSegments(xi, xi2, xj1, xj2);
            mini::SVector<Scalar, 3> xcp = (1 - st(1)) * xj1 + st(1) * xj2;
            mini::SVector<Scalar, 2> uv1{1 - st(0), st(0)};
            kernels::AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
                xi,
                xi2,
                xti,
                xti2,
                uv1,
                0 /*ilocal == 0 because i == eindsi(0)*/,
                xcp,
                xcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        });
    contact.ForEachDynamicPointContactOnTrianglesIncidentOnPoint(
        i,
        // Triangle-vertex contact
        [&](Eigen::Vector<Index, 3> const& finds, Index j) {
            int ilocal =
                /*(finds(0) == i) * 0 + */ (finds(1) == i) * 1 + (finds(2) == i) * 2;
            int jlocal = (ilocal + 1) % 3;
            int klocal = (ilocal + 2) % 3;
            mini::SVector<Scalar, 3> xb =
                FromEigen(params.xb.col(finds(jlocal)).template head<3>());
            mini::SVector<Scalar, 3> xc =
                FromEigen(params.xb.col(finds(klocal)).template head<3>());
            mini::SVector<Scalar, 3> xtb  = FromEigen(xt.col(finds(jlocal)).template head<3>());
            mini::SVector<Scalar, 3> xtc  = FromEigen(xt.col(finds(klocal)).template head<3>());
            mini::SVector<Scalar, 3> xcp  = FromEigen(params.xb.col(j).template head<3>());
            mini::SVector<Scalar, 3> xtcp = FromEigen(xt.col(j).template head<3>());
            mini::SVector<Scalar, 3> uvw =
                geometry::ClosestPointQueries::UvwPointInTriangle(xcp, xi, xb, xc);
            kernels::AccumulateTriangleVertexToClosestPointContactDerivatives(
                xi,
                xb,
                xc,
                xti,
                xtb,
                xtc,
                uvw,
                0 /*ilocal == 0, because finds(ilocal) == i*/,
                xcp,
                xtcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        });
    contact.ForEachStaticPointContactOnTrianglesIncidentOnPoint(
        i,
        // Triangle-vertex contact
        [&](Eigen::Vector<Index, 3> const& finds, Index j) {
            int ilocal =
                /*(finds(0) == i) * 0 + */ (finds(1) == i) * 1 + (finds(2) == i) * 2;
            int jlocal = (ilocal + 1) % 3;
            int klocal = (ilocal + 2) % 3;
            mini::SVector<Scalar, 3> xb =
                FromEigen(params.xb.col(finds(jlocal)).template head<3>());
            mini::SVector<Scalar, 3> xc =
                FromEigen(params.xb.col(finds(klocal)).template head<3>());
            mini::SVector<Scalar, 3> xtb = FromEigen(xt.col(finds(jlocal)).template head<3>());
            mini::SVector<Scalar, 3> xtc = FromEigen(xt.col(finds(klocal)).template head<3>());
            mini::SVector<Scalar, 3> xcp = FromEigen(Xenv.col(j).template head<3>());
            mini::SVector<Scalar, 3> uvw =
                geometry::ClosestPointQueries::UvwPointInTriangle(xcp, xi, xb, xc);
            kernels::AccumulateTriangleVertexToClosestPointContactDerivatives(
                xi,
                xb,
                xc,
                xti,
                xtb,
                xtc,
                uvw,
                0 /*ilocal == 0, because finds(ilocal) == i*/,
                xcp,
                xcp,
                rB,
                kcB,
                kcpB,
                bB,
                mu,
                epsvh,
                h2inv,
                gi,
                Hi);
            assert(
                not ToEigen(gi).hasNaN() and not ToEigen(Hi).hasNaN() and
                ToEigen(gi).allFinite() and ToEigen(Hi).allFinite());
        });
}

} // namespace detail

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Iterate");
    auto betaTildeBdf  = fem.bdf.BetaTilde();
    auto betaTildeBdf2 = betaTildeBdf * betaTildeBdf;
    auto xtildeBdf     = fem.bdf.Inertia(0).reshaped(fem.x.rows(), fem.x.cols());
    auto xt            = -xtildeBdf;
    typename contact::MeshDynamics<Scalar, Index>::Params const& contactParams =
        contact.GetParams();
    Scalar h               = betaTildeBdf;
    Scalar h2              = betaTildeBdf2;
    Scalar h2inv           = 1 / h2;
    Scalar rB              = contactParams.mOgcParams.r;
    Scalar kcB             = contactParams.kc;
    Scalar kcpB            = contactParams.kcp;
    Scalar bB              = contactParams.b;
    Scalar epsvh           = contactParams.epsv * h;
    Scalar mu              = contactParams.mu;
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
            // Vertex derivatives
            mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
            // Elastic energy
            detail::AccumulateElasticEnergy<TElasticEnergy>(i, fem, params, gi, Hi);
            gi *= h2;
            Hi *= h2;
            // Contact energy
            mini::SVector<Scalar, 3> xi  = FromEigen(fem.x.col(i).template head<3>());
            mini::SVector<Scalar, 3> xti = FromEigen(xt.col(i).template head<3>());
            detail::AccumulateContactEnergy(
                i,
                xi,
                xti,
                fem.x,
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
                gi,
                Hi);
            // "Kinetic" energy
            Scalar m                         = fem.m(i);
            mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).template head<3>());
            kernels::AddInertiaDerivatives(Scalar(1) /*h2*/, m, xtildei, xi, gi, Hi);
            kernels::AddDamping(Scalar(1) / h /*h*/, xti, xi, params.betaR, gi, Hi);
            kernels::IntegratePositions(gi, Hi, xi, params.detHZero);
            fem.x.col(i) = ToEigen(xi);
        });
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    [[maybe_unused]] Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeSolve");
    auto const xt   = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
    auto& ogcParams = contact.GetParams().mOgcParams;
    ogcParams.rq    = ogcParams.r + (fem.xtilde - xt).colwise().norm().maxCoeff();
    contact.ComputeDisplacementBounds(xt);
    contact.TruncateDisplacedPositions(fem.x, fem.dmask);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Solve");
    for (auto k = 0; k < params.nMaxIters; ++k)
    {
        if (contact.RequiresBoundsComputation())
            contact.ComputeDisplacementBounds(fem.x);
        Iterate<TElasticEnergy>(fem, contact, params);
        contact.TruncateDisplacedPositions(fem.x, fem.dmask);
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
