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
#include "pbat/fem/Tetrahedron.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Enums.h"
#include "pbat/io/Archive.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/algorithm/vbd/Kernels.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

#include <Eigen/Core>
#include <optional>
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
 * @details See @cite anka2024vbd
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
     * @brief BCD optimization initialization strategy
     * @param _strategy Initialization strategy
     * @return Reference to this
     */
    PBAT_API Params& WithInitializationStrategy(EInitializationStrategy _strategy);
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
    // Time integration optimization parameters
    EInitializationStrategy strategy{
        EInitializationStrategy::Inertia}; ///< BCD optimization initialization strategy
    Scalar detHZero{1e-7};                 ///< Numerical zero for hessian pseudo-singularity check
    Index nMaxIters{25};                   ///< Maximum number of VBD iterations
};

/**
 * @brief Finite element elasto dynamics problem for VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
using FemElastoDynamics =
    dynamics::FemElastoDynamics<fem::Tetrahedron<1>, 3, TElasticEnergy, Scalar, Index>;

/**
 * @brief Initialize VBD minimization solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, Params const& params);

/**
 * @brief One VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param ac Optional archive to serialize to
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    std::optional<io::Archive> ac = std::nullopt);

/**
 * @brief Back-substitute integrated positions into velocities
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void BackSubstituteIntegratedPositionsIntoVelocities(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params);

/**
 * @brief Integrate FEM elasto dynamics one step using VBD as the non-linear solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param ac Optional archive to serialize to
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    std::optional<io::Archive> ac = std::nullopt);

/**
 * @brief Serialize solver iteration data
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param k Current solver iteration
 * @param archive Archive to serialize to
 * @param bPostSolve Whether this is after the solve. If true, velocities are also serialized.
 * Default is false.
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void SerializeSolverIteration(
    FemElastoDynamics<TElasticEnergy>& fem,
    Index k,
    io::Archive& archive,
    bool bPostSolve = false);

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeSolve");
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::ToEigen;
    // NOTE:
    // We should make this initialization adapt to higher-order BDF schemes as well!
    // In this case, we would have
    // "xt" = -fem.bdf.Inertia(0), and
    // "vt" = -fem.bdf.Inertia(1),
    // and instead of the h, we would have fem.bdf.BetaTilde()
    // and h^2 would be fem.bdf.BetaTilde()^2.
    auto aext             = fem.aext();
    auto xt               = fem.bdf.CurrentState(0).reshaped(fem.x.rows(), fem.x.cols());
    auto vt               = fem.bdf.CurrentState(1).reshaped(fem.v.rows(), fem.v.cols());
    auto free             = fem.FreeNodes();
    auto const nFreeVerts = free.size();
    auto h                = fem.bdf.TimeStep();
    auto h2               = h * h;
    tbb::parallel_for(Index(0), nFreeVerts, [&](Index fi) {
        auto i = free(fi);
        auto x = kernels::InitialPositionsForSolve(
            FromEigen(xt.col(i).template head<3>()),
            FromEigen(vt.col(i).template head<3>()),
            FromEigen(fem.v.col(i).template head<3>()),
            FromEigen(aext.col(i).template head<3>()),
            h,
            h2,
            params.strategy);
        fem.x.col(i) = ToEigen(x);
    });
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Iterate");
    auto h  = fem.bdf.TimeStep();
    auto h2 = h * h;
    // NOTE:
    // If we want to support damping in the future, it would be nice to make it adapt
    // to higher-order BDF schemes. We have all the tools necessary in the Bdf class.
    // auto xt                = fem.bdf.CurrentState(0).reshaped(fem.x.rows(), fem.x.cols());
    // auto vt                = fem.bdf.CurrentState(1).reshaped(fem.v.rows(), fem.v.cols());
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
            auto begin = params.GVGp(i);
            auto end   = params.GVGp(i + 1);
            // Elastic energy
            mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
            for (auto n = begin; n < end; ++n)
            {
                auto ilocal = params.GVGilocal(n);
                auto e      = params.GVGe(n);
                auto lamee  = fem.lamegU.col(e);
                auto wg     = fem.wgU(e);
                auto ti     = fem.mesh.E.col(e);
                mini::SMatrix<Scalar, 4, 3> GPe =
                    FromEigen(fem.GNegU.template block<4, 3>(0, e * 3));
                mini::SMatrix<Scalar, 3, 4> xe =
                    FromEigen(fem.x(Eigen::placeholders::all, ti).template block<3, 4>(0, 0));
                mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
                TElasticEnergy Psi{};
                mini::SVector<Scalar, 9> gF;
                mini::SMatrix<Scalar, 9, 9> HF;
                Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
                kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
                kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
            }
            // "Kinetic" energy
            Scalar m = fem.m(i);
            // mini::SVector<Scalar, 3> xti     = FromEigen(xt.col(i).template head<3>());
            mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).template head<3>());
            mini::SVector<Scalar, 3> xi      = FromEigen(fem.x.col(i).template head<3>());
            // kernels::AddDamping(h, xti, xi, Scalar(0) /*Rayleigh damping*/, gi, Hi);
            kernels::AddInertiaDerivatives(h2, m, xtildei, xi, gi, Hi);
            // Update vertex position
            kernels::IntegratePositions(gi, Hi, xi, params.detHZero);
            fem.x.col(i) = ToEigen(xi);
        });
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    std::optional<io::Archive> ac)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Solve");
    std::optional<io::Archive> group;
    if (ac)
    {
        group = ac->GetOrCreateGroup("pbat.sim.algorithm.vbd.Solve");
    }
    InitializeSolve<TElasticEnergy>(fem, params);
    for (Index k = 0; k < params.nMaxIters; ++k)
    {
        if (group)
        {
            SerializeSolverIteration(fem, k, *group);
        }
        Iterate<TElasticEnergy>(fem, params);
    }
    BackSubstituteIntegratedPositionsIntoVelocities<TElasticEnergy>(fem, params);
    if (group)
    {
        SerializeSolverIteration(fem, params.nMaxIters, *group, true /* bPostSolve */);
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void BackSubstituteIntegratedPositionsIntoVelocities(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params)
{
    auto x                                = fem.x.reshaped();
    auto xt                               = fem.bdf.CurrentState(0);
    auto dt                               = fem.bdf.TimeStep();
    auto dofs                             = fem.FreeDofs();
    auto v                                = (x(dofs) - xt(dofs)) / dt;
    auto free                             = fem.FreeNodes();
    fem.v(Eigen::placeholders::all, free) = v.reshaped(fem.v.rows(), free.size());
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    std::optional<io::Archive> ac)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Integrate");
    fem.SetupTimeIntegrationOptimization();
    std::optional<io::Archive> group;
    if (ac)
    {
        group = ac->GetOrCreateGroup("pbat.sim.algorithm.vbd.Integrate");
        fem.Serialize(*group);
    }
    Solve<TElasticEnergy>(fem, params, group);
    fem.Step();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void SerializeSolverIteration(
    FemElastoDynamics<TElasticEnergy>& fem,
    Index k,
    io::Archive& archive,
    bool bPostSolve)
{
    Scalar f     = fem.Objective();
    Scalar gnorm = fem.Gradient().norm();
    // NOTE: Use 6 decimal positions for iteration index (allows up to 999999 iterations)
    io::Archive iter = archive[fmt::format("{:06d}", k)];
    iter.WriteData("x", fem.x);
    iter.WriteMetaData("f", f);
    iter.WriteMetaData("gnorm", gnorm);
    if (bPostSolve)
        iter.WriteData("v", fem.v);
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CORE_H
