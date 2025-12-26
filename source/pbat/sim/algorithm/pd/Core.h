/**
 * @file Core.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Core PD API.
 * @version 0.1
 * @date 2025-12-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_PD_CORE_H
#define PBAT_SIM_ALGORITHM_PD_CORE_H

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
#include "pbat/fem/Laplacian.h"
//#include "pbat/sim/algorithm/vbd/Kernels.h"
#include "pbat/sim/contact/MeshDynamics.h"

#include <Eigen/Core>
#include <exception>
#include <fmt/core.h>
#include <tbb/parallel_for.h>

namespace pbat::sim::algorithm::pd {
    using SparseMatrixType = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;
/**
 * @brief Construct a cell-element adjacency graph
 *
 * @param E Element connectivity `|# nodes per element| x |# elements|` array
 * @param nNodes Number of nodes in the mesh
 * @param GTGp `|# cell+1|` prefixes into GTGe
 * @param GTGe `|# of cell-elems adjacencies|` element indices s.t. `GTGe[k] for GTGp[i] <= k <
 * GTGp[i+1]` gives the element `e` adjacent to cell `i`
 * @param GTGilocal `|# of cell-elems adjacencies|` local cell indices s.t. `GTGilocal[k] for
 * GTGp[i] <= k < GTGp[i+1]` gives the element `e` adjacent to cell `i` GTGp[i+1]` gives the local
 * cell index of cell `i` in element `e=GTGe[k]`
 */
PBAT_API void CellElementAdjacencyGraph(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    Eigen::Ref<IndexVectorX> GTGp,
    Eigen::Ref<IndexVectorX> GTGe,
    Eigen::Ref<IndexVectorX> GTGilocal);

/**
 * @brief Compute cell colors using a greedy algorithm
 *
 * @param E `|# nodes per element| x |# elements|` element connectivity array
 * @param nNodes Number of nodes in the mesh
 * @param eOrdering Vertex color ordering strategy
 * @param eSelection Vertex color selection strategy
 * @param colors `|# verts| x 1` Vertex colors
 */
PBAT_API void CellColors(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    graph::EGreedyColorOrderingStrategy eOrdering,
    graph::EGreedyColorSelectionStrategy eSelection,
    Eigen::Ref<IndexVectorX> colors);

/**
 * @brief PD simulation configuration
 * @details See \cite bouaziz2014PD
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
    PBAT_API Params& WithCellElementAdjacencyGraph(
        Eigen::Ref<IndexVectorX const> const& _GTGp,
        Eigen::Ref<IndexVectorX const> const& _GTGe,
        Eigen::Ref<IndexVectorX const> const& _GTGilocal);
    /**
     * @brief Vertex colors
     * @param _colors Vertex colors
     * @return Reference to this
     */
    PBAT_API Params& WithCellColors(Eigen::Ref<IndexVectorX const> const& _colors);
    /**
     * @brief Rayleigh damping coefficient
     * @param _betaR Rayleigh damping coefficient
     * @return Reference to this
     */
    PBAT_API Params& WithDamping(Scalar _betaR);
    /**
     * @brief Maximum number of PD iterations
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
    IndexVectorX GTGp;      ///< `|# cells+1|` prefixes into GVGe
    IndexVectorX GTGe;      ///< `|# of cell-elems adjacencies|` element indices s.t.
                            ///< `GTGe[k] for GTGp[i] <= k < GTGp[i+1]` gives the element `e`
                            ///< adjacent to cell `i`
    IndexVectorX GTGilocal; ///< `|# of cell-elems adjacencies|` local cell indices s.t.
                            ///< `GTGilocal[k] for GTGp[i] <= k < GTGp[i+1]` gives the local cell
                            ///< index of cell `i` in element `e=GTGe[k]`
    // Parallelization
    IndexVectorX colors; ///< `|# cell|` map of cell colors
    IndexVectorX Pptr;   ///< `|# partitions+1|` partition pointers, s.t. the range `[Pptr[p],
                         ///< Pptr[p+1])` indexes into Padj from partition `p`
    IndexVectorX Padj;   ///< `|# cell|` partition vertices

    Scalar betaR{0};     ///< Rayleigh damping coefficient
    Index nMaxIters{25}; ///< Maximum number of PD iterations
    Scalar detHZero{0};  ///< Numerical zero for hessian pseudo-singularity check

    /**
     * @brief Read-write
     */
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xb; ///< `3 x |# nodes|` buffer positions
    SparseMatrixType lhs; // `|3 * # nodes | x | 3 * # nodes |` left-hand side of PD equation: M/h^2 + L
};

//////////////////////////////
///  Function Declaration  ///
//////////////////////////////

/**
 * @brief Initialize PD solve by preparing contact displacement bounds
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
 * @brief One PD minimization step
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
 * @brief Solve FEM elasto dynamics time integration minimization problem using PD
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
 * @brief Integrate FEM elasto dynamics one step using PD as the non-linear solver
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


/////////////////////////////////
///  Function Implementation  ///
/////////////////////////////////


template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.pd.Iterate");
    // get xtilde?
    // per iteration:
        // global solve: equation 10 of PD paper, fully parallelizable
        // local solve: equation 22, parallelizable per dual graph colour (quads)


    
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    [[maybe_unused]] Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.pd.InitializeSolve");
    
    
    using Triplet          = Eigen::Triplet<Scalar, Index>;

    auto constexpr nQuadPts         = fem.egU.size();
    auto constexpr kNodesPerElement = fem.mesh.kNodesPerElement;
    auto constexpr kDims            = fem.mesh.kDims;

    auto const dims = fem.mesh.X.rows();
    auto const nNodes = fem.mesh.X.cols();
    auto const nQuadPts = eg.size();
    auto const GNeg = fem.GNegU;

    std::vector<Triplet> triplets{};
    triplets.reserve(
        static_cast<std::size_t>(kNodesPerElement * kNodesPerElement * nQuadPts * dims));

    auto const numberOfDofs = dims * nNodes;
    params.lhs = SparseMatrixType(numberOfDofs, numberOfDofs);

    auto h_squared_inv = 1 / (fem.bdf.h * fem.bdf.h);

    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = fem.egU(g);
        auto const nodes = fem.mesh.E.col(e);
        auto const w     = fem.wgU(g);

        // Get shape function gradients at this quadrature point
        auto const GP = GNeg.template block<kNodesPerElement, kDims>(0, g * kDims);

        // Compute element Laplacian: -w * GP * GP^T
        auto const Leg = -w * GP * GP.transpose();

        // Add contributions for each dimension
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            auto const m_h_inv = fem.m(i) * h_squared_inv;
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = static_cast<Index>(dims * nodes(i) + d);
                    auto const nj = static_cast<Index>(dims * nodes(j) + d);
                    triplets.emplace_back(ni, nj, Leg(i, j));
                }
            }
            auto ind = nodes(i);
            for (auto d = 0; d < dims; ++d)
            {
                auto const ni = static_cast<Index>(dims * ind + d);
                triplets.emplace_back(ni, ni, fem.m(ind) * h_squared_inv);
            }
        }
    }

    params.lhs.setFromTriplets(triplets.begin(), triplets.end());


    // TODO: Initialize structures for contact when the time comes

    // auto const xt   = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
    // auto& ogcParams = contact.GetParams().mOgcParams;
    // ogcParams.rq    = ogcParams.r + (fem.xtilde - xt).colwise().norm().maxCoeff();
    // contact.ComputeDisplacementBounds(xt);
    // contact.TruncateDisplacedPositions(fem.x, fem.dmask);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.pd.Solve");
    for (auto k = 0; k < params.nMaxIters; ++k)
    {
        // if (contact.RequiresBoundsComputation())
        //     contact.ComputeDisplacementBounds(fem.x);
        Iterate<TElasticEnergy>(fem, contact, params);
        // contact.TruncateDisplacedPositions(fem.x, fem.dmask);
    }
    fem.BackSubstituteIntegratedPositionsIntoVelocities();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.pd.Integrate");
    Solve<TElasticEnergy>(fem, contact, params);
    fem.Step();
}

} // namespace pbat::sim::algorithm::pd

#endif // PBAT_SIM_ALGORITHM_PD_CORE_H
