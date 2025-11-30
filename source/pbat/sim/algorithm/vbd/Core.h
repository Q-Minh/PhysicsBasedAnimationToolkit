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
#include "pbat/geometry/HalfEdges.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Enums.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/algorithm/common/Common.h"
#include "pbat/sim/algorithm/vbd/Kernels.h"
#include "pbat/sim/contact/MeshDynamics.h"
#include "pbat/sim/contact/Potentials.h"

#include <Eigen/Core>
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
};

/**
 * @brief Initialize VBD minimization solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param meshDynamics Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params);

/**
 * @brief One VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param meshDynamics Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param meshDynamics Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params);

/**
 * @brief Integrate FEM elasto dynamics one step using VBD as the non-linear solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param meshDynamics Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params);

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeSolve");
    // Run collision detection with environment and get vertex displacement bounds
    // meshDynamics.UpdateEnvironmentContactConstraints(fem.x);
    // Truncate displacement
    // auto xt = fem.bdf.CurrentState(0).reshaped(fem.x.rows(), fem.x.cols());
    // tbb::parallel_for(Index(0), fem.x.cols(), [&](Index i) {
    //     Index const v = meshDynamics.mMeshes.GXV(i);
    //     if (v < 0)
    //         return;
    //     Scalar const db = meshDynamics.mMeshSdfContact.mVertexDisplacementBounds(v);
    //     Eigen::Vector<Scalar, 3> const dxi = fem.x.col(i) - xt.col(i);
    //     Scalar const dxiNorm               = dxi.norm();
    //     if (dxiNorm > db)
    //         fem.x.col(i) = xt.col(i) + (db / dxiNorm) * dxi;
    // });
    // // Initialize AL contact constraints
    // meshDynamics.PrepareEnvironmentContactsForDualIteration();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Iterate");
    auto betaTildeBdf  = fem.bdf.BetaTilde();
    auto betaTildeBdf2 = betaTildeBdf * betaTildeBdf;
    auto xtildeBdf     = fem.bdf.Inertia(0).reshaped(fem.x.rows(), fem.x.cols());
    contact::potentials::LaggedFriction friction;
    // Scalar const epsvh     = meshDynamics.mEnvContactDynamicsParams.epsv * fem.bdf.BetaTilde();
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
            // Vertex derivatives
            mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
            // Elastic energy
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
                Psi.GradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
                kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
                kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
            }
            // Environment contact energy
            mini::SVector<Scalar, 3> xi  = FromEigen(fem.x.col(i).template head<3>());
            mini::SVector<Scalar, 3> xti = -FromEigen(xtildeBdf.col(i).template head<3>());
            // Index const vi               = meshDynamics.mMeshes.GXV(i);
            // bool const bIsSurfaceVertex  = vi >= 0;
            // if (bIsSurfaceVertex)
            // {
            //     // Loop over triangles incident on i for augmented Lagrangian derivatives
            //     IndexVectorX const& heAdj = meshDynamics.mMeshes.GVHEadj;
            //     Index const heAdjOffset   = meshDynamics.mMeshes.GVHEp(i);
            //     Index const heAdjEnd      = meshDynamics.mMeshes.GVHEp(i + 1);
            //     for (Index he : heAdj(Eigen::seqN(heAdjOffset, heAdjEnd - heAdjOffset)))
            //     {
            //         Index f = geometry::FaceOfHalfEdge(he);
            //         if (not meshDynamics.mMeshSdfContact.mTriangleContactMask(f))
            //             continue;
            //         Eigen::Vector<Index, 3> const finds = meshDynamics.mMeshes.F.col(f);
            //         Eigen::Vector<Scalar, 2> const uv =
            //             meshDynamics.mMeshSdfContact.mTriangleContactPoints.col(f);
            //         Eigen::Matrix<Scalar, 3, 3> const xf = fem.x(Eigen::placeholders::all,
            //         finds); auto ilocal = /*(i==finds(0))*0 + */ (i == finds(1)) * 1 + (i ==
            //         finds(2)) * 2; sim::contact::MeshDynamics::EnvironmentContact& C =
            //         meshDynamics.CF[f]; Eigen::Vector<Scalar, 3> const xc =
            //             (1 - uv(0) - uv(1)) * xf.col(0) + uv(0) * xf.col(1) + uv(1) * xf.col(2);
            //         C.Eval(xc);
            //         // AL gradient + hessian
            //         bool const bIsPenetrating = C.C < 0;
            //         Scalar alpha = (ilocal == 0) * (1 - uv(0) - uv(1)) + (ilocal == 1) * uv(0) +
            //                        (ilocal == 2) * uv(1);
            //         Scalar lambdakn = C.ForceEstimate();
            //         Eigen::Vector<Scalar, 3> gradAL =
            //             -(bIsPenetrating * alpha) * (C.B.col(0) * lambdakn);
            //         Eigen::Matrix<Scalar, 3, 3> hessAL =
            //             (bIsPenetrating * alpha * alpha) *
            //             (C.k * (C.B.col(0) * C.B.col(0).transpose()));
            //         gi += FromEigen(gradAL);
            //         Hi += FromEigen(hessAL);
            //         // Tangential friction forces
            //         if (bIsPenetrating)
            //         {
            //             Eigen::Matrix<Scalar, 3, 3> const xtf =
            //                 xtildeBdf(Eigen::placeholders::all, finds);
            //             Eigen::Vector<Scalar, 3> const xct = (1 - uv(0) - uv(1)) * xtf.col(0) +
            //                                                  uv(0) * xtf.col(1) +
            //                                                  uv(1) * xtf.col(2);
            //             auto const Tk               = C.B.rightCols<2>();
            //             Eigen::Vector<Scalar, 2> uk = Tk.transpose() * (xc - xct);
            //             Eigen::Vector<Scalar, 2> gkf;
            //             Eigen::Matrix<Scalar, 2, 2> Hkf;
            //             auto gkfmini = FromEigen(gkf);
            //             auto Hkfmini = FromEigen(Hkf);
            //             friction
            //                 .GradAndHessian(FromEigen(uk), C.mu, lambdakn, epsvh, gkfmini,
            //                 Hkfmini);
            //             Eigen::Vector<Scalar, 3> gkfx    = Tk * gkf;
            //             Eigen::Matrix<Scalar, 3, 3> Hkfx = Tk * Hkf * Tk.transpose();
            //             gi += FromEigen(gkfx);
            //             Hi += FromEigen(Hkfx);
            //         }
            //     }
            //     // Loop over half-edges incident on i for augmented Lagrangian derivatives
            //     for (Index he : heAdj(Eigen::seqN(heAdjOffset, heAdjEnd - heAdjOffset)))
            //     {
            //         if (not meshDynamics.mMeshSdfContact.mHalfEdgeContactMask(he))
            //             continue;
            //         Scalar const u = meshDynamics.mMeshSdfContact.mHalfEdgeContactPoints(he);
            //         Index const ei = geometry::IncomingVertex(meshDynamics.mMeshes.F, he);
            //         Index const ej = geometry::OutgoingVertex(meshDynamics.mMeshes.F, he);
            //         Eigen::Vector<Scalar, 3> const xei = fem.x.col(ei);
            //         Eigen::Vector<Scalar, 3> const xej = fem.x.col(ej);
            //         auto ilocal                        = /*(i == ei) * 0 + */ (i == ej) * 1;
            //         sim::contact::MeshDynamics::EnvironmentContact& C = meshDynamics.CHE[he];
            //         Eigen::Vector<Scalar, 3> const xc                 = (1 - u) * xei + u * xej;
            //         C.Eval(xc);
            //         // AL gradient + hessian
            //         bool const bIsPenetrating = C.C < 0;
            //         Scalar alpha              = (ilocal == 0) * (1 - u) + (ilocal == 1) * u;
            //         Scalar const lambdakn     = C.ForceEstimate();
            //         Eigen::Vector<Scalar, 3> gradAL =
            //             -(bIsPenetrating * alpha) * (C.B.col(0) * lambdakn);
            //         Eigen::Matrix<Scalar, 3, 3> hessAL =
            //             (bIsPenetrating * alpha * alpha) *
            //             (C.k * (C.B.col(0) * C.B.col(0).transpose()));
            //         gi += FromEigen(gradAL);
            //         Hi += FromEigen(hessAL);
            //         // Tangential friction forces
            //         if (bIsPenetrating)
            //         {
            //             Eigen::Vector<Scalar, 3> const xtei = xtildeBdf.col(ei);
            //             Eigen::Vector<Scalar, 3> const xtej = xtildeBdf.col(ej);
            //             Eigen::Vector<Scalar, 3> const xct  = (1 - u) * xtei + u * xtej;
            //             auto const Tk                       = C.B.rightCols<2>();
            //             Eigen::Vector<Scalar, 2> uk         = Tk.transpose() * (xc - xct);
            //             Eigen::Vector<Scalar, 2> gkf;
            //             Eigen::Matrix<Scalar, 2, 2> Hkf;
            //             auto gkfmini = FromEigen(gkf);
            //             auto Hkfmini = FromEigen(Hkf);
            //             friction
            //                 .GradAndHessian(FromEigen(uk), C.mu, lambdakn, epsvh, gkfmini,
            //                 Hkfmini);
            //             Eigen::Vector<Scalar, 3> gkfx    = Tk * gkf;
            //             Eigen::Matrix<Scalar, 3, 3> Hkfx = Tk * Hkf * Tk.transpose();
            //             gi += FromEigen(gkfx);
            //             Hi += FromEigen(Hkfx);
            //         }
            //     }
            //     // Check i itself for augmented Lagrangian derivatives
            //     if (meshDynamics.mMeshSdfContact.mVertexContactMask(vi))
            //     {
            //         sim::contact::MeshDynamics::EnvironmentContact& C = meshDynamics.CV[vi];
            //         C.Eval(ToEigen(xi));
            //         // AL gradient + hessian
            //         bool const bIsPenetrating = C.C < 0;
            //         Scalar const lambdakn     = C.ForceEstimate();
            //         Eigen::Vector<Scalar, 3> gradAL =
            //             -static_cast<int>(bIsPenetrating) * (C.B.col(0) * lambdakn);
            //         Eigen::Matrix<Scalar, 3, 3> hessAL =
            //             (bIsPenetrating * C.k) * (C.B.col(0) * C.B.col(0).transpose());
            //         gi += FromEigen(gradAL);
            //         Hi += FromEigen(hessAL);
            //         // Tangential friction forces
            //         if (bIsPenetrating)
            //         {
            //             auto const Tk               = C.B.rightCols<2>();
            //             mini::SVector<Scalar, 2> uk = FromEigen(Tk).Transpose() * (xi - xti);
            //             Eigen::Vector<Scalar, 2> gkf;
            //             Eigen::Matrix<Scalar, 2, 2> Hkf;
            //             auto gkfmini = FromEigen(gkf);
            //             auto Hkfmini = FromEigen(Hkf);
            //             friction.GradAndHessian(uk, C.mu, lambdakn, epsvh, gkfmini, Hkfmini);
            //             Eigen::Vector<Scalar, 3> gkfx    = Tk * gkf;
            //             Eigen::Matrix<Scalar, 3, 3> Hkfx = Tk * Hkf * Tk.transpose();
            //             gi += FromEigen(gkfx);
            //             Hi += FromEigen(Hkfx);
            //         }
            //     }
            // }
            // dt2 scale potential energies' derivatives
            // "Kinetic" energy
            Scalar m                         = fem.m(i);
            mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).template head<3>());
            kernels::AddInertiaDerivatives(betaTildeBdf2, m, xtildei, xi, gi, Hi);
            kernels::AddDamping(betaTildeBdf, xti, xi, params.betaR, gi, Hi);
            kernels::IntegratePositions(gi, Hi, xi, params.detHZero);
            fem.x.col(i) = ToEigen(xi);
        });
    }
    // Update dual variables every VBD iteration
    // meshDynamics.DualUpdateEnvironmentContacts(fem.x);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Solve");
    InitializeSolve<TElasticEnergy>(fem, meshDynamics, params);
    for (Index k = 0; k < params.nMaxIters; ++k)
    {
        Iterate<TElasticEnergy>(fem, meshDynamics, params);
    }
    fem.BackSubstituteIntegratedPositionsIntoVelocities();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics& meshDynamics,
    Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Integrate");
    Solve<TElasticEnergy>(fem, meshDynamics, params);
    fem.Step();
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CORE_H
