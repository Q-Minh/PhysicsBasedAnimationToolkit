#ifndef PBAT_SIM_VBD_DATA_H
#define PBAT_SIM_VBD_DATA_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

PBAT_API struct Data
{
  public:
    Data&
    WithVolumeMesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& E);
    Data& WithSurfaceMesh(
        Eigen::Ref<IndexVectorX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& F);
    Data& WithVelocity(Eigen::Ref<MatrixX const> const& v);
    Data& WithAcceleration(Eigen::Ref<MatrixX const> const& aext);
    Data& WithMass(Eigen::Ref<VectorX const> const& m);
    Data& WithQuadrature(
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GP,
        Eigen::Ref<MatrixX const> const& lame);
    Data& WithVertexAdjacency(
        Eigen::Ref<IndexVectorX const> const& GVGp,
        Eigen::Ref<IndexVectorX const> const& GVGg,
        Eigen::Ref<IndexVectorX const> const& GVGe,
        Eigen::Ref<IndexVectorX const> const& GVGilocal);
    Data& WithPartitions(std::vector<std::vector<Index>> const& partitions);
    Data& WithDirichletConstrainedVertices(IndexVectorX const& dbc, bool bDbcSorted = true);
    Data& WithInitializationStrategy(EInitializationStrategy strategy);
    Data& WithRayleighDamping(Scalar kD);
    Data& WithCollisionPenalty(Scalar kC);
    Data& WithHessianDeterminantZeroUnder(Scalar zero);
    Data& Construct(bool bValidate = true);

  public:
    IndexVectorX V; ///< Collision vertices
    IndexMatrixX F; ///< Collision triangles (on the boundary of T)
    IndexMatrixX T; ///< Tetrahedra

    MatrixX x;    ///< Vertex positions
    MatrixX v;    ///< Vertex velocities
    MatrixX aext; ///< Vertex external accelerations
    VectorX m;    ///< Vertex masses

    MatrixX xt;      ///< Previous vertex positions
    MatrixX xtilde;  ///< Inertial target for vertex positions
    MatrixX xchebm2; ///< x^{k-2} used in Chebyshev semi-iterative method
    MatrixX xchebm1; ///< x^{k-1} used in Chebyshev semi-iterative method
    MatrixX vt;      ///< Previous vertex velocities

    VectorX wg;   ///< |#quad.pts.| quadrature weights
    MatrixX GP;   ///< |#elem.nodes|x|#dims*#quad.pts.| shape function gradients at quad. pts.
    MatrixX lame; ///< 2x|#quad.pts.| Lame coefficients

    IndexVectorX GVGp; ///< |#verts+1| prefixes into GVGg
    IndexVectorX
        GVGg; ///< |# of vertex-quad.pt. edges| neighbours s.t.
              ///< GVGg[k] for GVGp[i] <= k < GVGp[i+1] gives the quad.pts. involving vertex i
    IndexVectorX GVGe;      ///< |# of vertex-quad.pt. edges| element indices s.t.
                            ///< GVGe[k] for GVGp[i] <= k < GVGp[i+1] gives the element index of
                            ///< adjacent to vertex i for the neighbouring quad.pt.
    IndexVectorX GVGilocal; ///< |# of vertex-quad.pt. edges| local vertex indices s.t.
                            ///< GVGilocal[k] for GVGp[i] <= k < GVGp[i+1] gives the local index of
                            ///< vertex i for the neighbouring quad.pts.

    IndexVectorX dbc; ///< Dirichlet constrained vertices (sorted)

    std::vector<std::vector<Index>>
        partitions; ///< partitions[c] gives the c^{th} group of vertices which can all be
                    ///< integrated independently in parallel

    EInitializationStrategy strategy{
        EInitializationStrategy::AdaptivePbat}; ///< BCD optimization initialization strategy
    Scalar kD{0};                               ///< Uniform damping coefficient
    Scalar kC{1};                               ///< Uniform collision penalty
    Scalar detHZero{1e-7}; ///< Numerical zero for hessian pseudo-singularity check
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_XPBD_DATA_H