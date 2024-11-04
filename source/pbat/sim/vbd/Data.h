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
    Data& VolumeMesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& E);
    Data&
    SurfaceMesh(Eigen::Ref<IndexVectorX const> const& V, Eigen::Ref<IndexMatrixX const> const& F);
    Data& Velocity(Eigen::Ref<MatrixX const> const& v);
    Data& Acceleration(Eigen::Ref<MatrixX const> const& aext);
    Data& Mass(Eigen::Ref<VectorX const> const& m);
    Data& Quadrature(
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GP,
        Eigen::Ref<MatrixX const> const& lame);
    Data& VertexAdjacency(
        Eigen::Ref<IndexVectorX const> const& GVGp,
        Eigen::Ref<IndexVectorX const> const& GVGn,
        Eigen::Ref<IndexVectorX const> const& GVGe,
        Eigen::Ref<IndexVectorX const> const& GVGilocal);
    Data& Partitions(std::vector<std::vector<Index>> const& partitions);
    Data& DirichletConstrainedVertices(IndexVectorX const& dbc);
    Data& InitializationStrategy(EInitializationStrategy strategy);
    Data& RayleighDamping(Scalar kD);
    Data& CollisionPenalty(Scalar kC);
    Data& NumericalZero(Scalar zero);

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

    IndexVectorX GVGp; ///< |#verts+1| prefixes into GVGn
    IndexVectorX
        GVGn; ///< |# of vertex-quad.pt. edges| neighbours s.t.
              ///< GVGn[k] for GVGp[i] <= k < GVGp[i+1] gives the quad.pts. involving vertex i
    IndexVectorX GVGe;      ///< |# of vertex-quad.pt. edges| element indices s.t.
                            ///< GVGe[k] for GVGp[i] <= k < GVGp[i+1] gives the element index of
                            ///< adjacent to vertex i for the neighbouring quad.pt.
    IndexVectorX GVGilocal; ///< |# of vertex-quad.pt. edges| local vertex indices s.t.
                            ///< GVGilocal[k] for GVGp[i] <= k < GVGp[i+1] gives the local index of
                            ///< vertex i for the neighbouring quad.pts.

    IndexVectorX dbc; ///< Dirichlet constrained vertices

    std::vector<std::vector<Index>>
        partitions; ///< partitions[c] gives the c^{th} group of vertices which can all be
                    ///< integrated independently in parallel

    EInitializationStrategy initializationStrategy; ///< BCD optimization initialization strategy
    Scalar kD;                                      ///< Uniform damping coefficient
    Scalar kC;                                      ///< Uniform collision penalty
    Scalar detHZero; ///< Numerical zero for hessian pseudo-singularity check
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_XPBD_DATA_H