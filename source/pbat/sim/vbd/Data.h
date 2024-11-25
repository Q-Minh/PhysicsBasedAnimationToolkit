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
    /**
     * @brief
     * @param X 3x|#vertices| vertex positions
     * @param E 4x|#elements| tetrahedra
     * @return
     */
    Data&
    WithVolumeMesh(Eigen::Ref<MatrixX const> const& X, Eigen::Ref<IndexMatrixX const> const& E);
    /**
     * @brief
     * @param V Collision vertices
     * @param F 3x|#collision triangles| collision triangles (on the boundary of T)
     * @return
     */
    Data& WithSurfaceMesh(
        Eigen::Ref<IndexVectorX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& F);
    /**
     * @brief
     * @param v 3x|#verts| vertex velocities
     * @return
     */
    Data& WithVelocity(Eigen::Ref<MatrixX const> const& v);
    /**
     * @brief
     * @param aext 3x|#verts| vertex external accelerations
     * @return
     */
    Data& WithAcceleration(Eigen::Ref<MatrixX const> const& aext);
    /**
     * @brief
     * @param m 3x|#verts| vertex masses
     * @return
     */
    Data& WithMass(Eigen::Ref<VectorX const> const& m);
    /**
     * @brief
     * @param wg |#quad.pts.| quadrature weights
     * @param GP |#elem.nodes|x|#dims*#quad.pts.| shape function gradients at quad. pts.
     * @param lame 2x|#quad.pts.| Lame coefficients
     * @return
     */
    Data& WithQuadrature(
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GP,
        Eigen::Ref<MatrixX const> const& lame);
    /**
     * @brief
     * @param GVGp |#verts+1| prefixes into GVGg
     * @param GVGg |# of vertex-quad.pt. edges| neighbours s.t. GVGg[k] for GVGp[i] <= k < GVGp[i+1]
     * gives the quad.pts. involving vertex i
     * @param GVGe |# of vertex-quad.pt. edges| element indices s.t. GVGe[k] for GVGp[i] <= k <
     * GVGp[i+1] gives the element index of adjacent to vertex i for the neighbouring quad.pt.
     * @param GVGilocal |# of vertex-quad.pt. edges| local vertex indices s.t. GVGilocal[k] for
     * GVGp[i] <= k < GVGp[i+1] gives the local index of vertex i for the neighbouring quad.pts.
     * @return
     */
    Data& WithVertexAdjacency(
        Eigen::Ref<IndexVectorX const> const& GVGp,
        Eigen::Ref<IndexVectorX const> const& GVGg,
        Eigen::Ref<IndexVectorX const> const& GVGe,
        Eigen::Ref<IndexVectorX const> const& GVGilocal);
    /**
     * @brief
     * @param Pptr |#partitions+1| partition pointers, s.t. the range [Pptr[p], Pptr[p+1]) indexes
     * into Padj vertices from partition p
     * @param Padj Partition vertices
     * @return
     */
    Data& WithPartitions(
        Eigen::Ref<IndexVectorX const> const& Pptr,
        Eigen::Ref<IndexVectorX const> const& Padj);
    /**
     * @brief
     * @param dbc Dirichlet constrained vertices
     * @param bDbcSorted If false, dbc will be sorted
     * @return
     */
    Data& WithDirichletConstrainedVertices(IndexVectorX const& dbc, bool bDbcSorted = true);
    /**
     * @brief
     * @param strategy
     * @return
     */
    Data& WithInitializationStrategy(EInitializationStrategy strategy);
    /**
     * @brief
     * @param kD
     * @return
     */
    Data& WithRayleighDamping(Scalar kD);
    /**
     * @brief
     * @param kC
     * @return
     */
    Data& WithCollisionPenalty(Scalar kC);
    /**
     * @brief
     * @param zero
     * @return
     */
    Data& WithHessianDeterminantZeroUnder(Scalar zero);
    /**
     * @brief
     * @param bValidate Throw on detected ill-formed inputs
     * @return
     */
    Data& Construct(bool bValidate = true);

  public:
    IndexVectorX V; ///< Collision vertices
    IndexMatrixX F; ///< 3x|#collision triangles| collision triangles (on the boundary of T)
    IndexMatrixX T; ///< 4x|#elements| tetrahedra

    MatrixX x;    ///< 3x|#verts| vertex positions
    MatrixX v;    ///< 3x|#verts| vertex velocities
    MatrixX aext; ///< 3x|#verts| vertex external accelerations
    VectorX m;    ///< 3x|#verts| vertex masses

    MatrixX xt;      ///< 3x|#verts| previous vertex positions
    MatrixX xtilde;  ///< 3x|#verts| inertial target positions
    MatrixX xchebm2; ///< 3x|#verts| x^{k-2} used in Chebyshev semi-iterative method
    MatrixX xchebm1; ///< 3x|#verts| x^{k-1} used in Chebyshev semi-iterative method
    MatrixX vt;      ///< 3x|#verts| previous vertex velocities

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

    IndexVectorX Pptr; ///< |#partitions+1| partition pointers, s.t. the range [Pptr[p], Pptr[p+1])
                       ///< indexes into Padj vertices from partition p
    IndexVectorX Padj; ///< Partition vertices

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