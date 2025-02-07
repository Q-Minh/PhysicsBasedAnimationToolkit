#ifndef PBAT_SIM_VBD_DATA_H
#define PBAT_SIM_VBD_DATA_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/graph/Enums.h"

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
     *
     * @param B
     * @return Data&
     */
    Data& WithBodies(Eigen::Ref<IndexVectorX const> const& B);
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
     * @param rhoe |#elems| mass densities
     * @param mue |#elems| 1st Lame coefficients
     * @param lambdae |#elems| 2nd Lame coefficients
     * @return
     */
    Data& WithMaterial(
        Eigen::Ref<VectorX const> const& rhoe,
        Eigen::Ref<VectorX const> const& mue,
        Eigen::Ref<VectorX const> const& lambdae);
    /**
     * @brief
     * @param dbc Dirichlet constrained vertices
     * @param muD Dirichlet penalty coefficient
     * @param bDbcSorted If false, dbc will be sorted
     * @return
     */
    Data& WithDirichletConstrainedVertices(
        IndexVectorX const& dbc,
        Scalar muD      = Scalar(1),
        bool bDbcSorted = false);
    /**
     * @brief
     * @param eOrdering
     * @param eSelection
     * @return
     */
    Data& WithVertexColoringStrategy(
        graph::EGreedyColorOrderingStrategy eOrdering,
        graph::EGreedyColorSelectionStrategy eSelection);
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
     * 
     * @param muC 
     * @param muF 
     * @param epsv 
     * @return Data& 
     */
    Data& WithContactParameters(Scalar muC, Scalar muF, Scalar epsv);
    /**
     * @brief
     *
     * @param activeSetUpdateFrequency
     * @return Data&
     */
    Data& WithActiveSetUpdateFrequency(Index activeSetUpdateFrequency);
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
    MatrixX X;      ///< 3x|#verts| FEM nodal positions
    IndexMatrixX E; ///< 4x|#elems| FEM linear tetrahedral elements
    IndexVectorX B; ///< |#verts| array of body indices
    IndexVectorX V; ///< Collision vertices
    IndexMatrixX F; ///< 3x|#collision triangles| collision triangles (on the boundary of T)

    MatrixX x;    ///< 3x|#verts| vertex positions
    MatrixX v;    ///< 3x|#verts| vertex velocities
    MatrixX aext; ///< 3x|#verts| vertex external accelerations
    VectorX m;    ///< |#verts| vertex masses

    MatrixX xt;      ///< 3x|#verts| previous vertex positions
    MatrixX xtilde;  ///< 3x|#verts| inertial target positions
    MatrixX xchebm2; ///< 3x|#verts| x^{k-2} used in Chebyshev semi-iterative method
    MatrixX xchebm1; ///< 3x|#verts| x^{k-1} used in Chebyshev semi-iterative method
    MatrixX vt;      ///< 3x|#verts| previous vertex velocities

    VectorX wg;   ///< |#elems| quadrature weights
    MatrixX GP;   ///< |#elem.nodes|x|#dims*#elems| shape function gradients at elems
    VectorX rhoe; ///< |#elems| mass densities
    MatrixX lame; ///< 2x|#elems| Lame coefficients

    IndexVectorX GVGp;      ///< |#verts+1| prefixes into GVGg
    IndexVectorX GVGe;      ///< |# of vertex-elems edges| element indices s.t.
                            ///< GVGe[k] for GVGp[i] <= k < GVGp[i+1] gives the element index of
                            ///< adjacent to vertex i for the neighbouring elems
    IndexVectorX GVGilocal; ///< |# of vertex-elems edges| local vertex indices s.t.
                            ///< GVGilocal[k] for GVGp[i] <= k < GVGp[i+1] gives the local index of
                            ///< vertex i for the neighbouring elems

    Scalar muD{1};    ///< Dirichlet penalty coefficient
    IndexVectorX dbc; ///< Dirichlet constrained vertices (sorted)

    graph::EGreedyColorOrderingStrategy eOrdering{
        graph::EGreedyColorOrderingStrategy::LargestDegree}; ///< Vertex graph coloring ordering
                                                             ///< strategy
    graph::EGreedyColorSelectionStrategy eSelection{
        graph::EGreedyColorSelectionStrategy::LeastUsed}; ///< Vertex graph coloring selection
                                                          ///< strategy
    IndexVectorX colors;                                  ///< |#vertices| map of vertex colors
    IndexVectorX Pptr; ///< |#partitions+1| partition pointers, s.t. the range [Pptr[p], Pptr[p+1])
                       ///< indexes into Padj vertices from partition p
    IndexVectorX Padj; ///< Partition vertices

    EInitializationStrategy strategy{
        EInitializationStrategy::AdaptivePbat}; ///< BCD optimization initialization strategy
    Scalar kD{0};                               ///< Uniform damping coefficient
    Scalar muC{1e6};                            ///< Uniform collision penalty
    Scalar muF{0.3};                            ///< Uniform friction coefficient
    Scalar epsv{1e-3}; ///< IPC's relative velocity threshold for static to dynamic friction's
                       ///< smooth transition
    Index mActiveSetUpdateFrequency{1}; ///< Active set update frequency
    Scalar detHZero{1e-7};              ///< Numerical zero for hessian pseudo-singularity check
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_DATA_H
