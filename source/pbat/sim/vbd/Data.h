#ifndef PBAT_SIM_VBD_DATA_H
#define PBAT_SIM_VBD_DATA_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/graph/Enums.h"

namespace pbat::sim::vbd {

/**
 * @brief VBD simulation configuration
 */
PBAT_API struct Data
{
  public:
    /**
     * @brief Simulation mesh
     *
     * If body map has not been set, all vertices are assumed to belong to the same body.
     *
     * @param X 3x|#vertices| vertex positions
     * @param E 4x|#elements| tetrahedra
     * @return Reference to this
     */
    Data&
    WithVolumeMesh(Eigen::Ref<MatrixX const> const& X, Eigen::Ref<IndexMatrixX const> const& E);
    /**
     * @brief Collision mesh
     * @param V Collision vertices
     * @param F 3x|#collision triangles| collision triangles (on the boundary of T)
     * @return Reference to this
     * @pre WithVolumeMesh() must be called before this
     */
    Data& WithSurfaceMesh(
        Eigen::Ref<IndexVectorX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& F);
    /**
     * @brief Multibody map
     * @param B `|# vertices|` array of body indices s.t. B[i] is the body index of vertex i
     * @return Reference to this
     */
    Data& WithBodies(Eigen::Ref<IndexVectorX const> const& B);
    /**
     * @brief Vertex velocities
     * @param v `3x|# verts|` vertex velocities
     * @return Reference to this
     */
    Data& WithVelocity(Eigen::Ref<MatrixX const> const& v);
    /**
     * @brief Vertex external accelerations
     * @param aext `3x|# verts|` vertex external accelerations
     * @return Reference to this
     */
    Data& WithAcceleration(Eigen::Ref<MatrixX const> const& aext);
    /**
     * @brief Element material parameters
     * @param rhoe `|# elems|` mass densities
     * @param mue `|# elems|` 1st Lame coefficients
     * @param lambdae `|# elems|` 2nd Lame coefficients
     * @return Reference to this
     */
    Data& WithMaterial(
        Eigen::Ref<VectorX const> const& rhoe,
        Eigen::Ref<VectorX const> const& mue,
        Eigen::Ref<VectorX const> const& lambdae);
    /**
     * @brief Set Dirichlet constrained vertices
     * @param dbc Dirichlet constrained vertices
     * @param muD Dirichlet penalty coefficient
     * @param bDbcSorted If false, dbc will be sorted
     * @return Reference to this
     */
    Data& WithDirichletConstrainedVertices(
        IndexVectorX const& dbc,
        Scalar muD      = Scalar(1),
        bool bDbcSorted = false);
    /**
     * @brief Vertex graph coloring strategy to use
     * @param eOrdering Vertex visiting order
     * @param eSelection Color selection strategy
     * @return Reference to this
     */
    Data& WithVertexColoringStrategy(
        graph::EGreedyColorOrderingStrategy eOrdering,
        graph::EGreedyColorSelectionStrategy eSelection);
    /**
     * @brief BCD optimization initialization strategy
     * @param strategy Initialization strategy
     * @return Reference to this
     */
    Data& WithInitializationStrategy(EInitializationStrategy strategy);
    /**
     * @brief Uniform damping coefficient
     * @param kD Damping coefficient
     * @return Reference to this
     */
    Data& WithRayleighDamping(Scalar kD);
    /**
     * @brief Normal and frictional contact parameters
     *
     * Li et al. 2020 \cite li2020ipc
     *
     * @param muC Normal contact penalty
     * @param muF Friction coefficient
     * @param epsv Relative velocity threshold for static to dynamic friction's smooth transition.
     * See \cite li2020ipc.
     * @return Reference to this
     */
    Data& WithContactParameters(Scalar muC, Scalar muF, Scalar epsv);
    /**
     * @brief Active set update frequency
     * @param activeSetUpdateFrequency Active set update frequency
     * @return Reference to this
     */
    Data& WithActiveSetUpdateFrequency(Index activeSetUpdateFrequency);
    /**
     * @brief Numerical zero for hessian pseudo-singularity check
     * @param zero Numerical zero
     * @return Reference to this
     */
    Data& WithHessianDeterminantZeroUnder(Scalar zero);
    /**
     * @brief Use Chebyshev acceleration
     * @param rho Chebyshev acceleration estimated spectral radius
     * @return Reference to this
     */
    Data& WithChebyshevAcceleration(Scalar rho);
    /**
     * @brief Use Trust Region acceleration
     * @param eta Trust Region energy reduction accuracy threshold
     * @param tau Trust Region radius increase factor
     * @param bCurved Use curved accelerated path, otherwise use linear path. Default is true.
     * @return Reference to this
     */
    Data& WithTrustRegionAcceleration(Scalar eta, Scalar tau, bool bCurved = true);
    /**
     * @brief Construct the simulation data
     * @param bValidate Throw on detected ill-formed inputs
     * @return Reference to this
     */
    Data& Construct(bool bValidate = true);

  public:
    /**
     * Simulation mesh
     */
    MatrixX X;      ///< 3x|#verts| FEM nodal positions
    IndexMatrixX E; ///< 4x|#elems| FEM linear tetrahedral elements

    /**
     * Collision mesh
     */
    IndexVectorX B; ///< |#verts| array of body indices
    IndexVectorX V; ///< Collision vertices
    IndexMatrixX F; ///< 3x|#collision triangles| collision triangles (on the boundary of T)
    VectorX XVA;    ///< |#verts| vertex areas (i.e. triangle areas distributed onto vertices for
                    ///< boundary integration)
    VectorX FA;     ///< |#collision triangles| triangle areas

    /**
     * Vertex data
     */
    MatrixX x;    ///< 3x|#verts| vertex positions
    MatrixX v;    ///< 3x|#verts| vertex velocities
    MatrixX aext; ///< 3x|#verts| vertex external accelerations
    VectorX m;    ///< |#verts| vertex masses

    MatrixX xt;     ///< 3x|#verts| previous vertex positions
    MatrixX xtilde; ///< 3x|#verts| inertial target positions
    MatrixX vt;     ///< 3x|#verts| previous vertex velocities

    /**
     * Element data
     */
    VectorX wg;   ///< |#elems| quadrature weights
    MatrixX GP;   ///< |#elem.nodes|x|#dims*#elems| shape function gradients at elems
    VectorX rhoe; ///< |#elems| mass densities
    MatrixX lame; ///< 2x|#elems| Lame coefficients

    /**
     * Vertex-element adjacency graph
     */
    IndexVectorX GVGp;      ///< |#verts+1| prefixes into GVGg
    IndexVectorX GVGe;      ///< |# of vertex-elems edges| element indices s.t.
                            ///< GVGe[k] for GVGp[i] <= k < GVGp[i+1] gives the element index of
                            ///< adjacent to vertex i for the neighbouring elems
    IndexVectorX GVGilocal; ///< |# of vertex-elems edges| local vertex indices s.t.
                            ///< GVGilocal[k] for GVGp[i] <= k < GVGp[i+1] gives the local index of
                            ///< vertex i for the neighbouring elems

    /**
     * Dirichlet boundary conditions
     */
    Scalar muD{1};    ///< Dirichlet penalty coefficient
    IndexVectorX dbc; ///< Dirichlet constrained vertices (sorted)

    /**
     * Parallelization
     */
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

    /**
     * Time integration optimization parameters
     */
    EInitializationStrategy strategy{
        EInitializationStrategy::AdaptivePbat}; ///< BCD optimization initialization strategy
    Scalar kD{0};                               ///< Uniform damping coefficient
    Scalar muC{1e6};                            ///< Uniform collision penalty
    Scalar muF{0.3};                            ///< Uniform friction coefficient
    Scalar epsv{1e-3}; ///< IPC \cite li2020ipc 's relative velocity threshold for static to dynamic
                       ///< friction's smooth transition
    Index mActiveSetUpdateFrequency{1}; ///< Active set update frequency
    Scalar detHZero{1e-7};              ///< Numerical zero for hessian pseudo-singularity check
    EAccelerationStrategy eAcceleration{EAccelerationStrategy::None}; ///< Acceleration strategy

    /**
     * Chebyshev acceleration
     */
    Scalar rho{1};   ///< Chebyshev acceleration estimated spectral radius
    MatrixX xchebm2; ///< 3x|#verts| x^{k-2} used in Chebyshev semi-iterative method
    MatrixX xchebm1; ///< 3x|#verts| x^{k-1} used in Chebyshev semi-iterative method

    /**
     * Trust Region acceleration
     */
    Scalar eta{0.2};    ///< Trust Region energy reduction accuracy threshold
    Scalar tau{2};      ///< Trust Region radius increase factor
    bool bCurved{true}; ///< Use curved accelerated path, otherwise use linear path
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_DATA_H
