#ifndef PBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
#define PBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/MeshBoundary.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>

namespace pbat::sim::contact {

/**
 * @brief Multibody Tetrahedral Mesh System
 *
 * Holds a common representation of a multibody system composed of tetrahedral meshes.
 */
template <common::CIndex TIndex = Index>
struct MultibodyTetrahedralMeshSystem
{
    using IndexType = TIndex; ///< Index type used in the multibody system

    MultibodyTetrahedralMeshSystem() = default;
    /**
     * @brief Construct a new Multibody Tetrahedral Mesh System object
     * @tparam TDerivedE Eigen type of the input tetrahedral element matrix
     * @param T_ `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity
     * @param nNodes Number of nodes in the multibody system. If `nNodes == -1`, the number of nodes
     * is inferred from the input tetrahedral mesh `T_`
     */
    template <class TDerivedE>
    MultibodyTetrahedralMeshSystem(
        Eigen::DenseBase<TDerivedE> const& T_,
        Eigen::Index nNodes = Eigen::Index(-1));
    /**
     * @brief Construct a new Multibody Tetrahedral Mesh System object
     * @tparam TDerivedE Eigen type of the input tetrahedral element matrix
     * @param T_ `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity
     * @param nNodes Number of nodes in the multibody system. If `nNodes == -1`, the number of nodes
     * is inferred from the input tetrahedral mesh `T_`
     */
    template <class TDerivedE>
    void Construct(Eigen::DenseBase<TDerivedE> const& T_, Eigen::Index nNodes = Eigen::Index(-1));
    /**
     * @brief Get the number of bodies in the multibody system
     * @return The number of bodies
     */
    Eigen::Index NumberOfBodies() const { return VP.size() - 1; }
    /**
     * @brief Get vertices of body `o`
     * @param o Index of the body
     * @return `|# contact vertices of body o| x 1` indices into mesh vertices
     */
    auto VerticesOf(IndexType o) const { return V.segment(VP[o], VP[o + 1] - VP[o]); }
    /**
     * @brief Get edges of body `o`
     * @param o Index of the body
     * @return `2 x |# contact edges of body o|` edges into mesh vertices
     */
    auto EdgesOf(IndexType o) const { return E.middleCols(EP[o], EP[o + 1] - EP[o]); }
    /**
     * @brief Get triangles of body `o`
     * @param o Index of the body
     * @return `3 x |# contact triangles of body o|` triangles into mesh vertices
     */
    auto TrianglesOf(IndexType o) const { return F.middleCols(FP[o], FP[o + 1] - FP[o]); }
    /**
     * @brief Get tetrahedra of body `o`
     * @param o Index of the body
     * @return `4 x |# contact tetrahedra of body o|` tetrahedra into input tetrahedral mesh `T`
     */
    auto TetrahedraOf(IndexType o) const { return T.middleCols(TP[o], TP[o + 1] - TP[o]); }
    /**
     * @brief Get the body associated with vertex `v`
     * @param v Index of the vertex
     * @return Body index of vertex `v`
     */
    auto BodyOfVertex(IndexType v) const { return CC[V[v]]; }
    /**
     * @brief Get the body associated with edge `e`
     * @param e Index of the edge
     * @return Body index of edge `e`
     */
    auto BodyOfEdge(IndexType e) const { return CC[E(0, e)]; }
    /**
     * @brief Get the body associated with triangle `f`
     * @param f Index of the triangle
     * @return Body index of triangle `f`
     */
    auto BodyOfTriangle(IndexType f) const { return CC[F(0, f)]; }
    /**
     * @brief Get the body associated with tetrahedron `t`
     * @param t Index of the tetrahedron
     * @return Body index of tetrahedron `t`
     */
    auto BodyOfTetrahedron(IndexType t) const { return CC[T(0, t)]; }

    Eigen::Vector<TIndex, Eigen::Dynamic>
        V; ///< `|# contact vertices| x 1` indices into mesh vertices
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
        E; ///< `2 x |# contact edges|` edges into mesh vertices
    Eigen::Matrix<TIndex, 3, Eigen::Dynamic>
        F; ///< `3 x |# contact triangles|` triangles into mesh vertices

    Eigen::Vector<TIndex, Eigen::Dynamic> VP; ///< Prefix sum of vertex pointers into `V`
    Eigen::Vector<TIndex, Eigen::Dynamic> EP; ///< Prefix sum of edge pointers into `E`
    Eigen::Vector<TIndex, Eigen::Dynamic> FP; ///< Prefix sum of triangle pointers into `F`
    Eigen::Vector<TIndex, Eigen::Dynamic>
        TP; ///< Prefix sum of tetrahedron pointers into input tetrahedral mesh `T`

    Eigen::Vector<TIndex, Eigen::Dynamic> CC; ///< `|# vertices| x 1` vertex -> connected components
};

template <common::CIndex TIndex>
template <class TDerivedE>
inline MultibodyTetrahedralMeshSystem<TIndex>::MultibodyTetrahedralMeshSystem(
    Eigen::DenseBase<TDerivedE> const& T_,
    Eigen::Index nNodes)
    : MultibodyTetrahedralMeshSystem<TIndex>()
{
    Construct(T_.derived(), nNodes);
}

template <common::CIndex TIndex>
template <class TDerivedE>
inline void MultibodyTetrahedralMeshSystem<TIndex>::Construct(
    Eigen::DenseBase<TDerivedE> const& T_,
    Eigen::Index nNodes)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyTetrahedralMeshSystem.Construct");
    // TODO: 
    // Implement the construction of the multibody tetrahedral mesh system
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H