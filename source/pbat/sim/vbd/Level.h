#ifndef PBAT_SIM_VBD_LEVEL_H
#define PBAT_SIM_VBD_LEVEL_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Level
{
    struct Energy
    {
        /**
         * @brief
         *
         * @param wg |#quad.pts.| array of quadrature weights
         * @param sg |#quad.pts.| boolean array identifying singular quadrature points
         * @return Energy&
         */
        Energy& WithQuadrature(
            Eigen::Ref<VectorX const> const& wg,
            Eigen::Ref<Eigen::Vector<bool, Eigen::Dynamic> const> const& sg);
        /**
         * @brief
         *
         * @param GVGp |#verts + 1| prefix array of vertex-quad.pt. adjacency graph
         * @param GVGg |#vertex-quad.pt. adjacencies| array of adjacency graph edges
         * @param GVGe |#vertex-quad.pt. adjacencies| array of element indices associated with
         * adjacency graph edges
         * @param GVGilocal |#vertex-quad.pt. adjacencies| array of local vertex indices associated
         * with adjacency graph edges
         * @return Energy&
         */
        Energy& WithAdjacency(
            Eigen::Ref<IndexVectorX const> const& GVGp,
            Eigen::Ref<IndexVectorX const> const& GVGg,
            Eigen::Ref<IndexVectorX const> const& GVGe,
            Eigen::Ref<IndexVectorX const> const& GVGilocal);
        /**
         * @brief
         *
         * @param dwg |#Dirichlet quad.pts.| array of Dirichlet quadrature weights
         * @param dNcg 4x|#Dirichlet quad.pts.| array of coarse cage shape functions at Dirichlet
         * quadrature points
         * @param dxg 3x|#Dirichlet quad.pts.| array of Dirichlet boundary conditions at Dirichlet
         * quadrature points
         * @return Energy&
         */
        Energy& WithDirichletEnergy(
            Eigen::Ref<VectorX const> const& dwg,
            Eigen::Ref<MatrixX const> const& dNcg,
            Eigen::Ref<MatrixX const> const& dxg);
        /**
         * @brief
         *
         * @param GVDGp |#verts + 1| prefix array of vertex-Dirichlet quad.pt. adjacency graph
         * @param GVDGg |#vertex-Dirichlet quad.pt. adjacencies| array of Dirichlet quad. adjacency
         * graph edges
         * @param GVDGe |#vertex-Dirichlet quad.pt. adjacencies| array of element indices associated
         * with Dirichlet quad. adjacency graph edges
         * @param GVDGilocal |#vertex-Dirichlet quad.pt. adjacencies| array of local vertex indices
         * associated with Dirichlet quad. adjacency graph edges
         * @return Energy&
         */
        Energy& WithDirichletAdjacency(
            Eigen::Ref<IndexVectorX const> const& GVDGp,
            Eigen::Ref<IndexVectorX const> const& GVDGg,
            Eigen::Ref<IndexVectorX const> const& GVDGe,
            Eigen::Ref<IndexVectorX const> const& GVDGilocal);
        /**
         * @brief
         *
         * @param rhog |#quad.pts.| array of mass densities at quadrature points
         * @param Ncg 4x|#quad.pts.| array of coarse cage element shape functions at quadrature
         * points
         * @return Energy&
         */
        Energy& WithKineticEnergy(
            Eigen::Ref<VectorX const> const& rhog,
            Eigen::Ref<MatrixX const> const& Ncg);
        /**
         * @brief
         *
         * @param mug |#quad.pts.| array of first Lame coefficients at quadrature points
         * @param lambdag |#quad.pts.| array of second Lame coefficients at quadrature points
         * @param erg 4x|#quad.pts.| array of coarse element indices containing vertices of root
         * level element embedding quadrature point g
         * @param Nrg 4x|4*#quad.pts.| array of coarse cage element shape functions at root level
         * elements' 4 vertices associated with quadrature points
         * @param GNfg 4x|3*#quad.pts.| array of root level element shape function gradients at
         * quadrature points
         * @param GNcg 4x|3*#quad.pts.| array of coarse cage element shape function gradients at
         * quadrature points
         * @return Energy&
         */
        Energy& WithPotentialEnergy(
            Eigen::Ref<VectorX const> const& mug,
            Eigen::Ref<VectorX const> const& lambdag,
            Eigen::Ref<IndexMatrixX const> const& erg,
            Eigen::Ref<MatrixX const> const& Nrg,
            Eigen::Ref<MatrixX const> const& GNfg,
            Eigen::Ref<MatrixX const> const& GNcg);
        /**
         * @brief
         *
         * @param bValidate Throw on ill-formed input or not
         * @return Energy&
         */
        Energy& Construct(bool bValidate = true);

        /**
         * @brief Kinetic energy
         */

        MatrixX xtildeg; ///< 3x|#quad.pts.| array of inertial target positions at quadrature points
        VectorX rhog;    ///< |#quad.pts.| array of mass densities at quadrature points
        MatrixX Ncg; ///< 4x|#quad.pts.| array of coarse cage element shape functions at quadrature
                     ///< points

        /**
         * @brief Elastic energy
         */

        VectorX mug;      ///< |#quad.pts.| array of first Lame coefficients at quadrature points
        VectorX lambdag;  ///< |#quad.pts.| array of second Lame coefficients at quadrature points
        IndexMatrixX erg; ///< 4x|#quad.pts.| array of coarse element indices containing vertices of
                          ///< root level element embedding quadrature point g
        MatrixX Nrg;      ///< 4x|4*#quad.pts.| array of coarse cage element shape functions at root
                          ///< level elements' 4 vertices associated with quadrature points
        MatrixX GNfg; ///< 4x|3*#quad.pts.| array of root level element shape function gradients at
                      ///< quadrature points
        MatrixX GNcg; ///< 4x|3*#quad.pts.| array of coarse cage element shape function gradients at
                      ///< quadrature points

        /**
         * @brief Domain integration
         */

        Eigen::Vector<bool, Eigen::Dynamic>
            sg;     ///< |#quad.pts.| boolean array identifying singular quadrature points
        VectorX wg; ///< |#quad.pts.| array of quadrature weights

        IndexVectorX GVGp; ///< |#verts + 1| prefix array of vertex-quad.pt. adjacency graph
        IndexVectorX GVGg; ///< |#vertex-quad.pt. adjacencies| array of adjacency graph edges
        IndexVectorX GVGe; ///< |#vertex-quad.pt. adjacencies| array of element indices associated
                           ///< with adjacency graph edges
        IndexVectorX GVGilocal; ///< |#vertex-quad.pt. adjacencies| array of local vertex indices
                                ///< associated with adjacency graph edges

        /**
         * @brief Dirichlet energy
         */

        VectorX dwg;  ///< |#Dirichlet quad.pts.| array of Dirichlet quadrature weights
        MatrixX dNcg; ///< 4x|#Dirichlet quad.pts.| array of coarse cage shape functions at
                      ///< Dirichlet quadrature points
        MatrixX dxg;  ///< 3x|#Dirichlet quad.pts.| array of Dirichlet boundary conditions at
                      ///< Dirichlet quadrature points

        /**
         * @brief Boundary integration
         */

        IndexVectorX
            GVDGp; ///< |#verts + 1| prefix array of vertex-Dirichlet quad.pt. adjacency graph
        IndexVectorX GVDGg; ///< |#vertex-Dirichlet quad.pt. adjacencies| array of Dirichlet quad.
                            ///< adjacency graph edges
        IndexVectorX GVDGe; ///< |#vertex-Dirichlet quad.pt. adjacencies| array of element indices
                            ///< associated with Dirichlet quad. adjacency graph edges
        IndexVectorX GVDGilocal; ///< |#vertex-Dirichlet quad.pt. adjacencies| array of local vertex
                                 ///< indices associated with Dirichlet quad. adjacency graph edges

        // VectorX swg;       ///< |#surface quad.pts.| array of surface quadrature weights
        // IndexVectorX ecsg; ///< |#surface quad.pts.| array of element indices containing surface
        //                    ///< quadrature points
        // MatrixX Ncsg; ///< 4x|#surface quad.pts.| array of coarse cage shape functions at surface
        //               ///< quadrature points
        // MatrixX
        //     nsg; ///< 3x|#surface quad.pts.| array of outward normals at surface quadrature
        //     points

        // IndexVectorX
        //     GVSGp; ///< |#verts + 1| prefix array of vertex-surface quad.pt. adjacency graph
        // IndexVectorX
        //     GVSGg; ///< |#vertex-quad.pt. adjacencies| array of surface quad. adjacency graph
        //     edges
        // IndexVectorX GVSGe; ///< |#vertex-quad.pt. adjacencies| array of element indices
        // associated
        //                     ///< with surface quad. adjacency graph edges
        // IndexVectorX GVSGilocal; ///< |#vertex-quad.pt. adjacencies| array of local vertex
        // indices
        //                          ///< associated with surface quad. adjacency graph edges
    };

    struct Cage
    {
        /**
         * @brief Construct a new Cage object
         *
         * @param x 3x|#verts| array of mesh vertex positions
         * @param E 4x|#elements| array of mesh elements
         * @param ptr |#partitions + 1| prefix array of partition-vertex adjacency graph, i.e.
         * Padj[Pptr[i]:Pptr[i+1]] yield vertex indices of the i^{th} partition
         * @param adj |#free verts| array of partition-vertex adjacency graph edges
         */
        Cage(
            Eigen::Ref<MatrixX const> const& x,
            Eigen::Ref<IndexMatrixX const> const& E,
            Eigen::Ref<IndexVectorX const> const& ptr,
            Eigen::Ref<IndexVectorX const> const& adj);

        MatrixX x;      ///< 3x|#verts| array of mesh vertex positions
        IndexMatrixX E; ///< 4x|#elements| array of mesh elements

        IndexVectorX
            ptr; ///< |#partitions + 1| prefix array of partition-vertex adjacency graph, i.e.
                 ///< Padj[Pptr[i]:Pptr[i+1]] yield vertex indices of the i^{th} partition
        IndexVectorX adj; ///< |#free verts| array of partition-vertex adjacency graph edges
    };

    /**
     * @brief
     *
     * This allows transferring problem parameters from the root directly to a given coarser level,
     * for example inertial targets xtildeg.
     *
     */
    struct RootParameterBus
    {
        /**
         * @brief Construct a new Root Parameter Bus object
         *
         * @param ergIn
         * @param NrgIn
         */
        RootParameterBus(
            Eigen::Ref<IndexVectorX const> const& ergIn,
            Eigen::Ref<MatrixX const> const& NrgIn);

        IndexVectorX erg; ///< |#quad.pts.| array of root element associated with quadrature points
        MatrixX Nrg; ///< 4x|#quad.pts at level l| arrays of root level shape functions evaluated at
                     ///< quadrature points
    };

    /**
     * @brief Construct a new Level object
     *
     * @param C
     * @param E
     * @param RPB
     */
    Level(Cage C, Energy E, RootParameterBus RPB);

    Cage C;
    Energy E;
    RootParameterBus RPB;
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LEVEL_H