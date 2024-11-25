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
         * @param dt
         * @param xtildeg 3x|#quad.pts.| array of inertial target positions at quadrature points
         * @param rhog |#quad.pts.| array of mass densities at quadrature points
         * @param Ncg 4x|#quad.pts.| array of coarse cage element shape functions at quadrature
         * points
         * @return Energy&
         */
        Energy& WithKineticEnergy(
            Scalar dt,
            Eigen::Ref<MatrixX const> const& xtildeg,
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
            Eigen::Ref<IndexVectorX const> const& erg,
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

        Scalar dt;       ///< Time step
        MatrixX xtildeg; ///< 3x|#quad.pts.| array of inertial target positions at quadrature points
        VectorX rhog;    ///< |#quad.pts.| array of mass densities at quadrature points
        MatrixX Ncg; ///< 4x|#quad.pts.| array of coarse cage element shape functions at quadrature
                     ///< points

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

        Eigen::Vector<bool, Eigen::Dynamic>
            sg;     ///< |#quad.pts.| boolean array identifying singular quadrature points
        VectorX wg; ///< |#quad.pts.| array of quadrature weights

        IndexVectorX GVGp; ///< |#verts + 1| prefix array of vertex-quad.pt. adjacency graph
        IndexVectorX GVGg; ///< |#vertex-quad.pt. adjacencies| array of adjacency graph edges
        IndexVectorX GVGe; ///< |#vertex-quad.pt. adjacencies| array of element indices associated
                           ///< with adjacency graph edges
        IndexVectorX GVGilocal; ///< |#vertex-quad.pt. adjacencies| array of local vertex indices
                                ///< associated with adjacency graph edges
    };

    struct Cage
    {
        /**
         * @brief Construct a new Cage object
         *
         * @param E 4x|#elements| array of mesh elements
         * @param x 3x|#verts| array of mesh vertex positions
         * @param ptr |#partitions + 1| prefix array of partition-vertex adjacency graph, i.e.
         * Padj[Pptr[i]:Pptr[i+1]] yield vertex indices of the i^{th} partition
         * @param adj |#free verts| array of partition-vertex adjacency graph edges
         */
        Cage(
            Eigen::Ref<IndexMatrixX const> const& E,
            Eigen::Ref<MatrixX const> const& x,
            Eigen::Ref<IndexVectorX const> const& ptr,
            Eigen::Ref<IndexVectorX const> const& adj);

        IndexMatrixX E; ///< 4x|#elements| array of mesh elements
        MatrixX x;      ///< 3x|#verts| array of mesh vertex positions

        IndexVectorX
            ptr; ///< |#partitions + 1| prefix array of partition-vertex adjacency graph, i.e.
                 ///< Padj[Pptr[i]:Pptr[i+1]] yield vertex indices of the i^{th} partition
        IndexVectorX adj; ///< |#free verts| array of partition-vertex adjacency graph edges
    };

    /**
     * @brief Construct a new Level object
     * 
     * @param C 
     * @param E 
     */
    Level(Cage C, Energy E);

    Cage C;
    Energy E;
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LEVEL_H