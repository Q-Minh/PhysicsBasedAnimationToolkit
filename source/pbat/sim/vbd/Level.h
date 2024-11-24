#ifndef PBAT_SIM_VBD_LEVEL_H
#define PBAT_SIM_VBD_LEVEL_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Energy
{
    Scalar dt;       ///< Time step
    MatrixX xtildeg; ///< 3x|#quad.pts.| array of inertial target positions at quadrature points
    VectorX rhog;    ///< |#quad.pts.| array of mass densities at quadrature points
    MatrixX
        Ncg; ///< 4x|#quad.pts.| array of coarse cage element shape functions at quadrature points
    MatrixX GNcg;     ///< 4x|3*#quad.pts.| array of coarse cage element shape function gradients at
                      ///< quadrature points
    IndexMatrixX erg; ///< 4x|#quad.pts.| array of coarse element indices containing vertices of
                      ///< root level element embedding quadrature point g
    MatrixX Nrg;  ///< 4x|4*#quad.pts.| array of coarse cage element shape functions at root level
                  ///< elements' 4 vertices associated with quadrature points
    MatrixX GNfg; ///< 4x|3*#quad.pts.| array of root level element shape function gradients at
                  ///< quadrature points
    VectorX mug;  ///< |#quad.pts.| array of first Lame coefficients at quadrature points
    VectorX lambdag; ///< |#quad.pts.| array of second Lame coefficients at quadrature points
    IndexVectorX eg; ///< |#quad.pts.| array of elements associated with quadrature points
    Eigen::Vector<bool, Eigen::Dynamic>
        sg;     ///< |#quad.pts.| boolean array identifying singular quadrature points
    VectorX wg; ///< |#quad.pts.| array of quadrature weights
};

struct Adjacency
{
    IndexVectorX GVGp; ///< |#verts + 1| prefix array of vertex-quad.pt. adjacency graph
    IndexVectorX GVGg; ///< |#vertex-quad.pt. adjacencies| array of adjacency graph edges
    IndexVectorX GVGe; ///< |#vertex-quad.pt. adjacencies| array of element indices associated with
                       ///< adjacency graph edges
    IndexVectorX GVGilocal; ///< |#vertex-quad.pt. adjacencies| array of local vertex indices
                            ///< associated with adjacency graph edges
};

struct Partitions
{
    IndexVectorX ptr; ///< |#partitions + 1| prefix array of partition-vertex adjacency graph, i.e.
                      ///< Padj[Pptr[i]:Pptr[i+1]] yield vertex indices of the i^{th} partition
    IndexVectorX adj; ///< |#free verts| array of partition-vertex adjacency graph edges
};

struct Cage
{
    IndexMatrixX E; ///< 4x|#elements| array of mesh elements
    MatrixX x;      ///< 3x|#verts| array of mesh vertex positions
};

struct Level
{
    Cage C;
    Energy E;
    Adjacency VG;
    Partitions P;

    Level& WithMesh(Cage cage);
    Level& WithEnergy(Energy energy);
    Level& WithAdjacency(Adjacency adjacency);
    Level& WithPartitions(Partitions partitions);
    bool Construct();
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LEVEL_H