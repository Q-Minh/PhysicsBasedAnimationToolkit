#ifndef PBAT_SIM_VBD_MULTIGRID_LEVEL_H
#define PBAT_SIM_VBD_MULTIGRID_LEVEL_H

#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Level
{
    /**
     * @brief
     * @param data
     * @param mesh
     */
    Level(Data const& data, VolumeMesh mesh);

    /**
     * @brief
     * @param data
     */
    void Prolong(Data& data) const;
    /**
     * @brief
     * @param data
     */
    void Smooth(Data const& data);

    /**
     * Coarse mesh discretization
     */
    VolumeMesh mesh;         ///< Coarse FEM mesh
    MatrixX u;               ///< 3x|#cage verts| coarse displacement coefficients
    IndexVectorX colors;     ///< Coarse vertex graph coloring
    IndexVectorX Pptr, Padj; ///< Parallel vertex partitions

    /**
     * Elastic energy
     */
    IndexMatrixX ecVE; ///< 4x|#fine elems| coarse elements containing 4 vertices of fine elements
    MatrixX
        NecVE; ///< 4x|4*#fine elems| coarse element shape functions at 4 vertices of fine elements
    IndexMatrixX ilocalE; ///< 4x|#fine elems| coarse vertex local index w.r.t. coarse elements
                          ///< containing 4 vertices of fine elements
    IndexVectorX GEptr, GEadj; ///< Coarse vertex -> fine element adjacency graph

    /**
     * Kinetic energy
     */
    IndexVectorX ecK; ///< |#fine vertices| coarse elements containing fine vertices
    MatrixX NecK;     ///< 4x|#fine vertices| coarse element shape functions at fine vertices
    IndexVectorX GKptr, GKadj, GKilocal; ///< Coarse vertex -> fine vertex adjacency graph

    /**
     * Dirichlet energy
     */
    using BoolVector = Eigen::Vector<bool, Eigen::Dynamic>;
    BoolVector bIsDirichletVertex;
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_LEVEL_H