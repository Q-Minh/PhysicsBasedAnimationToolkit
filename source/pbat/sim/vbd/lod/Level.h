#ifndef PBAT_SIM_VBD_LOD_LEVEL_H
#define PBAT_SIM_VBD_LOD_LEVEL_H

#include "DirichletEnergy.h"
#include "ElasticEnergy.h"
#include "Mesh.h"
#include "MomentumEnergy.h"
#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

struct Level
{
    /**
     * @brief
     * @param CM This level's finite element mesh
     */
    Level(VolumeMesh CM);
    /**
     * @brief Constructs this level's cage quadrature
     * @param problem
     * @param params
     * @return
     */
    Level& WithCageQuadrature(Data const& problem, CageQuadratureParameters const& params);
    /**
     * @brief Constructs this level's Dirichlet quadrature
     * @param problem
     * @return
     */
    Level& WithDirichletQuadrature(Data const& problem);
    /**
     * @brief Constructs this level's momentum energy
     * @param problem
     * @return
     */
    Level& WithMomentumEnergy(Data const& problem);
    /**
     * @brief Constructs this level's elastic energy
     * @param problem
     * @return
     */
    Level& WithElasticEnergy(Data const& problem);
    /**
     * @brief
     * @param problem
     * @return
     */
    Level& WithDirichletEnergy(Data const& problem);

    MatrixX x;             ///< 3x|#cage verts| deformed positions
    IndexVectorX colors;   ///< Coarse vertex graph coloring
    IndexVectorX ptr, adj; ///< Parallel vertex partitions

    VolumeMesh mesh;                ///< Cage linear tetrahedral FEM mesh
    CageQuadrature Qcage;           ///< Cage volumetric quadrature
    DirichletQuadrature Qdirichlet; ///< Cage Dirichlet quadrature
    MomentumEnergy Ekinetic;        ///< \frac{1}{2} \rho || x^c - \Tilde{x} ||_2^2
    ElasticEnergy Epotential;       ///< \Psi(x^c)
    DirichletEnergy Edirichlet;     ///< \frac{1}{2} \mu_D || x^c - x_D ||_2^2
};

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LOD_LEVEL_H