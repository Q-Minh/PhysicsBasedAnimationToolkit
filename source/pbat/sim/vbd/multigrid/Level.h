#ifndef PBAT_SIM_VBD_MULTIGRID_LEVEL_H
#define PBAT_SIM_VBD_MULTIGRID_LEVEL_H

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
namespace multigrid {

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
     * @param eStrategy
     * @return
     */
    Level& WithCageQuadrature(
        Data const& problem,
        ECageQuadratureStrategy eStrategy = ECageQuadratureStrategy::PolynomialSubCellIntegration);
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

    VolumeMesh mesh;
    CageQuadrature Qcage;
    DirichletQuadrature Qdirichlet;
    MomentumEnergy Ekinetic;
    ElasticEnergy Epotential;
    DirichletEnergy Edirichlet;
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_LEVEL_H