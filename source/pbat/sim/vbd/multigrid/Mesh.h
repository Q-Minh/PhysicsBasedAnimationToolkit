#ifndef PBAT_SIM_VBD_MULTIGRID_MESH_H
#define PBAT_SIM_VBD_MULTIGRID_MESH_H

#include "pbat/fem/Mesh.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/fem/Triangle.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

using VolumeMesh  = fem::Mesh<fem::Tetrahedron<1>, 3>;
using SurfaceMesh = fem::Mesh<fem::Triangle<1>, 3>;

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_MESH_H