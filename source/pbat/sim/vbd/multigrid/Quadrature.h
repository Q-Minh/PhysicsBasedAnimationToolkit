#ifndef PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H
#define PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H

#include "pbat/Aliases.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/Tetrahedron.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

enum class ECageQuadratureStrategy { EmbeddedMesh, PolynomialSubCellIntegration };

struct CageQuadrature
{
    /**
     * @brief Computes a cage quadrature given a fine mesh FM and its embedding cage mesh CM.
     * @param FM Fine mesh
     * @param CM Coarse mesh
     */
    CageQuadrature(
        fem::Mesh<fem::Tetrahedron<1>, 3> const& FM,
        fem::Mesh<fem::Tetrahedron<1>, 3> const& CM,
        ECageQuadratureStrategy eStrategy = ECageQuadratureStrategy::PolynomialSubCellIntegration);

    using BoolVectorType = Eigen::Vector<bool, Eigen::Dynamic>;

    MatrixX Xg; ///< 3x|#quad.pts.| array of quadrature points
    VectorX wg; ///< |#quad.pts.| array of quadrature weights
    BoolVectorType
        sg;          ///< |#quad.pts.| boolean mask indicating quad.pts. outside the embedded domain
    IndexVectorX eg; ///< |#quad.pts.| array of cage elements containing corresponding quad.pts.

    IndexVectorX GVGp;      ///< |#cage verts + 1| prefix
    IndexVectorX GVGg;      ///< |#quad.pts.| cage vertex-quad.pt. adjacencies
    IndexVectorX GVGilocal; ///< |#quad.pts.| cage vertex local element index
};

enum class ESurfaceQuadratureStrategy { EmbeddedVertexSinglePointQuadrature };

struct SurfaceQuadrature
{
    /**
     * @brief Computes a surface quadrature on the boundary of a mesh FM embedded in mesh CM.
     * @param FM 
     * @param CM 
     * @param eStrategy 
     */
    SurfaceQuadrature(
        fem::Mesh<fem::Tetrahedron<1>, 3> const& FM,
        fem::Mesh<fem::Tetrahedron<1>, 3> const& CM,
        ESurfaceQuadratureStrategy eStrategy =
            ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature);

    MatrixX Xg;      ///< 3x|#quad.pts.| array of quadrature points
    VectorX wg;      ///< |#quad.pts.| array of quadrature weights
    IndexVectorX eg; ///< |#quad.pts.| array of cage elements containing corresponding quad.pts.

    IndexVectorX GVGp;      ///< |#cage verts + 1| prefix
    IndexVectorX GVGg;      ///< |#quad.pts.| cage vertex-quad.pt. adjacencies
    IndexVectorX GVGilocal; ///< |#quad.pts.| cage vertex local element index
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H