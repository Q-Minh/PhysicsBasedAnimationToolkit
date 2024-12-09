#ifndef PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H
#define PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H

#include "Mesh.h"
#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

enum class ECageQuadratureStrategy { CageMesh, EmbeddedMesh, PolynomialSubCellIntegration };

struct CageQuadrature
{
    CageQuadrature() = default;

    /**
     * @brief Computes a cage quadrature given a fine mesh FM and its embedding cage mesh CM.
     * @param FM Fine mesh
     * @param CM Coarse mesh
     */
    CageQuadrature(
        VolumeMesh const& FM,
        VolumeMesh const& CM,
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
    SurfaceQuadrature() = default;
    /**
     * @brief Computes a surface quadrature on the boundary of a mesh FM embedded in mesh CM.
     * @param FM
     * @param CM
     * @param eStrategy
     */
    SurfaceQuadrature(
        VolumeMesh const& FM,
        VolumeMesh const& CM,
        ESurfaceQuadratureStrategy eStrategy =
            ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature);

    MatrixX Xg;      ///< 3x|#quad.pts.| array of quadrature points
    VectorX wg;      ///< |#quad.pts.| array of quadrature weights
    IndexVectorX eg; ///< |#quad.pts.| array of cage elements containing corresponding quad.pts.

    IndexVectorX GVGp;      ///< |#cage verts + 1| prefix
    IndexVectorX GVGg;      ///< |#quad.pts.| cage vertex-quad.pt. adjacencies
    IndexVectorX GVGilocal; ///< |#quad.pts.| cage vertex local element index
};

struct DirichletQuadrature
{
    DirichletQuadrature() = default;

    /**
     * @brief
     * @param FM Fine mesh
     * @param CM Coarse mesh
     * @param m Lumped mass coefficients of FM
     * @param dbcs Dirichlet constrained vertices in FM
     */
    DirichletQuadrature(
        VolumeMesh const& FM,
        VolumeMesh const& CM,
        Eigen::Ref<VectorX const> const& m,
        Eigen::Ref<IndexVectorX const> const& dbcs);

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