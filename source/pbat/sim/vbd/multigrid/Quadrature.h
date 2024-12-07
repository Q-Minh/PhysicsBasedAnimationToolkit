#ifndef PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H
#define PBAT_SIM_VBD_MULTIGRID_QUADRATURE_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

enum class ECageQuadratureStrategy { EmbeddedMesh, PolynomialSubCellIntegration };

struct CageQuadrature
{
    /**
     * @brief Computes a cage quadrature given the domain (X,E) and embedding cage (XC,EC) as
     * meshes.
     * @param X
     * @param E
     * @param XC
     * @param EC
     */
    CageQuadrature(
        Eigen::Ref<MatrixX const> const& X,
        Eigen::Ref<IndexMatrixX const> const& E,
        Eigen::Ref<MatrixX const> const& XC,
        Eigen::Ref<IndexMatrixX const> const& EC,
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
    SurfaceQuadrature(
        Eigen::Ref<MatrixX const> const& X,
        Eigen::Ref<IndexMatrixX const> const& E,
        Eigen::Ref<MatrixX const> const& XC,
        Eigen::Ref<IndexMatrixX const> const& EC,
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