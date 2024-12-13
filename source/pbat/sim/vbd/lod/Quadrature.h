#ifndef PBAT_SIM_VBD_LOD_QUADRATURE_H
#define PBAT_SIM_VBD_LOD_QUADRATURE_H

#include "Mesh.h"
#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

enum class ECageQuadratureStrategy { CageMesh, EmbeddedMesh, PolynomialSubCellIntegration };

struct CageQuadratureParameters
{
    ECageQuadratureStrategy eStrategy{ECageQuadratureStrategy::EmbeddedMesh};
    int mCageMeshPointsOfOrder{4};
    int mPatchCellPointsOfOrder{2};
    Scalar mPatchTetVolumeError{1e-4};

    CageQuadratureParameters& WithStrategy(ECageQuadratureStrategy eStrategyIn);
    CageQuadratureParameters& WithCageMeshPointsOfOrder(int order);
    CageQuadratureParameters& WithPatchCellPointsOfOrder(int order);
    CageQuadratureParameters& WithPatchError(Scalar err);
};

struct CageQuadrature
{
    CageQuadrature() = default;

    /**
     * @brief Computes a cage quadrature given a fine mesh FM and its embedding cage mesh CM.
     * @param FM Fine mesh
     * @param CM Coarse mesh
     * @param params
     */
    CageQuadrature(
        VolumeMesh const& FM,
        VolumeMesh const& CM,
        CageQuadratureParameters const& params = CageQuadratureParameters{});

    using BoolVectorType = Eigen::Vector<bool, Eigen::Dynamic>;

    MatrixX Xg; ///< 3x|#quad.pts.| array of quadrature points
    VectorX wg; ///< |#quad.pts.| array of quadrature weights
    BoolVectorType
        sg;          ///< |#quad.pts.| boolean mask indicating quad.pts. outside the embedded domain
    IndexVectorX eg; ///< |#quad.pts.| array of cage elements containing corresponding quad.pts.

    MatrixX Ncg;      ///< 4x|#quad.pts.| coarse mesh shape functions at quad.pts.
    MatrixX GNcg;     ///< 4x|3*#quad.pts.| coarse mesh shape function gradients at quad.pts.
    IndexVectorX efg; ///< |#quad.pts.| fine mesh elements associated with quad.pts.
    MatrixX Nfg;      ///< 4x|#quad.pts.| fine mesh shape functions at quad.pts.
    MatrixX GNfg;     ///< 4x|3*#quad.pts.| fine mesh shape function gradients at quad.pts.

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

    MatrixX Ncg; ///< 4x|#quad.pts.| coarse mesh shape functions at quad.pts.

    IndexVectorX GVGp;      ///< |#cage verts + 1| prefix
    IndexVectorX GVGg;      ///< |#quad.pts.| cage vertex-quad.pt. adjacencies
    IndexVectorX GVGilocal; ///< |#quad.pts.| cage vertex local element index
};

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LOD_QUADRATURE_H