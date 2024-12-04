#ifndef PBAT_SIM_VBD_RESTRICTION_H
#define PBAT_SIM_VBD_RESTRICTION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Hierarchy;
struct Level;

struct Restriction
{
    /**
     * @brief
     *
     * @param lf Fine level
     * @return Restriction&
     */
    Restriction& From(Index lf);
    /**
     * @brief
     *
     * @param lc Coarse level
     * @return Restriction&
     */
    Restriction& To(Index lc);
    /**
     * @brief
     *
     * @param efg |#quad.pts.| array of fine cage elements associated with quadrature points
     * @param Nfg 4x|#quad.pts.| array of fine cage shape functions at coarse quadrature points
     * @return Restriction&
     */
    Restriction& WithFineShapeFunctions(
        Eigen::Ref<IndexVectorX const> const& efg,
        Eigen::Ref<MatrixX const> const& Nfg);
    /**
     * @brief
     *
     * @param iterations Number of descent iterations
     * @return Restriction&
     */
    Restriction& Iterate(Index iterations);
    /**
     * @brief
     *
     * @param bValidate Throw on ill-formed input
     * @return Restriction&
     */
    Restriction& Construct(bool bValidate = true);
    /**
     * @brief
     *
     * @return Index
     */
    Index StartLevel() const;
    /**
     * @brief
     *
     * @return Index
     */
    Index EndLevel() const;

    IndexVectorX
        efg;     ///< |#quad.pts.| array of fine cage elements associated with quadrature points
    MatrixX Nfg; ///< 4x|#quad.pts.| array of fine cage shape functions at coarse quadrature points
    MatrixX xfg; ///< 3x|#quad.pts.| array of fine cage shape target positions at quadrature points
    Index iterations; ///< Number of BCD iterations to compute restriction
    Index lc, lf; ///< Indices to coarse (lc) and fine (lf) levels, respectively, assuming lc > lf

    /**
     * @brief Restrict level lf to level lc
     *
     * @param H
     */
    void Apply(Hierarchy& H);
    /**
     * @brief
     * @param Ef
     * @param xf
     */
    void
    SetTargetShape(Eigen::Ref<IndexMatrixX const> const& Ef, Eigen::Ref<MatrixX const> const& xf);
    /**
     * @brief Restrict
     *
     * @param Lc
     * @param detHZero
     */
    Scalar DoApply(Level& Lc, Scalar detHZero) const;
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_RESTRICTION_H