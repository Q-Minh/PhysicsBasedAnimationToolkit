#ifndef PBAT_SIM_VBD_MULTIGRID_KERNELS_H
#define PBAT_SIM_VBD_MULTIGRID_KERNELS_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/vbd/Kernels.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {
namespace kernels {

namespace mini = math::linalg::mini;

template <
    class IndexType,
    physics::CHyperElasticEnergy TPsi,
    mini::CMatrix TMatrixXCG,
    mini::CMatrix TMatrixGNCG,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCG::ScalarType>
PBAT_HOST_DEVICE Scalar AccumulateElasticEnergy(
    IndexType ilocal,
    ScalarType wg,
    TPsi const& Psi,
    ScalarType mug,
    ScalarType lambdag,
    TMatrixXCG const& xcg,
    TMatrixGNCG const& GNcg,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    SMatrix<Scalar, 3, 3> F = xcg * GNcg;
    SVector<Scalar, 9> gF;
    SMatrix<Scalar, 9, 9> HF;
    Scalar E = Psi.evalWithGradAndHessian(F, mug, lambdag, gF, HF);
    pbat::sim::vbd::kernels::AccumulateElasticGradient(ilocal, wg, GNcg, gF, gi);
    pbat::sim::vbd::kernels::AccumulateElasticHessian(ilocal, wg, GNcg, HF, Hi);
    return E;
}

template <
    class IndexType,
    mini::CMatrix TMatrixXCG,
    mini::CMatrix TMatrixNCG,
    mini::CMatrix TMatrixXTL,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCG::ScalarType>
PBAT_HOST_DEVICE void AccumulateShapeMatchingEnergy(
    IndexType ilocal,
    ScalarType wg,
    ScalarType rhog,
    TMatrixXCG const& xcg,
    TMatrixNCG const& Ncg,
    TMatrixXTL const& xtarget,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    auto xc = xcg * Ncg;
    // Energy is 1/2 w_g rho_g || xc - xf ||_2^2
    gi += (wg * rhog * Ncg(ilocal)) * (xc - xtarget);
    Diag(Hi) += wg * rhog * Ncg(ilocal) * Ncg(ilocal);
}

} // namespace kernels
} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_KERNELS_H