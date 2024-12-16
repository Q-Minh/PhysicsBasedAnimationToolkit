#ifndef PBAT_SIM_VBD_MULTIGRID_KERNELS_H
#define PBAT_SIM_VBD_MULTIGRID_KERNELS_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/fem/DeformationGradient.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/vbd/Mesh.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {
namespace kernels {

namespace mini = math::linalg::mini;

template <
    physics::CHyperElasticEnergy TPsi,
    mini::CMatrix TMatrixIL,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixGN,
    mini::CMatrix TMatrixN,
    mini::CMatrix TMatrixGU,
    mini::CMatrix TMatrixHU,
    class ScalarType = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE void AccumulateElasticEnergy(
    TPsi const& Psi,
    TMatrixIL const& ilocal,
    ScalarType wg,
    ScalarType mug,
    ScalarType lambdag,
    TMatrixX const& xe,
    TMatrixGN const& GNe,
    TMatrixN const& N,
    TMatrixGU& gu,
    TMatrixHU& Hu)
{
    using namespace mini;
    SMatrix<ScalarType, 3, 3> F = xe * GNe;
    SVector<Scalar, 9> gF       = Zeros<Scalar, 9>();
    SMatrix<Scalar, 9, 9> HF    = Zeros<Scalar, 9, 9>();
    Psi.gradAndHessian(F, mug, lambdag, gF, HF);
    using Element             = typename VolumeMesh::ElementType;
    SMatrix<Scalar, 3, 3> dHu = Zeros<Scalar, 3, 3>();
    SVector<Scalar, 3> dgu    = Zeros<Scalar, 3>();
    for (auto i = 0; i < 4; ++i)
    {
        for (auto j = 0; j < 4; ++j)
        {
            if (ilocal(i) >= 0 and ilocal(j) >= 0)
            {
                auto Hij = fem::HessianBlockWrtDofs<Element, 3>(HF, GNe, i, j);
                dHu += N(ilocal(i), i) * N(ilocal(j), j) * Hij;
            }
        }
    }
    for (auto i = 0; i < 4; ++i)
    {
        if (ilocal(i) >= 0)
        {
            auto gi = fem::GradientSegmentWrtDofs<Element, 3>(gF, GNe, i);
            dgu += N(ilocal(i), i) * gi;
        }
    }
    Hu += wg * dHu;
    gu += wg * dgu;
}

} // namespace kernels
} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_KERNELS_H