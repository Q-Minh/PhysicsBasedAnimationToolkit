#ifndef PBAT_SIM_VBD_RESTRICTION_H
#define PBAT_SIM_VBD_RESTRICTION_H

#include "Kernels.h"
#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

class Restriction
{
  public:
    template <class TDerivedXTI>
    void SetTargetShape(Eigen::MatrixBase<TDerivedXTI> const& xtildeg);

    /**
     * @brief
     *
     * @tparam TDerivedX
     * @param x
     */
    template <class TDerivedX>
    void Apply(Eigen::MatrixBase<TDerivedX>& x, Index iterations = 20);

  public:
    IndexMatrixX E;  ///< 4x|#elements| array of tetrahedral element nodal indices
    IndexVectorX eg; ///< |#quad.pts.| array of elements associated with quadrature points
    MatrixX Ng;      ///< 4x|#quad.pts.| array of shape functions at quadrature points
    MatrixX GNeg; ///< 4x|#dims * #quad.pts.| array of shape function gradients at quadrature points
    VectorX wg;   ///< |#quad.pts.| array of quadrature weights
    VectorX rhog; ///< |#quad.pts.| array of mass densities at quadrature points
    VectorX mug;  ///< First Lame coefficients at quadrature points
    VectorX lambdag;   ///< Second Lame coefficients at quadrature points
    VectorX m;         ///< |#coarse nodes| lumped mass matrix diagonals
    MatrixX gtilde;    ///< 3x|#coarse nodes| array of target shape energy gradients
    VectorX Psitildeg; ///< |#quad.pts.| array of elastic energies at quadrature points
    Scalar Kpsi{1e-4}; ///< Stiffness of elastic energy matching

    IndexVectorX GVGp; ///< |#coarse nodes + 1| prefixes into GVGg
    IndexVectorX
        GVGg; ///< |# of coarse vertex-quad.pt. edges| neighbours s.t.
              ///< GVGg[k] for GVGp[i] <= k < GVGp[i+1] gives the quad.pts. involving vertex i
    IndexVectorX GVGe;      ///< |# of vertex-quad.pt. edges| element indices s.t.
                            ///< GVGe[k] for GVGp[i] <= k < GVGp[i+1] gives the element index of
                            ///< adjacent to vertex i for the neighbouring quad.pt.
    IndexVectorX GVGilocal; ///< |# of vertex-quad.pt. edges| local vertex indices s.t.
                            ///< GVGilocal[k] for GVGp[i] <= k < GVGp[i+1] gives the local index of
                            ///< vertex i for the neighbouring quad.pts.
    std::vector<std::vector<Index>> partitions; ///< Coarse vertex parallel partitions
};

template <class TDerivedXTI>
inline void Restriction::SetTargetShape(Eigen::MatrixBase<TDerivedXTI> const& xtildeg)
{
    gtilde.setZero();
    for (auto g = 0; g < xtildeg.cols(); ++g)
    {
        auto e     = eg(g);
        auto nodes = E.col(e).head<4>();
        for (auto i = 0; i < nodes.size(); ++i)
        {
            gtilde.col(nodes(i)) += wg(g) * rhog(g) * Ng(i, g) * xtildeg.col(g);
        }
    }
}

template <class TDerivedX>
inline void Restriction::Apply(Eigen::MatrixBase<TDerivedX>& x, Index iterations)
{
    for (auto k = 0; k < iterations; ++k)
    {
        for (auto const& partition : partitions)
        {
            tbb::parallel_for(std::size_t(0), partition.size(), [&](std::size_t v) {
                auto i     = partition[static_cast<std::size_t>(v)];
                auto begin = GVGp[i];
                auto end   = GVGp[i + 1];
                physics::StableNeoHookeanEnergy<3> Psi{};
                using namespace pbat::math::linalg;
                mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                for (auto n = begin; n < end; ++n)
                {
                    auto ilocal                     = GVGilocal[n];
                    auto e                          = GVGe[n];
                    auto g                          = GVGg[n];
                    auto mu                         = mug(g);
                    auto lambda                     = lambdag(g);
                    auto wgg                        = wg(g);
                    auto Te                         = E.col(e);
                    mini::SMatrix<Scalar, 4, 3> GNe = mini::FromEigen(GNeg.block<4, 3>(0, e * 3));
                    mini::SMatrix<Scalar, 3, 4> xe =
                        mini::FromEigen(x(Eigen::placeholders::all, Te).block<3, 4>(0, 0));
                    mini::SMatrix<Scalar, 3, 3> Fe = xe * GNe;
                    mini::SVector<Scalar, 9> gF;
                    mini::SMatrix<Scalar, 9, 9> HF;
                    auto Psig = Psi.evalWithGradAndHessian(Fe, mu, lambda, gF, HF);
                    mini::SMatrix<Scalar, 3, 3> Hg = mini::Zeros<Scalar, 3, 3>();
                    mini::SVector<Scalar, 3> gg    = mini::Zeros<Scalar, 3, 1>();
                    kernels::AccumulateElasticHessian(ilocal, Scalar(1), GNe, HF, Hg);
                    kernels::AccumulateElasticGradient(ilocal, Scalar(1), GNe, gF, gg);
                    auto dPsi = Psig - Psitildeg(g);
                    gi += wgg * dPsi * gg;
                    Hi += wgg * (gg * gg.Transpose() + dPsi * Hg);
                }
                gi *= Kpsi;
                if (not gtilde.col(i).isZero())
                    gi += mini::FromEigen((m(i) * x.col(i).head<3>() - gtilde.col(i).head<3>()));
                Hi *= Kpsi;
                mini::Diag(Hi) += m(i);
                mini::SVector<Scalar, 3> dx = mini::Inverse(Hi) * gi;
                x.col(i) -= mini::ToEigen(dx);
            });
        }
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_RESTRICTION_H