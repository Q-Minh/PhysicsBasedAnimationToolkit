/**
 * @file ValanisLandelQuasiNewton.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header for Valanis-Landel Quasi-Newton stiffness.
 * @version 0.1
 * @date 2025-09-25
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_PD_VALANISLANDELQUASINEWTON_H
#define PBAT_SIM_ALGORITHM_PD_VALANISLANDELQUASINEWTON_H

#include "pbat/common/Concepts.h"
#include "pbat/physics/Enums.h"

#include <Eigen/Core>
#include <cmath>

namespace pbat::sim::algorithm::pd {

/**
 * @brief Computes the element constraint stiffness for the quasi-Newton Projective Dynamics hessian
 * approximation @cite liu2017quasiPD .
 *
 * @tparam TScalar Scalar type
 * @param mu First Lame coefficient
 * @param lambda Second Lame coefficient
 * @param sigma Singular values of deformation gradient
 * @param sigmalo Lowest singular value of deformation gradient
 * @param sigmahi Highest singular value of deformation gradient
 * @param energy Hyperelastic energy model
 * @return Quasi-Newton stiffness for Projective Dynamics element constraints
 */
template <common::CArithmetic TScalar>
TScalar ValanisLandelQuasiNewtonStiffness(
    TScalar mu,
    TScalar lambda,
    Eigen::Vector<TScalar, 3> const& sigma,
    TScalar sigmalo,
    TScalar sigmahi,
    physics::EHyperElasticEnergy energy)
{
    TScalar kstar{1};
    switch (energy)
    {
        case physics::EHyperElasticEnergy::SaintVenantKirchhoff: {
            TScalar const a0  = ((sigmahi) * (sigmahi) * (sigmahi));
            TScalar const a1  = ((sigmahi) * (sigmahi));
            TScalar const a2  = ((sigmalo) * (sigmalo));
            TScalar const a3  = ((sigmalo) * (sigmalo) * (sigmalo));
            TScalar const a4  = 12 * sigmahi;
            TScalar const a5  = lambda * mu;
            TScalar const a6  = a4 * a5;
            TScalar const a7  = 12 * sigmalo;
            TScalar const a8  = a5 * a7;
            TScalar const a9  = ((lambda) * (lambda));
            TScalar const a10 = ((mu) * (mu));
            TScalar const a11 = 6 * a1;
            TScalar const a12 = a11 * a5;
            TScalar const a13 = 4 * a5;
            TScalar const a14 = ((sigmahi) * (sigmahi) * (sigmahi) * (sigmahi));
            TScalar const a15 = 3 * a5;
            TScalar const a16 = 6 * a2;
            TScalar const a17 = a16 * a5;
            TScalar const a18 = ((sigmalo) * (sigmalo) * (sigmalo) * (sigmalo));
            TScalar const a19 = ((sigma[1]) * (sigma[1]));
            TScalar const a20 = ((sigma[2]) * (sigma[2]));
            TScalar const a21 = sigma[1] * sigma[2];
            TScalar const a22 = a21 * a5;
            TScalar const a23 = 12 * a22;
            TScalar const a24 = 8 * a22;
            TScalar const a25 = a21 * a9;
            TScalar const a26 = 12 * a25;
            TScalar const a27 = 8 * a25;
            TScalar const a28 = a19 * a20 * a9;
            TScalar const a29 = 4 * a28;
            TScalar const a30 = 3 * a28;
            kstar             = (1.0 / 8.0) *
                    std::abs(
                        (a0 * a13 + a0 * a24 + a0 * a27 + a0 * a29 - a1 * a23 - a1 * a26 -
                         a10 * a11 + a10 * a16 + a10 * a4 - a10 * a7 - a11 * a9 - a12 * a19 -
                         a12 * a20 + a12 - a13 * a3 - a14 * a15 - a14 * a30 + a15 * a18 + a16 * a9 +
                         a17 * a19 + a17 * a20 - a17 + a18 * a30 + a19 * a6 - a19 * a8 + a2 * a23 +
                         a2 * a26 + a20 * a6 - a20 * a8 - a24 * a3 - a27 * a3 - a29 * a3 + a4 * a9 -
                         a6 - a7 * a9 + a8) /
                        (lambda * (a0 - 3 * a1 + 3 * a2 - a3 + 3 * sigmahi - 3 * sigmalo)));
            break;
        }
        case physics::EHyperElasticEnergy::StableNeoHookean: {
            TScalar const a0  = ((sigmahi) * (sigmahi) * (sigmahi));
            TScalar const a1  = ((sigmahi) * (sigmahi));
            TScalar const a2  = ((sigmalo) * (sigmalo));
            TScalar const a3  = ((sigmalo) * (sigmalo) * (sigmalo));
            TScalar const a4  = 270 * lambda;
            TScalar const a5  = 180 * mu;
            TScalar const a6  = 135 * lambda;
            TScalar const a7  = 60 * lambda;
            TScalar const a8  = ((sigmahi) * (sigmahi) * (sigmahi) * (sigmahi));
            TScalar const a9  = 45 * lambda;
            TScalar const a10 = std::pow(sigmahi, 5);
            TScalar const a11 = 6 * lambda;
            TScalar const a12 = std::pow(sigmahi, 6);
            TScalar const a13 = 5 * lambda;
            TScalar const a14 = ((sigmalo) * (sigmalo) * (sigmalo) * (sigmalo));
            TScalar const a15 = std::pow(sigmalo, 5);
            TScalar const a16 = std::pow(sigmalo, 6);
            TScalar const a17 = 90 * mu;
            TScalar const a18 = 40 * mu;
            TScalar const a19 = 30 * mu;
            TScalar const a20 = 12 * mu;
            TScalar const a21 = 10 * mu;
            TScalar const a22 = ((sigma[1]) * (sigma[1]));
            TScalar const a23 = 180 * lambda;
            TScalar const a24 = a23 * sigmahi;
            TScalar const a25 = ((sigma[1]) * (sigma[1]) * (sigma[1]) * (sigma[1]));
            TScalar const a26 = a25 * sigmahi;
            TScalar const a27 = 30 * lambda;
            TScalar const a28 = ((sigma[2]) * (sigma[2]));
            TScalar const a29 = ((sigma[2]) * (sigma[2]) * (sigma[2]) * (sigma[2]));
            TScalar const a30 = a29 * sigmahi;
            TScalar const a31 = a22 * sigmalo;
            TScalar const a32 = a27 * sigmalo;
            TScalar const a33 = a28 * sigmalo;
            TScalar const a34 = 120 * mu;
            TScalar const a35 = a34 * sigmahi;
            TScalar const a36 = 60 * mu;
            TScalar const a37 = a36 * sigmalo;
            TScalar const a38 = a1 * a22;
            TScalar const a39 = 90 * lambda;
            TScalar const a40 = 15 * lambda;
            TScalar const a41 = a1 * a40;
            TScalar const a42 = a1 * a28;
            TScalar const a43 = a22 * lambda;
            TScalar const a44 = 20 * a0;
            TScalar const a45 = a28 * lambda;
            TScalar const a46 = 15 * a8;
            TScalar const a47 = a2 * a39;
            TScalar const a48 = a2 * a40;
            TScalar const a49 = 20 * a3;
            TScalar const a50 = 15 * a14;
            TScalar const a51 = a1 * a19;
            TScalar const a52 = a2 * a36;
            TScalar const a53 = a19 * a2;
            TScalar const a54 = a22 * a28;
            kstar             = (1.0 / 20.0) *
                    std::abs(
                        (a0 * a18 + a0 * a7 + a1 * a17 + a1 * a6 - a10 * a11 - a10 * a20 +
                         a11 * a15 + a12 * a13 + a12 * a21 - a13 * a16 + a14 * a19 + a14 * a9 +
                         a15 * a20 - a16 * a21 - a17 * a2 - a18 * a3 - a19 * a8 - a2 * a27 * a54 -
                         a2 * a6 + a22 * a24 + a22 * a35 + a22 * a47 + a22 * a52 - a23 * a31 -
                         a23 * a33 + a24 * a28 + a25 * a32 + a25 * a37 + a25 * a41 - a25 * a48 +
                         a25 * a51 - a25 * a53 - a26 * a27 - a26 * a36 + a27 * a28 * a38 -
                         a27 * a30 + a28 * a31 * a7 + a28 * a35 + a28 * a47 + a28 * a52 +
                         a29 * a32 + a29 * a37 + a29 * a41 - a29 * a48 + a29 * a51 - a29 * a53 -
                         a3 * a7 - a30 * a36 - a31 * a34 - a33 * a34 - a36 * a38 - a36 * a42 -
                         a38 * a39 - a39 * a42 - a4 * sigmahi + a4 * sigmalo - a43 * a44 +
                         a43 * a46 + a43 * a49 - a43 * a50 - a44 * a45 + a45 * a46 + a45 * a49 -
                         a45 * a50 - a5 * sigmahi + a5 * sigmalo - a54 * a7 * sigmahi - a8 * a9) /
                        (a0 - 3 * a1 + 3 * a2 - a3 + 3 * sigmahi - 3 * sigmalo));
            break;
        }
        default: break;
    }
    return kstar;
}

/**
 * @brief Vectorized version of ValanisLandelQuasiNewtonStiffness.
 *
 * @tparam TDerivedMu Matrix type for mu
 * @tparam TDerivedLambda Matrix type for lambda
 * @tparam TDerivedSigmaLo Matrix type for sigmalo
 * @tparam TDerivedSigmaHi Matrix type for sigmahi
 * @tparam TDerivedK Matrix type for k
 * @param mu `|# constraints| x 1` matrix of first Lame coefficients
 * @param lambda `|# constraints| x 1` matrix of second Lame coefficients
 * @param sigma `|# constraints| x 3` matrix of singular values of deformation gradients
 * @param sigmalo `|# constraints| x 1` matrix of lowest singular values of deformation gradients
 * @param sigmahi `|# constraints| x 1` matrix of highest singular values of deformation gradients
 * @param energy Hyperelastic energy model
 * @param k `|# constraints| x 1` matrix of constraint stiffnesses
 */
template <
    class TDerivedMu,
    class TDerivedLambda,
    class TDerivedSigma,
    class TDerivedSigmaLo,
    class TDerivedSigmaHi,
    class TDerivedK>
void ValanisLandelQuasiNewtonStiffness(
    Eigen::DenseBase<TDerivedMu> const& mu,
    Eigen::DenseBase<TDerivedLambda> const& lambda,
    Eigen::DenseBase<TDerivedSigma> const& sigma,
    Eigen::DenseBase<TDerivedSigmaLo> const& sigmalo,
    Eigen::DenseBase<TDerivedSigmaHi> const& sigmahi,
    physics::EHyperElasticEnergy energy,
    Eigen::DenseBase<TDerivedK>& k)
{
    using ScalarType = typename TDerivedMu::Scalar;
    k.resize(mu.rows(), mu.cols());
    auto n = mu.size();
    for (Eigen::Index i = 0; i < n; ++i)
    {
        k(i) = ValanisLandelQuasiNewtonStiffness<ScalarType>(
            mu(i),
            lambda(i),
            sigma.col(i),
            sigmalo(i),
            sigmahi(i),
            energy);
    }
}

} // namespace pbat::sim::algorithm::pd

#endif // PBAT_SIM_ALGORITHM_PD_VALANISLANDELQUASINEWTON_H
