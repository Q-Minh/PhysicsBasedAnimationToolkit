from .. import codegen as cg
import sympy as sp


def IC(F):
    return (F.transpose() * F).trace()


def IIC(F):
    FTF = F.transpose() * F
    return (FTF.transpose() * FTF).trace()


def IIIC(F):
    return (F.transpose() * F).det()


def I1(S):
    return S.trace()


def I2(F):
    return (F.transpose() * F).trace()


def I3(F):
    return F.det()


def stvk(F, mu, llambda):
    I = sp.eye(F.shape[0])
    FtF = F.transpose() * F
    E = (FtF - I) / 2
    trE = E.trace()
    EtE = E.transpose() * E
    EddotE = EtE.trace()
    return mu * EddotE + (llambda / 2) * trE**2


def neohookean(F, mu, llambda):
    alpha = 1 + mu / llambda
    d = F.shape[0]
    return (mu / 2) * (I2(F) - d) + (llambda / 2) * (I3(F) - alpha) ** 2


def codegen(fpsi, energy_name: str):
    source = []

    header = f"""
/**
 * @file 
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief {energy_name} hyperelastic energy {"\cite smith2018snh" if energy_name == "StableNeoHookeanEnergy" else ""}
 * 
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_PHYSICS_{energy_name.upper()}_H
#define PBAT_PHYSICS_{energy_name.upper()}_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <cmath>

namespace pbat {{
namespace physics {{

template <int Dims>
struct {energy_name};
"""
    source.append(header)

    for d in range(1, 3 + 1):
        mu, llambda = sp.symbols("mu lambda", real=True)
        vecF = sp.Matrix(sp.MatrixSymbol("F", d * d, 1))
        F = vecF.reshape(d, d).transpose()
        psi = fpsi(F, mu, llambda)
        gradpsi = sp.derive_by_array(psi, vecF)
        hesspsi = sp.derive_by_array(gradpsi, vecF)[:, 0, :, 0]
        psicode = cg.codegen(psi, lhs=sp.Symbol("psi"), scalar_type="ScalarType")
        gradpsicode = cg.codegen(
            gradpsi, lhs=sp.MatrixSymbol("G", *gradpsi.shape), scalar_type="ScalarType"
        )
        hesspsicode = cg.codegen(
            hesspsi.transpose(),
            lhs=sp.MatrixSymbol("H", vecF.shape[0], vecF.shape[0]),
            scalar_type="ScalarType",
        )
        evalgradpsi = cg.codegen(
            [psi, gradpsi],
            lhs=[sp.Symbol("psi"), sp.MatrixSymbol("gF", *gradpsi.shape)],
            scalar_type="ScalarType",
        )
        evalgradhesspsi = cg.codegen(
            [psi, gradpsi, hesspsi],
            lhs=[
                sp.Symbol("psi"),
                sp.MatrixSymbol("gF", *gradpsi.shape),
                sp.MatrixSymbol("HF", vecF.shape[0], vecF.shape[0]),
            ],
            scalar_type="ScalarType",
        )
        gradhesspsi = cg.codegen(
            [gradpsi, hesspsi],
            lhs=[
                sp.MatrixSymbol("gF", *gradpsi.shape),
                sp.MatrixSymbol("HF", vecF.shape[0], vecF.shape[0]),
            ],
            scalar_type="ScalarType",
        )
        impl = f"""
/**
 * @brief {energy_name} hyperelastic energy for {d}D
 * 
 * @tparam Dims Dimension of the space
 */
template <>
struct {energy_name}<{d}>
{{
    public:
        template <class TScalar, int M, int N>
        using SMatrix = pbat::math::linalg::mini::SMatrix<TScalar, M, N>; ///< Scalar matrix type

        template <class TScalar, int M>
        using SVector = pbat::math::linalg::mini::SVector<TScalar, M>; ///< Scalar vector type

        static auto constexpr kDims = {d}; ///< Dimension of space

        /**
         * @brief Evaluate the elastic energy
         * 
         * @tparam TMatrix Matrix type
         * @param F Deformation gradient
         * @param mu First Lame coefficient
         * @param lambda Second Lame coefficient
         * @return ScalarType Energy
         */
        template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
        PBAT_HOST_DEVICE
        typename TMatrix::ScalarType
        eval(
            TMatrix const& F,
            typename TMatrix::ScalarType mu,
            typename TMatrix::ScalarType lambda) const;

        /**
         * @brief Evaluate the elastic energy gradient
         * 
         * @tparam TMatrix Matrix type
         * @param F Deformation gradient
         * @param mu First Lame coefficient
         * @param lambda Second Lame coefficient
         * @return ScalarType Energy gradient
         */
        template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
        PBAT_HOST_DEVICE
        SVector<typename TMatrix::ScalarType, {vecF.shape[0]}>
        grad(
            TMatrix const& F,
            typename TMatrix::ScalarType mu,
            typename TMatrix::ScalarType lambda) const;

        /**
         * @brief Evaluate the elastic energy hessian
         * 
         * @tparam TMatrix Matrix type
         * @param F Deformation gradient
         * @param mu First Lame coefficient
         * @param lambda Second Lame coefficient
         * @return ScalarType Energy hessian
         */
        template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
        PBAT_HOST_DEVICE
        SMatrix<typename TMatrix::ScalarType, {vecF.shape[0]},{vecF.shape[0]}>
        hessian(
            TMatrix const& F,
            typename TMatrix::ScalarType mu,
            typename TMatrix::ScalarType lambda) const;

        /**
         * @brief Evaluate the elastic energy and its gradient
         * 
         * @tparam TMatrix Matrix type
         * @param F Deformation gradient
         * @param mu First Lame coefficient
         * @param lambda Second Lame coefficient
         * @param gF Gradient w.r.t. F
         * @return ScalarType Energy and its gradient
         */
        template <
            math::linalg::mini::CReadableVectorizedMatrix TMatrix, 
            math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF
            >
        PBAT_HOST_DEVICE
        typename TMatrix::ScalarType
        evalWithGrad(
            TMatrix const& F,
            typename TMatrix::ScalarType mu,
            typename TMatrix::ScalarType lambda,
            TMatrixGF& gF) const;

        /**
         * @brief Evaluate the elastic energy with its gradient and hessian
         * 
         * @tparam TMatrix Matrix type
         * @param F Deformation gradient
         * @param mu First Lame coefficient
         * @param lambda Second Lame coefficient
         * @param gF Gradient w.r.t. F
         * @param HF Hessian w.r.t. F
         * @return ScalarType Energy and its gradient and hessian
         */
        template <
            math::linalg::mini::CReadableVectorizedMatrix TMatrix,
            math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF, 
            math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF
            >
        PBAT_HOST_DEVICE
        typename TMatrix::ScalarType
        evalWithGradAndHessian(
            TMatrix const& F,
            typename TMatrix::ScalarType mu,
            typename TMatrix::ScalarType lambda,
            TMatrixGF& gF,
            TMatrixHF& HF) const;

        /**
         * @brief Evaluate the elastic energy gradient and hessian
         * 
         * @tparam TMatrix Matrix type
         * @param F Deformation gradient
         * @param mu First Lame coefficient
         * @param lambda Second Lame coefficient
         * @param gF Gradient w.r.t. F
         * @param HF Hessian w.r.t. F
         */
        template <
            math::linalg::mini::CReadableVectorizedMatrix TMatrix,
            math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF, 
            math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF
            >
        PBAT_HOST_DEVICE
        void
        gradAndHessian(
            TMatrix const& F,
            typename TMatrix::ScalarType mu,
            typename TMatrix::ScalarType lambda,
            TMatrixGF& gF,
            TMatrixHF& HF) const;
}};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE
typename TMatrix::ScalarType
{energy_name}<{d}>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
{cg.tabulate(psicode, spaces=4)}
    return psi;
}}

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE
{energy_name}<{d}>::SVector<typename TMatrix::ScalarType, {vecF.shape[0]}>
{energy_name}<{d}>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, {vecF.shape[0]}> G;
{cg.tabulate(gradpsicode, spaces=4)}
    return G;
}}

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE
{energy_name}<{d}>::SMatrix<typename TMatrix::ScalarType, {vecF.shape[0]},{vecF.shape[0]}>
{energy_name}<{d}>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, {vecF.shape[0]},{vecF.shape[0]}> H;
{cg.tabulate(hesspsicode, spaces=4)}
    return H;
}}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF
    >
PBAT_HOST_DEVICE
typename TMatrix::ScalarType
{energy_name}<{d}>::evalWithGrad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF) const
{{
    static_assert(
        TMatrixGF::kRows == {vecF.shape[0]} and TMatrixGF::kCols == 1, 
        "Grad w.r.t. F must have dimensions {vecF.shape[0]}x1");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
{cg.tabulate(evalgradpsi, spaces=4)}
    return psi;
}}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF
    >
PBAT_HOST_DEVICE
typename TMatrix::ScalarType
{energy_name}<{d}>::evalWithGradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{{
    static_assert(
        TMatrixGF::kRows == {vecF.shape[0]} and TMatrixGF::kCols == 1, 
        "Grad w.r.t. F must have dimensions {vecF.shape[0]}x1");
    static_assert(
        TMatrixHF::kRows == {vecF.shape[0]} and TMatrixHF::kCols == {vecF.shape[0]}, 
        "Hessian w.r.t. F must have dimensions {vecF.shape[0]}x{vecF.shape[0]}");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
{cg.tabulate(evalgradhesspsi, spaces=4)}
    return psi;
}}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF
    >
PBAT_HOST_DEVICE
void
{energy_name}<{d}>::gradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{{
    static_assert(
        TMatrixGF::kRows == {vecF.shape[0]} and TMatrixGF::kCols == 1, 
        "Grad w.r.t. F must have dimensions {vecF.shape[0]}x1");
    static_assert(
        TMatrixHF::kRows == {vecF.shape[0]} and TMatrixHF::kCols == {vecF.shape[0]}, 
        "Hessian w.r.t. F must have dimensions {vecF.shape[0]}x{vecF.shape[0]}");
    using ScalarType = typename TMatrix::ScalarType;
{cg.tabulate(gradhesspsi, spaces=4)}
}}
"""
        source.append(impl)

    footer = f"""
}} // namespace physics
}} // namespace pbat

#endif // PBAT_PHYSICS_{energy_name.upper()}_H
"""

    source.append(footer)

    with open(f"{energy_name}.h", mode="w") as file:
        file.write("".join(source))


if __name__ == "__main__":
    energies = [
        (stvk, "SaintVenantKirchhoffEnergy"),
        (neohookean, "StableNeoHookeanEnergy"),
    ]
    for fpsi, energy_name in energies:
        codegen(fpsi, energy_name)
