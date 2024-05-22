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
    return mu*EddotE + (llambda / 2) * trE**2


def neohookean(F, mu, llambda):
    alpha = 1 + mu/llambda
    d = F.shape[0]
    return (mu/2) * (I2(F) - d) + (llambda / 2) * (I3(F) - alpha)**2


def codegen(fpsi, energy_name: str):
    source = []

    header = f"""
#ifndef PBA_CORE_PHYSICS_{energy_name.upper()}_H
#define PBA_CORE_PHYSICS_{energy_name.upper()}_H

#include "pba/aliases.h"

#include <cmath>
#include <tuple>

namespace pba {{
namespace physics {{

template <int Dims>
struct {energy_name};
"""
    source.append(header)

    for d in range(1, 3+1):
        mu, llambda = sp.symbols(
            "mu lambda", real=True)
        vecF = sp.Matrix(
            sp.MatrixSymbol("F", d*d, 1))
        F = vecF.reshape(d, d).transpose()
        psi = fpsi(F, mu, llambda)
        gradpsi = sp.derive_by_array(psi, vecF)
        hesspsi = sp.derive_by_array(
            gradpsi, vecF)[:, 0, :, 0]
        psicode = cg.codegen(psi, lhs=sp.Symbol("psi"))
        gradpsicode = cg.codegen(
            gradpsi, lhs=sp.MatrixSymbol("vecG", *gradpsi.shape))
        hesspsicode = cg.codegen(hesspsi.transpose(
        ), lhs=sp.MatrixSymbol("vecH", vecF.shape[0], vecF.shape[0]))
        evalgradpsi = cg.codegen([psi, gradpsi], lhs=[sp.Symbol(
            "psi"), sp.MatrixSymbol("vecG", *gradpsi.shape)])
        evalgradhesspsi = cg.codegen([psi, gradpsi, hesspsi], lhs=[
            sp.Symbol("psi"),
            sp.MatrixSymbol("vecG", *gradpsi.shape),
            sp.MatrixSymbol(
                "vecH", vecF.shape[0], vecF.shape[0])
        ])
        gradhesspsi = cg.codegen([gradpsi, hesspsi], lhs=[
            sp.MatrixSymbol("vecG", *gradpsi.shape),
            sp.MatrixSymbol(
                "vecH", vecF.shape[0], vecF.shape[0])
        ])
        impl = f"""
template <>
struct {energy_name}<{d}>
{{
    public:
        static auto constexpr kDims = {d};
    
        template <class Derived>
        Scalar
        eval(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Vector<{vecF.shape[0]}>
        grad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        Matrix<{vecF.shape[0]},{vecF.shape[0]}>
        hessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<{vecF.shape[0]}>>
        evalWithGrad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Scalar, Vector<{vecF.shape[0]}>, Matrix<{vecF.shape[0]},{vecF.shape[0]}>>
        evalWithGradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;

        template <class Derived>
        std::tuple<Vector<{vecF.shape[0]}>, Matrix<{vecF.shape[0]},{vecF.shape[0]}>>
        gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const;
}};

template <class Derived>
Scalar
{energy_name}<{d}>::eval(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Scalar psi;
{cg.tabulate(psicode, spaces=4)}
    return psi;
}}

template <class Derived>
Vector<{vecF.shape[0]}>
{energy_name}<{d}>::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Vector<{vecF.shape[0]}> G;
    auto vecG = G.reshaped();
{cg.tabulate(gradpsicode, spaces=4)}
    return G;
}}

template <class Derived>
Matrix<{vecF.shape[0]},{vecF.shape[0]}>
{energy_name}<{d}>::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Matrix<{vecF.shape[0]},{vecF.shape[0]}> H;
    auto vecH = H.reshaped();
{cg.tabulate(hesspsicode, spaces=4)}
    return H;
}}

template <class Derived>
std::tuple<Scalar, Vector<{vecF.shape[0]}>>
{energy_name}<{d}>::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Scalar psi;
    Vector<{vecF.shape[0]}> G;
    auto vecG = G.reshaped();
{cg.tabulate(evalgradpsi, spaces=4)}
    return {{psi, G}};
}}

template <class Derived>
std::tuple<Scalar, Vector<{vecF.shape[0]}>, Matrix<{vecF.shape[0]},{vecF.shape[0]}>>
{energy_name}<{d}>::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Scalar psi;
    Vector<{vecF.shape[0]}> G;
    Matrix<{vecF.shape[0]},{vecF.shape[0]}> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
{cg.tabulate(evalgradhesspsi, spaces=4)}
    return {{psi, G, H}};
}}

template <class Derived>
std::tuple<Vector<{vecF.shape[0]}>, Matrix<{vecF.shape[0]},{vecF.shape[0]}>>
{energy_name}<{d}>::gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
{{
    Vector<{vecF.shape[0]}> G;
    Matrix<{vecF.shape[0]},{vecF.shape[0]}> H;
    auto vecG = G.reshaped();
    auto vecH = H.reshaped();
{cg.tabulate(gradhesspsi, spaces=4)}
    return {{G, H}};
}}
"""
        source.append(impl)

    footer = f"""
}} // physics
}} // pba

#endif // PBA_CORE_PHYSICS_{energy_name.upper()}_H
"""

    source.append(footer)

    with open(f"{energy_name}.h", mode="w") as file:
        file.write("".join(source))


if __name__ == "__main__":
    energies = [
        (stvk, "SaintVenantKirchhoffEnergy"),
        (neohookean, "StableNeoHookeanEnergy")
    ]
    for fpsi, energy_name in energies:
        codegen(fpsi, energy_name)
