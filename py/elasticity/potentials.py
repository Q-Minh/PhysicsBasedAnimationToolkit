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
    E = (F.transpose() * F -
         sp.eye(F.shape[0])) / 2
    return mu*(E.transpose()*E).trace() + (llambda / 2) * E.trace()**2


def neohookean(F, mu, llambda):
    gamma = 1 + mu/llambda
    d = F.shape[0]
    return (mu/2) * (I2(F) - d) + (llambda / 2) * (I3(F) - gamma)**2


def codegen(fpsi, energy_name: str):
    d = 3
    mu, llambda = sp.symbols(
        "mu lambda", real=True)
    vecF = sp.Matrix(
        sp.MatrixSymbol("F", d*d, 1))
    F = vecF.reshape(d, d).transpose()
    psi = fpsi(F, mu, llambda)
    gradpsi = sp.derive_by_array(psi, vecF)
    hesspsi = sp.derive_by_array(gradpsi, vecF)[:, 0, :, 0]
    psicode = cg.codegen(psi, lhs=sp.Symbol("psi"))
    gradpsicode = cg.codegen(
        gradpsi, lhs=sp.MatrixSymbol("G", *gradpsi.shape))
    hesspsicode = cg.codegen(hesspsi.transpose(
    ), lhs=sp.MatrixSymbol("H", vecF.shape[0], vecF.shape[0]))
    evalgradpsi = cg.codegen([psi, gradpsi], lhs=[sp.Symbol(
        "psi"), sp.MatrixSymbol("G", *gradpsi.shape)])
    evalgradhesspsi = cg.codegen([psi, gradpsi, hesspsi], lhs=[sp.Symbol(
        "psi"), sp.MatrixSymbol("G", *gradpsi.shape), sp.MatrixSymbol("H", vecF.shape[0], vecF.shape[0])])

    source = f"""
#ifndef PBA_CORE_PHYSICS_{energy_name.upper()}_H
#define PBA_CORE_PHYSICS_{energy_name.upper()}_H

#include "pba/aliases.h"

#include <cmath>
#include <tuple>

namespace pba {{
namespace physics {{

struct {energy_name}
{{
    public:
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
}};

template <class Derived>
Scalar
{energy_name}::eval(
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
{energy_name}::grad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Vector<{vecF.shape[0]}> G;
{cg.tabulate(gradpsicode, spaces=4)}
    return G;
}}

template <class Derived>
Matrix<{vecF.shape[0]},{vecF.shape[0]}>
{energy_name}::hessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Matrix<{vecF.shape[0]},{vecF.shape[0]}> H;
{cg.tabulate(hesspsicode, spaces=4)}
    return H;
}}

template <class Derived>
std::tuple<Scalar, Vector<{vecF.shape[0]}>>
{energy_name}::evalWithGrad(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Scalar psi;
    Vector<{vecF.shape[0]}> G;
{cg.tabulate(evalgradpsi, spaces=4)}
    return {{psi, G}};
}}

template <class Derived>
std::tuple<Scalar, Vector<{vecF.shape[0]}>, Matrix<{vecF.shape[0]},{vecF.shape[0]}>>
{energy_name}::evalWithGradAndHessian(
    Eigen::DenseBase<Derived> const& F,
    Scalar mu,
    Scalar lambda) const
{{
    Scalar psi;
    Vector<{vecF.shape[0]}> G;
    Matrix<{vecF.shape[0]},{vecF.shape[0]}> H;
{cg.tabulate(evalgradhesspsi, spaces=4)}
    return {{psi, G, H}};
}}

}} // physics
}} // pba

#endif // PBA_CORE_PHYSICS_{energy_name.upper()}_H
"""

    with open(f"{energy_name}.h", mode="w") as file:
        file.write(source)


if __name__ == "__main__":
    energies = [
        (stvk, "SaintVenantKirchhoffEnergy"),
        (neohookean, "StableNeoHookeanEnergy")
    ]
    for fpsi, energy_name in energies:
        codegen(fpsi, energy_name)
