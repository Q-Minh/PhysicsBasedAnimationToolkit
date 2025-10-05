import sympy as sp
from .. import codegen as cg


def neohookean(sigma: sp.Matrix, mu: float, lam: float) -> sp.Expr:
    """
    Compute the Neo-Hookean energy density given the deformation gradient.

    Parameters:
    sigma (sp.Matrix): `d x d` singular value matrix of the deformation gradient.
    mu (float): The shear modulus.
    lam (float): The first Lamé parameter.

    Returns:
    sp.Expr: The Neo-Hookean energy density.
    """
    # \Psi = \frac{\mu}{2} ( \sum_i \sigma_i^2 - d ) - \frac{\lambda}{2} ( \prod_i \sigma_i - 1 )^2
    gamma = 1 + mu / lam
    d = sigma.shape[0]
    return (
        mu / 2 * (sp.Trace(sigma.T @ sigma) - d) + lam / 2 * (sigma.det() - gamma) ** 2
    ).doit()


def stvk(sigma: sp.Matrix, mu: float, lam: float) -> sp.Expr:
    """
    Compute the St. Venant-Kirchhoff energy density given the deformation gradient.

    Parameters:
    sigma (sp.Matrix): `d x d` singular value matrix of the deformation gradient.
    mu (float): The shear modulus.
    lam (float): The first Lamé parameter.

    Returns:
    sp.Expr: The St. Venant-Kirchhoff energy density.
    """
    # \Psi = \frac{\mu}{2} ( \sum_i (\sigma_i^2 - 1) ) + \frac{\lambda}{8} ( \sum_i (\sigma_i^2 - 1) )^2
    d = sigma.shape[0]
    S = sigma
    STS = sigma.T @ sigma
    Psi1 = mu * (sp.Trace(STS.T @ STS) - 2 * sp.Trace(S.T @ S) + d)
    Psi2 = lam / 2 * (sp.Trace(S.T @ S) ** 2 - 2 * d * sp.Trace(S.T @ S) + d**2)
    return (Psi1 + Psi2).doit()


def least_squares_error_against_linear_approx(
    k: sp.Expr, Psi: sp.Expr, sigmai: sp.Expr, sigmal: sp.Expr, sigmau: sp.Expr
):
    line = k * (sigmai - 1)
    integrand = (line - Psi) ** 2
    return sp.integrate(integrand.expand().doit(), (sigmai, sigmal, sigmau)).doit()


def stiffness(
    Psi: sp.Expr, k: sp.Expr, sigmai: sp.Expr, sigmal: sp.Expr, sigmau: sp.Expr
) -> sp.Expr:
    E = least_squares_error_against_linear_approx(k, Psi, sigmai, sigmal, sigmau)
    kstar = sp.Abs(sp.solve(sp.diff(E, k), k)[0])
    return kstar


if __name__ == "__main__":
    d = 3
    sigma = sp.MatrixSymbol("sigma", d, 1)
    mu, lam, gamma = sp.symbols("mu lambda gamma")
    k, sigmal, sigmau = sp.symbols("k sigmalo sigmahi")
    S = sp.Matrix([[sigma[0, 0], 0, 0], [0, sigma[1, 0], 0], [0, 0, sigma[2, 0]]])
    PsiSNH = neohookean(S, mu, lam)
    kstarSNH = stiffness(PsiSNH, k, sigma[0, 0], sigmal, sigmau)
    PsiSTVK = stvk(S, mu, lam)
    kstarSTVK = stiffness(PsiSTVK, k, sigma[0, 0], sigmal, sigmau)
    codeSNH = cg.codegen(kstarSNH, lhs=sp.Symbol("kstar"), scalar_type="TScalar")
    codeSTVK = cg.codegen(kstarSTVK, lhs=sp.Symbol("kstar"), scalar_type="TScalar")
    print("SNH:\n{}\n".format(codeSNH))
    print("STVK:\n{}\n".format(codeSTVK))
