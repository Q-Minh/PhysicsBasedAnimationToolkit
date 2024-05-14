import sympy as sp
from sympy.integrals import quadrature

def shifted_gauss_quadrature(order: int, dims: int = 1, ndigits=10):
    x, w = quadrature.gauss_legendre(order, ndigits)
    a, b = 0, 1
    # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
    x = [(b-a)/2 * xi + (a+b)/2 for xi in x]
    w = [(b-a)/2*wi for wi in w]
    if dims == 2:
        x = [[xi, xj] for xi in x for xj in x]
        w = [wi*wj for wi in w for wj in w]
    if dims == 3:
        x = [[xi, xj, xk] for xi in x for xj in x for xk in x]
        w = [wi*wj*wk for wi in w for wj in w for wk in w]
    return (x,w)

if __name__ == "__main__":
    pass