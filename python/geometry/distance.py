# type: ignore
from .. import codegen as cg
import sympy as sp


def point_point_distance():
    """
    Point-point distance: ||x - y||
    """
    x_0, x_1, x_2 = sp.symbols("x_0 x_1 x_2", real=True)
    y_0, y_1, y_2 = sp.symbols("y_0 y_1 y_2", real=True)
    x = sp.Matrix([x_0, x_1, x_2])
    y = sp.Matrix([y_0, y_1, y_2])

    diff = x - y
    d = diff.norm()

    grad_d = sp.derive_by_array(d, sp.Matrix([x, y]))
    hess_d = sp.derive_by_array(grad_d, sp.Matrix([x, y]))[:, 0, :, 0]

    d_code = cg.codegen([d], lhs=sp.Symbol("d"))
    grad_d_code = cg.codegen(grad_d, lhs=sp.MatrixSymbol("grad_d", *grad_d.shape))
    hess_d_code = cg.codegen(
        hess_d.transpose(), lhs=sp.MatrixSymbol("hess_d", 6, 6))

    print("=== Point-Point Distance ===")
    print(f"d code:\n{d_code}\n")
    print(f"grad_d code:\n{grad_d_code}\n")
    print(f"hess_d code:\n{hess_d_code}\n")


def point_edge_distance():
    """
    Point-edge distance: ||(b - a) x (x - a)|| / ||b - a||
    where (a, b) are edge endpoints and x is a point.
    """
    a_0, a_1, a_2 = sp.symbols("a_0 a_1 a_2", real=True)
    b_0, b_1, b_2 = sp.symbols("b_0 b_1 b_2", real=True)
    x_0, x_1, x_2 = sp.symbols("x_0 x_1 x_2", real=True)
    a = sp.Matrix([a_0, a_1, a_2])
    b = sp.Matrix([b_0, b_1, b_2])
    x = sp.Matrix([x_0, x_1, x_2])

    ab = b - a
    ax = x - a
    cross = ab.cross(ax)
    numerator = cross.norm()
    denominator = ab.norm()
    d = numerator / denominator

    vars = sp.Matrix([a, b, x])
    grad_d = sp.derive_by_array(d, vars)
    hess_d = sp.derive_by_array(grad_d, vars)[:, 0, :, 0]

    d_code = cg.codegen([d], lhs=sp.Symbol("d"))
    grad_d_code = cg.codegen(grad_d, lhs=sp.MatrixSymbol("grad_d", *grad_d.shape))
    hess_d_code = cg.codegen(
        hess_d.transpose(), lhs=sp.MatrixSymbol("hess_d", 9, 9))

    print("=== Point-Edge Distance ===")
    print(f"d code:\n{d_code}\n")
    print(f"grad_d code:\n{grad_d_code}\n")
    print(f"hess_d code:\n{hess_d_code}\n")


def point_triangle_distance():
    """
    Point-triangle distance: n_hat^T (x - a)
    where n_hat = n / ||n|| is the normalized triangle normal,
    n = (b - a) x (c - a) is the triangle normal,
    and x is a point.
    """
    a_0, a_1, a_2 = sp.symbols("a_0 a_1 a_2", real=True)
    b_0, b_1, b_2 = sp.symbols("b_0 b_1 b_2", real=True)
    c_0, c_1, c_2 = sp.symbols("c_0 c_1 c_2", real=True)
    x_0, x_1, x_2 = sp.symbols("x_0 x_1 x_2", real=True)
    a = sp.Matrix([a_0, a_1, a_2])
    b = sp.Matrix([b_0, b_1, b_2])
    c = sp.Matrix([c_0, c_1, c_2])
    x = sp.Matrix([x_0, x_1, x_2])

    ab = b - a
    ac = c - a
    n = ab.cross(ac)
    n_norm = n.norm()
    n_hat = n / n_norm
    d = (n_hat.T @ (x - a))[0, 0]

    vars = sp.Matrix([a, b, c, x])
    grad_d = sp.derive_by_array(d, vars)
    hess_d = sp.derive_by_array(grad_d, vars)[:, 0, :, 0]

    d_code = cg.codegen([d], lhs=sp.Symbol("d"))
    grad_d_code = cg.codegen(grad_d, lhs=sp.MatrixSymbol("grad_d", *grad_d.shape))
    hess_d_code = cg.codegen(
        hess_d.transpose(), lhs=sp.MatrixSymbol("hess_d", 12, 12))

    print("=== Point-Triangle Distance ===")
    print(f"d code:\n{d_code}\n")
    print(f"grad_d code:\n{grad_d_code}\n")
    print(f"hess_d code:\n{hess_d_code}\n")


def edge_edge_distance():
    """
    Edge-edge distance: n_hat^T (c - a)
    where n_hat = n / ||n||_eps is the approximately normalized edge-edge normal,
    n = (b - a) x (d - c) is the edge-edge normal,
    (a, b) are the first edge's endpoints,
    (c, d) are the second edge's endpoints,
    and ||n||_eps = sqrt(sum_i(n_i * n_i) + eps^2) is the mollified norm.
    """
    a_0, a_1, a_2 = sp.symbols("a_0 a_1 a_2", real=True)
    b_0, b_1, b_2 = sp.symbols("b_0 b_1 b_2", real=True)
    c_0, c_1, c_2 = sp.symbols("c_0 c_1 c_2", real=True)
    d_0, d_1, d_2 = sp.symbols("d_0 d_1 d_2", real=True)
    eps = sp.Symbol("eps", real=True, positive=True)
    a = sp.Matrix([a_0, a_1, a_2])
    b = sp.Matrix([b_0, b_1, b_2])
    c = sp.Matrix([c_0, c_1, c_2])
    d = sp.Matrix([d_0, d_1, d_2])

    ab = b - a
    cd = d - c
    n = ab.cross(cd)
    n_norm_eps = sp.sqrt((n.T @ n)[0, 0] + eps**2)
    n_hat = n / n_norm_eps
    dist = (n_hat.T @ (c - a))[0, 0]

    vars = sp.Matrix([a, b, c, d])
    grad_d = sp.derive_by_array(dist, vars)
    hess_d = sp.derive_by_array(grad_d, vars)[:, 0, :, 0]

    d_code = cg.codegen([dist], lhs=sp.Symbol("d"))
    grad_d_code = cg.codegen(grad_d, lhs=sp.MatrixSymbol("grad_d", *grad_d.shape))
    hess_d_code = cg.codegen(
        hess_d.transpose(), lhs=sp.MatrixSymbol("hess_d", 12, 12))

    print("=== Edge-Edge Distance ===")
    print(f"d code:\n{d_code}\n")
    print(f"grad_d code:\n{grad_d_code}\n")
    print(f"hess_d code:\n{hess_d_code}\n")


if __name__ == "__main__":
    point_point_distance()
    print("\n" + "="*60 + "\n")
    point_edge_distance()
    print("\n" + "="*60 + "\n")
    point_triangle_distance()
    print("\n" + "="*60 + "\n")
    edge_edge_distance()
