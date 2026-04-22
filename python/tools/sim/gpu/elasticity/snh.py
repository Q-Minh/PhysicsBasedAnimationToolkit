import warp as wp
from .. import types


@wp.func
def snh_eval(F: wp.mat33f, mu: wp.float32, llambda: wp.float32):
    I3 = wp.determinant(F)  # pyright: ignore[reportArgumentType]
    I2 = wp.ddot(F, F)  # pyright: ignore[reportArgumentType]
    I3min1 = I3 - float(1)
    psi = (
        float(0.5) * mu * (I2 - float(3))
        - mu * I3min1
        + float(0.5) * llambda * I3min1 * I3min1
    )
    return psi


@wp.func
def snh_grad(F: wp.mat33f, mu: wp.float32, llambda: wp.float32):
    I3 = wp.determinant(F)  # pyright: ignore[reportArgumentType]
    I3minAlpha = I3 - wp.float32(1) - mu / llambda
    f0 = F[:, 0]  # pyright: ignore[reportIndexIssue]
    f1 = F[:, 1]  # pyright: ignore[reportIndexIssue]
    f2 = F[:, 2]  # pyright: ignore[reportIndexIssue]
    # Cross-product columns of F
    c0 = wp.cross(f1, f2)  # pyright: ignore[reportArgumentType]
    c1 = wp.cross(f2, f0)  # pyright: ignore[reportArgumentType]
    c2 = wp.cross(f0, f1)  # pyright: ignore[reportArgumentType]
    g = types.vec9f()
    for d in range(3):
        g[d] = (
            mu * f0[d]
            + llambda * I3minAlpha * c0[d]  # pyright: ignore[reportIndexIssue]
        )
    for d in range(3):
        g[3 + d] = (
            mu * f1[d]
            + llambda * I3minAlpha * c1[d]  # pyright: ignore[reportIndexIssue]
        )
    for d in range(3):
        g[6 + d] = (
            mu * f2[d]
            + llambda * I3minAlpha * c2[d]  # pyright: ignore[reportIndexIssue]
        )
    return g


@wp.func
def snh_hess(F: wp.mat33f, mu: wp.float32, llambda: wp.float32):
    """Hessian d^2Psi/dF^2 as a 9x9 matrix (column-major F flattening)."""
    I3 = wp.determinant(F)  # pyright: ignore[reportArgumentType]
    I3minAlpha = I3 - float(1) - mu / llambda
    c = llambda * I3minAlpha
    f0 = F[:, 0]  # pyright: ignore[reportIndexIssue]
    f1 = F[:, 1]  # pyright: ignore[reportIndexIssue]
    f2 = F[:, 2]  # pyright: ignore[reportIndexIssue]
    # Cross-product columns of F
    c0 = wp.cross(f1, f2)  # pyright: ignore[reportArgumentType]
    c1 = wp.cross(f2, f0)  # pyright: ignore[reportArgumentType]
    c2 = wp.cross(f0, f1)  # pyright: ignore[reportArgumentType]
    # Flatten Fcross column-major
    fc = types.vec9f(
        c0[0],  # pyright: ignore[reportIndexIssue]
        c0[1],  # pyright: ignore[reportIndexIssue]
        c0[2],  # pyright: ignore[reportIndexIssue]
        c1[0],  # pyright: ignore[reportIndexIssue]
        c1[1],  # pyright: ignore[reportIndexIssue]
        c1[2],  # pyright: ignore[reportIndexIssue]
        c2[0],  # pyright: ignore[reportIndexIssue]
        c2[1],  # pyright: ignore[reportIndexIssue]
        c2[2],  # pyright: ignore[reportIndexIssue]
    )
    # Skew-symmetric matrices: [v]_x
    # skew(f2) -> H10 block, skew(f0) -> H21 block, skew(f1) -> H02 block
    S2 = wp.skew(f2)  # pyright: ignore[reportArgumentType]
    S0 = wp.skew(f0)  # pyright: ignore[reportArgumentType]
    S1 = wp.skew(f1)  # pyright: ignore[reportArgumentType]
    # Build H = 0
    H = types.mat99f()
    # Off-diagonal skew blocks
    for i in range(3):
        for j in range(3):
            # H10 = c * S2, H01 = -c * S2
            H[3 + i, j] = c * S2[i, j]  # pyright: ignore[reportIndexIssue]
            H[i, 3 + j] = -c * S2[i, j]  # pyright: ignore[reportIndexIssue]
            # H21 = c * S0, H12 = -c * S0
            H[6 + i, 3 + j] = c * S0[i, j]  # pyright: ignore[reportIndexIssue]
            H[3 + i, 6 + j] = -c * S0[i, j]  # pyright: ignore[reportIndexIssue]
            # H02 = c * S1, H20 = -c * S1
            H[i, 6 + j] = c * S1[i, j]  # pyright: ignore[reportIndexIssue]
            H[6 + i, j] = -c * S1[i, j]  # pyright: ignore[reportIndexIssue]
    # Outer product: lambda * fc * fc^T
    H += llambda * wp.outer(
        fc, fc  # pyright: ignore[reportArgumentType,reportOperatorIssue]
    )
    # Diagonal: += mu * I
    for i in range(9):
        H[i, i] += mu  # pyright: ignore[reportIndexIssue]
    return H


@wp.func
def snh_grad_and_hess(F: wp.mat33f, mu: wp.float32, llambda: wp.float32):
    """Evaluate gradient and hessian together."""
    I3 = wp.determinant(F)  # pyright: ignore[reportArgumentType]
    I3minAlpha = I3 - float(1) - mu / llambda
    f0 = F[:, 0]  # pyright: ignore[reportIndexIssue]
    f1 = F[:, 1]  # pyright: ignore[reportIndexIssue]
    f2 = F[:, 2]  # pyright: ignore[reportIndexIssue]
    # Cross-product columns of F
    c0 = wp.cross(f1, f2)  # pyright: ignore[reportArgumentType]
    c1 = wp.cross(f2, f0)  # pyright: ignore[reportArgumentType]
    c2 = wp.cross(f0, f1)  # pyright: ignore[reportArgumentType]
    # gradient
    g = types.vec9f()
    for d in range(3):
        g[d] = (
            mu * f0[d]
            + llambda * I3minAlpha * c0[d]  # pyright: ignore[reportIndexIssue]
        )
    for d in range(3):
        g[3 + d] = (
            mu * f1[d]
            + llambda * I3minAlpha * c1[d]  # pyright: ignore[reportIndexIssue]
        )
    for d in range(3):
        g[6 + d] = (
            mu * f2[d]
            + llambda * I3minAlpha * c2[d]  # pyright: ignore[reportIndexIssue]
        )
    # Hessian
    c = llambda * I3minAlpha
    fc = types.vec9f(
        c0[0],  # pyright: ignore[reportIndexIssue]
        c0[1],  # pyright: ignore[reportIndexIssue]
        c0[2],  # pyright: ignore[reportIndexIssue]
        c1[0],  # pyright: ignore[reportIndexIssue]
        c1[1],  # pyright: ignore[reportIndexIssue]
        c1[2],  # pyright: ignore[reportIndexIssue]
        c2[0],  # pyright: ignore[reportIndexIssue]
        c2[1],  # pyright: ignore[reportIndexIssue]
        c2[2],  # pyright: ignore[reportIndexIssue]
    )
    S2 = wp.skew(f2)  # pyright: ignore[reportArgumentType]
    S0 = wp.skew(f0)  # pyright: ignore[reportArgumentType]
    S1 = wp.skew(f1)  # pyright: ignore[reportArgumentType]
    H = types.mat99f()
    for i in range(3):
        for j in range(3):
            # H10 = c * S2, H01 = -c * S2
            H[3 + i, j] = c * S2[i, j]  # pyright: ignore[reportIndexIssue]
            H[i, 3 + j] = -c * S2[i, j]  # pyright: ignore[reportIndexIssue]
            # H21 = c * S0, H12 = -c * S0
            H[6 + i, 3 + j] = c * S0[i, j]  # pyright: ignore[reportIndexIssue]
            H[3 + i, 6 + j] = -c * S0[i, j]  # pyright: ignore[reportIndexIssue]
            # H02 = c * S1, H20 = -c * S1
            H[i, 6 + j] = c * S1[i, j]  # pyright: ignore[reportIndexIssue]
            H[6 + i, j] = -c * S1[i, j]  # pyright: ignore[reportIndexIssue]
    H += llambda * wp.outer(
        fc, fc  # pyright: ignore[reportArgumentType,reportOperatorIssue]
    )
    for i in range(9):
        H[i, i] += mu  # pyright: ignore[reportIndexIssue]
    return g, H


@wp.func
def snh_eval_with_grad_and_hess(F: wp.mat33f, mu: wp.float32, llambda: wp.float32):
    """Evaluate energy, gradient, and hessian together."""
    I3 = wp.determinant(F)  # pyright: ignore[reportArgumentType]
    I3min1 = I3 - float(1)
    I3minAlpha = I3min1 - mu / llambda
    f0 = F[:, 0]  # pyright: ignore[reportIndexIssue]
    f1 = F[:, 1]  # pyright: ignore[reportIndexIssue]
    f2 = F[:, 2]  # pyright: ignore[reportIndexIssue]
    # Cross-product columns of F
    c0 = wp.cross(f1, f2)  # pyright: ignore[reportArgumentType]
    c1 = wp.cross(f2, f0)  # pyright: ignore[reportArgumentType]
    c2 = wp.cross(f0, f1)  # pyright: ignore[reportArgumentType]
    # gradient
    g = types.vec9f()
    for d in range(3):
        g[d] = (
            mu * f0[d]
            + llambda * I3minAlpha * c0[d]  # pyright: ignore[reportIndexIssue]
        )
    for d in range(3):
        g[3 + d] = (
            mu * f1[d]
            + llambda * I3minAlpha * c1[d]  # pyright: ignore[reportIndexIssue]
        )
    for d in range(3):
        g[6 + d] = (
            mu * f2[d]
            + llambda * I3minAlpha * c2[d]  # pyright: ignore[reportIndexIssue]
        )
    # Hessian
    c = llambda * I3minAlpha
    fc = types.vec9f(
        c0[0],  # pyright: ignore[reportIndexIssue]
        c0[1],  # pyright: ignore[reportIndexIssue]
        c0[2],  # pyright: ignore[reportIndexIssue]
        c1[0],  # pyright: ignore[reportIndexIssue]
        c1[1],  # pyright: ignore[reportIndexIssue]
        c1[2],  # pyright: ignore[reportIndexIssue]
        c2[0],  # pyright: ignore[reportIndexIssue]
        c2[1],  # pyright: ignore[reportIndexIssue]
        c2[2],  # pyright: ignore[reportIndexIssue]
    )
    S2 = wp.skew(f2)  # pyright: ignore[reportArgumentType]
    S0 = wp.skew(f0)  # pyright: ignore[reportArgumentType]
    S1 = wp.skew(f1)  # pyright: ignore[reportArgumentType]
    H = types.mat99f()
    for i in range(3):
        for j in range(3):
            # H10 = c * S2, H01 = -c * S2
            H[3 + i, j] = c * S2[i, j]  # pyright: ignore[reportIndexIssue]
            H[i, 3 + j] = -c * S2[i, j]  # pyright: ignore[reportIndexIssue]
            # H21 = c * S0, H12 = -c * S0
            H[6 + i, 3 + j] = c * S0[i, j]  # pyright: ignore[reportIndexIssue]
            H[3 + i, 6 + j] = -c * S0[i, j]  # pyright: ignore[reportIndexIssue]
            # H02 = c * S1, H20 = -c * S1
            H[i, 6 + j] = c * S1[i, j]  # pyright: ignore[reportIndexIssue]
            H[6 + i, j] = -c * S1[i, j]  # pyright: ignore[reportIndexIssue]
    H += llambda * wp.outer(
        fc, fc  # pyright: ignore[reportArgumentType,reportOperatorIssue]
    )
    for i in range(9):
        H[i, i] += mu  # pyright: ignore[reportIndexIssue]
    # Eval
    I2 = wp.ddot(F, F)  # pyright: ignore[reportArgumentType]
    psi = (
        float(0.5) * mu * (I2 - float(3))
        - mu * I3min1
        + float(0.5) * llambda * I3min1 * I3min1
    )
    return psi, g, H


# --- Unit tests ---
import unittest
import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _snh_eval_np(F: np.ndarray, mu: float, lam: float) -> float:
    I2 = np.sum(F * F)
    I3 = np.linalg.det(F)
    I3min1 = I3 - 1.0
    return 0.5 * mu * (I2 - 3.0) - mu * I3min1 + 0.5 * lam * I3min1 * I3min1


def _snh_grad_np(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    I3 = np.linalg.det(F)
    I3minAlpha = I3 - 1.0 - mu / lam
    Fcross = np.empty((3, 3))
    Fcross[:, 0] = np.cross(F[:, 1], F[:, 2])
    Fcross[:, 1] = np.cross(F[:, 2], F[:, 0])
    Fcross[:, 2] = np.cross(F[:, 0], F[:, 1])
    G = mu * F + lam * I3minAlpha * Fcross
    return G.ravel(order="F")  # column-major


def _snh_hess_np(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    I3 = np.linalg.det(F)
    I3minAlpha = I3 - 1.0 - mu / lam
    c = lam * I3minAlpha
    f0, f1, f2 = F[:, 0], F[:, 1], F[:, 2]
    Fcross = np.empty((3, 3))
    Fcross[:, 0] = np.cross(f1, f2)
    Fcross[:, 1] = np.cross(f2, f0)
    Fcross[:, 2] = np.cross(f0, f1)
    fc = Fcross.ravel(order="F")
    H = np.zeros((9, 9))
    # Skew blocks
    H[3:6, 0:3] = c * _skew(f2)
    H[0:3, 3:6] = -c * _skew(f2)
    H[6:9, 3:6] = c * _skew(f0)
    H[3:6, 6:9] = -c * _skew(f0)
    H[0:3, 6:9] = c * _skew(f1)
    H[6:9, 0:3] = -c * _skew(f1)
    # Outer product
    H += lam * np.outer(fc, fc)
    # Diagonal
    H += mu * np.eye(9)
    return H


@wp.kernel
def _test_snh_kernel(
    F: wp.array[wp.mat33f],
    mu: wp.float32,
    llambda: wp.float32,
    psi: wp.array[wp.float32],
    grad: wp.array[types.vec9f],  # pyright: ignore[reportInvalidTypeForm]
    hess: wp.array[types.mat99f],  # pyright: ignore[reportInvalidTypeForm]
):
    g = wp.tid()
    Fg = F[g]
    psi[g] = snh_eval(  # pyright: ignore[reportIndexIssue]
        Fg, mu, llambda  # pyright: ignore[reportArgumentType]
    )
    grad[g] = snh_grad(  # pyright: ignore[reportIndexIssue]
        Fg, mu, llambda  # pyright: ignore[reportArgumentType]
    )
    hess[g] = snh_hess(  # pyright: ignore[reportIndexIssue]
        Fg, mu, llambda  # pyright: ignore[reportArgumentType]
    )


class TestStableNeoHookean(unittest.TestCase):
    def test_eval_grad_hess(self):
        np.random.seed(42)
        mu, lam = 1e5, 1e6
        # Random F close to identity (valid deformation gradient)
        F_np = np.eye(3, dtype=np.float32) + 0.1 * np.random.randn(3, 3).astype(
            np.float32
        )
        F_wp = wp.array([F_np], dtype=wp.mat33f)
        psi_out = wp.zeros(1, dtype=wp.float32)
        grad_out = wp.zeros(1, dtype=types.vec9f)
        hess_out = wp.zeros(1, dtype=types.mat99f)
        mu, lam = float(1e5), float(1e6)
        wp.launch(
            _test_snh_kernel, dim=1, inputs=[F_wp, mu, lam, psi_out, grad_out, hess_out]
        )
        wp.synchronize()
        # Ground truth
        psi_ref = _snh_eval_np(F_np, mu, lam)
        grad_ref = _snh_grad_np(F_np, mu, lam)
        hess_ref = _snh_hess_np(F_np, mu, lam)
        # Compare
        psi_gpu = psi_out.numpy()[0]
        grad_gpu = grad_out.numpy()[0]
        hess_gpu = hess_out.numpy()[0]
        self.assertAlmostEqual(float(psi_gpu), float(psi_ref), places=0)
        np.testing.assert_allclose(grad_gpu, grad_ref, rtol=1e-5)
        np.testing.assert_allclose(hess_gpu, hess_ref, rtol=1e-5)


if __name__ == "__main__":
    wp.init()
    unittest.main()
