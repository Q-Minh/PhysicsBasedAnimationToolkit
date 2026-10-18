"""
SPD correction of hyperelastic Hessians.

Mirrors ``pbat::math::linalg::FilterEigenvalues`` (``math/linalg/FilterEigenvalues.h``) as used by
``pbat::fem::HyperElasticPotential`` on the 9x9 Hessian d^2Psi/dF^2. Warp only provides 3x3
eigendecompositions, so the 9x9 symmetric eigenproblem is solved by cyclic Jacobi rotations.

The static Newton solver applies it to every quadrature point's d^2Psi/dF^2 before the chain rule, with
the mode from ``pbat.sim.algorithm.newton.Params.spd_correction`` (config key
``Solver.Newton.spd_correction``; C++'s default is ``Absolute``). Tests:
``python -m unittest python.tools.sim.gpu.elasticity.spd``.
"""

import warp as wp
import numpy as np
from pbatoolkit import pbat
from .. import types

# --- SPD correction modes (pbat.fem.HyperElasticSpdCorrection) ---
SPD_CORRECTION_NONE = wp.constant(int(pbat.fem.HyperElasticSpdCorrection.NoCorrection.value))
SPD_CORRECTION_PROJECTION = wp.constant(int(pbat.fem.HyperElasticSpdCorrection.Projection.value))
SPD_CORRECTION_ABSOLUTE = wp.constant(int(pbat.fem.HyperElasticSpdCorrection.Absolute.value))

_JACOBI_MAX_SWEEPS = wp.constant(12)
# Stop once the off-diagonal Frobenius norm is below this fraction of the full norm (float32).
_JACOBI_REL_TOL2 = wp.constant(wp.float32(1e-14))


@wp.func
def sym_eig9(A: types.mat9x9f):  # pyright: ignore[reportInvalidTypeForm]
    """Eigendecomposition A = V diag(d) V^T of a symmetric 9x9 matrix by cyclic Jacobi.

    Sweeps stop once the off-diagonal Frobenius norm falls below about 1e-7 of the matrix's norm, or
    after 12 sweeps.

    Args:
        A (mat9x9f): Symmetric matrix (only symmetric input is meaningful).

    Returns:
        (vec9f, mat9x9f): Eigenvalues d (unsorted) and the orthonormal eigenvectors as the columns of V.
    """
    V = wp.identity(n=9, dtype=wp.float32)
    norm2 = wp.ddot(A, A)
    for sweep in range(_JACOBI_MAX_SWEEPS):
        off2 = wp.float32(0)
        for p in range(9):
            for q in range(9):
                if p != q:
                    off2 += A[p, q] * A[p, q]
        if off2 <= _JACOBI_REL_TOL2 * norm2:
            break
        for p in range(8):
            for q in range(p + 1, 9):
                apq = A[p, q]
                if apq != wp.float32(0):
                    # Rotation J (J_pp = J_qq = c, J_pq = s, J_qp = -s) that zeroes A'_pq in J^T A J
                    theta = (A[q, q] - A[p, p]) / (wp.float32(2) * apq)
                    t = wp.float32(1) / (wp.abs(theta) + wp.sqrt(theta * theta + wp.float32(1)))
                    if theta < wp.float32(0):
                        t = -t
                    c = wp.float32(1) / wp.sqrt(t * t + wp.float32(1))
                    s = t * c
                    for k in range(9):
                        akp = A[k, p]
                        akq = A[k, q]
                        A[k, p] = c * akp - s * akq
                        A[k, q] = s * akp + c * akq
                    for k in range(9):
                        apk = A[p, k]
                        aqk = A[q, k]
                        A[p, k] = c * apk - s * aqk
                        A[q, k] = s * apk + c * aqk
                    for k in range(9):
                        vkp = V[k, p]
                        vkq = V[k, q]
                        V[k, p] = c * vkp - s * vkq
                        V[k, q] = s * vkp + c * vkq
    d = types.vec9f()
    for i in range(9):
        d[i] = A[i, i]
    return d, V


@wp.func
def filter_eigenvalues9(A: types.mat9x9f, mode: wp.int32):  # pyright: ignore[reportInvalidTypeForm]
    """SPD-correct a symmetric 9x9 matrix.

    ``SPD_CORRECTION_PROJECTION`` clamps negative eigenvalues to 0 (nearest PSD matrix in the
    2-norm), ``SPD_CORRECTION_ABSOLUTE`` flips their sign, and ``SPD_CORRECTION_NONE`` returns A.

    Args:
        A (mat9x9f): Symmetric matrix, e.g. d^2Psi/dF^2.
        mode (int32): One of the ``SPD_CORRECTION_*`` constants (``pbat.fem.HyperElasticSpdCorrection``
            values).

    Returns:
        mat9x9f: The corrected matrix. A itself if it has no negative eigenvalue or ``mode`` is none.
    """
    if mode == SPD_CORRECTION_NONE:
        return A
    d, V = sym_eig9(A)
    has_negative = bool(False)
    for i in range(9):
        if d[i] < wp.float32(0):
            has_negative = True
            if mode == SPD_CORRECTION_PROJECTION:
                d[i] = wp.float32(0)
            else:
                d[i] = -d[i]
    if not has_negative:
        return A
    B = types.mat9x9f()
    for i in range(9):
        for j in range(9):
            bij = wp.float32(0)
            for k in range(9):
                bij += V[i, k] * d[k] * V[j, k]
            B[i, j] = bij
    return B


import unittest


@wp.kernel
def _test_filter_kernel(
    A: wp.array[types.mat9x9f],  # pyright: ignore[reportInvalidTypeForm]
    mode: wp.int32,
    B: wp.array[types.mat9x9f],  # pyright: ignore[reportInvalidTypeForm]
):
    i = wp.tid()
    B[i] = filter_eigenvalues9(A[i], mode)


class TestFilterEigenvalues(unittest.TestCase):
    def _filter(self, A: np.ndarray, mode) -> np.ndarray:
        a = wp.array(A.astype(np.float32), dtype=types.mat9x9f)
        b = wp.zeros_like(a)
        wp.launch(_test_filter_kernel, dim=a.shape[0], inputs=[a, int(mode.value), b])
        return b.numpy().astype(np.float64)

    def _reference(self, A: np.ndarray, mode) -> np.ndarray:
        d, V = np.linalg.eigh(A)
        if mode == pbat.fem.HyperElasticSpdCorrection.Projection:
            d = np.maximum(d, 0)
        elif mode == pbat.fem.HyperElasticSpdCorrection.Absolute:
            d = np.abs(d)
        return np.einsum("bik,bk,bjk->bij", V, d, V)

    def test_random_symmetric(self):
        rng = np.random.default_rng(0)
        R = rng.standard_normal((64, 9, 9))
        A = 0.5 * (R + R.transpose(0, 2, 1))
        A[:16] = A[:16] @ A[:16].transpose(0, 2, 1)  # some already-PSD inputs
        A = A.astype(np.float32).astype(np.float64)
        for mode in pbat.fem.HyperElasticSpdCorrection.__members__.values():
            B = self._filter(A, mode)
            Bref = self._reference(A, mode)
            scale = np.abs(A).max(axis=(1, 2), keepdims=True)
            self.assertLess(np.abs(B - Bref).max() / scale.min(), 1e-4, mode.name)
            if mode != pbat.fem.HyperElasticSpdCorrection.NoCorrection:
                dmin = np.linalg.eigvalsh(0.5 * (B + B.transpose(0, 2, 1))).min(axis=1)
                self.assertTrue(np.all(dmin >= -1e-4 * scale.ravel()), mode.name)
        # Already-PSD matrices are returned unchanged
        B = self._filter(A[:16], pbat.fem.HyperElasticSpdCorrection.Projection)
        self.assertTrue(np.all(B == A[:16]))


if __name__ == "__main__":
    wp.init()
    unittest.main()
