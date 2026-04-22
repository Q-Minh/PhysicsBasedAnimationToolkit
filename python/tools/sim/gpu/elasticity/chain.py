"""
Chain rule functions for mapping derivatives of Psi(F) to derivatives w.r.t. FEM dofs x.

Replicates the C++ API in pbat/fem/DeformationGradient.h for 3D tetrahedral elements
(kNodes=4, Dims=3). GP is the 4x3 matrix of basis function gradients (grad N_i).

GF = d Psi / d vec(F)  is a vec9f  (9x1, column-major flattening of 3x3)
HF = d^2 Psi / d vec(F)^2  is a mat99f  (9x9)
GP = basis function gradients  is a mat43f  (4x3, row i = grad N_i)

Output sizes:
  GradientSegmentWrtDofs: vec3f (gradient for node i)
  GradientWrtDofs: vec12f (gradient for all 4 nodes, interleaved [x0,y0,z0,x1,...])
  HessianBlockWrtDofs: mat33f (hessian block for node pair i,j)
  HessianWrtDofs: mat1212f (full 12x12 element hessian)
"""

import warp as wp
from .. import types


@wp.func
def gradient_segment_wrt_dofs(
    GF: types.vec9f, GP: types.mat4x3f, i: int  # pyright: ignore[reportInvalidTypeForm]
):
    """
    Computes dPsi/dx_i = sum_k GP[i,k] * GF[k*3 : k*3+3], returning a vec3f.
    """
    result = wp.vec3f()
    for k in range(3):
        gpik = GP[i, k]  # pyright: ignore[reportIndexIssue]
        for d in range(3):
            result[d] += gpik * GF[k * 3 + d]  # pyright: ignore[reportIndexIssue]
    return result


@wp.func
def gradient_wrt_dofs(
    GF: types.vec9f, GP: types.mat4x3f  # pyright: ignore[reportInvalidTypeForm]
):
    """
    Computes dPsi/dx for all 4 nodes, returning a vec12f.
    Layout: [dx0, dy0, dz0, dx1, dy1, dz1, ..., dx3, dy3, dz3].
    """
    result = types.vec12f()
    for k in range(3):
        for i in range(4):
            gpik = GP[i, k]  # pyright: ignore[reportIndexIssue]
            for d in range(3):
                idx = i * 3 + d
                result[idx] = (
                    result[idx] + gpik * GF[k * 3 + d]
                )  # pyright: ignore[reportIndexIssue]
    return result


@wp.func
def hessian_block_wrt_dofs(
    HF: types.mat9x9f,  # pyright: ignore[reportInvalidTypeForm]
    GP: types.mat4x3f,  # pyright: ignore[reportInvalidTypeForm]
    i: int,
    j: int,
):
    """
    Computes d^2Psi/(dx_i dx_j) = sum_{ki,kj} GP[i,ki]*GP[j,kj]*HF[ki*3:ki*3+3, kj*3:kj*3+3].
    Returns a mat33f.
    """
    result = wp.mat33f()
    for kj in range(3):
        for ki in range(3):
            c = GP[i, ki] * GP[j, kj]  # pyright: ignore[reportIndexIssue]
            for row in range(3):
                for col in range(3):
                    result[row, col] = (  # pyright: ignore[reportIndexIssue]
                        result[row, col]  # pyright: ignore[reportIndexIssue]
                        + c
                        * HF[
                            ki * 3 + row, kj * 3 + col
                        ]  # pyright: ignore[reportIndexIssue]
                    )
    return result


@wp.func
def hessian_wrt_dofs(
    HF: types.mat9x9f, GP: types.mat4x3f  # pyright: ignore[reportInvalidTypeForm]
):
    """
    Computes the full 12x12 element hessian d^2Psi/dx^2.
    Layout: rows/cols ordered as [node0_xyz, node1_xyz, node2_xyz, node3_xyz].
    """
    result = types.mat12x12f()
    for kj in range(3):
        for ki in range(3):
            for j in range(4):
                for i in range(4):
                    c = GP[i, ki] * GP[j, kj]  # pyright: ignore[reportIndexIssue]
                    for row in range(3):
                        for col in range(3):
                            ri = i * 3 + row
                            ci = j * 3 + col
                            result[ri, ci] = (  # pyright: ignore[reportIndexIssue]
                                result[ri, ci]  # pyright: ignore[reportIndexIssue]
                                + c
                                * HF[
                                    ki * 3 + row, kj * 3 + col
                                ]  # pyright: ignore[reportIndexIssue]
                            )
    return result


# --- Unit tests ---
import unittest
import numpy as np


def _gradient_segment_wrt_dofs_np(GF: np.ndarray, GP: np.ndarray, i: int) -> np.ndarray:
    """GF: (9,), GP: (4,3). Returns (3,)."""
    result = np.zeros(3)
    for k in range(3):
        result += GP[i, k] * GF[k * 3 : k * 3 + 3]
    return result


def _gradient_wrt_dofs_np(GF: np.ndarray, GP: np.ndarray) -> np.ndarray:
    """GF: (9,), GP: (4,3). Returns (12,)."""
    result = np.zeros(12)
    for k in range(3):
        for i in range(4):
            result[i * 3 : i * 3 + 3] += GP[i, k] * GF[k * 3 : k * 3 + 3]
    return result


def _hessian_block_wrt_dofs_np(
    HF: np.ndarray, GP: np.ndarray, i: int, j: int
) -> np.ndarray:
    """HF: (9,9), GP: (4,3). Returns (3,3)."""
    result = np.zeros((3, 3))
    for kj in range(3):
        for ki in range(3):
            result += (
                GP[i, ki] * GP[j, kj] * HF[ki * 3 : ki * 3 + 3, kj * 3 : kj * 3 + 3]
            )
    return result


def _hessian_wrt_dofs_np(HF: np.ndarray, GP: np.ndarray) -> np.ndarray:
    """HF: (9,9), GP: (4,3). Returns (12,12)."""
    result = np.zeros((12, 12))
    for kj in range(3):
        for ki in range(3):
            for j in range(4):
                for i in range(4):
                    result[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] += (
                        GP[i, ki]
                        * GP[j, kj]
                        * HF[ki * 3 : ki * 3 + 3, kj * 3 : kj * 3 + 3]
                    )
    return result


@wp.kernel
def _test_chain_kernel(
    GF_arr: wp.array(dtype=types.vec9f),  # pyright: ignore[reportInvalidTypeForm]
    HF_arr: wp.array(dtype=types.mat9x9f),  # pyright: ignore[reportInvalidTypeForm]
    GP_arr: wp.array(dtype=types.mat4x3f),  # pyright: ignore[reportInvalidTypeForm]
    grad_seg_out: wp.array2d(dtype=wp.vec3f),  # pyright: ignore[reportInvalidTypeForm]
    grad_out: wp.array(dtype=types.vec12f),  # pyright: ignore[reportInvalidTypeForm]
    hess_block_out: wp.array2d(
        dtype=wp.mat33f
    ),  # pyright: ignore[reportInvalidTypeForm]
    hess_out: wp.array(dtype=types.mat12x12f),  # pyright: ignore[reportInvalidTypeForm]
):
    tid = wp.tid()
    GF = GF_arr[tid]
    HF = HF_arr[tid]
    GP = GP_arr[tid]
    # Gradient segments for each node
    for i in range(4):
        grad_seg_out[tid, i] = (
            gradient_segment_wrt_dofs(  # pyright: ignore[reportIndexIssue]
                GF, GP, i  # pyright: ignore[reportArgumentType]
            )
        )
    # Full gradient
    grad_out[tid] = gradient_wrt_dofs(  # pyright: ignore[reportIndexIssue]
        GF, GP  # pyright: ignore[reportArgumentType]
    )
    # Hessian blocks for each (i,j) pair — store in flat 4x4 = 16 entries
    for i in range(4):
        for j in range(4):
            hess_block_out[tid, i * 4 + j] = (
                hessian_block_wrt_dofs(  # pyright: ignore[reportIndexIssue]
                    HF, GP, i, j  # pyright: ignore[reportArgumentType]
                )
            )
    # Full hessian
    hess_out[tid] = hessian_wrt_dofs(  # pyright: ignore[reportIndexIssue]
        HF, GP  # pyright: ignore[reportArgumentType]
    )


class TestChainRule(unittest.TestCase):
    def test_gradient_and_hessian_wrt_dofs(self):
        np.random.seed(123)
        # Random GF (9,), HF (9,9 symmetric), GP (4,3)
        GF_np = np.random.randn(9).astype(np.float32)
        H_rand = np.random.randn(9, 9).astype(np.float32)
        HF_np = (H_rand + H_rand.T) * 0.5  # symmetric
        GP_np = np.random.randn(4, 3).astype(np.float32)

        GF_wp = wp.array([GF_np], dtype=types.vec9f)
        HF_wp = wp.array([HF_np], dtype=types.mat9x9f)
        GP_wp = wp.array([GP_np], dtype=types.mat4x3f)
        grad_seg_out = wp.zeros(shape=(1, 4), dtype=wp.vec3f)
        grad_out = wp.zeros(1, dtype=types.vec12f)
        hess_block_out = wp.zeros(shape=(1, 16), dtype=wp.mat33f)
        hess_out = wp.zeros(1, dtype=types.mat12x12f)

        wp.launch(
            _test_chain_kernel,
            dim=1,
            inputs=[
                GF_wp,
                HF_wp,
                GP_wp,
                grad_seg_out,
                grad_out,
                hess_block_out,
                hess_out,
            ],
        )
        wp.synchronize()

        # Reference
        grad_ref = _gradient_wrt_dofs_np(GF_np, GP_np)
        hess_ref = _hessian_wrt_dofs_np(HF_np, GP_np)

        # Compare full gradient
        grad_gpu = grad_out.numpy()[0]
        np.testing.assert_allclose(grad_gpu, grad_ref, rtol=1e-5, atol=1e-5)

        # Compare gradient segments
        grad_seg_gpu = grad_seg_out.numpy()[0]  # (4, 3)
        for i in range(4):
            seg_ref = _gradient_segment_wrt_dofs_np(GF_np, GP_np, i)
            np.testing.assert_allclose(grad_seg_gpu[i], seg_ref, rtol=1e-5, atol=1e-5)

        # Compare full hessian
        hess_gpu = hess_out.numpy()[0]
        np.testing.assert_allclose(hess_gpu, hess_ref, rtol=1e-4, atol=1e-4)

        # Compare hessian blocks
        hess_block_gpu = hess_block_out.numpy()[0]  # (16, 3, 3)
        for i in range(4):
            for j in range(4):
                block_ref = _hessian_block_wrt_dofs_np(HF_np, GP_np, i, j)
                np.testing.assert_allclose(
                    hess_block_gpu[i * 4 + j], block_ref, rtol=1e-4, atol=1e-4
                )


if __name__ == "__main__":
    wp.init()
    unittest.main()
