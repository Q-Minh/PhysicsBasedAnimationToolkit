import warp as wp
import numpy as np
from pbatoolkit import pbat

# --- Vertex linear solver constants ---
VLS_SOLVER_INVERSE = wp.constant(
    int(pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.Inverse.value)
)
VLS_SOLVER_LLT = wp.constant(
    int(pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.LLT.value)
)
VLS_SOLVER_QR = wp.constant(
    int(pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.QR.value)
)
VLS_SOLVER_EVD = wp.constant(
    int(pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.EVD.value)
)

_VLS_SOLVER_MAP = {
    pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.Inverse: VLS_SOLVER_INVERSE,
    pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.LLT: VLS_SOLVER_LLT,
    pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.QR: VLS_SOLVER_QR,
    pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.EVD: VLS_SOLVER_EVD,
}


@wp.struct
class ParamsData:
    """From `source/pbat/sim/algorithm/vbd/Core.h`."""

    # --- Vertex-element adjacency graph ---
    GVGp: wp.array[wp.int32]  # (N+1,) prefix sums into GVGe
    GVGe: wp.array[
        wp.int32
    ]  # (# of vertex-elems adjacencies,) element indices s.t. `GVGe[k]
    # for GVGp[i] <= k < GVGp[i+1]` gives the element `e` adjacent to
    # vertex `i`

    # --- Vertex-vertex adjacency graph ---
    GVVp: wp.array[wp.int32]  # (N+1,) prefix sums into GVVadj
    GVVadj: wp.array[wp.int32]  # (# vertex-vertex adjacencies,) adjacent vertex indices

    # --- Graph coloring ---
    colors: wp.array[wp.int32]  # (N,) map of vertex colors

    # --- Partitioning ---
    Pptr: wp.array[
        wp.int32
    ]  # (# colors + 1,) partition pointers s.t. the range `[Pptr[p], Pptr[p+1])` indexes into Padj from partition/color `p`
    Padj: wp.array[wp.int32]  # (# verts,) partition vertices

    # --- Iteration control ---
    n_max_iters: wp.int32  # max outer iterations
    n_subproblem_max_iters: wp.int32  # max VBD sweeps per subproblem
    gtol: wp.float32  # gradient norm convergence threshold

    # --- Damping ---
    betaR: wp.float32  # Rayleigh damping coefficient

    # --- Vertex linear solver ---
    hess_zero: wp.float32  # Hessian determinant zero threshold
    vls_eps: wp.float32  # vertex solver epsilon
    vls_max_iters: wp.int32  # max vertex linear solver iterations
    vls_solver: wp.int32  # vertex linear solver type

    # --- Stencil gradient acceleration ---
    betaG0: (
        wp.vec2f
    )  # initial stencil gradient augmentation scales (0: interior, 1: surface)
    rhohat: wp.vec2f  # Lipschitz-normalized step thresholds (0: interior, 1: surface)
    gammadown: wp.vec2f  # beta reduction factors (0: interior, 1: surface)
    gammaup: wp.vec2f  # beta increase factors (0: interior, 1: surface)

    # --- Read-write ---
    xb: wp.array[wp.vec3f]  # (N,) vertex positions
    gk: wp.array[wp.vec3f]  # (N,) vertex gradients
    xk: wp.array[wp.vec3f]  # (N,) vertex past iteration
    Hnk: wp.array[wp.float32]  # (N,) vertex Hessian norms
    betaG: wp.array[wp.float32]  # (N,) vertex stencil gradient augmentation scales
    Hk: wp.array[wp.mat33f]  # (N,) (3x3) block-diagonal Hessian


class Params:
    """VBD solver parameters, wrapping a C++ pbat.sim.algorithm.vbd.Params object for GPU execution.

    Usage:
        params_cpu = pbat.sim.algorithm.vbd.Params()
        # configuring params_cpu ...
        params = Params(params_cpu)
        # params.data can be passed to warp kernels
    """

    _data: ParamsData  # pyright: ignore[reportGeneralTypeIssues]

    def __init__(self, params: pbat.sim.algorithm.vbd.Params):
        self._data = ParamsData()
        # Vertex-element adjacency graph
        self._data.GVGp = wp.array(params.GVGp, dtype=wp.int32)
        self._data.GVGe = wp.array(params.GVGe, dtype=wp.int32)
        # Vertex-vertex adjacency graph
        self._data.GVVp = wp.array(params.GVVp, dtype=wp.int32)
        self._data.GVVadj = wp.array(params.GVVadj, dtype=wp.int32)
        # Graph coloring
        self._data.colors = wp.array(params.colors, dtype=wp.int32)
        # Partitioning
        self._data.Pptr = wp.array(params.Pptr, dtype=wp.int32, device="cpu")
        self._data.Padj = wp.array(params.Padj, dtype=wp.int32)
        # Iteration control
        self._data.n_max_iters = int(params.n_max_iters)
        self._data.n_subproblem_max_iters = int(params.n_subproblem_max_iters)
        self._data.gtol = float(params.gtol)
        # Damping
        self._data.betaR = float(params.betaR)
        # Vertex linear solver
        self._data.hess_zero = float(params.hess_zero)
        self._data.vls_eps = float(params.vls_eps)
        self._data.vls_max_iters = int(params.vls_max_iters)
        self._data.vls_solver = int(_VLS_SOLVER_MAP[params.vlinsolve])
        # Stencil gradient acceleration
        self._data.betaG0 = (float(params.betaG0), float(params.rhohatS))
        self._data.betaG0 = (float(params.betaG0), float(params.betaG0))
        self._data.rhohat = (float(params.rhohat), float(params.rhohatS))
        self._data.gammadown = (float(params.gammadown), float(params.gammadownS))
        self._data.gammaup = (float(params.gammaup), float(params.gammaupS))
        # Read-write
        n_nodes = params.colors.shape[0]
        self._data.xb = wp.zeros((n_nodes,), dtype=wp.vec3f)  # (N,) vertex positions
        self._data.gk = wp.zeros((n_nodes,), dtype=wp.vec3f)  # (N,) vertex gradients
        self._data.xk = wp.zeros((n_nodes,), dtype=wp.vec3f)  # (N,) vertex past iterate
        self._data.Hnk = wp.zeros(
            (n_nodes,), dtype=wp.float32
        )  # (N,) vertex Hessian norms
        self._data.betaG = wp.zeros(
            (n_nodes,), dtype=wp.float32
        )  # (N,) vertex stencil gradient augmentation scales
        self._data.Hk = wp.zeros(
            (n_nodes,), dtype=wp.mat33f
        )  # (N,) (3x3) block-diagonal Hessian

    @property
    def data(self) -> ParamsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data


import unittest


class TestParams(unittest.TestCase):
    def test_construction(self):
        C = np.array(
            [
                [0, 1, 3, 5],
                [3, 2, 0, 6],
                [5, 4, 6, 0],
                [6, 7, 5, 3],
                [0, 5, 3, 6],
            ],
            dtype=np.int64,
        )
        n_nodes = 8
        E = C.T  # |nodes_per_elem| x |num_elems|
        GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
            E, n_nodes
        )
        GVVp, GVVadj, colors = pbat.sim.algorithm.vbd.vertex_colors(E, n_nodes)
        params_cpu = pbat.sim.algorithm.vbd.Params()
        params_cpu.with_vertex_element_adjacency_graph(GVGp, GVGe, GVGilocal)
        params_cpu.with_vertex_colors(GVVp, GVVadj, colors)
        params_cpu.construct()

        params = Params(params_cpu)

        # Verify adjacency graph
        self.assertTrue(np.all(params.data.GVGp.numpy() == params_cpu.GVGp))
        self.assertTrue(np.all(params.data.GVGe.numpy() == params_cpu.GVGe))
        # Verify vertex-vertex adjacency
        self.assertTrue(np.all(params.data.GVVp.numpy() == params_cpu.GVVp))
        self.assertTrue(np.all(params.data.GVVadj.numpy() == params_cpu.GVVadj))
        # Verify coloring and partitioning
        self.assertTrue(np.all(params.data.colors.numpy() == params_cpu.colors))
        self.assertTrue(np.all(params.data.Pptr.numpy() == params_cpu.Pptr))
        self.assertTrue(np.all(params.data.Padj.numpy() == params_cpu.Padj))
        # Verify scalars
        self.assertEqual(params.data.n_max_iters, int(params_cpu.n_max_iters))
        self.assertEqual(
            params.data.n_subproblem_max_iters, int(params_cpu.n_subproblem_max_iters)
        )
        self.assertAlmostEqual(params.data.gtol, float(params_cpu.gtol))
        self.assertAlmostEqual(params.data.betaR, float(params_cpu.betaR))
        self.assertAlmostEqual(params.data.hess_zero, float(params_cpu.hess_zero))
        self.assertAlmostEqual(params.data.vls_eps, float(params_cpu.vls_eps))
        self.assertEqual(params.data.vls_max_iters, int(params_cpu.vls_max_iters))
        # Verify tuples
        self.assertEqual(
            params.data.rhohat, (float(params_cpu.rhohat), float(params_cpu.rhohatS))
        )
        self.assertEqual(
            params.data.gammadown,
            (float(params_cpu.gammadown), float(params_cpu.gammadownS)),
        )
        self.assertEqual(
            params.data.gammaup, (float(params_cpu.gammaup), float(params_cpu.gammaupS))
        )


if __name__ == "__main__":
    wp.init()
    unittest.main()
