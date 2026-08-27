import warp as wp
import numpy as np
import cupy as cp
from pbatoolkit import pbat

from ...common.fields import DocField

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
    GVGp: wp.array[wp.int32]  # (N+1,) prefix sums into GVGadj
    GVGadj: wp.array[
        wp.int32
    ]  # (# of vertex-elems adjacencies,) element indices s.t. `GVGadj[k]
    # for GVGp[i] <= k < GVGp[i+1]` gives the element `e` adjacent to
    # vertex `i`

    # --- Vertex-vertex adjacency graph ---
    GVVp: wp.array[wp.int32]  # (N+1,) prefix sums into GVVadj
    GVVadj: wp.array[wp.int32]  # (# vertex-vertex adjacencies,) adjacent vertex indices

    # --- Graph coloring ---
    colors: wp.array[wp.int32]  # (N,) map of vertex colors

    # --- Partitioning ---
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
    betaG: wp.array2d[wp.float32]  # (N,2) vertex stencil gradient augmentation scales
    Qnk: wp.array[wp.float32]  # (N,) max vertex normal contact Rayleigh quotients
    Qfk: wp.array[wp.float32]  # (N,) max vertex friction contact Rayleigh quotients


class Params:
    """VBD solver parameters, wrapping a C++ pbat.sim.algorithm.vbd.Params object for GPU execution.

    Usage:
        params_cpu = pbat.sim.algorithm.vbd.Params()
        # configuring params_cpu ...
        params = Params(params_cpu)
        # params.data can be passed to warp kernels
    """

    _data: ParamsData  # pyright: ignore[reportGeneralTypeIssues]
    _Pptr: np.ndarray  # (# colors + 1,) partition pointers on CPU, not part of GPU struct

    def __init__(self, params: pbat.sim.algorithm.vbd.Params):
        self._data = ParamsData()
        # Vertex-element adjacency graph
        self._data.GVGp = wp.array(params.GVGp, dtype=wp.int32)
        self._data.GVGadj = wp.array(params.GVGe, dtype=wp.int32)
        # Vertex-vertex adjacency graph
        self._data.GVVp = wp.array(params.GVVp, dtype=wp.int32)
        self._data.GVVadj = wp.array(params.GVVadj, dtype=wp.int32)
        # Graph coloring
        self._data.colors = wp.array(params.colors, dtype=wp.int32)
        # Partitioning — Pptr stays on CPU (used only to drive partition dispatch, never in GPU kernels)
        self._Pptr = params.Pptr.copy()
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
        self._data.betaG = wp.full((n_nodes, 2), params.betaG0, dtype=wp.float32)
        # NOTE: Ideally, Qnk,Qfk would have shape (# surface verts,) but we don't have 
        # mesh information in this constructor...
        self._data.Qnk = wp.zeros(
            (n_nodes,), dtype=wp.float32
        )  # (N,) max vertex normal contact Rayleigh quotients
        self._data.Qfk = wp.zeros(
            (n_nodes,), dtype=wp.float32
        )  # (N,) max vertex friction contact Rayleigh quotients

    @property
    def data(self) -> ParamsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

    @property
    def Pptr(self) -> np.ndarray:
        return self._Pptr


def serialize_vbd_cpu_params(params: pbat.sim.algorithm.vbd.Params, grp) -> None:
    """Serialize the scalar/enum fields of a VBD CPU Params object to an h5py group.

    Only settable float, int, bool, and enum properties are written (same set that
    ``draw_params`` displays in the UI).
    """
    import inspect
    import enum

    for name, value in inspect.getmembers(params):
        if name.startswith("_"):
            continue
        descriptor = getattr(type(params), name, None)
        if isinstance(descriptor, property) and descriptor.fset is None:
            continue
        if isinstance(value, float):
            grp.attrs[name] = value
        elif isinstance(value, bool):
            grp.attrs[name] = int(value)
        elif isinstance(value, int):
            grp.attrs[name] = value
        elif isinstance(value, enum.Enum):
            grp.attrs[name] = value.value


def deserialize_vbd_cpu_params(params: pbat.sim.algorithm.vbd.Params, grp) -> None:
    """Deserialize scalar/enum fields from an h5py group into a VBD CPU Params object.

    Silently skips keys absent in the group or attributes that cannot be set.
    """
    import inspect
    import enum

    for name, value in inspect.getmembers(params):
        if name.startswith("_"):
            continue
        descriptor = getattr(type(params), name, None)
        if isinstance(descriptor, property) and descriptor.fset is None:
            continue
        if name not in grp.attrs:
            continue
        raw = grp.attrs[name]
        try:
            if isinstance(value, enum.Enum):
                setattr(params, name, type(value)(int(raw)))
            elif isinstance(value, bool):
                setattr(params, name, bool(int(raw)))
            elif isinstance(value, float):
                setattr(params, name, float(raw))
            elif isinstance(value, int):
                setattr(params, name, int(raw))
        except Exception:
            pass


class ChebyshevParams(Params):
    """VBD solver parameters augmented with Chebyshev semi-iterative acceleration state, for
    GPU execution.

    Wraps a `pbat.sim.algorithm.vbd.Params` CPU object, inheriting all VBD parameters/buffers
    (see `Params`), and additionally stores the Chebyshev-specific GPU-resident state mirroring
    `pbat::sim::algorithm::vbd::ChebyshevParams` (`source/pbat/sim/algorithm/vbd/Chebyshev.h`):
      - rho: spectral radius estimate, `0 < rho < 1`, read once from the CPU
        `pbat.sim.algorithm.vbd.ChebyshevParams` object at construction time
      - omega: current relaxation weight, recomputed every VBD sweep
      - xkm1, xkm2: previous VBD-sweep iterates used by the momentum recurrence
    """

    def __init__(
        self,
        params: pbat.sim.algorithm.vbd.Params,
        cheb_params: pbat.sim.algorithm.vbd.ChebyshevParams,
    ):
        super().__init__(params)
        self.rho = cheb_params.rho
        self.omega = 1.0
        n_nodes = params.colors.shape[0]
        self.xkm1: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)
        self.xkm2: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)


def serialize_chebyshev_cpu_params(
    vbd_params: pbat.sim.algorithm.vbd.Params,
    cheb_params: pbat.sim.algorithm.vbd.ChebyshevParams,
    grp,
) -> None:
    """Serialize a VBD CPU `Params` object plus a Chebyshev CPU `ChebyshevParams` object's
    `rho` to an h5py group."""
    serialize_vbd_cpu_params(vbd_params, grp)
    grp.attrs["rho"] = float(cheb_params.rho)


def deserialize_chebyshev_cpu_params(
    vbd_params: pbat.sim.algorithm.vbd.Params,
    cheb_params: pbat.sim.algorithm.vbd.ChebyshevParams,
    grp,
) -> None:
    """Deserialize a VBD CPU `Params` object plus a Chebyshev CPU `ChebyshevParams` object's
    `rho` from an h5py group, in place. Silently skips `rho` if absent from the group."""
    deserialize_vbd_cpu_params(vbd_params, grp)
    if "rho" in grp.attrs:
        cheb_params.rho = float(grp.attrs["rho"])


class ChebyshevCpuParams:
    """Container bundling the two CPU-side parameter objects the Chebyshev-accelerated VBD
    solver needs, so that they can be stored/passed around as a single `params_cpu` entry in
    `ui.py`.

    `rho` is exposed as a top-level property delegating to the underlying
    `pbat.sim.algorithm.vbd.ChebyshevParams.rho`, so it's drawn directly by `draw_params`,
    while `vbd_params` (the underlying `pbat.sim.algorithm.vbd.Params`) is meant to be drawn
    as a nested sub-tree via `draw_params`'s `sub_params` mechanism (see
    `SOLVER_SUB_PARAMS[SolverType.Chebyshev] = {"vbd_params": None}` in `ui.py`), the same way
    Newton's `line_search` sub-params are handled.
    """

    def __init__(
        self,
        vbd_params: pbat.sim.algorithm.vbd.Params | None = None,
        cheb_params: pbat.sim.algorithm.vbd.ChebyshevParams | None = None,
    ):
        self.vbd_params = (
            vbd_params if vbd_params is not None else pbat.sim.algorithm.vbd.Params()
        )
        self._cheb_params = (
            cheb_params
            if cheb_params is not None
            else pbat.sim.algorithm.vbd.ChebyshevParams()
        )

    @property
    def cheb_params(self) -> pbat.sim.algorithm.vbd.ChebyshevParams:
        """The underlying `pbat.sim.algorithm.vbd.ChebyshevParams` CPU object."""
        return self._cheb_params

    @property
    def rho(self) -> float:
        """Spectral radius estimate `0 < rho < 1` for Chebyshev acceleration."""
        return self._cheb_params.rho

    @rho.setter
    def rho(self, value: float):
        self._cheb_params.rho = float(value)


class AndersonParams(Params):
    """VBD solver parameters augmented with Anderson acceleration state, for GPU execution.

    Wraps a `pbat.sim.algorithm.vbd.Params` CPU object, inheriting all VBD parameters/buffers
    (see `Params`), and additionally stores the Anderson-specific GPU-resident state mirroring
    `pbat::sim::algorithm::vbd::AndersonParams` (`source/pbat/sim/algorithm/vbd/Anderson.h`):
      - m: window size, read once from the CPU `pbat.sim.algorithm.vbd.AndersonParams` object
        at construction time
      - beta: mixing parameter, read once from the CPU object at construction time
      - xkm1: `(N,)` previous step's positions
      - fk, fkm1: `(N,)` current/past residuals `x - xkm1`
      - Xk, Fk: `(m, N, 3)` CuPy ndarrays of past step/residual differences (one slot per
        window entry, indexed modulo `m` as in `pbat::common::Modulo`), so that the Anderson
        mixing step (`gpu/vbd/aasolver.py`) can run `cupy.linalg.lstsq`/matmul directly on them
        as `(3N, mk)` matrices. Per-slot `wp.array` vec3f views (`Xk`/`Fk`, zero-copy over the
        same device memory as `Xk_cp`/`Fk_cp`) are exposed so the per-vertex bookkeeping
        kernels can keep writing into them exactly as before.
      - gammak: `(m,)` warp array of subspace mixing coefficients, solved for via
        `warp.optim.linear.cg` (see `gpu/vbd/aasolver.py`) applied to the normal equations
        `Fk[:mk]^T Fk[:mk] gammak[:mk] = Fk[:mk]^T fk`, using a custom
        `warp.optim.linear.LinearOperator` whose matvec runs `cupy` matmuls
      - b: `(m,)` warp array scratch buffer holding the CG right-hand-side `Fk[:mk]^T fk`
    """

    def __init__(
        self,
        params: pbat.sim.algorithm.vbd.Params,
        anderson_params: pbat.sim.algorithm.vbd.AndersonParams,
    ):
        super().__init__(params)
        self.m = int(anderson_params.m)
        self.beta = float(anderson_params.beta)
        n_nodes = params.colors.shape[0]
        self.xkm1: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)
        self.fk: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)
        self.fkm1: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)
        # Xk, Fk are stored as CuPy ndarrays (m, N, 3) rather than lists of independently
        # allocated warp arrays, so the Anderson mixing step can treat them as (3N, mk)
        # matrices for `cupy` matmuls without any gather/copy. Each slot's `wp.array` vec3f
        # view shares the exact same device memory (zero-copy), so per-vertex kernels
        # (`gpu/vbd/aasolver.py`) can still write into `Xk[dkl]`/`Fk[dkl]` directly.
        self.Xk_cp: cp.ndarray = cp.zeros((self.m, n_nodes, 3), dtype=cp.float32)
        self.Fk_cp: cp.ndarray = cp.zeros((self.m, n_nodes, 3), dtype=cp.float32)
        self.Xk: list[wp.array[wp.vec3f]] = [
            wp.array(
                ptr=self.Xk_cp[i].data.ptr,
                dtype=wp.vec3f,
                shape=(n_nodes,),
                ndim=1,
                copy=False,
            )
            for i in range(self.m)
        ]
        self.Fk: list[wp.array[wp.vec3f]] = [
            wp.array(
                ptr=self.Fk_cp[i].data.ptr,
                dtype=wp.vec3f,
                shape=(n_nodes,),
                ndim=1,
                copy=False,
            )
            for i in range(self.m)
        ]
        # gammak/b are plain warp arrays (rather than CuPy ndarrays) so they can be passed
        # directly as the `x`/`b` arguments of `warp.optim.linear.cg`.
        self.gammak: wp.array[wp.float32] = wp.zeros((self.m,), dtype=wp.float32)
        self.b: wp.array[wp.float32] = wp.zeros((self.m,), dtype=wp.float32)


def serialize_anderson_cpu_params(
    vbd_params: pbat.sim.algorithm.vbd.Params,
    anderson_params: pbat.sim.algorithm.vbd.AndersonParams,
    grp,
) -> None:
    """Serialize a VBD CPU `Params` object plus an Anderson CPU `AndersonParams` object's
    `m`, `beta`, and `cod_numerical_zero` to an h5py group."""
    serialize_vbd_cpu_params(vbd_params, grp)
    grp.attrs["m"] = int(anderson_params.m)
    grp.attrs["beta"] = float(anderson_params.beta)
    grp.attrs["cod_numerical_zero"] = float(anderson_params.cod_numerical_zero)


def deserialize_anderson_cpu_params(
    vbd_params: pbat.sim.algorithm.vbd.Params,
    anderson_params: pbat.sim.algorithm.vbd.AndersonParams,
    grp,
) -> None:
    """Deserialize a VBD CPU `Params` object plus an Anderson CPU `AndersonParams` object's
    `m`, `beta`, and `cod_numerical_zero` from an h5py group, in place. Silently skips keys
    absent from the group."""
    deserialize_vbd_cpu_params(vbd_params, grp)
    if "m" in grp.attrs:
        anderson_params.m = int(grp.attrs["m"])
    if "beta" in grp.attrs:
        anderson_params.beta = float(grp.attrs["beta"])
    if "cod_numerical_zero" in grp.attrs:
        anderson_params.cod_numerical_zero = float(grp.attrs["cod_numerical_zero"])


class AndersonCpuParams:
    """Container bundling the two CPU-side parameter objects the Anderson-accelerated VBD
    solver needs, so that they can be stored/passed around as a single `params_cpu` entry in
    `ui.py`.

    `m`, `beta`, and `cod_numerical_zero` are exposed as top-level properties delegating to the
    underlying `pbat.sim.algorithm.vbd.AndersonParams`, so they're drawn directly by
    `draw_params`, while `vbd_params` (the underlying `pbat.sim.algorithm.vbd.Params`) is meant
    to be drawn as a nested sub-tree via `draw_params`'s `sub_params` mechanism (see
    `SOLVER_SUB_PARAMS[SolverType.AndersonSolver] = {"vbd_params": None}` in `ui.py`), the same
    way Newton's `line_search` sub-params are handled.
    """

    def __init__(
        self,
        vbd_params: pbat.sim.algorithm.vbd.Params | None = None,
        anderson_params: pbat.sim.algorithm.vbd.AndersonParams | None = None,
    ):
        self.vbd_params = (
            vbd_params if vbd_params is not None else pbat.sim.algorithm.vbd.Params()
        )
        self._anderson_params = (
            anderson_params
            if anderson_params is not None
            else pbat.sim.algorithm.vbd.AndersonParams()
        )

    @property
    def anderson_params(self) -> pbat.sim.algorithm.vbd.AndersonParams:
        """The underlying `pbat.sim.algorithm.vbd.AndersonParams` CPU object."""
        return self._anderson_params

    @property
    def m(self) -> int:
        """Anderson acceleration window size."""
        return self._anderson_params.m

    @m.setter
    def m(self, value: int):
        self._anderson_params.m = int(value)

    @property
    def beta(self) -> float:
        """Anderson acceleration mixing parameter."""
        return self._anderson_params.beta

    @beta.setter
    def beta(self, value: float):
        self._anderson_params.beta = float(value)

    @property
    def cod_numerical_zero(self) -> float:
        """Numerical zero threshold for the COD least-squares solver."""
        return self._anderson_params.cod_numerical_zero

    @cod_numerical_zero.setter
    def cod_numerical_zero(self, value: float):
        self._anderson_params.cod_numerical_zero = float(value)


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
        self.assertTrue(np.all(params.data.GVGadj.numpy() == params_cpu.GVGe))
        # Verify vertex-vertex adjacency
        self.assertTrue(np.all(params.data.GVVp.numpy() == params_cpu.GVVp))
        self.assertTrue(np.all(params.data.GVVadj.numpy() == params_cpu.GVVadj))
        # Verify coloring and partitioning
        self.assertTrue(np.all(params.data.colors.numpy() == params_cpu.colors))
        self.assertTrue(np.all(params.Pptr == params_cpu.Pptr))
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
