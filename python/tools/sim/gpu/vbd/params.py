import warp as wp
import numpy as np
import cupy as cp
import enum
from pbatoolkit import pbat

from ...common.fields import DocField, SerializableMixin

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


class AndersonLinearSolver(enum.Enum):
    """GPU Anderson least-squares linear solve method.

    Both methods operate on the Gram matrix of ``[F f]``, formed in a single pass over ``F``:
      - NormalEquations: Cholesky solve of ``F^T F gamma = F^T f``, regularized only when a
        pivot falls below the scale-invariant threshold ``cod_numerical_zero``.
      - ConjugateGradient: Jacobi-preconditioned CG on the same system, without regularization.
    """

    NormalEquations = 0
    ConjugateGradient = 1


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
    hess_zero: wp.float32  # pseudo-singularity tolerance relative to H's scale (see kernels.integrate_positions)
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
    betaup: wp.float32  # upper limit that beta grows toward (GPU only; C++ hard-codes 1)
    lipschitz_normalization: wp.int32  # normalize rho by the vertex Lipschitz estimate (GPU only; C++ always does)
    surface_stencil_surface_neighbours_only: wp.int32  # surface vertices skip interior neighbours

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

    ``warm_start_beta`` says when ``initialize_solve`` and ``prepare_subproblem`` reset the stencil
    gradient augmentation scales ``betaG`` to ``betaG0``, as in C++ ``InitializeSolve`` and
    ``PrepareSubproblem``: ``Never`` warm starts them (reset every subproblem), ``Subproblem`` carries
    them across a time step's subproblems (reset every time step) and ``TimeStep`` never resets them.
    Only AAAVBD reads ``betaG``, so it is the only solver that passes the CPU params'
    ``warm_start_beta``. The default, ``TimeStep``, adds no resets.

    ``betaup`` and ``lipschitz_normalization`` are GPU-only AAAVBD parameters for ablation studies
    (see `AaaVbdCpuParams`). Their defaults reproduce C++ ``vbd::Iterate``.
    """

    _data: ParamsData  # pyright: ignore[reportGeneralTypeIssues]
    _Pptr: np.ndarray  # (# colors + 1,) partition pointers on CPU, not part of GPU struct

    def __init__(
        self,
        params: pbat.sim.algorithm.vbd.Params,
        warm_start_beta: pbat.sim.algorithm.vbd.StencilGradientBetaWarmStartMask = (
            pbat.sim.algorithm.vbd.StencilGradientBetaWarmStartMask.TimeStep
        ),
        betaup: float = 1.0,
        lipschitz_normalization: bool = True,
    ):
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
        if params.vlinsolve == pbat.sim.algorithm.vbd.VertexIntegrationLinearSolver.LLT:
            raise ValueError(
                "The LLT vertex solve isn't implemented on the GPU; use Inverse (default), QR or EVD"
            )
        self._data.vls_solver = int(_VLS_SOLVER_MAP[params.vlinsolve])
        # Stencil gradient acceleration
        self._data.betaG0 = (float(params.betaG0), float(params.betaG0))
        self._data.rhohat = (float(params.rhohat), float(params.rhohatS))
        self._data.gammadown = (float(params.gammadown), float(params.gammadownS))
        self._data.gammaup = (float(params.gammaup), float(params.gammaupS))
        self._data.betaup = float(betaup)
        self._data.lipschitz_normalization = int(bool(lipschitz_normalization))
        self._data.surface_stencil_surface_neighbours_only = int(
            params.surface_stencil_surface_neighbours_only
        )
        self.warm_start_beta = warm_start_beta  # host-side: decides which resets are recorded
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
        # Device-side convergence test (gpu.vbd.convergence.ConvergenceCheck), built for a given
        # FEM and contact problem by the owner of both. Without it, solves run n_max_iters outer
        # iterations.
        self.convergence = None

    @property
    def data(self) -> ParamsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

    @property
    def Pptr(self) -> np.ndarray:
        return self._Pptr


# Parameters that only AAAVBD (stencil gradient acceleration) reads: fields of
# pbat.sim.algorithm.vbd.Params, plus AaaVbdCpuParams' GPU-only ones. The GPU VBD, Chebyshev and
# Anderson solvers ignore them, so their parameter tables and GUI panels leave them out.
STENCIL_GRADIENT_ACCELERATION_KEYS = frozenset(
    {
        "betaG0",
        "rhohat",
        "rhohatS",
        "gammaup",
        "gammaupS",
        "gammadown",
        "gammadownS",
        "warm_start_beta",
        "surface_stencil_surface_neighbours_only",
        # GPU-only (AaaVbdCpuParams)
        "betaup",
        "lipschitz_normalization",
    }
)


def serialize_vbd_cpu_params(
    params: pbat.sim.algorithm.vbd.Params,
    grp,
    stencil_gradient_acceleration: bool = False,
) -> None:
    """Serialize the scalar/enum fields of a VBD CPU Params object to an h5py group.

    Only settable float, int, bool, and enum properties are written (same set that
    ``draw_params`` displays in the UI). The ``STENCIL_GRADIENT_ACCELERATION_KEYS`` are written
    only if ``stencil_gradient_acceleration`` is set (AAAVBD).
    """
    import inspect
    import enum

    for name, value in inspect.getmembers(params):
        if name.startswith("_"):
            continue
        if not stencil_gradient_acceleration and name in STENCIL_GRADIENT_ACCELERATION_KEYS:
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


def deserialize_vbd_cpu_params(
    params: pbat.sim.algorithm.vbd.Params,
    grp,
    stencil_gradient_acceleration: bool = False,
) -> None:
    """Deserialize scalar/enum fields from an h5py group into a VBD CPU Params object.

    Silently skips keys absent in the group or attributes that cannot be set. Unless
    ``stencil_gradient_acceleration`` is set (AAAVBD), the ``STENCIL_GRADIENT_ACCELERATION_KEYS``
    are ignored too, so older files that store them for every VBD-family solver still load.
    """
    import inspect
    import enum

    for name, value in inspect.getmembers(params):
        if name.startswith("_"):
            continue
        if not stencil_gradient_acceleration and name in STENCIL_GRADIENT_ACCELERATION_KEYS:
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


class AaaVbdCpuParams(SerializableMixin):
    """CPU-side AAAVBD parameters: a `pbat.sim.algorithm.vbd.Params` object plus GPU-only stencil
    gradient acceleration parameters for ablation studies, stored as a single `params_cpu` entry.

    The C++ object rejects new attributes, so this wrapper holds it as `vbd_params` (drawn as a nested
    sub-tree in the GUI, see `SOLVER_SUB_PARAMS` in `ui.py`) and forwards every other attribute to it,
    so `params_cpu[SolverType.AAAVBD].gammadown` and the like keep working. The GPU-only parameters
    are `DocField`s, serialized next to the VBD fields in the `Solver/AAAVBD` group
    (`serialize_aaavbd_cpu_params`). C++ `vbd::Iterate` has neither: it behaves as their defaults.
    """

    betaup = DocField(
        1.0,
        "Upper limit of the stencil gradient augmentation scale betaG: on a step with rho > rhohat, "
        "betaG += (betaup - betaG) * gammaup. C++ hard-codes 1. Values above 1 let betaG exceed 1 "
        "(over-relaxation). Both interior and surface vertices use it. GPU only.",
    )
    lipschitz_normalization = DocField(
        True,
        "Normalize the step ratio rho = |g_k - g_k-1| / |x_k - x_k-1| by the vertex Lipschitz estimate "
        "L = |H_i|_F + |g_k| / |x_k - x_k-1|, as C++ does. Off, rho is not normalized, so rhohat and "
        "rhohatS have stiffness units and their values aren't comparable with the normalized ones. GPU only.",
    )

    def __init__(self, vbd_params: pbat.sim.algorithm.vbd.Params | None = None):
        object.__setattr__(
            self,
            "vbd_params",
            vbd_params if vbd_params is not None else pbat.sim.algorithm.vbd.Params(),
        )

    def __getattr__(self, name):
        # Only reached for attributes this wrapper lacks
        if name == "vbd_params":
            raise AttributeError(name)
        return getattr(self.vbd_params, name)

    def __setattr__(self, name, value):
        if name.startswith("_") or hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.vbd_params, name, value)


def serialize_aaavbd_cpu_params(params: AaaVbdCpuParams, grp) -> None:
    """Serialize an `AaaVbdCpuParams` container to an h5py group: the VBD CPU `Params` with its
    stencil gradient acceleration fields, then the GPU-only `DocField`s."""
    serialize_vbd_cpu_params(params.vbd_params, grp, stencil_gradient_acceleration=True)
    params.serialize(grp)


def deserialize_aaavbd_cpu_params(params: AaaVbdCpuParams, grp) -> None:
    """Deserialize an `AaaVbdCpuParams` container from an h5py group, in place (see
    `serialize_aaavbd_cpu_params`). Silently skips keys absent from the group."""
    deserialize_vbd_cpu_params(params.vbd_params, grp, stencil_gradient_acceleration=True)
    params.deserialize(grp)


class ChebyshevParams(Params):
    """VBD solver parameters augmented with Chebyshev semi-iterative acceleration state, for
    GPU execution.

    Wraps a `pbat.sim.algorithm.vbd.Params` CPU object, inheriting all VBD parameters/buffers
    (see `Params`), and additionally stores the Chebyshev-specific GPU-resident state mirroring
    `pbat::sim::algorithm::vbd::ChebyshevParams` (`source/pbat/sim/algorithm/vbd/Chebyshev.h`):
      - rho: spectral radius estimate, `0 < rho < 1`, read once from the CPU
        `pbat.sim.algorithm.vbd.ChebyshevParams` object at construction time
      - omega: (1,) device array holding the current relaxation weight, recomputed every VBD sweep
      - xkm1, xkm2: previous VBD-sweep iterates used by the momentum recurrence
    """

    def __init__(
        self,
        params: pbat.sim.algorithm.vbd.Params,
        cheb_params: pbat.sim.algorithm.vbd.ChebyshevParams,
    ):
        super().__init__(params)
        self.rho = cheb_params.rho
        self.omega: wp.array[wp.float32] = wp.ones((1,), dtype=wp.float32)
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
      - FfT: `(m+1, 3N)` C-contiguous CuPy ndarray. Rows `0..m-1` (`FkT`) are the window slots of
        the residual-difference history, each storing one conceptual column of the `(3N, m)`
        Anderson matrix `F` contiguously, and row `m` stores `fk`. The Gram matrix of this buffer
        therefore holds both `F^T F` and `F^T f`.
      - XkT: `(m, 3N)` C-contiguous CuPy ndarray of step differences, laid out like `FkT`
      - Xk, Fk, fk: zero-copy `wp.vec3f` views over `XkT`, `FkT` and the last row of `FfT`
      - FfT_flat: zero-copy `(m+1, 3N)` `wp.float32` view over `FfT` for tile loads
      - G: `(m+1, m+1)` Gram matrix of `[F f]`
      - linear_solver: Cholesky or Jacobi-preconditioned CG solve of the normal equations
      - vbd_iters_per_anderson_iter: number of VBD sweeps composing the fixed-point map that
        one Anderson iteration applies (GPU only, not part of the CPU `AndersonParams`)
      - gammak: `(m,)` Warp array of subspace mixing coefficients
    """

    def __init__(
        self,
        params: pbat.sim.algorithm.vbd.Params,
        anderson_params: pbat.sim.algorithm.vbd.AndersonParams,
        linear_solver: AndersonLinearSolver = AndersonLinearSolver.NormalEquations,
        vbd_iters_per_anderson_iter: int = 1,
    ):
        super().__init__(params)
        self.m = int(anderson_params.m)
        self.beta = float(anderson_params.beta)
        self.cod_numerical_zero = float(anderson_params.cod_numerical_zero)
        self.linear_solver = linear_solver
        self.vbd_iters_per_anderson_iter = max(1, int(vbd_iters_per_anderson_iter))
        n_nodes = params.colors.shape[0]
        self.xkm1: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)
        self.fkm1: wp.array[wp.vec3f] = wp.zeros((n_nodes,), dtype=wp.vec3f)
        self.XkT: cp.ndarray = cp.zeros((self.m, n_nodes * 3), dtype=cp.float32)
        self.FfT: cp.ndarray = cp.zeros((self.m + 1, n_nodes * 3), dtype=cp.float32)
        self.FkT: cp.ndarray = self.FfT[: self.m]
        self.Xk: wp.array[wp.vec3f] = wp.array(
            self.XkT,
            dtype=wp.vec3f,
            shape=(self.m, n_nodes),
            copy=False,
        )
        self.Fk: wp.array[wp.vec3f] = wp.array(
            self.FkT,
            dtype=wp.vec3f,
            shape=(self.m, n_nodes),
            copy=False,
        )
        self.fk: wp.array[wp.vec3f] = wp.array(
            ptr=self.FfT[self.m].data.ptr,
            dtype=wp.vec3f,
            shape=(n_nodes,),
            device=self.Fk.device,
            copy=False,
        )
        self.FfT_flat: wp.array2d[wp.float32] = wp.array(  # pyright: ignore[reportAttributeAccessIssue]
            self.FfT, dtype=wp.float32, copy=False
        )
        self.G: wp.array2d[wp.float32] = wp.zeros(  # pyright: ignore[reportAttributeAccessIssue]
            (self.m + 1, self.m + 1), dtype=wp.float32
        )
        self.gammak: wp.array[wp.float32] = wp.zeros((self.m,), dtype=wp.float32)
        # Kernels are specialized on the window size m (static tile shapes), and must exist before
        # any CUDA graph capture starts, so that capture's module force-load includes them.
        from .aasolver import anderson_kernels

        self.kernels = anderson_kernels(self.m)


def serialize_anderson_cpu_params(params: "AndersonCpuParams", grp) -> None:
    """Serialize an `AndersonCpuParams` container to an h5py group: the VBD CPU `Params`, the
    Anderson CPU `AndersonParams`' `m`, `beta` and `cod_numerical_zero`, and the GPU-only
    `linear_solver` and `vbd_iters_per_anderson_iter`."""
    serialize_vbd_cpu_params(params.vbd_params, grp)
    grp.attrs["m"] = int(params.m)
    grp.attrs["beta"] = float(params.beta)
    grp.attrs["cod_numerical_zero"] = float(params.cod_numerical_zero)
    grp.attrs["linear_solver"] = params.linear_solver.value
    grp.attrs["vbd_iters_per_anderson_iter"] = int(params.vbd_iters_per_anderson_iter)


def deserialize_anderson_cpu_params(params: "AndersonCpuParams", grp) -> None:
    """Deserialize an `AndersonCpuParams` container from an h5py group, in place (see
    `serialize_anderson_cpu_params`). Silently skips keys absent from the group."""
    deserialize_vbd_cpu_params(params.vbd_params, grp)
    if "m" in grp.attrs:
        params.m = int(grp.attrs["m"])
    if "beta" in grp.attrs:
        params.beta = float(grp.attrs["beta"])
    if "cod_numerical_zero" in grp.attrs:
        params.cod_numerical_zero = float(grp.attrs["cod_numerical_zero"])
    if "linear_solver" in grp.attrs:
        params.linear_solver = AndersonLinearSolver(int(grp.attrs["linear_solver"]))
    if "vbd_iters_per_anderson_iter" in grp.attrs:
        params.vbd_iters_per_anderson_iter = int(grp.attrs["vbd_iters_per_anderson_iter"])


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
        self._linear_solver = AndersonLinearSolver.NormalEquations
        self._vbd_iters_per_anderson_iter = 1

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
        """Relative rank threshold: the CPU COD treats pivots below this fraction of the largest
        pivot as zero. The GPU Cholesky solve uses it the same way, as a relative singular-value
        cutoff on F, and regularizes only when a pivot falls below it."""
        return self._anderson_params.cod_numerical_zero

    @cod_numerical_zero.setter
    def cod_numerical_zero(self, value: float):
        self._anderson_params.cod_numerical_zero = float(value)

    @property
    def linear_solver(self) -> AndersonLinearSolver:
        """GPU least-squares solve method: NormalEquations (Cholesky, conditionally regularized)
        or ConjugateGradient (Jacobi-preconditioned CG, unregularized)."""
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, value: AndersonLinearSolver):
        self._linear_solver = AndersonLinearSolver(value)

    @property
    def vbd_iters_per_anderson_iter(self) -> int:
        """Number of VBD sweeps that make up one Anderson iteration (>= 1). Anderson accelerates
        the fixed-point map made of this many consecutive sweeps. n_subproblem_max_iters still
        counts sweeps: a subproblem runs n_subproblem_max_iters // vbd_iters_per_anderson_iter
        Anderson iterations, then the leftover sweeps as plain VBD sweeps."""
        return self._vbd_iters_per_anderson_iter

    @vbd_iters_per_anderson_iter.setter
    def vbd_iters_per_anderson_iter(self, value: int):
        self._vbd_iters_per_anderson_iter = max(1, int(value))


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
        self.assertEqual(
            params.data.surface_stencil_surface_neighbours_only,
            int(params_cpu.surface_stencil_surface_neighbours_only),
        )
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
        # GPU-only AAAVBD parameters: C++ behaviour by default
        self.assertEqual(params.data.betaup, 1.0)
        self.assertEqual(params.data.lipschitz_normalization, 1)
        params = Params(params_cpu, betaup=1.05, lipschitz_normalization=False)
        self.assertAlmostEqual(params.data.betaup, 1.05, places=6)
        self.assertEqual(params.data.lipschitz_normalization, 0)

    def test_stencil_gradient_acceleration_keys_are_aaavbd_only(self):
        import h5py

        params_cpu = pbat.sim.algorithm.vbd.Params()
        with h5py.File("params.h5", "w", driver="core", backing_store=False) as f:
            serialize_vbd_cpu_params(params_cpu, f.create_group("VBD"))
            serialize_aaavbd_cpu_params(AaaVbdCpuParams(params_cpu), f.create_group("AAAVBD"))
            self.assertFalse(STENCIL_GRADIENT_ACCELERATION_KEYS & set(f["VBD"].attrs))
            self.assertLessEqual(STENCIL_GRADIENT_ACCELERATION_KEYS, set(f["AAAVBD"].attrs))
            self.assertIn("gtol", f["VBD"].attrs)
            # Older files store the stencil keys for every VBD-family solver: only AAAVBD reads them.
            old = f.create_group("old")
            old.attrs["betaG0"] = 0.25
            old.attrs["gtol"] = 0.5
            vbd = pbat.sim.algorithm.vbd.Params()
            deserialize_vbd_cpu_params(vbd, old)
            self.assertAlmostEqual(vbd.betaG0, params_cpu.betaG0)
            self.assertAlmostEqual(vbd.gtol, 0.5)
            aaavbd = pbat.sim.algorithm.vbd.Params()
            deserialize_vbd_cpu_params(aaavbd, old, stencil_gradient_acceleration=True)
            self.assertAlmostEqual(aaavbd.betaG0, 0.25)

    def test_aaavbd_cpu_params(self):
        import h5py

        params = AaaVbdCpuParams()
        # GPU-only parameters, with C++ behaviour by default
        self.assertEqual(params.betaup, 1.0)
        self.assertIs(params.lipschitz_normalization, True)
        # Every other attribute is the wrapped C++ object's
        params.gammadown = 0.9
        self.assertAlmostEqual(params.vbd_params.gammadown, 0.9, places=6)
        self.assertAlmostEqual(params.gammadown, 0.9, places=6)
        with self.assertRaises(AttributeError):
            params.not_a_parameter = 1
        # Round trip through HDF5, next to the VBD fields
        params.betaup = 1.05
        params.lipschitz_normalization = False
        with h5py.File("params.h5", "w", driver="core", backing_store=False) as f:
            serialize_aaavbd_cpu_params(params, f.create_group("AAAVBD"))
            self.assertEqual(f["AAAVBD"].attrs["lipschitz_normalization"], 0)
            self.assertIn("gammadown", f["AAAVBD"].attrs)
            loaded = AaaVbdCpuParams()
            deserialize_aaavbd_cpu_params(loaded, f["AAAVBD"])
            self.assertAlmostEqual(loaded.betaup, 1.05)
            self.assertIs(loaded.lipschitz_normalization, False)
            self.assertAlmostEqual(loaded.gammadown, 0.9, places=6)
            # Files without the GPU-only keys keep their defaults
            old = f.create_group("old")
            old.attrs["gammadown"] = 0.95
            loaded = AaaVbdCpuParams()
            deserialize_aaavbd_cpu_params(loaded, old)
            self.assertEqual(loaded.betaup, 1.0)
            self.assertIs(loaded.lipschitz_normalization, True)
            self.assertAlmostEqual(loaded.gammadown, 0.95, places=6)


if __name__ == "__main__":
    wp.init()
    unittest.main()
