"""
Device-side convergence test of the VBD-family solves.

Mirrors ``pbat::sim::algorithm::vbd::CheckConvergence``: stop once ``|g|^2 <= gtol^2``, where ``g`` is
the gradient of the time step's Lagrangian at the current contact multipliers, over free DOFs
(:class:`gpu.gradient.GradientNorm` without the augmented-Lagrangian penalty term, as C++
``ToGradient(..., bForAugmentedLagrangian = false)``). Without contact it is the objective's gradient,
so the test means the same inside a subproblem as between subproblems.

The test runs between outer iterations, as in C++, and also every ``check_interval`` sweeps inside a
subproblem. Its outcome stays on the device in flags that conditional graph nodes
(``wp.capture_if``/``wp.capture_while``) read, so a captured step stops early too. See
``vbd.solver.solve_outer_loop`` for how the flags drive the loops.
"""

import warp as wp

from ...common.fields import DocField, SerializableMixin
from ..contact.dynamics import MeshDynamics as ContactDynamics
from ..elasticity.fem import FemElastoDynamics
from ..gradient import GradientNorm


class ConvergenceParams(SerializableMixin):
    """GPU-only settings of the VBD-family convergence test."""

    check_interval = DocField(
        20, "VBD sweeps between convergence tests inside a subproblem (each test costs about a quarter sweep)"
    )


@wp.func
def _is_converged(gnorm2: wp.array[wp.float32], gtol2: wp.float32) -> bool:
    return gnorm2[0] <= gtol2


@wp.kernel
def _start_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    n_max_iters: wp.int32,
    k: wp.array[wp.int32],
    keep: wp.array[wp.int32],
):
    k[0] = 0
    keep[0] = wp.where(n_max_iters > 0 and not _is_converged(gnorm2, gtol2), 1, 0)


@wp.kernel
def _begin_outer_kernel(keep: wp.array[wp.int32], active: wp.array[wp.int32]):
    active[0] = keep[0]


@wp.kernel
def _begin_inner_kernel(
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    inner_keep: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    blocks[0] = 0
    inner_keep[0] = wp.where(n_blocks > 0, 1, 0)
    subproblem_keep[0] = 1


@wp.kernel
def _after_block_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    inner_keep: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    b = blocks[0] + 1
    blocks[0] = b
    converged = _is_converged(gnorm2, gtol2)
    subproblem_keep[0] = wp.where(converged, 0, 1)
    inner_keep[0] = wp.where(not converged and b < n_blocks, 1, 0)


@wp.kernel
def _end_outer_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    n_max_iters: wp.int32,
    k: wp.array[wp.int32],
    keep: wp.array[wp.int32],
):
    kk = k[0] + 1
    k[0] = kk
    keep[0] = wp.where(kk < n_max_iters and not _is_converged(gnorm2, gtol2), 1, 0)


class ConvergenceCheck:
    """Outer-iteration counter and loop flags of a solve, on the device.

    Flags (each a (1,) int32 array, read by conditional graph nodes):

    - ``keep``: the step goes on (not converged and outer iterations left).
    - ``active``: the current outer iteration runs (``keep`` when it started).
    - ``inner_keep``: the current subproblem's sweep blocks go on.
    - ``subproblem_keep``: the current subproblem's remaining sweeps run (not converged inside it).

    ``k`` counts the outer iterations done. :meth:`converged` reads the last test on the host.
    """

    def __init__(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        params: ConvergenceParams | None = None,
    ):
        self.params = params or ConvergenceParams()
        self.gradient_norm = GradientNorm(fem, contact, augmented_lagrangian=False)
        self.k = wp.zeros(1, dtype=wp.int32)
        self.keep = wp.zeros(1, dtype=wp.int32)
        self.active = wp.zeros(1, dtype=wp.int32)
        self.blocks = wp.zeros(1, dtype=wp.int32)
        self.inner_keep = wp.zeros(1, dtype=wp.int32)
        self.subproblem_keep = wp.zeros(1, dtype=wp.int32)

    def _test(self) -> None:
        self.gradient_norm.compute(wp.get_stream())

    def start(self, gtol: float, n_max_iters: int) -> None:
        """Test the initial positions: ``k = 0``, ``keep`` = not converged."""
        self._test()
        wp.launch(
            _start_kernel,
            dim=1,
            inputs=[self.gradient_norm.gnorm2, float(gtol) ** 2, int(n_max_iters), self.k, self.keep],
        )

    def begin_outer(self) -> None:
        """Latch ``keep`` into ``active`` for the outer iteration that starts."""
        wp.launch(_begin_outer_kernel, dim=1, inputs=[self.keep, self.active])

    def begin_inner(self, n_blocks: int) -> None:
        """Start a subproblem's loop over ``n_blocks`` sweep blocks."""
        wp.launch(
            _begin_inner_kernel,
            dim=1,
            inputs=[int(n_blocks), self.blocks, self.inner_keep, self.subproblem_keep],
        )

    def after_block(self, gtol: float, n_blocks: int) -> None:
        """Test after a sweep block: a converged subproblem stops, and so do its remaining sweeps."""
        self._test()
        wp.launch(
            _after_block_kernel,
            dim=1,
            inputs=[
                self.gradient_norm.gnorm2,
                float(gtol) ** 2,
                int(n_blocks),
                self.blocks,
                self.inner_keep,
                self.subproblem_keep,
            ],
        )

    def end_outer(self, gtol: float, n_max_iters: int) -> None:
        """Test after an outer iteration (C++'s check before the next one): ``k += 1``, update ``keep``."""
        self._test()
        wp.launch(
            _end_outer_kernel,
            dim=1,
            inputs=[self.gradient_norm.gnorm2, float(gtol) ** 2, int(n_max_iters), self.k, self.keep],
        )

    def converged(self, gtol: float) -> bool:
        """Host-side ``|g|^2 <= gtol^2`` from the last test. Synchronizes, so not capturable."""
        return float(self.gradient_norm.gnorm2.numpy()[0]) <= float(gtol) * float(gtol)
