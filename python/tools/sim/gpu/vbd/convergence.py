"""
Device-side convergence test of the VBD-family outer (augmented-Lagrangian) loop.

Mirrors ``pbat::sim::algorithm::vbd::CheckConvergence``: stop once ``|g|^2 <= gtol^2``, where ``g`` is
the gradient of the time step's Lagrangian at the current contact multipliers, over free DOFs
(:class:`gpu.gradient.GradientNorm` without the augmented-Lagrangian penalty term, as C++
``ToGradient(..., bForAugmentedLagrangian = false)``). Without contact it is the objective's gradient,
so the test means the same inside a subproblem as between subproblems. The decision is kept on the device in ``keep_iterating``, so a captured CUDA graph can loop
on it with ``wp.capture_while`` and stop early.
"""

import warp as wp

from ..contact.dynamics import MeshDynamics as ContactDynamics
from ..elasticity.fem import FemElastoDynamics
from ..gradient import GradientNorm


@wp.kernel
def _update_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    k: wp.array[wp.int32],
    increment: wp.int32,
    n_max_iters: wp.int32,
    keep_iterating: wp.array[wp.int32],
):
    kk = k[0] + increment
    k[0] = kk  # pyright: ignore[reportIndexIssue]
    keep = wp.int32(0)
    if kk < n_max_iters and not (gnorm2[0] <= gtol2):
        keep = wp.int32(1)
    keep_iterating[0] = keep  # pyright: ignore[reportIndexIssue]


class ConvergenceCheck:
    """Outer-iteration counter ``k`` and convergence flag ``keep_iterating`` of a solve, on the device.

    Usage (inside a solve, captured or not)::

        check.start(gtol, n_max_iters)           # k = 0, test at the current x
        while keep_iterating:                    # e.g. wp.capture_while(check.keep_iterating, ...)
            ...one outer iteration...
            check.advance(gtol, n_max_iters)     # k += 1, test again
    """

    def __init__(self, fem: FemElastoDynamics, contact: ContactDynamics):
        self.gradient_norm = GradientNorm(fem, contact, augmented_lagrangian=False)
        self.k = wp.zeros(1, dtype=wp.int32)
        self.keep_iterating = wp.zeros(1, dtype=wp.int32)

    def start(self, gtol: float, n_max_iters: int) -> None:
        """Reset ``k`` to 0 and test convergence at the current positions."""
        self.k.zero_()
        self._update(gtol, n_max_iters, 0)

    def advance(self, gtol: float, n_max_iters: int) -> None:
        """Count one more outer iteration and test convergence at the current positions."""
        self._update(gtol, n_max_iters, 1)

    def _update(self, gtol: float, n_max_iters: int, increment: int) -> None:
        self.gradient_norm.compute(wp.get_stream())
        wp.launch(
            _update_kernel,
            dim=1,
            inputs=[
                self.gradient_norm.gnorm2,
                float(gtol) * float(gtol),
                self.k,
                increment,
                int(n_max_iters),
                self.keep_iterating,
            ],
        )

    def converged(self, gtol: float) -> bool:
        """Host-side ``|g|^2 <= gtol^2`` from the last test. Synchronizes, so not capturable."""
        return float(self.gradient_norm.gnorm2.numpy()[0]) <= float(gtol) * float(gtol)
