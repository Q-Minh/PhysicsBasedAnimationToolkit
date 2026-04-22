import cupy as cp
import numpy as np

# BDF coefficients (alpha, beta) for steps 1..6
_BDF_COEFFS = {
    1: (np.array([-1.0], dtype=np.float32), 1.0),
    2: (np.array([1.0 / 3, -4.0 / 3], dtype=np.float32), 2.0 / 3),
    3: (np.array([-2.0 / 11, 9.0 / 11, -18.0 / 11], dtype=np.float32), 6.0 / 11),
    4: (
        np.array([3.0 / 25, -16.0 / 25, 36.0 / 25, -48.0 / 25], dtype=np.float32),
        12.0 / 25,
    ),
    5: (
        np.array(
            [-12.0 / 137, 75.0 / 137, -200.0 / 137, 300.0 / 137, -300.0 / 137],
            dtype=np.float32,
        ),
        60.0 / 137,
    ),
    6: (
        np.array(
            [
                10.0 / 147,
                -72.0 / 147,
                225.0 / 147,
                -400.0 / 147,
                450.0 / 147,
                -360.0 / 147,
            ],
            dtype=np.float32,
        ),
        60.0 / 147,
    ),
}


class Bdf:
    """BDF (Backward Differentiation Formula) time integration scheme.

    Mirrors `pbat::sim::integration::Bdf` from `source/pbat/sim/integration/Bdf.h`.

    Storage layout:
        xt:     (order, step, N) circular buffer of past states per derivative order.
        xtilde: (order, N) aggregated inertia terms.

    Usage:
        bdf = Bdf(step=1, order=2, dt=0.01)
        bdf.set_initial_conditions(x0, v0)  # each (N,)
        for _ in range(num_steps):
            bdf.construct_equations()
            xn, vn = solve(bdf)
            bdf.step(xn, vn)

    Interop CuPy/warp:
        @wp.kernel
        def solve(
            a0: wp.array[wp.vec3f], h: float, x0: wp.array[wp.vec3f], v0: wp.array[wp.vec3f]
        ):
            i = wp.tid()
            v0[i] += h * a0[i]  # pyright: ignore[reportIndexIssue]
            x0[i] += h * v0[i]  # pyright: ignore[reportIndexIssue]

        bdf = Bdf(step=1, order=2, dt=1e-2)
        n = 10
        x0, v0 = wp.zeros((n,), dtype=wp.vec3f), wp.zeros((n,), dtype=wp.vec3f)
        a0 = cp.tile(cp.array([0.0, 0.0, -9.81], dtype=cp.float32), (n, 1))
        bdf.set_initial_conditions(cp.asarray(x0).ravel(), cp.asarray(v0).ravel())
        for _ in range(2):
            bdf.construct_equations()
            # option 1:
            wp.launch(solve, n, [a0, bdf.beta_tilde], [x0, v0])
            # option 2:
            v0 = cp.asarray(v0) + bdf.beta_tilde * a0
            x0 = cp.asarray(x0) + bdf.beta_tilde * cp.asarray(v0)
            bdf.step(cp.asarray(x0).ravel(), cp.asarray(v0).ravel())
    """

    def __init__(self, step: int = 1, order: int = 2, dt: float = 0.01):
        assert 1 <= step <= 6
        assert order > 0
        self._step = step
        self._order = order
        self._ti = 0
        self._h = dt
        self._alpha, self._beta = _BDF_COEFFS[step]
        self.xt: cp.ndarray = None  # (order, step, N)
        self.xtilde: cp.ndarray = None  # (order, N)

    @property
    def order(self) -> int:
        return self._order

    @property
    def num_steps(self) -> int:
        return self._step

    @property
    def ti(self) -> int:
        return self._ti

    @property
    def h(self) -> float:
        return self._h

    @h.setter
    def h(self, dt: float):
        assert dt > 0
        self._h = dt

    @property
    def alpha(self) -> cp.ndarray:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def beta_tilde(self) -> float:
        return self._beta * self._h

    def _state_index(self, k: int) -> int:
        """Map logical index k to circular buffer column: (ti + k) % step."""
        return (self._ti + k) % self._step

    def state(self, k: int, o: int = 0) -> cp.ndarray:
        """Return xt[o, (ti+k)%step] which is x^{(o)}_{ti - step + k}, shape (N,)."""
        return self.xt[o, self._state_index(k)]

    def current_state(self, o: int = 0) -> cp.ndarray:
        """Current state x^{(o)}_{ti}, shape (N,)."""
        return self.state(self._step - 1, o)

    def inertia(self, o: int = 0) -> cp.ndarray:
        """Inertia xtilde[o], shape (N,)."""
        return self.xtilde[o]

    def set_initial_conditions(self, *x0: cp.ndarray):
        """Set ICs. Pass `order` arrays each of shape (N,)."""
        assert len(x0) == self._order
        self._ti = 0
        n = x0[0].shape[0]
        self.xt = cp.zeros((self._order, self._step, n), dtype=cp.float32)
        self.xtilde = cp.zeros((self._order, n), dtype=cp.float32)
        for o in range(self._order):
            self.xt[o, :] = x0[o][cp.newaxis, :]

    def construct_equations(self):
        """Compute xtilde[o] = sum_k alpha[k] * State(k, o) for all o."""
        self.xtilde[:] = 0
        for o in range(self._order):
            # We could implement this as a matrix-multiplication by permuting
            # alpha similarly to our circular buffer on self.xt
            for k in range(self._step):
                self.xtilde[o] += self._alpha[k] * self.state(k, o)

    def step(self, *x_new: cp.ndarray):
        """Advance by one time step. Pass `order` arrays each of shape (N,)."""
        assert len(x_new) == self._order
        self._ti += 1
        kt = self._state_index(self._step - 1)
        for o in range(self._order):
            self.xt[o, kt] = x_new[o]


import unittest
import warp as wp


@wp.kernel
def _explicit_integration_kernel(
    a0: wp.array[wp.vec3f], h: float, x0: wp.array[wp.vec3f], v0: wp.array[wp.vec3f]
):
    i = wp.tid()
    v0[i] += h * a0[i]  # pyright: ignore[reportIndexIssue]
    x0[i] += h * v0[i]  # pyright: ignore[reportIndexIssue]


class TestBdf(unittest.TestCase):
    def test_bdf_step(self):
        bdf = Bdf(step=1, order=2, dt=1e-2)
        n = 10
        x0, v0 = wp.zeros((n,), dtype=wp.vec3f), wp.zeros((n,), dtype=wp.vec3f)
        a0 = cp.tile(cp.array([0.0, 0.0, -9.81], dtype=cp.float32), (n, 1))
        bdf.set_initial_conditions(cp.asarray(x0).ravel(), cp.asarray(v0).ravel())
        for _ in range(2):
            bdf.construct_equations()
            # v0 = cp.asarray(v0) + bdf.beta_tilde * a0
            # x0 = cp.asarray(x0) + bdf.beta_tilde * cp.asarray(v0)
            wp.launch(_explicit_integration_kernel, dim=n, inputs=[a0, bdf.beta_tilde, x0, v0])
            bdf.step(cp.asarray(x0).ravel(), cp.asarray(v0).ravel())


if __name__ == "__main__":
    wp.init()
    unittest.main()
