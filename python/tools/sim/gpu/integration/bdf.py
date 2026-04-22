import warp as wp
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


@wp.struct
class Bdf:
    """BDF (Backward Differentiation Formula) time integration scheme for a system of ODEs for an initial value problem (IVP)
    See `source/pbat/sim/integration/Bdf.h`
    """

    xt: wp.array3d[
        wp.vec3f
    ]  # (N,step,order) matrix of `N`-dimensional states and their time derivatives
    # s.t. \f$ xt[o,k] = x^(k)(t - k*dt) \f$ for \f$ k = 0, ..., step-1 \f$ and
    # \f$ o = 0, ..., \text{order}-1 \f$
    xbar: wp.array2d[
        wp.vec3f
    ]  # (N,order) matrix of `N`-dimensional aggregated past states and time
    # derivatives s.t. xtilde[o] = \f$ \frac{1}{\alpha_s} \sum_{k=t_i-s}^{s-1}
    # \alpha_k x_k \f$ for \f$ o = 0, ..., \text{order}-1 \f$
    t: wp.int32  # current time step index
    h: wp.float32  # time step size
    order: wp.int32  # ODE order (1 for quasistatics, 2 for elastodynamics)
    step: wp.int32  # BDF step (1..6)
    alpha: wp.array[wp.float32]  # (step,) BDF interpolation coefficients
    beta: wp.float32  # BDF forcing term coefficient
