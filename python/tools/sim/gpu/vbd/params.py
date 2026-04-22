import warp as wp


@wp.struct
class Params:
    """GPU params for Anderson-accelerated VBD solver (mirrors full CPU vbd::Params)."""

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

    # --- Stencil gradient acceleration ---
    betaG0: (
        wp.vec2f
    )  # initial stencil gradient augmentation scales (0: interior, 1: surface)
    rhohat: wp.vec2f  # Lipschitz-normalized step thresholds (0: interior, 1: surface)
    gammadown: wp.vec2f  # beta reduction factors (0: interior, 1: surface)
    gammaup: wp.vec2f  # beta increase factors (0: interior, 1: surface)
