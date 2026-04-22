import warp as wp
import warp.sparse
import warp.optim
import numpy as np


@wp.struct
class Params:
    """From `source/pbat/sim/newton/Core.h`"""

    # --- Iteration control ---
    n_max_iters: wp.int32  # max outer (linearized constraint subproblem) iterations
    k: wp.int32  # current iteration

    # --- Newton optimizer settings ---
    n_subproblem_max_iters: wp.int32  # max Newton iterations per subproblem
    gtol2: wp.float32  # squared gradient norm tolerance

    # --- Line search (backtracking Armijo) ---
    ls_max_iters: wp.int32  # max line search iterations
    ls_tau: wp.float32  # step size decrease factor
    ls_c: wp.float32  # Armijo slope scale
    ls_alpha: wp.float32  # initial step size

    # --- Hessian triplets (sparsity structure for BSR assembly) ---
    Hrows: wp.array[wp.int32]  # (nnz_blocks,) block row indices
    Hcols: wp.array[wp.int32]  # (nnz_blocks,) block column indices
    Hvals: wp.array[
        wp.mat33f
    ]  # (nnz_blocks,) block values (initialized to zero, filled in by GPU kernels)
    H: warp.sparse.BsrMatrix  # sparse Hessian matrix (assembled from triplets)
