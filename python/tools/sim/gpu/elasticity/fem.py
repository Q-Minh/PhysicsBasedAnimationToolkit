import warp as wp
from .. import types


@wp.struct
class FemElastoDynamics:
    """Elastodynamic IVP (initial value problem) using FEM spatial discretization and BDF time discretization."""

    # --- Mesh (linear) ---
    X: wp.array[wp.vec3f]  # (N,) rest positions
    E: wp.array[
        types.vec4i  # type: ignore
    ]  # (E,) element connectivity (4 nodes per tet)

    # --- Dynamic state ---
    x: wp.array[wp.vec3f]  # (N,) current positions
    v: wp.array[wp.vec3f]  # (N,) velocities

    # --- Time integration ---
    xtilde: wp.array[wp.vec3f]  # (N,) BDF inertial target

    # --- Mass ---
    m: wp.array[wp.float32]  # (N,) lumped mass per node

    # --- External forces ---
    fext: wp.array[wp.vec3f]  # (N,) external force per node

    # --- Elastic quadrature ---
    wg: wp.array[wp.float32]  # (Q,) quadrature weights
    GNeg: wp.array[
        types.mat43f  # type: ignore
    ]  # (Q,) shape function gradients (4x3 per quad pt) at quad pts
    mug: wp.array[wp.float32]  # (Q,) 1st Lame parameter at quad pts
    lambdag: wp.array[wp.float32]  # (Q,) 2nd Lame parameter at quad pts

    # --- Dirichlet BCs ---
    dmask: wp.array[wp.int32]  # (N,) 0 = free, 1+ = constrained
    ndbc: wp.int32  # number of dirichlet constrained nodes
    dbc: wp.array[
        wp.int32
    ]  # (N,) concatenated vector of Dirichlet unconstrained and constrained nodes, partitioned as [ dbc[0:N-ndbc], dbc[N-ndbc:] ]
