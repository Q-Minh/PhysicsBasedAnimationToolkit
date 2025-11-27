# type: ignore
from pbatoolkit import pbat
import numpy as np
import typing


def serialize_solver_iteration(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    contact: pbat.sim.contact.MeshDynamics,
    k: int,
    arc: pbat.io.Archive,
    f_serialize_more: typing.Callable[pbat.io.Archive, None] | None = None,
    post_solve: bool = False,
):
    """Serialize solver iteration data

    Args:
        fem (pbat.sim.dynamics.FemElastoDynamics): Finite element elasto dynamics
            problem
        contact (pbat.sim.contact.MeshDynamics): Contact dynamics problem
        k (int): Iteration index
        arc (pbat.io.Archive): Archive to store iteration data
        f_serialize_more (Callable[pbat.io.Archive, None], optional): Additional
            custom serialization function that takes in the iteration archive. Defaults to None.
        post_solve (bool, optional): Whether this is a post-solve iteration, in which case
            we also serialize velocity. Defaults to False.
    """
    iter = arc[f"{k:06d}"]
    iter.write_data("x", fem.x)
    # Time integration objective and its gradient
    f = fem.objective(fem.x.ravel())
    iter.write_metadata("f", f)
    g = fem.gradient(fem.x)
    iter.write_data("g", g)
    gnorm = np.linalg.norm(
        g
    )  # annoyingly, this returns numpy.float32 which nanobind does not cast automatically to C++ float
    iter.write_metadata("gnorm", float(gnorm))
    if f_serialize_more is not None:
        f_serialize_more(iter)
    # Velocities at post-solve
    if post_solve:
        iter.write_data("v", fem.v)
