import warp as wp
import numpy as np
from .. import types

# TODO: Implement contact dynamics data structure based on `source/pbat/sim/contact/MeshDynamics.h`.
# This is going to be a little complicated, since there's ogc state and multi-meshes for both dynamic and static meshes.
# Perhaps we should just straight away merge the static and dynamic meshes together into one.


@wp.struct
class ContactDynamics:
    """From `source/pbat/sim/contact/MeshDynamics.h`"""

    # --- Dynamic surface mesh ---
    dyn_V: wp.array[wp.int32]  # (nDV,) surface vertex indices into volume mesh
    dyn_F: wp.array[types.vec3i]  # type: ignore  # (nDF,) surface triangle connectivity
    dyn_E: wp.array[wp.int32]  # (2*nDE,) surface edge connectivity (flat pairs)
    dyn_n_vertices: wp.int32
    dyn_n_faces: wp.int32

    # --- Static surface mesh ---
    static_X: wp.array[wp.vec3f]  # (nSV,) static mesh positions
    static_V: wp.array[wp.int32]  # (nSV,) static vertex indices
    static_F: wp.array[
        types.vec3i  # type: ignore
    ]  # (nSF,) static triangle connectivity
    static_n_vertices: wp.int32
    static_n_faces: wp.int32

    # --- Contact parameters ---
    mu: wp.float32  # friction coefficient
    epsv: wp.float32  # IPC velocity threshold
    kc: wp.float32  # contact stiffness multiplier
    dmin: wp.float32  # target minimum contact distance
    gamma: wp.float32  # AL barrier multiplier (normal)
    gammaf: wp.float32  # AL barrier multiplier (friction)
