import warp as wp
from abc import ABC, abstractmethod
from . import pairs
from ..multimesh import MultiMesh


class ContactDetection(ABC):
    """Abstract base class for contact detection algorithms."""

    _name: str
    _xt: wp.array[wp.vec3f]
    _xk: wp.array[wp.vec3f]
    _x: wp.array[wp.vec3f]
    _xtilde: wp.array[wp.vec3f]
    _dmask: wp.array[wp.int32]
    _meshes: MultiMesh
    _contacts: pairs.ContactPairs

    def __init__(self, name: str):
        self._name = name

    def register_handles(
        self,
        xt: wp.array[wp.vec3f],
        xk: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        xtilde: wp.array[wp.vec3f],
        meshes: MultiMesh,
        contacts: pairs.ContactPairs,
        dmask: wp.array[wp.int32] | None = None,
    ):
        """Registers any necessary handles for contact detection. This is called
        once at the beginning of the simulation, so that every time step is CUDA
        graph compatible. If xk is None, the ContactDetection implementation should
        allocate its own handle for xk, and update it as needed.

        Args:
            xt (wp.array[wp.vec3f]): Point positions at the start of the time step.
            xk (wp.array[wp.vec3f]): Point positions from the last call to detect_contacts.
            x (wp.array[wp.vec3f]): Current point positions.
            xtilde (wp.array[wp.vec3f]): Time integrator inertial target.
            meshes (MultiMesh): Mesh data for contact detection queries.
            contacts (pairs.ContactPairs): Contact pairs to write to.
            dmask (wp.array[wp.int32] | None): (N,) Dirichlet mask (nonzero = constrained).
                filter_step never moves constrained points, like C++
                ``MeshDynamics::RestoreFeasibility(x, dmask)``. If None, no point is constrained.
        """
        self._xt = xt
        self._x = x
        self._xtilde = xtilde
        self._meshes = meshes
        self._contacts = contacts
        self._dmask = (
            dmask if dmask is not None else wp.zeros(x.shape[0], dtype=wp.int32)
        )
        if xk is not None:
            self._xk = xk
        else:
            self._xk = wp.zeros_like(x)

    @abstractmethod
    def on_time_step_started(self):
        """Callback for any one-time precomputation for the duration of the
        current time step.
        """
        pass

    @abstractmethod
    def detect_contacts(self, from_xt: bool = False, fresh: bool = False):
        """Detects and writes contact pairs (u, v), and contact
        count (prefix[-1]) for any detected vv, ve, vf and ee contact.

        Postconditions:
        - For any given i, all detected pairs (i,j) of type * in the arrays
        *.u and *.v are unique and sorted by j, but not necessarily
        contiguous.
        - The total number of detected contacts for each type * is stored
        in *.prefix[-1].
        - self._contacts.assemble_contacts() is called so that ContactPairs 
        are readable.

        Args:
            from_xt (bool): If True, detects contacts based on the positions at
            the start of the time step (xt) instead of the current positions (x).
            fresh (bool): If True, a full detection even when the detector keeps
            linearizations within a solve (:attr:`freezes_within_solve`): the
            solve's last detection, whose keys the next solve warm-starts from.
            Detectors that don't freeze ignore it.
        """
        pass

    @abstractmethod
    def filter_step(self):
        """Modifies x so that the step xk -> x is improved."""
        pass

    @abstractmethod
    def on_time_step_ended(self):
        """Callback for any one-time post-computation before the start of the next
        time step.
        """
        pass

    def prepare_detection(self):
        """Work of :meth:`detect_contacts` that a conditional CUDA graph body
        (``wp.capture_if``/``wp.capture_while``) can't hold, e.g. allocations.

        Solvers that record detection inside conditional bodies call this first, outside them, then
        ``detect_contacts(prepared=True)``. Does nothing by default.
        """
        pass

    @property
    def supports_conditional_capture(self) -> bool:
        """Whether ``detect_contacts(prepared=True)`` and :meth:`filter_step` can be recorded inside
        a conditional CUDA graph body, after :meth:`prepare_detection` ran outside it. Solvers keep
        detection outside conditional bodies otherwise. Conservatively False unless verified.
        """
        return False

    @property
    def freezes_within_solve(self) -> bool:
        """Whether re-detections within a solve keep the stencils of pairs the previous detection found (vertex
        SDF's ``freeze_within_solve``). Solvers then make the solve's last detection a full one: ``fresh=True`` on
        the last scheduled re-detection, and once more after the outer loop if :attr:`last_detection_frozen` is
        set (an early exit). False by default."""
        return False

    @property
    def last_detection_frozen(self) -> wp.array | None:
        """(1,) int32 device flag, nonzero while the last detection kept frozen stencils, for a capture-safe
        ``wp.capture_if`` after the outer loop; None for detectors that don't freeze."""
        return None

    @property
    def name(self) -> str:
        return self._name
