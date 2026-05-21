import warp as wp
from abc import ABC, abstractmethod
from . import pairs


class ContactDetection(ABC):
    """Abstract base class for contact detection algorithms."""

    _name: str
    _xt: wp.array[wp.vec3f]
    _xk: wp.array[wp.vec3f]
    _x: wp.array[wp.vec3f]
    _xtilde: wp.array[wp.vec3f]
    _contacts: pairs.ContactPairs

    def __init__(self, name: str):
        self._name = name

    def register_handles(
        self,
        xt: wp.array[wp.vec3f],
        xk: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        xtilde: wp.array[wp.vec3f],
        contacts: pairs.ContactPairs,
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
            contacts (pairs.ContactPairs): Contact pairs to write to.
        """
        self._xt = xt
        self._x = x
        self._xtilde = xtilde
        self._contacts = contacts
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
    def detect_contacts(self, from_xt: bool = False):
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

    @property
    def name(self) -> str:
        return self._name
