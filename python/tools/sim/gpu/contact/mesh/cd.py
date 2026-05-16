import warp as wp
from abc import ABC, abstractmethod
from . import pairs


class ContactDetection(ABC):
    """Abstract base class for contact detection algorithms."""

    _name: str

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def on_time_step_started(self, xt: wp.array[wp.vec3f], x: wp.array[wp.vec3f]):
        """Callback for any one-time precomputation for the duration of the
        current time step.

        Args:
            xt (wp.array[wp.vec3f]): Point positions at the start of the
            time step.
            x (wp.array[wp.vec3f]): Current point positions (may be
            different from xt).
        """
        pass

    @abstractmethod
    def on_contact_detection_starting(
        self, xt: wp.array[wp.vec3f], x: wp.array[wp.vec3f]
    ):
        """Callback for any one-time precomputation to be done right
        before the next contact detection.

        Args:
            xt (wp.array[wp.vec3f]): Point positions at the start of
            the time step.
            x (wp.array[wp.vec3f]): Current point positions.
        """
        pass

    @abstractmethod
    def detect_contacts(
        self,
        xt: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        contacts: pairs.ContactPairs,
    ):
        """Detects and writes contact pairs (u, v), and contact
        count (prefix[-1]) for any detected vv, ve, vf and ee contact.

        Postconditions:
        - For any given i, all detected pairs (i,j) of type * in the arrays
        *.u and *.v are unique and sorted by j, but not necessarily
        contiguous.
        - The total number of detected contacts for each type * is stored
        in *.prefix[-1].

        Args:
            xt (wp.array[wp.vec3f]): Point positions at the start of
            the time step.
            x (wp.array[wp.vec3f]): Current point positions.
            contacts (pairs.ContactPairs): Contact pairs to populate.
        """
        pass

    @abstractmethod
    def on_contact_detection_ended(self, xt: wp.array[wp.vec3f], x: wp.array[wp.vec3f]):
        """Callback for any one-time post-computation after the contact detection
        phase.

        Args:
            xt (wp.array[wp.vec3f]): Point positions at the start of
            the time step.
            x (wp.array[wp.vec3f]): Current point positions.
        """
        pass

    @abstractmethod
    def on_time_step_ended(self, xt: wp.array[wp.vec3f], x: wp.array[wp.vec3f]):
        """Callback for any one-time post-computation before the start of the next
        time step.

        Args:
            xt (wp.array[wp.vec3f]): Point positions at the start of
            the time step.
            x (wp.array[wp.vec3f]): Current point positions.
        """
        pass

    @property
    def name(self) -> str:
        return self._name
