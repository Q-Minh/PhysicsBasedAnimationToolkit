from typing import Dict

import warp as wp

from .ogc import ContactPairsData, OgcData, Ogc
from .. import common
from ..common.buffer import DoubleBuffer


# TODO: Review/refactor this code.
# 1. Implement contact basis construction + constraint linearization. We might want
#    to make OGC responsible for basis construction.
# 2. Implement dual update

# ---------------------------------------------------------------------------
# Kernels (type-independent — shared by all four contact types)
# ---------------------------------------------------------------------------


@wp.kernel
def _init_contact_set(
    fwd: ContactPairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_fwd_key: wp.int32,  # index into fwd.prefix that holds the total contact count
    cur_u: wp.array[wp.int32],
    cur_v: wp.array[wp.int32],
    cur_lambda: wp.array[wp.float32],
    cur_s: wp.array[wp.float32],
    cur_gamma: wp.array[wp.float32],
    cur_lambda_f: wp.array[wp.vec2f],
    cur_sigma: wp.array[wp.float32],
    cur_sigma_f: wp.array[wp.float32],
):
    """Initialize the k-th new OGC contact in the current buffer.

    Each thread k handles one contact from the new OGC forward list.  The pair
    keys (u, v) are copied from the OGC list and all augmented-Lagrangian state
    fields are set to their defaults:
      lambda = 0, s = 0, gamma = 1, lambda_f = [0,0], sigma = 0, sigma_f = 0.

    Fields that are computed/overwritten by the linearization pass (chat, gradc,
    c, T, W, cf, dhat) are intentionally left untouched.
    """
    k = wp.tid()
    nfwd = fwd.prefix[n_fwd_key]
    if k >= nfwd:
        return
    cur_u[k] = fwd.u[k]
    cur_v[k] = fwd.v[k]
    cur_lambda[k] = wp.float32(0)
    cur_s[k] = wp.float32(0)
    cur_gamma[k] = wp.float32(1)
    cur_lambda_f[k] = wp.vec2f(0, 0)
    cur_sigma[k] = wp.float32(0)
    cur_sigma_f[k] = wp.float32(0)


@wp.kernel
def _warm_start_contact_set(
    fwd: ContactPairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_fwd_key: wp.int32,  # index into fwd.prefix that holds the total contact count
    # Alternate (previous-step) buffer: read then invalidate ---------------
    alt_u: wp.array[wp.int32],
    alt_v: wp.array[wp.int32],
    alt_lambda: wp.array[wp.float32],
    alt_s: wp.array[wp.float32],
    alt_gamma: wp.array[wp.float32],
    alt_lambda_f: wp.array[wp.vec2f],
    alt_sigma: wp.array[wp.float32],
    alt_sigma_f: wp.array[wp.float32],
    # Current (new-step) buffer: overwrite defaults with warm-started state -
    cur_lambda: wp.array[wp.float32],
    cur_s: wp.array[wp.float32],
    cur_gamma: wp.array[wp.float32],
    cur_lambda_f: wp.array[wp.vec2f],
    cur_sigma: wp.array[wp.float32],
    cur_sigma_f: wp.array[wp.float32],
):
    """Warm-start the current buffer from the alternate buffer and invalidate the alternate.

    For each thread k:
      1. Read (u, v) from the alternate buffer and immediately invalidate the slot
         (u = v = -1) so stale entries are clearly marked for future steps.
      2. If the slot was already empty (u < 0), return.
      3. Binary-search for (u, v) in the new forward contact list.
      4. If found at position l, overwrite the defaults written by
         ``_init_contact_set`` with the warm-started augmented-Lagrangian state.

    Only the state that persists across steps is copied: lambda, s, gamma,
    lambda_f, sigma, sigma_f.  Geometric quantities (chat, gradc, c, T, W, cf,
    dhat) are recomputed by the subsequent linearization pass.
    """
    k = wp.tid()
    # Read pair keys and immediately invalidate the alternate slot.
    u = alt_u[k]
    v = alt_v[k]
    alt_u[k] = wp.int32(-1)
    alt_v[k] = wp.int32(-1)
    if u < wp.int32(0):
        return  # slot was already empty
    nfwd = fwd.prefix[n_fwd_key]
    l = common.lower_bound(fwd.u, fwd.v, nfwd, u, v)  # type: ignore
    if l >= nfwd or fwd.u[l] != u or fwd.v[l] != v:
        return  # contact dropped from the new set; discard prior state
    cur_lambda[l] = alt_lambda[k]
    cur_s[l] = alt_s[k]
    cur_gamma[l] = alt_gamma[k]
    cur_lambda_f[l] = alt_lambda_f[k]
    cur_sigma[l] = alt_sigma[k]
    cur_sigma_f[l] = alt_sigma_f[k]


# ---------------------------------------------------------------------------
# ContactConstraintSet
# ---------------------------------------------------------------------------


class ContactConstraintSet:
    """GPU contact constraint set backed by a dictionary of double-buffered arrays.

    Each field is a :class:`~..common.buffer.DoubleBuffer` whose ``current``
    array holds the active state for the current simulation step and whose
    ``alternate`` array holds the state from the previous step (used for
    warm-starting).

    Fields
    ------
    u, v       : ``int32``          contact pair keys; ``-1`` marks an empty slot.
    lambda     : ``float32``        normal Lagrange multiplier.
    s          : ``float32``        inequality slack variable.
    gamma      : ``float32``        constraint decay factor (default-initialised to 1).
    chat       : ``float32``        linearisation constant c(xk) - grad_c(xk)·xk.
    gradc      : ``vec[kDofs]``     constraint gradient at the linearisation point.
    c          : ``float32``        last evaluated constraint value c(x).
    T          : ``vec6f``          3×2 tangent basis (row-major flattened, 6 entries).
    W          : ``vec[kStencil]``  tangential operator weights.
    cf         : ``vec2f``          friction constraint value T^T W (x - x_t).
    dhat       : ``vec2f``          T^T W x_t.
    lambda_f   : ``vec2f``          friction Lagrange multiplier.
    sigma      : ``float32``        normal contact penalty parameter.
    sigma_f    : ``float32``        friction penalty parameter.

    Parameters
    ----------
    capacity : int
        Maximum number of contact constraints (matches the OGC contact list capacity).
    kStencil : int
        Number of mesh primitives in the contact stencil (2 for VV, 3 for VE, 4 for VF/EE).
    """

    def __init__(self, capacity: int, kStencil: int):
        kDofs = 3 * kStencil
        grad_t = wp.vec_t(kDofs, wp.float32)
        W_t = wp.vec_t(kStencil, wp.float32)
        T_t = wp.vec_t(6, wp.float32)  # 3×2 = 6 entries, same for all contact types

        def _pair(shape, dtype, *, fill=None):
            if fill is not None:
                a = wp.full(shape=shape, value=fill, dtype=dtype)
                b = wp.full(shape=shape, value=fill, dtype=dtype)
            else:
                a = wp.zeros(shape=shape, dtype=dtype)
                b = wp.zeros(shape=shape, dtype=dtype)
            return DoubleBuffer(a, b)

        self.capacity = capacity
        self.kStencil = kStencil
        self.kDofs = kDofs

        self.buffers: Dict[str, DoubleBuffer] = {
            # Contact pair keys — -1 = invalid/empty slot
            "u":        _pair((capacity,), wp.int32,   fill=-1),
            "v":        _pair((capacity,), wp.int32,   fill=-1),
            # Augmented-Lagrangian state (warm-started across steps)
            "lambda":   _pair((capacity,), wp.float32),
            "s":        _pair((capacity,), wp.float32),
            "gamma":    _pair((capacity,), wp.float32, fill=1.0),  # decay, default 1
            "sigma":    _pair((capacity,), wp.float32),
            "sigma_f":  _pair((capacity,), wp.float32),
            "lambda_f": _pair((capacity,), wp.vec2f),
            # Geometric / linearization state (recomputed each step, not warm-started)
            "chat":     _pair((capacity,), wp.float32),
            "gradc":    _pair((capacity,), grad_t),
            "c":        _pair((capacity,), wp.float32),
            "T":        _pair((capacity,), T_t),
            "W":        _pair((capacity,), W_t),
            "cf":       _pair((capacity,), wp.vec2f),
            "dhat":     _pair((capacity,), wp.vec2f),
        }

    def swap(self):
        """Swap current ↔ alternate for every field simultaneously."""
        for buf in self.buffers.values():
            buf.swap()

    @property
    def current(self) -> Dict[str, wp.array]:
        """Dictionary mapping field name → current (active) wp.array."""
        return {k: v.current for k, v in self.buffers.items()}

    @property
    def alternate(self) -> Dict[str, wp.array]:
        """Dictionary mapping field name → alternate (previous-step) wp.array."""
        return {k: v.alternate for k, v in self.buffers.items()}


# ---------------------------------------------------------------------------
# MeshDynamics
# ---------------------------------------------------------------------------


class MeshDynamics:
    """GPU contact mesh dynamics.

    Maintains four double-buffered contact constraint sets — vertex-vertex (VV),
    vertex-edge (VE), vertex-triangle (VF) and edge-edge (EE) — that are
    refreshed each step from a freshly computed OGC contact list.

    Contacts that persist across two consecutive steps are warm-started from the
    previous step's augmented-Lagrangian state (lambda, s, gamma, lambda_f,
    sigma, sigma_f).  Brand-new contacts receive default values.  Contacts that
    disappear from the new OGC list are silently dropped.
    """

    _vv: ContactConstraintSet
    _ve: ContactConstraintSet
    _vf: ContactConstraintSet
    _ee: ContactConstraintSet
    _streams: list[wp.Stream]

    def __init__(self, ogc: Ogc):
        """Construct the mesh dynamics state.

        Args:
            ogc (Ogc): Fully constructed OGC contact detector.  Contact
                capacities and mesh primitive counts are read from ``ogc`` at
                construction time and kept fixed for the lifetime of this object.
        """
        vv_cap, ve_cap, vf_cap, ee_cap = ogc.capacity
        self._vv = ContactConstraintSet(vv_cap, kStencil=2)
        self._ve = ContactConstraintSet(ve_cap, kStencil=3)
        self._vf = ContactConstraintSet(vf_cap, kStencil=4)
        self._ee = ContactConstraintSet(ee_cap, kStencil=4)

        # Each contact type's prefix array stores the total count at index `nu`,
        # where nu = ContactPairs.nu (the "u" dimension).
        self._n_vv_key: int = ogc._vv.nu  # = n_verts
        self._n_ve_key: int = ogc._ve.nu  # = n_verts
        self._n_vf_key: int = ogc._vf.nu  # = n_verts
        self._n_ee_key: int = ogc._ee.nu  # = n_half_edges

        self._streams = [wp.Stream() for _ in range(4)]

    def update_contact_set(self, ogc_data: OgcData):
        """Update all contact constraint sets from a freshly computed OGC contact list.

        Algorithm
        ---------
        1. **Swap** current ↔ alternate for every field in all four sets (CPU
           pointer swap, zero GPU cost).  The prior step's active constraints
           now live in the alternate buffers.
        2. **Init** every new OGC contact in the current buffer with default
           augmented-Lagrangian state (lambda=0, s=0, gamma=1, lambda_f=[0,0],
           sigma=0, sigma_f=0); pair keys (u, v) are copied from the OGC list.
        3. **Warm-start** contacts that persist: for each alternate entry whose
           (u, v) keys are valid, binary-search for that pair in the new forward
           contact list and, if found, overwrite the just-initialised defaults
           with the prior step's state.
        4. **Invalidate** each alternate entry (u = v = -1) — done inline during
           step 3 so no extra kernel is needed.

        A cross-stream barrier between steps 2 and 3 ensures the initialisation
        writes are visible before the warm-start writes begin.  Steps 2–4 are
        overlapped across the four contact types via four independent CUDA streams.

        Args:
            ogc_data (OgcData): Contact pairs produced by ``Ogc.detect_contacts()``.
        """
        main_stream = wp.get_stream()

        # -- Step 1: CPU-side pointer swap (free) ----------------------------------
        self._vv.swap()
        self._ve.swap()
        self._vf.swap()
        self._ee.swap()

        # Tuples: (ContactConstraintSet, forward ContactPairsData, n_fwd_key, stream index)
        contact_types = [
            (self._vv, ogc_data.vv, self._n_vv_key, 0),
            (self._ve, ogc_data.ve, self._n_ve_key, 1),
            (self._vf, ogc_data.vf, self._n_vf_key, 2),
            (self._ee, ogc_data.ee, self._n_ee_key, 3),
        ]

        # -- Step 2: Initialize new contacts in parallel across contact types ------
        for cs, fwd, n_key, si in contact_types:
            cur = cs.current
            with wp.ScopedStream(self._streams[si]):
                wp.launch(
                    _init_contact_set,
                    dim=cs.capacity,
                    inputs=[
                        fwd,
                        wp.int32(n_key),
                        cur["u"],
                        cur["v"],
                        cur["lambda"],
                        cur["s"],
                        cur["gamma"],
                        cur["lambda_f"],
                        cur["sigma"],
                        cur["sigma_f"],
                    ],
                )

        # Cross-stream barrier: warm-start must not begin until all inits are done.
        # Pattern: main_stream gathers all side streams, then side streams re-sync
        # from main_stream, establishing a full barrier across streams 0-3.
        for s in self._streams:
            main_stream.wait_stream(s)
        for s in self._streams:
            s.wait_stream(main_stream)

        # -- Steps 3 & 4: Warm-start from alternate + invalidate, in parallel ------
        for cs, fwd, n_key, si in contact_types:
            alt = cs.alternate
            cur = cs.current
            with wp.ScopedStream(self._streams[si]):
                wp.launch(
                    _warm_start_contact_set,
                    dim=cs.capacity,
                    inputs=[
                        fwd,
                        wp.int32(n_key),
                        # alternate (old state, will be invalidated in-kernel)
                        alt["u"],
                        alt["v"],
                        alt["lambda"],
                        alt["s"],
                        alt["gamma"],
                        alt["lambda_f"],
                        alt["sigma"],
                        alt["sigma_f"],
                        # current (warm-start targets)
                        cur["lambda"],
                        cur["s"],
                        cur["gamma"],
                        cur["lambda_f"],
                        cur["sigma"],
                        cur["sigma_f"],
                    ],
                )

        # Final fence: caller can safely read results on the main stream.
        for s in self._streams:
            main_stream.wait_stream(s)
