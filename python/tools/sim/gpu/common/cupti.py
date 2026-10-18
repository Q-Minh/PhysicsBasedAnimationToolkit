"""
Minimal ctypes binding to the CUPTI activity API, for in-process GPU timing.

NVIDIA's ``cupti-python`` has no Windows wheels, so this module loads the CUPTI DLL shipped by the pip
package ``nvidia-cuda-cupti-cu12`` (pinned in ``requirements.txt``) and reads only what the profiler
needs: every kernel, memcpy and memset with its GPU start and end timestamps, and external correlation
records that map each launch to the scope id that was pushed when it was issued. It captures CUB and
cupy kernels too, unlike Warp's activity timing.

Record layouts follow ``cupti_activity.h`` of CUPTI 12.9 (``CUpti_ActivityKernel9``,
``CUpti_ActivityMemcpy6``, ``CUpti_ActivityMemset4``, ``CUpti_ActivityExternalCorrelation``). Only
the leading fields are read, which have been stable across recent record versions.
"""

import atexit
import ctypes
import dataclasses
import importlib.metadata
import threading

# CUpti_ActivityKind
_KIND_MEMCPY = 1
_KIND_MEMSET = 2
_KIND_DRIVER = 4
_KIND_RUNTIME = 5
_KIND_CONCURRENT_KERNEL = 10
_KIND_EXTERNAL_CORRELATION = 39
_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 3
_FLAG_FLUSH_FORCED = 1
_SUCCESS = 0
_BUFFER_SIZE = 8 * 1024 * 1024
_ALIGNMENT = 8

# Kinds enabled while collecting. Driver and runtime API records are needed for CUPTI to emit
# external correlation records; they are not stored.
_KINDS = (
    _KIND_CONCURRENT_KERNEL,
    _KIND_MEMCPY,
    _KIND_MEMSET,
    _KIND_DRIVER,
    _KIND_RUNTIME,
    _KIND_EXTERNAL_CORRELATION,
)

# Labels as Nsight Systems prints them, so the two backends name activities alike.
_MEMCPY_KINDS = {1: "Host-to-Device", 2: "Device-to-Host", 8: "Device-to-Device", 10: "Peer-to-Peer"}


def short_name(name: str) -> str:
    """The function name of an Itanium-mangled C++ kernel name (``_Z20memtile_value_kernelIiEvPT_S0_y``
    -> ``memtile_value_kernel``; nested names keep their last component). Other names are unchanged."""
    if not name.startswith("_Z"):
        return name
    i = 2
    nested = i < len(name) and name[i] == "N"
    i += nested
    last = name
    while i < len(name) and name[i].isdigit():
        j = i
        while j < len(name) and name[j].isdigit():
            j += 1
        length = int(name[i:j])
        last = name[j : j + length]
        i = j + length
        if not nested:
            break
        while i < len(name) and name[i] == "I":  # skip template arguments of a nested component
            depth, i = 1, i + 1
            while i < len(name) and depth:
                depth += {"I": 1, "N": 1, "L": 1, "E": -1}.get(name[i], 0)
                i += 1
    return last


@dataclasses.dataclass
class Activity:
    """One GPU activity, with timestamps in nanoseconds."""

    name: str
    start: int
    end: int
    correlation_id: int
    stream_id: int
    grid: int = 0
    block: int = 0
    registers: int = 0


_DISTRIBUTION = "nvidia-cuda-cupti-cu12"


def _find_dll() -> str:
    """The CUPTI DLL of the installed ``nvidia-cuda-cupti-cu12`` distribution."""
    try:
        files = importlib.metadata.distribution(_DISTRIBUTION).files or []
    except importlib.metadata.PackageNotFoundError:
        files = []
    dlls = sorted(f for f in files if f.name.startswith("cupti64_") and f.suffix == ".dll")
    if not dlls:
        raise RuntimeError(
            f"CUPTI not found. Install the {_DISTRIBUTION} version pinned in "
            "python/tools/sim/requirements.txt."
        )
    return str(dlls[-1].locate())


_REQUEST = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
)
_COMPLETE = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t
)


def _u8(p, o):
    return ctypes.c_uint8.from_address(p + o).value


def _u16(p, o):
    return ctypes.c_uint16.from_address(p + o).value


def _u32(p, o):
    return ctypes.c_uint32.from_address(p + o).value


def _i32(p, o):
    return ctypes.c_int32.from_address(p + o).value


def _u64(p, o):
    return ctypes.c_uint64.from_address(p + o).value


class ActivityCollector:
    """Collects GPU activity records and launch-to-external-id correlations from CUPTI.

    Usage::

        collector = ActivityCollector()
        collector.enable()
        collector.push(1)  # launches issued now correlate with external id 1
        ...
        collector.pop()
        activities, external = collector.flush()  # after wp.synchronize()
    """

    def __init__(self, dll_path: str | None = None):
        self._lib = ctypes.CDLL(dll_path or _find_dll())
        self._lib.cuptiActivityGetNextRecord.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.cuptiActivityPushExternalCorrelationId.argtypes = [ctypes.c_int, ctypes.c_uint64]
        self._lib.cuptiActivityPopExternalCorrelationId.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.cuptiGetResultString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._buffers: dict[int, ctypes.Array] = {}
        self._lock = threading.Lock()
        self._activities: list[Activity] = []
        self._external: dict[int, int] = {}
        # Keep the callbacks alive for as long as CUPTI may call them.
        self._request_cb = _REQUEST(self._on_request)
        self._complete_cb = _COMPLETE(self._on_complete)
        self._check(self._lib.cuptiActivityRegisterCallbacks(self._request_cb, self._complete_cb))
        self._enabled = False
        # CUPTI delivers its last buffers through the Python callbacks when the process exits. Deliver
        # them and detach CUPTI while the interpreter is still alive.
        atexit.register(self.close)

    def _check(self, result: int):
        """Raise RuntimeError with CUPTI's message if result is not CUPTI_SUCCESS."""
        if result != _SUCCESS:
            message = ctypes.c_char_p()
            self._lib.cuptiGetResultString(result, ctypes.byref(message))
            raise RuntimeError(f"CUPTI error {result}: {(message.value or b'').decode()}")

    def _on_request(self, buffer, size, max_records):
        """CUPTI callback: hand out an aligned activity buffer."""
        raw = ctypes.create_string_buffer(_BUFFER_SIZE + _ALIGNMENT)
        address = (ctypes.addressof(raw) + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
        with self._lock:
            self._buffers[address] = raw
        buffer[0] = address
        size[0] = _BUFFER_SIZE
        max_records[0] = 0

    def _on_complete(self, context, stream_id, buffer, size, valid_size):
        """CUPTI callback: parse a filled buffer into activities and correlations, then free it."""
        record = ctypes.c_void_p()
        activities, external = [], {}
        while (
            self._lib.cuptiActivityGetNextRecord(buffer, valid_size, ctypes.byref(record)) == _SUCCESS
        ):
            p = record.value
            kind = _u32(p, 0)
            if kind == _KIND_CONCURRENT_KERNEL:
                name = ctypes.string_at(ctypes.c_void_p.from_address(p + 104).value).decode(
                    errors="replace"
                )
                activities.append(
                    Activity(
                        name=short_name(name),
                        start=_u64(p, 16),
                        end=_u64(p, 24),
                        correlation_id=_u32(p, 92),
                        stream_id=_u32(p, 48),
                        grid=_i32(p, 52) * _i32(p, 56) * _i32(p, 60),
                        block=_i32(p, 64) * _i32(p, 68) * _i32(p, 72),
                        registers=_u16(p, 6),
                    )
                )
            elif kind == _KIND_MEMCPY:
                copy_kind = _MEMCPY_KINDS.get(_u8(p, 4), str(_u8(p, 4)))
                activities.append(
                    Activity(f"memcpy {copy_kind}", _u64(p, 16), _u64(p, 24), _u32(p, 44), _u32(p, 40))
                )
            elif kind == _KIND_MEMSET:
                activities.append(
                    Activity("memset", _u64(p, 16), _u64(p, 24), _u32(p, 44), _u32(p, 40))
                )
            elif kind == _KIND_EXTERNAL_CORRELATION:
                if _u32(p, 4) == _EXTERNAL_CORRELATION_KIND_CUSTOM0:
                    external[_u32(p, 16)] = _u64(p, 8)
        with self._lock:
            self._activities.extend(activities)
            self._external.update(external)
            self._buffers.pop(buffer, None)

    def enable(self):
        """Start collecting the activity kinds this module reads."""
        if not self._enabled:
            for kind in _KINDS:
                self._check(self._lib.cuptiActivityEnable(kind))
            self._enabled = True

    def disable(self):
        """Stop collecting activity records."""
        if self._enabled:
            for kind in _KINDS:
                self._check(self._lib.cuptiActivityDisable(kind))
            self._enabled = False

    def close(self):
        """Disable collection, deliver pending records and detach CUPTI from the process."""
        self.disable()
        self._lib.cuptiActivityFlushAll(_FLAG_FLUSH_FORCED)
        self._lib.cuptiFinalize()

    def push(self, external_id: int):
        """Correlate launches issued from now on (on this thread) with ``external_id``."""
        self._check(
            self._lib.cuptiActivityPushExternalCorrelationId(
                _EXTERNAL_CORRELATION_KIND_CUSTOM0, external_id
            )
        )

    def pop(self):
        """Stop correlating launches with the most recently pushed external id."""
        last = ctypes.c_uint64()
        self._check(
            self._lib.cuptiActivityPopExternalCorrelationId(
                _EXTERNAL_CORRELATION_KIND_CUSTOM0, ctypes.byref(last)
            )
        )

    def flush(self) -> tuple[list[Activity], dict[int, int]]:
        """Deliver all pending records and return (activities, correlation id -> external id)
        collected since the previous flush. Synchronize the device first so every activity is done."""
        self._check(self._lib.cuptiActivityFlushAll(_FLAG_FLUSH_FORCED))
        with self._lock:
            activities, external = self._activities, self._external
            self._activities, self._external = [], {}
        return activities, external


import unittest


class TestShortName(unittest.TestCase):

    def test_short_names(self):
        self.assertEqual(short_name("_Z20memtile_value_kernelIiEvPT_S0_y"), "memtile_value_kernel")
        self.assertEqual(
            short_name("_ZN2wp16bvh_refit_kernelEiPKiPiS1_PNS_17BVHPackedNodeHalfES4_"), "bvh_refit_kernel"
        )
        self.assertEqual(
            short_name("_ZN3cub6detail16DeviceScanKernelINS0_10policy_hubIiEEPiEEvT_T0_"),
            "DeviceScanKernel",
        )
        self.assertEqual(short_name("_vertex_solve_kernel_cuda_kernel_forward"), "_vertex_solve_kernel_cuda_kernel_forward")


if __name__ == "__main__":
    unittest.main()
