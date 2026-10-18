"""
Profiling scopes for the GPU simulator.

Wrap algorithm phases in ``with profiling.scope("anderson.gram"): ...`` using hierarchical, dotted names
(e.g. ``vbd.sweep``, ``contact.detect``, ``newton.linear_solve``). Scopes nest at run time.

This is the Phase 0 stub (see plans/02-profiling.md): profiling is always off, so ``scope`` returns a
shared no-op context manager. It performs no allocation and no device work, so it is safe inside CUDA
graph capture. Task 2 adds the backends (kernel-level GPU busy time attributed to the innermost open
scope, NVTX ranges) and their selection without changing this interface.
"""

import contextlib

_NO_OP_SCOPE = contextlib.nullcontext()


def scope(name: str) -> contextlib.AbstractContextManager:
    """Return a context manager that profiles the enclosed code under ``name``.

    Args:
        name (str): Hierarchical, dot-separated phase name, e.g. ``"vbd.sweep"``.

    Returns:
        contextlib.AbstractContextManager: A no-op while profiling is off.
    """
    return _NO_OP_SCOPE
