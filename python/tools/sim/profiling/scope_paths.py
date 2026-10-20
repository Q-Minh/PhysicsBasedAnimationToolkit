"""
Profiling scope paths, shared by the recorder (``gpu/common/profiling.py``) and the result files and reports
(``profiling/results.py``). Kept free of GPU imports, so reading results doesn't load Warp, cupy or pbatoolkit.
"""

PATH_SEPARATOR = "/"


def path_ancestors(path: str) -> list[str]:
    """The path and its ancestor paths: ``"a/b/c"`` -> ``["a", "a/b", "a/b/c"]``. Work outside any
    scope (``""``) has no ancestors."""
    if not path:
        return [path]
    parts = path.split(PATH_SEPARATOR)
    return [PATH_SEPARATOR.join(parts[: n + 1]) for n in range(len(parts))]
