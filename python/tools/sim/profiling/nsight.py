"""
Locate the Nsight Systems (``nsys``) and Nsight Compute (``ncu``) executables.

:func:`find_tool` resolves a tool at run time, in this order:

1. an explicit path (the ``--nsys`` / ``--ncu`` command line argument);
2. the environment variable ``NSYS`` / ``NCU``;
3. the tool on ``PATH`` (``shutil.which``);
4. the standard install locations, newest version first:
   - Windows: ``%ProgramFiles%/NVIDIA Corporation/Nsight Systems <version>/target-windows-x64/nsys.exe``
     and ``%ProgramFiles%/NVIDIA Corporation/Nsight Compute <version>/ncu.exe`` or
     ``.../target/<platform>/ncu.exe``;
   - Linux: ``/opt/nvidia/nsight-systems/<version>/bin/nsys``, ``/opt/nvidia/nsight-compute/<version>/ncu``,
     then ``$CUDA_PATH/bin`` and ``/usr/local/cuda/bin``.

On Windows, Nsight Compute puts an ``ncu.bat`` wrapper on ``PATH``. Batch files run through ``cmd.exe``,
which strips the ``^`` of kernel name regexes, so a wrapper found on ``PATH`` is replaced by the
``ncu.exe`` it wraps when that exists.
"""

import glob
import os
import re
import shutil
import sys

# Tool name -> (environment variable, command line option)
_TOOLS = {"nsys": ("NSYS", "--nsys"), "ncu": ("NCU", "--ncu")}


def _version_key(path: str) -> tuple[int, ...]:
    """Sort key of a versioned install directory, e.g. ``Nsight Systems 2026.1.3`` -> (2026, 1, 3)."""
    return tuple(int(x) for x in re.findall(r"\d+", os.path.basename(path)))


def _newest_first(pattern: str) -> list[str]:
    """Directories matching ``pattern``, highest version first."""
    return sorted((d for d in glob.glob(pattern) if os.path.isdir(d)), key=_version_key, reverse=True)


def _first_file(patterns: list[str]) -> str | None:
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return path
    return None


def _ncu_exe_in(root: str) -> str | None:
    """The ``ncu.exe`` of a Windows Nsight Compute install directory (which the ``ncu.bat`` wrapper there
    launches), if present."""
    return _first_file([os.path.join(root, "ncu.exe"), os.path.join(root, "target", "*", "ncu.exe")])


def _install_candidates(name: str) -> list[str]:
    """Executables of ``name`` in the standard install locations, newest version first."""
    found = []
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles")
        if not program_files:
            return []
        nvidia = os.path.join(program_files, "NVIDIA Corporation")
        if name == "nsys":
            for d in _newest_first(os.path.join(nvidia, "Nsight Systems *")):
                found += glob.glob(os.path.join(d, "target-windows-x64", "nsys.exe"))
        else:
            for d in _newest_first(os.path.join(nvidia, "Nsight Compute *")):
                exe = _ncu_exe_in(d)
                found += [exe] if exe else []
    else:
        if name == "nsys":
            for d in _newest_first("/opt/nvidia/nsight-systems/*"):
                found.append(os.path.join(d, "bin", "nsys"))
        else:
            for d in _newest_first("/opt/nvidia/nsight-compute/*"):
                found.append(os.path.join(d, "ncu"))
        cuda_bins = ([os.path.join(os.environ["CUDA_PATH"], "bin")] if os.environ.get("CUDA_PATH") else []) + [
            "/usr/local/cuda/bin"
        ]
        found += [os.path.join(b, name) for b in cuda_bins]
    return [p for p in found if os.path.isfile(p)]


def find_tool(name: str, path: str | None = None) -> str:
    """Path of the Nsight executable ``name`` (``"nsys"`` or ``"ncu"``), resolved as the module docstring
    describes.

    Args:
        name (str): ``"nsys"`` or ``"ncu"``.
        path (str | None): Explicit path (the ``--nsys`` / ``--ncu`` argument), used as given.

    Raises:
        FileNotFoundError: If the tool is found nowhere.
    """
    env, option = _TOOLS[name]
    if path:
        return path
    if os.environ.get(env):
        return os.environ[env]
    on_path = shutil.which(name)
    if on_path:
        if name == "ncu" and on_path.lower().endswith((".bat", ".cmd")):
            return _ncu_exe_in(os.path.dirname(on_path)) or on_path
        return on_path
    candidates = _install_candidates(name)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"{name} not found on PATH or in the standard install locations. Pass {option} path/to/{name}, "
        f"or set the {env} environment variable to the executable."
    )


import unittest
from unittest import mock


class TestFindTool(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, *parts: str) -> str:
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def _env(self, **env):
        return mock.patch.dict(os.environ, env, clear=True)

    def test_order(self):
        on_path = self._touch("bin", "nsys.exe")
        with self._env(NSYS="from/env/nsys"), mock.patch("shutil.which", return_value=on_path):
            self.assertEqual(find_tool("nsys", "from/arg/nsys"), "from/arg/nsys")
            self.assertEqual(find_tool("nsys"), "from/env/nsys")
        with self._env(), mock.patch("shutil.which", return_value=on_path):
            self.assertEqual(find_tool("nsys"), on_path)

    @mock.patch.object(sys, "platform", "win32")
    def test_windows_install_locations_newest_first(self):
        nvidia = ("NVIDIA Corporation",)
        self._touch(*nvidia, "Nsight Systems 2025.9.1", "target-windows-x64", "nsys.exe")
        new = self._touch(*nvidia, "Nsight Systems 2026.10.0", "target-windows-x64", "nsys.exe")
        self._touch(*nvidia, "Nsight Systems 2026.2.0", "target-windows-x64", "nsys.exe")
        ncu = self._touch(*nvidia, "Nsight Compute 2026.2.1", "target", "windows-desktop-win7-x64", "ncu.exe")
        with self._env(ProgramFiles=self.root), mock.patch("shutil.which", return_value=None):
            self.assertEqual(find_tool("nsys"), new)
            self.assertEqual(find_tool("ncu"), ncu)

    @mock.patch.object(sys, "platform", "win32")
    def test_windows_ncu_wrapper_on_path(self):
        wrapper = self._touch("Nsight Compute 2026.2.1", "ncu.bat")
        exe = self._touch("Nsight Compute 2026.2.1", "target", "windows-desktop-win7-x64", "ncu.exe")
        with self._env(), mock.patch("shutil.which", return_value=wrapper):
            self.assertEqual(find_tool("ncu"), exe)
        os.remove(exe)
        with self._env(), mock.patch("shutil.which", return_value=wrapper):
            self.assertEqual(find_tool("ncu"), wrapper)

    @mock.patch.object(sys, "platform", "linux")
    def test_linux_cuda_path(self):
        exe = self._touch("cuda", "bin", "ncu")
        real_glob = glob.glob
        no_opt = lambda p: [] if p.startswith("/opt/") else real_glob(p)  # ignore a real Nsight install
        with self._env(CUDA_PATH=os.path.join(self.root, "cuda")), mock.patch("shutil.which", return_value=None):
            with mock.patch("glob.glob", side_effect=no_opt):
                self.assertEqual(find_tool("ncu"), exe)

    def test_not_found(self):
        with self._env(), mock.patch("shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError) as e:
                find_tool("ncu")
            self.assertIn("--ncu", str(e.exception))
            self.assertIn("NCU", str(e.exception))


if __name__ == "__main__":
    unittest.main()
