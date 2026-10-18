"""
Run a Python module inside a virtual environment from the base interpreter.

Nsight Systems hangs when it launches a venv's ``python.exe`` on Windows (the venv launcher stub
re-spawns the base interpreter). Launch the base interpreter on this script instead::

    nsys profile ... <base python.exe> python/tools/sim/profiling/venvrun.py <venv dir> <module> [args...]

It points ``sys.prefix`` at the venv before adding its site-packages, because ``cuda.pathfinder``
(used by ``cuda.compute``) locates NVIDIA libraries relative to the prefix; setting ``PYTHONPATH``
alone is not enough. The current directory is put first on ``sys.path``, as ``python -m`` does.
"""

import os
import runpy
import site
import sys


def main():
    venv, module, *args = sys.argv[1:]
    sys.prefix = sys.exec_prefix = venv
    os.environ["VIRTUAL_ENV"] = venv
    site.PREFIXES[:] = [venv]
    site.addsitedir(os.path.join(venv, "Lib", "site-packages"))
    sys.path.insert(0, os.getcwd())
    sys.argv = [module] + args
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
