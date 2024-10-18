import os
import platform

# Python 3.8 and over doesn't search PATH directories for DLLs automatically https://bugs.python.org/issue36085
if platform.system() == "Windows":
    for dll_search_dir in os.environ["PATH"].split(";"):
        if os.path.isdir(dll_search_dir):
            os.add_dll_directory(dll_search_dir)

import pbatoolkit.fem
import pbatoolkit.geometry

# Some users may not have CUDA Toolkit libraries installed or discoverable.
# They should still be allowed to use pbatoolkit's CPU APIs.
try:
    import pbatoolkit.gpu
except ImportError:
    import warnings
    warnings.warn("Could not load submodule gpu of module pbatoolkit")
    has_gpu = False
else:
    has_gpu = True

import pbatoolkit.profiling
import pbatoolkit.math
import pbatoolkit.math.linalg
