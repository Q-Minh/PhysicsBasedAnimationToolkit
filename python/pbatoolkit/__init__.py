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
# NOTE: 
# Scenarios (best first, worst last):
# 1. cuda libraries are loaded on first function call to a CUDA API
# 2. cuda libraries are loaded as soon as the gpu submodule is loaded
# 3. cuda libraries are loaded as soon as the _pbat.* dynamic library is loaded
# More details -> https://stackoverflow.com/questions/50786247/when-is-dynamic-linking-between-a-program-and-a-shared-library-performed
# - Scenario 3. forces pbatoolkit users to have a CUDA GPU and libraries. 
#   In this case, I will need to either distribute 2 packages, pbatoolkit (CPU only) 
#   and pbatoolkit-cuda (with GPU), or use dlopen and LoadLibrary calls.
# - Scenario 2. is handled by the following try/except guard.
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
