import os
import platform

# Python 3.8 and over doesn't search PATH directories for DLLs automatically https://bugs.python.org/issue36085
if platform.system() == "Windows":
    for dll_search_dir in os.environ["PATH"].split(";"):
        if os.path.isdir(dll_search_dir):
            os.add_dll_directory(dll_search_dir)

import pbatoolkit.fem
import pbatoolkit.geometry
import pbatoolkit.sim
import pbatoolkit.profiling
import pbatoolkit.math
import pbatoolkit.math.linalg

# Some users may not have CUDA Toolkit libraries installed or discoverable.
# They should still be allowed to use pbatoolkit's CPU APIs.
# NOTE:
# Scenarios (best first, worst last):
# 1. cuda libraries are loaded on first function call to a CUDA API
# 2. cuda libraries are loaded as soon as the gpu submodule is loaded
# 3. cuda libraries are loaded as soon as the _pbat.* dynamic library is loaded
# More details -> https://stackoverflow.com/questions/50786247/when-is-dynamic-linking-between-a-program-and-a-shared-library-performed
# - Scenario 1 is the best case, we don't need to do anything special!
# - Scenario 2 is handled by wrapping the gpu submodule import by a try/except guard.
# - Scenario 3 forces pbatoolkit users to have a CUDA GPU and libraries.
# Because lazy loading of dynamic libraries is hard to achieve on linux,
# I distribute 2 packages, pbatoolkit (CPU only) and pbatoolkit-gpu (CPU and GPU).
import pbatoolkit.gpu
