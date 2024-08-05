import os
import platform

# Python 3.8 and over doesn't search PATH directories for DLLs automatically https://bugs.python.org/issue36085
if platform.system() == "Windows":
    for dll_search_dir in os.environ["PATH"].split(";"):
        if os.path.isdir(dll_search_dir):
            os.add_dll_directory(dll_search_dir)

import pbatoolkit.fem
import pbatoolkit.geometry
import pbatoolkit.gpu
import pbatoolkit.profiling
import pbatoolkit.math
import pbatoolkit.math.linalg
