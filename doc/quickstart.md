\page quickstart Quick start

<div class="tabbed">
- <b class="tab-title">C++</b>
  Take a look at the unit tests, found in the library's source (`.cpp` or `.cu`) files.
  ```cpp
  #include <pbat/Pbat.h>
  ```
- <b class="tab-title">Python</b>
  To download and install from PyPI, run in command line
  ```bash
  pip install pbatoolkit
  ```
  or, alternatively
  ```bash
  pip install pbatoolkit-gpu
  ```
  if your environment is properly setup to use our GPU algorithms.
  
  Verify [`pbatoolkit`](https://pypi.org/project/pbatoolkit/)'s contents in a Python shell
  ```python
  import pbatoolkit as pbat
  help(pbat.fem)
  help(pbat.geometry)
  help(pbat.profiling)
  help(pbat.math)
  help(pbat.gpu)
  ```

  A bunch of Python scripts demonstrating usage of [`pbatoolkit`](https://pypi.org/project/pbatoolkit/) can be found in the [examples folder](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/tree/master/python/examples), along with their associated [`requirements.txt`](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/tree/master/python/examples/requirements.txt) for easily downloading necessary dependencies via `pip install -r path/to/requirements.txt`. 
  
  Their command line interface follows the pattern
  ```bash
  python[.exe] path/to/examples/[example].py -i path/to/input/mesh
  ```

  The full interface is always revealed by `-h` or `--help`, i.e. 
  ```bash
  python[.exe] path/to/examples/[example].py -h
  ```

  \note The examples assume the user provides the meshes to [`pbatoolkit`](https://pypi.org/project/pbatoolkit/). Triangle (surface) meshes can easily be obtained via [Thingi10K](https://ten-thousand-models.appspot.com/), [TurboSquid](https://www.turbosquid.com/Search/3D-Models/free) or authored yourself in [Blender](https://www.blender.org/). Tools like [TetWild](https://github.com/Yixin-Hu/TetWild), [fTetWild](https://github.com/wildmeshing/fTetWild) and [TetGen](https://wias-berlin.de/software/index.jsp?id=TetGen&lang=1) can then convert them into tetrahedral (volume) meshes. We provide [helper scripts](python/tools/mesh/) to facilitate mesh processing and their associated [`requirements.txt`](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/blob/master/python/tools/mesh/requirements.txt).

</div>


Example results are showcased in our [Gallery](https://github.com/Q-Minh/PhysicsBasedAnimationToolkit/tree/master?tab=readme-ov-file#gallery).

<div class="section_buttons">

| Previous      |         Next |
|:--------------|-------------:|
| \ref features | \ref gallery |

</div>