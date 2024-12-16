# Profiling Module

This module provides Python bindings for profiling functionality in the Physics-Based Animation Toolkit (PBAT). The `Profiler` class allows for performance measurement and function profiling, with integration into the Tracy profiler.

## Classes and Functions

### **Profiler**
An interface to the profiler used by the Physics-Based Animation Toolkit. This class allows you to profile different parts of your code, measure execution time, and visualize performance using the Tracy profiler.

#### Methods

- **begin_frame(frame_name: str)**  
  Begins a new frame for profiling, associated with the given `frame_name`.
  
- **end_frame(frame_name: str)**  
  Ends the profiling of the current frame, associated with the given `frame_name`.

- **profile(zone_name: str, func)**  
  Profiles the execution of a given function (`func`), associating it with the specified `zone_name` for display in the profiler.
  
  **Args**:
    - `zone_name` (str): The name of the zone in the Tracy profiler to display the function execution.
    - `func`: A callable function that takes no arguments and returns tuples of matrices (NumPy arrays).
  
  **Returns**:  
  The return value of the function `func`.

- **wait_for_server_connection(timeout=10, retry=0.1)**  
  Waits for the profiler server to establish a connection. You can specify the `timeout` and `retry` duration to control the connection attempt.

#### Readonly Properties

- **is_connected_to_server**  
  A boolean property that indicates whether the profiler is connected to the server.