from ._pbat import profiling as _profiling

import time


class Profiler:
    """Interface to profiler used by Physics Based Animation Toolkit"""

    @property
    def is_connected_to_server(self):
        return _profiling.is_connected_to_server()

    def wait_for_server_connection(self, timeout=10, retry=0.1):
        start = time.time()
        while not self.is_connected_to_server and time.time() - start < timeout:
            time.sleep(retry)
        return self.is_connected_to_server

    def begin_frame(self, frame_name: str):
        _profiling.begin_frame(frame_name)

    def end_frame(self, frame_name: str):
        _profiling.end_frame(frame_name)
        
    def profile(self, zone_name: str, func):
        return _profiling.profile(zone_name, func)
