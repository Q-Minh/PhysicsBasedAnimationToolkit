import warp as wp

class Stream:

    def __init__(self, stream: wp.Stream):
        self._stream = stream

    def __cuda_stream__(self):
        return (0, self._stream.cuda_stream)