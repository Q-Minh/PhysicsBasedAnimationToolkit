import warp as wp

_sync_threads_cpp = """
    __syncthreads();
"""


@wp.func_native(_sync_threads_cpp)
def sync_threads(): ...