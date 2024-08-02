from ._pbat import gpu as _gpu
import sys

__module = sys.modules[__name__]
for name in dir(_gpu):
    setattr(__module, name, getattr(_gpu, name))