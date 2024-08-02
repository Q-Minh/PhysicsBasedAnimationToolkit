from ._pbat import gpu as _gpu
import sys, inspect

__module = sys.modules[__name__]
for name, attr in inspect.getmembers(_gpu):
    setattr(__module, name, attr)