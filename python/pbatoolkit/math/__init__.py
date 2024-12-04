from .._pbat import math as _math
import sys
import inspect


__module = sys.modules[__name__]
for _name, _attr in inspect.getmembers(_math):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)