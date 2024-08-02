from ._pbat import fem as _fem
import sys, inspect

__module = sys.modules[__name__]
for name, attr in inspect.getmembers(_fem):
    setattr(__module, name, attr)