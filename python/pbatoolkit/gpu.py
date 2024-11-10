from ._pbat import gpu as _gpu
import sys, inspect
import contextlib
import io

__module = sys.modules[__name__]
_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_gpu)
_strio.seek(0)
setattr(__module, "__doc__", _strio.read())

for _name, _attr in inspect.getmembers(_gpu):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)