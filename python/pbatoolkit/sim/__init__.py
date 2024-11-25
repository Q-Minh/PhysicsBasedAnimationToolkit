from .._pbat import sim as _sim
import sys
import inspect
import contextlib
import io

__module = sys.modules[__name__]
_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_sim)
_strio.seek(0)
setattr(__module, "__doc__", _strio.read())

__module = sys.modules[__name__]
for _name, _attr in inspect.getmembers(_sim):
    should_set = not _name.startswith("__") and _name.find(
        "vbd") == -1 and _name.find("xpbd") == -1
    if should_set:
        setattr(__module, _name, _attr)
