from ..._pbat.sim import vbd as _vbd
import sys
import inspect
import contextlib
import io

__module = sys.modules[__name__]
_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_vbd)
_strio.seek(0)
setattr(__module, "__doc__", _strio.read())

__module = sys.modules[__name__]
for _name, _attr in inspect.getmembers(_vbd):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)
