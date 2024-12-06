from ._pbat import graph as _graph
import sys
import inspect
import contextlib
import io

__module = sys.modules[__name__]
_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_graph)
_strio.seek(0)
setattr(__module, "__doc__", _strio.read())

for _name, _attr in inspect.getmembers(_graph):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)
