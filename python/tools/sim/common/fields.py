class DocField:
    """Descriptor for a scalar field with a default value and docstring.

    Allows per-field documentation to be retrieved via
    ``getattr(type(obj), name).__doc__``, which is used by ``try_draw_tooltip``
    in ui.py to display imgui tooltips.

    Example::

        class MyParams:
            lr = DocField(1e-3, "Learning rate.")
    """

    def __init__(self, default: float, doc: str = ""):
        self.default = default
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        self._attr = f"_field_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._attr, self.default)

    def __set__(self, obj, value):
        setattr(obj, self._attr, value)
