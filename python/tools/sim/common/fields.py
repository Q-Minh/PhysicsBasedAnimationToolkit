from enum import Enum

class DocField:
    """Descriptor for a scalar field with a default value and docstring.

    Allows per-field documentation to be retrieved via
    ``getattr(type(obj), name).__doc__``, which is used by ``try_draw_tooltip``
    in ui.py to display imgui tooltips.

    Example::

        class MyParams:
            lr = DocField(1e-3, "Learning rate.")
    """

    def __init__(self, default: float | int | bool | Enum, doc: str = ""):
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


class SerializableMixin:
    """Mixin for Params classes using :class:`DocField` that adds h5py serialize/deserialize.

    Iterates all :class:`DocField` descriptors on the class, storing each value as an
    HDF5 group attribute. Enum fields are stored as their integer value. Bool fields
    are stored as int to ensure portability.
    """

    def serialize(self, grp) -> None:
        """Write all DocField values to HDF5 group attributes."""
        from enum import Enum

        for name in dir(type(self)):
            if isinstance(getattr(type(self), name, None), DocField):
                value = getattr(self, name)
                if isinstance(value, Enum):
                    grp.attrs[name] = value.value
                elif isinstance(value, bool):
                    grp.attrs[name] = int(value)
                else:
                    grp.attrs[name] = value

    def deserialize(self, grp) -> None:
        """Read DocField values from HDF5 group attributes, skipping missing keys."""
        from enum import Enum

        for name in dir(type(self)):
            descriptor = getattr(type(self), name, None)
            if not isinstance(descriptor, DocField):
                continue
            if name not in grp.attrs:
                continue
            default = descriptor.default
            raw = grp.attrs[name]
            try:
                if isinstance(default, Enum):
                    value = type(default)(int(raw))
                elif isinstance(default, bool):
                    value = bool(int(raw))
                elif isinstance(default, int):
                    value = int(raw)
                else:
                    value = float(raw)
                setattr(self, name, value)
            except Exception:
                pass
