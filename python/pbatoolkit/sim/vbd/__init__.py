from ..._pbat.sim import vbd as _vbd
import sys
import inspect
import contextlib
import io
import numpy as np

from ...graph import mesh_adjacency_graph, mesh_primal_graph, colors

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


def vertex_element_adjacency(V: np.ndarray, C: np.ndarray, data: np.ndarray = None):
    """Computes adjacency graph of edges (v,c) where v in V and c in C. (v,c) can have 
    associated propertys in data.

    Args:
        V (np.ndarray): |#verts|x|#dims| array of vertices
        C (np.ndarray): |#elements|x|#verts per element| array of elements
        data (np.ndarray, optional): |#elements|x|#verts per element| or 
        |#elements * #verts per element| property values. Defaults to None.

    Returns:
        GVT: Adjacency graph GVT of edges (v,c) in compressed sparse column matrix format.
    """
    GVT = mesh_adjacency_graph(V, C, data).asformat("csc")
    return GVT


def partitions(V: np.ndarray, C: np.ndarray, dbcs: np.ndarray | list = None):
    """Computes VBD parallel vertex partitions, accounting for 
    Dirichlet boundary conditions dbcs.

    NOTE: Requires networkx to be available!

    Args:
        V (np.ndarray): |#verts|x|#dims| array of vertices
        C (np.ndarray): |#elements|x|#verts per element| array of elements
        dbcs (np.ndarray | list, optional): List of Dirichlet constrained vertices. 
        Defaults to None.

    Returns:
        list: List of lists of independent vertices, disregarding any Dirichlet 
        constrained vertex.
    """
    GVV = mesh_primal_graph(V, C)
    GC = colors(GVV, strategy="random_sequential")
    npartitions = GC.max() + 1
    partitions = []
    for p in range(npartitions):
        vertices = np.nonzero(GC == p)[0]
        # Remove Dirichlet constrained vertices from partitions.
        # In other words, internal forces will not be applied to constrained vertices.
        if dbcs is not None:
            vertices = np.setdiff1d(vertices, dbcs).tolist()
        if len(vertices) > 0:
            partitions.append(vertices)
    return partitions, GC
