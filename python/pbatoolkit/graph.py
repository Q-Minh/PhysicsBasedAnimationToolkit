from ._pbat import graph as _graph
import sys
import inspect
import contextlib
import io
import numpy as np
import math
import scipy as sp

__module = sys.modules[__name__]
_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_graph)
_strio.seek(0)
setattr(__module, "__doc__", _strio.read())

for _name, _attr in inspect.getmembers(_graph):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)


def mesh_adjacency_graph(V: np.ndarray, C: np.ndarray, data: np.ndarray = None):
    """Computes the adjacency list (c,v), where c in C and v in V. Edges (c,v) can 
    have associated property values in data.

    Args:
        V (np.ndarray): |#verts|x|#dims| array of vertices
        C (np.ndarray): |#elements|x|#verts per element| array of elements
        data (np.ndarray, optional): |#elements|x|#verts per element| or 
        |#elements * #verts per element| property values. Defaults to None.

    Raises:
        ValueError: Given data must be associated to edges (c,v).

    Returns:
        G: Graph (c,v) in compressed sparse row format
    """
    row = np.repeat(range(C.shape[0]), C.shape[1])
    col = C.flatten(order="C")
    if data is None:
        data = np.ones_like(C)
    else:
        length = math.prod(data.shape)
        nnz = math.prod(C.shape)
        if length != nnz:
            raise ValueError(f"Expected len(data)={nnz}")
    G = sp.sparse.csr_array((data.flatten(order="C"), (row, col)), shape=(
        C.shape[0], V.shape[0]))
    return G


def mesh_dual_graph(V: np.ndarray, C: np.ndarray):
    """Computes the graph of adjacent elements (ci,cj), where ci in C and cj in C.

    Args:
        V (np.ndarray): |#verts|x|#dims| array of vertices
        C (np.ndarray): |#elements|x|#verts per element| array of elements

    Returns:
        GGT: |#elements|x|#elements| dual graph of adjacent elements as a sparse matrix 
        whose entries (ci,cj) count the number of shared vertices between elements ci and cj.
    """
    G = mesh_adjacency_graph(V, C)
    GGT = G @ G.T
    return GGT


def mesh_primal_graph(V: np.ndarray, C: np.ndarray):
    """Computes the graph of adjacent vertices (vi,vj), where vi in V and vj in V.

    Args:
        V (np.ndarray): |#verts|x|#dims| array of vertices
        C (np.ndarray): |#elements|x|#verts per element| array of elements

    Returns:
        GTG: |#verts|x|#verts| primal graph of adjacent vertices as a sparse matrix 
        whose entries (vi,vj) count the number of shared elements between vertices vi and vj.
    """
    G = mesh_adjacency_graph(V, C)
    GTG = G.T @ G
    return GTG
