from ._pbat import graph as _graph
import sys
import inspect
import contextlib
import io
import numpy as np
import math
import scipy as sp
import itertools
import networkx as nx

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


def _color_dict_to_array(Cdict):
    keys = np.array(list(Cdict.keys()), dtype=np.int64)
    values = np.array(list(Cdict.values()), dtype=np.int64)
    n = keys.max() + 1
    C = np.zeros(n, dtype=np.int64)
    C[keys] = values
    return C


def colors(G: sp.sparse.csc_array | sp.sparse.csr_array | sp.sparse.csc_matrix | sp.sparse.csr_matrix, strategy: str = "random_sequential"):
    """Computes a greedy coloring on graph G.

    Args:
        G (sp.sparse.csc_array | sp.sparse.csr_array | sp.sparse.csc_matrix | sp.sparse.csr_matrix): Input graph in sparse matrix format.
        strategy (str, optional): One of 'largest_first' | 'random_sequential' | 'smallest_last' | 'independent_set' | 
        'connected_sequential_bfs' | 'connected_sequential_dfs' | 'saturation_largest_first'. 
        Defaults to "random_sequential".

    Returns:
        np.ndarray: Returns the color map C s.t. C[i] yields the color of node i in G.
    """
    G = nx.Graph(G)
    C = nx.greedy_color(G, strategy=strategy)
    C = _color_dict_to_array(C)
    return C


def lil_to_adjacency(partitions: list[list[int]], dtype=np.int64):
    """Converts list of lists (i.e. partitions, clusters, elements, etc.) to an adjacency graph.

    Args:
        partitions (list[list[int]]): Groups of objects.
        dtype: Object type. Defaults to np.int64.

    Returns:
        (np.ndarray, np.ndarray): Adjacency graph (ptr, adj) in compressed sparse storage format
    """
    psizes = np.zeros(len(partitions) + 1, dtype=dtype)
    psizes[1:] = [len(partition) for partition in partitions]
    ptr = np.array(list(itertools.accumulate(psizes)))
    adj = np.array(list(itertools.chain.from_iterable(partitions)))
    return ptr, adj


def map_to_adjacency(map: np.ndarray | list[int], dtype=np.int64, n=None):
    """Converts flat array (i.e. some partition map) to an adjacency graph.

    Args:
        map (np.ndarray | list[int]): Flat array mapping objects to partitions/clusters/groups/elements, 
        i.e. p = map[i] for object i and partition p, if the edge (p,i) exists in the graph.
        dtype: Object type. Defaults to np.int64.

    Returns:
        (np.ndarray, np.ndarray): Adjacency graph (ptr, adj) in compressed sparse storage format
    """
    if n is None:
        n = map.max() + 1
    psizes = np.zeros(n + 1, dtype=dtype)
    np.add.at(psizes[1:], map, 1)
    ptr = np.array(list(itertools.accumulate(psizes)))
    adj = np.argsort(map)
    return ptr, adj
