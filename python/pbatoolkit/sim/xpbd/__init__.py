from ..._pbat.sim import xpbd as _xpbd
import sys
import inspect
import contextlib
import io
import numpy as np
import itertools
import scipy as sp

from ...graph import mesh_dual_graph, map_to_adjacency, colors

__module = sys.modules[__name__]
_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_xpbd)
_strio.seek(0)
setattr(__module, "__doc__", _strio.read())

__module = sys.modules[__name__]
for _name, _attr in inspect.getmembers(_xpbd):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)


def partition_clustered_mesh_constraint_graph(V, C):
    """Computes the clustered constraint graph's partitioning from 
    Ton-That, Quoc-Minh, Paul G. Kry, and Sheldon Andrews. 
    "Parallel block Neo-Hookean XPBD using graph clustering." 
    Computers & Graphics 110 (2023): 1-10.

    Args:
        C (np.ndarray): Tetrahedral mesh elements

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): 
        The tuple (SGptr, SGadj, Cptr, Cadj, clustering, SGC) where (SGptr, SGadj) 
        yields the clustered graph partitions, (Cptr, Cadj) yields the map from 
        clusters to constraints, clustering[c] yields the cluster which constraint c 
        belongs to, and SGC is the coloring of the clusters.
    """
    GGT = mesh_dual_graph(V, C)
    # NOTE:
    # The sparse matrix representation of GGT counts the number of shared vertices
    # n(ei,ej) for each adjacent element pair (ei, ej). We define a weight function
    # w(ei,ej) = 10^{n(ei,ej)},
    # such that adjacent elements with many shared vertices have large weight.
    # We will try to partition this weighted dual graph such that the edge cut
    # is minimized, i.e. "highly connected" elements (ei,ej) with large w(ei,ej)
    # will be in the same partition (as much as possible).
    weights = 10**np.array(2*(GGT.data - 1), dtype=np.int64)
    # For tetrahedral elements, 5-element partitions is a good choice
    cluster_size = 5
    # Cluster our constraint graph via (edge-)cut-minimizing graph partitioning into
    # a supernodal graph
    from ...graph import partition
    clustering = np.array(partition(
        GGT.indptr, GGT.indices, weights, int(C.shape[0] / cluster_size)))
    # Construct adjacency graph of the map clusters -> constraints
    Cptr, Cadj = map_to_adjacency(clustering)
    # Compute edges between the clusters, i.e. the supernodal graph's edges
    GGTsizes = GGT.indptr[1:] - GGT.indptr[:-1]
    SGu = clustering[np.repeat(np.linspace(
        0, clustering.shape[0]-1, clustering.shape[0], dtype=np.int64), GGTsizes)]
    SGv = clustering[GGT.indices]
    inds = np.unique(SGu + clustering.shape[0]*SGv, return_index=True)[1]
    SGu, SGv = SGu[inds], SGv[inds]
    # Construct the supernodal graph
    SGM = sp.sparse.coo_array(
        (np.ones(SGu.shape[0]), (SGu, SGv))).asformat('csr')
    import networkx as nx
    # Color the supernodal graph
    SGC = colors(SGM, strategy="random_sequential")
    # Construct supernode partitions
    SGptr, SGadj = map_to_adjacency(SGC)
    return SGptr, SGadj, Cptr, Cadj, clustering, SGC


def partition_mesh_constraints(V, C):
    """Computes partition of mesh (V,C)'s constraint graph, 
    assuming constraints are associated with mesh elements, 
    and degrees of freedom are attached to mesh vertices.

    Args:
        V (np.ndarray): |#verts|x|#dims| array of mesh vertex positions
        C (np.ndarray): |#elements|x|#verts-per-element| array of mesh elements

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): 
            (ptr, adj, GC), where (ptr,adj) is the adjacency graph 
            mapping partitions to constraints in compressed sparse format, 
            and GC is the color map on mesh elements.
    """
    GGT = mesh_dual_graph(V, C)
    GC = colors(GGT, strategy="random_sequential")
    ptr, adj = map_to_adjacency(GC)
    return ptr, adj, GC
