from ..._pbat.graph import mesh_dual_graph
from ..._pbat.graph import map_to_adjacency
from ..._pbat.graph import greedy_color
from ..._pbat.graph import GreedyColorOrderingStrategy
from ..._pbat.graph import GreedyColorSelectionStrategy
import numpy as np
import scipy as sp

def partition_clustered_mesh_constraint_graph(
        X: np.ndarray,
        E: np.ndarray,
        ordering: GreedyColorOrderingStrategy = GreedyColorOrderingStrategy.LargestDegree,
        selection: GreedyColorSelectionStrategy = GreedyColorSelectionStrategy.LeastUsed):
    """Computes the clustered constraint graph's partitioning from 
    Ton-That, Quoc-Minh, Paul G. Kry, and Sheldon Andrews. 
    "Parallel block Neo-Hookean XPBD using graph clustering." 
    Computers & Graphics 110 (2023): 1-10.

    Args:
        X (np.ndarray): |#dims|x|#verts| array of mesh vertex positions
        E (np.ndarray): |#verts-per-element|x|#elements| array of mesh elements
        ordering (_pbat.graph.GreedyColorOrderingStrategy): Vertex coloring ordering strategy. 
        Defaults to GreedyColorOrderingStrategy.LargestDegree.
        selection (_pbat.graph.GreedyColorSelectionStrategy): Vertex coloring selection strategy. 
        Defaults to GreedyColorSelectionStrategy.LeastUsed.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): 
        The tuple (SGptr, SGadj, Cptr, Cadj, clustering, SGC) where (SGptr, SGadj) 
        yields the clustered graph partitions, (Cptr, Cadj) yields the map from 
        clusters to constraints, clustering[c] yields the cluster which constraint c 
        belongs to, and SGC is the coloring of the clusters.
    """
    GGT = mesh_dual_graph(E, X.shape[1])
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
    from ...graph import partition, PartitioningCoarseningStrategy
    n_clusters = int(E.shape[1] / cluster_size)
    clustering = partition(
        GGT.indptr, GGT.indices, weights, n_clusters,
        # Default | RandomMatching | SortedHeavyEdgeMatching
        coarsening=PartitioningCoarseningStrategy.SortedHeavyEdgeMatching,
        seed=0,
        minimize_degree=True,
        contiguous_parts=True,
        identify_conn_comp=True)
    # Construct adjacency graph of the map clusters -> constraints
    Cptr, Cadj = map_to_adjacency(clustering)
    # Compute edges between the clusters, i.e. the supernodal graph's edges
    GGTsizes = GGT.indptr[1:] - GGT.indptr[:-1]
    SGu = clustering[np.repeat(np.arange(clustering.shape[0], dtype=np.int64), GGTsizes)]
    SGv = clustering[GGT.indices]
    inds = np.unique(SGu + clustering.shape[0]*SGv, return_index=True)[1]
    SGu, SGv = SGu[inds], SGv[inds]
    # Construct the supernodal graph
    SGM = sp.sparse.coo_array(
        (np.ones(SGu.shape[0]), (SGu, SGv))).asformat('csr')
    # Color the supernodal graph
    SGC = greedy_color(SGM.indptr, SGM.indices,
                       ordering=ordering, selection=selection)
    # Construct supernode partitions
    SGptr, SGadj = map_to_adjacency(SGC)
    return SGptr, SGadj, Cptr, Cadj, clustering, SGC


def partition_mesh_constraints(
        X: np.ndarray,
        E: np.ndarray,
        ordering: GreedyColorOrderingStrategy = GreedyColorOrderingStrategy.LargestDegree,
        selection: GreedyColorSelectionStrategy = GreedyColorSelectionStrategy.LeastUsed):
    """Computes partition of mesh (X,E)'s constraint graph, 
    assuming constraints are associated with mesh elements, 
    and degrees of freedom are attached to mesh vertices.

    Args:
        X (np.ndarray): |#dims|x|#verts| array of mesh vertex positions
        E (np.ndarray): |#verts-per-element|x|#elements| array of mesh elements

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): 
            (ptr, adj, GC), where (ptr,adj) is the adjacency graph 
            mapping partitions to constraints in compressed sparse format, 
            and GC is the color map on mesh elements.
    """
    GGT = mesh_dual_graph(E, X.shape[1])
    GC = greedy_color(GGT.indptr, GGT.indices,
                      ordering=ordering, selection=selection)
    ptr, adj = map_to_adjacency(GC)
    return ptr, adj, GC
