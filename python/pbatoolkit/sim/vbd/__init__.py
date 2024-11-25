from ..._pbat.sim import vbd as _vbd
import sys
import inspect
import contextlib
import io
import numpy as np

from ...graph import mesh_adjacency_graph, mesh_primal_graph, colors, lil_to_adjacency

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
    Pptr, Padj = lil_to_adjacency(partitions)
    return Pptr, Padj, GC


def _transfer_quadrature(cmesh, cbvh, wgf, Xgf, order=1):
    from ... import fem, math
    Xgc = cmesh.quadrature_points(order)
    n_elems = cmesh.E.shape[1]
    egc = np.arange(n_elems, dtype=np.int64)
    n_quad_pts = Xgc.shape[1]
    n_quad_pts_per_elem = int(n_quad_pts / n_elems)
    egc = np.repeat(egc, n_quad_pts_per_elem)
    Xic = fem.reference_positions(cmesh, egc, Xgc)
    ecf = np.array(cbvh.nearest_primitives_to_points(
        Xgf, parallelize=True)[0])
    Xif = fem.reference_positions(cmesh, ecf, Xgf)
    wgc, err = math.transfer_quadrature(
        egc, Xic, ecf, Xif, wgf, order=order, with_error=True, max_iters=50, precision=1e-10)
    return wgc, Xgc, egc, err


def _patch_quadrature(cmesh, wgc, Xgc, egc, err=1e-4, numerical_zero=1e-10):
    from ... import fem, math
    import scipy as sp
    import qpsolvers

    wtotal = np.zeros(cmesh.E.shape[1])
    np.add.at(wtotal, egc, wgc)
    ezero = np.argwhere(wtotal < numerical_zero).squeeze()
    ezeroinds = np.isin(egc, ezero)
    egczero = np.copy(egc[ezeroinds])
    Xg1zero = np.copy(Xgc[:, ezeroinds])
    MM, BM, PM = math.reference_moment_fitting_systems(
        egczero, Xg1zero, egczero, Xg1zero, np.zeros(Xg1zero.shape[1]))
    MM = math.block_diagonalize_moment_fitting(MM, PM)
    n = np.count_nonzero(ezeroinds)
    P = -sp.sparse.eye(n).asformat('csc')
    G = MM.asformat('csc')
    h = np.full(G.shape[0], err)
    lb = np.full(n, 0.)
    q = np.full(n, 0.)
    wzero = qpsolvers.solve_qp(P, q, G=G, h=h, lb=lb, initvals=np.zeros(
        n), solver="cvxopt")
    wgcp = np.copy(wgc)
    wgcp[ezeroinds] = wzero
    return wgcp, ezeroinds


def hierarchy(data: _vbd.Data, V: list[np.ndarray], C: list[np.ndarray]):
    from ... import fem, geometry
    rbvh = geometry.bvh(data.x, data.T, cell=geometry.Cell.Tetrahedron)
    bvhs = [geometry.bvh(VC, CC, cell=geometry.Cell.Tetrahedron)
            for VC, CC in zip(V, C)]
    cages = [None]*len(V)
    for c in range(len(V)):
        X, E = V[c].T, C[c].T
        ptr, adj = partitions(V[c], C[c])
        cages[c] = _vbd.level.Cage(X, E, ptr, adj)
    energies = [None]*len(V)
    from ...fem import inner_product_weights

    reference_quadrature_order = 1
    rmesh = fem.Mesh(data.x, data.T, element=fem.Element.Tetrahedron, order=1)
    Xgr = rmesh.quadrature_points(reference_quadrature_order)
    wgr = fem.inner_product_weights(
        rmesh, quadrature_order=reference_quadrature_order
    ).flatten(order="F")
    coarse_quadrature_order = 1
    for c in range(len(V)):
        # Quadrature
        cmesh = fem.Mesh(
            V[c].T, C[c].T, element=fem.Element.Tetrahedron, order=1)
        cbvh = bvhs[c]
        wgc, Xgc, egc, err = _transfer_quadrature(cmesh, cbvh, wgr, Xgr,
                                                  order=coarse_quadrature_order)

        # Adjacency

        # Kinetic energy

        # Potential energy
