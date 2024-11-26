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


class Quadrature:
    def __init__(self, wg, Xg, eg, sg):
        self.wg = wg
        self.Xg = Xg
        self.eg = eg
        self.sg = sg


def hierarchy(
        data: _vbd.Data,
        V: list[np.ndarray],
        C: list[np.ndarray],
        QL: list[Quadrature],
        rrhog: np.ndarray | float = 1e3):
    from ... import fem, geometry
    rmesh = fem.Mesh(data.x, data.T, element=fem.Element.Tetrahedron)
    rbvh = geometry.bvh(data.x, data.T, cell=geometry.Cell.Tetrahedron)
    bvhs = [geometry.bvh(VC, CC, cell=geometry.Cell.Tetrahedron)
            for VC, CC in zip(V, C)]
    if isinstance(rrhog, float):
        rrhog = np.full(data.T.shape[1], rrhog)
    cages = [None]*len(V)
    for c in range(len(V)):
        X, E = V[c].T, C[c].T
        ptr, adj = partitions(V[c], C[c])
        cages[c] = _vbd.level.Cage(X, E, ptr, adj)
    energies = [None]*len(V)

    for c in range(len(V)):
        # Quadrature
        wg, Xg, eg, sg = QL[c].wg, QL[c].Xg, QL[c].eg, QL[c].sg
        reg = rbvh.nearest_primitives_to(Xg)
        # Adjacency
        VC, CC = V[c], C[c]
        CG = CC[eg, :]
        ilocal = np.repeat(np.arange(CG.shape[1])[
                           np.newaxis, :], CG.shape[0], axis=0)
        GVG = mesh_adjacency_graph(VC, CG, ilocal)
        e = np.repeat(eg[:, np.newaxis], CG.shape[1])
        GVGp = GVG.indptr
        GVGg = GVG.indices
        GVGilocal = GVG.data
        GVGe = mesh_adjacency_graph(VC, CG, e).data
        # Kinetic energy
        cmesh = fem.Mesh(VC.T, CC.T, element=fem.Element.Tetrahedron)
        Xig = fem.reference_positions(cmesh, eg, Xg)
        Ncg = fem.shape_functions_at(cmesh, Xig)
        rhog = rrhog[reg]
        # Potential energy
        nquadpts = Xg.shape[1]
        rmug = data.lame[0, :]
        rlambdag = data.lame[1, :]
        mug = rmug[reg]
        lambdag = rlambdag[reg]
        nshapef = CC.shape[1]
        Nrg = np.zeros(nshapef, nshapef*nquadpts)
        erg = np.zeros((nshapef, nquadpts), dtype=np.int64)
        cbvh = bvhs[c]
        for v in range(nshapef):
            rXv = data.x[:, data.T[v, reg]]
            erg[v, :] = cbvh.primitives_containing_points(rXv)
            cXi = fem.reference_positions(cmesh, erg[v, :], rXv)
            Nrg[:, v::nshapef] = fem.shape_functions_at(cmesh, erg[v, :], cXi)
        # Gradients of linear shape functions on root mesh are constant,
        # so the specific reference points Xi are unimportant
        rXi = np.zeros((rmesh.dims, nquadpts))
        GNfg = fem.shape_function_gradients_at(
            rmesh, reg, rXi)
        GNcg = fem.shape_function_gradients_at(cmesh, eg, Xig)
        # Store energy
        energies[c] = _vbd.level.Energy(
        ).with_quadrature(
            wg, sg
        ).with_adjacency(
            GVGp, GVGg, GVGe, GVGilocal
        ).with_kinetic_energy(
            rhog, Ncg
        ).with_potential_energy(
            mug, lambdag, erg, Nrg, GNfg, GNcg
        ).construct()

    levels = [_vbd.Level(cage, energy)
              for cage, energy in zip(cages, energies)]
