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


class Transition:
    def __init__(self, li, lj, riters=10):
        self.li = li
        self.lj = lj
        self.riters = riters


def hierarchy(
        data: _vbd.Data,
        V: list[np.ndarray],
        C: list[np.ndarray],
        QL: list[Quadrature],
        cycle: list[Transition],
        schedule: list[int],
        rrhog: np.ndarray | float = 1e3):
    """Constructs a multiscale hierarchy for VBD integration

    Args:
        data (_vbd.Data): The problem description
        V (list[np.ndarray]): List of cage mesh vertices
        C (list[np.ndarray]): List of cage mesh elements
        QL (list[Quadrature]): Quadrature schemes for each cage
        cycle (list[Transition]): The multigrid cycle as a list of transitions between levels 
        (l = -1 is the root level, l >= 0 are the coarser levels in order of increasing coarsening).
        schedule (list[int]): Size len(cycle)+1 list of smoothing iterations at each level in the cycle.
        rrhog (np.ndarray | float, optional): |#elems| mass densities at root level. Defaults to 1e3.

    Raises:
        ValueError: 

    Returns:
        (_pbat.sim.vbd.Hierarchy): The multiscale VBD hierarchy
    """

    from ... import fem, geometry
    rmesh = fem.Mesh(data.x, data.T, element=fem.Element.Tetrahedron)
    cmeshes = [fem.Mesh(VC.T, CC.T, element=fem.Element.Tetrahedron)
               for VC, CC in zip(V, C)]
    VR, CR = data.x.T, data.T.T
    rbvh = geometry.bvh(VR.T, CR.T, cell=geometry.Cell.Tetrahedron)
    cbvhs = [geometry.bvh(VC.T, CC.T, cell=geometry.Cell.Tetrahedron)
             for VC, CC in zip(V, C)]
    if isinstance(rrhog, float):
        rrhog = np.full(data.T.shape[1], rrhog)

    cages = [None]*len(V)
    for l in range(len(V)):
        X, E = V[l].T, C[l].T
        ptr, adj, colors = partitions(V[l], C[l])
        cages[l] = _vbd.level.Cage(X, E, ptr, adj)

    energies = [None]*len(V)
    for l in range(len(V)):
        # Quadrature
        wg, Xg, eg, sg = QL[l].wg, QL[l].Xg, QL[l].eg, QL[l].sg
        erg = rbvh.nearest_primitives_to_points(Xg)[0]
        VC, CC = V[l], C[l]
        CG = CC[eg, :]
        cmesh = cmeshes[l]
        # Kinetic energy
        Xig = fem.reference_positions(cmesh, eg, Xg)
        Ncg = fem.shape_functions_at(cmesh, Xig)
        rhog = rrhog[erg]
        # Potential energy
        nquadpts = Xg.shape[1]
        rmug = data.lame[0, :]
        rlambdag = data.lame[1, :]
        mug = rmug[erg]
        lambdag = rlambdag[erg]
        nshapef = CC.shape[1]
        Nrg = np.zeros((nshapef, nshapef*nquadpts))
        ervg = np.zeros((nshapef, nquadpts), dtype=np.int64)
        cbvh = cbvhs[l]
        for v in range(nshapef):
            rXv = data.x[:, data.T[v, erg]]
            ervg[v, :] = cbvh.nearest_primitives_to_points(rXv)[0]
            cXi = fem.reference_positions(cmesh, ervg[v, :], rXv)
            Nrg[:, v::nshapef] = fem.shape_functions_at(cmesh, cXi)
        # Shape function gradients on root mesh at quad.pts.
        rXi = fem.reference_positions(rmesh, erg, Xg)
        GNfg = fem.shape_function_gradients_at(
            rmesh, erg, rXi)
        # Shape function gradients on coarse mesh at quad.pts.
        GNcg = fem.shape_function_gradients_at(cmesh, eg, Xig)
        # Adjacency
        ilocal = np.repeat(np.arange(CG.shape[1])[
                           np.newaxis, :], CG.shape[0], axis=0)
        GVG = vertex_element_adjacency(VC, CG, data=ilocal)
        e = np.repeat(eg[:, np.newaxis], CG.shape[1])
        GVGp = GVG.indptr
        GVGg = GVG.indices
        GVGilocal = GVG.data
        GVGe = vertex_element_adjacency(VC, CG, data=e).data
        # Store energy
        energies[l] = _vbd.level.Energy(
        ).with_quadrature(
            wg, sg
        ).with_adjacency(
            GVGp, GVGg, GVGe, GVGilocal
        ).with_kinetic_energy(
            rhog, Ncg
        ).with_potential_energy(
            mug, lambdag, ervg, Nrg, GNfg, GNcg
        ).construct()

    buses = [None]*len(V)
    for l in range(len(V)):
        Xgl = QL[l].Xg
        ervg = rbvh.nearest_primitives_to_points(Xgl)[0]
        Xigl = fem.reference_positions(rmesh, ervg, Xgl)
        Nrg = fem.shape_functions_at(rmesh, Xigl)
        buses[l] = _vbd.level.RootParameterBus(ervg, Nrg)

    levels = [_vbd.Level(cage, energy, bus)
              for cage, energy, bus in zip(cages, energies, buses)]

    transitions = [None]*len(cycle)
    smoothers = [None]*len(cycle)
    for t, step in enumerate(cycle):
        li, lj, riters = step.li, step.lj, step.riters
        if li == lj:
            raise ValueError(
                f"Invalid transition from level={li} to level={lj}"
            )
        is_prolongation = li > lj
        if is_prolongation:
            lc, lf = li, lj
            is_lf_root = lf == -1
            Xf = data.x if is_lf_root else V[lc].T
            ecf = cbvhs[lc].primitives_containing_points(Xf)
            Xif = fem.reference_positions(cmeshes[lc], ecf, Xf)
            Ncf = fem.shape_functions_at(cmeshes[lc], Xif)
            transitions[t] = _vbd.Prolongation(
            ).from_level(
                lc
            ).to_level(
                lf
            ).with_coarse_shape_functions(
                ecf, Ncf
            ).construct()
        else:
            lf, lc = li, lj
            is_lf_root = lf == -1
            Xcg = QL[lc].Xg
            fbvh = rbvh if is_lf_root else cbvhs[lf]
            fmesh = rmesh if is_lf_root else cmeshes[lf]
            efg = fbvh.nearest_primitives_to_points(Xcg)[0]
            Xif = fem.reference_positions(fmesh, efg, Xcg)
            Nfg = fem.shape_functions_at(fmesh, Xif)
            transitions[t] = _vbd.Restriction(
            ).from_level(
                lf
            ).to_level(
                lc
            ).with_fine_shape_functions(
                efg, Nfg
            ).iterate(
                riters
            ).construct()

    smoothers = [_vbd.Smoother(siters) for siters in schedule]

    hierarchy = _vbd.Hierarchy(data, levels, transitions, smoothers)
    return hierarchy
