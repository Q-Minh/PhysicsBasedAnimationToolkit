import pbatoolkit as pbat
import meshio
import numpy as np
import scipy as sp
import igl
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse
import networkx as nx
import itertools


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    Csizes = [Ci.shape[0] for Ci in C]
    Voffsets = list(itertools.accumulate(Vsizes))
    Coffsets = list(itertools.accumulate(Csizes))
    C = [C[i] + Voffsets[i] - Vsizes[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, Vsizes, C, Coffsets, Csizes


def boundary_triangles(C: np.ndarray, Coffsets: list, Csizes: list):
    F = [None]*len(Csizes)
    for i in range(len(F)):
        begin = Coffsets[i] - Csizes[i]
        end = begin + Csizes[i]
        F[i] = igl.boundary_facets(C[begin:end, :])
        F[i][:, :2] = np.roll(F[i][:, :2], shift=1, axis=1)
    Fsizes = [Fi.shape[0] for Fi in F]
    F = np.vstack(F)
    return F, Fsizes


def vertex_tetrahedron_adjacency_graph(V, C):
    row = np.repeat(range(C.shape[0]), C.shape[1])
    col = C.flatten()
    data = np.zeros_like(C)
    for i in range(C.shape[1]):
        data[:, i] = i
    data = data.flatten()
    GVT = sp.sparse.coo_array((data, (row, col)), shape=(
        C.shape[0], V.shape[0])).asformat("csc")
    return GVT


def color_dict_to_array(Cdict, n):
    C = np.zeros(n)
    keys = [key for key in Cdict.keys()]
    values = [value for value in Cdict.values()]
    C[keys] = values
    return C


def partition_vertices(GVT, dbcs):
    GVV = GVT.T @ GVT
    Gprimal = nx.Graph(GVV)
    GC = nx.greedy_color(Gprimal, strategy="random_sequential")
    GC = color_dict_to_array(GC, GVT.shape[1]).astype(np.int32)
    npartitions = GC.max() + 1
    partitions = []
    for p in range(npartitions):
        vertices = np.nonzero(GC == p)[0]
        # Remove Dirichlet constrained vertices from partitions.
        # In other words, internal forces will not be applied to constrained vertices.
        vertices = np.setdiff1d(vertices, dbcs).tolist()
        if len(vertices) > 0:
            partitions.append(vertices)
    return partitions, GC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VBD elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", nargs="+",
                        dest="inputs", required=True)
    parser.add_argument("-o", "--output", help="Path to output",
                        dest="output", default=".")
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("-t", "--translation", help="Distance in z axis between every input mesh as multiplier of input mesh extents", type=float,
                        dest="translation", default=0.1)
    parser.add_argument("--percent-fixed", help="Percentage, in the fixed axis, of the scene's mesh to fix", type=float,
                        dest="percent_fixed", default=0.01)
    parser.add_argument("--fixed-axis", help="Axis 0 | 1 | 2 (x=0,y=1,z=2) in which to fix a certain percentage of the scene's mesh", type=int,
                        dest="fixed_axis", default=2)
    parser.add_argument("--fixed-end", help="min | max, whether to fix from the min or the max of the scene mesh's bounding box", type=str, default="min",
                        dest="fixed_end")
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imeshes = [meshio.read(input) for input in args.inputs]
    V, C = [imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes], [
        imesh.cells_dict["tetra"] for imesh in imeshes]
    for i in range(len(V) - 1):
        extent = V[i][:, -1].max() - V[i][:, -1].min()
        offset = V[i][:, -1].max() - V[i+1][:, -1].min()
        V[i+1][:, -1] += offset + extent*args.translation
    V, Vsizes, C, Coffsets, Csizes = combine(V, C)
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    F, Fsizes = boundary_triangles(C, Coffsets, Csizes)

    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
    M = pbat.fem.MassMatrix(mesh, detJeM, rho=rho,
                            dims=1, quadrature_order=2).to_matrix()
    m = np.array(M.sum(axis=0)).squeeze()

    # Construct load vector from gravity field
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    qgf = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.sparse.diags_array([qgf], offsets=[0])
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    fe = np.tile(rho*g[:, np.newaxis], mesh.E.shape[1])
    f = fe @ Qf @ Nf
    a = f / m

    # Compute material (Lame) constants
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    mue = Y / (2*(1+nu))
    lambdae = (Y*nu) / ((1+nu)*(1-2*nu))

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    extent = Xmax - Xmin
    if args.fixed_end == "min":
        Xmax[args.fixed_axis] = Xmin[args.fixed_axis] + \
            args.percent_fixed*extent[args.fixed_axis]
        Xmin[args.fixed_axis] -= args.percent_fixed*extent[args.fixed_axis]
    elif args.fixed_end == "max":
        Xmin[args.fixed_axis] = Xmax[args.fixed_axis] - \
            args.percent_fixed*extent[args.fixed_axis]
        Xmax[args.fixed_axis] += args.percent_fixed * extent[args.fixed_axis]
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    a[:, vdbc] = 0.  # Allow no acceleration, i.e. no external forces in fixed vertices

    # Setup VBD
    Vcollision = np.unique(F)
    VC = Vcollision[:, np.newaxis].T
    vbd = pbat.gpu.vbd.Vbd(V.T, VC, F.T, C.T)
    vbd.a = a
    vbd.m = m
    vbd.wg = detJeU / 6
    vbd.GNe = GNeU
    vbd.lame = np.vstack((mue, lambdae))
    GVT = vertex_tetrahedron_adjacency_graph(V, C)
    vbd.GVT = GVT.indptr, GVT.indices, GVT.data
    vbd.kD = 0
    GVTtopology = GVT.copy()
    GVTtopology.data[:] = 1  # Set all edge weights to 1
    partitions, GC = partition_vertices(GVTtopology, vdbc)
    vbd.partitions = partitions
    thread_block_size = 64
    vbd.set_gpu_block_size(thread_block_size)
    initialization_strategies = [
        pbat.gpu.vbd.InitializationStrategy.Position,
        pbat.gpu.vbd.InitializationStrategy.Inertia,
        pbat.gpu.vbd.InitializationStrategy.KineticEnergyMinimum,
        pbat.gpu.vbd.InitializationStrategy.AdaptiveVbd,
        pbat.gpu.vbd.InitializationStrategy.AdaptivePbat
    ]
    initialization_strategy = initialization_strategies[2]
    vbd.initialization_strategy = initialization_strategy

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Vertex Block Descent")
    ps.init()
    vm = ps.register_volume_mesh("Simulation mesh", V, C)
    vm.add_scalar_quantity("Coloring", GC, defined_on="vertices", cmap="jet")
    pc = ps.register_point_cloud("Dirichlet", V[vdbc, :])
    dt = 0.01
    iterations = 20
    substeps = 1
    rho_chebyshev = 1.
    RdetH = 1e-10
    kD = 0.
    animate = False
    export = False
    t = 0

    profiler = pbat.profiling.Profiler()

    def callback():
        global dt, iterations, substeps, rho_chebyshev, thread_block_size, initialization_strategy, RdetH, kD
        global animate, export, t
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, iterations = imgui.InputInt("Iterations", iterations)
        changed, substeps = imgui.InputInt("Substeps", substeps)
        changed, rho_chebyshev = imgui.InputFloat(
            "Chebyshev rho", rho_chebyshev)
        changed, kD = imgui.InputFloat(
            "Damping", kD, format="%.8f")
        changed, RdetH = imgui.InputFloat(
            "Residual det(H)", RdetH, format="%.15f")
        changed, thread_block_size = imgui.InputInt(
            "Thread block size", thread_block_size)
        changed = imgui.BeginCombo(
            "Initialization strategy", str(initialization_strategy).split('.')[-1])
        if changed:
            for i in range(len(initialization_strategies)):
                _, selected = imgui.Selectable(
                    str(initialization_strategies[i]).split('.')[-1], initialization_strategy == initialization_strategies[i])
                if selected:
                    initialization_strategy = initialization_strategies[i]
            imgui.EndCombo()
        vbd.initialization_strategy = initialization_strategy
        vbd.kD = kD
        vbd.RdetH = RdetH
        changed, animate = imgui.Checkbox("Animate", animate)
        changed, export = imgui.Checkbox("Export", export)
        step = imgui.Button("Step")
        reset = imgui.Button("Reset")

        if reset:
            vbd.x = mesh.X
            vbd.v = np.zeros(mesh.X.shape)
            vm.update_vertex_positions(mesh.X.T)
            t = 0

        vbd.set_gpu_block_size(thread_block_size)

        if animate or step:
            profiler.begin_frame("Physics")
            vbd.step(dt, iterations, substeps, rho_chebyshev)
            profiler.end_frame("Physics")

            # Update visuals
            V = vbd.x.T
            if export:
                ps.screenshot(f"{args.output}/{t}.png")
                # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
                # meshio.write(f"{args.output}/{t}.mesh", omesh)

            vm.update_vertex_positions(V)
            t = t+1

        imgui.Text(f"Frame={t}")

    ps.set_user_callback(callback)
    ps.show()
