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


def bodies(Vsizes: list, Fsizes: list):
    BV = np.hstack([np.full(Vsizes[i], i) for i in range(len(Vsizes))])
    BF = np.hstack([np.full(Fsizes[i], i) for i in range(len(Fsizes))])
    return BV, BF


def color_dict_to_array(Cdict, n):
    C = np.zeros(n)
    keys = [key for key in Cdict.keys()]
    values = [value for value in Cdict.values()]
    C[keys] = values
    return C


def partition_constraints(C):
    row = np.repeat(range(C.shape[0]), C.shape[1])
    col = C.flatten()
    data = np.ones(math.prod(C.shape))
    G = sp.sparse.coo_array((data, (row, col)), shape=(
        C.shape[0], V.shape[0])).asformat("csr")
    G = nx.Graph(G @ G.T)
    GC = nx.greedy_color(G, strategy="random_sequential")
    GC = color_dict_to_array(GC, C.shape[0]).astype(int)
    npartitions = GC.max() + 1
    partitions = [None]*npartitions
    for p in range(npartitions):
        constraints = np.nonzero(GC == p)[0]
        partitions[p] = constraints.tolist()
    return partitions, GC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="XPBD elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument("-i", "--input", help="Paths to input mesh", nargs="+",
                        dest="inputs", required=True)
    parser.add_argument("-o", "--output", help="Path to output meshes",
                        dest="output", default=".")
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("-t", "--translation", help="Distance in z axis between every input mesh as multiplier of input mesh extents", type=float,
                        dest="translation", default=0.1)
    parser.add_argument("--percent-fixed", help="Percentage, in the z-axis, of scene mesh to fix", type=float,
                        dest="percent_fixed", default=0.01)
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
    V = mesh.X.T
    C = mesh.E.T
    F, Fsizes = boundary_triangles(C, Coffsets, Csizes)
    BV, BF = bodies(Vsizes, Fsizes)

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

    # Compute material (Lame) constants
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    mue = Y / (2*(1+nu))
    lambdae = (Y*nu) / ((1+nu)*(1-2*nu))

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    extent = Xmax - Xmin
    Xmax[-1] = Xmin[-1] + args.percent_fixed*extent[-1]
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    minv = 1 / m
    minv[vdbc] = 0.  # XPBD allows fixing particles by zeroing out their mass

    # Setup XPBD
    Vcollision = np.unique(F)
    VC = Vcollision[:, np.newaxis].T
    BV = BV[Vcollision]
    max_overlaps = 20 * mesh.X.shape[1]
    max_contacts = 8*max_overlaps
    xpbd = pbat.gpu.xpbd.Xpbd(mesh.X, VC, F.T, mesh.E,
                              BV, BF, max_overlaps, max_contacts)
    xpbd.f = f
    xpbd.minv = minv
    xpbd.lame = np.vstack((mue, lambdae))
    partitions, GC = partition_constraints(mesh.E.T)
    xpbd.partitions = partitions
    alphac = 0
    xpbd.set_compliance(
        alphac * np.ones(VC.shape[1]), pbat.gpu.xpbd.ConstraintType.Collision)
    xpbd.prepare()

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("Simulation mesh", mesh.X.T, mesh.E.T)
    vm.add_scalar_quantity("Coloring", GC, defined_on="cells", cmap="jet")
    pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
    dt = 0.01
    iterations = 1
    substeps = 50
    animate = False
    export = False
    t = 0

    profiler = pbat.profiling.Profiler()

    def callback():
        global dt, iterations, substeps, alphac
        global animate, export, t
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, iterations = imgui.InputInt("Iterations", iterations)
        changed, substeps = imgui.InputInt("Substeps", substeps)
        alphac_changed, alphac = imgui.InputFloat(
            "Collision compliance", alphac, format="%.10f")
        changed, animate = imgui.Checkbox("Animate", animate)
        changed, export = imgui.Checkbox("Export", export)
        step = imgui.Button("Step")
        reset = imgui.Button("Reset")

        if reset:
            xpbd.x = mesh.X
            xpbd.v = np.zeros(mesh.X.shape)
            vm.update_vertex_positions(mesh.X.T)
            t = 0

        if alphac_changed:
            xpbd.set_compliance(
                alphac * np.ones(VC.shape[1]), pbat.gpu.xpbd.ConstraintType.Collision)

        if animate or step:
            profiler.begin_frame("Physics")
            xpbd.step(dt, iterations, substeps)
            profiler.end_frame("Physics")

            # Update visuals
            V = xpbd.x.T
            min, max = np.min(V, axis=0), np.max(V, axis=0)
            xpbd.scene_bounding_box = min, max
            if export:
                ps.screenshot(f"{args.output}/{t}.png")
                # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
                # meshio.write(f"{args.output}/{t}.mesh", omesh)

            vm.update_vertex_positions(V)
            t = t+1

        imgui.Text(f"Frame={t}")

    ps.set_user_callback(callback)
    ps.show()
