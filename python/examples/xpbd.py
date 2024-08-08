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
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imesh = meshio.read(args.inputs[0])  # TODO: Combine all input meshes into 1
    V, C = imesh.points, imesh.cells_dict["tetra"]
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    v = np.zeros(x.shape[0])
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
    Xmax[0] = Xmin[0]+1e-4
    Xmin[0] = Xmin[0]-1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    # m[vdbc] = 0.  # XPBD allows fixing particles by zeroing out their mass

    # Setup XPBD
    F = igl.boundary_facets(mesh.E.T)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    xpbd = pbat.gpu.xpbd.Xpbd(mesh.X, F.T, mesh.E)
    xpbd.f = f
    xpbd.m = m
    xpbd.lame = np.vstack((mue, lambdae))
    partitions, GC = partition_constraints(mesh.E.T)
    xpbd.partitions = partitions
    xpbd.prepare()

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("world model", xpbd.x.T, mesh.E.T)
    vm.add_scalar_quantity("Coloring", GC, defined_on="cells", cmap="jet")
    pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
    dt = 0.01
    iterations = 5
    substeps = 10
    animate = False

    profiler = pbat.profiling.Profiler()

    def callback():
        global animate, dt, iterations, substeps
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, iterations = imgui.InputInt("iterations", iterations)
        changed, substeps = imgui.InputInt("substeps", substeps)
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            profiler.begin_frame("Physics")
            xpbd.step(dt, iterations, substeps)
            profiler.end_frame("Physics")

            # Update visuals
            vm.update_vertex_positions(xpbd.x.T)

    ps.set_user_callback(callback)
    ps.show()
