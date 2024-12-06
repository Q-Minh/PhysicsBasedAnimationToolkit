import pbatoolkit as pbat
import meshio
import numpy as np
import scipy as sp
import igl
import polyscope as ps
import polyscope.imgui as imgui
import argparse
import networkx as nx
import itertools


def combine(V: list, C: list):
    NV = [Vi.shape[0] for Vi in V]
    offsets = list(itertools.accumulate(NV))
    C = [C[i] + offsets[i] - NV[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    BV = np.hstack([np.full(NV[i], i) for i in range(len(NV))])
    return V, C, BV


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
    parser.add_argument("--betaSNH", help="Stable Neo-Hookean constraint damping", type=float,
                        dest="betaSNH", default=0.)
    parser.add_argument("--alphaC", help="Vertex collision constraint compliance", type=float,
                        dest="alphaC", default=0.)
    parser.add_argument("--betaC", help="Vertex collision constraint damping", type=float,
                        dest="betaC", default=0.)
    parser.add_argument("--muC", help="Vertex collision penalty", type=float,
                        dest="muC", default=1e1)
    parser.add_argument("-t", "--translation", help="Distance in z axis between every input mesh as multiplier of input mesh extents", type=float,
                        dest="translation", default=0.1)
    parser.add_argument("--fixed-axis", help="Axis of scene to fix", type=int,
                        dest="fixed_axis", default=2)
    parser.add_argument("--percent-fixed", help="Percentage, in the z-axis, of scene mesh to fix", type=float,
                        dest="percent_fixed", default=0.01)
    parser.add_argument("--use-gpu", help="Use GPU implementation", action="store_true",
                        dest="gpu", default=False)
    parser.add_argument("--use-clustering", help="Use Ton-That et al. 2023's clustered constraint graph parallelization strategy", action="store_true",
                        dest="cluster", default=False)
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imeshes = [meshio.read(input) for input in args.inputs]
    V, C = [imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes], [
        imesh.cells_dict["tetra"] for imesh in imeshes]
    for i in range(len(V) - 1):
        extent = V[i][:, -1].max() - V[i][:, -1].min()
        offset = V[i][:, -1].max() - V[i+1][:, -1].min()
        V[i+1][:, -1] += offset + extent*args.translation
    V, C, BV = combine(V, C)
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)

    # Compute mass
    M, detJeM = pbat.fem.mass_matrix(mesh, rho=args.rho, dims=1, lump=True)
    m = np.array(M.diagonal()).squeeze()

    # Compute material (Lame) constants
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    mue = Y / (2*(1+nu))
    lambdae = (Y*nu) / ((1+nu)*(1-2*nu))

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    extent = Xmax - Xmin
    Xmax[args.fixed_axis] = Xmin[args.fixed_axis] + \
        args.percent_fixed*extent[args.fixed_axis]
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    minv = 1 / m

    # Setup XPBD
    VC = np.unique(F)
    dblA = igl.doublearea(V, F)
    muC = np.zeros(V.shape[0])
    for d in range(3):
        muC[F[:, d]] += (1/6)*dblA
    muC = args.muC*muC[VC]
    Pptr, Padj, GC = pbat.sim.xpbd.partition_mesh_constraints(
        mesh.X, mesh.E)
    data = pbat.sim.xpbd.Data(
    ).with_volume_mesh(
        mesh.X, mesh.E
    ).with_surface_mesh(
        VC, F.T
    ).with_bodies(
        BV
    ).with_mass_inverse(
        minv
    ).with_elastic_material(
        np.vstack((mue, lambdae))
    ).with_damping(
        np.full(
            mesh.E.shape[1]*2, args.betaSNH), pbat.sim.xpbd.Constraint.StableNeoHookean
    ).with_compliance(
        np.full(VC.shape[0],
                args.alphaC), pbat.sim.xpbd.Constraint.Collision
    ).with_damping(
        np.full(VC.shape[0],
                args.betaC), pbat.sim.xpbd.Constraint.Collision
    ).with_collision_penalties(
        muC
    ).with_friction_coefficients(
        0.6, 0.4
    ).with_dirichlet_vertices(
        vdbc
    ).with_partitions(
        Pptr, Padj
    )

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("eXtended Position Based Dynamics")
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

    has_partitioning = getattr(pbat.graph, "partition") is not None
    if has_partitioning and args.cluster:
        SGptr, SGadj, Cptr, Cadj, clustering, SGC = pbat.sim.xpbd.partition_clustered_mesh_constraint_graph(
            mesh.X, mesh.E)
        data.with_cluster_partitions(SGptr, SGadj, Cptr, Cadj)
        ecolors = SGC[clustering]
        max_color = GC.max()
        vm.add_scalar_quantity("Clustered Coloring",
                               ecolors, defined_on="cells", cmap="jet", vminmax=(0, max_color))
    data.construct()

    integrator_type = pbat.gpu.xpbd.Integrator if args.gpu else pbat.sim.xpbd.Integrator
    max_overlaps = 20 * mesh.X.shape[1]
    max_contacts = 8 * max_overlaps
    xpbd = integrator_type(
        data,
        max_vertex_tetrahedron_overlaps=max_overlaps,
        max_vertex_triangle_contacts=max_contacts
    )

    profiler = pbat.profiling.Profiler()

    def callback():
        global dt, iterations, substeps, alphac
        global animate, export, t
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, iterations = imgui.InputInt("Iterations", iterations)
        changed, substeps = imgui.InputInt("Substeps", substeps)
        # alphac_changed, alphac = imgui.InputFloat(
        #     "Collision compliance", alphac, format="%.10f")
        changed, animate = imgui.Checkbox("Animate", animate)
        changed, export = imgui.Checkbox("Export", export)
        step = imgui.Button("Step")
        reset = imgui.Button("Reset")

        if reset:
            xpbd.x = mesh.X
            xpbd.v = np.zeros(mesh.X.shape)
            vm.update_vertex_positions(mesh.X.T)
            t = 0

        # if alphac_changed:
        #     xpbd.set_compliance(
        #         alphac * np.ones(VC.shape[1]), pbat.sim.xpbd.ConstraintType.Collision)

        if animate or step:
            profiler.begin_frame("Physics")
            xpbd.step(dt, iterations, substeps)
            profiler.end_frame("Physics")

            # Update visuals
            V = xpbd.x.T
            if hasattr(pbat.gpu.xpbd, "Integrator") and isinstance(xpbd, pbat.gpu.xpbd.Integrator):
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
