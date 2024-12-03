import pbatoolkit as pbat
import meshio
import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as imgui
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VBD elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh",
                        dest="input", required=True)
    parser.add_argument("-c", "--cage", help="Path to cage tetrahedral mesh", type=str,
                        dest="cage", required=True)
    parser.add_argument("-o", "--output", help="Path to output",
                        dest="output", default=".")
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("--percent-fixed", help="Percentage, in the fixed axis, of the scene's mesh to fix", type=float,
                        dest="percent_fixed", default=0.01)
    parser.add_argument("--fixed-axis", help="Axis 0 | 1 | 2 (x=0,y=1,z=2) in which to fix a certain percentage of the scene's mesh", type=int,
                        dest="fixed_axis", default=2)
    parser.add_argument("--fixed-end", help="min | max, whether to fix from the min or the max of the scene mesh's bounding box", type=str, default="min",
                        dest="fixed_end")
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imesh = meshio.read(args.input)
    V, C = imesh.points.astype(
        np.float64), imesh.cells_dict["tetra"].astype(np.int64)
    cmesh = meshio.read(args.cage)
    VC, CC = cmesh.points.astype(
        np.float64), cmesh.cells_dict["tetra"].astype(np.int64)

    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron)
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    cmesh = pbat.fem.Mesh(VC.T, CC.T, element=pbat.fem.Element.Tetrahedron)

    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
    M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho, dims=1, lump=True)
    m = np.array(M.diagonal()).squeeze()

    # Construct load vector from gravity field
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    f, detJeF = pbat.fem.load_vector(mesh, rho*g, detJe=detJeU, flatten=False)
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

    # Setup VBD
    ilocal = np.repeat(np.arange(4)[np.newaxis, :], C.shape[0], axis=0)
    GVT = pbat.sim.vbd.vertex_element_adjacency(V, C, data=ilocal)
    Pptr, Padj, GC = pbat.sim.vbd.partitions(V, C, vdbc)
    data = pbat.sim.vbd.Data().with_volume_mesh(
        V.T, C.T
    ).with_surface_mesh(
        np.unique(F), F.T
    ).with_acceleration(
        a
    ).with_mass(
        m
    ).with_quadrature(
        detJeU[0, :] / 6, GNeU, np.vstack((mue, lambdae))
    ).with_vertex_adjacency(
        GVT.indptr, GVT.indices, GVT.data
    ).with_partitions(
        Pptr, Padj
    ).with_dirichlet_vertices(
        vdbc
    ).with_initialization_strategy(
        pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum
    ).construct(validate=False)
    thread_block_size = 64

    # Setup multiscale VBD
    Transition = pbat.sim.vbd.Transition
    Quadrature = pbat.sim.vbd.Quadrature
    ibvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
    cbvh = pbat.geometry.bvh(
        VC.T, CC.T, cell=pbat.geometry.Cell.Tetrahedron)
    cXg, cwg, ceg, csg, iXg, iwg, err = pbat.fem.fit_output_quad_to_input_quad(
        mesh,
        cmesh,
        ibvh,
        cbvh,
        iorder=1,
        oorder=2,
        selection=pbat.fem.QuadraturePointSelection.FromOutputQuadrature,
        fitting_strategy=pbat.fem.QuadratureFittingStrategy.Ignore,
        singular_strategy=pbat.fem.QuadratureSingularityStrategy.Constant,
        volerr=1e-3
    )
    hierarchy = pbat.sim.vbd.hierarchy(
        data,
        V=[VC], C=[CC],
        QL=[Quadrature(cwg, cXg, ceg, csg)],
        cycle=[Transition(-1, 0, riters=20), Transition(0, -1)],
        schedule=[10, 10, 10],
        rrhog=rho
    )

    # vbd = None
    # if args.gpu:
    #     vbd = pbat.gpu.vbd.Integrator(data)
    #     vbd.set_gpu_block_size(thread_block_size)
    # else:
    #     vbd = pbat.sim.vbd.Integrator(data)
    vbd = pbat.sim.vbd.MultiScaleIntegrator()

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
    substeps = 1
    RdetH = 1e-10
    animate = False
    export = False
    t = 0

    profiler = pbat.profiling.Profiler()

    def callback():
        global dt, substeps
        global RdetH
        global thread_block_size
        global animate, export, t
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, substeps = imgui.InputInt("Substeps", substeps)
        changed, animate = imgui.Checkbox("Animate", animate)
        changed, export = imgui.Checkbox("Export", export)
        step = imgui.Button("Step")
        reset = imgui.Button("Reset")

        if animate or step:
            profiler.begin_frame("Physics")
            vbd.step(dt, substeps, hierarchy)
            profiler.end_frame("Physics")

            # Update visuals
            V = hierarchy.root.x.T
            if export:
                ps.screenshot(f"{args.output}/{t}.png")
                # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
                # meshio.write(f"{args.output}/{t}.mesh", omesh)

            vm.update_vertex_positions(V)
            t = t+1

        imgui.Text(f"Frame={t}")
        imgui.Text("Using CPU Multi-Scale VBD integrator")

    ps.set_user_callback(callback)
    ps.show()
