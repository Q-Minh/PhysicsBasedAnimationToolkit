import pbatoolkit as pbat
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Multigrid VBD elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    parser.add_argument(
        "-c",
        "--cages",
        help="Path to cage tetrahedral meshes",
        nargs="+",
        dest="cages",
        required=True,
    )
    parser.add_argument("--cycle", help="Multigrid cycle",
                        nargs="+", dest="cycle")
    parser.add_argument(
        "--smoothing-iters",
        help="Smoothing iterations to spend at each level in the cycle",
        nargs="+",
        dest="siters",
    )
    parser.add_argument(
        "-o", "--output", help="Path to output", dest="output", default="."
    )
    parser.add_argument(
        "-m",
        "--mass-density",
        help="Mass density",
        type=float,
        dest="rho",
        default=1000.0,
    )
    parser.add_argument(
        "-Y",
        "--young-modulus",
        help="Young's modulus",
        type=float,
        dest="Y",
        default=1e6,
    )
    parser.add_argument(
        "-n",
        "--poisson-ratio",
        help="Poisson's ratio",
        type=float,
        dest="nu",
        default=0.45,
    )
    parser.add_argument(
        "--percent-fixed",
        help="Percentage, in the fixed axis, of the scene's mesh to fix",
        type=float,
        dest="percent_fixed",
        default=0.01,
    )
    parser.add_argument(
        "--fixed-axis",
        help="Axis 0 | 1 | 2 (x=0,y=1,z=2) in which to fix a certain percentage of the scene's mesh",
        type=int,
        dest="fixed_axis",
        default=2,
    )
    parser.add_argument(
        "--fixed-end",
        help="min | max, whether to fix from the min or the max of the scene mesh's bounding box",
        type=str,
        default="min",
        dest="fixed_end",
    )
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imesh = meshio.read(args.input)
    V, C = imesh.points.astype(
        np.float64), imesh.cells_dict["tetra"].astype(np.int64)
    center = V.mean(axis=0)
    scale = V.max() - V.min()
    V = (V - center) / scale
    icmeshes = [meshio.read(cage) for cage in args.cages]
    VC, CC = [icmesh.points.astype(np.float64) for icmesh in icmeshes], [
        icmesh.cells_dict["tetra"].astype(np.int64) for icmesh in icmeshes
    ]
    for c in range(len(VC)):
        VC[c] = (VC[c] - center) / scale

    mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron)
    cmeshes = [
        pbat.fem.Mesh(VCc.T, CCc.T, element=pbat.fem.Element.Tetrahedron)
        for (VCc, CCc) in zip(VC, CC)
    ]

    # Compute material (Lame) constants
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    mue = Y / (2 * (1 + nu))
    lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
    rhoe = np.full(mesh.E.shape[1], args.rho)

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    extent = Xmax - Xmin
    if args.fixed_end == "min":
        Xmax[args.fixed_axis] = (
            Xmin[args.fixed_axis] + args.percent_fixed *
            extent[args.fixed_axis]
        )
        Xmin[args.fixed_axis] -= args.percent_fixed * extent[args.fixed_axis]
    elif args.fixed_end == "max":
        Xmin[args.fixed_axis] = (
            Xmax[args.fixed_axis] - args.percent_fixed *
            extent[args.fixed_axis]
        )
        Xmax[args.fixed_axis] += args.percent_fixed * extent[args.fixed_axis]
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = np.array(aabb.contained(mesh.X))

    # Setup VBD
    VF, FF = pbat.geometry.simplex_mesh_boundary(C.T, V.shape[0])
    data = (
        pbat.sim.vbd.Data()
        .with_volume_mesh(V.T, C.T)
        .with_surface_mesh(VF, FF)
        .with_material(rhoe, mue, lambdae)
        .with_dirichlet_vertices(vdbc, muD=args.Y)
        .with_initialization_strategy(pbat.sim.vbd.InitializationStrategy.AdaptivePbat)
        .construct(validate=True)
    )

    # Setup multigrid VBD
    cycle = [int(l) for l in args.cycle]
    siters = [int(iters) for iters in args.siters]
    hierarchy = pbat.sim.vbd.multigrid.Hierarchy(
        data, cmeshes, cycle=cycle, siters=siters
    )
    vbd = pbat.sim.vbd.multigrid.Integrator()

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Vertex Block Descent")
    ps.init()
    vm = ps.register_volume_mesh("Simulation mesh", V, C)
    vm.add_scalar_quantity(
        "Coloring", hierarchy.data.colors, defined_on="vertices", cmap="jet"
    )
    pc = ps.register_point_cloud("Dirichlet", V[vdbc, :])
    for l, level in enumerate(hierarchy.levels):
        vm.add_scalar_quantity(
            f"Level {l} active elements",
            level.hyper_reduction.active_elements,
            defined_on="cells",
            cmap="coolwarm",
        )
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
            V = hierarchy.data.x.T
            if export:
                ps.screenshot()
                # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
                # meshio.write(f"{args.output}/{t}.mesh", omesh)

            vm.update_vertex_positions(V)
            t = t + 1

        imgui.Text(f"Frame={t}")
        imgui.Text("Using CPU Multi-Scale VBD integrator")

    ps.set_user_callback(callback)
    ps.show()
