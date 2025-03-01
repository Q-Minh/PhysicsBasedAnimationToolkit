import pbatoolkit as pbat
import meshio
import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as imgui
import argparse
import itertools


def combine(V: list, C: list):
    NV = [Vi.shape[0] for Vi in V]
    offsets = list(itertools.accumulate(NV))
    C = [C[i] + offsets[i] - NV[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    B = np.hstack([np.full(NV[i], i) for i in range(len(NV))])
    return V, C, B


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VBD elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input mesh",
        nargs="+",
        dest="inputs",
        required=True,
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
        "-t",
        "--translation",
        help="Distance in z axis between every input mesh as multiplier of input mesh extents",
        type=float,
        dest="translation",
        default=0.1,
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
    parser.add_argument(
        "--muC", help="Collision penalty", type=float, default=1e6, dest="muC"
    )
    parser.add_argument(
        "--muF", help="Friction coefficient", type=float, default=0.3, dest="muF"
    )
    parser.add_argument(
        "--epsv",
        help="Relative tangential velocity threshold for contact constraints",
        type=float,
        default=1e-2,
        dest="epsv",
    )
    parser.add_argument(
        "--use-gpu",
        help="Run GPU implementation of VBD",
        action="store_true",
        dest="gpu",
        default=False,
    )
    parser.add_argument(
        "--rho-chebyshev",
        help="Chebyshev estimated spectral radius. Chebyshev acceleration is disabled if 0 < rho < 1 is not true",
        type=float,
        default=1.0,
        dest="rho_chebyshev",
    )
    parser.add_argument(
        "--use-trust-region",
        help="Use trust region acceleration",
        action="store_true",
        dest="use_trust_region",
        default=False,
    )
    parser.add_argument(
        "--use-curved-tr",
        help="Use curved trust region path",
        action="store_true",
        dest="use_curved_tr",
        default=False,
    )
    parser.add_argument(
        "--tr-eta",
        help="Trust region energy reduction ratio threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tr-tau",
        help="Trust region radius scaling factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--trace",
        help="Enable trace output",
        action="store_true",
        dest="trace",
        default=False,
    )
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imeshes = [meshio.read(input) for input in args.inputs]
    V, C = [
        imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes
    ], [imesh.cells_dict["tetra"] for imesh in imeshes]
    for i in range(len(V) - 1):
        extent = V[i][:, -1].max() - V[i][:, -1].min()
        offset = V[i][:, -1].max() - V[i + 1][:, -1].min()
        V[i + 1][:, -1] += offset + extent * args.translation
    V, C, B = combine(V, C)
    mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)

    # Compute material (Lame) constants
    rhoe = np.full(mesh.E.shape[1], args.rho)
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    mue = Y / (2 * (1 + nu))
    lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    extent = Xmax - Xmin
    if args.fixed_end == "min":
        Xmax[args.fixed_axis] = (
            Xmin[args.fixed_axis] + args.percent_fixed * extent[args.fixed_axis]
        )
        Xmin[args.fixed_axis] -= args.percent_fixed * extent[args.fixed_axis]
    elif args.fixed_end == "max":
        Xmin[args.fixed_axis] = (
            Xmax[args.fixed_axis] - args.percent_fixed * extent[args.fixed_axis]
        )
        Xmax[args.fixed_axis] += args.percent_fixed * extent[args.fixed_axis]
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)

    # Setup VBD
    data = (
        pbat.sim.vbd.Data()
        .with_volume_mesh(V.T, C.T)
        .with_surface_mesh(np.unique(F), F.T)
        .with_bodies(B)
        .with_material(rhoe, mue, lambdae)
        .with_dirichlet_vertices(vdbc)
        .with_initialization_strategy(
            pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum
        )
        .with_contact_parameters(args.muC, args.muF, args.epsv)
    )
    if args.rho_chebyshev < 1.0 and args.rho_chebyshev > 0.0:
        data = data.with_chebyshev_acceleration(args.rho_chebyshev)
    if args.use_trust_region:
        data = data.with_trust_region_acceleration(
            args.tr_eta, args.tr_tau, args.use_curved_tr
        )
    data = data.construct(validate=True)
    thread_block_size = 64

    vbd = None
    if args.gpu:
        vbd = pbat.gpu.vbd.Integrator(data)
        vbd.gpu_block_size = thread_block_size
    else:
        vbd = pbat.sim.vbd.Integrator(data)

    # Setup visuals
    initialization_strategies = [
        pbat.sim.vbd.InitializationStrategy.Position,
        pbat.sim.vbd.InitializationStrategy.Inertia,
        pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum,
        pbat.sim.vbd.InitializationStrategy.AdaptiveVbd,
        pbat.sim.vbd.InitializationStrategy.AdaptivePbat,
    ]
    initialization_strategy = initialization_strategies[2]
    vbd.strategy = initialization_strategy

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Vertex Block Descent")
    ps.init()
    vm = ps.register_volume_mesh("Simulation mesh", V, C)
    vm.add_scalar_quantity("Coloring", data.colors, defined_on="vertices", cmap="jet")
    pc = ps.register_point_cloud("Dirichlet", V[vdbc, :])
    dt = 0.01
    iterations = 20
    substeps = 1
    rho_chebyshev = 1.0
    RdetH = 1e-10
    kD = 0.0
    animate = False
    export = False
    t = 0

    profiler = pbat.profiling.Profiler()

    def callback():
        global dt, iterations, substeps
        global rho_chebyshev, initialization_strategy, RdetH, kD
        global thread_block_size
        global animate, export, t
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, iterations = imgui.InputInt("Iterations", iterations)
        changed, substeps = imgui.InputInt("Substeps", substeps)
        changed, kD = imgui.InputFloat("Damping", kD, format="%.8f")
        changed, RdetH = imgui.InputFloat("Residual det(H)", RdetH, format="%.15f")
        changed, thread_block_size = imgui.InputInt(
            "Thread block size", thread_block_size
        )
        changed = imgui.BeginCombo(
            "Initialization strategy", str(initialization_strategy).split(".")[-1]
        )
        if changed:
            for i in range(len(initialization_strategies)):
                _, selected = imgui.Selectable(
                    str(initialization_strategies[i]).split(".")[-1],
                    initialization_strategy == initialization_strategies[i],
                )
                if selected:
                    initialization_strategy = initialization_strategies[i]
            imgui.EndCombo()
        vbd.strategy = initialization_strategy
        vbd.kD = kD
        vbd.detH_residual = RdetH
        changed, animate = imgui.Checkbox("Animate", animate)
        changed, export = imgui.Checkbox("Export", export)
        step = imgui.Button("Step")
        reset = imgui.Button("Reset")

        if reset:
            vbd.x = mesh.X
            vbd.v = np.zeros(mesh.X.shape)
            vm.update_vertex_positions(mesh.X.T)
            t = 0

        if args.gpu:
            vbd.gpu_block_size = thread_block_size

        if animate or step:
            profiler.begin_frame("Physics")
            if args.trace:
                vbd.traced_step(dt, iterations, substeps, t, dir=args.output)
            else:
                vbd.step(dt, iterations, substeps)
            profiler.end_frame("Physics")

            # Update visuals
            V = vbd.x.T
            if hasattr(pbat.gpu.vbd, "Integrator") and isinstance(
                vbd, pbat.gpu.vbd.Integrator
            ):
                min, max = np.min(V, axis=0), np.max(V, axis=0)
                vbd.scene_bounding_box = min, max
            if export:
                ps.screenshot()
                # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
                # meshio.write(f"{args.output}/{t}.mesh", omesh)

            vm.update_vertex_positions(V)
            t = t + 1

        imgui.Text(f"Frame={t}")
        if args.gpu:
            imgui.Text("Using GPU VBD integrator")
        else:
            imgui.Text("Using CPU VBD integrator")

    ps.set_user_callback(callback)
    ps.show()
