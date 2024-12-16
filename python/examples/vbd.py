import pbatoolkit as pbat
import meshio
import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as imgui
import argparse
import itertools


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    Csizes = [Ci.shape[0] for Ci in C]
    Voffsets = list(itertools.accumulate(Vsizes))
    C = [C[i] + Voffsets[i] - Vsizes[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C


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
    parser.add_argument("--use-gpu", help="Run GPU implementation of VBD", action="store_true",
                        dest="gpu", default=False)
    args = parser.parse_args()

    # Construct FEM quantities for simulation
    imeshes = [meshio.read(input) for input in args.inputs]
    V, C = [imesh.points / (imesh.points.max() - imesh.points.min()) for imesh in imeshes], [
        imesh.cells_dict["tetra"] for imesh in imeshes]
    for i in range(len(V) - 1):
        extent = V[i][:, -1].max() - V[i][:, -1].min()
        offset = V[i][:, -1].max() - V[i+1][:, -1].min()
        V[i+1][:, -1] += offset + extent*args.translation
    V, C = combine(V, C)
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)

    # Compute material (Lame) constants
    rhoe = np.full(mesh.E.shape[1], args.rho)
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
    data = pbat.sim.vbd.Data().with_volume_mesh(
        V.T, C.T
    ).with_surface_mesh(
        np.unique(F), F.T
    ).with_material(
        rhoe, mue, lambdae
    ).with_dirichlet_vertices(
        vdbc
    ).with_initialization_strategy(
        pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum
    ).construct(validate=True)
    thread_block_size = 64

    vbd = None
    if args.gpu:
        vbd = pbat.gpu.vbd.Integrator(data)
        vbd.set_gpu_block_size(thread_block_size)
    else:
        vbd = pbat.sim.vbd.Integrator(data)

    # Setup visuals
    initialization_strategies = [
        pbat.sim.vbd.InitializationStrategy.Position,
        pbat.sim.vbd.InitializationStrategy.Inertia,
        pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum,
        pbat.sim.vbd.InitializationStrategy.AdaptiveVbd,
        pbat.sim.vbd.InitializationStrategy.AdaptivePbat
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
    rho_chebyshev = 1.
    RdetH = 1e-10
    kD = 0.
    use_parallel_reduction = False
    animate = False
    export = False
    t = 0

    profiler = pbat.profiling.Profiler()

    def callback():
        global dt, iterations, substeps
        global rho_chebyshev, initialization_strategy, RdetH, kD
        global use_parallel_reduction, thread_block_size
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
        changed, use_parallel_reduction = imgui.Checkbox(
            "2-level parallelism", use_parallel_reduction)
        changed = imgui.BeginCombo(
            "Initialization strategy", str(initialization_strategy).split('.')[-1])
        if changed:
            for i in range(len(initialization_strategies)):
                _, selected = imgui.Selectable(
                    str(initialization_strategies[i]).split('.')[-1], initialization_strategy == initialization_strategies[i])
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
            vbd.set_gpu_block_size(thread_block_size)
            vbd.use_parallel_reduction(use_parallel_reduction)

        if animate or step:
            profiler.begin_frame("Physics")
            vbd.step(dt, iterations, substeps, rho_chebyshev)
            profiler.end_frame("Physics")

            # Update visuals
            V = vbd.x.T
            if export:
                ps.screenshot()
                # omesh = meshio.Mesh(V, {"tetra": mesh.E.T})
                # meshio.write(f"{args.output}/{t}.mesh", omesh)

            vm.update_vertex_positions(V)
            t = t+1

        imgui.Text(f"Frame={t}")
        if args.gpu:
            imgui.Text("Using GPU VBD integrator")
        else:
            imgui.Text("Using CPU VBD integrator")

    ps.set_user_callback(callback)
    ps.show()
