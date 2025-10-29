# type: ignore
from pbatoolkit import pbat, pypbat
import meshio
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import tkinter as tk
from tkinter import filedialog


def serialize_solver_iteration(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    k: int,
    arc: pbat.io.Archive,
    post_solve: bool = False,
):
    """Serialize solver iteration data

    Args:
        fem (pbat.sim.dynamics.FemElastoDynamics): Finite element elasto dynamics
            problem
        k (int): Iteration index
        arc (pbat.io.Archive): Archive to store iteration data
        post_solve (bool, optional): Whether this is a post-solve iteration, in which case
            we also serialize velocity. Defaults to False.
    """
    iter = arc[f"{k:06d}"]
    iter.write_data("x", fem.x)
    # Time integration objective and its gradient
    f = fem.objective()
    iter.write_metadata("f", f)
    g = fem.gradient()
    iter.write_data("g", g)
    gnorm = np.linalg.norm(
        g
    )  # annoyingly, this returns numpy.float32 which nanobind does not cast automatically to C++ float
    iter.write_metadata("gnorm", float(gnorm))
    # Velocities at post-solve
    if post_solve:
        iter.write_data("v", fem.v)


def vbd_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    archive: pbat.io.Archive | None = None,
):
    """Python-side VBD solve with optional serialization per iteration."""
    grp = None
    if archive is not None:
        grp = archive["pbat.sim.algorithm.vbd.Solve"]
    pbat.sim.algorithm.vbd.initialize_solve(fem, params)
    for k in range(params.n_max_iters):
        if grp is not None:
            serialize_solver_iteration(fem, k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, params)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, params.n_max_iters, grp, post_solve=True)


def vbd_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    archive: pbat.io.Archive | None = None,
):
    """Python-side VBD integrate with optional serialization, mirroring C++ Integrate."""
    fem.setup_time_integration_optimization()
    grp = None
    if archive is not None:
        grp = archive["pbat.sim.algorithm.vbd.Integrate"]
        fem.serialize(grp)
    vbd_solve(fem, params, archive=grp)
    fem.step()


def anderson_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    anderson: pbat.sim.algorithm.vbd.AndersonParams,
    archive: pbat.io.Archive | None = None,
):
    grp = (
        archive["pbat.sim.algorithm.vbd.Anderson.Solve"]
        if archive is not None
        else None
    )
    pbat.sim.algorithm.vbd.initialize_solve(fem, params, anderson)
    while anderson.k < params.n_max_iters:
        if grp is not None:
            serialize_solver_iteration(fem, anderson.k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, params, anderson)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, anderson.k, grp, post_solve=True)


def anderson_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    anderson: pbat.sim.algorithm.vbd.AndersonParams,
    archive: pbat.io.Archive | None = None,
):
    fem.setup_time_integration_optimization()
    grp = (
        archive["pbat.sim.algorithm.vbd.Anderson.Integrate"]
        if archive is not None
        else None
    )
    if grp is not None:
        fem.serialize(grp)
    anderson_solve(fem, params, anderson, archive=grp)
    fem.step()


def broyden_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    broyden: pbat.sim.algorithm.vbd.BroydenParams,
    archive: pbat.io.Archive | None = None,
):
    grp = (
        archive["pbat.sim.algorithm.vbd.Broyden.Solve"] if archive is not None else None
    )
    pbat.sim.algorithm.vbd.initialize_solve(fem, params, broyden)
    while broyden.k < params.n_max_iters:
        if grp is not None:
            serialize_solver_iteration(fem, broyden.k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, params, broyden)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, broyden.k, grp, post_solve=True)


def broyden_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    broyden: pbat.sim.algorithm.vbd.BroydenParams,
    archive: pbat.io.Archive | None = None,
):
    fem.setup_time_integration_optimization()
    grp = (
        archive["pbat.sim.algorithm.vbd.Broyden.Integrate"]
        if archive is not None
        else None
    )
    if grp is not None:
        fem.serialize(grp)
    broyden_solve(fem, params, broyden, archive=grp)
    fem.step()


def chebyshev_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    cheb: pbat.sim.algorithm.vbd.ChebyshevParams,
    archive: pbat.io.Archive | None = None,
):
    grp = (
        archive["pbat.sim.algorithm.vbd.Chebyshev.Solve"]
        if archive is not None
        else None
    )
    pbat.sim.algorithm.vbd.initialize_solve(fem, params, cheb)
    while cheb.k < params.n_max_iters:
        if grp is not None:
            serialize_solver_iteration(fem, cheb.k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, params, cheb)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, cheb.k, grp, post_solve=True)


def chebyshev_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    cheb: pbat.sim.algorithm.vbd.ChebyshevParams,
    archive: pbat.io.Archive | None = None,
):
    fem.setup_time_integration_optimization()
    grp = (
        archive["pbat.sim.algorithm.vbd.Chebyshev.Integrate"]
        if archive is not None
        else None
    )
    if grp is not None:
        fem.serialize(grp)
    chebyshev_solve(fem, params, cheb, archive=grp)
    fem.step()


if __name__ == "__main__":
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Vertex Block Descent Simple API")
    ps.init()

    dynamics = pbat.sim.dynamics.FemElastoDynamics()
    Y = 1e6  # Young's modulus
    nu = 0.45  # Poisson's ratio
    rho = 1e3  # Mass density
    aext = np.array([0.0, 0.0, -9.81])  # Acceleration
    b = np.array([0.0, 0.0, 0.0])  # Body forces
    v0 = np.array([0.0, 0.0, 0.0])  # Initial velocity
    d_axis = 0  # Dirichlet axis
    d_percent = 0.01  # Dirichlet percentage
    d_extremity = 0  # Dirichlet extremity
    d_nodes = []  # Dirichlet nodes
    dt = 1e-2  # Time step
    s = 1  # Backward differentiation order
    vbd_params = pbat.sim.algorithm.vbd.Params()
    anderson_params = pbat.sim.algorithm.vbd.AndersonParams()
    broyden_params = pbat.sim.algorithm.vbd.BroydenParams()
    chebyshev_params = pbat.sim.algorithm.vbd.ChebyshevParams()
    solver_names = ["Base", "Anderson", "Broyden", "Chebyshev"]
    i_solver = 0  # Non-linear solver index
    init_strategies = [
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position,
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.FreeTrajectory,
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithExternalLoad,
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithFdLoad,
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithProjectedFdLoad,
    ]
    i_init_strategy = 2  # Initialization strategy index
    n_max_iters = 25  # Maximum iterations
    e_broyden_l2_solvers = [
        pbat.sim.algorithm.vbd.EBroydenLeastSquaresSolver.QR,
        pbat.sim.algorithm.vbd.EBroydenLeastSquaresSolver.COD,
        pbat.sim.algorithm.vbd.EBroydenLeastSquaresSolver.LSCG,
        pbat.sim.algorithm.vbd.EBroydenLeastSquaresSolver.OneStepSteepestDescent,
    ]
    i_broyden_l2_solver = 0
    e_broyden_jacobian_estimates = [
        pbat.sim.algorithm.vbd.EBroydenJacobianEstimate.Identity,
        pbat.sim.algorithm.vbd.EBroydenJacobianEstimate.ScaledIdentity,
        pbat.sim.algorithm.vbd.EBroydenJacobianEstimate.QuasiCauchyRelationDiagonalUpdating,
        pbat.sim.algorithm.vbd.EBroydenJacobianEstimate.UsdDiagonal,
        pbat.sim.algorithm.vbd.EBroydenJacobianEstimate.DiagonalCauchySchwarz,
    ]
    i_broyden_jacobian_estimate = 0
    animate = False
    export = False
    archive: pbat.io.Archive = None  # pbat.io.Archive or None
    archive_path = ""
    archive_flush_period = 100
    t = 0
    vm: ps.VolumeMesh = None
    dpc: ps.PointCloud = None

    def callback():
        global Y, nu, rho, aext, b, v0, d_axis, d_percent, d_extremity, d_nodes
        global dt, s
        global vbd_params, anderson_params, broyden_params, chebyshev_params
        global solver_names, i_solver, i_init_strategy, n_max_iters
        global i_broyden_l2_solver, i_broyden_jacobian_estimate
        global animate, export, t, vm, dpc
        global archive, archive_path, archive_flush_period

        dirty = False
        is_new_mesh = False
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select tetrahedral mesh file",
                    defaultextension=".mesh",
                    filetypes=[
                        ("Tetrahedral mesh files (ASCII)", "*.mesh"),
                        ("Tetrahedral mesh files (binary)", "*.msh"),
                        ("All files", "*.*"),
                    ],
                )
                if file_path:
                    imesh = meshio.read(file_path)
                    V, C = imesh.points, imesh.cells_dict["tetra"]
                    dynamics.construct(V.T, C.T)
                    vm = ps.register_volume_mesh("Mesh", dynamics.X.T, dynamics.E.T)
                    is_new_mesh = True
                root.destroy()

            if imgui.Button("Select Export File", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                save_path = filedialog.asksaveasfilename(
                    title="Create HDF5 archive",
                    defaultextension=".h5",
                    filetypes=[
                        ("HDF5 files", "*.h5;*.hdf5"),
                        ("All files", "*.*"),
                    ],
                )
                if save_path:
                    try:
                        # Overwrite existing or create new file
                        archive = pbat.io.Archive(
                            save_path, pbat.io.AccessMode.Overwrite
                        )
                        archive_path = save_path
                    except Exception as e:
                        archive = None
                        archive_path = ""
                        print(f"Failed to open archive: {e}")
                root.destroy()

            _, archive_flush_period = imgui.InputInt(
                "Archive Flush Period", archive_flush_period
            )
            if archive_path:
                imgui.Text(f"Archive: {archive_path}")
            imgui.TreePop()

        dirty |= is_new_mesh
        if imgui.TreeNode("Material Parameters"):
            Y_updated, Y = imgui.InputFloat("Young's Modulus", Y)
            nu_updated, nu = imgui.InputFloat("Poisson's Ratio", nu)
            rho_updated, rho = imgui.InputFloat("Mass Density", rho)
            dirty |= Y_updated or nu_updated or rho_updated
            imgui.TreePop()

        if imgui.TreeNode("Dynamics"):
            a_updated, aext = imgui.InputFloat3("External acceleration", aext)
            b_updated, b = imgui.InputFloat3("Body forces", b)
            v0_updated, v0 = imgui.InputFloat3("Initial velocity", v0)
            betaR_updated, vbd_params.betaR = imgui.InputFloat(
                "Rayleigh damping", vbd_params.betaR, format="%.8f"
            )
            dirty |= (
                a_updated or b_updated or v0_updated
            )  # no need to trigger dirty on betaR_updated
            imgui.TreePop()

        if imgui.TreeNode("Dirichlet Constraints"):
            d_axis_updated, d_axis = imgui.InputInt("Axis (0=x,1=y,2=z)", d_axis)
            d_percent_updated, d_percent = imgui.InputFloat("Percentage", d_percent)
            d_extremity_updated, d_extremity = imgui.InputInt(
                "Extremity (0=min,1=max)", d_extremity
            )
            dirty |= d_axis_updated or d_percent_updated or d_extremity_updated
            imgui.TreePop()

        if imgui.TreeNode("Time Integration"):
            dt_updated, dt = imgui.InputFloat("Time step", dt, format="%.5f")
            s_updated, s = imgui.InputInt("BDF step", s)
            dirty |= dt_updated or s_updated
            imgui.TreePop()

        if imgui.TreeNode("VBD"):
            _, i_solver = imgui.Combo(
                "Non-linear Solver",
                i_solver,
                solver_names,
            )
            _, i_init_strategy = imgui.Combo(
                "Initialization Strategy",
                i_init_strategy,
                [init_strategy.name for init_strategy in init_strategies],
            )
            vbd_params.strategy = init_strategies[i_init_strategy]
            _, vbd_params.n_max_iters = imgui.InputInt(
                "Maximum Iterations", vbd_params.n_max_iters
            )
            _, vbd_params.detH_zero = imgui.InputFloat(
                "detH0", vbd_params.detH_zero, format="%.10f"
            )
            if imgui.TreeNode("Anderson"):
                _, anderson_params.m = imgui.InputInt("m", anderson_params.m)
                _, anderson_params.beta = imgui.InputFloat("beta", anderson_params.beta)
                _, anderson_params.cod_numerical_zero = imgui.InputFloat(
                    "cod_numerical_zero", anderson_params.cod_numerical_zero
                )
                imgui.TreePop()

            if imgui.TreeNode("Broyden"):
                _, broyden_params.m = imgui.InputInt("m", broyden_params.m)
                _, i_broyden_l2_solver = imgui.Combo(
                    "L2 Solver",
                    i_broyden_l2_solver,
                    [solver.name for solver in e_broyden_l2_solvers],
                )
                _, broyden_params.eps_l2_solve = imgui.InputFloat(
                    "L2 solve epsilon", broyden_params.eps_l2_solve
                )
                _, broyden_params.max_l2_solver_iters = imgui.InputInt(
                    "Max L2 solve iters", broyden_params.max_l2_solver_iters
                )
                _, i_broyden_jacobian_estimate = imgui.Combo(
                    "Jacobian Estimate",
                    i_broyden_jacobian_estimate,
                    [estimate.name for estimate in e_broyden_jacobian_estimates],
                )
                broyden_params.l2_solver = e_broyden_l2_solvers[i_broyden_l2_solver]
                broyden_params.jacobian_estimate = e_broyden_jacobian_estimates[
                    i_broyden_jacobian_estimate
                ]
                _, broyden_params.broyden_beta_F = imgui.InputFloat(
                    "beta_F", broyden_params.broyden_beta_F
                )
                _, broyden_params.broyden_beta_B = imgui.InputFloat(
                    "beta_B", broyden_params.broyden_beta_B
                )
                imgui.TreePop()

            if imgui.TreeNode("Chebyshev"):
                _, chebyshev_params.rho = imgui.InputFloat("rho", chebyshev_params.rho)
                imgui.TreePop()
            imgui.TreePop()

        # Initialize VBD parameters
        if is_new_mesh:
            n_nodes = dynamics.X.shape[1]
            GVGp, GVGe, GVGilocal = (
                pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
                    dynamics.E, n_nodes
                )
            )
            colors = pbat.sim.algorithm.vbd.vertex_colors(dynamics.E, n_nodes)
            vbd_params.with_initialization_strategy(
                init_strategies[i_init_strategy]
            ).with_maximum_iterations(n_max_iters).with_vertex_element_adjacency_graph(
                GVGp, GVGe, GVGilocal
            ).with_vertex_colors(
                colors
            ).construct()

        # Update scenario
        if dirty:
            # Time integration
            dynamics.set_time_integration_scheme(dt, s)
            # Material
            mu, llambda = pypbat.fem.lame_coefficients(Y, nu)
            dynamics.set_elastic_energy(mu, llambda)
            dynamics.set_mass_matrix(rho)
            # Dynamics
            fext = np.asarray(b) + rho * np.asarray(aext)
            dynamics.set_external_load(fext)
            # Dirichlet
            aabb: pbat.geometry.AxisAlignedBoundingBox3 = pypbat.geometry.aabb(
                dynamics.X
            )
            Xmin, Xmax = aabb.min.copy(), aabb.max.copy()
            extent = Xmax - Xmin
            if d_extremity == 0:
                Xmax[d_axis] = Xmin[d_axis] + d_percent * extent[d_axis]
                Xmin[d_axis] -= d_percent * extent[d_axis]
            else:
                Xmin[d_axis] = Xmax[d_axis] - d_percent * extent[d_axis]
                Xmax[d_axis] += d_percent * extent[d_axis]
            aabb.min, aabb.max = Xmin, Xmax
            d_nodes = aabb.contained(dynamics.X)
            d_mask = np.zeros(dynamics.X.shape[1], dtype=bool)
            d_mask[d_nodes] = True
            dynamics.constrain(d_mask)
            dpc = ps.register_point_cloud("Dirichlet Nodes", dynamics.x[:, d_nodes].T)
            # NOTE: If the time integration scheme has changed, the BDF integrator
            # needs to be re-initialized. However, if we haven't asked to "reset" the
            # simulation, then we need to continue simulating from the current state.
            # Resetting the initial conditions, but to the current state, approximately
            # achieves this.
            dynamics.set_initial_conditions(dynamics.x, dynamics.v)

        _, animate = imgui.Checkbox("Animate", animate)
        _, export = imgui.Checkbox("Export", export)
        step = imgui.Button("Step")
        reset = imgui.Button("Reset")
        # Initial value problem
        if reset or is_new_mesh:
            n_nodes = dynamics.X.shape[1]
            x0 = dynamics.X
            xdot0 = np.repeat(np.asarray(v0)[:, np.newaxis], n_nodes, axis=1)
            dynamics.set_initial_conditions(
                x0,
                xdot0,
            )
            t = 0

        # Simulate
        if animate or step:
            # Prepare frame archive group if saving is enabled
            frame_group: pbat.io.Archive = None
            if archive is not None:
                try:
                    # Create/obtain group for this frame index
                    frame_group = archive[f"frames/{t:06d}"]
                except Exception as e:
                    frame_group = None
                    print(f"Archive group error: {e}")
            if i_solver == 0:
                vbd_integrate(dynamics, vbd_params, archive=frame_group)
            elif i_solver == 1:
                anderson_integrate(
                    dynamics, vbd_params, anderson_params, archive=frame_group
                )
            elif i_solver == 2:
                broyden_integrate(
                    dynamics, vbd_params, broyden_params, archive=frame_group
                )
            elif i_solver == 3:
                chebyshev_integrate(
                    dynamics, vbd_params, chebyshev_params, archive=frame_group
                )
            if export:
                ps.screenshot()
            t = t + 1
            if archive is not None and (t % archive_flush_period == 0):
                try:
                    archive.flush()
                except Exception as e:
                    print(f"Archive flush error: {e}")

        # Update visuals
        vis_dirty = animate or step or reset
        if vis_dirty:
            if dpc:
                dpc.update_point_positions(dynamics.x[:, d_nodes].T)
            if vm:
                vm.update_vertex_positions(dynamics.x.T)
        imgui.Text(f"Frame={t}")
        imgui.Text(f"Using {solver_names[i_solver]} solver")

    ps.set_user_callback(callback)
    ps.show()
