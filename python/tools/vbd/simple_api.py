# type: ignore
from pbatoolkit import pbat, pypbat
import meshio
import polyscope as ps
from polyscope import imgui
import numpy as np
import tkinter as tk
from tkinter import filedialog
import h5py
import scipy as sp

import utils.transform_library as tlib
from utils.transform_library import TransformType


def _read_mesh_and_state(integrate_grp: h5py.Group):
    # Fem group and datasets
    fem_path = "pbat.sim.dynamics.FemElastoDynamics"
    mesh_path = f"{fem_path}/pbat.fem.Mesh"
    if fem_path not in integrate_grp:
        return None, None, None
    fem_grp = integrate_grp[fem_path]
    # Read mesh connectivity and positions
    X = None
    E = None
    if mesh_path in integrate_grp:
        mesh_grp = integrate_grp[mesh_path]
    else:
        # Fallback: nested under fem_grp
        mesh_grp = fem_grp.get("pbat.fem.Mesh", None)
    if mesh_grp is not None:
        if "X" in mesh_grp and "E" in mesh_grp:
            X = np.array(mesh_grp["X"])  # shape (3, n)
            E = np.array(mesh_grp["E"])  # shape (4, m)
    dmask = (
        np.array(fem_grp["dmask"], dtype=np.int32)
        if "dmask" in fem_grp
        else np.zeros(X.shape[1], dtype=bool)
    )
    lamegU = (
        np.array(fem_grp["lamegU"], dtype=np.float64)
        if "lamegU" in fem_grp
        else np.zeros(E.shape[1], dtype=np.float64)
    )
    # Deformed positions x at this frame
    x = np.array(fem_grp["x"]) if "x" in fem_grp else None
    # print("DA VALS:", X, E, x, dmask)
    return X, E, x, dmask, lamegU


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
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    archive: pbat.io.Archive | None = None,
):
    """Python-side VBD solve with optional serialization per iteration."""
    grp = None
    if archive is not None:
        grp = archive["pbat.sim.algorithm.vbd.Solve"]
    pbat.sim.algorithm.vbd.initialize_solve(fem, mesh_dynamics, params)
    for k in range(params.n_max_iters):
        if grp is not None:
            serialize_solver_iteration(fem, k, grp)
        if (k + 1) % 5 == 0:
            mesh_dynamics.update_environment_contact_constraints(fem.x)
        pbat.sim.algorithm.vbd.iterate(fem, mesh_dynamics, params)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, params.n_max_iters, grp, post_solve=True)


def vbd_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    archive: pbat.io.Archive | None = None,
):
    """Python-side VBD integrate with optional serialization, mirroring C++ Integrate."""
    fem.setup_time_integration_optimization()
    grp = None
    if archive is not None:
        grp = archive["pbat.sim.algorithm.vbd.Integrate"]
        fem.serialize(grp)
    vbd_solve(fem, mesh_dynamics, params, archive=grp)
    fem.step()


def anderson_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    anderson: pbat.sim.algorithm.vbd.AndersonParams,
    archive: pbat.io.Archive | None = None,
):
    grp = (
        archive["pbat.sim.algorithm.vbd.Anderson.Solve"]
        if archive is not None
        else None
    )
    pbat.sim.algorithm.vbd.initialize_solve(fem, mesh_dynamics, params, anderson)
    while anderson.k < params.n_max_iters:
        if grp is not None:
            serialize_solver_iteration(fem, anderson.k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, mesh_dynamics, params, anderson)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, anderson.k, grp, post_solve=True)


def anderson_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
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
    anderson_solve(fem, mesh_dynamics, params, anderson, archive=grp)
    fem.step()


def broyden_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    broyden: pbat.sim.algorithm.vbd.BroydenParams,
    archive: pbat.io.Archive | None = None,
):
    grp = (
        archive["pbat.sim.algorithm.vbd.Broyden.Solve"] if archive is not None else None
    )
    pbat.sim.algorithm.vbd.initialize_solve(fem, mesh_dynamics, params, broyden)
    while broyden.k < params.n_max_iters:
        if grp is not None:
            serialize_solver_iteration(fem, broyden.k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, mesh_dynamics, params, broyden)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, broyden.k, grp, post_solve=True)


def broyden_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
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
    broyden_solve(fem, mesh_dynamics, params, broyden, archive=grp)
    fem.step()


def chebyshev_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
    params: pbat.sim.algorithm.vbd.Params,
    cheb: pbat.sim.algorithm.vbd.ChebyshevParams,
    archive: pbat.io.Archive | None = None,
):
    grp = (
        archive["pbat.sim.algorithm.vbd.Chebyshev.Solve"]
        if archive is not None
        else None
    )
    pbat.sim.algorithm.vbd.initialize_solve(fem, mesh_dynamics, params, cheb)
    while cheb.k < params.n_max_iters:
        if grp is not None:
            serialize_solver_iteration(fem, cheb.k, grp)
        pbat.sim.algorithm.vbd.iterate(fem, mesh_dynamics, params, cheb)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_solver_iteration(fem, cheb.k, grp, post_solve=True)


def chebyshev_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
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
    chebyshev_solve(fem, mesh_dynamics, params, cheb, archive=grp)
    fem.step()


def serialize_newton_solver_iteration(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    k: int,
    newton: pbat.math.optimization.Newton,
    grp: pbat.io.Archive,
    post_solve: bool = False,
):
    """Serialize Newton solver iteration data

    Args:
        fem (pbat.sim.dynamics.FemElastoDynamics): Finite element elasto dynamics
            problem
        k (int): Iteration index
        grp (pbat.io.Archive): Archive to store iteration data
    """
    iter = grp[f"{k:06d}"]
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
    newton.serialize(iter)
    if post_solve:
        iter.write_data("v", fem.v)


def newton_solve(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.newton.Params,
    archive: pbat.io.Archive | None = None,
):
    """Python-side Newton solve with optional serialization per iteration."""
    grp = archive["pbat.sim.algorithm.newton.Solve"] if archive is not None else None
    pbat.sim.algorithm.newton.initialize_solve(fem, params)
    newton: pbat.math.optimization.Newton = params.newton
    while newton.k < newton.n_max_iters:
        if grp is not None:
            serialize_newton_solver_iteration(fem, newton.k, newton, grp)
        if newton.gknorm2 < newton.gtol2:
            break
        if not pbat.sim.algorithm.newton.iterate(fem, params):
            break
        pbat.sim.algorithm.newton.prepare_next_iteration(fem, params)
    fem.back_substitute_integrated_positions_into_velocities()
    if grp is not None:
        serialize_newton_solver_iteration(fem, newton.k, newton, grp, post_solve=True)


def newton_integrate(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    params: pbat.sim.algorithm.newton.Params,
    initialization_strategy: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization,
    archive: pbat.io.Archive | None = None,
):
    """Python-side Newton integrate (one time step) with optional archiving."""
    fem.setup_time_integration_optimization(
        initialization_strategy=initialization_strategy
    )
    grp = (
        archive["pbat.sim.algorithm.newton.Integrate"] if archive is not None else None
    )
    if grp is not None:
        fem.serialize(grp)
    newton_solve(fem, params, archive=grp)
    fem.back_substitute_integrated_positions_into_velocities()
    fem.step()


def register_contact_frames_in_polyscope(
    mesh_dynamics: pbat.sim.contact.MeshDynamics,
    fcinds: np.ndarray,
    hecinds: np.ndarray,
    vcinds: np.ndarray,
):
    pcc = []
    if fcinds.shape[0] > 0:
        constraints = [mesh_dynamics.CF[f] for f in fcinds]
        for f, constraint in zip(fcinds, constraints):
            origins = np.stack([constraint.O, constraint.O, constraint.O])
            pcfc = ps.register_point_cloud(f"Contact frame f={f}", origins)
            pcfc.add_vector_quantity("basis", constraint.B.T, enabled=True)
            pcfc.add_vector_quantity("normal", np.stack([constraint.B[:,0]]*3), enabled=False)
            pcfc.add_vector_quantity(
                "forces",
                np.stack(
                    [
                        constraint.B[:, 0]
                        * (constraint.lagrange - constraint.k * constraint.C)
                        for _ in range(3)
                    ]
                ),
                enabled=False,
            )
            pcc.append(pcfc)
    if hecinds.shape[0] > 0:
        constraints = [mesh_dynamics.CHE[he] for he in hecinds]
        for he, constraint in zip(hecinds, constraints):
            origins = np.stack([constraint.O, constraint.O, constraint.O])
            pchec = ps.register_point_cloud(f"Contact frame he={he}", origins)
            pchec.add_vector_quantity("basis", constraint.B.T, enabled=True)
            pchec.add_vector_quantity("normal", np.stack([constraint.B[:,0]]*3), enabled=False)
            pchec.add_vector_quantity(
                "forces",
                np.stack(
                    [
                        constraint.B[:, 0]
                        * (constraint.lagrange - constraint.k * constraint.C)
                        for _ in range(3)
                    ]
                ),
                enabled=False,
            )
            pcc.append(pchec)
    if vcinds.shape[0] > 0:
        constraints = [mesh_dynamics.CV[v] for v in vcinds]
        for v, constraint in zip(vcinds, constraints):
            origins = np.stack([constraint.O, constraint.O, constraint.O])
            pcvc = ps.register_point_cloud(f"Contact frame v={v}", origins)
            pcvc.add_vector_quantity("basis", constraint.B.T, enabled=True)
            pcvc.add_vector_quantity("normal", np.stack([constraint.B[:,0]]*3), enabled=False)
            pcvc.add_vector_quantity(
                "forces",
                np.stack(
                    [
                        constraint.B[:, 0]
                        * (constraint.lagrange - constraint.k * constraint.C)
                        for _ in range(3)
                    ]
                ),
                enabled=False,
            )
            pcc.append(pcvc)
    return pcc


if __name__ == "__main__":
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Vertex Block Descent Simple API")
    ps.init()

    dynamics = pbat.sim.dynamics.FemElastoDynamics()
    mesh_dynamics = pbat.sim.contact.MeshDynamics()
    Y = 1e7  # Young's modulus
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
    # Newton solver params
    newton_params = (
        pbat.sim.algorithm.newton.Params()
        .with_optimizer(
            pbat.math.optimization.Newton(
                n_max_iters=10,
                gtol=1e-4,
                line_search=pbat.math.optimization.BackTrackingLineSearch(),
            )
        )
        .with_spd_correction(pbat.fem.HyperElasticSpdCorrection.Absolute)
        .construct()
    )
    newton_linsolvers = [
        pbat.sim.algorithm.newton.ELinearSolver.LLT,
        pbat.sim.algorithm.newton.ELinearSolver.PCGJacobi,
        pbat.sim.algorithm.newton.ELinearSolver.PCGIC,
        pbat.sim.algorithm.newton.ELinearSolver.PCGILUT,
        pbat.sim.algorithm.newton.ELinearSolver.PCGLaplacian,
    ]
    i_newton_linsol = 0  # Linear solver index
    newton_linsol_maxiters = 100
    newton_linsol_tol = 1e-6
    # Non-linear solvers
    solver_names = ["Base", "Anderson", "Broyden", "Chebyshev", "Newton"]
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

    # SDF visualization grid
    sdf_grid: ps.VolumeGrid = None
    sdf_grid_dims = (50, 50, 50)
    sdf_grid_bmin = -np.ones(3, dtype=np.float64)
    sdf_grid_bmax = np.ones(3, dtype=np.float64)
    sdf_transform_R = np.eye(3)
    sdf_transform_t = np.zeros(3)

    # Contact frame vis
    pcc = []
    show_contact_frames = False

    # transform_library = tlib.TransformLibrary()
    # transform_library.deserialize("primitive_transforms.h5")
    original_mask = None

    def callback():
        global Y, nu, rho, aext, b, v0, d_axis, d_percent, d_extremity, d_nodes
        global dt, s
        global vbd_params, anderson_params, broyden_params, chebyshev_params, newton_params
        global i_newton_linsol, newton_linsolvers, newton_linsol_maxiters, newton_linsol_tol
        global solver_names, i_solver, i_init_strategy, n_max_iters
        global i_broyden_l2_solver, i_broyden_jacobian_estimate
        global animate, export, t, vm, dpc
        global archive, archive_path, archive_flush_period
        # global transform_library, original_mask
        global mesh_dynamics
        global sdf_grid, sdf_grid_dims, sdf_grid_bmin, sdf_grid_bmax
        global pcc, show_contact_frames
        global sdf_transform_R, sdf_transform_t

        dirty = False
        is_new_mesh = False
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load mesh file", [imgui.GetWindowWidth() / 2.1, 0]):
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

            if imgui.Button("Load SDF", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select SDF forest",
                    defaultextension=".h5",
                    filetypes=[("SDF forest", "*.h5"), ("All files", "*.*")],
                )
                if file_path:
                    try:
                        archive_sdf = pbat.io.Archive(
                            file_path, pbat.io.AccessMode.ReadOnly
                        )
                        sdf_forest = pbat.geometry.sdf.Forest()
                        sdf_forest.deserialize(archive_sdf)
                        for r in sdf_forest.roots:
                            sdf_forest.transforms[r].R = sdf_transform_R
                            sdf_forest.transforms[r].t = sdf_transform_t
                        mesh_dynamics.set_static_geometry(sdf_forest)
                        # Create grid if it doesn't exist
                        if sdf_grid is None:
                            sdf_grid = ps.register_volume_grid(
                                "SDF Domain",
                                sdf_grid_dims,
                                sdf_grid_bmin,
                                sdf_grid_bmax,
                            )
                            sdf_grid.set_transform(np.eye(4))
                            sdf_grid.set_transparency(0.75)
                        # Sample SDF on grid
                        x, y, z = np.meshgrid(
                            np.linspace(
                                sdf_grid_bmin[0], sdf_grid_bmax[0], sdf_grid_dims[0]
                            ),
                            np.linspace(
                                sdf_grid_bmin[1], sdf_grid_bmax[1], sdf_grid_dims[1]
                            ),
                            np.linspace(
                                sdf_grid_bmin[2], sdf_grid_bmax[2], sdf_grid_dims[2]
                            ),
                            indexing="ij",
                        )
                        Xs = np.vstack([np.ravel(z), np.ravel(y), np.ravel(x)]).astype(
                            np.float64
                        )
                        sd = mesh_dynamics.sdf.eval(Xs).reshape(
                            sdf_grid_dims, order="F"
                        )
                        sdf_grid.add_scalar_quantity(
                            "SDF",
                            sd,
                            defined_on="nodes",
                            cmap="coolwarm",
                            isolines_enabled=True,
                            enable_isosurface_viz=True,
                            enable_gridcube_viz=False,
                            enabled=True,
                        )
                    except Exception as e:
                        print(f"Failed to load SDF: {e}")
                root.destroy()

            if imgui.Button("Load h5", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Open HDF5 simulation file",
                    defaultextension="*.*",
                    filetypes=[("All files", "*.*"), ("HDF5 files", "*.h5;*.hdf5")],
                )
                if file_path:
                    h5 = h5py.File(file_path, "r")
                    X, E, x, d_mask, lamegU = _read_mesh_and_state(h5)
                    if (X is None) or (E is None):
                        print("oops")
                        return
                    V = (x if x is not None else X).T  # to shape (n,3)
                    C = E.T  # to shape (m,4)
                    dynamics.construct(V.T, C.T)
                    dynamics.constrain(d_mask.ravel())
                    original_mask = d_mask.copy()
                    # dynamics.set_elastic_energy(dynamics.E, dynamics.wgU, dynamics.X, dynamics.lamegU[0], dynamics.lamegU[1])
                    element = pbat.fem.Element.Tetrahedron
                    order = 1  # linear shape functions only
                    qorder_U = order
                    n_elems = E.shape[1]

                    wgU = pbat.fem.mesh_quadrature_weights(
                        E, X, element, order=order, quadrature_order=qorder_U
                    )
                    egU = pbat.fem.mesh_quadrature_elements(E, wgU)
                    XgU = pbat.fem.mesh_reference_quadrature_points(
                        n_elems, element=element, order=order, quadrature_order=qorder_U
                    )
                    dynamics.set_elastic_energy(
                        np.ravel(egU, order="F"),
                        np.ravel(wgU, order="F"),
                        XgU,
                        np.ravel(lamegU[0], order="F"),
                        np.ravel(lamegU[1], order="F"),
                    )

                    vm = ps.register_volume_mesh("Mesh", dynamics.X.T, dynamics.E.T)
                    dpc = ps.register_point_cloud(
                        "Dirichlet Nodes", dynamics.x[:, d_nodes].T
                    )
                    is_new_mesh = True
                    d_axis = 3  # Don't apply default Dirichlet constraints
                root.destroy()
                # imesh = meshio.read(file_path)
                # V, C = imesh.points, imesh.cells_dict["tetra"]
                # dynamics.construct(V.T, C.T)
                # vm = ps.register_volume_mesh("Mesh", dynamics.X.T, dynamics.E.T)
                # is_new_mesh = True

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
            d_axis_updated, d_axis = imgui.InputInt("Axis (0=x,1=y,2=z,3=None)", d_axis)
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
            imgui.TreePop()

        if imgui.TreeNode("VBD"):
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

        if imgui.TreeNode("Contact"):
            if imgui.TreeNode("Mesh-SDF Contact"):
                params = mesh_dynamics.mesh_sdf_contact.params
                _, params.sigmaR = imgui.SliderFloat(
                    "sigmaR", params.sigmaR, 1e-2, 1.0, format="%.2f"
                )
                _, params.sigmaB = imgui.SliderFloat(
                    "sigmaB", params.sigmaB, 1e-2, 10.0, format="%.2f"
                )
                _, params.tauAred = imgui.SliderFloat(
                    "tauAred", params.tauAred, 1e-5, 1e-3, format="%.5f"
                )
                _, params.tauPred = imgui.SliderFloat(
                    "tauPred", params.tauPred, 1e-5, 1e-3, format="%.5f"
                )
                _, params.n_max_opt_iters = imgui.SliderInt(
                    "# max opt iters",
                    params.n_max_opt_iters,
                    1,
                    200,
                )
                _, params.coord_zero = imgui.SliderFloat(
                    "coord_zero",
                    params.coord_zero,
                    0,
                    1e-2,
                    format="%.8f",
                )
                _, params.hfd = imgui.SliderFloat(
                    "hfd", params.hfd, 1e-8, 1e-2, format="%.8f"
                )
                _, params.r = imgui.SliderFloat(
                    "r", params.r, 1e-3, 1e-1, format="%.8f"
                )
                imgui.TreePop()

            if imgui.TreeNode("Environment Contact Dynamics"):
                params = mesh_dynamics.env_contact_dynamics_params
                _, params.mu = imgui.InputFloat("Friction Coefficient", params.mu)
                _, params.beta = imgui.InputFloat("beta", params.beta)
                _, params.Fnmax = imgui.SliderFloat(
                    "Max Normal Force", params.Fnmax, 1e2, 1e8, format="%.1f"
                )
                _, params.gamma = imgui.SliderFloat(
                    "gamma", params.gamma, 1e-2, 1.0, format="%.6f"
                )
                _, params.kstart = imgui.SliderFloat(
                    "kstart", params.kstart, 1e1, 1e6, format="%.1f"
                )
                _, params.epsv = imgui.SliderFloat(
                    "epsv", params.epsv, 1e-5, 1e-1, format="%.6f"
                )
                imgui.TreePop()

            if imgui.TreeNode("SDF Visualization"):
                # Grid bounds
                bmin = int(sdf_grid_bmin[0])
                bmin_changed, bmin = imgui.InputInt("Grid Min", bmin)
                sdf_grid_bmin[:] = bmin
                bmax = int(sdf_grid_bmax[0])
                bmax_changed, bmax = imgui.InputInt("Grid Max", bmax)
                sdf_grid_bmax[:] = bmax
                # Grid resolution
                grid_res = sdf_grid_dims[0]
                dims_changed, grid_res = imgui.InputInt("Grid Resolution", grid_res)
                sdf_grid_dims = (grid_res, grid_res, grid_res)
                # Transform
                euler_angles = sp.spatial.transform.Rotation.from_matrix(
                    sdf_transform_R
                ).as_euler("xyz", degrees=True)
                R_changed, euler_angles = imgui.SliderFloat3(
                    "Euler angles", euler_angles, -180, 180
                )
                if R_changed:
                    euler_angles = [5 * round(ri / 5) for ri in euler_angles]
                    sdf_transform_R = sp.spatial.transform.Rotation.from_euler(
                        "xyz", euler_angles, degrees=True
                    ).as_matrix()
                t_changed, sdf_transform_t = imgui.SliderFloat3(
                    "Translation", sdf_transform_t, -1, 1
                )
                sdf_transform_t = np.array(sdf_transform_t)
                if R_changed or t_changed:
                    for r in mesh_dynamics.sdf_forest.roots:
                        mesh_dynamics.sdf_forest.transforms[r].R = sdf_transform_R
                        mesh_dynamics.sdf_forest.transforms[r].t = sdf_transform_t
                if (
                    bmin_changed
                    or bmax_changed
                    or dims_changed
                    or R_changed
                    or t_changed
                ):
                    try:
                        # Create or update grid
                        if (
                            sdf_grid is None
                            or bmin_changed
                            or bmax_changed
                            or dims_changed
                        ):
                            if sdf_grid is not None:
                                ps.remove_volume_grid("SDF Domain")
                            sdf_grid = ps.register_volume_grid(
                                "SDF Domain",
                                sdf_grid_dims,
                                sdf_grid_bmin,
                                sdf_grid_bmax,
                            )
                            sdf_grid.set_transform(np.eye(4))
                            sdf_grid.set_transparency(0.75)
                        # Sample SDF on grid
                        x, y, z = np.meshgrid(
                            np.linspace(
                                sdf_grid_bmin[0], sdf_grid_bmax[0], sdf_grid_dims[0]
                            ),
                            np.linspace(
                                sdf_grid_bmin[1], sdf_grid_bmax[1], sdf_grid_dims[1]
                            ),
                            np.linspace(
                                sdf_grid_bmin[2], sdf_grid_bmax[2], sdf_grid_dims[2]
                            ),
                            indexing="ij",
                        )
                        Xs = np.vstack([np.ravel(z), np.ravel(y), np.ravel(x)]).astype(
                            np.float64
                        )
                        sd = mesh_dynamics.sdf.eval(Xs).reshape(
                            sdf_grid_dims, order="F"
                        )
                        sdf_grid.add_scalar_quantity(
                            "SDF",
                            sd,
                            defined_on="nodes",
                            cmap="coolwarm",
                            isolines_enabled=True,
                            enable_gridcube_viz=False,
                            enable_isosurface_viz=True,
                            enabled=True,
                        )
                    except Exception as e:
                        print(f"Failed to visualize SDF: {e}")

                imgui.TreePop()

            _, show_contact_frames = imgui.Checkbox(
                "Visualize Contact Frames", show_contact_frames
            )
            imgui.TreePop()

        if imgui.TreeNode("Newton"):
            # Basic Newton parameters
            _, newton_params.newton.n_max_iters = imgui.InputInt(
                "Maximum Iterations", newton_params.newton.n_max_iters
            )
            _, gtol = imgui.InputFloat(
                "Gradient Tol", np.sqrt(newton_params.newton.gtol2), format="%.8f"
            )
            if isinstance(
                newton_params.newton.line_search,
                pbat.math.optimization.BackTrackingLineSearch,
            ):
                _, newton_params.newton.line_search.alpha = imgui.InputFloat(
                    "Initial step size",
                    newton_params.newton.line_search.alpha,
                    format="%.8f",
                )
                _, newton_params.newton.line_search.c = imgui.InputFloat(
                    "Armijo slope scale",
                    newton_params.newton.line_search.c,
                    format="%.8f",
                )
                _, newton_params.newton.line_search.tau = imgui.InputFloat(
                    "Step size reduction",
                    newton_params.newton.line_search.tau,
                    format="%.8f",
                )
                _, newton_params.newton.line_search.n_max_iters = imgui.InputInt(
                    "Max line search iters",
                    newton_params.newton.line_search.n_max_iters,
                )
            newton_params.newton.gtol2 = gtol * gtol
            linsol_changed, i_newton_linsol = imgui.Combo(
                "Linear Solver",
                i_newton_linsol,
                [linsol.name for linsol in newton_linsolvers],
            )
            linsol_maxiters_changed, newton_linsol_maxiters = imgui.InputInt(
                "Linear Solver Max Iters", newton_linsol_maxiters
            )
            linsol_tol_changed, newton_linsol_tol = imgui.InputFloat(
                "Linear Solver Tol", newton_linsol_tol, format="%.8f"
            )
            if linsol_changed or linsol_maxiters_changed or linsol_tol_changed:
                newton_params.with_linear_solver(
                    newton_linsolvers[i_newton_linsol],
                    newton_linsol_maxiters,
                    newton_linsol_tol,
                )
            imgui.TreePop()

        # Initialize VBD parameters
        if is_new_mesh:
            # Compute connected components
            XCC = np.zeros(dynamics.X.shape[1], dtype=np.int32)
            ECC = np.zeros(dynamics.E.shape[1], dtype=np.int32)
            Xord = np.zeros(dynamics.X.shape[1], dtype=np.int32)
            Eord = np.zeros(dynamics.E.shape[1], dtype=np.int32)
            Xord, Eord, XCC, ECC, n_components = (
                pbat.graph.sorted_connected_component_ordering(dynamics.X, dynamics.E)
            )
            # Reindex mesh by connected components
            dynamics.X, dynamics.E, XCC, ECC = (
                pbat.graph.reindex_mesh_by_connected_components(
                    dynamics.X, dynamics.E, XCC, ECC, Xord, Eord
                )
            )
            # Setup VBD parameters
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

            # Create MultiMesh via BoundaryTriangulation
            multi_mesh = pbat.sim.contact.MultiMesh(dynamics.E, XCC, n_components)
            # Construct MeshDynamics
            mesh_dynamics.construct(multi_mesh, mesh_dynamics.sdf_forest)
            # Initialize mesh-environment contact detection
            mesh_dynamics.initialize_mesh_environment_contact_detection()

            # Set SDF grid bounds based on mesh bounding box
            aabb_mesh = pypbat.geometry.aabb(dynamics.X)
            mesh_extents = aabb_mesh.max - aabb_mesh.min
            margin = 0.2 * np.max(mesh_extents)  # Add 20% margin

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
            if d_axis in [0, 1, 2]:
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
                d_mask = np.zeros(dynamics.X.shape[1], dtype=int)
                d_mask[d_nodes] = 1
                dynamics.constrain(d_mask)
            elif d_axis == 3:
                d_nodes = np.array([], dtype=int)
                d_mask = np.zeros(dynamics.X.shape[1], dtype=int)
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
            # dynamics.constrain(original_mask.ravel() if original_mask is not None else np.zeros(n_nodes, dtype=int))
            t = 0
            for pcci in pcc:
                pcci.remove()
            pcc = []

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
            # for transform in transform_library.transforms:
            #     v = dynamics.x[:, dynamics.dmask == transform.id]
            #     transformed_v = transform.apply(t, dt, v)
            #     dynamics.x[:, dynamics.dmask == transform.id] = transformed_v
            #     if transform.expired(t * dt) and transform.transform_type == TransformType.FIXED:
            #         dynamics.dmask[dynamics.dmask == transform.id] = 0
            #         dynamics.constrain(dynamics.dmask.ravel())
            # for i in range(dynamics.x.shape[1]):
            #     if dynamics.dmask[i] == transform.id:
            #         dynamics.x[:, i] = transformed_v[:, i]

            if i_solver == 0:
                vbd_integrate(dynamics, mesh_dynamics, vbd_params, archive=frame_group)
            elif i_solver == 1:
                anderson_integrate(
                    dynamics,
                    mesh_dynamics,
                    vbd_params,
                    anderson_params,
                    archive=frame_group,
                )
            elif i_solver == 2:
                broyden_integrate(
                    dynamics,
                    mesh_dynamics,
                    vbd_params,
                    broyden_params,
                    archive=frame_group,
                )
            elif i_solver == 3:
                chebyshev_integrate(
                    dynamics,
                    mesh_dynamics,
                    vbd_params,
                    chebyshev_params,
                    archive=frame_group,
                )
            elif i_solver == 4:
                newton_integrate(
                    dynamics,
                    newton_params,
                    init_strategies[i_init_strategy],
                    archive=frame_group,
                )
            if export:
                ps.screenshot()
            t = t + 1
            if archive is not None and (t % archive_flush_period == 0):
                try:
                    archive.flush()
                except Exception as e:
                    print(f"Archive flush error: {e}")

            # Show contact frames and clear old ones
            for pcci in pcc:
                try:
                    pcci.remove()
                except Exception as e:
                    print(f"Point cloud removal error: {e}")
            pcc = []
            if show_contact_frames:
                fmask = mesh_dynamics.mesh_sdf_contact.triangle_contact_mask
                hemask = mesh_dynamics.mesh_sdf_contact.half_edge_contact_mask
                vmask = mesh_dynamics.mesh_sdf_contact.vertex_contact_mask
                fcinds = np.where(fmask)[0]
                hecinds = np.where(hemask)[0]
                vcinds = np.where(vmask)[0]
                pcc = register_contact_frames_in_polyscope(
                    mesh_dynamics, fcinds, hecinds, vcinds
                )

        if not show_contact_frames and len(pcc) > 0:
            for pcci in pcc:
                pcci.remove()
            pcc = []

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
