# type: ignore
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
import meshio
from pbatoolkit import pbat


def barycentric_to_world(
    V: np.ndarray, F: np.ndarray, BC: np.ndarray, tricounts: np.ndarray
):
    """Convert barycentric contact points to world coordinates.

    Args:
        V: (|V|, 3) array of vertex positions.
        F: (|F|, 3) array of triangle indices.
        BC: (|contacts|, 3) array of barycentric contact coordinates.
        tricounts: (|F|,) array of contact counts per triangle.

    Returns:
        WC: (|contacts|, 3) array of world contact positions.
    """
    WC = np.zeros_like(BC)
    f = np.repeat(np.arange(F.shape[0]), tricounts)
    for i in range(3):
        WC += BC[:, i][:,np.newaxis] * V[F[f, i], :]
    return WC


def main():

    # Init Polyscope
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Mesh-SDF Contact")
    ps.init()

    # Volume grid for SDF visualization
    bmin = -np.ones(3, dtype=np.float64)
    bmax = np.ones(3, dtype=np.float64)
    dims = (100, 100, 100)
    grid = ps.register_volume_grid("SDF Domain", dims, bmin, bmax)
    grid.set_transparency(0.75)
    # Prepare sampling points (polyscope expects x fastest, then y, then z)
    x, y, z = np.meshgrid(
        np.linspace(bmin[0], bmax[0], dims[0]),
        np.linspace(bmin[1], bmax[1], dims[1]),
        np.linspace(bmin[2], bmax[2], dims[2]),
        indexing="ij",
    )
    Xs = np.vstack([np.ravel(z), np.ravel(y), np.ravel(x)]).astype(np.float64)
    forest = pbat.geometry.sdf.Forest()
    composite: pbat.geometry.sdf.Composite = None

    # Contact engine setup
    mesh_sdf_contact_params = pbat.sim.contact.MeshSdfContactParams()
    mesh_sdf_contact = pbat.sim.contact.MeshSdfContact()
    V: np.ndarray = None
    F: np.ndarray = None
    sm: ps.SurfaceMesh = None
    pc: ps.PointCloud = None

    # UI state
    auto_detect = False
    step_detect = False

    def ui_callback():
        nonlocal forest, composite
        nonlocal mesh_sdf_contact_params, mesh_sdf_contact
        nonlocal V, F
        nonlocal sm, pc
        nonlocal auto_detect, step_detect
        # Allow forest reload
        if imgui.TreeNode("I/O"):
            if imgui.Button("Load SDF", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select SDF forest",
                    defaultextension=".h5",
                    filetypes=[("SDF forest", "*.h5"), ("All files", "*.*")],
                )
                if file_path:
                    archive = pbat.io.Archive(file_path, pbat.io.AccessMode.ReadOnly)
                    forest.deserialize(archive)
                    composite = pbat.geometry.sdf.Composite(forest)
                    # Re-sample SDF volume for visualization
                    sd_new = composite.eval(Xs).reshape(dims, order="F")
                    grid.add_scalar_quantity(
                        "SDF",
                        sd_new,
                        defined_on="nodes",
                        cmap="coolwarm",
                        vminmax=(-np.max(bmax - bmin), np.max(bmax - bmin)),
                        isolines_enabled=True,
                        enable_isosurface_viz=True,
                        enabled=True,
                    )
                root.destroy()
            if imgui.Button("Load Mesh", [imgui.GetWindowWidth() / 2.1, 0]):
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename(
                    title="Select Mesh",
                    defaultextension=".obj",
                    filetypes=[
                        ("Mesh files", "*.obj;*.off;*.stl;*.ply"),
                        ("All files", "*.*"),
                    ],
                )
                if file_path:
                    mesh = meshio.read(file_path)
                    V = mesh.points
                    # Find triangle cells
                    if "triangle" in mesh.cells_dict:
                        F = mesh.cells_dict["triangle"]
                    elif "tri" in mesh.cells_dict:
                        F = mesh.cells_dict["tri"]
                    elif "tetra" in mesh.cells_dict:
                        T = mesh.cells_dict["tetra"]
                        _, F = pbat.geometry.simplex_mesh_boundary(T.T, V.shape[0])
                    else:
                        raise ValueError("Mesh must contain triangle elements.")
                    sm = ps.register_surface_mesh("Mesh", V, F)
                    mesh_sdf_contact.initialize(V.T, F.T, mesh_sdf_contact_params)
                root.destroy()
            imgui.TreePop()

        # Controls: Auto/Step detection and parameter editing
        do_detect = False
        params_changed = False
        if imgui.TreeNode("Mesh-SDF Contact"):
            # Parameters UI
            imgui.Separator()
            imgui.TextUnformatted("Parameters")
            _, mesh_sdf_contact_params.sigmaR = imgui.SliderFloat(
                "sigmaR", mesh_sdf_contact_params.sigmaR, 1e-6, 1.0
            )
            _, mesh_sdf_contact_params.sigmaB = imgui.SliderFloat(
                "sigmaB", mesh_sdf_contact_params.sigmaB, 1e-6, 10.0
            )
            _, mesh_sdf_contact_params.tauAred = imgui.SliderFloat(
                "tauAred", mesh_sdf_contact_params.tauAred, 1e-8, 1.0
            )
            _, mesh_sdf_contact_params.tauPred = imgui.SliderFloat(
                "tauPred", mesh_sdf_contact_params.tauPred, 1e-8, 1.0
            )
            _, mesh_sdf_contact_params.n_max_contacts_per_triangle = imgui.SliderInt(
                "n_max_contacts_per_triangle",
                mesh_sdf_contact_params.n_max_contacts_per_triangle,
                1,
                32,
            )
            _, mesh_sdf_contact_params.n_max_opt_iters_per_triangle = imgui.SliderInt(
                "n_max_opt_iters_per_triangle",
                mesh_sdf_contact_params.n_max_opt_iters_per_triangle,
                1,
                200,
            )
            _, mesh_sdf_contact_params.coord_zero = imgui.SliderFloat(
                "coord_zero", mesh_sdf_contact_params.coord_zero, 1e-8, 1e-2
            )
            _, mesh_sdf_contact_params.hfd = imgui.SliderFloat(
                "hfd", mesh_sdf_contact_params.hfd, 1e-8, 1e-2
            )
            _, mesh_sdf_contact_params.r = imgui.SliderFloat(
                "r", mesh_sdf_contact_params.r, 1e-8, 1e-1
            )
            imgui.TreePop()

        # Run controls
        _, auto_detect = imgui.Checkbox("Auto detect each frame", auto_detect)
        do_detect = imgui.Button("Step Detect", [imgui.GetWindowWidth() / 2.1, 0])

        # Auto detection flag
        do_detect = do_detect or auto_detect

        if (
            do_detect
            and (V is not None)
            and (F is not None)
            and (composite is not None)
        ):
            # Transform point positions
            VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
            T = sm.get_transform()
            VT = (T @ VH)[:3, :].T
            # Re-initialize engine if parameters changed so it picks up new config
            mesh_sdf_contact.initialize(VT.T, F.T, mesh_sdf_contact_params)
            mesh_sdf_contact.triangle_sdf_contact_detection(VT.T, F.T, composite)
            xc = mesh_sdf_contact.flat_contacts
            if xc.shape[1] > 0:
                xc = barycentric_to_world(
                    VT,
                    F,
                    xc.T,
                    mesh_sdf_contact.triangle_contact_counts,
                )
                pc = ps.register_point_cloud("Contact Points", xc)
            else:
                if pc is not None:
                    ps.remove_point_cloud("Contact Points")
                    pc = None

            # Per-face contact density (counts)
            if F is not None and F.shape[0] > 0:
                sm.add_scalar_quantity(
                    "Contact Density",
                    mesh_sdf_contact.triangle_contact_counts,
                    defined_on="faces",
                    enabled=True,
                )

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
