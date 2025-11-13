# type: ignore
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
import meshio
from pbatoolkit import pbat, pypbat


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
    sdf: pbat.geometry.sdf.Composite = None

    # Contact engine setup
    mesh_sdf_contact = pbat.sim.contact.MeshSdfContact()
    X: np.ndarray = None
    V: np.ndarray = None
    F: np.ndarray = None
    GHEF: np.ndarray = None
    GXV: np.ndarray = None
    HE: np.ndarray = None
    sm: ps.SurfaceMesh = None
    pcf: ps.PointCloud = None  # for face contacts
    pche: ps.PointCloud = None  # for half-edge contacts
    pcv: ps.PointCloud = None  # for vertex contacts

    profiler = pypbat.profiling.Profiler()

    # UI state
    auto_detect = False
    step_detect = False
    deduplicate_contacts = False

    def ui_callback():
        nonlocal forest, sdf
        nonlocal mesh_sdf_contact
        nonlocal X, V, F, GHEF, GXV, HE
        nonlocal sm, pcf, pche, pcv
        nonlocal auto_detect, step_detect, deduplicate_contacts
        nonlocal profiler

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
                    sdf = pbat.geometry.sdf.Composite(forest)
                    # Re-sample SDF volume for visualization
                    sd_new = sdf.eval(Xs).reshape(dims, order="F")
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
                    X = mesh.points.T
                    # Find triangle cells
                    if "triangle" in mesh.cells_dict:
                        V = np.arange(X.shape[1])
                        F = mesh.cells_dict["triangle"].T
                    elif "tri" in mesh.cells_dict:
                        V = np.arange(X.shape[1])
                        F = mesh.cells_dict["tri"].T
                    elif "tetra" in mesh.cells_dict:
                        T = mesh.cells_dict["tetra"].T
                        V, F = pbat.geometry.simplex_mesh_boundary(T, X.shape[1])
                    else:
                        raise ValueError("Mesh must contain triangle elements.")
                    GHEF = pbat.geometry.half_edge_face_adjacency(F)
                    GXV = np.full(X.shape[1], -1)
                    GXV[V] = np.arange(V.shape[0])
                    HE = pbat.geometry.half_edges(F)
                    sm = ps.register_surface_mesh("Mesh", X.T, F.T)
                    mesh_sdf_contact.initialize(V, F)
                root.destroy()
            imgui.TreePop()

        # Controls: Auto/Step detection and parameter editing
        do_detect = False
        if imgui.TreeNode("Mesh-SDF Contact"):
            # Parameters UI
            imgui.Separator()
            imgui.TextUnformatted("Parameters")
            _, mesh_sdf_contact.params.sigmaR = imgui.SliderFloat(
                "sigmaR", mesh_sdf_contact.params.sigmaR, 1e-6, 1.0, format="%.6f"
            )
            _, mesh_sdf_contact.params.sigmaB = imgui.SliderFloat(
                "sigmaB", mesh_sdf_contact.params.sigmaB, 1e-6, 10.0, format="%.6f"
            )
            _, mesh_sdf_contact.params.tauAred = imgui.SliderFloat(
                "tauAred", mesh_sdf_contact.params.tauAred, 1e-8, 1.0, format="%.8f"
            )
            _, mesh_sdf_contact.params.tauPred = imgui.SliderFloat(
                "tauPred", mesh_sdf_contact.params.tauPred, 1e-8, 1.0, format="%.8f"
            )
            _, mesh_sdf_contact.params.n_max_contacts_per_triangle = imgui.SliderInt(
                "n_max_contacts_per_triangle",
                mesh_sdf_contact.params.n_max_contacts_per_triangle,
                1,
                32,
            )
            _, mesh_sdf_contact.params.n_max_opt_iters_per_triangle = imgui.SliderInt(
                "n_max_opt_iters_per_triangle",
                mesh_sdf_contact.params.n_max_opt_iters_per_triangle,
                1,
                200,
            )
            _, mesh_sdf_contact.params.coord_zero = imgui.SliderFloat(
                "coord_zero",
                mesh_sdf_contact.params.coord_zero,
                1e-8,
                1e-2,
                format="%.8f",
            )
            _, mesh_sdf_contact.params.hfd = imgui.SliderFloat(
                "hfd", mesh_sdf_contact.params.hfd, 1e-8, 1e-2, format="%.8f"
            )
            _, mesh_sdf_contact.params.r = imgui.SliderFloat(
                "r", mesh_sdf_contact.params.r, 1e-8, 1e-1, format="%.8f"
            )
            _, deduplicate_contacts = imgui.Checkbox(
                "Deduplicate contacts", deduplicate_contacts
            )
            imgui.TreePop()

        # Run controls
        _, auto_detect = imgui.Checkbox("Auto detect each frame", auto_detect)
        do_detect = imgui.Button("Step Detect", [imgui.GetWindowWidth() / 2.1, 0])

        # Auto detection flag
        do_detect = do_detect or auto_detect

        if do_detect and (V is not None) and (F is not None) and (sdf is not None):
            # Transform point positions
            XH = np.vstack([X, np.ones((1, X.shape[1]))])
            T = sm.get_transform()
            XT = (T @ XH)[:3, :]
            # Re-initialize engine if parameters changed so it picks up new config
            mesh_sdf_contact.initialize(V, F)
            # Run contact detection
            profiler.begin_frame("Physics")
            mesh_sdf_contact.prepare_iteration()
            mesh_sdf_contact.triangle_sdf_contact_detection(XT, F, sdf)
            if deduplicate_contacts:
                mesh_sdf_contact.deduplicate_contact_set(F, GHEF, GXV)
            profiler.end_frame("Physics")
            uvcf = mesh_sdf_contact.triangle_contacts
            uche = mesh_sdf_contact.half_edge_contacts
            vc = mesh_sdf_contact.vertex_contacts
            if uvcf.shape[1] > 0:
                # Get the first barycentric coordinate as 1-u-v
                u = uvcf[0, :]
                v = uvcf[1, :]
                w = 1 - u - v
                # Get the triangle indices associated with each contact
                f = np.repeat(
                    np.arange(F.shape[1]), mesh_sdf_contact.triangle_contact_counts
                )
                # Compute world contact positions via triangle barycentric interpolation
                xcf = (
                    XT[:, F[0, f]] * w[np.newaxis, :]
                    + XT[:, F[1, f]] * u[np.newaxis, :]
                    + XT[:, F[2, f]] * v[np.newaxis, :]
                )
                pcf = ps.register_point_cloud("Face Contacts", xcf.T)
            else:
                if ps.has_point_cloud("Face Contacts"):
                    ps.remove_point_cloud("Face Contacts")
            if uche.shape[0] > 0:
                # Get the first barycentric coordinate as 1-u
                u = uche
                v = 1 - u
                # Get the half-edge indices associated with each contact
                he = np.repeat(
                    np.arange(HE.shape[1]),
                    mesh_sdf_contact.half_edge_contact_counts,
                )
                # Compute world contact positions via half-edge barycentric interpolation
                xche = (
                    XT[:, HE[0, he]] * v[np.newaxis, :]
                    + XT[:, HE[1, he]] * u[np.newaxis, :]
                )
                pche = ps.register_point_cloud("Half-Edge Contacts", xche.T)
            else:
                if ps.has_point_cloud("Half-Edge Contacts"):
                    ps.remove_point_cloud("Half-Edge Contacts")
            if vc.shape[0] > 0:
                xcv = XT[:, V[vc]]
                pcv = ps.register_point_cloud("Vertex Contacts", xcv.T)
            else:
                if ps.has_point_cloud("Vertex Contacts"):
                    ps.remove_point_cloud("Vertex Contacts")

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
