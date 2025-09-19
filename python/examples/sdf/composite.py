import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import scipy as sp


def node_ui(node, transform) -> bool:
    dirty = False
    if imgui.TreeNode(type(node).__name__):
        # Parameters
        if isinstance(node, pbat.geometry.sdf.Box):
            he_updated, he = imgui.SliderFloat3("Half Extent", node.he, 0.1, 10.0)
            if he_updated:
                node.he = np.array(he)
            dirty = he_updated
        elif isinstance(node, pbat.geometry.sdf.Sphere):
            r_updated, r = imgui.SliderFloat("Radius", node.R, 0.1, 10.0)
            if r_updated:
                node.R = r
            dirty = r_updated
        elif isinstance(node, pbat.geometry.sdf.Elongate):
            h_updated, h = imgui.SliderFloat3("Length", node.h, 0.1, 10.0)
            if h_updated:
                node.h = np.array(h)
            dirty = h_updated
        elif isinstance(node, pbat.geometry.sdf.Bend):
            k_updated, k = imgui.SliderFloat("Curvature", node.k, -0.5, 0.5)
            if k_updated:
                node.k = k
            dirty = k_updated
        elif isinstance(node, pbat.geometry.sdf.Plane):
            pass
        elif isinstance(node, pbat.geometry.sdf.SmoothUnion):
            k_updated, k = imgui.SliderFloat("Smoothness", node.k, 0.0, 10.0)
            if k_updated:
                node.k = k
            dirty = k_updated
        elif isinstance(node, pbat.geometry.sdf.Difference):
            pass

        # Transform
        t_updated, t = imgui.SliderFloat3("Translation", transform.t, -10.0, 10.0)
        if t_updated:
            transform.t = np.array(t)

        euler_angles = sp.spatial.transform.Rotation.from_matrix(transform.R).as_euler(
            "xyz", degrees=True
        )
        r_updated, r = imgui.SliderFloat3("Rotation XYZ", euler_angles, -180.0, 180.0)
        if r_updated:
            transform.R = sp.spatial.transform.Rotation.from_euler(
                "xyz", r, degrees=True
            ).as_matrix()

        dirty |= t_updated or r_updated
        imgui.TreePop()
    return dirty


if __name__ == "__main__":
    # Domain
    bmin = np.array([-10, -10, -10])
    bmax = np.array([10, 10, 10])
    dims = (50, 50, 50)
    x, y, z = np.meshgrid(
        np.linspace(bmin[0], bmax[0], dims[0]),
        np.linspace(bmin[1], bmax[1], dims[1]),
        np.linspace(bmin[2], bmax[2], dims[2]),
    )
    X = np.vstack([np.ravel(x), np.ravel(y), np.ravel(z)]).astype(np.float64)

    # SDF nodes
    sphere = pbat.geometry.sdf.Sphere(R=1.0)
    box = pbat.geometry.sdf.Box(he=np.array([0.5, 0.5, 0.5]))
    elongate = pbat.geometry.sdf.Elongate(h=np.array([0.0, 0.0, 0.0]))
    bend = pbat.geometry.sdf.Bend(k=0.0)
    plane = pbat.geometry.sdf.Plane()
    sunion = pbat.geometry.sdf.SmoothUnion(k=0.25)
    sdifference = pbat.geometry.sdf.Difference()

    # Composite
    nodes = [
        sphere,
        box,
        elongate,
        bend,
        plane,
        sunion,
        sdifference,
    ]
    children = [
        (-1, -1),
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, -1),
        (2, 3),
        (5, 4)
    ]
    transforms = [pbat.geometry.sdf.Transform.eye() for _ in nodes]
    roots, parents = pbat.geometry.sdf.roots_and_parents(children)
    forest = pbat.geometry.sdf.Forest(nodes, transforms, children, roots)
    composite = pbat.geometry.sdf.Composite(forest)
    if composite.status != pbat.geometry.sdf.ECompositeStatus.Valid:
        raise ValueError("Composite SDF is not valid")
    sd_composite = composite.eval(X).reshape(dims)

    # Polyscope visualization
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("SDF composite")
    ps.init()

    slice_plane = ps.add_scene_slice_plane()
    slice_plane.set_draw_plane(False)
    slice_plane.set_draw_widget(True)
    isolines = True
    enable_isosurface_viz = True
    vminmax = (-10, 10)
    cmap = "coolwarm"
    grid = ps.register_volume_grid("Domain", dims, bmin, bmax)
    grid.add_scalar_quantity(
        "Composite",
        sd_composite,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
        enabled=True,
    )

    def callback():
        global sd_composite

        # Box UI
        dirty = False
        for i, (node, transform) in enumerate(zip(nodes, transforms)):
            imgui.PushID(i)
            dirty |= node_ui(node, transform)
            imgui.PopID()
        # Update SDF
        if dirty:
            forest = pbat.geometry.sdf.Forest(nodes, transforms, children, roots)
            composite = pbat.geometry.sdf.Composite(forest)
            sd_composite = composite.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Composite",
                sd_composite,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
                enable_isosurface_viz=enable_isosurface_viz,
                enabled=True,
            )

    ps.set_user_callback(callback)
    ps.show()
