import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np

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

    # SDFs
    sphere = pbat.geometry.sdf.Sphere(R=1.0)
    sd_sphere = sphere.eval(X).reshape(dims)

    box = pbat.geometry.sdf.Box(he=np.array([0.5, 0.5, 0.5]))
    sd_box = box.eval(X).reshape(dims)

    # SDF binary nodes
    union_node = pbat.geometry.sdf.Union()
    sd_union = np.array(
        [
            union_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    difference_node = pbat.geometry.sdf.Difference()
    sd_difference = np.array(
        [
            difference_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    intersection_node = pbat.geometry.sdf.Intersection()
    sd_intersection = np.array(
        [
            intersection_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    xor_node = pbat.geometry.sdf.ExclusiveOr()
    sd_xor = np.array(
        [
            xor_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    smooth_union_node = pbat.geometry.sdf.SmoothUnion(k=0.25)
    sd_smooth_union = np.array(
        [
            smooth_union_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    smooth_difference_node = pbat.geometry.sdf.SmoothDifference(k=0.25)
    sd_smooth_difference = np.array(
        [
            smooth_difference_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    smooth_intersection_node = pbat.geometry.sdf.SmoothIntersection(k=0.25)
    sd_smooth_intersection = np.array(
        [
            smooth_intersection_node.eval(sds, sdb)
            for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
        ]
    ).reshape(dims)

    # Polyscope visualization
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("SDF binary nodes")
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
        "Box",
        sd_box,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Sphere",
        sd_sphere,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Union",
        sd_union,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Difference",
        sd_difference,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Intersection",
        sd_intersection,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Xor",
        sd_xor,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Smooth Union",
        sd_smooth_union,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Smooth Difference",
        sd_smooth_difference,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Smooth Intersection",
        sd_smooth_intersection,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )

    def callback():
        global sd_sphere, sd_box
        global sd_union, sd_difference, sd_intersection, sd_xor
        global sd_smooth_union, sd_smooth_difference, sd_smooth_intersection
        # Box UI
        box_updated = False
        sphere_updated = False
        if imgui.TreeNode("Box"):
            hex_updated, hex = imgui.SliderFloat("Half Extent X", box.he[0], 0.1, 10.0)
            hey_updated, hey = imgui.SliderFloat("Half Extent Y", box.he[1], 0.1, 10.0)
            hez_updated, hez = imgui.SliderFloat("Half Extent Z", box.he[2], 0.1, 10.0)
            box_updated = hex_updated or hey_updated or hez_updated
            if box_updated:
                box.he = np.array([hex, hey, hez])
                sd_box = box.eval(X).reshape(dims)
                grid.add_scalar_quantity(
                    "Box",
                    sd_box,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Sphere UI
        if imgui.TreeNode("Sphere"):
            sphere_updated, r = imgui.SliderFloat("Radius", sphere.R, 0.1, 5.0)
            if sphere_updated:
                sphere.R = r
                sd_sphere = sphere.eval(X).reshape(dims)
                grid.add_scalar_quantity(
                    "Sphere",
                    sd_sphere,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Union UI
        if imgui.TreeNode("Union"):
            if box_updated or sphere_updated:
                sd_union = np.array(
                    [
                        union_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Union",
                    sd_union,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Difference UI
        if imgui.TreeNode("Difference"):
            if box_updated or sphere_updated:
                sd_difference = np.array(
                    [
                        difference_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Difference",
                    sd_difference,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Intersection UI
        if imgui.TreeNode("Intersection"):
            if box_updated or sphere_updated:
                sd_intersection = np.array(
                    [
                        intersection_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Intersection",
                    sd_intersection,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Xor UI
        if imgui.TreeNode("Xor"):
            if box_updated or sphere_updated:
                sd_xor = np.array(
                    [
                        xor_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Xor",
                    sd_xor,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Smooth Union UI
        if imgui.TreeNode("Smooth Union"):
            k_updated, k = imgui.SliderFloat(
                "Smoothness", smooth_union_node.k, 0.01, 10.0
            )
            if box_updated or sphere_updated or k_updated:
                smooth_union_node.k = k
                sd_smooth_union = np.array(
                    [
                        smooth_union_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Smooth Union",
                    sd_smooth_union,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Smooth Difference UI
        if imgui.TreeNode("Smooth Difference"):
            k_updated, k = imgui.SliderFloat(
                "Smoothness", smooth_difference_node.k, 0.01, 10.0
            )
            if box_updated or sphere_updated or k_updated:
                smooth_difference_node.k = k
                sd_smooth_difference = np.array(
                    [
                        smooth_difference_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Smooth Difference",
                    sd_smooth_difference,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Smooth Intersection UI
        if imgui.TreeNode("Smooth Intersection"):
            k_updated, k = imgui.SliderFloat(
                "Smoothness", smooth_intersection_node.k, 0.01, 10.0
            )
            if box_updated or sphere_updated or k_updated:
                smooth_intersection_node.k = k
                sd_smooth_intersection = np.array(
                    [
                        smooth_intersection_node.eval(sds, sdb)
                        for sds, sdb in zip(np.ravel(sd_sphere), np.ravel(sd_box))
                    ]
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Smooth Intersection",
                    sd_smooth_intersection,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()
