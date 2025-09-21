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
    box = pbat.geometry.sdf.Box(he=np.array([0.5, 0.5, 0.5]))
    sd_box = box.eval(X).reshape(dims)

    # SDF unary nodes
    scale_node = pbat.geometry.sdf.Scale(s=2.0)
    sd_scale = scale_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    elongate_node = pbat.geometry.sdf.Elongate(h=np.array([0.0, 0.0, 0.0]))
    sd_elongate = elongate_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    round_node = pbat.geometry.sdf.Round(r=0.0)
    sd_round = round_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    onion_node = pbat.geometry.sdf.Onion(t=0.0)
    sd_onion = onion_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    symmetrize_node = pbat.geometry.sdf.Symmetrize()
    sd_symmetrize = symmetrize_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    repeat_node = pbat.geometry.sdf.Repeat(s=1.0, l=np.array([5.0, 5.0, 5.0]))
    sd_repeat = repeat_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    bump_node = pbat.geometry.sdf.Bump(
        f=np.array([20.0, 20.0, 20.0]), g=np.array([0.1, 0.1, 0.1])
    )
    sd_bump = bump_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    twist_node = pbat.geometry.sdf.Twist(k=0.1)
    sd_twist = twist_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    bend_node = pbat.geometry.sdf.Bend(k=0.1)
    sd_bend = bend_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)

    # Polyscope visualization
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("SDF unary nodes")
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
        "Scale",
        sd_scale,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Elongate",
        sd_elongate,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Round",
        sd_round,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Onion",
        sd_onion,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Symmetrize",
        sd_symmetrize,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Repeat",
        sd_repeat,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Bump",
        sd_bump,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Twist",
        sd_twist,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )
    grid.add_scalar_quantity(
        "Bend",
        sd_bend,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        enable_isosurface_viz=enable_isosurface_viz,
    )

    def callback():
        global sd_box
        global sd_scale, sd_elongate, sd_round, sd_onion, sd_symmetrize, sd_repeat, sd_bump, sd_twist, sd_bend
        # Box UI
        box_updated = False
        if imgui.TreeNode("Box"):
            he_updated, he = imgui.SliderFloat3("Half Extents", box.he, 0.1, 10.0)
            box_updated = he_updated
            if box_updated:
                box.he = np.array(he)
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
        # Scale UI
        if imgui.TreeNode("Scale"):
            s_updated, s = imgui.SliderFloat("Scale", scale_node.s, 0.01, 5.0)
            if s_updated or box_updated:
                scale_node.s = s
                sd_scale = scale_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Scale",
                    sd_scale,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Elongate UI
        if imgui.TreeNode("Elongate"):
            h_updated, h = imgui.SliderFloat3(
                "Elongate Half Extents", elongate_node.h, 0.0, 10.0
            )
            updated = h_updated
            if updated or box_updated:
                elongate_node.h = np.array(h)
                sd_elongate = elongate_node.eval(X, sdf=lambda x: box.eval(x)).reshape(
                    dims
                )
                grid.add_scalar_quantity(
                    "Elongate",
                    sd_elongate,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Round UI
        if imgui.TreeNode("Round"):
            r_updated, r = imgui.SliderFloat("Round Radius", round_node.r, 0.0, 5.0)
            if r_updated or box_updated:
                round_node.r = r
                sd_round = round_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Round",
                    sd_round,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Onion UI
        if imgui.TreeNode("Onion"):
            t_updated, t = imgui.SliderFloat("Onion Thickness", onion_node.t, 0.0, 5.0)
            if t_updated or box_updated:
                onion_node.t = t
                sd_onion = onion_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Onion",
                    sd_onion,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Symmetrize UI
        if imgui.TreeNode("Symmetrize"):
            if box_updated:
                sd_symmetrize = symmetrize_node.eval(
                    X, sdf=lambda x: box.eval(x)
                ).reshape(dims)
                grid.add_scalar_quantity(
                    "Symmetrize",
                    sd_symmetrize,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Repeat UI
        if imgui.TreeNode("Repeat"):
            s_updated, s = imgui.SliderFloat("Scale", repeat_node.s, 0.01, 10.0)
            l_updated, l = imgui.SliderFloat3("Extents", repeat_node.l, 0.01, 10.0)
            updated = s_updated or l_updated
            if updated or box_updated:
                repeat_node.s = s
                repeat_node.l = np.array(l)
                sd_repeat = repeat_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Repeat",
                    sd_repeat,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Bump UI
        if imgui.TreeNode("Bump"):
            f_updated, f = imgui.SliderFloat3("Frequency", bump_node.f, 1.0, 50.0)
            g_updated, g = imgui.SliderFloat3("Amplitude", bump_node.g, 0.01, 10.0)
            updated = f_updated or g_updated
            if updated or box_updated:
                bump_node.f = np.array(f)
                bump_node.g = np.array(g)
                sd_bump = bump_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Bump",
                    sd_bump,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Twist UI
        if imgui.TreeNode("Twist"):
            k_updated, k = imgui.SliderFloat("Twist Rate", twist_node.k, -1.0, 1.0)
            if k_updated or box_updated:
                twist_node.k = k
                sd_twist = twist_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Twist",
                    sd_twist,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()
        # Bend UI
        if imgui.TreeNode("Bend"):
            k_updated, k = imgui.SliderFloat("Bend Rate", bend_node.k, -1.0, 1.0)
            if k_updated or box_updated:
                bend_node.k = k
                sd_bend = bend_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Bend",
                    sd_bend,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                )
            imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()
