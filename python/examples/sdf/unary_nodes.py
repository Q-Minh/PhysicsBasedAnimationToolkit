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
    ps.set_program_name("SDF primitives")
    ps.init()

    slice_plane = ps.add_scene_slice_plane()
    slice_plane.set_draw_plane(False)
    slice_plane.set_draw_widget(True)
    isolines = True
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
    )
    grid.add_scalar_quantity(
        "Scale",
        sd_scale,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Elongate",
        sd_elongate,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Round",
        sd_round,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Onion",
        sd_onion,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Symmetrize",
        sd_symmetrize,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Repeat",
        sd_repeat,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Bump",
        sd_bump,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Twist",
        sd_twist,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Bend",
        sd_bend,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )

    def callback():
        # Box UI
        box_updated = False
        if imgui.TreeNode("Box"):
            hex_updated, hex = imgui.SliderFloat("Half Extent X", box.he[0], 0.1, 5.0)
            hey_updated, hey = imgui.SliderFloat("Half Extent Y", box.he[1], 0.1, 5.0)
            hez_updated, hez = imgui.SliderFloat("Half Extent Z", box.he[2], 0.1, 5.0)
            box_updated = hex_updated or hey_updated or hez_updated
            if box_updated:
                box.he = np.array([hex, hey, hez])
                sd = box.eval(X).reshape(dims)
                grid.add_scalar_quantity(
                    "Box",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Scale UI
        if imgui.TreeNode("Scale"):
            s_updated, s = imgui.SliderFloat("Scale", scale_node.s, 0.01, 5.0)
            if s_updated or box_updated:
                scale_node.s = s
                sd = scale_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Scale",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Elongate UI
        if imgui.TreeNode("Elongate"):
            h0_updated, h0 = imgui.SliderFloat(
                "Elongate X", elongate_node.h[0], 0.0, 10.0
            )
            h1_updated, h1 = imgui.SliderFloat(
                "Elongate Y", elongate_node.h[1], 0.0, 10.0
            )
            h2_updated, h2 = imgui.SliderFloat(
                "Elongate Z", elongate_node.h[2], 0.0, 10.0
            )
            updated = h0_updated or h1_updated or h2_updated
            if updated or box_updated:
                elongate_node.h = np.array([h0, h1, h2])
                sd = elongate_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Elongate",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Round UI
        if imgui.TreeNode("Round"):
            r_updated, r = imgui.SliderFloat("Round Radius", round_node.r, 0.0, 5.0)
            if r_updated or box_updated:
                round_node.r = r
                sd = round_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Round",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Onion UI
        if imgui.TreeNode("Onion"):
            t_updated, t = imgui.SliderFloat("Onion Thickness", onion_node.t, 0.0, 5.0)
            if t_updated or box_updated:
                onion_node.t = t
                sd = onion_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Onion",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Symmetrize UI
        if imgui.TreeNode("Symmetrize"):
            if box_updated:
                sd = symmetrize_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Symmetrize",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Repeat UI
        if imgui.TreeNode("Repeat"):
            s_updated, s = imgui.SliderFloat("Scale", repeat_node.s, 0.01, 10.0)
            l0_updated, l0 = imgui.SliderFloat("Extent X", repeat_node.l[0], 0.01, 10.0)
            l1_updated, l1 = imgui.SliderFloat("Extent Y", repeat_node.l[1], 0.01, 10.0)
            l2_updated, l2 = imgui.SliderFloat("Extent Z", repeat_node.l[2], 0.01, 10.0)
            updated = s_updated or l0_updated or l1_updated or l2_updated
            if updated or box_updated:
                repeat_node.s = s
                repeat_node.l = np.array([l0, l1, l2])
                sd = repeat_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Repeat",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Bump UI
        if imgui.TreeNode("Bump"):
            f0_updated, f0 = imgui.SliderFloat("Frequency X", bump_node.f[0], 1.0, 50.0)
            f1_updated, f1 = imgui.SliderFloat("Frequency Y", bump_node.f[1], 1.0, 50.0)
            f2_updated, f2 = imgui.SliderFloat("Frequency Z", bump_node.f[2], 1.0, 50.0)
            g0_updated, g0 = imgui.SliderFloat("Amplitude X", bump_node.g[0], 0.1, 10.0)
            g1_updated, g1 = imgui.SliderFloat("Amplitude Y", bump_node.g[1], 0.1, 10.0)
            g2_updated, g2 = imgui.SliderFloat("Amplitude Z", bump_node.g[2], 0.1, 10.0)
            updated = (
                f0_updated
                or f1_updated
                or f2_updated
                or g0_updated
                or g1_updated
                or g2_updated
            )
            if updated or box_updated:
                bump_node.f = np.array([f0, f1, f2])
                bump_node.g = np.array([g0, g1, g2])
                sd = bump_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Bump",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Twist UI
        if imgui.TreeNode("Twist"):
            k_updated, k = imgui.SliderFloat("Twist Rate", twist_node.k, -1.0, 1.0)
            if k_updated or box_updated:
                twist_node.k = k
                sd = twist_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Twist",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()
        # Bend UI
        if imgui.TreeNode("Bend"):
            k_updated, k = imgui.SliderFloat("Bend Rate", bend_node.k, -1.0, 1.0)
            if k_updated or box_updated:
                bend_node.k = k
                sd = bend_node.eval(X, sdf=lambda x: box.eval(x)).reshape(dims)
                grid.add_scalar_quantity(
                    "Bend",
                    sd,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                )
            imgui.TreePop()

    ps.set_user_callback(callback)
    ps.show()
