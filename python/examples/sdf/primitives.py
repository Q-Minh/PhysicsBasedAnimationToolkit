import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np


def sphere_ui(grid, sphere, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Sphere"):
        updated, sphere.R = imgui.SliderFloat("Radius", sphere.R, 0.1, 10.0)
        if updated:
            sd = sphere.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Sphere",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def box_ui(grid, box, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Box"):
        he_updated, he = imgui.SliderFloat3("Half Extents", box.he, 0.1, 10.0)
        updated = he_updated
        if updated:
            box.he = np.array(he)
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


def box_frame_ui(grid, box_frame, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Box Frame"):
        he_updated, he = imgui.SliderFloat3("Half Extents", box_frame.he, 0.1, 10.0)
        t_updated, t = imgui.SliderFloat("Thickness", box_frame.t, 0.01, 1.0)
        updated = he_updated or t_updated
        if updated:
            box_frame.he = np.array(he)
            box_frame.t = t
            sd = box_frame.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Box Frame",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def torus_ui(grid, torus, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Torus"):
        r_updated, r = imgui.SliderFloat2("Radii", torus.t, 0.1, 10.0)
        updated = r_updated
        if updated:
            torus.t = np.array(r)
            sd = torus.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Torus",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def capped_torus_ui(
    grid, capped_torus, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Capped Torus"):
        sc_updated, sc = imgui.SliderFloat2("Sin/Cos", capped_torus.sc, -1.0, 1.0)
        ra_updated, ra = imgui.SliderFloat("Radius 1", capped_torus.ra, 0.01, 10.0)
        rb_updated, rb = imgui.SliderFloat("Radius 2", capped_torus.rb, 0.01, 10.0)
        updated = sc_updated or ra_updated or rb_updated
        if updated:
            capped_torus.sc = np.array(sc)
            capped_torus.ra = ra
            capped_torus.rb = rb
            sd = capped_torus.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Capped Torus",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def link_ui(grid, link, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Link"):
        t_updated, t = imgui.SliderFloat2("Radii", link.t, 0.1, 10.0)
        le_updated, le = imgui.SliderFloat("Length", link.le, 0.1, 10.0)
        updated = t_updated or le_updated
        if updated:
            link.t = np.array(t)
            link.le = le
            sd = link.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Link",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def infinite_cylinder_ui(
    grid, cylinder, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Infinite Cylinder"):
        c_updated, c = imgui.SliderFloat2("Center", cylinder.c[:2], -10.0, 10.0)
        r_updated, r = imgui.SliderFloat("Radius", cylinder.c[2], 0.1, 10.0)
        updated = c_updated or r_updated
        if updated:
            cylinder.c = np.array([c[0], c[1], r])
            sd = cylinder.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Infinite Cylinder",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def cone_ui(grid, cone, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Cone"):
        sc_updated, sc = imgui.SliderFloat2("Sin/Cos", cone.sc, -1.0, 1.0)
        r_updated, r = imgui.SliderFloat("Height", cone.h, 0.1, 10.0)
        updated = sc_updated or r_updated
        if updated:
            cone.sc = np.array(sc)
            cone.h = r
            sd = cone.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Cone",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def infinite_cone_ui(grid, cone, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Infinite Cone"):
        sc_updated, sc = imgui.SliderFloat2("Sin/Cos", cone.sc, -1.0, 1.0)
        updated = sc_updated
        if updated:
            cone.sc = np.array(sc)
            sd = cone.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Infinite Cone",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def plane_ui(grid, plane, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Plane"):
        imgui.TreePop()


def hexagonal_prism_ui(grid, prism, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Hexagonal Prism"):
        h_updated, h = imgui.SliderFloat2("Radii", prism.h, 0.1, 10.0)
        updated = h_updated
        if updated:
            prism.h = np.array(h)
            sd = prism.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Hexagonal Prism",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def capsule_ui(grid, capsule, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Capsule"):
        a_updated, a = imgui.SliderFloat3("Endpoint A", capsule.a, -10.0, 10.0)
        b_updated, b = imgui.SliderFloat3("Endpoint B", capsule.b, -10.0, 10.0)
        r_updated, r = imgui.SliderFloat("Radius", capsule.r, 0.1, 5.0)
        updated = a_updated or b_updated or r_updated
        if updated:
            capsule.a = np.array(a)
            capsule.b = np.array(b)
            capsule.r = r
            sd = capsule.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Capsule",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def vertical_capsule_ui(
    grid, vertical_capsule, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Vertical Capsule"):
        h_updated, h = imgui.SliderFloat("Height", vertical_capsule.h, 0.1, 20.0)
        r_updated, r = imgui.SliderFloat("Radius", vertical_capsule.r, 0.1, 5.0)
        updated = h_updated or r_updated
        if updated:
            vertical_capsule.h = h
            vertical_capsule.r = r
            sd = vertical_capsule.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Vertical Capsule",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def capped_cylinder_ui(
    grid, capped_cylinder, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Capped Cylinder"):
        a_updated, a = imgui.SliderFloat3("Endpoint A", capped_cylinder.a, -10.0, 10.0)
        b_updated, b = imgui.SliderFloat3("Endpoint B", capped_cylinder.b, -10.0, 10.0)
        r_updated, r = imgui.SliderFloat("Radius", capped_cylinder.r, 0.1, 5.0)
        updated = a_updated or b_updated or r_updated
        if updated:
            capped_cylinder.a = np.array(a)
            capped_cylinder.b = np.array(b)
            capped_cylinder.r = r
            sd = capped_cylinder.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Capped Cylinder",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def vertical_capped_cylinder_ui(
    grid, vertical_capped_cylinder, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Vertical Capped Cylinder"):
        h_updated, h = imgui.SliderFloat(
            "Height", vertical_capped_cylinder.h, 0.1, 20.0
        )
        r_updated, r = imgui.SliderFloat("Radius", vertical_capped_cylinder.r, 0.1, 5.0)
        updated = h_updated or r_updated
        if updated:
            vertical_capped_cylinder.h = h
            vertical_capped_cylinder.r = r
            sd = vertical_capped_cylinder.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Vertical Capped Cylinder",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def rounded_cylinder_ui(
    grid, rounded_cylinder, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Rounded Cylinder"):
        h_updated, h = imgui.SliderFloat("Height", rounded_cylinder.h, 0.1, 10.0)
        ra_updated, ra = imgui.SliderFloat("Radius", rounded_cylinder.ra, 0.1, 5.0)
        rb_updated, rb = imgui.SliderFloat(
            "Corner Radius", rounded_cylinder.rb, 0.01, 5.0
        )
        updated = h_updated or ra_updated or rb_updated
        if updated:
            rounded_cylinder.h = h
            rounded_cylinder.ra = ra
            rounded_cylinder.rb = rb
            sd = rounded_cylinder.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Rounded Cylinder",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def vertical_capped_cone_ui(
    grid, vertical_capped_cone, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Vertical Capped Cone"):
        h_updated, h = imgui.SliderFloat("Height", vertical_capped_cone.h, 0.1, 20.0)
        r1_updated, r1 = imgui.SliderFloat(
            "Bottom Radius", vertical_capped_cone.r1, 0.1, 5.0
        )
        r2_updated, r2 = imgui.SliderFloat(
            "Top Radius", vertical_capped_cone.r2, 0.1, 5.0
        )
        updated = h_updated or r1_updated or r2_updated
        if updated:
            vertical_capped_cone.h = h
            vertical_capped_cone.r1 = r1
            vertical_capped_cone.r2 = r2
            sd = vertical_capped_cone.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Vertical Capped Cone",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def cut_hollow_sphere_ui(
    grid, cut_hollow_sphere, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Cut Hollow Sphere"):
        r_updated, r = imgui.SliderFloat("Radius", cut_hollow_sphere.r, 0.1, 5.0)
        h_updated, h = imgui.SliderFloat("Height", cut_hollow_sphere.h, 0.1, 10.0)
        t_updated, t = imgui.SliderFloat("Thickness", cut_hollow_sphere.t, 0.01, 5.0)
        updated = r_updated or h_updated or t_updated
        if updated:
            cut_hollow_sphere.r = r
            cut_hollow_sphere.h = h
            cut_hollow_sphere.t = t
            sd = cut_hollow_sphere.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Cut Hollow Sphere",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def vertical_round_cone_ui(
    grid, vertical_round_cone, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Vertical Rounded Cone"):
        h_updated, h = imgui.SliderFloat("Height", vertical_round_cone.h, 0.1, 20.0)
        r1_updated, r1 = imgui.SliderFloat(
            "Bottom Radius", vertical_round_cone.r1, 0.1, 5.0
        )
        r2_updated, r2 = imgui.SliderFloat(
            "Top Radius", vertical_round_cone.r2, 0.1, 5.0
        )
        updated = h_updated or r1_updated or r2_updated
        if updated:
            vertical_round_cone.h = h
            vertical_round_cone.r1 = r1
            vertical_round_cone.r2 = r2
            sd = vertical_round_cone.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Vertical Rounded Cone",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def octahedron_ui(grid, octahedron, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Octahedron"):
        s_updated, s = imgui.SliderFloat("Size", octahedron.s, 0.1, 10.0)
        updated = s_updated
        if updated:
            octahedron.s = s
            sd = octahedron.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Octahedron",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def pyramid_ui(grid, pyramid, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Pyramid"):
        h_updated, h = imgui.SliderFloat("Height", pyramid.h, 0.1, 10.0)
        updated = h_updated
        if updated:
            pyramid.h = h
            sd = pyramid.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Pyramid",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def triangle_ui(grid, triangle, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Triangle"):
        a_updated, a = imgui.SliderFloat3("Vertex A", triangle.a, -10.0, 10.0)
        b_updated, b = imgui.SliderFloat3("Vertex B", triangle.b, -10.0, 10.0)
        c_updated, c = imgui.SliderFloat3("Vertex C", triangle.c, -10.0, 10.0)
        updated = a_updated or b_updated or c_updated
        if updated:
            triangle.a = np.array(a)
            triangle.b = np.array(b)
            triangle.c = np.array(c)
            sd = triangle.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Triangle",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


def quadrilateral_ui(
    grid, quadrilateral, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Quadrilateral"):
        a_updated, a = imgui.SliderFloat3("Vertex A", quadrilateral.a, -10.0, 10.0)
        b_updated, b = imgui.SliderFloat3("Vertex B", quadrilateral.b, -10.0, 10.0)
        c_updated, c = imgui.SliderFloat3("Vertex C", quadrilateral.c, -10.0, 10.0)
        d_updated, d = imgui.SliderFloat3("Vertex D", quadrilateral.d, -10.0, 10.0)
        updated = a_updated or b_updated or c_updated or d_updated
        if updated:
            quadrilateral.a = np.array(a)
            quadrilateral.b = np.array(b)
            quadrilateral.c = np.array(c)
            quadrilateral.d = np.array(d)
            sd = quadrilateral.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Quadrilateral",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )
        imgui.TreePop()


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
    sphere = pbat.geometry.sdf.Sphere(R=1)
    sd_sphere = sphere.eval(X).reshape(dims)

    box = pbat.geometry.sdf.Box(he=np.array([1, 1, 1]))
    sd_box = box.eval(X).reshape(dims)

    box_frame = pbat.geometry.sdf.BoxFrame(he=np.array([1.0, 1.0, 1.0]), t=0.1)
    sd_box_frame = box_frame.eval(X).reshape(dims)

    torus = pbat.geometry.sdf.Torus(t=np.array([5.0, 2.0]))
    sd_torus = torus.eval(X).reshape(dims)

    capped_torus = pbat.geometry.sdf.CappedTorus(
        sc=np.array([0.833, -0.545]), ra=5.0, rb=1.0
    )
    sd_capped_torus = capped_torus.eval(X).reshape(dims)

    link = pbat.geometry.sdf.Link(t=np.array([1.0, 2.0]), le=5.0)
    sd_link = link.eval(X).reshape(dims)

    infinite_cylinder = pbat.geometry.sdf.InfiniteCylinder(c=np.array([0.0, 0.0, 1.0]))
    sd_infinite_cylinder = infinite_cylinder.eval(X).reshape(dims)

    cone = pbat.geometry.sdf.Cone(sc=np.array([0.5, 0.5]), h=5.0)
    sd_cone = cone.eval(X).reshape(dims)

    infinite_cone = pbat.geometry.sdf.InfiniteCone(sc=np.array([0.5, 0.5]))
    sd_infinite_cone = infinite_cone.eval(X).reshape(dims)

    plane = pbat.geometry.sdf.Plane()
    sd_plane = plane.eval(X).reshape(dims)

    hexagonal_prism = pbat.geometry.sdf.HexagonalPrism(h=np.array([1.0, 2.0]))
    sd_hexagonal_prism = hexagonal_prism.eval(X).reshape(dims)

    capsule = pbat.geometry.sdf.Capsule(
        a=np.array([-5.0, 0.0, 0.0]), b=np.array([5.0, 0.0, 0.0]), r=1.0
    )
    sd_capsule = capsule.eval(X).reshape(dims)

    vertical_capsule = pbat.geometry.sdf.VerticalCapsule(h=5.0, r=1.0)
    sd_vertical_capsule = vertical_capsule.eval(X).reshape(dims)

    capped_cylinder = pbat.geometry.sdf.CappedCylinder(
        a=np.array([-5.0, 0.0, 0.0]), b=np.array([5.0, 0.0, 0.0]), r=1.0
    )
    sd_capped_cylinder = capped_cylinder.eval(X).reshape(dims)

    vertical_capped_cylinder = pbat.geometry.sdf.VerticalCappedCylinder(h=5.0, r=1.0)
    sd_vertical_capped_cylinder = vertical_capped_cylinder.eval(X).reshape(dims)

    rounded_cylinder = pbat.geometry.sdf.RoundedCylinder(h=5.0, ra=1.0, rb=0.2)
    sd_rounded_cylinder = rounded_cylinder.eval(X).reshape(dims)

    vertical_capped_cone = pbat.geometry.sdf.VerticalCappedCone(h=5.0, r1=2.0, r2=1.0)
    sd_vertical_capped_cone = vertical_capped_cone.eval(X).reshape(dims)

    cut_hollow_sphere = pbat.geometry.sdf.CutHollowSphere(h=1.0, r=2.0, t=0.2)
    sd_cut_hollow_sphere = cut_hollow_sphere.eval(X).reshape(dims)

    vertical_round_cone = pbat.geometry.sdf.VerticalRoundCone(h=5.0, r1=2.0, r2=1.0)
    sd_vertical_round_cone = vertical_round_cone.eval(X).reshape(dims)

    octahedron = pbat.geometry.sdf.Octahedron(s=2.0)
    sd_octahedron = octahedron.eval(X).reshape(dims)

    pyramid = pbat.geometry.sdf.Pyramid(h=2.0)
    sd_pyramid = pyramid.eval(X).reshape(dims)

    triangle = pbat.geometry.sdf.Triangle(
        a=np.array([-1.0, 0.0, 0.0]),
        b=np.array([1.0, 0.0, 0.0]),
        c=np.array([0.0, 0.0, 2.0]),
    )
    sd_triangle = triangle.eval(X).reshape(dims)

    quadrilateral = pbat.geometry.sdf.Quadrilateral(
        a=np.array([-1.0, 0.0, -1.0]),
        b=np.array([1.0, 0.0, -1.0]),
        c=np.array([1.0, 0.0, 1.0]),
        d=np.array([-1.0, 0.0, 1.0]),
    )
    sd_quadrilateral = quadrilateral.eval(X).reshape(dims)

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
        "Sphere",
        sd_sphere,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Box",
        sd_box,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Box Frame",
        sd_box_frame,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Torus",
        sd_torus,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Capped Torus",
        sd_capped_torus,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Link",
        sd_link,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Infinite Cylinder",
        sd_infinite_cylinder,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Cone",
        sd_cone,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Infinite Cone",
        sd_infinite_cone,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Plane",
        sd_plane,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Hexagonal Prism",
        sd_hexagonal_prism,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Capsule",
        sd_capsule,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Vertical Capsule",
        sd_vertical_capsule,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Capped Cylinder",
        sd_capped_cylinder,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Vertical Capped Cylinder",
        sd_vertical_capped_cylinder,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Rounded Cylinder",
        sd_rounded_cylinder,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Vertical Capped Cone",
        sd_vertical_capped_cone,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Cut Hollow Sphere",
        sd_cut_hollow_sphere,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Vertical Rounded Cone",
        sd_vertical_round_cone,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Octahedron",
        sd_octahedron,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Pyramid",
        sd_pyramid,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Triangle",
        sd_triangle,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )
    grid.add_scalar_quantity(
        "Quadrilateral",
        sd_quadrilateral,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
    )

    def callback():
        sphere_ui(grid, sphere, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        box_ui(grid, box, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        box_frame_ui(grid, box_frame, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        torus_ui(grid, torus, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        capped_torus_ui(
            grid, capped_torus, X, dims, cmap, vminmax=vminmax, isolines=isolines
        )
        link_ui(grid, link, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        infinite_cylinder_ui(
            grid,
            infinite_cylinder,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        cone_ui(grid, cone, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        infinite_cone_ui(
            grid,
            infinite_cone,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        plane_ui(grid, plane, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        hexagonal_prism_ui(
            grid, hexagonal_prism, X, dims, cmap, vminmax=vminmax, isolines=isolines
        )
        capsule_ui(grid, capsule, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        vertical_capsule_ui(
            grid,
            vertical_capsule,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        capped_cylinder_ui(
            grid,
            capped_cylinder,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        vertical_capped_cylinder_ui(
            grid,
            vertical_capped_cylinder,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        rounded_cylinder_ui(
            grid,
            rounded_cylinder,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        vertical_capped_cone_ui(
            grid,
            vertical_capped_cone,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        cut_hollow_sphere_ui(
            grid,
            cut_hollow_sphere,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        vertical_round_cone_ui(
            grid,
            vertical_round_cone,
            X,
            dims,
            cmap,
            vminmax=vminmax,
            isolines=isolines,
        )
        octahedron_ui(
            grid, octahedron, X, dims, cmap, vminmax=vminmax, isolines=isolines
        )
        pyramid_ui(grid, pyramid, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        triangle_ui(grid, triangle, X, dims, cmap, vminmax=vminmax, isolines=isolines)
        quadrilateral_ui(
            grid, quadrilateral, X, dims, cmap, vminmax=vminmax, isolines=isolines
        )

    ps.set_user_callback(callback)
    ps.show()
