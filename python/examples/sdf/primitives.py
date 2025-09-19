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


def box_ui(grid, box, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Box"):
        hex_updated, hex = imgui.SliderFloat("Half Extent X", box.he[0], 0.1, 10.0)
        hey_updated, hey = imgui.SliderFloat("Half Extent Y", box.he[1], 0.1, 10.0)
        hez_updated, hez = imgui.SliderFloat("Half Extent Z", box.he[2], 0.1, 10.0)
        updated = hex_updated or hey_updated or hez_updated
        if updated:
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


def box_frame_ui(grid, box_frame, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Box Frame"):
        hex_updated, hex = imgui.SliderFloat(
            "Half Extent X", box_frame.he[0], 0.1, 10.0
        )
        hey_updated, hey = imgui.SliderFloat(
            "Half Extent Y", box_frame.he[1], 0.1, 10.0
        )
        hez_updated, hez = imgui.SliderFloat(
            "Half Extent Z", box_frame.he[2], 0.1, 10.0
        )
        t_updated, t = imgui.SliderFloat("Thickness", box_frame.t, 0.01, 1.0)
        updated = hex_updated or hey_updated or hez_updated or t_updated
        if updated:
            box_frame.he = np.array([hex, hey, hez])
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


def torus_ui(grid, torus, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Torus"):
        r1_updated, r1 = imgui.SliderFloat("Minor Radius", torus.t[0], 0.1, 10.0)
        r2_updated, r2 = imgui.SliderFloat("Major Radius", torus.t[1], 0.1, 5.0)
        updated = r1_updated or r2_updated
        if updated:
            torus.t = np.array([r1, r2])
            sd = torus.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Torus",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )


def capped_torus_ui(
    grid, capped_torus, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Capped Torus"):
        r1_updated, r1 = imgui.SliderFloat(
            "Minor Radius", capped_torus.sc[0], 0.1, 10.0
        )
        r2_updated, r2 = imgui.SliderFloat("Major Radius", capped_torus.sc[1], 0.1, 5.0)
        ra_updated, ra = imgui.SliderFloat("Cap Radius", capped_torus.ra, 0.1, 5.0)
        rb_updated, rb = imgui.SliderFloat("Cap Radius", capped_torus.rb, 0.1, 5.0)
        updated = r1_updated or r2_updated or ra_updated or rb_updated
        if updated:
            capped_torus.sc = np.array([r1, r2])
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


def link_ui(grid, link, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Link"):
        t1_updated, t1 = imgui.SliderFloat("Radius 1", link.t[0], 0.1, 5.0)
        t2_updated, t2 = imgui.SliderFloat("Radius 2", link.t[1], 0.1, 5.0)
        le_updated, le = imgui.SliderFloat("Length", link.le, 0.1, 10.0)
        updated = t1_updated or t2_updated or le_updated
        if updated:
            link.t = np.array([t1, t2])
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


def infinite_cylinder_ui(
    grid, cylinder, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Infinite Cylinder"):
        cx_updated, cx = imgui.SliderFloat("Center X", cylinder.c[0], -5.0, 5.0)
        cy_updated, cy = imgui.SliderFloat("Center Y", cylinder.c[1], -5.0, 5.0)
        r_updated, r = imgui.SliderFloat("Radius", cylinder.c[2], 0.1, 5.0)
        updated = cx_updated or cy_updated or r_updated
        if updated:
            cylinder.c = np.array([cx, cy, r])
            sd = cylinder.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Infinite Cylinder",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )


def cone_ui(grid, cone, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Cone"):
        s_updated, s = imgui.SliderFloat("Sin", cone.c[0], 0.0, 1.0)
        c_updated, c = imgui.SliderFloat("Cos", cone.c[1], 0.0, 1.0)
        r_updated, r = imgui.SliderFloat("Height", cone.h, 0.1, 10.0)
        updated = s_updated or c_updated or r_updated
        if updated:
            cone.c = np.array([s, c])
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


def infinite_cone_ui(grid, cone, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Infinite Cone"):
        s_updated, s = imgui.SliderFloat("Sin", cone.c[0], 0.0, 1.0)
        c_updated, c = imgui.SliderFloat("Cos", cone.c[1], 0.0, 1.0)
        updated = s_updated or c_updated
        if updated:
            cone.c = np.array([s, c])
            sd = cone.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Infinite Cone",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )


def plane_ui(grid, plane, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Plane"):
        pass


def hexagonal_prism_ui(grid, prism, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Hexagonal Prism"):
        h1_updated, h1 = imgui.SliderFloat("In-radius", prism.h[0], 0.1, 5.0)
        h2_updated, h2 = imgui.SliderFloat("Circumradius", prism.h[1], 0.1, 10.0)
        updated = h1_updated or h2_updated
        if updated:
            prism.h = np.array([h1, h2])
            sd = prism.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Hexagonal Prism",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )


def capsule_ui(grid, capsule, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Capsule"):
        ax_updated, ax = imgui.SliderFloat("X of Endpoint A", capsule.a[0], -10.0, 10.0)
        ay_updated, ay = imgui.SliderFloat("Y of Endpoint A", capsule.a[1], -10.0, 10.0)
        az_updated, az = imgui.SliderFloat("Z of Endpoint A", capsule.a[2], -10.0, 10.0)
        bx_updated, bx = imgui.SliderFloat("X of Endpoint B", capsule.b[0], -10.0, 10.0)
        by_updated, by = imgui.SliderFloat("Y of Endpoint B", capsule.b[1], -10.0, 10.0)
        bz_updated, bz = imgui.SliderFloat("Z of Endpoint B", capsule.b[2], -10.0, 10.0)
        r_updated, r = imgui.SliderFloat("Radius", capsule.r, 0.1, 5.0)
        updated = (
            ax_updated
            or ay_updated
            or az_updated
            or bx_updated
            or by_updated
            or bz_updated
            or r_updated
        )
        if updated:
            capsule.a = np.array([ax, ay, az])
            capsule.b = np.array([bx, by, bz])
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


def capped_cylinder_ui(
    grid, capped_cylinder, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Capped Cylinder"):
        ax_updated, ax = imgui.SliderFloat(
            "X of Endpoint A", capped_cylinder.a[0], -10.0, 10.0
        )
        ay_updated, ay = imgui.SliderFloat(
            "Y of Endpoint A", capped_cylinder.a[1], -10.0, 10.0
        )
        az_updated, az = imgui.SliderFloat(
            "Z of Endpoint A", capped_cylinder.a[2], -10.0, 10.0
        )
        bx_updated, bx = imgui.SliderFloat(
            "X of Endpoint B", capped_cylinder.b[0], -10.0, 10.0
        )
        by_updated, by = imgui.SliderFloat(
            "Y of Endpoint B", capped_cylinder.b[1], -10.0, 10.0
        )
        bz_updated, bz = imgui.SliderFloat(
            "Z of Endpoint B", capped_cylinder.b[2], -10.0, 10.0
        )
        r_updated, r = imgui.SliderFloat("Radius", capped_cylinder.r, 0.1, 5.0)
        updated = (
            ax_updated
            or ay_updated
            or az_updated
            or bx_updated
            or by_updated
            or bz_updated
            or r_updated
        )
        if updated:
            capped_cylinder.a = np.array([ax, ay, az])
            capped_cylinder.b = np.array([bx, by, bz])
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


def triangle_ui(grid, triangle, X, dims, cmap, vminmax=(-10, 10), isolines=True):
    if imgui.TreeNode("Triangle"):
        ax_updated, ax = imgui.SliderFloat("X of Vertex A", triangle.a[0], -10.0, 10.0)
        ay_updated, ay = imgui.SliderFloat("Y of Vertex A", triangle.a[1], -10.0, 10.0)
        az_updated, az = imgui.SliderFloat("Z of Vertex A", triangle.a[2], -10.0, 10.0)
        bx_updated, bx = imgui.SliderFloat("X of Vertex B", triangle.b[0], -10.0, 10.0)
        by_updated, by = imgui.SliderFloat("Y of Vertex B", triangle.b[1], -10.0, 10.0)
        bz_updated, bz = imgui.SliderFloat("Z of Vertex B", triangle.b[2], -10.0, 10.0)
        cx_updated, cx = imgui.SliderFloat("X of Vertex C", triangle.c[0], -10.0, 10.0)
        cy_updated, cy = imgui.SliderFloat("Y of Vertex C", triangle.c[1], -10.0, 10.0)
        cz_updated, cz = imgui.SliderFloat("Z of Vertex C", triangle.c[2], -10.0, 10.0)
        updated = (
            ax_updated
            or ay_updated
            or az_updated
            or bx_updated
            or by_updated
            or bz_updated
            or cx_updated
            or cy_updated
            or cz_updated
        )
        if updated:
            triangle.a = np.array([ax, ay, az])
            triangle.b = np.array([bx, by, bz])
            triangle.c = np.array([cx, cy, cz])
            sd = triangle.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Triangle",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )


def quadrilateral_ui(
    grid, quadrilateral, X, dims, cmap, vminmax=(-10, 10), isolines=True
):
    if imgui.TreeNode("Quadrilateral"):
        ax_updated, ax = imgui.SliderFloat(
            "X of Vertex A", quadrilateral.a[0], -10.0, 10.0
        )
        ay_updated, ay = imgui.SliderFloat(
            "Y of Vertex A", quadrilateral.a[1], -10.0, 10.0
        )
        az_updated, az = imgui.SliderFloat(
            "Z of Vertex A", quadrilateral.a[2], -10.0, 10.0
        )
        bx_updated, bx = imgui.SliderFloat(
            "X of Vertex B", quadrilateral.b[0], -10.0, 10.0
        )
        by_updated, by = imgui.SliderFloat(
            "Y of Vertex B", quadrilateral.b[1], -10.0, 10.0
        )
        bz_updated, bz = imgui.SliderFloat(
            "Z of Vertex B", quadrilateral.b[2], -10.0, 10.0
        )
        cx_updated, cx = imgui.SliderFloat(
            "X of Vertex C", quadrilateral.c[0], -10.0, 10.0
        )
        cy_updated, cy = imgui.SliderFloat(
            "Y of Vertex C", quadrilateral.c[1], -10.0, 10.0
        )
        cz_updated, cz = imgui.SliderFloat(
            "Z of Vertex C", quadrilateral.c[2], -10.0, 10.0
        )
        dx_updated, dx = imgui.SliderFloat(
            "X of Vertex D", quadrilateral.d[0], -10.0, 10.0
        )
        dy_updated, dy = imgui.SliderFloat(
            "Y of Vertex D", quadrilateral.d[1], -10.0, 10.0
        )
        dz_updated, dz = imgui.SliderFloat(
            "Z of Vertex D", quadrilateral.d[2], -10.0, 10.0
        )
        updated = (
            ax_updated
            or ay_updated
            or az_updated
            or bx_updated
            or by_updated
            or bz_updated
            or cx_updated
            or cy_updated
            or cz_updated
            or dx_updated
            or dy_updated
            or dz_updated
        )
        if updated:
            quadrilateral.a = np.array([ax, ay, az])
            quadrilateral.b = np.array([bx, by, bz])
            quadrilateral.c = np.array([cx, cy, cz])
            quadrilateral.d = np.array([dx, dy, dz])
            sd = quadrilateral.eval(X).reshape(dims)
            grid.add_scalar_quantity(
                "Quadrilateral",
                sd,
                defined_on="nodes",
                cmap=cmap,
                vminmax=vminmax,
                isolines_enabled=isolines,
            )


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

    torus = pbat.geometry.sdf.Torus(t=np.array([1.0, 3.0]))
    sd_torus = torus.eval(X).reshape(dims)

    capped_torus = pbat.geometry.sdf.CappedTorus(
        sc=np.array([1.0, 3.0]), ra=1.0, rb=1.0
    )
    sd_capped_torus = capped_torus.eval(X).reshape(dims)

    link = pbat.geometry.sdf.Link(t=np.array([1.0, 2.0]), le=5.0)
    sd_link = link.eval(X).reshape(dims)

    infinite_cylinder = pbat.geometry.sdf.InfiniteCylinder(c=np.array([0.0, 0.0, 1.0]))
    sd_infinite_cylinder = infinite_cylinder.eval(X).reshape(dims)

    cone = pbat.geometry.sdf.Cone(c=np.array([0.5, 0.5]), h=5.0)
    sd_cone = cone.eval(X).reshape(dims)

    infinite_cone = pbat.geometry.sdf.InfiniteCone(c=np.array([0.5, 0.5]))
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
