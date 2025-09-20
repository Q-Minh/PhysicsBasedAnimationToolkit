from typing import Tuple
import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import scipy as sp


def node_ui(id, nodes, transforms, children, visited) -> Tuple[bool, int, bool]:
    node = nodes[id]
    transform = transforms[id]
    deleted_id = -1
    dirty = False
    descendant_updated = False
    if visited[id]:
        return dirty, deleted_id, descendant_updated
    visited[id] = True
    # Visit
    imgui.PushID(id)
    if imgui.TreeNode("{} - {}".format(id, type(node).__name__)):
        # Primitives
        if isinstance(node, pbat.geometry.sdf.Sphere):
            r_updated, r = imgui.SliderFloat("Radius", node.R, 0.1, 10.0)
            if r_updated:
                node.R = r
            dirty = r_updated
        elif isinstance(node, pbat.geometry.sdf.Box):
            he_updated, he = imgui.SliderFloat3("Half Extent", node.he, 0.1, 10.0)
            if he_updated:
                node.he = np.array(he)
            dirty = he_updated
        elif isinstance(node, pbat.geometry.sdf.BoxFrame):
            he_updated, he = imgui.SliderFloat3("Half Extent", node.he, 0.1, 10.0)
            t_updated, t = imgui.SliderFloat("Thickness", node.t, 0.01, 1.0)
            updated = he_updated or t_updated
            if updated:
                node.he = np.array(he)
                node.t = t
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Torus):
            t_updated, t = imgui.SliderFloat2("Radii", node.t, 0.1, 10.0)
            updated = t_updated
            if updated:
                node.t = np.array(t)
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.CappedTorus):
            sc_updated, sc = imgui.SliderFloat2("Radii", node.sc, 0.1, 10.0)
            ra_updated, ra = imgui.SliderFloat("Cap Radius a", node.ra, 0.1, 10.0)
            rb_updated, rb = imgui.SliderFloat("Cap Radius b", node.rb, 0.1, 10.0)
            updated = sc_updated or ra_updated or rb_updated
            if updated:
                node.sc = np.array(sc)
                node.ra = ra
                node.rb = rb
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Link):
            t_updated, t = imgui.SliderFloat2("Radii", node.t, 0.1, 10.0)
            le_updated, le = imgui.SliderFloat("Length", node.le, 0.1, 10.0)
            updated = t_updated or le_updated
            if updated:
                node.t = np.array(t)
                node.le = le
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.InfiniteCylinder):
            c_updated, c = imgui.SliderFloat2("Center", node.c[:2], -10.0, 10.0)
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 10.0)
            updated = c_updated or r_updated
            if r_updated:
                node.c = np.array([c[0], c[1], r])
            dirty = r_updated
        elif isinstance(node, pbat.geometry.sdf.Cone):
            sc_updated, sc = imgui.SliderFloat2("Sin/Cos", node.c[:2], 0.0, 1.0)
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            updated = h_updated or sc_updated
            if updated:
                node.c = np.array(sc)
                node.h = h
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.InfiniteCone):
            sc_updated, sc = imgui.SliderFloat2("Sin/Cos", node.c[:2], 0.0, 1.0)
            updated = sc_updated
            if updated:
                node.c = np.array(sc)
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Plane):
            pass
        elif isinstance(node, pbat.geometry.sdf.HexagonalPrism):
            h_updated, h = imgui.SliderFloat2("Radii", node.h, 0.1, 10.0)
            updated = h_updated
            if updated:
                node.h = np.array(h)
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Capsule):
            a_updated, a = imgui.SliderFloat3("Endpoint A", node.a, -10.0, 10.0)
            b_updated, b = imgui.SliderFloat3("Endpoint B", node.b, -10.0, 10.0)
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 10.0)
            updated = a_updated or b_updated or r_updated
            if updated:
                node.a = np.array(a)
                node.b = np.array(b)
                node.r = r
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.VerticalCapsule):
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 10.0)
            updated = h_updated or r_updated
            if updated:
                node.h = h
                node.r = r
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.CappedCylinder):
            a_updated, a = imgui.SliderFloat3("Endpoint A", node.a, -10.0, 10.0)
            b_updated, b = imgui.SliderFloat3("Endpoint B", node.b, -10.0, 10.0)
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 10.0)
            updated = a_updated or b_updated or r_updated
            if updated:
                node.a = np.array(a)
                node.b = np.array(b)
                node.r = r
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.VerticalCappedCylinder):
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 10.0)
            updated = h_updated or r_updated
            if updated:
                node.h = h
                node.r = r
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.RoundedCylinder):
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 10.0)
            cr_updated, cr = imgui.SliderFloat("Corner Radius", node.cr, 0.01, 1.0)
            updated = h_updated or r_updated or cr_updated
            if updated:
                node.h = h
                node.ra = r
                node.rb = cr
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.VerticalCappedCone):
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            r1_updated, r1 = imgui.SliderFloat("Bottom Radius", node.r1, 0.1, 10.0)
            r2_updated, r2 = imgui.SliderFloat("Top Radius", node.r2, 0.1, 10.0)
            updated = h_updated or r1_updated or r2_updated
            if updated:
                node.h = h
                node.r1 = r1
                node.r2 = r2
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.CutHollowSphere):
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.1, 5.0)
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            t_updated, t = imgui.SliderFloat("Thickness", node.t, 0.01, 5.0)
            updated = r_updated or h_updated or t_updated
            if updated:
                node.r = r
                node.h = h
                node.t = t
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.VerticalRoundCone):
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 20.0)
            r1_updated, r1 = imgui.SliderFloat("Bottom Radius", node.r1, 0.1, 5.0)
            r2_updated, r2 = imgui.SliderFloat("Top Radius", node.r2, 0.1, 5.0)
            updated = h_updated or r1_updated or r2_updated
            if updated:
                node.h = h
                node.r1 = r1
                node.r2 = r2
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Octahedron):
            s_updated, s = imgui.SliderFloat("Radius", node.s, 0.1, 10.0)
            if s_updated:
                node.s = s
            dirty = s_updated
        elif isinstance(node, pbat.geometry.sdf.Pyramid):
            h_updated, h = imgui.SliderFloat("Height", node.h, 0.1, 10.0)
            updated = h_updated
            if updated:
                node.h = h
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Triangle):
            a_updated, a = imgui.SliderFloat3("Vertex A", node.a, -10.0, 10.0)
            b_updated, b = imgui.SliderFloat3("Vertex B", node.b, -10.0, 10.0)
            c_updated, c = imgui.SliderFloat3("Vertex C", node.c, -10.0, 10.0)
            updated = a_updated or b_updated or c_updated
            if updated:
                node.a = np.array(a)
                node.b = np.array(b)
                node.c = np.array(c)
            dirty = updated
        elif isinstance(node, pbat.geometry.sdf.Quadrilateral):
            a_updated, a = imgui.SliderFloat3("Vertex A", node.a, -10.0, 10.0)
            b_updated, b = imgui.SliderFloat3("Vertex B", node.b, -10.0, 10.0)
            c_updated, c = imgui.SliderFloat3("Vertex C", node.c, -10.0, 10.0)
            d_updated, d = imgui.SliderFloat3("Vertex D", node.d, -10.0, 10.0)
            updated = a_updated or b_updated or c_updated or d_updated
            if updated:
                node.a = np.array(a)
                node.b = np.array(b)
                node.c = np.array(c)
                node.d = np.array(d)
            dirty = updated

        # Unary nodes
        is_unary_node = False
        if isinstance(node, pbat.geometry.sdf.Scale):
            s_updated, s = imgui.SliderFloat("Scale", node.s, 0.01, 5.0)
            if s_updated:
                node.s = s
            dirty = s_updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Elongate):
            h_updated, h = imgui.SliderFloat3("Length", node.h, 0.1, 10.0)
            if h_updated:
                node.h = np.array(h)
            dirty = h_updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Round):
            r_updated, r = imgui.SliderFloat("Radius", node.r, 0.01, 1.0)
            if r_updated:
                node.r = r
            dirty = r_updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Onion):
            t_updated, t = imgui.SliderFloat("Thickness", node.t, 0.01, 1.0)
            if t_updated:
                node.t = t
            dirty = t_updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Symmetrize):
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Repeat):
            s_updated, s = imgui.SliderFloat("Scale", node.s, 0.01, 10.0)
            l_updated, l = imgui.SliderFloat3("Extents", node.l, 0.01, 10.0)
            updated = s_updated or l_updated
            if updated:
                node.s = s
                node.l = np.array(l)
            dirty = updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Bump):
            a_updated, a = imgui.SliderFloat3("Amplitude", node.g, 0.0, 1.0)
            f_updated, f = imgui.SliderFloat3("Frequency", node.f, 0.0, 10.0)
            updated = a_updated or f_updated
            if updated:
                node.g = np.array(a)
                node.f = np.array(f)
            dirty = updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Twist):
            k_updated, k = imgui.SliderFloat("Twist", node.k, -0.5, 0.5)
            if k_updated:
                node.k = k
            dirty = k_updated
            is_unary_node = True
        elif isinstance(node, pbat.geometry.sdf.Bend):
            k_updated, k = imgui.SliderFloat("Curvature", node.k, -0.5, 0.5)
            if k_updated:
                node.k = k
            dirty = k_updated
            is_unary_node = True

        # Binary nodes
        is_binary_node = False
        if isinstance(node, pbat.geometry.sdf.Union):
            is_binary_node = True
        elif isinstance(node, pbat.geometry.sdf.Difference):
            is_binary_node = True
        elif isinstance(node, pbat.geometry.sdf.Intersection):
            is_binary_node = True
        elif isinstance(node, pbat.geometry.sdf.ExclusiveOr):
            is_binary_node = True
        elif isinstance(node, pbat.geometry.sdf.SmoothUnion):
            k_updated, k = imgui.SliderFloat("Smoothness", node.k, 0.0, 10.0)
            if k_updated:
                node.k = k
            dirty = k_updated
            is_binary_node = True
        elif isinstance(node, pbat.geometry.sdf.SmoothDifference):
            k_updated, k = imgui.SliderFloat("Smoothness", node.k, 0.0, 10.0)
            if k_updated:
                node.k = k
            dirty = k_updated
            is_binary_node = True
        elif isinstance(node, pbat.geometry.sdf.SmoothIntersection):
            k_updated, k = imgui.SliderFloat("Smoothness", node.k, 0.0, 10.0)
            if k_updated:
                node.k = k
            dirty = k_updated
            is_binary_node = True

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

        # Children
        ci, cj = children[id]
        if is_unary_node:
            unary_node_child_selection_requested = imgui.BeginCombo(
                "Child", str(children[id][0])
            )
            if unary_node_child_selection_requested:
                for i in range(len(nodes)):
                    name = type(nodes[i]).__name__
                    _, selected = imgui.Selectable(name, i == children[id][0])
                    if selected:
                        ci = i
                imgui.EndCombo()
        if is_binary_node:
            binary_node_left_child_selection_requested = imgui.BeginCombo(
                "Left Child", str(children[id][0])
            )
            if binary_node_left_child_selection_requested:
                for i in range(len(nodes)):
                    name = "{} - {}".format(i, type(nodes[i]).__name__)
                    _, selected = imgui.Selectable(name, i == ci)
                    if selected:
                        ci = i
                imgui.EndCombo()
            binary_node_right_child_selection_requested = imgui.BeginCombo(
                "Right Child", str(children[id][1])
            )
            if binary_node_right_child_selection_requested:
                for j in range(len(nodes)):
                    name = "{} - {}".format(j, type(nodes[j]).__name__)
                    _, selected = imgui.Selectable(name, j == cj)
                    if selected:
                        cj = j
                imgui.EndCombo()

        if is_unary_node:
            descendant_updated = ci != children[id][0]
            dirty |= descendant_updated
        if is_binary_node:
            descendant_updated = (ci != children[id][0]) or (cj != children[id][1])
            dirty |= descendant_updated

        # Deletion
        if imgui.Button("Delete"):
            deleted_id = id

        # Recurse
        c0, c1 = children[id]
        if c0 >= 0:
            c0_dirty, c0_deleted_id, c0_children_updated = node_ui(
                c0, nodes, transforms, children, visited
            )
            dirty |= c0_dirty
            descendant_updated |= c0_children_updated
            if c0_deleted_id != -1:
                deleted_id = c0_deleted_id
        if c1 >= 0:
            c1_dirty, c1_deleted_id, c1_children_updated = node_ui(
                c1, nodes, transforms, children, visited
            )
            dirty |= c1_dirty
            descendant_updated |= c1_children_updated
            if c1_deleted_id != -1:
                deleted_id = c1_deleted_id

        children[id] = (ci, cj)
        imgui.TreePop()
    imgui.PopID()
    return dirty, deleted_id, descendant_updated


if __name__ == "__main__":
    # Domain
    bmin = np.array([-10, -10, -10])
    bmax = np.array([10, 10, 10])
    dims = (100, 100, 100)
    x, y, z = np.meshgrid(
        np.linspace(bmin[0], bmax[0], dims[0]),
        np.linspace(bmin[1], bmax[1], dims[1]),
        np.linspace(bmin[2], bmax[2], dims[2]),
    )
    X = np.vstack([np.ravel(x), np.ravel(y), np.ravel(z)]).astype(np.float64)

    # Composite
    nodes = []
    children = []
    transforms = []
    roots = []
    forest = pbat.geometry.sdf.Forest()
    composite = pbat.geometry.sdf.Composite(forest)
    sd_composite = np.ones(dims)

    # Polyscope visualization
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("SDF editor")
    ps.init()

    slice_plane = ps.add_scene_slice_plane()
    slice_plane.set_draw_plane(False)
    slice_plane.set_draw_widget(True)
    isolines = True
    enable_isosurface_viz = True
    isoline_contour_thickness = 0.3
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
        # isoline_contour_thickness=isoline_contour_thickness,
        enable_isosurface_viz=enable_isosurface_viz,
        enabled=True,
    )
    grid.add_scalar_quantity(
        "All Primitives",
        sd_composite,
        defined_on="nodes",
        cmap=cmap,
        vminmax=vminmax,
        isolines_enabled=isolines,
        # isoline_contour_thickness=isoline_contour_thickness,
        enable_isosurface_viz=enable_isosurface_viz,
        enabled=False,
    )

    # GUI items
    primitive_node_types = [
        pbat.geometry.sdf.Sphere,
        pbat.geometry.sdf.Box,
        pbat.geometry.sdf.BoxFrame,
        pbat.geometry.sdf.Torus,
        pbat.geometry.sdf.CappedTorus,
        pbat.geometry.sdf.Link,
        pbat.geometry.sdf.InfiniteCylinder,
        pbat.geometry.sdf.Cone,
        pbat.geometry.sdf.InfiniteCone,
        pbat.geometry.sdf.Plane,
        pbat.geometry.sdf.HexagonalPrism,
        pbat.geometry.sdf.Capsule,
        pbat.geometry.sdf.VerticalCapsule,
        pbat.geometry.sdf.CappedCylinder,
        pbat.geometry.sdf.VerticalCappedCylinder,
        pbat.geometry.sdf.RoundedCylinder,
        pbat.geometry.sdf.VerticalCappedCone,
        pbat.geometry.sdf.CutHollowSphere,
        pbat.geometry.sdf.VerticalRoundCone,
        pbat.geometry.sdf.Octahedron,
        pbat.geometry.sdf.Pyramid,
        pbat.geometry.sdf.Triangle,
        pbat.geometry.sdf.Quadrilateral,
    ]
    unary_node_types = [
        pbat.geometry.sdf.Scale,
        pbat.geometry.sdf.Elongate,
        pbat.geometry.sdf.Round,
        pbat.geometry.sdf.Onion,
        pbat.geometry.sdf.Symmetrize,
        pbat.geometry.sdf.Repeat,
        pbat.geometry.sdf.Bump,
        pbat.geometry.sdf.Twist,
        pbat.geometry.sdf.Bend,
    ]
    binary_node_types = [
        pbat.geometry.sdf.Union,
        pbat.geometry.sdf.Difference,
        pbat.geometry.sdf.Intersection,
        pbat.geometry.sdf.ExclusiveOr,
        pbat.geometry.sdf.SmoothUnion,
        pbat.geometry.sdf.SmoothIntersection,
        pbat.geometry.sdf.SmoothDifference,
    ]

    selected_primitive_node_type = primitive_node_types[0]
    selected_unary_node_type = unary_node_types[0]
    selected_binary_node_type = binary_node_types[0]

    def callback():
        global nodes, transforms, children, roots, sd_composite, composite
        global selected_primitive_node_type, selected_unary_node_type, selected_binary_node_type

        dirty = False
        if composite.status != pbat.geometry.sdf.ECompositeStatus.Valid:
            dirty = True
        # Node creation UI
        primitive_node_created = False
        unary_node_created = False
        binary_node_created = False

        primitive_node_selection_changed = imgui.BeginCombo(
            "Primitive Nodes", selected_primitive_node_type.__name__
        )
        if primitive_node_selection_changed:
            for node_type in primitive_node_types:
                name = node_type.__name__
                _, selected = imgui.Selectable(
                    name, selected_primitive_node_type.__name__ == name
                )
                if selected:
                    selected_primitive_node_type = node_type
            imgui.EndCombo()
        if imgui.Button("Add Primitive Node"):
            primitive_node_created = True

        unary_node_selection_changed = imgui.BeginCombo(
            "Unary Nodes", selected_unary_node_type.__name__
        )
        if unary_node_selection_changed:
            for node_type in unary_node_types:
                name = node_type.__name__
                _, selected = imgui.Selectable(
                    name, selected_unary_node_type.__name__ == name
                )
                if selected:
                    selected_unary_node_type = node_type
            imgui.EndCombo()
        if imgui.Button("Add Unary Node"):
            unary_node_created = True

        binary_node_selection_changed = imgui.BeginCombo(
            "Binary Nodes", selected_binary_node_type.__name__
        )
        if binary_node_selection_changed:
            for node_type in binary_node_types:
                name = node_type.__name__
                _, selected = imgui.Selectable(
                    name, selected_binary_node_type.__name__ == name
                )
                if selected:
                    selected_binary_node_type = node_type
            imgui.EndCombo()
        if imgui.Button("Add Binary Node"):
            binary_node_created = True

        # Update forest if a node was created
        node_created = (
            primitive_node_created or unary_node_created or binary_node_created
        )
        if node_created:
            node = None
            if primitive_node_created:
                node = selected_primitive_node_type()
                children.append((-1, -1))
            elif unary_node_created:
                node = selected_unary_node_type()
                children.append((len(nodes) - 1, -1))
            elif binary_node_created:
                node = selected_binary_node_type()
                children.append((len(nodes) - 2, len(nodes) - 1))
            nodes.append(node)
            transforms.append(pbat.geometry.sdf.Transform.eye())
            roots, _ = pbat.geometry.sdf.roots_and_parents(children)
            dirty = True

        # Node modification UIs
        deleted_id = -1
        visited = [False] * len(nodes)
        for i in roots:
            # Top-down forest traversal to show UI for each node
            node_dirty, node_deleted_id, ci_descendants_updated = node_ui(
                i, nodes, transforms, children, visited
            )
            was_descendant_deleted = node_deleted_id >= 0
            dirty |= node_dirty or ci_descendants_updated or was_descendant_deleted
            if was_descendant_deleted:
                deleted_id = node_deleted_id

        # Update forest if a node was deleted
        if deleted_id >= 0:
            nodes = nodes[:deleted_id] + nodes[deleted_id + 1 :]
            transforms = transforms[:deleted_id] + transforms[deleted_id + 1 :]
            children = children[:deleted_id] + children[deleted_id + 1 :]
            # Update children indices
            for i in range(len(children)):
                c0, c1 = children[i]
                if c0 >= deleted_id:
                    c0 -= 1
                if c1 >= deleted_id:
                    c1 -= 1
                children[i] = (c0, c1)

        # Update SDF
        if dirty:
            roots, _ = pbat.geometry.sdf.roots_and_parents(children)
            forest = pbat.geometry.sdf.Forest(nodes, transforms, children, roots)
            composite = pbat.geometry.sdf.Composite(forest)
            if composite.status == pbat.geometry.sdf.ECompositeStatus.Valid:
                # Update the composite view
                sd_composite = composite.eval(X).reshape(dims)
                grid.add_scalar_quantity(
                    "Composite",
                    sd_composite,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                    # isoline_contour_thickness=isoline_contour_thickness,
                )
                # Update the primitive nodes view
                primitive_node_inds = [
                    i for i in range(len(children)) if children[i] == (-1, -1)
                ]
                primitive_children = [(-1, -1) for _ in range(len(primitive_node_inds))]
                primitive_roots, _ = pbat.geometry.sdf.roots_and_parents(primitive_children)
                primitive_nodes = [nodes[i] for i in primitive_node_inds]
                primitive_transforms = [transforms[i] for i in primitive_node_inds]
                primitive_forest = pbat.geometry.sdf.Forest(
                    primitive_nodes, primitive_transforms, primitive_children, primitive_roots
                )
                primitive_composite = pbat.geometry.sdf.Composite(primitive_forest)
                primitive_sd_composite = primitive_composite.eval(X).reshape(dims)
                grid.add_scalar_quantity(
                    "All Primitives",
                    primitive_sd_composite,
                    defined_on="nodes",
                    cmap=cmap,
                    vminmax=vminmax,
                    isolines_enabled=isolines,
                    enable_isosurface_viz=enable_isosurface_viz,
                    # isoline_contour_thickness=isoline_contour_thickness,
                )
            else:
                imgui.Text("{}".format(composite.status.name))

    ps.set_user_callback(callback)
    ps.show()
