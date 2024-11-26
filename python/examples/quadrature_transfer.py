import pbatoolkit as pbat
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import igl
import meshio
import argparse
import qpsolvers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Quadrature Transfer",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-c", "--cage", help="Path to cage tetrahedral mesh", type=str,
                        dest="cage", required=True)
    args = parser.parse_args()

    # Load input meshes
    imesh, icmesh = meshio.read(args.input), meshio.read(args.cage)
    V, C = imesh.points.astype(
        np.float64, order='c'), imesh.cells_dict["tetra"].astype(np.int64, order='c')
    CV, CC = icmesh.points.astype(
        np.float64, order='c'), icmesh.cells_dict["tetra"].astype(np.int64, order='c')
    maxcoord = V.max()
    V = V / maxcoord
    CV = CV / maxcoord
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    CF = igl.boundary_facets(CC)
    CF[:, :2] = np.roll(CF[:, :2], shift=1, axis=1)
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron)
    cmesh = pbat.fem.Mesh(
        CV.T, CC.T, element=pbat.fem.Element.Tetrahedron)

    ibvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
    cbvh = pbat.geometry.bvh(CV.T, CC.T, cell=pbat.geometry.Cell.Tetrahedron)
    iorder = 1
    corder = 1
    cwg = pbat.fem.inner_product_weights(cmesh, corder).flatten("F")
    iwg = pbat.fem.inner_product_weights(mesh, iorder).flatten("F")
    err = np.array([(iwg.sum() - cwg.sum())**2])

    # Visualize
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()

    ism = ps.register_surface_mesh(
        "Input", V, F, transparency=0.25, edge_width=1)
    csm = ps.register_surface_mesh(
        "Cage", CV, CF, transparency=0.25, edge_width=1)

    radius = 1e2
    selection_strategies = [
        pbat.fem.QuadraturePointSelection.FromOutputQuadrature,
        pbat.fem.QuadraturePointSelection.FromInputRandomSampling
    ]
    selection_strategy = selection_strategies[0]
    fitting_strategies = [
        pbat.fem.QuadratureFittingStrategy.FitOutputQuadrature,
        pbat.fem.QuadratureFittingStrategy.FitInputQuadrature
    ]
    fitting_strategy = fitting_strategies[0]
    singular_strategies = [
        pbat.fem.QuadratureSingularityStrategy.Ignore,
        pbat.fem.QuadratureSingularityStrategy.Constant
    ]
    singular_strategy = singular_strategies[0]

    def callback():
        global cwg, iwg, radius, iorder, corder, err
        global selection_strategy, fitting_strategy, singular_strategy

        changed, iorder = imgui.InputInt("Input quad. order", iorder)
        changed, corder = imgui.InputInt("Coarse quad. order", corder)

        changed = imgui.BeginCombo(
            "Quad.Pt. Selection", str(selection_strategy).split(".")[-1])
        if changed:
            for i in range(len(selection_strategies)):
                _, selected = imgui.Selectable(
                    str(selection_strategies[i]).split(".")[-1], selection_strategy == selection_strategies[i])
                if selected:
                    selection_strategy = selection_strategies[i]
            imgui.EndCombo()

        changed = imgui.BeginCombo(
            "Fitting strategy", str(fitting_strategy).split(".")[-1])
        if changed:
            for i in range(len(fitting_strategies)):
                _, selected = imgui.Selectable(
                    str(fitting_strategies[i]).split(".")[-1], fitting_strategy == fitting_strategies[i])
                if selected:
                    fitting_strategy = fitting_strategies[i]
            imgui.EndCombo()

        changed = imgui.BeginCombo(
            "Singular strategy", str(singular_strategy).split(".")[-1])
        if changed:
            for i in range(len(singular_strategies)):
                _, selected = imgui.Selectable(
                    str(singular_strategies[i]).split(".")[-1], singular_strategy == singular_strategies[i])
                if selected:
                    singular_strategy = singular_strategies[i]
            imgui.EndCombo()

        changed, radius = imgui.SliderFloat(
            "Point radius", radius, v_min=1, v_max=1e3)

        if imgui.Button("Compute coarse quadrature"):
            cXg, cwg, ceg, csg, iXg, iwg, err = pbat.fem.fit_output_quad_to_input_quad(
                mesh,
                cmesh,
                ibvh,
                cbvh,
                iorder,
                corder,
                selection=selection_strategy,
                fitting_strategy=fitting_strategy,
                singular_strategy=singular_strategy
            )
            ipc = ps.register_point_cloud("Input quadrature", iXg.T)
            ipc.add_scalar_quantity("weights", radius*iwg,
                                    cmap="reds", enabled=True)
            ipc.set_point_radius_quantity("weights", autoscale=False)
            cpc = ps.register_point_cloud("Cage quadrature", cXg.T)
            cpc.add_scalar_quantity("weights", radius*cwg,
                                    cmap="reds", enabled=True)
            cpc.set_point_radius_quantity("weights", autoscale=False)

        imgui.Text(f"Fine quad volume={iwg.sum()}")
        imgui.Text(f"Coarse quad volume={cwg.sum()}")
        imgui.Text(f"Integration error={err.sum()}")

    ps.set_user_callback(callback)
    ps.show()
