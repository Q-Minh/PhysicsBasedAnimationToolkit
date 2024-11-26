import pbatoolkit as pbat
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import igl
import meshio
import argparse
import qpsolvers
from enum import Enum


class QuadraturePointSelection(Enum):
    FromCageQuadrature = 0
    FromInputRandomSampling = 1
    FromInputSmartSampling = 2


class QuadratureFittingStrategy(Enum):
    FitCageQuadrature = 0
    FitInputQuadrature = 1


def cage_quadrature_points(cmesh, cbvh, iXg, selection=QuadraturePointSelection.FromCageQuadrature, corder=1):
    """Selects quadrature points on coarse cage cmesh, given input mesh quadrature points iXg.

    Args:
        cmesh: Coarse cage mesh
        cbvh: BVH over coarse cage mesh
        iXg (np.ndarray): 3x|#input quad.pts.| array of input mesh quadrature points
        selection (QuadraturePointSelection, optional): Coarse quadrature point selection scheme. 
        Defaults to QuadraturePointSelection.FromCageQuadrature.
        corder (int, optional): Quadrature order on coarse cage. Defaults to 1.

    Raises:
        ValueError: The selection scheme QuadraturePointSelection.FromInputSmartSampling is not yet supported

    Returns:
        (np.ndarray, np.ndarray): (cXg, ceg) where cXg is a 3x|#coarse quad.pts.| array of coarse quadrature points, 
        and ceg is the |#coarse quad.pts.| array of corresponding coarse elements
    """
    cXg = cmesh.quadrature_points(corder)
    n_coarse_elements = cmesh.E.shape[1]
    if selection == QuadraturePointSelection.FromCageQuadrature:
        return cXg
    if selection == QuadraturePointSelection.FromInputRandomSampling:
        inds = np.arange(iXg.shape[1])
        np.random.shuffle(inds)
        n_random_samples = cXg.shape[1]
        inds = inds[:n_random_samples]
        cXg = iXg[:, inds]
        cen = cbvh.primitives_containing_points(cXg)
        ces = np.setdiff1d(list(range(n_coarse_elements)), cen)
        cXg = np.hstack((cXg, cmesh.quadrature_points(1)[:, ces]))
        ceg = np.hstack((cen, ces))
        eorder = np.argsort(ceg)
        cXg = cXg[:, eorder]
        return cXg
    if selection == QuadraturePointSelection.FromInputSmartSampling:
        raise ValueError(
            f"selection={QuadraturePointSelection.FromInputSmartSampling} not yet supported")


def fit_cage_quad_to_fine_quad(imesh, cmesh, ibvh, cbvh, iorder=1, corder=1, selection=QuadraturePointSelection.FromCageQuadrature, fitting_strategy=QuadratureFittingStrategy.FitCageQuadrature):
    iwg = pbat.fem.inner_product_weights(imesh, iorder).flatten(order="F")
    iXg = imesh.quadrature_points(iorder)
    cXg = cage_quadrature_points(
        cmesh, cbvh, iXg, selection=selection, corder=corder)
    if fitting_strategy == QuadratureFittingStrategy.FitInputQuadrature:
        ieg = np.array(ibvh.primitives_containing_points(iXg))
        ceg = np.array(ibvh.primitives_containing_points(cXg))
        singular = np.nonzero(ceg < 0)[0].astype(np.int64)
        valid = np.nonzero(ceg >= 0)[0].astype(np.int64)
        iXi = pbat.fem.reference_positions(imesh, ieg, iXg)
        cXi = pbat.fem.reference_positions(
            imesh, ceg[valid], cXg[:, valid])
        ieg = np.array(cbvh.primitives_containing_points(iXg))
        ceg = np.array(cbvh.primitives_containing_points(
            cXg))
        cwg = np.zeros(cXg.shape[1])
        cwg[valid], err = pbat.math.transfer_quadrature(
            ceg[valid], cXi, ieg, iXi, iwg, order=corder, with_error=True, max_iters=50, precision=1e-10)
    elif fitting_strategy == QuadratureFittingStrategy.FitCageQuadrature:
        ieg = np.array(cbvh.primitives_containing_points(iXg))
        ceg = np.array(cbvh.primitives_containing_points(cXg))
        iXi = pbat.fem.reference_positions(cmesh, ieg, iXg)
        cXi = pbat.fem.reference_positions(cmesh, ceg, cXg)
        cwg, err = pbat.math.transfer_quadrature(
            ceg, cXi, ieg, iXi, iwg, order=corder, with_error=True, max_iters=50, precision=1e-10)
    return cXg, cwg, iXg, iwg, err


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
        QuadraturePointSelection.FromCageQuadrature,
        QuadraturePointSelection.FromInputRandomSampling
    ]
    selection_strategy = selection_strategies[0]
    fitting_strategies = [
        QuadratureFittingStrategy.FitCageQuadrature,
        QuadratureFittingStrategy.FitInputQuadrature
    ]
    fitting_strategy = fitting_strategies[0]

    def callback():
        global cwg, iwg, radius
        global selection_strategy, fitting_strategy, iorder, corder, err

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

        changed, radius = imgui.SliderFloat(
            "Point radius", radius, v_min=1, v_max=1e3)

        if imgui.Button("Compute coarse quadrature"):
            cXg, cwg, iXg, iwg, err = fit_cage_quad_to_fine_quad(
                mesh,
                cmesh,
                ibvh,
                cbvh,
                iorder,
                corder,
                selection=selection_strategy,
                fitting_strategy=fitting_strategy
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
