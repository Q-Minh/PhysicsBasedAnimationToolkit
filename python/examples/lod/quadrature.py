import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import meshio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Embedded FEM simulation quadrature",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input tetrahedral mesh",
        type=str,
        dest="input",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cages",
        help="Ordered paths to cage tetrahedral meshes",
        nargs="+",
        dest="cages",
        required=True,
    )
    args = parser.parse_args()

    # Load input meshes
    icmeshes = [meshio.read(cage) for cage in args.cages]
    imesh = meshio.read(args.input)
    root_mesh = pbat.fem.Mesh(
        imesh.points.T,
        imesh.cells_dict["tetra"].T,
        element=pbat.fem.Element.Tetrahedron,
    )
    cage_meshes = [
        pbat.fem.Mesh(
            icmesh.points.T,
            icmesh.cells_dict["tetra"].T,
            element=pbat.fem.Element.Tetrahedron,
        )
        for icmesh in icmeshes
    ]

    # Create visual meshes
    VR, FR = pbat.geometry.simplex_mesh_boundary(root_mesh.E, n=root_mesh.X.shape[1])
    VC, FC = zip(
        *[
            pbat.geometry.simplex_mesh_boundary(cage_mesh.E, n=cage_mesh.X.shape[1])
            for cage_mesh in cage_meshes
        ]
    )

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()

    rsm = ps.register_surface_mesh("Domain", root_mesh.X.T, FR.T)
    min_transparency, max_transparency = 0.25, 0.75
    scene_size = cage_meshes[-1].X.max() - cage_meshes[-1].X.min()
    rsm.set_transparency(max_transparency)
    n_cages = len(cage_meshes)
    for c, cage_mesh in enumerate(cage_meshes):
        csm = ps.register_surface_mesh(f"Cage {c}", cage_mesh.X.T, FC[c].T)
        transparency = (
            min_transparency + c * (max_transparency - min_transparency) / n_cages
        )
        csm.set_transparency(transparency)

    cage_quadrature_strategies = [
        pbat.sim.vbd.lod.CageQuadratureStrategy.CageMesh,
        pbat.sim.vbd.lod.CageQuadratureStrategy.EmbeddedMesh,
        pbat.sim.vbd.lod.CageQuadratureStrategy.PolynomialSubCellIntegration,
    ]
    cage_quadrature_strategy = cage_quadrature_strategies[0]
    cage_mesh_pts = 3
    patch_cell_pts = 2
    patch_error = 1e-4
    rwg = pbat.fem.inner_product_weights(root_mesh)
    volume = rwg.sum()
    scale = rwg.max()
    cvolume = [0.]*n_cages

    def callback():
        global cage_quadrature_strategy, cage_mesh_pts, patch_cell_pts, patch_error
        global volume, cvolume
        changed = imgui.BeginCombo(
            "Quadrature Strategy", str(cage_quadrature_strategy).split(".")[-1]
        )
        if changed:
            for i in range(len(cage_quadrature_strategies)):
                _, selected = imgui.Selectable(
                    str(cage_quadrature_strategies[i]).split(".")[-1],
                    cage_quadrature_strategy == cage_quadrature_strategies[i],
                )
                if selected:
                    cage_quadrature_strategy = cage_quadrature_strategies[i]
            imgui.EndCombo()
            
        changed, cage_mesh_pts = imgui.InputInt("Cage mesh order", cage_mesh_pts)
        changed, patch_cell_pts = imgui.InputInt("Patch cell order", patch_cell_pts)
        changed, patch_error = imgui.InputFloat("Patch error", patch_error, format="%.6f")

        if imgui.Button("Compute"):
            cage_quad_params = (
                pbat.sim.vbd.lod.CageQuadratureParameters()
                .with_strategy(cage_quadrature_strategy)
                .with_cage_mesh_pts(cage_mesh_pts)
                .with_patch_cell_pts(patch_cell_pts)
                .with_patch_error(patch_error)
            )
            cage_quadratures = [
                pbat.sim.vbd.lod.CageQuadrature(
                    root_mesh, cage_mesh, params=cage_quad_params
                )
                for cage_mesh in cage_meshes
            ]
            for c, cage_quadrature in enumerate(cage_quadratures):
                cpc = ps.register_point_cloud(
                    f"Cage quadrature {c}", cage_quadrature.Xg.T
                )
                cpc.add_scalar_quantity(
                    "Weights",
                    cage_quadrature.wg / scene_size * 50,
                    cmap="reds",
                    enabled=True,
                )
                cpc.set_point_radius_quantity("Weights", autoscale=False)
                cvolume[c] = cage_quadrature.wg.sum()

        imgui.Text(f"Volume={volume}")
        for c, vol in enumerate(cvolume):
            imgui.Text(f"Cage {c} volume={vol}")

    ps.set_user_callback(callback)
    ps.show()
