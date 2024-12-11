import pbatoolkit as pbat
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import time
import meshio
import argparse
import math


class VbdRestrictionOperator:
    def __init__(
        self,
        MD: pbat.fem.Mesh,
        MT: pbat.fem.Mesh,
        Y=1e6, nu=0.45, rho=1e3,
        cage_quad_params=None,
        iters: int = 10
    ):
        # Construct VBD problem
        VR, FR = pbat.geometry.simplex_mesh_boundary(MD.E, n=MD.X.shape[1])
        rhoe = np.full(mesh.E.shape[1], rho)
        mue, lambdae = pbat.fem.lame_coefficients(
            np.full(mesh.E.shape[1], Y),
            np.full(mesh.E.shape[1], nu)
        )
        data = pbat.sim.vbd.Data().with_volume_mesh(
            MD.X, MD.E
        ).with_surface_mesh(
            VR, FR
        ).with_material(
            rhoe, mue, lambdae
        ).construct()
        # Construct quadrature on coarse mesh
        if cage_quad_params is None:
            cage_quad_params = pbat.sim.vbd.multigrid.CageQuadratureParameters(
            ).with_strategy(
                pbat.sim.vbd.multigrid.CageQuadratureStrategy.EmbeddedMesh
            ).with_cage_mesh_pts(
                3
            ).with_patch_cell_pts(
                2
            ).with_patch_error(
                1e-3
            )
        self.coarse_level = pbat.sim.vbd.multigrid.Level(
            MT
        ).with_cage_quadrature(
            data,
            params=cage_quad_params
        ).with_elastic_energy(
            data
        ).with_momentum_energy(
            data
        )
        # Construct Restriction operator
        self.restriction = pbat.sim.vbd.multigrid.Restriction(
            self.coarse_level.Qcage)
        self.iters = iters
        self.fine_level = pbat.sim.vbd.multigrid.Level(MD)

    def __matmul__(self, u):
        UD = u.reshape(self.fine_level.X.shape, order='F')
        xf = self.fine_level.X + UD
        self.fine_level.x = xf
        energy = self.restriction.do_apply(
            self.iters, self.fine_level.x, self.fine_level.E, self.coarse_level)
        xc = self.coarse_level.x
        UC = xc - self.coarse_level.X
        uc = UC.flatten(order="F")
        return uc


def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c*np.sin(k*w*t)*v
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Linear FEM shape transfer",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-c", "--cage", help="Path to cage tetrahedral mesh", type=str,
                        dest="cage", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("-k", "--num-modes", help="Number of modes to compute", type=int,
                        dest="modes", default=30)
    args = parser.parse_args()

    # Cube test
    # V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
    #              1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    # V = V - V.mean(axis=0)
    # C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [
    #              6, 7, 5, 3], [0, 5, 3, 6]], dtype=np.int64)
    # CV = 2*V
    # CC = C

    # Load input meshes
    imesh, icmesh = meshio.read(args.input), meshio.read(args.cage)
    V, C = imesh.points.astype(
        np.float64, order='c'), imesh.cells_dict["tetra"].astype(np.int64, order='c')
    CV, CC = icmesh.points.astype(
        np.float64, order='c'), icmesh.cells_dict["tetra"].astype(np.int64, order='c')

    # Rescale problem
    center = V.mean(axis=0).reshape(1, 3)
    scale = V.max() - V.min()
    V = (V - center) / scale
    CV = (CV - center) / scale

    # Construct FEM meshes
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron)
    cmesh = pbat.fem.Mesh(
        CV.T, CC.T, element=pbat.fem.Element.Tetrahedron)
    F = pbat.geometry.simplex_mesh_boundary(mesh.E, mesh.X.shape[1])[1]
    CF = pbat.geometry.simplex_mesh_boundary(cmesh.E, cmesh.X.shape[1])[1]

    # Precompute quantities
    w, L = pbat.fem.rest_pose_hyper_elastic_modes(
        mesh, rho=args.rho, Y=args.Y, nu=args.nu, modes=args.modes)

    # Construct restriction operator
    cage_quad_params = pbat.sim.vbd.multigrid.CageQuadratureParameters(
    ).with_strategy(
        pbat.sim.vbd.multigrid.CageQuadratureStrategy.PolynomialSubCellIntegration
    ).with_cage_mesh_pts(
        4
    ).with_patch_cell_pts(
        2
    ).with_patch_error(
        1e-5
    )
    Fvbd = VbdRestrictionOperator(
        mesh,
        cmesh,
        cage_quad_params=cage_quad_params,
        iters=20
    )

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_surface_mesh("model", mesh.X.T, F.T)
    vbdvm = ps.register_surface_mesh("VBD cage", cmesh.X.T, CF.T)
    vbdvm.set_transparency(0.5)
    vbdvm.set_edge_width(1)
    mode = 6
    t0 = time.time()
    t = 0
    c = 3.
    k = 0.1
    theta = 0
    dtheta = np.pi/120
    animate = False
    step = False
    screenshot = False

    def callback():
        global mode, c, k, theta, dtheta
        global animate, step, screenshot
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)
        changed, animate = imgui.Checkbox("animate", animate)
        changed, screenshot = imgui.Checkbox("screenshot", screenshot)
        step = imgui.Button("step")

        if animate or step:
            t = time.time() - t0

            R = sp.spatial.transform.Rotation.from_quat(
                [0, np.sin(theta/2), 0, np.cos(theta/4)]).as_matrix()
            X = (V - V.mean(axis=0)) @ R.T + V.mean(axis=0)
            uf = signal(w[mode], L[:, mode], t, c, k)
            ur = (X - V).flatten(order="C")
            ut = 1e-1*np.ones(math.prod(X.shape))
            u = ut + ur + uf
            # XCnewton = CV + (Fnewton @ u).reshape(CV.shape)
            XCvbd = CV + (Fvbd @ u).reshape(CV.shape)
            # XCrank = CV + (Frank @ u).reshape(CV.shape)
            vm.update_vertex_positions(V + u.reshape(V.shape))
            # newtonvm.update_vertex_positions(XCnewton)
            vbdvm.update_vertex_positions(XCvbd)
            # rankvm.update_vertex_positions(XCrank)

            theta += dtheta
            if theta > 2*np.pi:
                theta = 0

        if screenshot:
            ps.screenshot()

    ps.set_user_callback(callback)
    ps.show()
