import pbatoolkit as pbat
import pbatoolkit.math.linalg
import pbatoolkit.geometry
import igl
import meshio
import polyscope as ps
import numpy as np
import argparse


def harmonic_field(V: np.ndarray, C: np.ndarray, order: int, eps: float = 0.1):
    # Construct order order mesh and its Laplacian
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=order)
    quadrature_order = max(int(2*(order-1)), 1)
    detJeL = pbat.fem.jacobian_determinants(
        mesh, quadrature_order=quadrature_order)
    GNeL = pbat.fem.shape_function_gradients(
        mesh, quadrature_order=quadrature_order)
    L = pbat.fem.Laplacian(
        mesh, detJeL, GNeL, quadrature_order=quadrature_order).to_matrix()
    # Set Dirichlet boundary conditions at bottom and top of the model
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    extents = Xmax - Xmin
    extents[:-1] = 0
    aabblo = pbat.geometry.aabb(
        np.vstack((Xmin - eps*extents, Xmax - (1-eps)*extents)).T)
    gammalo = aabblo.contained(mesh.X)
    aabbhi = pbat.geometry.aabb(
        np.vstack((Xmin + (1-eps)*extents, Xmax + eps*extents)).T)
    gammahi = aabbhi.contained(mesh.X)
    gamma = np.concatenate((gammalo, gammahi))
    uk = np.concatenate((np.ones(len(gammalo)), np.zeros(len(gammahi))))

    # Solve boundary value problem
    n = mesh.X.shape[1]
    dofs = np.setdiff1d(list(range(n)), gamma)
    Lu = L.tocsr()[dofs, :]
    Luu = Lu.tocsc()[:, dofs]
    Luk = Lu.tocsc()[:, gamma]
    Luuinv = pbat.math.linalg.ldlt(Luu)
    Luuinv.compute(Luu)
    b = -Luk @ uk
    uu = Luuinv.solve(b).squeeze()
    u = np.zeros(n)
    u[gamma] = uk
    u[dofs] = uu
    return u, mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Higher order FEM demo",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-r", "--refined-input", help="Path to refined input tetrahedral mesh", type=str,
                        dest="rinput", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C, = imesh.points, imesh.cells_dict["tetra"]
    V = np.copy(V, order='c')
    C = C.astype(np.int64, order='c')
    u1, mesh1 = harmonic_field(V, C, order=1)
    u2, mesh2 = harmonic_field(V, C, order=2)
    bvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
    rmesh = meshio.read(args.rinput)
    Vrefined, Crefined = rmesh.points.astype(
        np.float64, order='c'), rmesh.cells_dict["tetra"].astype(np.int64, order='c')
    Frefined = igl.boundary_facets(Crefined)
    Frefined[:,:2] = np.roll(Frefined[:,:2], shift=1, axis=1)

    e = bvh.nearest_primitives_to_points(Vrefined.T)
    Xi1 = pbat.fem.reference_positions(mesh1, e, Vrefined.T)
    Xi2 = pbat.fem.reference_positions(mesh2, e, Vrefined.T)
    phi1 = pbat.fem.shape_functions_at(mesh1, Xi1)
    phi2 = pbat.fem.shape_functions_at(mesh2, Xi2)
    u1ref = (u1[mesh1.E[:, e]] * phi1).sum(axis=0)
    u2ref = (u2[mesh2.E[:, e]] * phi2).sum(axis=0)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("domain refined", Vrefined, Crefined)
    vm.add_scalar_quantity("Order 1 harmonic solution", u1ref, enabled=True, cmap="turbo")
    vm.add_scalar_quantity("Order 2 harmonic solution", u2ref, cmap="turbo")
    niso = 15
    def isolines(V, F, u, niso):
        # Code for libigl 2.5.1
        diso = (u.max() - u.min()) / (niso+2)
        isovalues = np.array([(i+1)*diso for i in range(niso)])
        Viso, Eiso, Iiso = igl.isolines(V, F, u, isovalues)
        # Uncomment for libigl 2.4.1
        # Viso1, Eiso1 = igl.isolines(V, F, u, niso)
        return Viso, Eiso
    Viso1, Eiso1 = isolines(Vrefined, Frefined, u1ref, niso)
    Viso2, Eiso2 = isolines(Vrefined, Frefined, u2ref, niso)
    cn1 = ps.register_curve_network("Order 1 contours", Viso1, Eiso1)
    cn1.set_radius(0.002)
    cn1.set_color((0, 0, 0))
    cn2 = ps.register_curve_network("Order 2 contours", Viso2, Eiso2)
    cn2.set_radius(0.002)
    cn2.set_color((0, 0, 0))
    cn2.set_enabled(False)
    ps.show()
