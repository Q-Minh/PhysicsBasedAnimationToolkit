import pbatoolkit as pbat
import igl
import meshio
import polyscope as ps
import numpy as np
import argparse


def harmonic_field(
    V: np.ndarray,
    C: np.ndarray,
    element: pbat.fem.Element,
    order: int,
    eps: float = 0.1,
):
    # Construct order order mesh and its Laplacian
    X, E = pbat.fem.mesh(V.T, C.T, element=element, order=order)
    L = pbat.fem.laplacian_matrix(E, X, element=element, order=order)
    # Set Dirichlet boundary conditions at bottom and top of the model
    Xmin = X.min(axis=1)
    Xmax = X.max(axis=1)
    extents = Xmax - Xmin
    extents[:-1] = 0
    aabblo = pbat.geometry.aabb(
        np.vstack((Xmin - eps * extents, Xmax - (1 - eps) * extents)).T
    )
    gammalo = aabblo.contained(X)
    aabbhi = pbat.geometry.aabb(
        np.vstack((Xmin + (1 - eps) * extents, Xmax + eps * extents)).T
    )
    gammahi = aabbhi.contained(X)
    gamma = np.concatenate((gammalo, gammahi))
    uk = np.concatenate((np.ones(len(gammalo)), np.zeros(len(gammahi))))

    # Solve boundary value problem
    n = X.shape[1]
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
    return u, X, E


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Higher order FEM demo",
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
        "-r",
        "--refined-input",
        help="Path to refined input tetrahedral mesh",
        type=str,
        dest="rinput",
        required=True,
    )
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    (
        V,
        C,
    ) = (
        imesh.points,
        imesh.cells_dict["tetra"],
    )
    V = V.astype(np.float64, order="c")
    C = C.astype(np.int64, order="c")
    element = pbat.fem.Element.Tetrahedron
    u1, X1, E1 = harmonic_field(V, C, element, order=1)
    u2, X2, E2 = harmonic_field(V, C, element, order=2)
    bvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
    rmesh = meshio.read(args.rinput)
    Vrefined, Crefined = rmesh.points.astype(np.float64, order="c"), rmesh.cells_dict[
        "tetra"
    ].astype(np.int64, order="c")
    Frefined = igl.boundary_facets(Crefined)
    Frefined[:, :2] = np.roll(Frefined[:, :2], shift=1, axis=1)

    e, d = bvh.nearest_primitives_to_points(Vrefined.T)
    Xi1 = pbat.fem.reference_positions(E1, X1, e, Vrefined.T, element=element, order=1)
    Xi2 = pbat.fem.reference_positions(E2, X2, e, Vrefined.T, element=element, order=2)
    phi1 = pbat.fem.shape_functions_at(Xi1, element, order=1)
    phi2 = pbat.fem.shape_functions_at(Xi2, element, order=2)
    u1ref = (u1[E1[:, e]] * phi1).sum(axis=0)
    u2ref = (u2[E2[:, e]] * phi2).sum(axis=0)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("domain refined", Vrefined, Crefined)
    vm.add_scalar_quantity(
        "Order 1 harmonic solution", u1ref, enabled=True, cmap="turbo"
    )
    vm.add_scalar_quantity("Order 2 harmonic solution", u2ref, cmap="turbo")
    niso = 15

    def isolines(V, F, u, niso):
        # Code for libigl 2.5.1
        diso = (u.max() - u.min()) / (niso + 2)
        isovalues = np.array([(i + 1) * diso for i in range(niso)])
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
