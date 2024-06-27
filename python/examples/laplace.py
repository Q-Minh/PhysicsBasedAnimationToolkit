import pbatoolkit as pbat
import pbatoolkit.fem
import pbatoolkit.math.linalg
import pbatoolkit.geometry
import meshio
import polyscope as ps
import numpy as np
import argparse

def harmonic_field(V: np.ndarray, C: np.ndarray, order: int):
    # Construct order order mesh and its Laplacian
    mesh = pbat.fem.mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=order)
    quadrature_order = max(int(2*(order-1)), 1)
    detJeL = pbat.fem.jacobian_determinants(
        mesh, quadrature_order=quadrature_order)
    GNeL = pbat.fem.shape_function_gradients(
        mesh, quadrature_order=quadrature_order)
    L = pbat.fem.laplacian_matrix(
        mesh, detJeL, GNeL, quadrature_order=quadrature_order).to_matrix()
    # Set Dirichlet boundary conditions at bottom and top of the model
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    dXmin = np.zeros(mesh.dims)
    dXmax = np.zeros(mesh.dims)
    eps = 1e-4
    dXmin[-1] = -eps
    dXmax[-1] = -(Xmax[-1] - Xmin[-1]) + eps
    aabblo = pbat.geometry.aabb(np.vstack((Xmin + dXmin, Xmax + dXmax)).T)
    gammalo = aabblo.contained(mesh.X)
    dXmin[-1] = (Xmax[-1] - Xmin[-1]) - eps
    dXmax[-1] = eps
    aabbhi = pbat.geometry.aabb(np.vstack((Xmin + dXmin, Xmax + dXmax)).T)
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
    Vrefined, Crefined = rmesh.points, rmesh.cells_dict["tetra"]
    
    e = bvh.nearest_primitives_to_points(Vrefined.T)
    Xi1 = pbat.fem.reference_positions(mesh1, e, Vrefined.T)
    Xi2 = pbat.fem.reference_positions(mesh2, e, Vrefined.T)
    phi1 = pbat.fem.shape_functions_at(mesh1, Xi1)
    phi2 = pbat.fem.shape_functions_at(mesh2, Xi2)
    u1ref = (u1[mesh1.E[:,e]] * phi1).sum(axis=0)
    u2ref = (u2[mesh2.E[:,e]] * phi2).sum(axis=0)
    
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    ps.register_volume_mesh("domain", V, C)
    vm = ps.register_volume_mesh("domain refined", Vrefined, Crefined)
    vm.add_scalar_quantity("Order 1 harmonic solution", u1ref)
    vm.add_scalar_quantity("Order 2 harmonic solution", u2ref)
    ps.show()
