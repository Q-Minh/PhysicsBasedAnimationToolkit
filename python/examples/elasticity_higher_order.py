import pbatoolkit as pbat
import meshio
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple 3D elastic simulation using quadratic FEM tetrahedra",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-r", "--refined-input", help="Path to refined input mesh", type=str,
                        dest="rinput", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    args = parser.parse_args()
    
    # Load domain and visual meshes
    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    V = V.astype(np.float64, order='c')
    C = C.astype(np.int64, order='c')
    rimesh = meshio.read(args.rinput)
    VR, FR = rimesh.points, rimesh.cells_dict["triangle"]
    VR = VR.astype(np.float64, order='c')
    FR = FR.astype(np.int64, order='c')
    
    # Construct FEM quantities for simulation
    mesh = pbat.fem.Mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=2)
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    v = np.zeros(x.shape[0])
    rho = args.rho
    
    # Mass matrix
    M, detJeM = pbat.fem.mass_matrix(mesh, rho=rho, quadrature_order=4)
    Minv = pbat.math.linalg.ldlt(M)
    Minv.compute(M)

    # Construct load vector from gravity field
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    f, detJeF = pbat.fem.load_vector(mesh, rho*g, quadrature_order=2)
    a = Minv.solve(f).squeeze()

    # Create hyper elastic potential
    Y, nu, psi = args.Y, args.nu, pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep, detJeU, GNeU = pbat.fem.hyper_elastic_potential(
        mesh, Y=Y, nu=nu, energy=psi, quadrature_order=4)

    # Set Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    Xmax[0] = Xmin[0]+1e-4
    Xmin[0] = Xmin[0]-1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    dbcs = np.array(vdbc)[:, np.newaxis]
    dbcs = np.repeat(dbcs, mesh.dims, axis=1)
    for d in range(mesh.dims):
        dbcs[:, d] = mesh.dims*dbcs[:, d]+d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    n = x.shape[0]
    dofs = np.setdiff1d(list(range(n)), dbcs)

    # Setup linear solver
    Hdd = hep.to_matrix()[:, dofs].tocsr()[dofs, :]
    Hddinv = pbat.math.linalg.ldlt(Hdd)
    Hddinv.analyze(Hdd)
    
    # Setup visual mesh
    bvh = pbat.geometry.bvh(V.T, C.T, cell=pbat.geometry.Cell.Tetrahedron)
    e = bvh.nearest_primitives_to_points(VR.T)
    Xi = pbat.fem.reference_positions(mesh, e, VR.T)
    phi = pbat.fem.shape_functions_at(mesh, Xi)

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Higher Order Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("Domain", V, C)
    sm = ps.register_surface_mesh("Visual", VR, FR)
    pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
    dt = 0.01
    animate = False
    dx = np.zeros(n)

    profiler = pbat.profiling.Profiler()

    def callback():
        global x, v, dx, hep, dt, M, Minv, f, animate, step, profiler
        changed, dt = imgui.InputFloat("dt", dt)
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            profiler.begin_frame("Physics")
            # 1 Newton step
            hep.compute_element_elasticity(x, grad=True, hessian=True)
            gradU, HU = hep.gradient(), hep.hessian()
            dt2 = dt**2
            xtilde = x + dt*v + dt2*a
            A = M + dt2 * HU
            b = -(M @ (x - xtilde) + dt2*gradU)
            Add = A.tocsc()[:, dofs].tocsr()[dofs, :]
            bd = b[dofs]
            Hddinv.factorize(Add)
            dx[dofs] = Hddinv.solve(bd).squeeze()
            v = dx / dt
            x = x + dx
            profiler.end_frame("Physics")

            # Update visuals
            X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order='f')
            xv = (X[:,mesh.E[:,e]] * phi).sum(axis=1)
            sm.update_vertex_positions(xv.T)

    ps.set_user_callback(callback)
    ps.show()
