import pbatoolkit as pbat
import pbatoolkit.fem
import pbatoolkit.geometry
import pbatoolkit.profiling
import pbatoolkit.math.linalg
import meshio
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple 3D elastic simulation using linear FEM tetrahedra",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    args = parser.parse_args()
    
    # Construct FEM quantities for simulation
    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    mesh = pbat.fem.mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='f')
    v = np.zeros(x.shape[0])
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
    M = pbat.fem.mass_matrix(mesh, detJeM, rho=rho,
                             dims=3, quadrature_order=2).to_matrix()
    Minv = pbat.math.linalg.ldlt(M)
    Minv.compute(M)
    # Could also lump the mass matrix like this
    # lumpedm = M.sum(axis=0)
    # M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    # Minv = sp.sparse.spdiags(
    #     1./lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])

    # Construct load vector from gravity field
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    fe = np.tile(rho*g[:, np.newaxis], mesh.E.shape[1])
    f = pbat.fem.load_vector(mesh, detJeU, fe, quadrature_order=1).to_vector()
    a = Minv.solve(f).squeeze()

    # Create hyper elastic potential
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.hyper_elastic_potential(
        mesh, detJeU, GNeU, Y, nu, psi=psi, quadrature_order=1)
    hep.precompute_hessian_sparsity()

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

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_autocenter_structures(False)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_autoscale_structures(False)
    ps.set_ground_plane_mode("shadow_only")
    ps.set_program_name("Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("world model", mesh.X.T, mesh.E.T)
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
            hep.compute_element_elasticity(x, grad=True, hess=True)
            gradU, HU = hep.to_vector(), hep.to_matrix()
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
            vm.update_vertex_positions(X.T)
            pc.update_point_positions(mesh.X[:, vdbc].T)

    ps.set_user_callback(callback)
    ps.show()
