import gpytoolbox as gpt
import numpy as np
import meshio
import argparse

from pbatoolkit import pbat, pypbat
import igl
import polyscope as ps
import polyscope.imgui as imgui
import polyscope.implot as implot
import numpy as np
import scipy as sp
import argparse
import meshio
import scipy


def beam(
    dims: np.ndarray | list[float],
    resolution: np.ndarray | list[int],
    subdivision: str = "5",
    center: bool = False,
    normalize: bool = False,
):
    """
    Generate a beam mesh.

    Parameters
    ----------
    dims : list of float
        Dimensions of the beam in the format: dx dy dz
    resolution : list of int
        Mesh resolution for the beam in the format: nx ny nz
    center : bool
        Center the mesh at the origin.
    normalize : bool
        Rescale the mesh so that the largest dimension is unit.

    Returns
    -------
    Tuple[np.ndarray[float], np.ndarray[int]] : The generated beam mesh (V,T).
    """
    V, T = gpt.regular_cube_mesh(
        resolution[0],
        resolution[1],
        resolution[2],
        type=(
            "rotationally-symmetric"
            if subdivision == "5"
            else "reflectionally-symmetric"
        ),
    )
    
    return V, T
 

def define_args():
    parser = argparse.ArgumentParser(description="Generate a beam mesh.")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Dimensions of the beam in the format: dx dy dz",
    )
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        default=[10, 10, 10],
        help="Mesh resolution for the beam in the format: nx ny nz",
    )
    parser.add_argument(
        "--subdivision",
        type=str,
        default="5",
        help="5 | 6 tets per cube (default: 5).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the mesh at the origin.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Rescale the mesh so that the largest dimension is unit.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="spaghetti-nwew.mesh",
        help="Output filename for the mesh.",
    )
    return parser.parse_args()


def get_cylinder(args):
    print(args.input)
    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]
    return V, F


def elastic_potential(x, X, E, eg, wg, GNeg, mug, lambdag):
    U, _, _ = pbat.fem.hyper_elastic_potential(
        E,
        X.shape[1],
        eg=eg,
        wg=wg,
        GNeg=GNeg,
        mug=mug,
        lambdag=lambdag,
        x=np.ravel(x, order="F"),
        energy=pbat.fem.HyperElasticEnergy.StableNeoHookean,
        flags=pbat.fem.ElementElasticityComputationFlags.Potential,
        spd_correction=pbat.fem.HyperElasticSpdCorrection.Absolute,
        element=pbat.fem.Tetrahedron,
        order=1,
        dims=3
    )
    return U


def elastic_potential_gradient(x, X, E, eg, wg, GNeg, mug, lambdag):
    _, gradU, _ = pbat.fem.hyper_elastic_potential(
        E,
        X.shape[1],
        eg=eg,
        wg=wg,
        GNeg=GNeg,
        mug=mug,
        lambdag=lambdag,
        x=np.ravel(x, order="F"),
        energy=pbat.fem.HyperElasticEnergy.StableNeoHookean,
        flags=pbat.fem.ElementElasticityComputationFlags.Gradient,
        spd_correction=pbat.fem.HyperElasticSpdCorrection.Absolute,
        element=pbat.fem.Tetrahedron,
        order=1,
        dims=3
    )
    return gradU


def infinite_cylinder_sdf(V, center, radius=1, return_normals=False):
    """ Compute the sdf of an infinite cylinder along the y axis."""
    position = V[:, [0, 2]] - center

    length = np.linalg.norm(position, axis=1)
    length = length[:, np.newaxis]
    sdf = length - radius
    if return_normals:
        position_3 = np.vstack((position[:, 0], np.zeros_like(position[:, 0]), position[:, 1])).T
        normal = position_3 / length
        return sdf, normal
    return sdf


#def shape_matching_penalty(x, b, Vref, Fref, mu = 1e5):
def shape_matching_penalty(x, b, mu = 1e5, radius=0.5):
    V = x.reshape((-1, 3), order="C")
    #g, _, _ = igl.signed_distance(V[b, :], Vref, Fref, return_normals=False)
    g = infinite_cylinder_sdf(V[b, :], center=np.array([0.5,0.5]), radius=radius, return_normals=False)
    return 0.5 * mu * g.T @ g

#def shape_matching_penalty_gradient(x, b, Vref, Fref, mu = 1e5):
def shape_matching_penalty_gradient(x, b, mu = 1e5, radius=0.5):
    V = x.reshape((-1, 3), order="C")
    #g, _, _, N = igl.signed_distance(V[b, :], Vref, Fref, return_normals=True)
    g, N = infinite_cylinder_sdf(V[b, :], center=np.array([0.5,0.5]), radius=radius, return_normals=True)
    n_constraints = b.shape[0]
    n_nodes = V.shape[0]
    data = N.flatten(order="C")
    rows = np.vstack([
        3*b, 
        3*b+1, 
        3*b+2]).flatten(order="F")
    cols = np.repeat(np.arange(n_constraints), 3)
    gradg = sp.sparse.csc_matrix(
        (data, (rows, cols)), shape=(3*n_nodes, n_constraints)
    )
    return np.ravel(mu * gradg @ g)


if __name__ == "__main__":
    args = define_args()

    V, T = beam(
        dims=args.dims,
        resolution=args.resolution,
        subdivision=args.subdivision,
        center=args.center,
        normalize=args.normalize,
    )

    Fb = igl.boundary_facets(T)
    b = np.sort(np.unique(Fb))
    # Vref, Fref = get_cylinder(args)

    # Initialize material parameters
    Y, nu = 1e6, 0.45
    mu, llambda = pypbat.fem.lame_coefficients(Y, nu)
    x = V.flatten(order="C")
    X = V.T
    E = T.T
    mug = np.full(E.shape[1], mu)
    lambdag = np.full(E.shape[1], llambda)
    egU = np.arange(E.shape[1])
    element = pbat.fem.Element.Tetrahedron
    GNegU = pbat.fem.shape_function_gradients(
        E, X, element=element, dims=3, order=1
    )
    wgU = pbat.fem.mesh_quadrature_weights(E, X, element, order=1, quadrature_order=1).flatten()
    mu_eq = 1e5
    g0 = elastic_potential_gradient(x, X, E, egU, wgU, GNegU, mug, lambdag)

    # Callbacks for potential and gradient
    fun = lambda x: (
        elastic_potential(x, X, E, egU, wgU, GNegU, mug, lambdag) + 
        shape_matching_penalty(x, b, mu=mu_eq, radius=radius)
    )
    jac = lambda x: (
        elastic_potential_gradient(x, X, E, egU, wgU, GNegU, mug, lambdag) + 
        shape_matching_penalty_gradient(x, b, mu=mu_eq, radius=radius)
    )

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Spaghett maker")
    ps.init()
    vm_domain = ps.register_volume_mesh("Domain", V, T)
    vm_model = ps.register_volume_mesh("Model", V, T)

    maxiter = 100
    maxcor = 10
    fk = []
    radius = 0.5

    # Set bounds for L-BFGS. We want x and z to be unbounded, and y to be equal to original position
    bounds = [("-inf", "inf")] * x.shape[0]
    for i in range(x.shape[0]//3):
        bounds[3*i+1] = (V[i, 1], V[i, 1])

    def callback():
        global vm_domain, vm_model, fun, jac, x, mu_eq, maxiter, maxcor, fk, radius

        _, mu_eq = imgui.InputFloat("mu", mu_eq)
        _, maxiter = imgui.InputInt("maxiter", maxiter)
        _, maxcor = imgui.InputInt("maxcor", maxcor)
        _, radius = imgui.InputFloat("radius", radius)

        if imgui.Button("Minimize"):
            fk = []
            def on_lbfgs_iterate(intermediate_result: scipy.optimize.OptimizeResult):
                res = intermediate_result
                fk.append(res.fun)
                vm_model.update_vertex_positions(res.x.reshape((-1, 3), order="C"))
            res = scipy.optimize.minimize(
                fun,
                x,
                method="L-BFGS-B",
                jac=jac,
                options={"maxiter": maxiter, "maxcor": maxcor},
                callback=on_lbfgs_iterate,
                bounds=bounds
            )

        if imgui.Button("Reset"):
            vm_model.update_vertex_positions(V)
            x = V.flatten(order="C")

        flags = implot.ImPlotAxisFlags_None
        if imgui.Button("Fit Axes"):
            flags = flags + implot.ImPlotAxisFlags_AutoFit
        if implot.BeginPlot("Objective"):
            implot.SetupAxes(
                "Iteration",
                "f",
                flags,
                flags,
            )
            implot.PlotLine(
                "L-BFGS",
                np.arange(len(fk)),
                np.array(fk),
            )
            implot.EndPlot()
    ps.set_user_callback(callback)
    ps.show()


    # TODO: bring back mesh transform options
    # V *= np.array(dims)
    # # Center the mesh at the origin if requested
    # if center:
    #     V -= np.array(dims) / 2
    # # Normalize the mesh if requested
    # if normalize:
    #     max_dim = max(dims)
    #     V /= max_dim
    

    # TODO: save the mesh    
    # omesh = meshio.Mesh(res.x.reshape((-1, 3)), [("tetra", T)])
    # meshio.write(args.output, omesh)






