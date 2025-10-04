import numpy as np
from pbatoolkit import pbat
import scipy as sp
import argparse
import os
import matplotlib.pyplot as plt


def load_descent_path(input: str, frame: int, substep: int):
    files = [f for f in os.listdir(input) if f.endswith(".mtx")]

    def key(f):
        tokens = f.split(".")
        if len(tokens) < 3:
            return f
        t = int(tokens[2])
        s = int(tokens[4])
        k = 0 if len(tokens) < 7 else int(tokens[6])
        return f"{t}".zfill(6) + f"{s}".zfill(2) + f"{k}".zfill(6)

    files.sort(key=key)
    x = [
        sp.io.mmread(f"{input}/{f}")
        for f in files
        if f.startswith(f"x.t.{frame}.s.{substep}")
    ]
    nverts = x[0].shape[0]
    dims = x[0].shape[1]
    x = np.vstack([xi.flatten() for xi in x]).T
    xtilde = next(
        sp.io.mmread(f"{input}/{f}")
        for f in files
        if f.startswith(f"xtilde.t.{frame}.s.{substep}")
    ).flatten()
    return x, xtilde, nverts, dims


class DeformableModel:
    T: np.ndarray
    ntets: int
    lame: np.ndarray
    M: np.ndarray
    mesh: pbat.fem.Mesh
    GNeU: np.ndarray
    wgU: np.ndarray
    egU: np.ndarray
    U: pbat.fem.HyperElasticPotential

    def __init__(self, input: str, X: np.ndarray):
        T = sp.io.mmread(f"{args.input}/T.mtx").astype(np.int64)
        ntets = T.shape[0]
        lame = sp.io.mmread(f"{args.input}/lame.mtx")
        M = sp.io.mmread(f"{args.input}/M.mtx").flatten()
        M = np.repeat(M, dims)
        GNeU = sp.io.mmread(f"{args.input}/GP.mtx").astype(np.float64, order="F")
        wgU = sp.io.mmread(f"{args.input}/wg.mtx").flatten()
        egU = np.arange(ntets, dtype=np.int64)
        mesh = pbat.fem.Mesh(
            X,
            T.T,
            element=pbat.fem.Element.Tetrahedron,
        )
        energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
        U, egU, wgU, GNeU = pbat.fem.hyper_elastic_potential(
            mesh, energy=energy, wg=wgU, eg=egU, GNeg=GNeU
        )
        U.mug, U.lambdag = lame[:, 0], lame[:, 1]

        self.T = T
        self.ntets = ntets
        self.lame = lame
        self.M = M
        self.mesh = mesh
        self.GNeU = GNeU
        self.wgU = wgU
        self.egU = egU
        self.U = U

    def f(self, xk, xtilde, dt):
        self.U.compute_element_elasticity(xk, grad=False, hessian=False)
        dx = xk - xtilde
        EK = np.dot(dx, self.M * dx)
        EU = self.U.eval()
        return 0.5 * EK + dt**2 * EU

    def df(self, xk, xtilde, dt):
        self.U.compute_element_elasticity(xk, grad=True)
        dx = xk - xtilde
        dEK = self.M * dx
        dEU = self.U.gradient()
        return dEK + dt**2 * dEU


def load_simulation_model(input: str, x: np.ndarray, nverts: int, dims: int):
    X = x[:, 0].reshape(dims, nverts, order="F")
    return DeformableModel(input, X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBD path analysis")
    parser.add_argument(
        "-i", "--input", type=str, help="Path to the data", dest="input", required=True
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Frame to analyze.",
        dest="frame",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--substep",
        type=int,
        help="Substep to analyze.",
        dest="substep",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Time step.",
        dest="dt",
        required=False,
        default=1e-2,
    )
    parser.add_argument(
        "--degree",
        type=int,
        help="Degree of the polynomial fit.",
        dest="degree",
        required=False,
        default=2,
    )
    parser.add_argument(
        "--ndata",
        type=int,
        help="Number of data points to fit.",
        dest="ndata",
        required=False,
        default=3,
    )
    parser.add_argument(
        "--zero",
        type=float,
        help="Numerical zero.",
        dest="zero",
        required=False,
        default=1e-5,
    )
    args = parser.parse_args()

    # Load descent path
    x, xtilde, nverts, dims = load_descent_path(args.input, args.frame, args.substep)

    # Load mesh objective function
    dt = args.dt
    model = load_simulation_model(args.input, x, nverts, dims)

    # Analyze path vs polynomial fit
    xmin = np.array([np.min(x[:, d::dims]) for d in range(dims)])
    xmax = np.array([np.max(x[:, d::dims]) for d in range(dims)])
    diag = np.linalg.norm(xmax - xmin)
    ndofs = x.shape[0]
    niterates = x.shape[1]
    k = 0
    degree = args.degree
    ndata = args.ndata
    nt = max(degree + 1, ndata)
    nwindows = x.shape[1] - nt
    zero = args.zero
    npredicted = [0] * nwindows
    for k in range(nwindows):
        xp = [
            np.polynomial.Polynomial.fit(
                k + np.arange(nt), x[i, k : k + nt], degree, symbol="t"
            )
            for i in range(ndofs)
        ]
        for j in range(k + nt, niterates):
            e = np.linalg.norm([xp[i](j) - x[i, j] for i in range(ndofs)]) / diag
            if e <= zero:
                npredicted[k] += 1
            else:
                break
    npredicted = np.array([0] * (nt - 1) + npredicted + [0])

    # Plot objective function
    t = range(niterates)
    ft = np.array([model.f(x[:, k], xtilde, dt) for k in range(niterates)])
    dft = ft[:-1] - ft[1:]
    dx = np.array([np.linalg.norm(x[:, k + 1] - x[:, k]) for k in range(niterates - 1)])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), sharex=True)
    fig.suptitle(f"VBD Path Analysis for frame={args.frame} of {args.input}")

    # Plot change in objective function and step sizes in the first subplot
    ax1.plot(
        t[1:],
        dft,
        label="Change in Objective Function",
        linestyle="--",
        color="r",
        markersize=5,
    )
    ax1.plot(
        t[1:],
        dx,
        label="Step Sizes",
        linestyle="-",
        color="m",
        markersize=5,
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Plot npredicted in the second subplot
    ax2.plot(
        t,
        npredicted,
        label="Number of Predicted Future Iterates",
        linestyle="-",
        color="g",
        markersize=5,
    )
    ax2.set_xlabel("Iteration")
    ax2.plot([], [], " ", label=f"zero={zero}")
    ax2.plot([], [], " ", label=f"bbox diag={diag}")
    ax2.plot([], [], " ", label=f"poly degree={degree}")
    ax2.plot([], [], " ", label=f"n data={ndata}")
    ax2.legend(loc="upper right")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Plot dft over dx in the third subplot
    ax3.plot(
        t[1:],
        dft / dx,
        label="Change in Objective Function over Step Sizes",
        linestyle="-",
        color="b",
        markersize=5,
    )
    ax3.set_yscale("log")
    ax3.set_xlabel("Step Sizes")
    ax3.set_ylabel("Change in Objective Function")
    ax3.legend(loc="upper right")
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5)

    # plt.tight_layout()
    plt.show()
