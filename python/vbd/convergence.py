import numpy as np
import scipy as sp
import argparse
import os
import pathlib
import matplotlib.pyplot as plt
import pbatoolkit as pbat

accelerators = [
    "vbd",
    "chebyshev",
    "anderson",
]


def initialization_strategy(args):
    if args.initialization_strategy == "position":
        return pbat.sim.vbd.InitializationStrategy.Position
    elif args.initialization_strategy == "inertia":
        return pbat.sim.vbd.InitializationStrategy.Inertia
    elif args.initialization_strategy == "kinetic_energy":
        return pbat.sim.vbd.InitializationStrategy.KineticEnergyMinimum
    elif args.initialization_strategy == "adaptive_vbd":
        return pbat.sim.vbd.InitializationStrategy.AdaptiveVbd
    elif args.initialization_strategy == "adaptive_pbat":
        return pbat.sim.vbd.InitializationStrategy.AdaptivePbat
    else:
        raise ValueError(
            f"Unknown initialization strategy: {args.initialization_strategy}"
        )


def load_data(args):
    V = sp.io.mmread(os.path.join(args.input, "V.mtx"))
    C = sp.io.mmread(os.path.join(args.input, "C.mtx"))
    F = sp.io.mmread(os.path.join(args.input, "F.mtx"))
    B = sp.io.mmread(os.path.join(args.input, "B.mtx"))
    aext = sp.io.mmread(os.path.join(args.input, "aext.mtx"))
    rhoe = sp.io.mmread(os.path.join(args.input, "rhoe.mtx"))
    mue = sp.io.mmread(os.path.join(args.input, "mue.mtx"))
    lambdae = sp.io.mmread(os.path.join(args.input, "lambdae.mtx"))
    vdbc = sp.io.mmread(os.path.join(args.input, "vdbc.mtx"))
    data = (
        pbat.sim.vbd.Data()
        .with_volume_mesh(V.T, C.T)
        .with_surface_mesh(np.unique(F), F.T)
        .with_bodies(B)
        .with_acceleration(aext)
        .with_material(rhoe, mue, lambdae)
        .with_dirichlet_vertices(vdbc)
        .with_initialization_strategy(initialization_strategy(args))
        .with_contact_parameters(args.muC, args.muF, args.epsv)
    ).construct(validate=True)
    # Set the initial state
    x = sp.io.mmread(os.path.join(args.input, f"{args.frame}.x.mtx"))
    v = sp.io.mmread(os.path.join(args.input, f"{args.frame}.v.mtx"))
    xt = sp.io.mmread(os.path.join(args.input, f"{args.frame-1}.x.mtx"))
    vt = sp.io.mmread(os.path.join(args.input, f"{args.frame-1}.v.mtx"))
    data.x = x
    data.v = v
    data.xt = xt
    data.vt = vt
    return data


def simulate(args, vbd, name):
    outdir = os.path.join(args.output, name)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    vbd.trace_next_step(outdir, args.frame)
    vbd.step(args.dt, args.iters, args.substeps)


def run(args):
    # Run base VBD
    data = load_data(args)
    data.accelerator = pbat.sim.vbd.AccelerationStrategy.Base
    simulate(args, pbat.sim.vbd.Integrator(data), accelerators[0])

    # Run Chebyshev accelerated VBD
    data = load_data(args)
    data.with_chebyshev_acceleration(args.chebyshev_rho)
    simulate(args, pbat.sim.vbd.Integrator(data), accelerators[1])

    # Run Anderson accelerated VBD
    data = load_data(args)
    data.with_anderson_acceleration(args.anderson_window)
    simulate(args, pbat.sim.vbd.Integrator(data), accelerators[2])


def analyze(args):
    # Load objective function values and gradients
    paths = [os.path.join(args.input, accelerator) for accelerator in accelerators]
    grads = [
        sp.io.mmread(os.path.join(path, f"{args.frame}.{args.substep}.grad.mtx"))
        for path in paths
    ]
    fs = [
        sp.io.mmread(os.path.join(path, f"{args.frame}.{args.substep}.f.mtx"))
        for path in paths
    ]
    gnorms = [
        np.array([np.linalg.norm(grad[:, k]) for k in range(grad.shape[1])])
        for grad in grads
    ]
    names = [os.path.split(path)[-1] for path in paths]

    # Plot objective function values and gradient norms
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Convergence", fontsize=16)
    faxs = axs[0]
    gaxs = axs[1]

    # Plot objective function values
    for f, name in zip(fs, names):
        faxs.plot(f[:-1] - f[1:], label=name)
    faxs.set_title("Objective Function")
    faxs.set_xlabel("Iteration")
    faxs.set_ylabel("Objective Value")
    faxs.legend()
    faxs.grid(True)

    # Plot gradient norms
    for gnorm, name in zip(gnorms, names):
        gaxs.plot(gnorm, label=name)
    gaxs.set_title("Gradient Norm")
    gaxs.set_xlabel("Iteration")
    gaxs.set_ylabel("Gradient Norm")
    gaxs.legend()
    gaxs.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run vanilla VBD and its accelerated variants starting from given ground truth states.",
        default=False,
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze the convergence of VBD variants.",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the data. In run mode, this is the "
        "path to a directory containing a ground truth simulation "
        "trajectory (i.e. pickled files of pbat.sim.vbd.Data). "
        "In analyze mode, these are the paths to the outputs of "
        "the run mode execution.",
        dest="input",
        required=True,
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Time step size to use in run mode.",
        dest="dt",
        required=False,
        default=1e-2,
    )
    parser.add_argument(
        "-t",
        "--frame",
        type=int,
        help="Frame to start from in run mode or to consider in analyze mode. Must be > 0.",
        dest="frame",
        required=True,
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Number of iterations to run in run mode.",
        dest="iters",
        required=False,
        default=1000,
    )
    parser.add_argument(
        "-s",
        "--substeps",
        type=int,
        help="Number of substeps to run in run mode.",
        dest="substeps",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--substep",
        type=int,
        help="Substep to analyze in analyze mode.",
        dest="substep",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--init-strategy",
        type=str,
        help="Initialization strategy to use.",
        dest="initialization_strategy",
        required=False,
        default="kinetic_energy",
    )
    parser.add_argument(
        "--muC", help="Collision penalty", type=float, default=1e6, dest="muC"
    )
    parser.add_argument(
        "--muF", help="Friction coefficient", type=float, default=0.3, dest="muF"
    )
    parser.add_argument(
        "--epsv",
        help="Relative tangential velocity threshold for contact constraints",
        type=float,
        default=1e-2,
        dest="epsv",
    )
    parser.add_argument(
        "--chebyshev-rho",
        type=float,
        help="Chebyshev estimated spectral radius.",
        dest="chebyshev_rho",
        required=False,
        default=0.9,
    )
    parser.add_argument(
        "--anderson-window",
        type=int,
        help="Anderson acceleration window size.",
        dest="anderson_window",
        required=False,
        default=5,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output directory for run mode.",
        dest="output",
        required=False,
    )
    args = parser.parse_args()

    if args.run:
        run(args)
    elif args.analyze:
        analyze(args)
    else:
        print("Please specify either --run or --analyze.")
        exit(1)
