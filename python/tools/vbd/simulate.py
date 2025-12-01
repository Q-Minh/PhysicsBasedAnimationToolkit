# type: ignore
from pbatoolkit import pbat, pypbat
import argparse
import h5py as h5
import gc

solver_params = {
    "vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
        },
        "solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.solve(
            fem, contact, params
        ),
    },
    "anderson_vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
            "anderson": pbat.sim.algorithm.vbd.AndersonParams,
        },
        "solve": lambda fem, contact, vbd_params, anderson_params: pbat.sim.algorithm.vbd.solve(
            fem, contact, vbd_params, anderson_params
        ),
    },
    "broyden_vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
            "broyden": pbat.sim.algorithm.vbd.BroydenParams,
        },
        "solve": lambda fem, contact, vbd_params, broyden_params: pbat.sim.algorithm.vbd.solve(
            fem, contact, vbd_params, broyden_params
        ),
    },
    "chebyshev_vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
            "chebyshev": pbat.sim.algorithm.vbd.ChebyshevParams,
        },
        "solve": lambda fem, contact, vbd_params, chebyshev_params: pbat.sim.algorithm.vbd.solve(
            fem, contact, vbd_params, chebyshev_params
        ),
    },
    "newton": {
        "params": {
            "newton": pbat.sim.algorithm.newton.Params,
        },
        "solve": lambda fem, contact, params: pbat.sim.algorithm.newton.solve(
            fem, params  # TODO: Add contact dynamics to newton solve
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation Tool")
    parser.add_argument(
        "--fem-elasto-dynamics",
        nargs=2,
        required=True,
        help=(
            "FEM elasto dynamics problem as a space-separated pair 'path/to/file1.h5:path/to/hdf5/group1' 'path/to/file2.h5:path/to/hdf5/group2' "
            "where group1 contains a 'pbat.sim.dynamics.FemElastoDynamics' and group2 contains attributes 'dt, bdf_scheme, fem_dynamics_init_strategy'"
        ),
        dest="fem_elasto_dynamics",
    )
    # TODO: Add contact dynamics argument
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="vbd",
        help=f"Solver type. Available type are {', '.join(solver_params.keys())}",
        dest="solver",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        help=(
            f"Space-separated list of params_name=path/to/file.h5:path/to/hdf5/group for solver parameters. "
            f"The hdf5 group should contain the appropriate parameter objects for the selected solver. "
            f"Available solvers and their parameter object names are: {', '.join([f'solver={solver} params_names=({', '.join(data['params'].keys())})' for solver, data in solver_params.items()])}"
        ),
        dest="params",
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        help="List of solver parameter overrides in the format params_name.path.to.attribute=value, where 'params_name' is one of the params entries for the selected solver.",
        dest="overrides",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (s)",
        dest="duration",
    )
    parser.add_argument(
        "--dirichlet-constraints",
        type=str,
        default="",
        help="Pair path/to/file.h5:path/to/group containing a `TransformLibrary` of procedural/kinematic Dirichlet constraints",
        dest="dirichlet_constraints",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.h5:sim",
        required=True,
        help="Output file.h5:group to store simulation trajectory",
        dest="output",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"--fem-elasto-dynamics=\n{args.fem_elasto_dynamics}")
    print(f"--solver={args.solver}")
    print(f"--params={args.params}")
    print(f"--overrides={args.overrides}")
    print(f"--duration={args.duration}")
    print(f"--dirichlet-constraints={args.dirichlet_constraints}")
    print(f"--output={args.output}")


if __name__ == "__main__":
    main()
