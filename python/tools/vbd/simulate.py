# type: ignore
from pbatoolkit import pbat, pypbat
from .ui.utils.transform_library import TransformLibrary
from .ui.tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
from .ui.static_mesh_collider import StaticMeshCollider
from .editor import make_fem_dynamics_object, make_contact_dynamics_object
import enum
import argparse
import h5py as h5
import gc
import inspect
from tqdm import tqdm
import numpy as np
import typing


def vbd_prepare(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    contact: pbat.sim.contact.MeshDynamics,
    params: pbat.sim.algorithm.vbd.Params,
):
    n_nodes = fem.X.shape[1]
    GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
        fem.E, n_nodes
    )
    colors = pbat.sim.algorithm.vbd.vertex_colors(fem.E, n_nodes)
    params.with_vertex_element_adjacency_graph(
        GVGp, GVGe, GVGilocal
    ).with_vertex_colors(colors).construct()


_archive_solver_groups = {
    "vbd": "Solver/VBD",
    "anderson": "Solver/Anderson",
    "broyden": "Solver/Broyden",
    "chebychev": "Solver/Chebychev",
    "newton": "Solver/Newton",
}

_archive_tlib_group = "transform_library"

_solver_params = {
    "vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
        },
        "initialize_solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.initialize_solve(
            fem, contact, params["vbd"]
        ),
        "solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.solve(
            fem, contact, params["vbd"]
        ),
        "prepare": lambda fem, contact, params: vbd_prepare(
            fem, contact, params["vbd"]
        ),
    },
    "anderson_vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
            "anderson": pbat.sim.algorithm.vbd.AndersonParams,
        },
        "initialize_solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.initialize_solve(
            fem, contact, params["vbd"], params["anderson"]
        ),
        "solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.solve(
            fem, contact, params["vbd"], params["anderson"]
        ),
        "prepare": lambda fem, contact, params: vbd_prepare(
            fem, contact, params["vbd"]
        ),
    },
    "broyden_vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
            "broyden": pbat.sim.algorithm.vbd.BroydenParams,
        },
        "initialize_solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.initialize_solve(
            fem, contact, params["vbd"], params["broyden"]
        ),
        "solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.solve(
            fem, contact, params["vbd"], params["broyden"]
        ),
        "prepare": lambda fem, contact, params: vbd_prepare(
            fem, contact, params["vbd"]
        ),
    },
    "chebyshev_vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
            "chebyshev": pbat.sim.algorithm.vbd.ChebyshevParams,
        },
        "initialize_solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.initialize_solve(
            fem, contact, params["vbd"], params["chebyshev"]
        ),
        "solve": lambda fem, contact, params: pbat.sim.algorithm.vbd.solve(
            fem, contact, params["vbd"], params["chebyshev"]
        ),
        "prepare": lambda fem, contact, params: vbd_prepare(
            fem, contact, params["vbd"]
        ),
    },
    "newton": {
        "params": {
            "newton": pbat.sim.algorithm.newton.Params,
        },
        "initialize_solve": lambda fem, contact, params: pbat.sim.algorithm.newton.initialize_solve(
            fem, contact, params["newton"]
        ),
        "solve": lambda fem, contact, params: pbat.sim.algorithm.newton.solve(
            fem, contact, params["newton"]
        ),
        "prepare": lambda fem, contact, params: None,
    },
}


# TODO: Also print recursively, for example, a parameter object may have a member which is another parameter object!!!!
def print_param_obj_spec(solver: str):
    if solver not in _solver_params:
        raise ValueError(
            f"Unsupported solver '{solver}'. Available solvers are: {', '.join(_solver_params.keys())}"
        )
    data = _solver_params[solver]
    message = ""
    for param_name, param_cls in data["params"].items():
        param_obj = param_cls()
        for name, member in inspect.getmembers(param_obj):
            if name.startswith("_") or not isinstance(
                member, (int, float, bool, str, enum.Enum)
            ):
                continue
            t = type(member)
            fqname = f"{t.__module__}.{t.__qualname__}"
            if fqname.startswith("builtins."):
                fqname = fqname[len("builtins.") :]
            message += f"{param_name}.{name} = {fqname} ({getattr(type(param_obj), name).__doc__})\n"
    return message


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation Tool")
    parser.add_argument(
        "-p",
        "--sim-params",
        "--simulation_params",
        "--simulation-params",
        type=str,
        default="",
        help=(
            "Path to an h5 file containing solver and contact parameters. "
            "The root group must contain attributes 'dt, bdf_scheme, fem_dynamics_init_strategy', "
            "and it should contain at least one subgroup holding the parameters of a solver. "
            f"Available solvers and their parameter object names are: {', '.join([f'solver={solver} params_names=({', '.join(data['params'].keys())})' for solver, data in _solver_params.items()])}"
        ),
        dest="simulation_params",
    )
    parser.add_argument(
        "-sess",
        "--session",
        "--scene",
        type=str,
        default="",
        help=(
            "Path to an h5 file containing a session to simulate"
            "The root group must contain a list of TetrahedralElastodynamicsBodies as well as a transform library."
        ),
        dest="session",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="vbd",
        help=f"Solver type. Available types are {', '.join(_solver_params.keys())}",
        dest="solver",
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        help=(
            "List of solver parameter overrides in the format params_name.path.to.attribute=value, "
            "where 'params_name' is one of the params entries for the selected solver."
        ),
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
        "--spec",
        action="store_true",
        help="Print solver parameter specification and exit",
        dest="spec",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.h5:sim",
        help="Output file.h5:group to store simulation trajectory",
        dest="output",
    )
    args = parser.parse_args()
    return args


def load_time_integration(path: str):
    try:
        with h5.File(path, "r") as f:
            h5grp = f
            dt = h5grp.attrs["dt"]
            bdf_scheme = h5grp.attrs["bdf_scheme"]
            fem_dynamics_init_strategy = (
                pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization(
                    h5grp.attrs["fem_dynamics_init_strategy"]
                )
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load time integration from '{arg}': {e}") from e
    return dt, bdf_scheme, fem_dynamics_init_strategy


def load_simulation_scenario_from_scene(path: str):
    try:
        tet_elastic_bodies = []
        static_mesh_colliders = []
        with h5.File(path, "r") as f:
            grp = f["fem_tet_elastic_bodies"]
            num_bodies = grp.attrs["num_tet_elastic_bodies"]
            for b in range(num_bodies):
                body_grp = grp[f"{b}"]
                body = TetrahedralElastodynamicsBody()
                body.deserialize(body_grp, headless=True)
                tet_elastic_bodies.append(body)

            grp = f["static_mesh_colliders"]
            num_static_meshes = grp.attrs["num_static_mesh_colliders"]
            for b in range(num_static_meshes):
                body_grp = grp[f"{b}"]
                body = StaticMeshCollider()
                body.deserialize(body_grp, headless=True)
                static_mesh_colliders.append(body)

            transform_library = TransformLibrary()
            tlib_h5grp = f[_archive_tlib_group]
            transform_library.deserialize(tlib_h5grp)
            tet_elastic_body_names = tlib_h5grp["mesh_names"][:].astype(str).tolist()

        fem, vert_counts, XP = make_fem_dynamics_object(
            tet_elastic_bodies, get_external_info=True
        )
        contact = make_contact_dynamics_object(
            fem, len(tet_elastic_bodies), vert_counts, static_mesh_colliders
        )
        return fem, contact, transform_library, tet_elastic_body_names, XP
    except Exception as e:
        raise RuntimeError(
            f"Failed to load simulation scenario from '{path}': {e}"
        ) from e


def load_solver_params(
    solver: str, solver_params: str | None = None, overrides: list[str] | None = None
) -> typing.Any:
    if solver not in _solver_params:
        raise ValueError(
            f"Unsupported solver '{solver}'. Available solvers are: {', '.join(_solver_params.keys())}"
        )
    param_objs = _solver_params[solver]["params"]
    param_objs = {name: cls() for name, cls in param_objs.items()}

    for param_name in param_objs:
        try:
            archive = pbat.io.Archive(solver_params, flags=pbat.io.AccessMode.ReadOnly)
            param_objs[param_name].deserialize(
                archive.get(_archive_solver_groups[param_name])
            )
            archive = None
            gc.collect()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load solver params from '{path}:{group}': {e}"
            ) from e
    if overrides is not None:
        for override in overrides:
            lhs, rhs = override.split("=")[:2]
            param_name, *attr_path = lhs.split(".")
            if param_name not in param_objs:
                raise ValueError(
                    f"Invalid override parameter name '{param_name}'. Available parameter names are: {', '.join(param_objs.keys())}"
                )
            obj = param_objs[param_name]
            for attr in attr_path[:-1]:
                obj = getattr(obj, attr)
            final_attr = attr_path[-1]
            current_value = getattr(obj, final_attr)
            value_type = type(current_value)
            if value_type == bool:
                value = rhs.lower() in ("true", "1", "yes", "on")
            elif value_type == enum.Enum:
                enum_values = list(value_type)
                matched = False
                for enum_value in enum_values:
                    if enum_value.name == rhs:
                        value = enum_value
                        matched = True
                        break
                if not matched:
                    raise ValueError(
                        f"Invalid enum value '{rhs}' for attribute '{final_attr}'. Available values are: {', '.join([ev.name for ev in enum_values])}"
                    )
            else:
                value = value_type(rhs)
            setattr(obj, final_attr, value)
    return param_objs


def load_contact_dynamics_params(
    path: str | None = None,
) -> pbat.sim.contact.MeshDynamicsParams:
    try:
        archive = pbat.io.Archive(path, flags=pbat.io.AccessMode.ReadOnly)
        contact_dynamics_params = pbat.sim.contact.MeshDynamicsParams()
        contact_dynamics_params.deserialize(archive["Contact"])
        archive = None
        gc.collect()
    except Exception as e:
        raise RuntimeError(f"Failed to load contact dynamics from '{path}': {e}") from e
    return contact_dynamics_params


def main():
    args = parse_args()
    if args.spec:
        print(print_param_obj_spec(args.solver))
        return
    (
        fem_elasto_dynamics,
        contact_dynamics,
        dirichlet_constraints,
        fem_elastic_mesh_names,
        XP,
    ) = load_simulation_scenario_from_scene(args.session)

    dt, bdf_scheme, fem_dynamics_init_strategy = load_time_integration(
        args.simulation_params
    )
    solver_params = load_solver_params(
        args.solver, args.simulation_params, args.overrides
    )
    contact_dynamics_params = load_contact_dynamics_params(args.simulation_params)
    contact_dynamics.params = contact_dynamics_params.construct()

    out_file, out_group = args.output.split(":")[:2]
    # Prepare elasto dynamics problem
    fem_elasto_dynamics.set_time_integration_scheme(dt, bdf_scheme)
    fem_elasto_dynamics.set_initial_conditions(
        fem_elasto_dynamics.x, fem_elasto_dynamics.v
    )
    # Prepare contact dynamics problem
    device_config = pbat.geometry.DeviceConfig()
    device_config.threads = 0
    device_config.start_threads = 1
    device = pbat.geometry.Device(device_config)
    contact_dynamics.initialize(device)
    # Prepare solver
    prepare = _solver_params[args.solver]["prepare"]
    prepare(fem_elasto_dynamics, contact_dynamics, solver_params)

    xD = None if dirichlet_constraints is None else fem_elasto_dynamics.x.copy()
    t = 0
    archive = pbat.io.Archive(out_file, flags=pbat.io.AccessMode.Overwrite)
    fem_elasto_dynamics.serialize(archive[f"{out_group}/{t:08d}"])
    pbar = tqdm(total=int(args.duration / dt), desc="Simulating", unit="step")

    #################
    #  RUN THE SIM  #
    #################

    solve = _solver_params[args.solver]["solve"]
    initialize_solve = _solver_params[args.solver]["initialize_solve"]
    while t * dt < args.duration:
        # Apply procedural constraints
        fem_elasto_dynamics.dmask[:] = 0
        if len(dirichlet_constraints.transforms) > 0:
            for start, tup in zip(
                XP[:-1], dirichlet_constraints.all_transformed_nodes(t, dt)
            ):
                _, dnodes = tup
                fem_elasto_dynamics.dmask[start + dnodes] = 1
            fem_elasto_dynamics.constrain(fem_elasto_dynamics.dmask)
            xD = fem_elasto_dynamics.x
            for start, end, name in zip(XP[:-1], XP[1:], fem_elastic_mesh_names):
                xD[:, start:end] = dirichlet_constraints.apply(
                    name, xD[:, start:end], t, dt
                )
            fem_elasto_dynamics.x = xD
        # Initialize time step optimization
        fem_elasto_dynamics.setup_time_integration_optimization(
            initialization_strategy=fem_dynamics_init_strategy
        )
        # Solve
        initialize_solve(fem_elasto_dynamics, contact_dynamics, solver_params)
        solve(fem_elasto_dynamics, contact_dynamics, solver_params)
        # Step
        fem_elasto_dynamics.step()
        t += 1
        # Write output
        fem_elasto_dynamics.serialize(archive[f"{out_group}/{t:08d}"])
        # Update progress bar
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
