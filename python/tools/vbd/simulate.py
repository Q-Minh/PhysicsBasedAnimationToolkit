# type: ignore
from pbatoolkit import pbat, pypbat
from .ui.utils.transform_library import TransformLibrary
import enum
import numpy as np
import argparse
import h5py as h5
import gc
import inspect


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


_solver_params = {
    "vbd": {
        "params": {
            "vbd": pbat.sim.algorithm.vbd.Params,
        },
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
        "solve": lambda fem, contact, params: pbat.sim.algorithm.newton.solve(
            fem, params["newton"]  # TODO: Add contact dynamics to newton solve
        ),
        "prepare": lambda fem, contact, params: None,
    },
}


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
        "-t",
        "--time-integration",
        type=str,
        default="",
        help=(
            "Time integration scheme as 'path/to/file.h5:path/to/hdf5/group' "
            "where 'group' contains attributes 'dt, bdf_scheme, fem_dynamics_init_strategy'"
        ),
        dest="time_integration",
    )
    parser.add_argument(
        "--fem-elasto-dynamics",
        type=str,
        help=(
            "FEM elasto dynamics problem as 'path/to/file.h5:path/to/hdf5/group' "
            "where group contains a 'pbat.sim.dynamics.FemElastoDynamics'"
        ),
        dest="fem_elasto_dynamics",
    )
    # TODO: Add contact dynamics argument
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="vbd",
        help=f"Solver type. Available type are {', '.join(_solver_params.keys())}",
        dest="solver",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        help=(
            f"Space-separated list of params_name=path/to/file.h5:path/to/hdf5/group for solver parameters. "
            f"The hdf5 group should contain the appropriate parameter objects for the selected solver. "
            f"Available solvers and their parameter object names are: {', '.join([f'solver={solver} params_names=({', '.join(data['params'].keys())})' for solver, data in _solver_params.items()])}"
        ),
        dest="params",
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
        "--dirichlet-constraints",
        type=str,
        default="",
        help=(
            "Pair path/to/file.h5:path/to/group,path/to/scene where group contains a `TransformLibrary` "
            "of procedural/kinematic Dirichlet constraints and scene contains datasets 'XP' and "
            "'tet_elastic_body_names'"
        ),
        dest="dirichlet_constraints",
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


def load_time_integration(arg: str):
    if arg == "":
        dt = 0.01
        bdf_scheme = 1
        fem_dynamics_init_strategy = (
            pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithExternalLoad
        )
    else:
        path, group = arg.split(":")[:2]
        try:
            with h5.File(path, "r") as f:
                h5grp = f if group is None or group == "" else f[group]
                dt = h5grp.attrs["dt"]
                bdf_scheme = h5grp.attrs["bdf_scheme"]
                fem_dynamics_init_strategy = (
                    pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization(
                        h5grp.attrs["fem_dynamics_init_strategy"]
                    )
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load time integration from '{arg}': {e}"
            ) from e
    return dt, bdf_scheme, fem_dynamics_init_strategy


def load_fem_elasto_dynamics(arg: str):
    path, group = arg.split(":")[:2]
    try:
        archive = pbat.io.Archive(path, flags=pbat.io.AccessMode.ReadOnly)
        fem = pbat.sim.dynamics.FemElastoDynamics()
        fem.deserialize(archive if group is None or group == "" else archive[group])
        archive = None
        gc.collect()
        return fem
    except Exception as e:
        raise RuntimeError(
            f"Failed to load fem elasto dynamics from '{arg}': {e}"
        ) from e


def load_solver_params(
    solver: str, params: str | None = None, overrides: list[str] | None = None
):
    if solver not in _solver_params:
        raise ValueError(
            f"Unsupported solver '{solver}'. Available solvers are: {', '.join(_solver_params.keys())}"
        )
    param_objs = _solver_params[solver]["params"]
    param_objs = {name: cls() for name, cls in param_objs.items()}
    params = [p.split("=")[:2] for p in params]
    params = [(p[0], p[1].split(":")[:2]) for p in params]
    for param_name, (path, group) in params:
        try:
            archive = pbat.io.Archive(path, flags=pbat.io.AccessMode.ReadOnly)
            param_objs[param_name].deserialize(
                archive if group is None or group == "" else archive.get(group)
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


def load_dirichlet_constraints(arg: str):
    if arg == "":
        return None
    path, groups = arg.split(":")[:2]
    tlib_group, scene_group = groups.split(",")[:2]
    try:
        with h5.File(path, "r") as f:
            transform_library = TransformLibrary()
            tlib_h5grp = f if tlib_group is None or tlib_group == "" else f[tlib_group]
            transform_library.deserialize(tlib_h5grp)
            scene_h5grp = (
                f if scene_group is None or scene_group == "" else f[scene_group]
            )
            XP = scene_h5grp["XP"][:]
            tet_elastic_body_names = (
                scene_h5grp["tet_elastic_body_names"][:].astype(str).tolist()
            )
            return transform_library, XP, tet_elastic_body_names
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dirichlet constraints from '{arg}': {e}"
        ) from e


def main():
    args = parse_args()
    if args.spec:
        print(print_param_obj_spec(args.solver))
        return
    dt, bdf_scheme, fem_dynamics_init_strategy = load_time_integration(
        args.time_integration
    )
    fem_elasto_dynamics = load_fem_elasto_dynamics(args.fem_elasto_dynamics)

    # TODO: Actually deserialize contact dynamics
    contact_dynamics = pbat.sim.contact.MeshDynamics()
    Xordering, Eordering, XCC, ECC, n_components = (
        pbat.graph.sorted_connected_component_ordering(
            fem_elasto_dynamics.X, fem_elasto_dynamics.E
        )
    )
    contact_meshes = pbat.sim.contact.MultiMesh(
        fem_elasto_dynamics.E, XCC, n_components=n_components
    )
    contact_dynamics.set_dynamic_geometry(contact_meshes)
    contact_dynamics.allocate_environment_contact_data_structures()
    contact_dynamics.initialize_mesh_environment_contact_detection()
    params = load_solver_params(args.solver, args.params, args.overrides)
    prepare = _solver_params[args.solver]["prepare"]
    prepare(fem_elasto_dynamics, contact_dynamics, params)
    dirichlet_constraints, XP, fem_elastic_mesh_names = load_dirichlet_constraints(
        args.dirichlet_constraints
    )
    out_file, out_group = args.output.split(":")[:2]
    fem_elasto_dynamics.set_time_integration_scheme(dt, bdf_scheme)
    fem_elasto_dynamics.set_initial_conditions(
        fem_elasto_dynamics.x, fem_elasto_dynamics.v
    )
    xD = None if dirichlet_constraints is None else fem_elasto_dynamics.x.copy()
    t = 0
    archive = pbat.io.Archive(out_file, flags=pbat.io.AccessMode.Overwrite)
    fem_elasto_dynamics.serialize(archive[f"{out_group}/{t:08d}"])
    while t * dt < args.duration:
        solve = _solver_params[args.solver]["solve"]
        # Apply procedural constraints
        fem_elasto_dynamics.dmask[:] = 0
        if dirichlet_constraints is not None:
            for _, dnodes in dirichlet_constraints.all_transformed_nodes(t, dt):
                fem_elasto_dynamics.dmask[dnodes] = 1
            fem_elasto_dynamics.constrain(fem_elasto_dynamics.dmask)
            xD = fem_elasto_dynamics.x
            for start, end, name in zip(XP[:-1], XP[1:], fem_elastic_mesh_names):
                xD[:, start:end] = dirichlet_constraints.apply(
                    name, xD[:, start:end], t, dt
                )
            fem_elasto_dynamics.x = xD
        # Setup time step optimization problem
        fem_elasto_dynamics.setup_time_integration_optimization(
            initialization_strategy=fem_dynamics_init_strategy
        )
        # Solve time step
        solve(fem_elasto_dynamics, contact_dynamics, params)
        # Step
        fem_elasto_dynamics.step()
        t += 1
        # Write output
        fem_elasto_dynamics.serialize(archive[f"{out_group}/{t:08d}"])


if __name__ == "__main__":
    main()
