"""Generates static and dynamic elasticity benchmark problems from microcanonical samples of a quadratic-manifold ROM.

For each scene and total energy E, a pool of microcanonical samples (q, qdot) is drawn on the scene's quadratic
manifold, and an evenly spaced subset of it becomes the benchmark problems. The total energy E = K + U includes the
gravity potential -(M g)^T u(q) in U when sampling with gravity: by default the dynamic problems are sampled with
gravity (consistent with their inertial targets) and the static ones without (--static-gravity, --dynamic-gravity).
Sets sampled with the same setting share one pool, i.e. the same problems; the dynamic sets always share theirs.

The whole benchmark is one HDF5 file (--output) holding one group per named scene (-i name=scene.h5, repeated),
which benchmark.py and benchmark_browser.py read:

    <scene>/fem/                       the scene's FemElastoDynamics (pbat.io.Archive), whose Dirichlet nodes are held
                                       at rest by every problem. The benchmark solves the scene's problems on it.
    <scene>/X, <scene>/basis/          rest positions and the quadratic-manifold basis, stored once per scene (see
                                       below).
    <scene>/pools/pool/E{E}/           all pooled samples without gravity.
    <scene>/pools/pool_gravity/E{E}/   all pooled samples with gravity.
    <scene>/static/E{E}/               static problems: release x = X + u(q) into a static solve with no loads. The
                                       solution is the rest shape X.
    <scene>/dynamic_h{h}/E{E}/         dynamic problems: one implicit Euler (BDF1) step of size h from
                                       (x, v = J(q) qdot), i.e. min_x 1/(2 h^2) |x - xtilde|_M^2 + Psi(x) with
                                       xtilde = x + h v + h^2 g. samples/x_ref holds a reference solution from a
                                       converged full-order Newton solve.

    scene attributes                   input (the scene file), label (a display name for plots, e.g. "Y = 1e6", when
                                       given with --label), sets and pools (the scene's groups, as paths from the
                                       file root), and the ROM's settings (n_linear_modes, rigid_modes_excluded, ...).
    root attributes                    dataset_format = "pbat-benchmark-dataset", dataset_version = 2, scenes (the
                                       names, in order), sets (every problem set group, e.g. ["Y1e6/static/E10000",
                                       "Y1e6/dynamic_h0.01/E10000", ...]), pools, git_commit, and the generator's
                                       settings.

A problem set is addressed as dataset.h5:<scene>/<set>/E{E}, e.g. dataset.h5:Y1e6/static/E10000.

Each input is a CPU editor scene file (e.g. data/scenarios/trampoline_circle_Y1e6.h5), whose FemElastoDynamics is
built the way the editor builds it, or a stored FemElastoDynamics given as filepath:h5path (e.g. an editor Dump's
fem group). It provides the mesh, masses, Stable Neo-Hookean material, Dirichlet constraints and gravity. The
problems live in the time span [0, max(--dt)] with the Dirichlet nodes fixed at rest and no collision geometry, so
a scene's constraints acting during that span must keep their nodes fixed throughout, and it can't have static
colliders (constraints entirely outside the span are ignored). Its Dirichlet nodes must sit at rest with zero
velocity, since every problem holds them at X. Its time integrator is only recorded: the dynamic problems are always
BDF1 steps with the --dt time steps.

Every pool and problem set group follows the sample-file contract (see rom_mcmc.write_samples), with
ensemble = "total" and sampler = "microcanonical"; its X and basis are hard links to its scene's, so a group reads
like a sample file's root. Dirichlet DOFs stay at rest in every state. The groups add:

    group attributes     problem ("pool", "static" or "dynamic"), problem_definition, scene (the scene's name), fem
                         (the path of the scene's FEM group, e.g. "Y1e6/fem"), pool (the pool group the problems were
                         drawn from, e.g. "Y1e6/pools/pool/E10000"; problem sets only), pool_indices,
                         gravity (3,) the acceleration in the sampled potential (zero without gravity),
                         scene_dt and scene_bdf_steps (the input's BDF time step in s and step count).
    dynamic sets only    attributes h (s), bdf_order = 1, aext (3,) the scene gravity (m/s^2),
                         reference_time (s), and the datasets
                         samples/v                       (S, N, 3) float32 start velocities J(q) qdot,
                         samples/xtilde                  (S, N, 3) float32 inertial targets x + h v + h^2 aext
                                                         (aext on the free DOFs only),
                         samples/x_ref                   (S, N, 3) float32 reference solutions for the stored
                                                         xtilde, from Newton converged to pbat's float32 energy
                                                         noise floor,
                         samples/ref_newton_iterations   (S,) Newton iterations of the reference solves,
                         samples/ref_newton_decrement    (S,) their last Newton decrement,
                         samples/ref_newton_correction   (S,) the largest entry (m) of their last Newton
                                                         correction,
                         samples/ref_error_mnorm         (S,) the M-norm (free DOFs, as the benchmark's error
                                                         e = |x - x_ref|_M) of that last correction: the
                                                         reference's accuracy (an estimate, at the noise floor)
                                                         in the benchmark's own metric,
                         samples/ref_min_J               (S,) minimum det(F) of the reference solutions.

Sampling and the reference solves run on the CPU. The samples' quality is in each pool group: check rhat < 1.05
(otherwise increase --pool or --burn-in) and energy_noise_warning = False.

Usage:
    # Two scenes at the default level (10000 J), 10 problems each, h = 1e-4 and 1e-2 s
    python make_benchmark.py -i Y1e6=path/to/PhysicsBasedAnimationToolkit/data/scenarios/trampoline_circle_Y1e6.h5 \\
        -i Y1e7=path/to/PhysicsBasedAnimationToolkit/data/scenarios/trampoline_circle_Y1e7.h5 \\
        -o path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines.h5

    # Chosen energies, problem counts and time steps; e.g. 100 problems at 10000 J from a pool of 2000
    python make_benchmark.py -i <name>=<scene>.h5 -o <dataset>.h5 --energies 10000 --problems 100 --pool 2000 --dt 1e-4 1e-2

    # All sets sampled with gravity, sharing one pool (paired static and dynamic problems)
    python make_benchmark.py -i <name>=<scene>.h5 -o <dataset>.h5 --static-gravity

    # Static problems only (no dynamic sets)
    python make_benchmark.py -i <name>=<scene>.h5 -o <dataset>.h5 --dt

    # Quick test set (rhat isn't meaningful with such small pools)
    python make_benchmark.py -i <name>=<scene>.h5 -o <dataset>.h5 --problems 3 --pool 40 --chains 2 --burn-in 100 --pilot 100
"""

import argparse
import gc
import os
import time
import typing

import h5py
import numpy as np
from pbatoolkit import pbat

import modalrom
import rom_mcmc
import rom_reach


def subset(batch: rom_mcmc.SampleBatch, indices: np.ndarray) -> rom_mcmc.SampleBatch:
    """The samples of a microcanonical batch at the given indices, with the batch's statistics."""
    return rom_mcmc.SampleBatch(
        batch.q[indices],
        batch.energy[indices],
        batch.log_target[indices],
        dict(batch.stats),
        qdot=batch.qdot[indices],
    )


def parse_scene(spec: str) -> typing.Tuple[str, str]:
    """Splits an --input value name=input into the scene's name and input.

    Args:
        spec (str): name=input, or just input, whose file name without extension is then the name (e.g.
            trampoline_circle_Y1e6 for data/scenarios/trampoline_circle_Y1e6.h5).

    Returns:
        typing.Tuple[str, str]: The scene's name (its group in the dataset) and the input.

    Raises:
        ValueError: If the name is empty or contains / or :, which HDF5 paths and dataset.h5:group addresses reserve.
    """
    name, sep, input = spec.partition("=")
    if not sep:
        input = spec
        i = input.rfind(":")
        name = os.path.splitext(os.path.basename(input[:i] if i > 1 else input))[0]
    if not name or "/" in name or ":" in name:
        raise ValueError(f"Invalid scene name {name!r} in {spec!r}: give a name without / or : as name=input")
    return name, input


def load_scene(input: str, t_end: float = 0.0):
    """Loads the FemElastoDynamics of a CPU editor scene file, or the one stored at filepath:h5path.

    Args:
        input (str): A scene file (e.g. data/scenarios/trampoline.h5), built as the editor does, or filepath:h5path
            of an hdf5 file storing a FemElastoDynamics (e.g. scene.Dump.h5:fem).
        t_end (float): End of the problems' time span in s (the largest time step), for checking a scene's
            constraints.

    Returns:
        pbat.sim.dynamics.FemElastoDynamics: The scene.

    Raises:
        ValueError: If input is a plain mesh file, or a scene with static colliders or with constraints acting
            during [0, t_end] that move their nodes or start or end within it (see modalrom.check_scene_constraints).
    """
    kind = modalrom.input_kind(input)
    if kind == "mesh":
        raise ValueError(
            f"{input} is a mesh: the benchmark needs a scene file (e.g. data/scenarios/trampoline.h5) or "
            "filepath:h5path of a stored FemElastoDynamics (e.g. scene.Dump.h5:fem)"
        )
    return modalrom.load_scene_fem(input, t_end) if kind == "scene" else modalrom.load_fem(input)


def check_scene(fem, atol: float = 1e-6) -> typing.List[str]:
    """Checks that a scene satisfies the problems' assumptions.

    Args:
        fem: The FemElastoDynamics (or any object with X, dirichlet_nodes, dirichlet_coordinates,
            dirichlet_velocities and bdf.s members).
        atol (float): Absolute tolerance on the Dirichlet positions (m) and velocities (m/s).

    Returns:
        typing.List[str]: Warnings about assumptions that don't hold but don't invalidate the problems.

    Raises:
        ValueError: If Dirichlet nodes aren't at rest or move, since every problem holds them at X.
    """
    nodes = np.asarray(fem.dirichlet_nodes)
    X = np.asarray(fem.X)
    if nodes.size:
        drift = np.abs(np.asarray(fem.dirichlet_coordinates) - X[:, nodes]).max()
        speed = np.abs(np.asarray(fem.dirichlet_velocities)).max()
        if drift > atol or speed > atol:
            raise ValueError(
                f"The scene's Dirichlet nodes must be at rest with zero velocity (max displacement {drift:.3g} m, "
                f"max speed {speed:.3g} m/s), since the problems hold them at X"
            )
    warnings = []
    if fem.bdf.s != 1:
        warnings.append(
            f"The scene integrates with BDF{fem.bdf.s}, but the dynamic problems are BDF1 steps (its "
            "scene_bdf_steps and scene_dt are recorded)"
        )
    return warnings


def write_fem(path: str, fems: typing.Dict[str, typing.Any]):
    """Creates (or overwrites) the dataset file with each scene's FemElastoDynamics in its <scene>/fem group.

    Args:
        path (str): The dataset file.
        fems (typing.Dict[str, pbat.sim.dynamics.FemElastoDynamics]): The FEM each scene's problems are built on, by
            scene name.
    """
    for i, (name, fem) in enumerate(fems.items()):
        mode = pbat.io.AccessMode.Overwrite if i == 0 else pbat.io.AccessMode.ReadWrite
        archive = pbat.io.Archive(path, flags=mode)
        fem.serialize(archive[f"{name}/fem"])
        archive = None
        gc.collect()  # Closes the archive before it's reopened, or h5py opens the file


class SamplingPlan(typing.NamedTuple):
    """One pool of samples per energy, and the problem sets drawn from it."""

    pool: str
    gravity: bool
    static: bool
    dynamic: bool


def sampling_plan(static_gravity: bool, dynamic_gravity: bool, has_dynamic: bool) -> typing.List[SamplingPlan]:
    """The pools to sample for the requested gravity settings.

    Static and dynamic sets share one pool, so they're paired (the same samples), when they're sampled with the
    same gravity setting. Otherwise each gets its own pool and the pairing is only kept between the dynamic sets.

    Args:
        static_gravity (bool): Whether the static problems' samples include the gravity potential.
        dynamic_gravity (bool): Whether the dynamic problems' samples include the gravity potential.
        has_dynamic (bool): Whether any dynamic set is requested.

    Returns:
        typing.List[SamplingPlan]: The pools, the scene's groups pools/pool (without gravity) or pools/pool_gravity
        (with gravity).
    """
    pool = lambda gravity: "pools/pool_gravity" if gravity else "pools/pool"
    if not has_dynamic:
        return [SamplingPlan(pool(static_gravity), static_gravity, True, False)]
    if static_gravity == dynamic_gravity:
        return [SamplingPlan(pool(static_gravity), static_gravity, True, True)]
    return [
        SamplingPlan(pool(static_gravity), static_gravity, True, False),
        SamplingPlan(pool(dynamic_gravity), dynamic_gravity, False, True),
    ]


def min_det_F(E, egU, GNegU, model, qs) -> np.ndarray:
    """Minimum det(F) over the mesh of each shape X + u(q) for q in qs (<= 0 means inverted elements)."""
    return np.array(
        [
            modalrom.deformation_gradient_determinants(
                E, egU, GNegU, model.x0 + model.displacement(q)
            ).min()
            for q in qs
        ]
    )


def write_scene(dataset: h5py.File, name: str, input: str, scene, args) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """Samples one scene's pools and writes them and its problem sets into the scene's group.

    Args:
        dataset (h5py.File): The dataset file, whose group name/fem already holds the scene's FemElastoDynamics.
        name (str): The scene's name, its group in the dataset.
        input (str): The scene's input, recorded.
        scene (pbat.sim.dynamics.FemElastoDynamics): The scene.
        args (argparse.Namespace): The generator's settings.

    Returns:
        typing.Tuple[typing.List[str], typing.List[str]]: The scene's problem set and pool groups, as paths from the
        file root.
    """
    element = pbat.fem.Element.Tetrahedron
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, gravity = modalrom.model_from_fem(scene)
    n, N = X.size, X.shape[1]
    w, U, Q, Z, ws = modalrom.build_rom(
        E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, energy, element, args.modes, args.eps
    )
    psi = modalrom.elastic_energy(E, X, egU, wgU, GNegU, mug, lambdag, energy, element)
    x0 = X.flatten(order="F").astype(np.float64)
    m = M.diagonal().astype(np.float64)
    rigid = 6 if len(freedofs) == n else 0
    model, wr = modalrom.reduced_model(
        "quadratic", U, Q, w, m, x0, psi, rigid, "mass"
    )
    fixed = np.setdiff1d(np.arange(n), freedofs)
    # Gravity acts on the free DOFs only; Dirichlet DOFs stay at rest
    accel = rom_reach.gravity_load(np.ones(n), gravity)
    accel[fixed] = 0.0
    # The same model with the gravity potential -(M g)^T u(q), for pools sampled with gravity
    model_gravity, _ = modalrom.reduced_model(
        "quadratic", U, Q, w, m, x0, psi, rigid, "mass", rom_reach.gravity_load(m, gravity)
    )
    group = dataset[name]
    settings = dict(
        input=input,
        n_linear_modes=args.modes,
        rigid_modes_excluded=rigid,
        modal_derivatives=True,
        mode_normalization="mass",
        eps=args.eps,
        scene_dt=float(scene.bdf.h),
        scene_bdf_steps=int(scene.bdf.s),
    )
    common = dict(
        settings,
        sampler="microcanonical",
        seed=args.seed,
        git_commit=args.git_commit,
        scene=name,
        fem=f"{name}/fem",
    )
    plans = sampling_plan(args.static_gravity, args.dynamic_gravity, len(args.dt) > 0)
    group.attrs.update(settings)
    group.create_dataset("X", data=np.ascontiguousarray(X.T))
    rom_mcmc.write_basis(group.create_group("basis"), model, wr)
    sets, pools = [], []

    for Etot in args.energies:
        level = f"E{Etot:g}"
        for plan in plans:
            sampled_model = model_gravity if plan.gravity else model
            t0 = time.time()
            pool = rom_mcmc.run_chains(
                lambda rng: rom_mcmc.make_sampler("microcanonical", sampled_model, Etot, rng),
                args.chains,
                args.pool,
                args.burn_in,
                args.pilot,
                seed=args.seed,
            )
            attrs = dict(
                common,
                gravity=np.asarray(gravity) if plan.gravity else np.zeros(3),
                target_energy=Etot,
                sampling_time=time.time() - t0,
            )
            pool_group = f"{plan.pool}/{level}"
            rom_mcmc.write_samples(
                group.create_group(pool_group),
                X,
                sampled_model,
                wr,
                pool,
                dict(attrs, problem="pool"),
                min_det_F(E, egU, GNegU, sampled_model, pool.q),
                shared=group,
            )
            pools.append(f"{name}/{pool_group}")
            indices = np.round(np.linspace(0, pool.q.shape[0] - 1, args.problems)).astype(np.int64)
            problems = subset(pool, indices)
            problem_attrs = dict(attrs, pool=f"{name}/{pool_group}", pool_indices=indices)
            start_min_J = min_det_F(E, egU, GNegU, sampled_model, problems.q)
            if plan.static:
                rom_mcmc.write_samples(
                    group.create_group(f"static/{level}"),
                    X,
                    sampled_model,
                    wr,
                    problems,
                    dict(
                        problem_attrs,
                        problem="static",
                        problem_definition="Release samples/x into a static solve of Psi with no loads and the "
                        "Dirichlet DOFs at rest. The solution is X.",
                    ),
                    start_min_J,
                    shared=group,
                )
                sets.append(f"{name}/static/{level}")
            # Dynamic problems: one BDF1 step from (x, v) for each time step. Future work: BDF schemes with s > 1
            # steps need s past states per problem, e.g. synthesized by integrating the reduced dynamics backward
            # from the sample, and the inertial target of that scheme.
            if plan.dynamic:
                ux = np.array([sampled_model.displacement(q) for q in problems.q])
                v = np.array([sampled_model.jacobian(q) @ qd for q, qd in zip(problems.q, problems.qdot)])
            for h in args.dt if plan.dynamic else []:
                t = time.time()
                xtilde, xref, iterations, decrements, corrections, errors, ref_min_J = [], [], [], [], [], [], []
                e0s = []
                for k in range(args.problems):
                    ut = ux[k] + h * v[k] + h**2 * accel
                    ut[fixed] = 0.0
                    # Solve for the inertial target as stored (float32), which is what the benchmark's solvers read
                    xtilde32 = (x0 + ut).astype(np.float32)
                    ut = xtilde32.astype(np.float64) - x0
                    uref, info = rom_reach.static_equilibrium(
                        psi,
                        x0,
                        freedofs,
                        inertia=(m / h**2, ut),
                        u0=ut,
                        steps=1,
                        max_iters=args.newton_iters,
                    )
                    xtilde.append(xtilde32.reshape(N, 3))
                    xref.append((x0 + uref).reshape(N, 3))
                    iterations.append(info["iterations"])
                    decrements.append(info["decrement"])
                    corrections.append(info["correction"])
                    du = info["last_correction"]
                    errors.append(float(np.sqrt(du @ (m * du))))
                    step = uref - ux[k]
                    e0s.append(float(np.sqrt(step @ (m * step))))
                    ref_min_J.append(
                        modalrom.deformation_gradient_determinants(E, egU, GNegU, x0 + uref).min()
                    )
                rom_mcmc.write_samples(
                    group.create_group(f"dynamic_h{h:g}/{level}"),
                    X,
                    sampled_model,
                    wr,
                    problems,
                    dict(
                        problem_attrs,
                        problem="dynamic",
                        h=h,
                        bdf_order=1,
                        aext=np.asarray(gravity),
                        reference_time=time.time() - t,
                        problem_definition="One implicit Euler step: minimize 1/(2 h^2) |x - xtilde|_M^2 + Psi(x) "
                        "with the Dirichlet DOFs at rest, starting from samples/x with velocity samples/v. "
                        "xtilde = x + h v + h^2 aext on the free DOFs. samples/x_ref is a reference solution for the "
                        "stored xtilde from full-order Newton, converged to the float32 energy noise floor; "
                        "samples/ref_error_mnorm is the M-norm of its last Newton correction, an estimate of its accuracy "
                        "in the benchmark's error metric |x - x_ref|_M.",
                    ),
                    start_min_J,
                    dict(
                        v=v.reshape(-1, N, 3).astype(np.float32),
                        xtilde=np.array(xtilde, dtype=np.float32),
                        x_ref=np.array(xref, dtype=np.float32),
                        ref_newton_iterations=np.array(iterations),
                        ref_newton_decrement=np.array(decrements),
                        ref_newton_correction=np.array(corrections),
                        ref_error_mnorm=np.array(errors),
                        ref_min_J=np.array(ref_min_J),
                    ),
                    shared=group,
                )
                sets.append(f"{name}/dynamic_h{h:g}/{level}")
                print(
                    f"{name}, E={Etot:g} J, h={h:g} s: reference Newton iterations {np.min(iterations)}-"
                    f"{np.max(iterations)}, accuracy |last correction|_M <= {max(e / e0 for e, e0 in zip(errors, e0s)):.1e} "
                    f"of e0 = |x - x_ref|_M, {time.time() - t:.1f} s"
                )
            stats = pool.stats
            print(
                f"{name}, E={Etot:g} J, {plan.pool} ({'with' if plan.gravity else 'without'} gravity, for "
                f"{' and '.join(kind for kind, used in [('static', plan.static), ('dynamic', plan.dynamic)] if used)} "
                f"problems): {pool.q.shape[0]} samples in {attrs['sampling_time']:.1f} s (split R-hat "
                f"{stats['rhat']:.3f}, ESS {stats['ess']:.0f}), {args.problems} problems, start min det F "
                f"{start_min_J.min():.3f} ({int((start_min_J <= 0).sum())} inverted)"
            )
            group.attrs["sets"], group.attrs["pools"] = sets, pools
            dataset.flush()
    return sets, pools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Static and dynamic elasticity benchmark problems from microcanonical ROM samples",
        description="Writes one benchmark dataset file (--output) holding, for each named scene, its FEM, pools, and "
        "static and dynamic_h{h} problem sets per total energy. "
        "See the module docstring for the file contents and representative commands.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="A scene of the benchmark as name=input, repeated for several scenes; the name is the scene's group in "
        "the dataset (no / or :; default: the input's file name without extension). The input is a CPU editor scene "
        "file (e.g. data/scenarios/trampoline_circle_Y1e6.h5) or filepath:h5path of an hdf5 file storing "
        "a FemElastoDynamics (e.g. scene.Dump.h5:fem), which provides the mesh, material, masses, Dirichlet "
        "constraints and gravity. A scene's constraints acting during [0, max(--dt)] must keep their nodes fixed "
        "throughout, and it can't have static colliders; "
        "meshes are rejected",
        type=str,
        action="append",
        dest="inputs",
        required=True,
    )
    parser.add_argument(
        "--label",
        help="A scene's display name for plots as name=label (e.g. 'Y1e6=Y = 1e6'), repeated per scene; stored as "
        "the scene group's label attribute (default: none)",
        type=str,
        action="append",
        dest="labels",
        default=[],
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output dataset file (.h5), overwritten",
        type=str,
        dest="output",
        required=True,
    )
    parser.add_argument(
        "--energies",
        help="Total energies E = K + U in J, one problem set per scene, energy and kind. On the trampoline, 50 / 500 / "
        "5000 J give median peak displacements of ~14 / 24 / 36 cm (default: %(default)s)",
        type=float,
        nargs="+",
        dest="energies",
        default=[10000.0],
    )
    parser.add_argument(
        "--problems",
        help="Number of problems per scene and energy, an evenly spaced subset of the pool; at most --pool, and "
        "preferably well below its effective sample size (default: %(default)s)",
        type=int,
        dest="problems",
        default=10,
    )
    parser.add_argument(
        "--pool",
        help="Number of pooled samples per scene and energy; large enough for rhat < 1.05 (default: %(default)s)",
        type=int,
        dest="pool",
        default=400,
    )
    parser.add_argument(
        "--chains",
        help="Number of independent chains per pool (default: %(default)s)",
        type=int,
        dest="chains",
        default=4,
    )
    parser.add_argument(
        "--burn-in",
        help="Burn-in steps per chain, during which the step size adapts (default: %(default)s)",
        type=int,
        dest="burn_in",
        default=300,
    )
    parser.add_argument(
        "--pilot",
        help="Steps per chain used to estimate the autocorrelation time for thinning (default: %(default)s)",
        type=int,
        dest="pilot",
        default=500,
    )
    parser.add_argument(
        "--dt",
        help="Time steps h in s of the dynamic problem sets, one set per value; pass no value for static problems "
        "only (default: %(default)s)",
        type=float,
        nargs="*",
        dest="dt",
        default=[1e-4, 1e-2],
    )
    parser.add_argument(
        "--static-gravity",
        help="Include the scene's gravity potential -(M g)^T u(q) in the total energy when sampling the static "
        "problems (default: off). The static problems themselves are solved without loads either way",
        action=argparse.BooleanOptionalAction,
        dest="static_gravity",
        default=False,
    )
    parser.add_argument(
        "--dynamic-gravity",
        help="Include the scene's gravity potential -(M g)^T u(q) in the total energy when sampling the dynamic "
        "problems (default: on), consistent with their inertial targets, which always include gravity. When static "
        "and dynamic settings differ, the two kinds of sets use separate pools and no longer share samples",
        action=argparse.BooleanOptionalAction,
        dest="dynamic_gravity",
        default=True,
    )
    parser.add_argument(
        "-k",
        "--num-modes",
        help="Number of vibration modes m of each scene's quadratic manifold (default: %(default)s)",
        type=int,
        dest="modes",
        default=20,
    )
    parser.add_argument(
        "--eps",
        help="Finite-difference step of the modal derivatives, as a fraction of the bounding box diagonal "
        "(default: %(default)s)",
        type=float,
        dest="eps",
        default=1e-3,
    )
    parser.add_argument(
        "--newton-iters",
        help="Maximum Newton iterations of each reference solve of a dynamic problem (default: %(default)s)",
        type=int,
        dest="newton_iters",
        default=100,
    )
    parser.add_argument(
        "--seed",
        help="Seed of the chains' random streams, the same for every scene (default: %(default)s)",
        type=int,
        dest="seed",
        default=0,
    )
    args = parser.parse_args()
    if args.problems > args.pool:
        parser.error(f"--problems ({args.problems}) can't exceed --pool ({args.pool}): problems are drawn from the pool")
    scenes = {}
    try:
        for spec in args.inputs:
            name, input = parse_scene(spec)
            if name in scenes:
                raise ValueError(f"Scene name {name!r} is given twice: name the scenes as name=input")
            scene = load_scene(input, max(args.dt, default=0.0))
            for warning in check_scene(scene):
                print(f"Warning ({name}): {warning}")
            scenes[name] = (input, scene)
        labels = dict(label.split("=", 1) for label in args.labels if "=" in label)
        if len(labels) != len(args.labels) or not set(labels) <= set(scenes):
            raise ValueError(f"--label takes name=label with the name of a scene ({', '.join(scenes)})")
    except ValueError as error:
        parser.error(str(error))
    args.git_commit = rom_mcmc.git_commit(os.path.dirname(os.path.abspath(__file__)))

    write_fem(args.output, {name: scene for name, (_, scene) in scenes.items()})
    dataset = h5py.File(args.output, "a")
    dataset.attrs.update(
        dataset_format="pbat-benchmark-dataset",
        dataset_version=2,
        scenes=list(scenes),
        energies=np.asarray(args.energies, dtype=np.float64),
        dt=np.asarray(args.dt, dtype=np.float64),
        problems=args.problems,
        pool_size=args.pool,
        chains=args.chains,
        burn_in=args.burn_in,
        pilot=args.pilot,
        newton_iters=args.newton_iters,
        static_gravity=args.static_gravity,
        dynamic_gravity=args.dynamic_gravity,
        n_linear_modes=args.modes,
        eps=args.eps,
        sampler="microcanonical",
        seed=args.seed,
        git_commit=args.git_commit,
    )
    sets, pools = [], []
    for name, (input, scene) in scenes.items():
        t = time.time()
        if name in labels:
            dataset[name].attrs["label"] = labels[name]
        scene_sets, scene_pools = write_scene(dataset, name, input, scene, args)
        sets += scene_sets
        pools += scene_pools
        dataset.attrs["sets"], dataset.attrs["pools"] = sets, pools
        dataset.flush()
        print(f"{name}: {len(scene_sets)} problem sets in {time.time() - t:.1f} s")
    dataset.close()
