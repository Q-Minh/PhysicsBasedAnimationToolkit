"""Generates static and dynamic elasticity benchmark problems from microcanonical samples of a quadratic-manifold ROM.

For each total energy E, a pool of microcanonical samples (q, qdot) is drawn on the quadratic manifold, and an
evenly spaced subset of it becomes the benchmark problems. The same subset is used in every problem set:

    <output>/pool/E{E}.h5          all pooled samples.
    <output>/static/E{E}.h5        static problems: release x = X + u(q) into a static solve with no loads.
                                   The solution is the rest shape X.
    <output>/dynamic_h{h}/E{E}.h5  dynamic problems: one implicit Euler (BDF1) step of size h from
                                   (x, v = J(q) qdot), i.e. min_x 1/(2 h^2) |x - xtilde|_M^2 + Psi(x) with
                                   xtilde = x + h v + h^2 g. samples/x_ref holds a reference solution from a
                                   converged full-order Newton solve.

All files follow the sample-file contract (see rom_mcmc.write_samples), with ensemble = "total" and
sampler = "microcanonical". Dirichlet DOFs stay at rest in every state. Problem files add:

    root attributes      problem ("pool", "static" or "dynamic"), problem_definition, pool_file, pool_indices.
    dynamic files only   root attributes h (s), bdf_order = 1, aext (3,) the scene gravity (m/s^2),
                         reference_time (s), and the datasets
                         samples/v                       (S, N, 3) float32 start velocities J(q) qdot,
                         samples/xtilde                  (S, N, 3) float32 inertial targets x + h v + h^2 aext
                                                         (aext on the free DOFs only),
                         samples/x_ref                   (S, N, 3) float32 reference solutions,
                         samples/ref_newton_iterations   (S,) Newton iterations of the reference solves,
                         samples/ref_newton_decrement    (S,) their last Newton decrement,
                         samples/ref_newton_correction   (S,) the largest entry (m) of their last Newton
                                                         correction, an accuracy estimate,
                         samples/ref_min_J               (S,) minimum det(F) of the reference solutions.

Sampling and the reference solves run on the CPU. The samples' quality is in each pool file: check rhat < 1.05
(otherwise increase --pool or --burn-in) and energy_noise_warning = False.

Usage:
    # Default levels (50, 500, 5000 J), 10 problems each, h = 1e-4 and 1e-2 s
    python make_benchmark.py -i C:\\git\\PhysicsBasedAnimationToolkit\\data\\scenarios\\trampoline.Dump.h5:fem -o C:\\git\\PhysicsBasedAnimationToolkit\\data\\benchmarks\\trampoline

    # Chosen energies, problem counts and time steps
    python make_benchmark.py -i <Dump>.h5:fem -o <dir> --energies 50 5000 --problems 5 --dt 1e-4 1e-2

    # Static problems only (no dynamic sets)
    python make_benchmark.py -i <Dump>.h5:fem -o <dir> --dt

    # Quick test set (rhat isn't meaningful with such small pools)
    python make_benchmark.py -i <Dump>.h5:fem -o <dir> --energies 50 5000 --problems 3 --pool 40 --chains 2 --burn-in 100 --pilot 100
"""

import argparse
import os
import time

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Static and dynamic elasticity benchmark problems from microcanonical ROM samples",
        description="Writes pool/, static/ and dynamic_h{h}/ problem files per total energy under --output. "
        "See the module docstring for the file contents and representative commands.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="filepath:h5path of an hdf5 file containing an FemElastoDynamics (e.g. scene.Dump.h5:fem), which "
        "provides the mesh, material, masses, Dirichlet constraints and gravity",
        type=str,
        dest="input",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory; existing files with the same names are overwritten",
        type=str,
        dest="output",
        required=True,
    )
    parser.add_argument(
        "--energies",
        help="Total energies E = K + U in J, one problem file per energy and set. On the trampoline, 50 / 500 / "
        "5000 J give median peak displacements of ~14 / 24 / 36 cm (default: %(default)s)",
        type=float,
        nargs="+",
        dest="energies",
        default=[50.0, 500.0, 5000.0],
    )
    parser.add_argument(
        "--problems",
        help="Number of problems per energy, an evenly spaced subset of the pool shared by all sets "
        "(default: %(default)s)",
        type=int,
        dest="problems",
        default=10,
    )
    parser.add_argument(
        "--pool",
        help="Number of pooled samples per energy; large enough for rhat < 1.05 (default: %(default)s)",
        type=int,
        dest="pool",
        default=400,
    )
    parser.add_argument(
        "--chains",
        help="Number of independent chains per energy, pooled (default: %(default)s)",
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
        "-k",
        "--num-modes",
        help="Number of vibration modes m of the quadratic manifold (default: %(default)s)",
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
        help="Seed of the chains' random streams (default: %(default)s)",
        type=int,
        dest="seed",
        default=0,
    )
    args = parser.parse_args()

    element = pbat.fem.Element.Tetrahedron
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, gravity = modalrom.load_model(
        args.input, 0.0, 0.0, 0.0, element, energy
    )
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
    common = dict(
        input=args.input,
        n_linear_modes=args.modes,
        rigid_modes_excluded=rigid,
        modal_derivatives=True,
        mode_normalization="mass",
        eps=args.eps,
        sampler="microcanonical",
        gravity=np.zeros(3),
        seed=args.seed,
        git_commit=rom_mcmc.git_commit(os.path.dirname(os.path.abspath(__file__))),
    )
    for directory in ["pool", "static"] + [f"dynamic_h{h:g}" for h in args.dt]:
        os.makedirs(os.path.join(args.output, directory), exist_ok=True)

    for Etot in args.energies:
        t0 = time.time()
        pool = rom_mcmc.run_chains(
            lambda rng: rom_mcmc.make_sampler("microcanonical", model, Etot, rng),
            args.chains,
            args.pool,
            args.burn_in,
            args.pilot,
            seed=args.seed,
        )
        attrs = dict(common, target_energy=Etot, sampling_time=time.time() - t0)
        name = f"E{Etot:g}.h5"
        rom_mcmc.write_samples(
            os.path.join(args.output, "pool", name),
            X,
            model,
            wr,
            pool,
            dict(attrs, problem="pool"),
            min_det_F(E, egU, GNegU, model, pool.q),
        )
        indices = np.round(np.linspace(0, pool.q.shape[0] - 1, args.problems)).astype(np.int64)
        problems = subset(pool, indices)
        problem_attrs = dict(attrs, pool_file=os.path.join("pool", name), pool_indices=indices)
        start_min_J = min_det_F(E, egU, GNegU, model, problems.q)
        rom_mcmc.write_samples(
            os.path.join(args.output, "static", name),
            X,
            model,
            wr,
            problems,
            dict(
                problem_attrs,
                problem="static",
                problem_definition="Release samples/x into a static solve of Psi with no loads and the "
                "Dirichlet DOFs at rest. The solution is X.",
            ),
            start_min_J,
        )
        # Dynamic problems: one BDF1 step from (x, v) for each time step
        ux = np.array([model.displacement(q) for q in problems.q])
        v = np.array([model.jacobian(q) @ qd for q, qd in zip(problems.q, problems.qdot)])
        for h in args.dt:
            t = time.time()
            xtilde, xref, iterations, decrements, corrections, ref_min_J = [], [], [], [], [], []
            for s in range(args.problems):
                ut = ux[s] + h * v[s] + h**2 * accel
                ut[fixed] = 0.0
                uref, info = rom_reach.static_equilibrium(
                    psi,
                    x0,
                    freedofs,
                    inertia=(m / h**2, ut),
                    u0=ut,
                    steps=1,
                    max_iters=args.newton_iters,
                )
                xtilde.append((x0 + ut).reshape(N, 3))
                xref.append((x0 + uref).reshape(N, 3))
                iterations.append(info["iterations"])
                decrements.append(info["decrement"])
                corrections.append(info["correction"])
                ref_min_J.append(
                    modalrom.deformation_gradient_determinants(E, egU, GNegU, x0 + uref).min()
                )
            rom_mcmc.write_samples(
                os.path.join(args.output, f"dynamic_h{h:g}", name),
                X,
                model,
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
                    "xtilde = x + h v + h^2 aext on the free DOFs. samples/x_ref is a reference solution from "
                    "full-order Newton, converged to the float32 energy noise floor; samples/ref_newton_correction is the "
                    "largest entry (m) of its last Newton correction, an estimate of its accuracy.",
                ),
                start_min_J,
                dict(
                    v=v.reshape(-1, N, 3).astype(np.float32),
                    xtilde=np.array(xtilde, dtype=np.float32),
                    x_ref=np.array(xref, dtype=np.float32),
                    ref_newton_iterations=np.array(iterations),
                    ref_newton_decrement=np.array(decrements),
                    ref_newton_correction=np.array(corrections),
                    ref_min_J=np.array(ref_min_J),
                ),
            )
            print(
                f"E={Etot:g} J, h={h:g} s: reference Newton iterations {np.min(iterations)}-{np.max(iterations)}, "
                f"{time.time() - t:.1f} s"
            )
        s = pool.stats
        print(
            f"E={Etot:g} J: pool of {pool.q.shape[0]} samples in {attrs['sampling_time']:.1f} s "
            f"(split R-hat {s['rhat']:.3f}, ESS {s['ess']:.0f}), {args.problems} problems, "
            f"start min det F {start_min_J.min():.3f}"
        )
