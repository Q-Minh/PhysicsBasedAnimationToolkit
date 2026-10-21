"""Recomputes the energy-noise attributes of a benchmark dataset's microcanonical pools in place, without resampling.

Datasets generated before the microcanonical sampler measured the energy noise at its samples (rom_mcmc.
samples_energy_noise) carry a noise estimate taken at a probe on the linearized level set U = E. On soft scenes that
probe lies far outside U < E (e.g. 1e13 J at E = 1e4 J), so energy_noise, energy_tolerance and energy_noise_warning
were grossly overstated; the samples themselves don't depend on it. This tool rebuilds each scene's reduced model
from the dataset (its stored FEM and basis), recomputes the statistics from each pool's stored samples exactly as
the sampler now does (rom_mcmc.pool_energy_noise), and writes them to the pool and to every problem set drawn from
it, with energy_noise_recomputed set to this checkout's git commit.

Usage (from python/examples):

    python recompute_energy_noise.py -d path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines.h5 --dry-run
    python recompute_energy_noise.py -d path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines.h5
"""

import argparse
import gc
import os

import h5py
import numpy as np
from pbatoolkit import pbat

import modalrom
import rom_mcmc
import rom_reach

KEYS = ("energy_noise", "energy_tolerance", "energy_noise_warning")


def scene_model(dataset: str, scene: str, basis: dict, gravity: np.ndarray) -> rom_mcmc.QuadraticManifold:
    """The reduced model a scene's pools were sampled on, from the dataset's stored FEM and basis.

    Args:
        dataset (str): The dataset file.
        scene (str): The scene's group name.
        basis (dict): The scene's basis/{U, Q, M, scale} arrays.
        gravity (np.ndarray): 3 gravitational acceleration of the sampled potential (zero without gravity).

    Returns:
        rom_mcmc.QuadraticManifold: The model, as make_benchmark builds it.
    """
    fem = modalrom.load_fem(f"{dataset}:{scene}/fem")
    E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, _ = modalrom.model_from_fem(fem)
    psi = modalrom.elastic_energy(
        E, X, egU, wgU, GNegU, mug, lambdag, pbat.fem.HyperElasticEnergy.StableNeoHookean, pbat.fem.Element.Tetrahedron
    )
    del fem
    gc.collect()
    m = basis["M"]
    fext = rom_reach.gravity_load(m, gravity) if np.any(gravity) else None
    x0 = X.flatten(order="F").astype(np.float64)
    return rom_mcmc.QuadraticManifold(basis["U"], basis["Q"], m, x0, psi, basis["scale"], "mass", fext)


def recompute(dataset: str, dry_run: bool = False) -> dict:
    """Recomputes the energy-noise statistics of every microcanonical pool of a dataset.

    Args:
        dataset (str): The dataset file (format version 2).
        dry_run (bool): Only compute, don't write.

    Returns:
        dict: {pool path: (old statistics, new statistics)}.
    """
    with h5py.File(dataset, "r") as f:
        if f.attrs.get("dataset_format") != "pbat-benchmark-dataset" or f.attrs.get("dataset_version") != 2:
            raise ValueError(f"{dataset} isn't a version-2 benchmark dataset")
        scenes = {}
        for scene in f.attrs["scenes"]:
            basis = {k: f[f"{scene}/basis/{k}"][()] for k in ("U", "Q", "M", "scale")}
            pools = {}
            for pool in f[scene].attrs["pools"]:
                g = f[pool]
                if g.attrs["sampler"] != "microcanonical":
                    continue
                pools[str(pool)] = dict(
                    q=g["samples/q"][()],
                    energy=g["samples/energy"][()],
                    chains=int(g.attrs["n_chains"]),
                    target_energy=float(g.attrs["target_energy"]),
                    gravity=np.asarray(g.attrs["gravity"], dtype=np.float64),
                    old={k: g.attrs[k] for k in KEYS},
                )
            scenes[str(scene)] = (basis, pools)
        sets = {str(s): str(f[s].attrs["pool"]) for s in f.attrs["sets"]}

    results = {}
    for scene, (basis, pools) in scenes.items():
        for path, pool in pools.items():
            model = scene_model(dataset, scene, basis, pool["gravity"])
            # The rebuilt model must reproduce the stored sample energies, or the recomputed noise would be meaningless
            if not np.isclose(model.energy(pool["q"][0]), pool["energy"][0], rtol=1e-6, atol=1e-6 * pool["target_energy"]):
                raise RuntimeError(f"{path}: the rebuilt model doesn't reproduce the stored sample energies")
            noise = rom_mcmc.pool_energy_noise(model, pool["q"], pool["chains"])
            results[path] = (pool["old"], rom_mcmc.energy_noise_stats(noise, pool["target_energy"]))

    if not dry_run:
        commit = rom_mcmc.git_commit(os.path.dirname(os.path.abspath(__file__)))
        with h5py.File(dataset, "r+") as f:
            for path, (_, new) in results.items():
                for group in [path] + [s for s, p in sets.items() if p == path]:
                    f[group].attrs.update(new, energy_noise_recomputed=commit)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Recompute energy noise",
        description="Recomputes the energy-noise attributes of a benchmark dataset's microcanonical pools (and their "
        "problem sets) in place from the stored samples, without resampling.",
    )
    parser.add_argument("-d", "--dataset", required=True, help="Benchmark dataset (.h5, format version 2)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Print the new values without writing")
    args = parser.parse_args()
    for path, (old, new) in recompute(args.dataset, args.dry_run).items():
        print(
            f"{path}: energy_noise {old['energy_noise']:.3g} -> {new['energy_noise']:.3g} J, energy_tolerance "
            f"{old['energy_tolerance']:.3g} -> {new['energy_tolerance']:.3g} J, energy_noise_warning "
            f"{bool(old['energy_noise_warning'])} -> {new['energy_noise_warning']}"
        )
    print("(dry run, nothing written)" if args.dry_run else f"Updated {args.dataset}")
