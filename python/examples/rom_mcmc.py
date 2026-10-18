"""Sampling of fixed-energy shapes on reduced (linear or quadratic) displacement manifolds.

The target set is the energy level set {q : Psi(x0 + u(q)) - Psi(x0) = E*}, where u(q) is the
reduced displacement map and Psi the full-space hyperelastic energy. Samples follow the density
pi(q) ~ w(q) delta(Psi_hat(q) - E*) dq, where the weight w selects the distribution:

- "mass": sqrt(det(J(q)^T M J(q))), the microcanonical measure in the mass metric of the shapes,
  which does not depend on how q is scaled,
- "uniform": 1, the microcanonical (coarea) measure in q,
- "area": |grad_q Psi_hat|, the Euclidean surface measure of the level set in q.

Samplers work in whitened coordinates y = q / scale, in which the linearized energy is |y|^2 / 2.
They all expose `burn_in`, `estimate_thinning` and `sample`, so that `run_chain` can drive any of them.
"""

import dataclasses
import subprocess
import typing

import numpy as np
import scipy.optimize

WEIGHTS = ("mass", "uniform", "area")


class QuadraticManifold:
    """Reduced model u(q) = U q + 1/2 Q:qq (or u(q) = U q if Q is None) over a full-space energy."""

    def __init__(
        self,
        U: np.ndarray,
        Q: np.ndarray | None,
        M: np.ndarray,
        x0: np.ndarray,
        potential: typing.Callable,
        scale: np.ndarray,
        weight: str = "mass",
    ):
        """
        Args:
            U (np.ndarray): n x m linear basis.
            Q (np.ndarray | None): n x m x m symmetric quadratic basis, or None for the linear map.
            M (np.ndarray): n lumped mass matrix diagonal.
            x0 (np.ndarray): n rest positions in F order (interleaved per node).
            potential (typing.Callable): potential(x, gradient) returns Psi(x), or (Psi(x), grad Psi(x)) if gradient.
            scale (np.ndarray): m whitening scales s.t. q = scale * y.
            weight (str): One of WEIGHTS.
        """
        if weight not in WEIGHTS:
            raise ValueError(f"Unknown weight {weight}, expected one of {WEIGHTS}")
        self.U = np.ascontiguousarray(U)
        self.Q = np.ascontiguousarray(Q) if Q is not None else None
        self.M = M
        self.x0 = x0
        self.potential = potential
        self.scale = scale
        self.weight = weight
        self.dims = U.shape[1]
        self.rest_energy = float(potential(x0, False))
        if self.Q is not None:
            n, m = self.U.shape
            # 2D views of Q for the quadratic term and the Jacobian (no copies, since Q is contiguous)
            self._Qnmm = self.Q.reshape(n, m * m)
            self._Qnm_m = self.Q.reshape(n * m, m)

    def displacement(self, q: np.ndarray) -> np.ndarray:
        u = self.U @ q
        if self.Q is not None:
            u += 0.5 * (self._Qnmm @ np.outer(q, q).ravel())
        return u

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        if self.Q is None:
            return self.U
        return self.U + (self._Qnm_m @ q).reshape(self.U.shape)

    def energy(self, q: np.ndarray) -> float:
        """Deformation energy Psi(x0 + u(q)) - Psi(x0)."""
        return float(self.potential(self.x0 + self.displacement(q), False)) - self.rest_energy

    def energy_and_gradient(self, q: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        e, g = self.potential(self.x0 + self.displacement(q), True)
        return float(e) - self.rest_energy, self.jacobian(q).T @ g

    def ray_energy(self, d: np.ndarray) -> typing.Callable[[float], float]:
        """Returns r -> energy(r * d), using u(r d) = r U d + r^2 (1/2 Q:dd) to avoid the map's products per call."""
        a = self.U @ d
        b = 0.5 * (self._Qnmm @ np.outer(d, d).ravel()) if self.Q is not None else None

        def energy(r: float) -> float:
            x = self.x0 + r * a
            if b is not None:
                x += (r * r) * b
            return float(self.potential(x, False)) - self.rest_energy

        return energy

    def log_weight(self, q: np.ndarray, gq: np.ndarray) -> float:
        """Log of the weight w(q), given the reduced energy gradient gq at q."""
        if self.weight == "uniform":
            return 0.0
        if self.weight == "area":
            return float(np.log(np.linalg.norm(gq)))
        J = self.jacobian(q)
        _, logdet = np.linalg.slogdet(J.T @ (self.M[:, np.newaxis] * J))
        return 0.5 * logdet


class StarShapeError(RuntimeError):
    pass


def radial_root(
    model,
    d: np.ndarray,
    energy: float,
    r0: float,
    rtol: float = 1e-6,
    max_expansions: int = 60,
    monotonicity_rtol: float = 1e-6,
    monotonicity_atol: float = 0.0,
) -> typing.Tuple[float, bool]:
    """Find r > 0 s.t. model.energy(r * d) = energy.

    Starts from the quadratic estimate r1 = r0 sqrt(energy / Psi_hat(r0 d)), brackets the root with
    geometrically growing steps around r1, then refines it with Brent's method.

    Args:
        model: Reduced model exposing energy(q).
        d (np.ndarray): Search direction in q.
        energy (float): Target energy.
        r0 (float): Initial guess for r.
        rtol (float): Relative tolerance on r.
        max_expansions (int): Maximum number of bracket expansions.
        monotonicity_rtol (float): Energy decreases along the ray smaller than
            max(monotonicity_rtol * energy, monotonicity_atol) are attributed to round-off.
        monotonicity_atol (float): See monotonicity_rtol, e.g. a multiple of the energy's noise floor.

    Returns:
        typing.Tuple[float, bool]: The root r, and False if the energy was found to decrease
        along the ray at the evaluated points (the level set may not be star-shaped).
    """
    # Models may provide a cheaper evaluation restricted to the ray
    along_ray = model.ray_energy(d) if hasattr(model, "ray_energy") else lambda r: model.energy(r * d)
    # Cached, so that brentq's re-evaluation of the bracket ends can't flip their signs under noise
    evaluations = {}

    def f(r):
        if r not in evaluations:
            evaluations[r] = along_ray(r) - energy
        return evaluations[r]

    psi0 = f(r0) + energy
    r1 = r0 * np.sqrt(energy / psi0) if psi0 > 0 else r0
    f1 = f(r1)
    growth = 1.01
    if f1 < 0:
        lo, hi = r1, r1 * growth
        for _ in range(max_expansions):
            if f(hi) >= 0:
                break
            growth *= growth
            lo, hi = hi, hi * growth
        else:
            raise StarShapeError(f"Could not bracket energy {energy} along ray")
    else:
        lo, hi = r1 / growth, r1
        for _ in range(max_expansions):
            if f(lo) < 0:
                break
            growth *= growth
            lo, hi = lo / growth, lo
        else:
            raise StarShapeError(f"Could not bracket energy {energy} along ray")
    root = scipy.optimize.brentq(f, lo, hi, rtol=rtol)
    fs = np.array([evaluations[r] for r in sorted(evaluations)])
    monotone = bool(np.all(np.diff(fs) >= -max(monotonicity_rtol * energy, monotonicity_atol)))
    return root, monotone


def estimate_energy_noise(model, q: np.ndarray, n: int = 33, rel: float = 1e-3) -> float:
    """Estimates the round-off noise of model.energy near q.

    Evaluates the energy at (1 + t) q for n values of t in [-rel, rel], over which the exact energy is
    a smooth function, and returns the standard deviation of the residual of a cubic fit. rel must be
    large enough for the probed positions to differ in the precision the energy is evaluated in.

    Args:
        model: Reduced model exposing energy(q).
        q (np.ndarray): Point to probe, ideally near the level set.
        n (int): Number of evaluations.
        rel (float): Relative half-width of the probed segment.

    Returns:
        float: The estimated (absolute) energy noise standard deviation.
    """
    t = np.linspace(-rel, rel, n)
    e = np.array([model.energy((1 + ti) * q) for ti in t])
    residual = e - np.polyval(np.polyfit(t, e, 3), t)
    return float(np.std(residual))


def integrated_autocorrelation_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of a scalar chain, with Sokal's automatic window.

    Args:
        x (np.ndarray): |# steps| chain of a scalar observable.
        c (float): Window constant.

    Returns:
        float: The estimated integrated autocorrelation time (1 for an uncorrelated chain).
    """
    n = x.shape[0]
    x = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = np.arange(n) >= c * taus
    m = np.argmax(window) if np.any(window) else n - 1
    return max(float(taus[m]), 1.0)


@dataclasses.dataclass
class SampleBatch:
    q: np.ndarray
    energy: np.ndarray
    log_target: np.ndarray
    stats: dict


class _Chain:
    """Shared Metropolis machinery: step size adaptation, thinning, and sample collection.

    Subclasses set self.q, self.e, self.logp and implement _step() -> bool (accepted) and
    _observables() -> np.ndarray, the scalars whose autocorrelation sets the thinning.

    The energy may be noisy (e.g. pbat evaluates it in float32). The noise floor is measured once
    near the level set, and energy tolerances are kept above noise_factor times it.
    """

    def __init__(
        self,
        model,
        energy: float,
        rng: np.random.Generator,
        step: float,
        target_acceptance: float,
        noise_factor: float = 10.0,
    ):
        self.model = model
        self.target_energy = energy
        self.rng = rng
        self.step = step
        self.target_acceptance = target_acceptance
        self.star_violations = 0
        self.n_steps = 0
        self.n_accepted = 0
        self.thin = None
        self.tau = None
        q = np.sqrt(2 * energy) * model.scale * self._random_direction()
        self.energy_noise = estimate_energy_noise(model, q)
        self.energy_atol = noise_factor * self.energy_noise

    def _random_direction(self) -> np.ndarray:
        theta = self.rng.standard_normal(self.model.dims)
        return theta / np.linalg.norm(theta)

    def _radial_point(self, theta: np.ndarray, r0: float):
        """The level-set point along whitened direction theta, and its polar log density."""
        d = self.model.scale * theta
        r, monotone = radial_root(
            self.model, d, self.target_energy, r0, monotonicity_atol=self.energy_atol
        )
        self.star_violations += not monotone
        q = r * d
        e, gq = self.model.energy_and_gradient(q)
        drdpsi = gq @ d
        if drdpsi <= 0:
            self.star_violations += 1
            return r, q, e, gq, -np.inf
        logp = self.model.log_weight(q, gq) + (self.model.dims - 1) * np.log(r) - np.log(drdpsi)
        return r, q, e, gq, logp

    def burn_in(self, n: int):
        """Run n steps, adapting the step size toward the target acceptance rate."""
        log_step = np.log(self.step)
        for k in range(n):
            accepted = self._step()
            log_step += (float(accepted) - self.target_acceptance) / np.sqrt(k + 1.0)
            log_step = np.clip(log_step, np.log(1e-4), np.log(4.0))
            self.step = float(np.exp(log_step))

    def estimate_thinning(self, n: int) -> int:
        """Run n steps at a fixed step size and set the thinning to the autocorrelation time."""
        obs = np.empty((n, self._observables().shape[0]))
        for k in range(n):
            self._advance()
            obs[k] = self._observables()
        self.tau = max(integrated_autocorrelation_time(o) for o in obs.T)
        self.thin = int(np.ceil(self.tau))
        return self.thin

    def _advance(self):
        self.n_accepted += self._step()
        self.n_steps += 1

    def sample(self, n: int, thin: int | None = None) -> SampleBatch:
        thin = thin or self.thin or 1
        steps0, accepted0 = self.n_steps, self.n_accepted
        q = np.empty((n, self.model.dims))
        e = np.empty(n)
        logp = np.empty(n)
        obs = np.empty((n * thin, self._observables().shape[0]))
        for k in range(n * thin):
            self._advance()
            obs[k] = self._observables()
            if (k + 1) % thin == 0:
                i = (k + 1) // thin - 1
                q[i], e[i], logp[i] = self.q, self.e, self.logp
        tau = max(integrated_autocorrelation_time(o) for o in obs.T)
        steps = self.n_steps - steps0
        stats = dict(
            acceptance=(self.n_accepted - accepted0) / steps,
            step=self.step,
            thin=thin,
            tau=tau,
            ess=steps / tau,
            star_violations=self.star_violations,
            importance_weights_required=isinstance(self, RaySampler),
            energy_noise=self.energy_noise,
            energy_tolerance=self.energy_atol,
            energy_noise_warning=self.energy_atol > 1e-3 * self.target_energy,
        )
        return SampleBatch(q, e, logp, stats)


class PolarMetropolis(_Chain):
    """Random-walk Metropolis on whitened directions theta, with q = r(theta) * scale * theta exactly on the level set.

    The target on the sphere is p(theta) ~ w(q) r^(m-1) / (d Psi_hat / dr). The proposal
    theta' = normalize(theta + step * xi), xi ~ N(0, I), is symmetric. Requires every ray to cross
    the level set once (star-shaped level set); violations found while bracketing are counted.
    """

    def __init__(self, model, energy: float, rng: np.random.Generator, step: float = 0.3, target_acceptance: float = 0.3):
        super().__init__(model, energy, rng, step, target_acceptance)
        self.theta = self._random_direction()
        self.r, self.q, self.e, _, self.logp = self._radial_point(self.theta, np.sqrt(2 * energy))

    def _step(self) -> bool:
        theta = self.theta + self.step * self.rng.standard_normal(self.model.dims)
        theta /= np.linalg.norm(theta)
        r, q, e, _, logp = self._radial_point(theta, self.r)
        if np.log(self.rng.uniform()) < logp - self.logp:
            self.theta, self.r, self.q, self.e, self.logp = theta, r, q, e, logp
            return True
        return False

    def _observables(self) -> np.ndarray:
        return np.append(self.theta, self.logp)


class ConstrainedRandomWalk(_Chain):
    """Metropolis random walk on the level set with tangent proposals and projection along the normal.

    Follows Zappa, Holmes-Cerfon & Goodman (2018), "Monte Carlo on manifolds: sampling densities and
    integrating functions", for a codimension-1 level set in whitened coordinates y. The target
    density with respect to the surface measure in y is f(y) = w(q) / |grad_y Psi_hat|. A move is
    rejected if the projection fails, or if the reverse projection doesn't return to the start
    (the reversibility check).
    """

    def __init__(
        self,
        model,
        energy: float,
        rng: np.random.Generator,
        step: float = 0.3,
        target_acceptance: float = 0.3,
        projection_rtol: float = 5e-5,
        reverse_rtol: float = 1e-3,
        max_newton: int = 30,
    ):
        super().__init__(model, energy, rng, step, target_acceptance)
        # Projections can't be more accurate than the energy's noise floor. Since Psi_hat ~ r^2,
        # an energy band of +-tol around E* is a radial band of relative width ~tol/E*.
        self.projection_atol = max(projection_rtol * energy, self.energy_atol)
        self.reverse_rtol = max(reverse_rtol, 10 * self.projection_atol / energy)
        self.max_newton = max_newton
        self.n_projection_failures = 0
        self.n_reverse_failures = 0
        theta = self._random_direction()
        r, q, e, gq, _ = self._radial_point(theta, np.sqrt(2 * energy))
        self._set_state(q / model.scale, q, e, gq)

    def _set_state(self, y, q, e, gq):
        self.y, self.q, self.e = y, q, e
        gy = self.model.scale * gq
        self.normal = gy / np.linalg.norm(gy)
        self.logp = self.model.log_weight(q, gq) - np.log(np.linalg.norm(gy))

    def _project(self, z: np.ndarray, n: np.ndarray):
        """Solve Psi_hat(scale * (z + a n)) = E* for a by Newton's method; returns (y, q, e, gq) or None."""
        a = 0.0
        tol = self.projection_atol
        for _ in range(self.max_newton):
            y = z + a * n
            q = self.model.scale * y
            e, gq = self.model.energy_and_gradient(q)
            if abs(e - self.target_energy) <= tol:
                return y, q, e, gq
            slope = (self.model.scale * gq) @ n
            if slope <= 0:
                return None
            a -= (e - self.target_energy) / slope
        return None

    def _step(self) -> bool:
        xi = self.step * self.rng.standard_normal(self.model.dims)
        v = xi - (xi @ self.normal) * self.normal
        projected = self._project(self.y + v, self.normal)
        if projected is None:
            self.n_projection_failures += 1
            return False
        y, q, e, gq = projected
        gy = self.model.scale * gq
        normal = gy / np.linalg.norm(gy)
        logp = self.model.log_weight(q, gq) - np.log(np.linalg.norm(gy))
        w = self.y - y
        vr = w - (w @ normal) * normal
        log_alpha = logp - self.logp - (vr @ vr - v @ v) / (2 * self.step**2)
        if np.log(self.rng.uniform()) >= log_alpha:
            return False
        reverse = self._project(y + vr, normal)
        if reverse is None or np.linalg.norm(reverse[0] - self.y) > self.reverse_rtol * np.linalg.norm(self.y):
            self.n_reverse_failures += 1
            return False
        self.y, self.q, self.e, self.normal, self.logp = y, q, e, normal, logp
        return True

    def _observables(self) -> np.ndarray:
        return np.append(self.y / np.linalg.norm(self.y), self.logp)

    def sample(self, n: int, thin: int | None = None) -> SampleBatch:
        batch = super().sample(n, thin)
        batch.stats.update(
            projection_failures=self.n_projection_failures,
            reverse_failures=self.n_reverse_failures,
        )
        return batch


class RaySampler(_Chain):
    """Independent uniform whitened directions, each solved to the level set.

    Samples are not distributed according to the target. Each sample's log_target is the log target
    density on the sphere, so log importance weights relative to the uniform direction proposal.
    """

    def __init__(self, model, energy: float, rng: np.random.Generator):
        super().__init__(model, energy, rng, step=1.0, target_acceptance=1.0)
        self.thin = 1
        self.tau = 1.0
        self._step()

    def _step(self) -> bool:
        self.theta = self._random_direction()
        _, self.q, self.e, _, self.logp = self._radial_point(self.theta, np.sqrt(2 * self.target_energy))
        return True

    def burn_in(self, n: int):
        pass

    def estimate_thinning(self, n: int) -> int:
        return 1

    def _observables(self) -> np.ndarray:
        return np.append(self.theta, self.logp)


SAMPLERS = {
    "polar": PolarMetropolis,
    "crw": ConstrainedRandomWalk,
    "ray": RaySampler,
}


def make_sampler(kind: str, model, energy: float, rng: np.random.Generator):
    if kind not in SAMPLERS:
        raise ValueError(f"Unknown sampler {kind}, expected one of {tuple(SAMPLERS)}")
    return SAMPLERS[kind](model, energy, rng)


def run_chain(sampler, n_samples: int, burn_in: int, pilot: int, thin: int | None = None) -> SampleBatch:
    """Burn in, then thin by the pilot's autocorrelation time (unless thin is given), then sample."""
    sampler.burn_in(burn_in)
    auto_thin = thin is None
    if auto_thin:
        thin = sampler.estimate_thinning(pilot)
    batch = sampler.sample(n_samples, thin)
    batch.stats.update(burn_in=burn_in, pilot=pilot if auto_thin else 0, auto_thin=auto_thin)
    return batch


def git_commit(path: str) -> str:
    """The HEAD commit of the repository containing path, suffixed with -dirty if it has local changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=path).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_samples(
    path: str,
    X: np.ndarray,
    model: QuadraticManifold,
    omega: np.ndarray,
    batch: SampleBatch,
    attrs: dict,
):
    """Write a sample batch to HDF5 (see the sample-file contract in plans/MASTER.md).

    Args:
        path (str): Output .h5 file path.
        X (np.ndarray): 3 x |# nodes| rest positions, in pbat's layout.
        model (QuadraticManifold): The reduced model the samples were drawn on.
        omega (np.ndarray): m frequencies of the linear modes in model.U.
        batch (SampleBatch): The samples.
        attrs (dict): Extra root attributes (input, sampler settings, target energy, seed, ...).
    """
    import h5py

    N = X.shape[1]
    S = batch.q.shape[0]
    x = np.empty((S, N, 3), dtype=np.float32)
    energy = np.empty(S)
    for s in range(S):
        u = model.displacement(batch.q[s])
        x[s] = (model.x0 + u).reshape(N, 3)
        energy[s] = model.energy(batch.q[s])
    with h5py.File(path, "w") as f:
        f.attrs.update(attrs)
        f.attrs.update({k: v for k, v in batch.stats.items()})
        f.attrs["weight"] = model.weight
        f.attrs["importance_weights_required"] = bool(batch.stats.get("importance_weights_required", False))
        f.attrs["log_target"] = (
            "samples/log_target is the log unnormalized target density per sample. "
            "If importance_weights_required is False (MCMC), samples already follow the target: "
            "average uniformly and treat log_target as a diagnostic. Otherwise (ray sampler), "
            "use self-normalized importance weights exp(log_target - max) / sum(exp(log_target - max))."
        )
        f.attrs["map"] = "linear" if model.Q is None else "quadratic"
        f.attrs["n_modes"] = model.dims
        f.attrs["energy_definition"] = "Psi(X + u(q)) - Psi(X) in J (deformation energy, rest energy subtracted)"
        f.attrs["rest_energy"] = model.rest_energy
        f.attrs["layout"] = (
            "X: (N, 3), row i is node i (column i of pbat's 3 x N X). "
            "samples/x: (S, N, 3), same node order. "
            "basis/U, basis/Q: rows indexed 3 * i + d for node i and coordinate d, i.e. X.flatten(order='F'). "
            "u(q) = U q + 1/2 Q:qq (Q is zero-sized for the linear map)."
        )
        f.create_dataset("X", data=np.ascontiguousarray(X.T))
        g = f.create_group("samples")
        g.create_dataset("q", data=batch.q)
        g.create_dataset("x", data=x)
        g.create_dataset("energy", data=energy)
        g.create_dataset("log_target", data=batch.log_target)
        b = f.create_group("basis")
        b.create_dataset("U", data=model.U)
        b.create_dataset("Q", data=model.Q if model.Q is not None else np.zeros((0, 0, 0)))
        b.create_dataset("omega", data=omega)
        b.create_dataset("scale", data=model.scale)
