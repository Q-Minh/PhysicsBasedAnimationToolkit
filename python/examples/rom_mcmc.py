"""Sampling of fixed-energy states on reduced (linear or quadratic) displacement manifolds.

Two ensembles are supported.

Fixed potential energy (samplers "polar", "crw", "ray"): shapes on the energy level set
{q : Psi(x0 + u(q)) - Psi(x0) = E*}, where u(q) is the reduced displacement map and Psi the
full-space hyperelastic energy. Samples follow the density pi(q) ~ w(q) delta(Psi_hat(q) - E*) dq,
where the weight w selects the distribution:

- "mass": sqrt(det(J(q)^T M J(q))), the microcanonical measure in the mass metric of the shapes,
  which does not depend on how q is scaled,
- "uniform": 1, the microcanonical (coarea) measure in q,
- "area": |grad_q Psi_hat|, the Euclidean surface measure of the level set in q.

Fixed total energy (sampler "microcanonical"): positions and velocities (q, qdot) of the
microcanonical ensemble delta(H(q, p) - E) dq dp, with H = 1/2 p^T M_r(q)^{-1} p + U(q) and
M_r(q) = J(q)^T M J(q). See MicrocanonicalMetropolis.

Samplers work in whitened coordinates y = q / scale, in which the linearized energy is |y|^2 / 2.
They all expose `burn_in`, `estimate_thinning` and `sample`, so that `run_chain` can drive any of them.

The derivations behind each sampler are in docs/sampling-methods.md.

Which sampler to use:

- "microcanonical" (the preferred default): states (q, qdot) at total energy E. Shapes span a range of
  deformation energies below E (U/E averages ~0.5 at small E and ~0.3 at large E), and velocities come for
  free, which dynamic problems need. Steps are cheap (~1 ms, no root solves) but strongly correlated
  (tau ~ 100-250 steps), so pool several chains (run_chains, 4 by default) with >= 400 samples and check
  split R-hat < 1.05. Used for the benchmark datasets (make_benchmark.py).
- "polar": shapes exactly on the level set U(q) = E*, when every sample must have the same deformation
  energy. Mixes fast (tau ~ 10-20 steps, ~3 ms/step with the radial root solve). Requires a star-shaped
  level set; check star_violations = 0.
- "crw": the same target as polar by an independent method, about 2x costlier per independent sample. Use
  it to cross-check polar, or if star_violations appear.
- "ray": independent directions, not distributed as the target (files set importance_weights_required).
  Only for browsing shapes or debugging, never for statistics.

Energies come from pbat in float32, so they carry an absolute round-off noise floor (about 1e-4 J on
MediumLongBendingBeam). Each chain measures it and keeps its tolerances above it, and batches report it as
`energy_noise`, `energy_tolerance` and `energy_noise_warning` (set when the tolerance exceeds 1e-3 E).

Usage (modalrom.py builds the model; see its build_rom and reduced_model):

    model, omega = modalrom.reduced_model("quadratic", U, Q, w, m, x0, potential, rigid, "mass")
    batch = rom_mcmc.run_chains(
        lambda rng: rom_mcmc.make_sampler("microcanonical", model, 10.0, rng),
        n_chains=4, n_samples=400, burn_in=300, pilot=500, seed=0,
    )
    assert batch.stats["rhat"] < 1.05  # the chains agree
    rom_mcmc.write_samples("batch.h5", X, model, omega, batch, attrs={"input": "scene.Dump.h5:fem"})
"""

import dataclasses
import subprocess
import typing

import numpy as np
import scipy.linalg
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
        fext: np.ndarray | None = None,
    ):
        """
        Args:
            U (np.ndarray): n x m linear basis.
            Q (np.ndarray | None): n x m x m symmetric quadratic basis, or None for the linear map.
            M (np.ndarray): n lumped mass matrix diagonal.
            x0 (np.ndarray): n rest positions in F order (interleaved per node).
            potential (typing.Callable): potential(x, gradient) returns Psi(x), or (Psi(x), grad Psi(x)) if gradient.
            scale (np.ndarray): m whitening scales s.t. q = scale * y. The samplers propose moves in y, where the
                linearized energy is |y|^2 / 2, so one step size suits all modes. modalrom.reduced_model sets
                scale_i = 1 / (omega_i sqrt(U_i^T M U_i)), i.e. 1 / omega_i for mass-normalized modes.
            weight (str): One of WEIGHTS; selects the fixed-potential samplers' target distribution on the level
                set (see the module docstring). The microcanonical sampler ignores it.
            fext (np.ndarray | None): n constant external force (e.g. gravity M g), adding the potential
                -fext^T u(q) to the energy. Note that its map u(q) is quadratic in q on a quadratic manifold.
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
        self.fext = fext
        self.dims = U.shape[1]
        self.rest_energy = float(potential(x0, False))
        if self.Q is not None:
            n, m = self.U.shape
            # 2D views of Q for the quadratic term and the Jacobian (no copies, since Q is contiguous)
            self._Qnmm = self.Q.reshape(n, m * m)
            self._Qnm_m = self.Q.reshape(n * m, m)

    def displacement(self, q: np.ndarray) -> np.ndarray:
        """The n full-space displacement u(q) = U q + 1/2 Q:qq (or U q) at the m coordinates q."""
        u = self.U @ q
        if self.Q is not None:
            u += 0.5 * (self._Qnmm @ np.outer(q, q).ravel())
        return u

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """The n x m Jacobian J(q) = du/dq = U + Q:q of the map at q (Q symmetric)."""
        if self.Q is None:
            return self.U
        return self.U + (self._Qnm_m @ q).reshape(self.U.shape)

    def energy(self, q: np.ndarray) -> float:
        """Potential energy U(q) = Psi(x0 + u(q)) - Psi(x0) - fext^T u(q) (the deformation energy without fext)."""
        u = self.displacement(q)
        e = float(self.potential(self.x0 + u, False)) - self.rest_energy
        return e - self.fext @ u if self.fext is not None else e

    def energy_and_gradient(self, q: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        """The potential energy U(q) (see energy) and its m gradient J(q)^T (grad Psi - fext)."""
        u = self.displacement(q)
        e, g = self.potential(self.x0 + u, True)
        e = float(e) - self.rest_energy
        if self.fext is not None:
            e, g = e - self.fext @ u, g - self.fext
        return e, self.jacobian(q).T @ g

    def ray_energy(self, d: np.ndarray) -> typing.Callable[[float], float]:
        """Returns r -> energy(r * d), using u(r d) = r U d + r^2 (1/2 Q:dd) to avoid the map's products per call."""
        a = self.U @ d
        b = 0.5 * (self._Qnmm @ np.outer(d, d).ravel()) if self.Q is not None else None
        fa = self.fext @ a if self.fext is not None else 0.0
        fb = self.fext @ b if self.fext is not None and b is not None else 0.0

        def energy(r: float) -> float:
            x = self.x0 + r * a
            if b is not None:
                x += (r * r) * b
            return float(self.potential(x, False)) - self.rest_energy - (r * fa + r * r * fb)

        return energy

    def reduced_mass(self, q: np.ndarray) -> np.ndarray:
        """The m x m configuration-dependent reduced mass matrix M_r(q) = J(q)^T M J(q)."""
        J = self.jacobian(q)
        return J.T @ (self.M[:, np.newaxis] * J)

    def log_weight(self, q: np.ndarray, gq: np.ndarray) -> float:
        """Log of the weight w(q) that selects the fixed-potential distribution (see the module docstring).

        Args:
            q (np.ndarray): m reduced coordinates.
            gq (np.ndarray): m reduced energy gradient at q (used by the "area" weight).

        Returns:
            float: log w(q).
        """
        if self.weight == "uniform":
            return 0.0
        if self.weight == "area":
            return float(np.log(np.linalg.norm(gq)))
        _, logdet = np.linalg.slogdet(self.reduced_mass(q))
        return 0.5 * logdet


class StarShapeError(RuntimeError):
    """Raised when the target energy can't be bracketed along a ray, e.g. because the energy never reaches it."""

    pass


def radial_root(
    model,
    d: np.ndarray,
    energy: float,
    r0: float,
    rtol: float = 1e-6,
    max_expansions: int = 60,
    crossing_rtol: float = 1e-6,
    crossing_atol: float = 0.0,
) -> typing.Tuple[float, bool]:
    """Find r > 0 s.t. model.energy(r * d) = energy.

    Algorithm:
        1. Evaluate the energy at the guess r0 and rescale it by the quadratic estimate
           r1 = r0 sqrt(energy / U(r0 d)), exact when the energy is quadratic along the ray (U(r0 d) <= 0, e.g.
           under gravity, keeps r1 = r0).
        2. Bracket the root around r1 with geometrically growing steps: the growth factor starts at 1.01 and is
           squared after each expansion, so a good guess costs one or two evaluations and a bad one stays
           logarithmic in the error.
        3. Refine with Brent's method to the relative tolerance rtol on r.
        4. Count the target crossings among all evaluated points (ignoring those within the round-off band).

    Args:
        model: Reduced model exposing energy(q), and optionally ray_energy(d) for cheaper evaluations along d.
        d (np.ndarray): Search direction in q (scale * theta for a whitened unit direction theta).
        energy (float): Target energy in J.
        r0 (float): Initial guess for r, e.g. the previous chain state's root, or sqrt(2 energy) (the linearized
            root in whitened coordinates).
        rtol (float): Relative tolerance on r. Since the energy grows like r^2, the energy error is ~2 rtol energy;
            the samplers set it to the energy's noise floor (no point solving more accurately than the energy).
        max_expansions (int): Maximum number of bracket expansions before giving up with StarShapeError.
        crossing_rtol (float): Energies within max(crossing_rtol * energy, crossing_atol) of the target
            are attributed to round-off when counting crossings.
        crossing_atol (float): See crossing_rtol, e.g. a multiple of the energy's noise floor.

    Returns:
        typing.Tuple[float, bool]: The root r, and False if the evaluated energies along the ray cross
        the target more than once (the level set is not star-shaped). The energy may decrease along the
        ray without crossing, e.g. under gravity, where it first dips below zero.

    Raises:
        StarShapeError: If the target can't be bracketed within max_expansions.
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
    signs = np.sign(fs[np.abs(fs) > max(crossing_rtol * energy, crossing_atol)])
    single_crossing = bool(np.count_nonzero(np.diff(signs)) <= 1)
    return root, single_crossing


def estimate_energy_noise(model, q: np.ndarray, n: int = 33, rel: float = 1e-3) -> float:
    """Estimates the round-off noise of model.energy near q.

    Evaluates the energy at (1 + t) q for n values of t in [-rel, rel], over which the exact energy is
    a smooth function, and returns the standard deviation of the residual of a cubic fit. rel must be
    large enough for the probed positions to differ in the precision the energy is evaluated in.

    Args:
        model: Reduced model exposing energy(q).
        q (np.ndarray): Point to probe, ideally near the level set (the noise of a float32 sum scales with the
            magnitude of its terms, so it depends on where it's measured).
        n (int): Number of evaluations; more gives a steadier estimate of the standard deviation.
        rel (float): Relative half-width of the probed segment. 1e-3 makes float32 positions differ (at ~1e-6
            relative, most probes would round to the same positions and underestimate the noise) while the smooth
            part of the energy stays cubic to ~1e-12 relative.

    Returns:
        float: The estimated (absolute) energy noise standard deviation.
    """
    t = np.linspace(-rel, rel, n)
    e = np.array([model.energy((1 + ti) * q) for ti in t])
    residual = e - np.polyval(np.polyfit(t, e, 3), t)
    return float(np.std(residual))


def integrated_autocorrelation_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of a scalar chain, with Sokal's automatic window.

    tau = 1 + 2 sum_{k=1}^{M} rho_k, with rho_k the normalized autocorrelation at lag k (computed by FFT), and the
    window M the smallest lag with M >= c tau(M). A chain of n steps then holds about n / tau independent samples
    (its effective sample size).

    Args:
        x (np.ndarray): |# steps| chain of a scalar observable.
        c (float): Window constant. Larger values sum more lags (less bias, more noise); 5 is Sokal's usual choice.
            The estimate is only reliable when the chain is much longer than c tau.

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


def split_rhat(traces: typing.Sequence[np.ndarray]) -> float:
    """Split-R-hat (Gelman et al., Bayesian Data Analysis, 3rd ed.) of chains of observables.

    Each chain is cut into two halves of n steps (detecting chains that are still drifting), and for each observable
    R-hat = sqrt(((n - 1) / n W + B / n) / W), with W the mean within-half variance and B / n the variance of the
    half means. It approaches 1 from above as all halves sample the same distribution.

    Args:
        traces (typing.Sequence[np.ndarray]): One |# steps| x |# observables| trace per chain.

    Returns:
        float: The largest split-R-hat over the observables. Values near 1 (e.g. < 1.05) indicate
        that the chains sample the same distribution.
    """
    n = min(t.shape[0] for t in traces) // 2
    halves = np.stack([h for t in traces for h in (t[:n], t[n : 2 * n])])
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var = (n - 1) / n * W + B / n
    rhat = np.sqrt(np.divide(var, W, out=np.ones_like(W), where=W > 0))
    return float(rhat.max())


@dataclasses.dataclass
class SampleBatch:
    """A batch of samples drawn by a sampler.

    Attributes:
        q (np.ndarray): S x m reduced coordinates.
        energy (np.ndarray): S potential energies U(q) in J (= E* on the level set for fixed-potential samplers).
        log_target (np.ndarray): S log unnormalized target densities in the sampler's own parametrization.
        stats (dict): Chain statistics (acceptance, step, thin, tau, ess, star_violations,
            importance_weights_required, energy_noise, energy_tolerance, energy_noise_warning, and from
            run_chain/run_chains burn_in, pilot, auto_thin, n_chains, rhat).
        qdot (np.ndarray | None): S x m reduced velocities (microcanonical sampler only).
        trace (np.ndarray | None): |# steps| x |# observables| per-step observables of the sampling run, used for
            the autocorrelation time and split R-hat.
    """

    q: np.ndarray
    energy: np.ndarray
    log_target: np.ndarray
    stats: dict
    qdot: np.ndarray | None = None
    trace: np.ndarray | None = None


class _Chain:
    """Shared Metropolis machinery: step size adaptation, thinning, and sample collection.

    Subclasses set self.q, self.e, self.logp and implement _step() -> bool (accepted) and
    _observables() -> np.ndarray, the scalars whose autocorrelation sets the thinning.

    The energy may be noisy (e.g. pbat evaluates it in float32). The noise floor is measured once
    near the level set, and energy tolerances are kept above noise_factor times it.

    A run has three phases (see run_chain):
        burn_in            adapts the step size sigma with the Robbins-Monro update
                           log sigma <- log sigma + (a_k - target_acceptance) / sqrt(k + 1), where a_k is 1 if step k
                           was accepted, clipped to sigma in [1e-4, 4]. The decaying gain lets sigma settle; the
                           step size is then frozen, so the later phases are a valid (non-adaptive) Markov chain.
        estimate_thinning  runs a pilot at the frozen step size and sets thin = ceil(tau), tau being the largest
                           integrated autocorrelation time over the observables.
        sample             keeps every thin-th state, so consecutive samples are about independent.

    Args:
        model: Reduced model (see QuadraticManifold).
        energy (float): Target energy in J.
        rng (np.random.Generator): Random generator.
        step (float): Initial proposal step size sigma, in whitened coordinates.
        target_acceptance (float): Acceptance rate the burn-in adapts sigma toward.
        noise_factor (float): Energy tolerances (crossing checks, projections) are noise_factor times the measured
            noise standard deviation, i.e. well above round-off.
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
        r, single_crossing = radial_root(
            self.model,
            d,
            self.target_energy,
            r0,
            # Psi_hat ~ r^2, so a relative error eps in r is an energy error ~2 eps E*: stop at the noise floor
            rtol=max(1e-6, 0.5 * self.energy_noise / self.target_energy),
            crossing_atol=self.energy_atol,
        )
        self.star_violations += not single_crossing
        q = r * d
        e, gq = self.model.energy_and_gradient(q)
        drdpsi = gq @ d
        if drdpsi <= 0:
            self.star_violations += 1
            return r, q, e, gq, -np.inf
        logp = self.model.log_weight(q, gq) + (self.model.dims - 1) * np.log(r) - np.log(drdpsi)
        return r, q, e, gq, logp

    def burn_in(self, n: int):
        """Run n steps, adapting the step size toward the target acceptance rate (see the class docstring).

        Args:
            n (int): Number of burn-in steps. They also move the chain away from its start, so use enough for both
                (a few hundred suffice for the step size; strongly correlated chains need more to forget the start).
        """
        log_step = np.log(self.step)
        for k in range(n):
            accepted = self._step()
            log_step += (float(accepted) - self.target_acceptance) / np.sqrt(k + 1.0)
            log_step = np.clip(log_step, np.log(1e-4), np.log(4.0))
            self.step = float(np.exp(log_step))

    def estimate_thinning(self, n: int) -> int:
        """Run n steps at a fixed step size and set the thinning to the autocorrelation time.

        Args:
            n (int): Number of pilot steps. Should be many times the autocorrelation time for a reliable estimate;
                a short pilot tends to over- or underestimate the thinning.

        Returns:
            int: The thinning ceil(tau).
        """
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
        """Runs n * thin steps and keeps every thin-th state.

        Args:
            n (int): Number of samples.
            thin (int | None): Thinning, or None for the one estimated by estimate_thinning (or 1).

        Returns:
            SampleBatch: The samples, with statistics of this run (the autocorrelation time tau and the
            effective sample size ess are estimated from its steps).
        """
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
        qdot = None
        if hasattr(self, "sample_velocity"):
            qdot = np.array([self.sample_velocity(qi, ei) for qi, ei in zip(q, e)])
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
        return SampleBatch(q, e, logp, stats, qdot=qdot, trace=obs)


class PolarMetropolis(_Chain):
    """Random-walk Metropolis on whitened directions theta, with q = r(theta) * scale * theta exactly on the level set.

    The target on the sphere is p(theta) ~ w(q) r^(m-1) / (d Psi_hat / dr). The proposal
    theta' = normalize(theta + step * xi), xi ~ N(0, I), is symmetric. Requires every ray to cross
    the level set once (star-shaped level set); violations found while bracketing are counted.

    Algorithm (one step):
        1. Propose theta' = normalize(theta + sigma xi), with xi ~ N(0, I_m).
        2. Solve the radial root r' with U(r' scale theta') = E* (radial_root, warm-started at the current r).
        3. Evaluate log p(theta') = log w(q') + (m - 1) log r' - log(dU/dr(q')) at q' = r' scale theta'.
        4. Accept with probability min(1, p(theta') / p(theta)).
    Since the energy is exact on the level set at every step, there is no projection and no rejection for failing
    to reach it.

    Args:
        model: Reduced model (QuadraticManifold, or any object with the same dims, scale, energy,
            energy_and_gradient, log_weight and reduced_mass members).
        energy (float): Target potential energy E* in J; every sample lies on U(q) = E*.
        rng (np.random.Generator): Random generator.
        step (float): Initial angular step size sigma of the direction proposal, adapted during burn-in. Small
            values make local moves on the sphere; sigma >~ 2 makes proposals nearly independent directions, so the
            adaptation's upper clip (4) still allows global moves when the target is broad.
        target_acceptance (float): Acceptance rate the burn-in adapts sigma toward (0.3, a common choice for
            random-walk proposals; if sigma hits its upper clip the realized rate is higher).
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

    Algorithm (one step, at y on the level set with unit normal n = grad_y U / |grad_y U|):
        1. Draw a tangent step v = P (sigma xi), xi ~ N(0, I_m), with P = I - n n^T.
        2. Project y + v back onto the level set along n: solve U(scale (y + v + a n)) = E* for a by Newton's
           method. Reject if it fails to converge or the slope along n isn't positive.
        3. Compute the reverse tangent step v' at the proposal y' (the tangent part of y - y') and accept with
           probability min(1, f(y') exp(-|v'|^2 / (2 sigma^2)) / (f(y) exp(-|v|^2 / (2 sigma^2)))).
        4. Reversibility check: project y' + v' along n'; reject unless it returns to y. Without this check the
           chain would be biased wherever the projection isn't unique.

    Args:
        model: Reduced model (QuadraticManifold, or any object with the same dims, scale, energy,
            energy_and_gradient, log_weight and reduced_mass members).
        energy (float): Target potential energy E* in J; every sample lies on U(q) = E* to the projection tolerance.
        rng (np.random.Generator): Random generator.
        step (float): Initial standard deviation sigma of the tangent step in whitened coordinates (where the level
            set's radius is ~sqrt(2 E*)), adapted during burn-in. Large steps fail to project more often.
        target_acceptance (float): Acceptance rate the burn-in adapts sigma toward; failed projections and failed
            reversibility checks count as rejections.
        projection_rtol (float): Relative energy tolerance |U - E*| <= projection_rtol E* of the projections, raised
            to noise_factor times the energy noise when that's larger (Newton can't converge below round-off).
        reverse_rtol (float): Relative position tolerance |y_reverse - y| <= reverse_rtol |y| of the reversibility
            check, raised to 10 projection_atol / E* (projection_atol being the effective energy tolerance above) so that
            round-off alone can't fail it.
            It only has to tell a different projection branch from the start point.
        max_newton (int): Maximum Newton iterations per projection before rejecting the move.
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

    Args:
        model: Reduced model (see PolarMetropolis).
        energy (float): Target energy E* in J.
        rng (np.random.Generator): Random generator.
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


class MicrocanonicalMetropolis(_Chain):
    """Positions and velocities of the microcanonical ensemble at total energy E (Liouville measure dq dp).

    Integrating the momenta out of delta(H(q, p) - E), with H = 1/2 p^T M_r(q)^{-1} p + U(q), leaves
    the position marginal pi(q) ~ sqrt(det M_r(q)) (E - U(q))^((d-2)/2) on U(q) < E. It is sampled by
    random-walk Metropolis in whitened coordinates, starting from the rest state q = 0. Given q, the
    momentum is uniform on the sphere |M_r^{-1/2} p|^2 = 2 (E - U), which sample_velocity draws exactly.
    The sample energies are the potential energies U(q); the kinetic energies are E - U(q). Chains are
    strongly correlated (tau ~ 100-250 steps on the beam and trampoline), so pool several chains with
    run_chains and check that split R-hat < 1.05.

    Algorithm (one step):
        1. Propose y' = y + sigma xi in whitened coordinates (q = scale * y), xi ~ N(0, I_m).
        2. Evaluate U(q'); reject immediately if U(q') >= E (no kinetic energy left).
        3. Accept with probability min(1, pi(q') / pi(q)), pi as above (one log-determinant of M_r per step).
    Velocities are drawn only for the kept (thinned) samples, by sample_velocity.

    Args:
        model: Reduced model (QuadraticManifold, or any object with the same dims, scale, energy and reduced_mass
            members); its energy may include gravity (fext), in which case U < 0 is possible.
        energy (float): Total energy E = K + U in J.
        rng (np.random.Generator): Random generator.
        step (float): Initial standard deviation sigma of the Gaussian random-walk proposal in whitened coordinates,
            where the energy surface U = E has radius ~sqrt(2 E); adapted during burn-in.
        target_acceptance (float): Acceptance rate the burn-in adapts sigma toward (0.25, near the 0.234 that is
            optimal for random-walk Metropolis in many dimensions, Roberts, Gelman & Gilks 1997).
    """

    def __init__(self, model, energy: float, rng: np.random.Generator, step: float = 0.3, target_acceptance: float = 0.25):
        super().__init__(model, energy, rng, step, target_acceptance)
        self.y = np.zeros(model.dims)
        self.q = np.zeros(model.dims)
        self.e = model.energy(self.q)
        self.logp = self._log_target(self.q, self.e)

    def _log_target(self, q: np.ndarray, e: float) -> float:
        kinetic = self.target_energy - e
        if kinetic <= 0:
            return -np.inf
        _, logdet = np.linalg.slogdet(self.model.reduced_mass(q))
        return 0.5 * logdet + 0.5 * (self.model.dims - 2) * np.log(kinetic)

    def _step(self) -> bool:
        y = self.y + self.step * self.rng.standard_normal(self.model.dims)
        q = self.model.scale * y
        e = self.model.energy(q)
        if e >= self.target_energy:
            return False
        logp = self._log_target(q, e)
        if np.log(self.rng.uniform()) < logp - self.logp:
            self.y, self.q, self.e, self.logp = y, q, e, logp
            return True
        return False

    def _observables(self) -> np.ndarray:
        return np.append(self.y, self.logp)

    def sample_velocity(self, q: np.ndarray, e: float) -> np.ndarray:
        """Draws qdot given q, s.t. 1/2 qdot^T M_r(q) qdot = E - U(q), from the exact conditional distribution.

        Args:
            q (np.ndarray): m reduced coordinates.
            e (float): The potential energy U(q).

        Returns:
            np.ndarray: m reduced velocities qdot.
        """
        kinetic = max(self.target_energy - e, 0.0)
        C = np.linalg.cholesky(self.model.reduced_mass(q))
        z = self.rng.standard_normal(self.model.dims)
        z /= np.linalg.norm(z)
        # With M_r = C C^T, qdot = C^{-T} z sqrt(2K) satisfies qdot^T M_r qdot = 2K
        return np.sqrt(2.0 * kinetic) * scipy.linalg.solve_triangular(C.T, z, lower=False)


SAMPLERS = {
    "polar": PolarMetropolis,
    "crw": ConstrainedRandomWalk,
    "ray": RaySampler,
    "microcanonical": MicrocanonicalMetropolis,
}


def make_sampler(kind: str, model, energy: float, rng: np.random.Generator):
    """Creates a sampler with default settings.

    Args:
        kind (str): One of SAMPLERS: "polar", "crw", "ray" (fixed potential energy E*) or "microcanonical"
            (fixed total energy E = K + U).
        model: Reduced model, e.g. a QuadraticManifold.
        energy (float): Target energy in J (E* or E, depending on kind).
        rng (np.random.Generator): Random generator.

    Returns:
        The sampler, ready for run_chain.
    """
    if kind not in SAMPLERS:
        raise ValueError(f"Unknown sampler {kind}, expected one of {tuple(SAMPLERS)}")
    return SAMPLERS[kind](model, energy, rng)


def run_chain(sampler, n_samples: int, burn_in: int, pilot: int, thin: int | None = None) -> SampleBatch:
    """Burn in, then thin by the pilot's autocorrelation time (unless thin is given), then sample.

    Args:
        sampler: A sampler from make_sampler.
        n_samples (int): Number of samples.
        burn_in (int): Burn-in steps, during which the step size adapts.
        pilot (int): Steps used to estimate the autocorrelation time when thin is None.
        thin (int | None): Fixed thinning, or None for automatic thinning.

    Returns:
        SampleBatch: The samples, with burn_in, pilot and auto_thin added to its statistics.
    """
    sampler.burn_in(burn_in)
    auto_thin = thin is None
    if auto_thin:
        thin = sampler.estimate_thinning(pilot)
    batch = sampler.sample(n_samples, thin)
    batch.stats.update(burn_in=burn_in, pilot=pilot if auto_thin else 0, auto_thin=auto_thin)
    return batch


def run_chains(
    make: typing.Callable[[np.random.Generator], typing.Any],
    n_chains: int,
    n_samples: int,
    burn_in: int,
    pilot: int,
    thin: int | None = None,
    seed: int = 0,
) -> SampleBatch:
    """Runs independent chains with run_chain and pools their samples.

    Args:
        make (typing.Callable): make(rng) creates a sampler using the random generator rng.
        n_chains (int): Number of independent chains.
        n_samples (int): Total number of pooled samples, split evenly across the chains.
        burn_in (int): Burn-in steps per chain.
        pilot (int): Pilot steps per chain for automatic thinning.
        thin (int | None): Fixed thinning, or None for automatic thinning per chain.
        seed (int): Seed from which the chains' independent random streams are spawned.

    Returns:
        SampleBatch: The pooled samples, with aggregated statistics and the chains' split-R-hat.
    """
    per_chain = -(-n_samples // n_chains)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_chains)]
    batches = [run_chain(make(rng), per_chain, burn_in, pilot, thin) for rng in rngs]

    def pool(field):
        arrays = [getattr(b, field) for b in batches]
        return None if arrays[0] is None else np.concatenate(arrays)[:n_samples]

    stats = [b.stats for b in batches]
    pooled = dict(stats[0])
    pooled.update(
        n_chains=n_chains,
        acceptance=float(np.mean([s["acceptance"] for s in stats])),
        step=float(np.mean([s["step"] for s in stats])),
        thin=int(max(s["thin"] for s in stats)),
        tau=float(max(s["tau"] for s in stats)),
        ess=float(sum(s["ess"] for s in stats)),
        star_violations=int(sum(s["star_violations"] for s in stats)),
        energy_noise=float(max(s["energy_noise"] for s in stats)),
        energy_tolerance=float(max(s["energy_tolerance"] for s in stats)),
        energy_noise_warning=bool(any(s["energy_noise_warning"] for s in stats)),
        rhat=split_rhat([b.trace for b in batches]),
    )
    return SampleBatch(pool("q"), pool("energy"), pool("log_target"), pooled, qdot=pool("qdot"))


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
    min_det_F: np.ndarray | None = None,
    extra_samples: dict | None = None,
):
    """Write a sample batch to HDF5, following the sample-file contract below.

    Sample-file contract (one target energy per file; S samples, N nodes, m reduced coordinates):

    Datasets:
        X                  (N, 3) float32 rest positions; row i is node i, i.e. pbat's 3 x N X transposed.
        samples/q          (S, m) reduced coordinates.
        samples/x          (S, N, 3) float32 positions X + u(q), in the same node order as X. Dirichlet nodes are
                           exactly at rest, because U and Q vanish on their DOFs.
        samples/energy     (S,) potential energy U(q) = Psi(X + u(q)) - Psi(X) [- (M g)^T u(q) with gravity], in J.
        samples/log_target (S,) log unnormalized target density per sample (see the root attribute log_target).
        samples/qdot       (S, m) reduced velocities, for ensemble "total" only. Full-space velocities are
                           J(q) qdot with J(q) = U + Q:q, and 1/2 qdot^T J^T M J qdot = target_energy - U(q).
        samples/min_J      (S,) minimum det(F) over the mesh per sample, if given (<= 0: inverted elements).
        samples/<name>     extra per-sample datasets, if given (e.g. the benchmark's v, xtilde, x_ref).
        basis/U            (3N, m) mass-normalized vibration modes (U^T M U = I). Row 3i + d is node i, coordinate d.
        basis/Q            (3N, m, m) symmetric quadratic tensor, M-orthogonal to U; (0, 0, 0) for the linear map.
        basis/omega        (m,) mode frequencies in rad/s.
        basis/scale        (m,) whitening scales, q = scale * y.
        basis/M            (3N,) lumped masses per DOF.

    Root attributes:
        From attrs (as written by modalrom.py and make_benchmark.py): input, n_linear_modes,
            rigid_modes_excluded, modal_derivatives, mode_normalization, eps, sampler, target_energy (E* for
            ensemble "potential", E = K + U for ensemble "total"), gravity (3,), seed, git_commit, sampling_time.
        From batch.stats: acceptance, step, thin, tau, ess, star_violations, importance_weights_required,
            energy_noise, energy_tolerance, energy_noise_warning, burn_in, pilot, auto_thin, and from run_chains
            n_chains and rhat (split R-hat).
        Written here: weight, log_target (how to use samples/log_target), map ("linear" or "quadratic"), n_modes,
            ensemble ("potential" or "total"), energy_definition, rest_energy (pbat's float32 energy at rest,
            round-off only) and layout (these conventions).

    Consuming samples: MCMC batches (importance_weights_required = False) already follow the target, so average
    uniformly over samples. Ray batches need the self-normalized importance weights exp(log_target - max) /
    sum(exp(log_target - max)). Check rhat < 1.05 when present, and energy_noise_warning = False (otherwise
    samples only lie on the level set to within energy_tolerance J).

    Args:
        path (str): Output .h5 file path.
        X (np.ndarray): 3 x |# nodes| rest positions, in pbat's layout.
        model (QuadraticManifold): The reduced model the samples were drawn on.
        omega (np.ndarray): m frequencies of the linear modes in model.U.
        batch (SampleBatch): The samples.
        attrs (dict): Extra root attributes (input, sampler settings, target energy, seed, ...).
        min_det_F (np.ndarray | None): Per-sample minimum det(F) over the mesh, a diagnostic for inversion.
        extra_samples (dict | None): Additional per-sample datasets written under samples/, by name.
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
        microcanonical = batch.qdot is not None
        f.attrs["ensemble"] = "total" if microcanonical else "potential"
        f.attrs["energy_definition"] = (
            "samples/energy is the potential energy U(q) = Psi(X + u(q)) - Psi(X) in J (deformation energy, "
            "rest energy subtracted), minus fext^T u(q) if attribute gravity is nonzero (fext = M g). "
            + (
                "Ensemble 'total': target_energy is the total energy E = K + U, the kinetic energy is "
                "E - U = 1/2 qdot^T M_r(q) qdot with M_r = J^T M J, and full-space velocities are J(q) qdot "
                "with J(q) = U + Q:q."
                if microcanonical
                else "Ensemble 'potential': target_energy is E*, and samples lie on U(q) = E*."
            )
        )
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
        if microcanonical:
            g.create_dataset("qdot", data=batch.qdot)
        if min_det_F is not None:
            g.create_dataset("min_J", data=min_det_F)
        for name, data in (extra_samples or {}).items():
            g.create_dataset(name, data=data)
        b = f.create_group("basis")
        b.create_dataset("U", data=model.U)
        b.create_dataset("Q", data=model.Q if model.Q is not None else np.zeros((0, 0, 0)))
        b.create_dataset("omega", data=omega)
        b.create_dataset("scale", data=model.scale)
        b.create_dataset("M", data=model.M)
