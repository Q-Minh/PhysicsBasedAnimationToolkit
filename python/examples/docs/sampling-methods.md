# Sampling fixed-energy states of elastic bodies on quadratic manifolds

A technical reference for the sampling strategies in `python/examples`:

- the reduced model they sample on;
- the target distributions and their derivations, and each sampler's algorithm;
- the Markov chain machinery;
- how samples become static and dynamic benchmark problems;
- how shape matching tests which configurations the reduced model can express.

Equations are numbered with explicit tags and cited as (n), since GitHub doesn't resolve `\label`/`\eqref`. When you
add, remove or reorder an equation, renumber the tags and citations that follow it.

**Contents**

1. [Setting and notation](#1-setting-and-notation)
2. [Quadratic-manifold reduced model](#2-quadratic-manifold-reduced-model)
3. [Fixed potential energy](#3-fixed-potential-energy)
4. [Fixed total energy (microcanonical)](#4-fixed-total-energy-microcanonical)
5. [Markov chain machinery](#5-markov-chain-machinery)
6. [Single-precision energy noise](#6-single-precision-energy-noise)
7. [Which sampler to use](#7-which-sampler-to-use)
8. [Benchmark problems](#8-benchmark-problems)
9. [Shape matching: what can the reduced model express?](#9-shape-matching-what-can-the-reduced-model-express)
10. [Validation](#10-validation)
11. [Code map](#11-code-map)
12. [References](#references)

## 1. Setting and notation

An elastic body is discretized with linear tetrahedra:

- $N$ nodes with rest positions $X \in \mathbb{R}^{3N}$, with degrees of freedom interleaved per node;
- deformed positions $x = X + u$;
- a lumped, diagonal mass matrix $M$.

Dirichlet nodes are fixed at rest, so all displacements vanish on their degrees of freedom. The elastic energy is
pbat's Stable Neo-Hookean energy integrated over the mesh,

$$
\Psi(x) = \sum_e V_e \Big[ \tfrac{\mu}{2}\big(\lVert F_e \rVert_F^2 - 3\big) - \mu\,(J_e - 1) + \tfrac{\lambda}{2}(J_e - 1)^2 \Big],
\qquad J_e = \det F_e ,
\tag{1}
$$

with $F_e$ the deformation gradient of element $e$ (constant on linear tetrahedra), $V_e$ its rest volume and
$\mu, \lambda$ its Lamé parameters. This form is zero at rest and finite for inverted elements ($J_e \le 0$). An
optional constant external force $f$, typically gravity $f = Mg$, adds the potential $-f^\top u$. pbat evaluates
$\Psi$ in single precision, which matters for tolerances (Section 6).

## 2. Quadratic-manifold reduced model

**Vibration modes.** The $m$ lowest modes solve $K U_i = \omega_i^2 M U_i$ at rest ($K = \nabla^2 \Psi(X)$).
They are computed by shift-invert Lanczos with a slightly negative shift, so that rigid modes don't make the
factorization singular, and mass-normalized, $U^\top M U = I$. Without Dirichlet constraints, the six rigid modes
($\omega_i = 0$) are excluded from sampling.

**Modal derivatives.** For each pair $(i, j)$, the derivative of mode $i$ along mode $j$ solves the bordered
vibration-mode-derivative system (Idelsohn & Cardona 1985)

$$
\begin{bmatrix} K - \omega_i^2 M & -M U_i \\ -U_i^\top M & 0 \end{bmatrix}
\begin{bmatrix} \Theta_{ij} \\ \mu_{ij} \end{bmatrix}
= \begin{bmatrix} -\dfrac{\partial K}{\partial \eta_j} U_i \\ 0 \end{bmatrix},
\qquad
\frac{\partial K}{\partial \eta_j} \approx \lVert U_j \rVert \,
\frac{K\big(X + h\,\hat U_j\big) - K\big(X - h\,\hat U_j\big)}{2h},
\tag{2}
$$

with $\hat U_j = U_j / \lVert U_j \rVert$ and $h$ a fraction (default $10^{-3}$) of the bounding-box diagonal. The
derivatives are then projected $M$-orthogonally off the span of the modes and symmetrized:

$$
\Theta \leftarrow \Theta - U\,(U^\top M U)^{-1} U^\top M\,\Theta,
\qquad Q_{ij} = \tfrac12\big(\Theta_{ij} + \Theta_{ji}\big),
\qquad U^\top M Q_{ij} = 0 .
\tag{3}
$$

> **Why the projection matters.** When two frequencies nearly coincide (bending pairs of a square beam:
> $\omega_0 = 1.294$, $\omega_1 = 1.303$), $K - \omega_0^2 M$ is nearly singular along $U_1$ as well, and
> $\Theta_{00}$ acquires a huge spurious component along $U_1$. On the beam, deforming along mode 0 to a 0.3 m tip
> displacement costs 93.5 J linearly, $5 \times 10^6$ J with the raw derivative, and 11.6 J with the projected one.
> With the projected derivative, the energy along $u = rU_0 + \tfrac{c}{2} r^2 Q_{00}$ is lowest at exactly $c = 1$.
> An earlier finite difference also divided by $h$ instead of $2h$, doubling $Q$.

**The map.** Reduced coordinates $q \in \mathbb{R}^m$ give the displacements, their Jacobian and a
configuration-dependent reduced mass matrix,

$$
u(q) = Uq + \tfrac12\, Q{:}qq,
\qquad J(q) = \frac{\partial u}{\partial q} = U + Q{:}q,
\qquad M_r(q) = J(q)^\top M\, J(q),
\tag{4}
$$

and the reduced potential energy is

$$
\mathcal{U}(q) = \Psi\big(X + u(q)\big) - \Psi(X) - f^\top u(q) .
\tag{5}
$$

Because $u(q)$ is quadratic, the gravity potential $-f^\top u(q)$ is quadratic in $q$, not the linear
$-q^\top U^\top f$ of a linear basis. $\Psi(X)$ is zero in exact arithmetic; it is subtracted because single precision
makes it about $5 \times 10^{-4}$ J on the beam.

**Whitening.** All samplers move in coordinates $y$ with $q = S y$, where $S = \mathrm{diag}(s_i)$ and
$s_i = 1 / \big(\omega_i \sqrt{U_i^\top M U_i}\big) = 1/\omega_i$. In $y$ the linearized energy is
$\tfrac12 \lVert y \rVert^2$, so a single step size suits all modes.

## 3. Fixed potential energy

This ensemble draws shapes exactly on the energy level set $\Sigma = \lbrace q : \mathcal{U}(q) = E^\star \rbrace$.

### 3.1 Target measures on the level set

Distributions on a level set are densities times the delta measure,

$$
\pi(\mathrm{d}q) \propto w(q)\, \delta\big(\mathcal{U}(q) - E^\star\big)\, \mathrm{d}q
= \frac{w(q)}{\lVert \nabla \mathcal{U}(q) \rVert}\, \mathrm{d}\sigma(q),
\tag{6}
$$

by the coarea formula, with $\mathrm{d}\sigma$ the surface measure of $\Sigma$ in $q$. Three weights are
implemented:

- $w = 1$ (`"uniform"`): the microcanonical (coarea) measure in $q$.
- $w = \lVert \nabla \mathcal{U} \rVert$ (`"area"`): uniform surface measure of $\Sigma$ in $q$.
- $w = \sqrt{\det M_r(q)}$ (`"mass"`, default): the delta measure in the mass metric $g = J^\top M J$ that the map
  induces from kinetic energy. Under a reparametrization $q = \varphi(\tilde q)$, $\mathrm{d}q = \lvert \det D\varphi
  \rvert\, \mathrm{d}\tilde q$ and $\sqrt{\det \tilde M_r} = \lvert \det D\varphi \rvert \sqrt{\det M_r}$, so this
  choice depends only on the set of shapes, not on how $q$ is scaled. It is also the $q$-dependence of the
  total-energy marginal (9).

### 3.2 Polar Metropolis

On all tested meshes, every ray from the rest shape crosses $\Sigma$ once (the level set is star-shaped). $\Sigma$ is
then parametrized by whitened directions $\theta \in S^{m-1}$: $q(\theta) = r(\theta)\, S\theta$ with
$\mathcal{U}(q(\theta)) = E^\star$. With $y = r\theta$, $\mathrm{d}q = \det S \; r^{m-1}\, \mathrm{d}r\,
\mathrm{d}\Omega(\theta)$ and $\delta(\mathcal{U} - E^\star) = \delta(r - r(\theta)) / \partial_r \mathcal{U}$, so (6)
becomes an ordinary density on the sphere:

$$
p(\theta) \propto \frac{w\big(q(\theta)\big)\, r(\theta)^{m-1}}{\partial_r \mathcal{U}\big(q(\theta)\big)},
\qquad \partial_r \mathcal{U} = \nabla_q \mathcal{U}^\top S \theta > 0 .
\tag{7}
$$

**One polar step:**

1. Propose $\theta' = (\theta + \sigma\xi) / \lVert \theta + \sigma\xi \rVert$, $\xi \sim \mathcal{N}(0, I_m)$. The
   density of $\theta'$ given $\theta$ depends only on $\theta^\top \theta'$, so the proposal is symmetric.
2. Solve $r'$ with $\mathcal{U}(r' S\theta') = E^\star$:
   - start from the current $r$ and rescale it by the quadratic estimate $r_1 = r_0 \sqrt{E^\star /
     \mathcal{U}(r_0 S\theta')}$;
   - bracket the root with a growth factor that starts at 1.01 and is squared after each expansion;
   - refine with Brent's method to a relative tolerance at the energy noise floor.
3. Accept with probability $\min\lbrace 1, p(\theta') / p(\theta) \rbrace$.

Along a ray, $u(rd) = r\, Ud + r^2\, \tfrac12 Q{:}dd$, so the two vectors are computed once per ray and the root
solve's energy evaluations need no matrix product. Every evaluated point is checked for a second crossing of
$E^\star$, which would violate the star shape. Under gravity, the energy may first dip below zero along a ray; that
is not a violation.

### 3.3 Constrained random walk

An independent method for the same targets follows Zappa, Holmes-Cerfon & Goodman (2018) in whitened coordinates. There
$\mathrm{d}q \propto \mathrm{d}y$, and the target density with respect to the surface measure $\mathrm{d}\sigma_y$ is
$f(y) = w(q) / \lVert \nabla_y \mathcal{U} \rVert$.

**One constrained step** at $y \in \Sigma$, with unit normal $n = \nabla_y \mathcal{U} / \lVert \nabla_y \mathcal{U} \rVert$:

1. Take a tangent step $v = (I - nn^\top)\, \sigma\xi$.
2. Project $y + v$ onto $\Sigma$ along $n$: solve $\mathcal{U}\big(S(y + v + an)\big) = E^\star$ for $a$ by Newton's
   method, and reject on failure.
3. Let $v'$ be the tangent part of $y - y'$ at the proposal $y'$. Accept with probability

$$
\alpha = \min\Big\lbrace 1,\ \frac{f(y')\, \exp\big(-\lVert v' \rVert^2 / 2\sigma^2\big)}{f(y)\, \exp\big(-\lVert v \rVert^2 / 2\sigma^2\big)} \Big\rbrace .
$$

4. Reversibility check: project $y' + v'$ along $n'$, and reject unless it returns to $y$. Without this check the
   chain is biased wherever the projection isn't unique.

### 3.4 Ray sampler

Independent directions $\theta_k \sim \mathrm{Unif}(S^{m-1})$, each solved to $\Sigma$, are not distributed as (7).
Each sample stores $\log p(\theta_k)$, so statistics need the self-normalized importance weights
$w_k = p(\theta_k) / \sum_l p(\theta_l)$. Use it to browse shapes only.

## 4. Fixed total energy (microcanonical)

This ensemble samples positions and velocities at a total mechanical energy $E$. With momenta $p = M_r(q)\,\dot q$,
the Hamiltonian is

$$
H(q, p) = \tfrac12\, p^\top M_r(q)^{-1} p + \mathcal{U}(q),
\tag{8}
$$

and the target is the microcanonical ensemble $\rho(q, p) \propto \delta\big(H(q, p) - E\big)$ with respect to the
Liouville measure $\mathrm{d}q\, \mathrm{d}p$, which Hamiltonian flow preserves.

### 4.1 Position marginal

Let $K(q) = E - \mathcal{U}(q)$. Three steps evaluate the integral over momenta:

- substitute $p = M_r^{1/2} s$, so that $\mathrm{d}p = \sqrt{\det M_r}\, \mathrm{d}s$;
- pass to spherical coordinates $s = \varrho\, \hat s$;
- resolve the delta at $\varrho = \sqrt{2K}$.

This gives

$$
\pi(q) = \int \delta\big(\tfrac12 p^\top M_r^{-1} p - K\big)\, \mathrm{d}p
= \sqrt{\det M_r}\; \lvert S^{d-1} \rvert \int_0^\infty \delta\big(\tfrac12 \varrho^2 - K\big)\, \varrho^{d-1}\, \mathrm{d}\varrho
\;\propto\; \sqrt{\det M_r(q)}\; \big(E - \mathcal{U}(q)\big)^{\frac{d-2}{2}}\; \mathbb{1}\big[\mathcal{U}(q) \lt E\big],
\tag{9}
$$

with $d = m$ the number of reduced coordinates. Taking $\mathrm{d}q\, \mathrm{d}\dot q$ instead of
$\mathrm{d}q\, \mathrm{d}p$ would replace $\sqrt{\det M_r}$ by $(\det M_r)^{-1/2}$; the Liouville measure is the
physically invariant choice. For a constant $M_r$ the determinant cancels; on a quadratic manifold it doesn't.

### 4.2 Exact velocities

Given $q$, the conditional distribution of $s = M_r^{-1/2} p$ is uniform on the sphere $\lVert s \rVert = \sqrt{2K}$, so
velocities are drawn exactly. With the Cholesky factorization $M_r = CC^\top$,

$$
\dot q = \sqrt{2\big(E - \mathcal{U}(q)\big)}\; C^{-\top} \frac{z}{\lVert z \rVert},
\qquad z \sim \mathcal{N}(0, I_d),
\qquad \tfrac12\, \dot q^\top M_r\, \dot q = E - \mathcal{U}(q) .
\tag{10}
$$

$C^{-\top} = M_r^{-1/2} R$ for an orthogonal $R$, so $C^{-\top} z / \lVert z \rVert$ has the same law as
$M_r^{-1/2} \hat s$. The result is the pushforward of the uniform sphere measure, not the uniform surface measure of
the kinetic-energy ellipsoid. Full-space velocities are $v = J(q)\, \dot q$, and $K + \mathcal{U} = E$ holds to
round-off for every sample.

### 4.3 The sampler

**One microcanonical step:**

1. Propose $y' = y + \sigma\xi$, $\xi \sim \mathcal{N}(0, I_m)$, $q' = Sy'$.
2. Reject immediately if $\mathcal{U}(q') \ge E$.
3. Accept with probability $\min\lbrace 1, \pi(q') / \pi(q) \rbrace$ from (9). This costs one energy evaluation and
   one $m \times m$ log-determinant.

Chains start at the rest shape $q = 0$, which is always admissible when $E > 0$. Velocities (10) are drawn only for
the kept samples. No root solve is needed, so a step costs about 1 ms on the test meshes. Successive states are
strongly correlated, though (Section 10).

## 5. Markov chain machinery

All Metropolis samplers share three phases.

**Burn-in with step-size adaptation.** A Robbins–Monro update drives the acceptance rate toward a target $a^\star$:

$$
\log \sigma_{k+1} = \mathrm{clip}\Big( \log \sigma_k + \frac{a_k - a^\star}{\sqrt{k+1}},\ \log 10^{-4},\ \log 4 \Big),
\tag{11}
$$

with $a_k \in \lbrace 0, 1 \rbrace$ the acceptance of step $k$. The target is $a^\star = 0.3$ for the level-set
samplers and $0.25$ for the microcanonical one, near the optimal 0.234 of random-walk Metropolis in many dimensions
(Roberts, Gelman & Gilks 1997). The step size is then frozen, so the later phases are a valid non-adaptive chain.

**Pilot and thinning.** A pilot run estimates the integrated autocorrelation time of each observable (whitened
coordinates or directions, and the log target) with Sokal's automatic window,

$$
\tau = 1 + 2 \sum_{k=1}^{W} \rho_k,
\qquad W = \min\lbrace W' : W' \ge c\, \tau(W') \rbrace,\ c = 5,
\tag{12}
$$

and the chain keeps every $\lceil \tau_{\max} \rceil$-th state. The effective sample size of a run of $n$ steps is
$n / \tau$.

**Several chains.** Independent chains from independent random streams are pooled. Convergence is checked with
split-$\hat R$: each chain's trace is halved, and

$$
\hat R = \sqrt{\frac{\frac{n-1}{n} W + \frac{1}{n} B}{W}},
\tag{13}
$$

with $W$ the mean within-half variance and $B/n$ the variance of the half means, maximized over observables. A batch
is accepted when $\hat R \lt 1.05$.

## 6. Single-precision energy noise

pbat sums element energies in float32. Around $F \approx I$ the invariants cancel, leaving an absolute round-off of
roughly $10^{-4}$ J whatever the energy level: a standard deviation of $8 \times 10^{-5}$ J on the beam, and
$1.3$–$4 \times 10^{-5}$ J on the trampoline.

Each chain estimates it once, as the residual standard deviation $\varepsilon$ of a cubic fit to
$\mathcal{U}\big((1+t) q\big)$ over $t \in [-10^{-3}, 10^{-3}]$ (33 points). A narrower range would leave the float32
positions unchanged and underestimate it. The fixed-potential samplers measure it at a random point
$q = \sqrt{2E}\, s \odot \theta$ of the linearized level set, near where they sample. The microcanonical sampler
re-measures it at its samples instead, as the largest $\varepsilon$ over 4 evenly spaced samples of each chain
(`rom_mcmc.samples_energy_noise`). On soft scenes that probe point lies far outside $\mathcal{U} < E$, e.g. at
$10^{13}$ J for $E = 10^4$ J on the $Y = 10^5$ circle trampoline, and overstated the noise by up to $10^9$×.
`recompute_energy_noise.py` corrects datasets generated before this change in place, without resampling.
Tolerances then follow $\varepsilon$:

- the star-shape crossing check ignores points within $10\varepsilon$ of $E^\star$;
- radial roots stop at a relative tolerance $\max(10^{-6}, \varepsilon / 2E^\star)$, since $\mathcal{U} \sim r^2$;
- constrained projections accept $\lvert \mathcal{U} - E^\star \rvert \le \max(5 \times 10^{-5} E^\star,
  10\varepsilon)$, and the reversibility check scales accordingly;
- a batch is flagged when $10\varepsilon \gt 10^{-3} E$, which happens below roughly 0.8 J on the beam.

## 7. Which sampler to use

| Sampler | Target | Cost and mixing | Use |
|---|---|---|---|
| `microcanonical` | $(q, \dot q)$ at $K + \mathcal{U} = E$, (9)–(10) | ~1 ms/step, $\tau \approx 100$–$250$; pool 4 chains and ≥ 400 samples | Default. Benchmark datasets, and whenever velocities or a spread of deformation scales are wanted |
| `polar` | $\mathcal{U} = E^\star$ exactly, (7) | ~3 ms/step, $\tau \approx 10$–$20$ | When every shape needs the same deformation energy |
| `crw` | same as polar | about 2× polar per independent sample; many projection failures | Cross-checking polar; level sets that aren't star-shaped |
| `ray` | biased; needs importance weights | independent, cheap | Browsing only |

## 8. Benchmark problems

For each total energy $E$, a pool of microcanonical samples is drawn, and an evenly spaced subset of it becomes the
problems. A sample $(q_s, \dot q_s)$ gives the start state $x_s = X + u(q_s)$ and velocity $v_s = J(q_s)\, \dot q_s$.
A benchmark dataset is one HDF5 file with a group per named scene (e.g. the trampoline at several Young's moduli),
each holding the FEM its problems are posed on, its pools, and one problem set per kind and energy
(`make_benchmark.py`'s docstring has the layout). Each scene gets its own reduced model and pools.

- **Static problem.** Start from $x_s$ and minimize $\Psi(x)$ with Dirichlet nodes at rest. The solution is exactly
  $X$, so errors are $\lVert x_k - X \rVert$. Static pools are sampled without gravity by default.
- **Dynamic problem.** One implicit Euler (BDF1) step of size $h$,

$$
x^\star = \arg\min_x\ \frac{1}{2h^2} \lVert x - \tilde x \rVert_M^2 + \Psi(x),
\qquad \tilde x = x_s + h\, v_s + h^2 a_{\mathrm{ext}},
\tag{14}
$$

  with the scene gravity $a_{\mathrm{ext}}$ applied to the free degrees of freedom only.
  - **Sampling:** dynamic pools are sampled with gravity in $\mathcal{U}$ by default, consistent with the
    gravity-loaded dynamics conserving $E$.
  - **Reference solution:** $x^\star$ solves (14) for $\tilde x$ exactly as stored (float32, which the solvers
    read). It comes from full-order Newton with SPD-corrected Hessians $\nabla^2 \Psi + M/h^2$ and Armijo
    backtracking, run to the round-off floor of pbat's float32 energies (gradients accurate to about 3 N), and is
    stored in float32.
  - **Reference accuracy:** estimated per problem by the $M$-norm of the last Newton correction (`ref_error_mnorm`),
    the same metric as the benchmark's error $e=\lVert x - x^\star\rVert_M$.
    - On the trampoline, the stored references are within $6.5 \times 10^{-4}\,e_0$ of exact ones (float64 energies,
      converged to machine precision); the estimate is within a factor of 3 of that.
    - That is enough for error reductions up to 99% ($10^{-2}\,e_0$), the tightest the benchmark reports. Tighter
      targets would need float64 energies, and the float32 GPU solvers stop near the same floor anyway.

The time step regularizes the problem through the mass term. With $\lambda_{\min}, \lambda_{\max}$ the extreme
eigenvalues of $M^{-1} K$, the mass-scaled condition number of the step's Hessian is roughly

$$
\kappa(h) \approx \frac{\lambda_{\max} + h^{-2}}{\lambda_{\min} + h^{-2}} .
\tag{15}
$$

On the trampoline, $\lambda_{\min} = 9.4$ and $\lambda_{\max} = 5.6 \times 10^9$, so
$\kappa(\infty) \approx 6.0 \times 10^8$ (static), $\kappa(10^{-2}) \approx 5.6 \times 10^5$ and
$\kappa(10^{-4}) \approx 57$.

The two dynamic time steps always share samples: $10^{-2}$ s, a typical step without substepping, and $10^{-4}$ s,
the equivalent of 100 substeps. Static and dynamic problems share samples when both are sampled with the same gravity
setting.

## 9. Shape matching: what can the reduced model express?

Sampling only explores shapes the manifold can produce. To verify the range of configurations it can express,
physically meaningful full-order target shapes are generated, and the reduced models are matched to them both
geometrically and energetically.

### 9.1 Targets

A target is a full-order static equilibrium $u_t$ of $\Psi$ under one of three authored conditions, chosen in the GUI
with sliders (no mesh picking):

- **Gravity:** a load $f = Mg$ for a chosen acceleration $g$, e.g. a cantilever bending down.
- **Region force:** a total force $F$ on the nodes of a box region $\mathcal{B}$, distributed in proportion to their
  masses, $f_a = \frac{m_a}{\sum_{b \in \mathcal{B}} m_b} F$ for $a \in \mathcal{B}$. The box is given per axis as
  fractions of the mesh's bounding box, e.g. $x \in [0.95, 1]$ for the tip of a beam.
- **Handle displacement:** the nodes of a box region are moved by a prescribed $d$ (handles), and the rest of the mesh
  relaxes to equilibrium.

Each target solves

$$
u_t = \arg\min_u\ \Psi(X + u) - f^\top u
\quad \text{s.t. } u_D = 0,\ u_{\mathcal{B}} = d \ \text{(handles only)},
\tag{16}
$$

by load-stepped Newton. The load or handle displacement is ramped over a few increments, and each increment is solved
by Newton from the previous one. Newton uses SPD-corrected Hessians and Armijo backtracking, and runs to the float32
round-off floor. A target takes 2–6 s on the 5k-DOF beam.

### 9.2 Matching the reduced models

Three models are compared:

- $U$, the $m$ vibration modes.
- $[U\ Z]$, an augmented linear basis. $Z$ holds the $M$-orthogonal SVD directions of the projected modal derivatives
  (3) whose singular values exceed 0.02 of the largest (95 columns on the beam).
- The quadratic manifold (4), with the same $m$ coordinates as $U$.

They are matched to a target in three ways:

$$
\begin{aligned}
&\text{geometric:} && \min_q\ \tfrac12 \lVert u(q) - u_t \rVert_M^2 && (17)\\
&\text{energy-regularized:} && \min_q\ \frac{\lVert u(q) - u_t \rVert_M^2}{2 \lVert u_t \rVert_M^2}
  + \alpha\, \frac{\Psi(X + u(q)) - \Psi(X)}{\Psi_t},
  \qquad \Psi_t = \Psi(X + u_t) - \Psi(X) && (18)\\
&\text{ROM equilibrium:} && \min_q\ \Psi(X + u(q)) - \Psi(X) - f^\top u(q)
  \ \Big[ + \tfrac{\kappa}{2} \sum_{a \in \mathcal{B}} m_a \lVert u_a(q) - d_a \rVert^2 \Big] && (19)
\end{aligned}
$$

- **Geometric best fit (17)** asks whether the model *contains* the shape.
  - For a linear basis $B$ it is the $M$-orthogonal projection $u = B\,(B^\top M B)^{-1} B^\top M u_t$.
  - For the quadratic manifold it is a nonlinear least-squares problem in $m$ unknowns, solved by
    Levenberg–Marquardt with residual $M^{1/2}(u(q) - u_t)$ and Jacobian $M^{1/2} J(q)$, starting from the projection
    onto $U$.
- **Energy-regularized fit (18)** adds $\alpha$ times the deformation energy to the geometric misfit. Both terms are
  normalized by the target's, so $\alpha$ is dimensionless.
  - It is solved by L-BFGS with gradient $J^\top M (u - u_t) / \lVert u_t \rVert_M^2 + \alpha\, \nabla_q
    \mathcal{U} / \Psi_t$.
  - It trades geometric fidelity for plausibility. In practice $\alpha \approx 10^{-3}$–$10^{-2}$ removes spurious
    strain while staying close, and $\alpha \approx 1$ already collapses the shape toward rest.
- **ROM equilibrium (19)** is what the reduced model *predicts* under the same conditions: its own static equilibrium.
  - The load is ramped over the same increments, with warm-started L-BFGS, so the solve stays on the load path from
    rest.
  - Handles become a mass-weighted penalty, because a reduced model generally can't match many prescribed degrees of
    freedom exactly. The penalty uses $\kappa = 100\, \omega_{\max}^2$, so it is stiffer than any mode.
  - The linear-mode ROM equilibrium (same energy, $u = Uq$) is also available as a baseline.

### 9.3 Metrics

Each fit $u$ is scored by:

- the relative mass-weighted error $\lVert u - u_t \rVert_M / \lVert u_t \rVert_M$;
- the largest vertex error $\max_a \lVert u_a - u_{t,a} \rVert$;
- its deformation energy $\Psi(X + u) - \Psi(X)$, against the target's $\Psi_t$.

The GUI draws the fit with the target as a transparent ghost, and the per-vertex error as a color.

### 9.4 Why linear-mode targets aren't enough

Since $U^\top M Q = 0$, for any linear target $U\eta$,

$$
\lVert u(q) - U\eta \rVert_M^2 = \lVert U(q - \eta) \rVert_M^2 + \tfrac14 \lVert Q{:}qq \rVert_M^2 .
\tag{20}
$$

Fitting the manifold to linear-mode targets, with or without the energy term, can only shrink the quadratic
correction, never enhance it. The "enhanced" counterpart of a linear shape is the manifold point $q = \eta$ itself.
Meaningful tests therefore need genuinely nonlinear targets such as (16).

### 9.5 Findings

Cantilever beam under gravity $g$ (relative $M$-norm errors):

| $g$ (m/s²) | tip sag | $U$ best fit (20 modes) | $[U\ Z]$ best fit (95 columns) | quadratic best fit (20 coordinates) | quadratic ROM equilibrium |
|---|---|---|---|---|---|
| 0.05 | 0.046 m | 0.1% | 0.0% | 0.0% | 1.2% |
| 0.40 | 0.366 m | 0.8% | 0.0% | 0.1% | 11% |
| 0.80 | 0.709 m | 1.5% | 0.1% | 0.4% | 32% |

Energies at $g = 0.4$ (target $\Psi_t = 7.9$ J):

| Fit | relative error | max vertex error | energy |
|---|---|---|---|
| $U$ best fit | 0.8% | 4 mm | 152 J |
| $[U\ Z]$ best fit | 0.0% | < 1 mm | 8.2 J |
| quadratic best fit | 0.1% | 1 mm | 8.4 J |
| quadratic, energy-regularized ($\alpha = 10^{-3}$) | 0.3% | 1 mm | 8.4 J |
| quadratic ROM equilibrium | 11% | 43 mm | 6.6 J |
| linear-mode ROM equilibrium | 68% | 262 mm | 1.9 J |

With a handle moving the beam's tip 0.5 m down, the quadratic ROM equilibrium is within 4.9% (relative). The
linear-mode ROM is within 7.3%, but stores 97 J against the target's 26 J.

> **Interpretation.** The quadratic manifold *contains* large bending shapes geometrically, with nearly the right
> energy (8.4 J vs 7.9 J). A linear basis that fits as well geometrically carries 20× too much strain. But the
> reduced model's own equilibrium is too stiff: it is off by 11–32% at 12–24% tip sag. This is a locking-like effect:
> the manifold has shapes that *look* right, but its energy landscape doesn't lead to them. Reachability should
> therefore be judged with the energy-aware fits (18)–(19) as well as the geometric one (17).

## 10. Validation

- **Level-set samplers agree.** On the beam at $E^\star = 10$ J, polar (two seeds) and the constrained random walk
  give mean peak displacements of 0.112, 0.115 and 0.119 m (±0.0025), with consistent per-mode energy shares. Every
  sample satisfies $\lvert \mathcal{U} - E^\star \rvert \le 5 \times 10^{-5} E^\star$.
- **Analytic tests.** For $\mathcal{U} = \tfrac12 q^\top A q$, the weight-1 level-set samplers make $A^{1/2} q$ uniform
  in direction. A polar chain with the Jacobian terms of (7) removed fails this test: its direction shares are
  0.14 / 0.33 / 0.53 instead of 1/3 each.
- **Microcanonical equipartition.**
  - For a harmonic $\mathcal{U}$ with constant mass, $\mathcal{U}/E \sim \mathrm{Beta}(d/2, d/2)$.
  - The same system written in nonlinear coordinates $z = q + cq^3$, so that $M_r(q) =
    \mathrm{diag}\big((1 + 3cq^2)^2\big)$, must give the same law. Without $\sqrt{\det M_r}$, the mean of
    $\mathcal{U}/E$ drops to 0.40 instead of 0.5.
- **Beam, $E = 10$ J.**
  - $K + \mathcal{U} = E$ to $5 \times 10^{-16}$ relative, recomputed from the exported file.
  - $\mathcal{U}/E$ has mean 0.498 and standard deviation 0.111. $\mathrm{Beta}(10, 10)$ predicts 0.5 and 0.109 for
    20 modes in the near-harmonic regime.
  - Split-$\hat R$ is 1.044.
- **Trampoline mixing.** Pools of 400 from 4 chains give $\hat R = 1.05 / 1.26 / 1.14$ at $E = 50 / 500 / 5000$ J (ESS
  92 / 42 / 46), so high energies need larger pools or longer chains.
- **Trampoline shapes.** At $E = 50 / 500 / 5000$ J, the median peak displacement is 14 / 24 / 36 cm on the 2 m plate,
  and $\mathcal{U}/E \approx 0.35 / 0.32 / 0.28$. At 5000 J some elements invert (min $\det F = -1.6$).
- **Mode normalization.** Switching from Euclidean to mass-normalized modes reproduces $Q$ up to the entrywise
  rescaling $s_i s_j Q_{ij}$, to 0.3% (the float32 accuracy of (2)), with identical shape statistics.

## 11. Code map

| Topic | Implementation (`python/examples`) |
|---|---|
| Modes, modal derivatives, projection (2)–(3) | `modalrom.vibration_modes`, `modal_derivatives`, `m_orthogonal_complement`, `build_rom` |
| Reduced model, whitening (4)–(5) | `modalrom.reduced_model`, `rom_mcmc.QuadraticManifold` |
| Level-set samplers (Section 3) | `rom_mcmc.PolarMetropolis`, `radial_root`, `ConstrainedRandomWalk`, `RaySampler` |
| Microcanonical sampler (Section 4) | `rom_mcmc.MicrocanonicalMetropolis` |
| Adaptation, thinning, $\hat R$ (Section 5) | `rom_mcmc._Chain`, `integrated_autocorrelation_time`, `split_rhat`, `run_chain`, `run_chains` |
| Noise floor (Section 6) | `rom_mcmc.estimate_energy_noise`, `samples_energy_noise`, `pool_energy_noise` |
| Sample files | `rom_mcmc.write_samples` (the file contract) |
| Benchmark problems and the dataset file (Section 8) | `make_benchmark.py`, `rom_reach.static_equilibrium` (with inertia) |
| Shape matching (Section 9) | `rom_reach.py`, and the reachability panel in `modalrom.py` |
| Browsing datasets | `benchmark_browser.py` |

## References

1. S. R. Idelsohn and A. Cardona. A reduction method for nonlinear structural dynamic analysis. *Computer Methods in
   Applied Mechanics and Engineering* 49(3):253–279, 1985.
2. J. Barbič and D. L. James. Real-time subspace integration for St. Venant–Kirchhoff deformable models. *ACM
   Transactions on Graphics* 24(3), 2005.
3. S. Jain, P. Tiso, J. B. Rutzmoser and D. J. Rixen. A quadratic manifold for model order reduction of nonlinear
   structural dynamics. *Computers & Structures* 188:80–94, 2017.
4. B. Smith, F. de Goes and T. Kim. Stable Neo-Hookean flesh simulation. *ACM Transactions on Graphics* 37(2), 2018.
5. E. Zappa, M. Holmes-Cerfon and J. Goodman. Monte Carlo on manifolds: sampling densities and integrating functions.
   *Communications on Pure and Applied Mathematics* 71(12):2609–2647, 2018.
6. G. O. Roberts, A. Gelman and W. R. Gilks. Weak convergence and optimal scaling of random walk Metropolis algorithms.
   *Annals of Applied Probability* 7(1):110–120, 1997.
7. H. Robbins and S. Monro. A stochastic approximation method. *Annals of Mathematical Statistics* 22(3):400–407, 1951.
8. C. Andrieu and J. Thoms. A tutorial on adaptive MCMC. *Statistics and Computing* 18:343–373, 2008.
9. A. D. Sokal. Monte Carlo methods in statistical mechanics: foundations and new algorithms. In *Functional
   Integration*, Springer, 1997.
10. A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin. *Bayesian Data Analysis*, 3rd ed.,
    CRC Press, 2013.
