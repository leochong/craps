# Feasibility Study — Physics-Based State Estimation and Stochastic Simulation of Electronic Bubble Craps
## Mathematical Foundations, Prototype Design, and Empirical Validation

---

**Department:** Mathematics
**Project type:** Research feasibility study (prototype creation + empirical testing)
**Suggested duration:** 14–16 weeks (one term), extendable to two terms
**Primary audience:** Math department faculty, casino gaming-integrity and regulatory stakeholders
**Keywords:** inverse problems, differentiable contact dynamics, system identification, stochastic simulation, Monte Carlo estimation, expected value, electronic gaming device integrity

---

## Abstract

Electronic *bubble craps* is a faster variant of the traditional dice game, played on a sealed, air-blast-driven dome in which two dice are launched and settle into a terminal roll whose sum determines the outcome. The game's outcome is a physical process: the initial pose and launch energy of the dice, combined with the contact properties of the machine, jointly determine the terminal face states. This raises a natural mathematical and engineering question of feasibility: to what extent is the uncertain terminal outcome *predictable*, and how tightly can its probability distribution be bounded, given a noisy pre-launch observation of the dice?

This document proposes a feasibility study that answers that question with a rigorous, quantitative methodology. The project constructs a research prototype that (1) extracts six-degree-of-freedom (6DoF) dice pose from high-frame-rate video, (2) calibrates an *inverse* physics model of the machine via differentiable contact mechanics and system identification, (3) propagates residual uncertainty through a GPU-accelerated Monte Carlo simulator to obtain a terminal-outcome probability vector, and (4) translates that vector into an expected-value (EV) analysis of the game's wager set.

The study treats the machine as a *vulnerability-assessment target* and is framed for a casino-operator and regulatory audience as much as for the mathematical community. The central deliverables are not a consumer product but: (a) a formal statement of identifiability and convergence for the inverse/estimation pipeline, (b) a calibrated digital twin, (c) an empirical measurement of how far the achievable edge can rise above the published house edge, and (d) documented countermeasures. All fieldwork is to be conducted with operator consent and under an ethics review.

---

## 1. Introduction and Research Questions

### 1.1 Motivation

Casino gaming devices derive their integrity from two assumptions: that the underlying process is effectively random, and that no observation available to a player contains exploitable information about the outcome. Traditional bubble craps is marketed as a physical (rather than software RNG) process — the dice are mechanically launched and physically settle. The physicality that makes the game appear "fair" is also the property that makes it, in principle, *observable*. A camera with sufficient frame rate can record the pre-launch resting pose of the dice and the initial launch transient.

The foundational question is therefore not "can dice be predicted" but rather a quantitative one:

- Given an observation with bounded measurement error, how much information about the terminal outcome is actually recoverable, and how rapidly does estimation error grow as the process becomes chaotic after successive impacts?

This is, at its core, a problem in **sensitivity of dynamical systems to initial conditions coupled with systematic (non-random) parameters** — a well-posed research program in applied mathematics.

### 1.2 Research Questions

The study is organized around five research questions (RQs):

1. **RQ1 — Identifiability.** Given a noisy 6DoF pose observation and a video sequence of the launch, are the machine's physical parameters (friction field $\mu$, restitution $e$, air-blast impulse $\mathbf{J}_{\text{air}}$) *identifiable* — that is, uniquely recoverable in the limit of infinite data up to the measurement noise floor?
2. **RQ2 — Controllability of uncertainty.** How does measurement error in pose propagate through the estimated launch impulse into the terminal-outcome distribution $P_{2..12}$, and can that propagation be bounded by an adjoint sensitivity?
3. **RQ3 — Convergence.** How many Monte Carlo rollouts are required to estimate $P_{2..12}$ to a specified $\ell_1$ error with a specified confidence, and how do sub-step integrator error and contact heterogeneity trade off against rollout count?
4. **RQ4 — Predicted edge.** For each legal wager in the bubble-craps set, what is the achievable player edge relative to the published house edge, and does any estimated edge exceed a binary positive-EV decision threshold?
5. **RQ5 — Feasibility and countermeasures.** Under realistic noise, latency, and hardware constraints, is a statistically significant edge achievable in practice, and what machine-design or operational countermeasures restore the integrity assumption?

---

## 2. Background and Literature

### 2.1 Electronic gaming device integrity

Electronic table games and "bubble" craps machines (Interblock, Aruze/IGT) replace the croupier with electro-mechanical actuators. Device integrity rests on hardware RNG certification and, for physical-launch games, on the assumed unpredictability of the mechanical outcome. A useful mathematics-adjacent literature concerns *mechanical randomness* and the physical detection of bias in dice and wheels — see, e.g., the analysis of roulette-wheel physics (Small & Tse, "Predicting the outcome of roulette", *Chaos*, 2012), which demonstrated that a low-cost camera and a simple dynamical model can recover exploitable information from a physical casino game. This project generalizes that approach from a continuous (wheel) system to a *contact-rich, impulsive, stochastic* system.

### 2.2 Rigid-body dynamics and contact mechanics

The dice in a bubble machine undergo a sequence of impacts (dice–dice, dice–felt, dice–dome, dice–bumper). The relevant theory is *nonsmooth mechanics*: Signorini complementarity, Coulomb friction, and Newton/Poisson restitution. Modern solvers — Sequential Impulse (PBD) and complementarity-based (Anitescu & Potra) methods — trade physical fidelity against fixed-timestep stability. Signed Distance Fields (SDFs) provide $\mathcal{O}(1)$ collision queries and are standard in GPU rigid-body engines.

### 2.3 Inverse problems and differentiable simulation

System identification from trajectory data is an *inverse problem*. Differentiable simulators (which propagate gradients through the forward physics) make gradient-based identification tractable; the adjoint method is the classical machinery for efficient gradient computation. The project draws on differentiable rigid-body simulators (e.g., Brax, Warp, Isaac) but, given the target hardware, pivots toward a bespoke Metal/C++ engine with hand-derived analytical adjoints.

### 2.4 Stochastic simulation and estimation theory

The pipeline's terminal stage is a Monte Carlo estimator of a categorical distribution. Relevant theory: strong laws for empirical categorical means, Hoeffding/Bernstein/Dvoretzky–Kiefer–Wolfowitz bounds for distribution estimation, importance sampling and control-variate variance reduction, and the relationship between sub-step integration error and the effective dimensionality of the uncertain initial ensemble.

### 2.5 Expected value and betting mathematics

Craps EV is standard combinatorial probability: each wager is a Bernoulli/multinomial payoff against a known house edge. The bubble-craps wager set (see §6) includes Place and Buy bets with house edges typically 1.5–5.5%. The Kelly criterion and edge-threshold decisions supply the decision-theoretic layer that converts a probability estimate into a wagering signal. This literature is well-established and pedagogically clean, which makes it an appropriate venue for validating the estimation pipeline.

---

## 3. Mathematical Formulation

### 3.1 Forward model: rigid-body dynamics with contact

Let the system state be $\mathbf{x} = (\mathbf{q}_1, \dot{\mathbf{q}}_1, \mathbf{q}_2, \dot{\mathbf{q}}_2)$ for two dice, where $\mathbf{q}_i \in SE(3)$ is the pose (position + orientation quaternion) of die $i$. The forward dynamics follow

$$
\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) = \mathbf{f}_{\text{ext}} + \mathbf{f}_{\text{contact}}(\mathbf{q},\dot{\mathbf{q}};\, \mu, e) ,
$$

with $\mathbf{f}_{\text{ext}}$ the gravity and launch impulse, and $\mathbf{f}_{\text{contact}}$ resolving contact via the SDF-based closest-distance map. The contact model is summarized by the parameter vector

$$
\boldsymbol{\theta} = \big(\mu_{\text{felt}},\, \mu_{\text{dome}},\, e_{\text{dice}},\, e_{\text{bumper}},\, \mathbf{J}_{\text{air}}(t)\big) .
$$

Given an initial state $\mathbf{x}_0$ (the pre-launch pose plus launch energy) and parameters $\boldsymbol{\theta}$, the forward map is

$$
\mathcal{F} : (\mathbf{x}_0, \boldsymbol{\theta}) \mapsto \mathbf{x}_f,
$$

where $\mathbf{x}_f$ is the terminal resting state. The game outcome is the *face-up pair*; we write the outcome as a function $S(\mathbf{x}_f) \in \{2,\dots,12\}$.

**Sub-step integrator.** Because impacts are impulsive and the target latency is tight, the engine uses fixed sub-steps $\Delta t = 1.0\,\text{ms}$ with a Semi-Implicit Euler scheme and Sequential Impulse contact resolution. Integrator order and stability are analyzed in §7 (risk on contact discontinuities).

### 3.2 Inverse problem and adjoint sensitivity (RQ1–RQ2)

Given $M$ observed trajectories $\{\mathbf{x}^{(k)}(t_j)\}$ extracted from video, parameter identification solves

$$
\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \sum_{k=1}^{M} \sum_{j} \big\| \Pi\big(\mathcal{F}(\mathbf{x}_0^{(k)},\boldsymbol{\theta}; t_j)\big) - \mathbf{y}^{(k)}_j \big\|^2_{\Sigma_j} ,
$$

where $\Pi$ is the camera projection, $\mathbf{y}_j^{(k)}$ the measured keypoints, and $\Sigma_j$ the per-measurement covariance. Gradients $\partial \mathcal{L}/\partial \boldsymbol{\theta}$ are computed via the **adjoint method** (forward pass then a backward pass solving the adjoint of the tangential dynamics), enabling L-BFGS.

**Identifiability.** Local identifiability is assessed through the rank and conditioning of the empirical Fisher/pseudo-Hessian at convergence; where the gradient stalls at contact discontinuities, a derivative-free **CMA-ES** fallback provides global robustness. This dual optimization (local gradient + global evolutionary) is itself a research deliverable documenting *when* the inverse problem is ill-conditioned.

### 3.3 Stochastic simulation and uncertainty quantification (RQ3)

Pose measurement error induces an ensemble of plausible initial states $\{\mathbf{x}_0^{(i)}\}_{i=1}^{N}$ sampled from the uncertain observation. Propagating the ensemble through $\mathcal{F}$ yields terminal faces and an empirical distribution

$$
\widehat{P}_s = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\{S(\mathbf{x}_f^{(i)}) = s\}, \qquad s = 2,\dots,12 .
$$

Concentration of $\widehat{P}$ around the true $P$ is bounded via the DKW/Hoeffding inequalities; variance reduction (stratified sampling over pose priors, control variates on contact count) reduces the rollout count for a fixed error target. The **effective information gain** of the prediction is measured by the Kullback–Leibler divergence

$$
D_{\text{KL}}\big(\widehat{P} \,\|\, P^{\text{fair}}\big),
$$

where $P^{\text{fair}}$ is the uniform die-face distribution. This is the central quantity that the whole pipeline must push measurably above zero.

### 3.4 Estimation theory: pose extraction

The 6DoF pose is estimated from detected die keypoints (corners/pips) via Perspective-n-Point (PnP). The measurement model is

$$
\mathbf{z} = \Pi(\mathbf{x}) + \boldsymbol{\eta}, \qquad \boldsymbol{\eta} \sim \mathcal{N}(0, \Sigma),
$$

with $\Sigma$ estimated from detection residuals. The inverse covariance-weighted PnP solution provides a MAP pose estimate whose uncertainty is propagated into the launch-impulse estimate and thence into §3.3. This closes the loop: **pose noise → impulse noise → terminal-distribution broadening**.

### 3.5 Expected value and decision theory (RQ4)

For a wager $w$ with outcome probability $P_w$ (derived from $\widehat{P}$) and payout odds $\text{odds}(w)$, the expected value is

$$
\text{EV}(w) = P_w\cdot \text{payout}(w) + (1-P_w)\cdot(-1).
$$

The edge is $\text{EV}(w) \times 100\%$. A binary bet flag fires when edge exceeds threshold $\tau = +2.0\%$. Optimal stake-sizing, conditional on the estimate and the decision, follows fractional Kelly,

$$
f^* = \max\left(0,\ \frac{P_w \cdot \text{odds}(w) - (1-P_w)}{ (1-P_w)}\right) ,
$$

with a fractional-Kelly multiplier applied to respect estimation uncertainty. The study's analytical contribution here is the **sensitivity of the decision boundary to estimation error** — i.e., how much the edge must exceed $\tau$ before the expected loss from mis-calibrated $\widehat{P}$ is dominated by the true edge.

---

## 4. Prototype Design (Prototyping Creation)

The prototype is a self-contained research instrument, not a product. Its purpose is to *measure*, with known error bars, the quantities of §3.

### 4.1 System architecture

```
[ Live Video / Frame Grab ]  (1080p @ 120/240 fps, AVFoundation)
              |
              v
[ M1 · CV & Pose Pipeline ]
  · floor-plane homography & (0,0,0) calibration
  · YOLOv8-Keypoint (ANE/CoreML) corner + pip detection
  · PnP 6DoF pose with covariance
              |
              v
[ M2 · Inverse Launch Physics ]
  · adjoint L-BFGS (CMA-ES fallback)
  · recovers μ, e, J_air from observed transient
              |
              v
[ M3 · Digital Twin (Metal/C++) ]
  · SDF dome + sub-step sequential-impulse solver (Δt = 1 ms)
              |
              v
[ M4 · Monte Carlo Engine ]
  · 1,000–2,000 parallel rollouts (Metal compute)
  · terminal distribution P_{2..12}
              |
              v
[ M5 · EV / Decision Engine ]
  · per-wager EV table, edge threshold, Kelly sizing
```

### 4.2 Latency budget (target)

| Stage | iPhone 15/16 Pro | Next-gen | Unit |
|:---|:---|:---|:---|
| Frame grab | 8.33 ms (120 fps) | 4.16 ms (240 fps) | `CVPixelBuffer` |
| Vision + pose | 10.0 ms | 4.0 ms | ANE + CoreML |
| Inverse optimization | 15.0 ms | 8.0 ms | Metal compute |
| Monte Carlo (1k–2k) | 12.0 ms | 5.0 ms | Metal shaders |
| **End-to-end** | **≈45.3 ms** | **≈21.2 ms** | — |

Note: latency is an *instrument* metric here — small enough that the estimator can be treated as operating inside a single game instance; the feasibility study's comparisons are against the machine's launch-to-lockout window.

### 4.3 Host and dependency assumptions

- Native iOS (Swift/UIKit) host with AVFoundation capture at 120/240 fps.
- Metal compute for the physics and Monte Carlo stages.
- CoreML/ANE for the keypoint detector; OpenCV for homography and geometry.
- `draft-project-plan.md` in the project root remains the dated technical spec and is superseded by this document for the math-department submission.

---

## 5. Experimental Design and Testing

### 5.1 Datasets

| Dataset | Source | Purpose |
|:---|:---|:---|
| Reference capture | `IMG_2111.MOV` (~3.3 GB raw) | Initial pose/homography and transient analysis |
| Annotation corpus | 50–100 self-captured clips | CV training + pose ground truth |
| Calibration corpus | 30–50 high-fps machine clips | System identification (RQ1) |
| Held-out validation | 15–20 clips | Generalization and edge measurement (RQ4) |

### 5.2 Metrics and acceptance criteria

- **Pose residual:** PnP reprojection residual $\le 1.5$ px; tracking latency $< 10$ ms (Milestone 1).
- **Simulator throughput:** $1{,}000$ rollouts of 2.0 s in $\le 12.5$ ms on an A17/A18 device (Milestone 2).
- **Trajectory fidelity:** cumulative spatial divergence $< 8\%$ across the first three impacts (Milestone 3).
- **Probability calibration:** reliability diagrams and Brier score for $\widehat{P}$ vs. held-out frequencies.
- **Edge:** measured $D_{\text{KL}}(\widehat{P}\|P^{\text{fair}})$ and per-wager empirical edge vs. published house edge (Milestone 4).
- **End-to-end latency:** $\le 45$ ms sustained.

### 5.3 Test matrix

1. **Unit** — integrator convergence against analytical single-impact benchmarks (order-of-convergence study on $\Delta t$).
2. **Integration** — pose → impulse → simulator → distribution, with injected synthetic noise, measuring the RQ2 propagation curve.
3. **Field** — operator-consented calibration session on a real machine; in-lab with controlled launches otherwise.

---

## 6. Betting-Strategy and Casino-Audience Study (RQ4)

### 6.1 Wager set

For each of the sum outcomes in $\{2,\dots,12\}$ (and, in *crapless* variants, the absence of the "craps" immediate-loss rule on 2, 3, 12), the study evaluates **Place** and **Buy** wagers. Outcomes carry the standard published house edges; the research value is in measuring the *predicted* edge under the model.

### 6.2 EV table and thresholding

For each wager the study tabulates: $P_w$ (fair), $\widehat{P}_w$ (model), payout odds, house edge, predicted edge, and Kelly fraction. A positive-EV flag fires when predicted edge $> +2.0\%$ and the lower 95% confidence bound on $\widehat{P}_w$ remains profitable.

### 6.3 Casino-audience framing

This module is written to be directly useful to two external stakeholders:

- **Operators** — a quantitative vulnerability report: *"how predictable is the physical outcome under a camera-assisted adversary, and what is the worst-case achievable edge?"*
- **Regulators / integrity labs** — a reproducible methodology and metric set for auditing physical-launch electronic games, extending the roulette-physics precedent to contact-rich dice games.

### 6.4 Countermeasures analysis

Given measured predictability, the study evaluates countermeasures and their quantitative effect (reduction in $D_{\text{KL}}$): randomizing launch impulse, adding post-launch actuator jitter, reducing pre-launch visual exposure, screen/dome opacity, and increasing settle-time entropy.

---

## 7. Timeline (14–16-week mapped plan)

| Phase | Weeks | Focus | Milestone |
|:---|:---|:---|:---|
| P1: CV & HITL | 1–4 | camera pipeline, homography, pose, annotation UI | M1: 6DoF tracking < 10 ms |
| P2: Physics engine | 5–8 | SDF dome, contact solver, adjoint gradients | M2: 1k rollouts in 12.5 ms |
| P3: System ID | 9–12 | dataset processing, L-BFGS/CMA-ES fitting, friction grid | M3: < 8% divergence over 3 impacts |
| P4: EV & validation | 13–16 | Monte Carlo aggregation, EV/Kelly engine, field validation, report | M4: full feasibility report |

---

## 8. Deliverables

1. **Feasibility report** — this document matured with empirical results and the RQ answers.
2. **Annotated dataset** — pose-labeled corpus with a documented labeling protocol.
3. **Research codebase** — the five-module pipeline and reproducibility scripts.
4. **Calibrated digital-twin parameters** — $\mu,\, e,\, \mathbf{J}_{\text{air}}$ and the spatial friction grid.
5. **Integrity/countermeasure report** — for operator/regulator stakeholders.
6. **Reproducibility package** — environment, hyperparameters, and random seeds.

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|:---|:---|:---|
| Dome refraction / glare corrupts pose | High | Shader calibration grid for acrylic distortion; ROI masking on floor plate |
| Contact-discontinuity stalls L-BFGS | Medium | CMA-ES fallback; sub-step regularization |
| Identifiability failure (RQ1) | High | Reduced parameter set; spatial grid coarsening; simulated-annealing warm starts |
| Motion blur / low-fps sources | Medium | RIFE frame interpolation; optical-flow (CoTracker) bootstrapping |
| Insufficient field access | Medium | Lab-controlled launch rig as validation surrogate |
| Estimation-error swamps edge | High | Decision threshold raised; uncertainty-quantified betting (Kelly multiplier) |

---

## 10. Ethics, Legal, and Regulatory Considerations

- The study is a **feasibility and vulnerability assessment**, intended for casino-integrity and regulatory use, and for publication as applied mathematics.
- Any use of predictive devices on live casino premises is subject to jurisdiction-specific prohibitions (e.g., Nevada device statutes) and operator terms; consequently, all field data collection requires **explicit operator consent and an ethics-board approval**.
- The prototype will not be distributed, and the report will de-identify specific venues and machines unless permission is granted.
- Findings are framed to *restore* device integrity (countermeasures) rather than to provide an unfair-advantage tool.

---

## 11. References (indicative)

- Small, M. & Tse, C.-K. (2012). Predicting the outcome of roulette. *Chaos*, 22(3).
- Anitescu, M. & Potra, F. (1997). Formulating dynamic multi-rigid-body contact with friction as solvable LCP. *Nonlinear Dynamics*.
- Catto, E. (2011/2013). Iterative dynamics / soft constraints (Sequential Impulse). GDC.
- Osher, S. & Fedkiw, R. (2003). *Level Set Methods and Dynamic Implicit Surfaces*. Springer (SDF background).
- Cao, Q., et al. — differentiable rigid-body simulators (Brax/Warp) for adjoint contacts.
- Dvoretzky, Kiefer, Wolfowitz (1956). Asymptotic minimax character of the sample distribution function.
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*.
- Hannum, R. & Cabot, A. — house edge and gambling mathematics (survey).

*(References to be finalized and expanded during literature review.)*

---

## Appendix A — Parameter summary table

| Symbol | Meaning | Units |
|:---|:---|:---|
| $\mu_{\text{felt}}, \mu_{\text{dome}}$ | Coulomb friction coefficients | dimensionless |
| $e_{\text{dice}}, e_{\text{bumper}}$ | coefficients of restitution | dimensionless |
| $\mathbf{J}_{\text{air}}(t)$ | air-blast impulse profile | N·s |
| $\Delta t$ | simulation sub-step | 1.0 ms |
| $\tau$ | positive-EV decision threshold | +2.0% |

## Appendix B — Notation

| Notation | Meaning |
|:---|:---|
| $\mathbf{q}_i$ | pose of die $i$, $SE(3)$ |
| $\mathcal{F}$ | forward map (initial state + params → terminal state) |
| $\Pi$ | camera projection |
| $P_s, \widehat{P}_s$ | true / estimated terminal-sum distribution |
| $P^{\text{fair}}$ | uniform fair-face distribution |
| $D_{\text{KL}}$ | Kullback–Leibler divergence |
| $f^*$ | fractional-Kelly stake |