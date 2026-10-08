Markdown# Crapless Bubble Craps Prediction Engine: Technical Specification & Master Implementation Plan

## Executive Summary
This document details the end-to-end technical specification, system architecture, and 16-week implementation plan for an iOS-based predictive system targeting electronic Crapless Bubble Craps machines (e.g., Interblock, Aruze). The system captures pre-launch 6DoF dice poses, leverages an inverse physics digital twin calibrated via differentiable contact mechanics, executes parallel stochastic Monte Carlo rollouts, and identifies positive expected value ($\text{EV}$) betting opportunities within a strict $< 50\text{ ms}$ processing latency budget.

---

## 1. System Architecture Pipeline

[ Live Video / Frame Grab ] (1080p @ 120fps / 240fps)|v[ Module 1: Computer Vision & HITL Pipeline ]Floor plane homography & (0,0,0) origin calibrationNeural keypoint detection & Perspective-n-Point (PnP) 6DoF pose solverActive-learning Human-in-the-Loop (HITL) annotation overlay|v[ Module 2: Inverse Launch Physics & Parameter Fitting ]Adjoint gradient-descent optimizer (L-BFGS)Calibrates friction ($\mu$), restitution ($e$), and air blast impulse ($\mathbf{J}_{\text{air}}$)|v[ Module 3: Digital Twin Physics Engine ]GPU-accelerated Metal/C++ rigid-body simulationSigned Distance Field (SDF) dome geometry & sub-step contact dynamics ($\Delta t = 1.0\text{ ms}$)|v[ Module 4: Stochastic Monte Carlo Simulation Engine ]Runs 1,000–2,000 parallel rollouts on Metal compute shaders ($< 15\text{ ms}$)Terminal state face probability distribution vector $P_{2..12}$|v[ Module 5: Expected Value (EV) Decision Engine ]Kelly criterion sizing & positive-EV place/buy bet flags ($> +2.0\%\text{ EV}$)
---

## 2. Hardware Profiles & Latency Targets

### Total Execution Budget: $< 50\text{ ms}$

| Pipeline Stage | Baseline Target (iPhone 15/16 Pro) | Next-Gen Target (iPhone 17/18 Pro) | Target Hardware Unit |
| :--- | :--- | :--- | :--- |
| **Sensor Frame Grab** | $8.33\text{ ms}$ ($120\text{ fps}$) | $4.16\text{ ms}$ ($240\text{ fps}$) | AVFoundation `CVPixelBuffer` |
| **Vision & Keypoint Pose Extraction** | $10.00\text{ ms}$ | $4.00\text{ ms}$ | Apple Neural Engine (ANE) + CoreML |
| **Inverse Vector Optimization** | $15.00\text{ ms}$ | $8.00\text{ ms}$ | Metal Compute C++ / Adjoint Solver |
| **Monte Carlo Ensemble (1,000x–2,000x)** | $12.00\text{ ms}$ | $5.00\text{ ms}$ | Metal Parallel Shaders |
| **End-to-End Latency** | **$\approx 45.33\text{ ms}$** | **$\approx 21.16\text{ ms}$** | **Sub-lockout window execution** |

---

## 3. Master 16-Week Implementation Roadmap

WEEKS  1-4  : [ Phase 1: CV Pipeline & HITL Active Learning ] =======> Milestone 1 DeliveredWEEKS  5-8  : [ Phase 2: Differentiable Rigid-Body Engine ] ========> Milestone 2 DeliveredWEEKS  9-12 : [ Phase 3: System Identification & Calibration ] ======> Milestone 3 DeliveredWEEKS 13-16 : [ Phase 4: Edge Deployment & EV Betting Engine ] ======> Milestone 4 Delivered crapless-bubble-craps-prediction Technical architecture and 16-week execution plan for building an iOS-based computer vision and differentiable physics prediction engine for Crapless Bubble Craps.
Instructions
Crapless Bubble Craps Prediction Engine: System Architecture & Master Implementation Plan
Executive Summary
This document details the end-to-end technical specification, system architecture, and 16-week implementation plan for an iOS-based predictive system targeting electronic Crapless Bubble Craps machines (e.g., Interblock, Aruze). The system captures pre-launch 6DoF dice poses, leverages an inverse physics digital twin calibrated via differentiable contact mechanics, executes parallel stochastic Monte Carlo rollouts, and identifies positive expected value ($\text{EV}$) betting opportunities within a strict $< 50\text{ ms}$ processing latency budget.

1. System Architecture Pipeline
  [ Live Video / Frame Grab ] (1080p @ 120fps / 240fps)
               |
               v
  [ Module 1: Computer Vision & HITL Pipeline ]
  * Floor plane homography & (0,0,0) origin calibration
  * Neural keypoint detection & Perspective-n-Point (PnP) 6DoF pose solver
  * Active-learning Human-in-the-Loop (HITL) annotation overlay
               |
               v
  [ Module 2: Inverse Launch Physics & Parameter Fitting ]
  * Adjoint gradient-descent optimizer (L-BFGS)
  * Calibrates friction ($\mu$), restitution ($e$), and air blast impulse ($\mathbf{J}_{\text{air}}$)
               |
               v
  [ Module 3: Digital Twin Physics Engine ]
  * GPU-accelerated Metal/C++ rigid-body simulation
  * Signed Distance Field (SDF) dome geometry & sub-step contact dynamics ($\Delta t = 1.0\text{ ms}$)
               |
               v
  [ Module 4: Stochastic Monte Carlo Simulation Engine ]
  * Runs 1,000–2,000 parallel rollouts on Metal compute shaders ($< 15\text{ ms}$)
  * Terminal state face probability distribution vector $P_{2..12}$
               |
               v
  [ Module 5: Expected Value (EV) Decision Engine ]
  * Kelly criterion sizing & positive-EV place/buy bet flags ($> +2.0\%\text{ EV}$)


2. Hardware Profiles & Latency Targets
Total Execution Budget: $< 50\text{ ms}$

Pipeline StageBaseline Target (iPhone 15/16 Pro)Next-Gen Target (iPhone 17/18 Pro)Target Hardware UnitSensor Frame Grab$8.33\text{ ms}$ ($120\text{ fps}$)$4.16\text{ ms}$ ($240\text{ fps}$)AVFoundation CVPixelBufferVision & Keypoint Pose Extraction$10.00\text{ ms}$$4.00\text{ ms}$Apple Neural Engine (ANE) + CoreMLInverse Vector Optimization$15.00\text{ ms}$$8.00\text{ ms}$Metal Compute C++ / Adjoint SolverMonte Carlo Ensemble (1,000x–2,000x)$12.00\text{ ms}$$5.00\text{ ms}$Metal Parallel ShadersEnd-to-End Latency$\approx 45.33\text{ ms}$$\approx 21.16\text{ ms}$Sub-lockout window execution


3. Master 16-Week Implementation Roadmap
WEEKS  1-4  : [ Phase 1: CV Pipeline & HITL Active Learning ] =======> Milestone 1 Delivered
WEEKS  5-8  : [ Phase 2: Differentiable Rigid-Body Engine ] ========> Milestone 2 Delivered
WEEKS  9-12 : [ Phase 3: System Identification & Calibration ] ======> Milestone 3 Delivered
WEEKS 13-16 : [ Phase 4: Edge Deployment & EV Betting Engine ] ======> Milestone 4 Delivered

Phase 1: High-Speed CV, Pose Extraction & HITL Framework (Weeks 1–4)

Task 1.1 (Camera Pipeline): Lock exposure, white balance, and manual focus at $120\text{ fps}$ via AVFoundation. Implement zero-copy CVPixelBuffer transfers directly to Metal memory buffers.
Task 1.2 (Pop-Plate Homography): OpenCV circular edge detection (HoughCircles) anchors world coordinate origin $(0,0,0)$ on the green felt plate.
Task 1.3 (Pose Estimation): Train quantized YOLOv8-Keypoint model on Apple Neural Engine to identify die corners and face pips, feeding a Perspective-n-Point (PnP) solver for 6DoF pose extraction.
Task 1.4 (Human-in-the-Loop Active Learning UI):

Swift/UIKit overlay rendering color-coded 3D bounding cubes over predicted poses.
Drag handles for keypoint corner corrections and 3-point pop-plate edge calibration.
$120\text{ fps}$ timeline scrubber and auto-accept thresholding ($\text{IoU} \ge 0.85$, PnP residual $\le 1.5\text{ px}$).


Task 1.5 (Data Collection): Capture and annotate 50–100 video clips across varied roll speeds, resting poses, and edge impact occlusions.


Milestone 1 (Week 4): Live camera feed tracks 6DoF dice coordinates and floor origin with $< 10\text{ ms}$ latency; HITL interface reduces clip annotation time to $< 30\text{ seconds}$ per video.


Phase 2: Differentiable Rigid-Body Physics Engine (Weeks 5–8)

Task 2.1 (SDF Dome Geometry): Analytical Signed Distance Fields (SDFs) representing glass cylinder walls, rubber bumper ring, and floor plate for $\mathcal{O}(1)$ collision queries.
Task 2.2 (Impulse Contact Solver): Metal compute shaders executing Sequential Impulse dynamics with surface friction ($\mu$) and coefficient of restitution ($e$) under sub-step numerical integration ($\Delta t = 1.0\text{ ms}$).
Task 2.3 (Adjoint Differentiation): Implement analytical forward/backward gradient passes ($\frac{\partial \text{Loss}}{\partial \mathbf{\theta}}$) to enable real-time parameter optimization.


Milestone 2 (Week 8): GPU simulation engine executes 1,000 parallel 2.0-second dice rollouts in $12.5\text{ ms}$ on an A17/A18 Pro device.


Phase 3: System Identification & Machine Calibration (Weeks 9–12)

Task 3.1 (Video Dataset Processing): Process 30–50 recorded high-frame-rate video clips of target bubble craps machines through the inverse physics pipeline.
Task 3.2 (Global Parameter Convergence): Execute L-BFGS adjoint optimization over training batches to fit global physical parameters ($\mu_{\text{felt}}, e_{\text{dice}}, \mathbf{J}_{\text{air}}$).
Task 3.3 (Spatial Surface Mapping): Construct a $5 \times 5$ spatial friction grid ($\mu_{x,y}$) across the pop-plate to model non-uniform felt wear and pop impulse tilt.


Milestone 3 (Week 12): Digital twin physics trajectory matches real video ground truth across the first 3 bounce impacts with $< 8%$ cumulative spatial divergence.


Phase 4: Edge Deployment, EV Engine & Field Verification (Weeks 13–16)

Task 4.1 (Monte Carlo Probability Aggregator): Map terminal dice orientations across 2,000 rollouts into terminal outcome probabilities $P_2$ through $P_{12}$.
Task 4.2 (Expected Value & Kelly Engine): Calculate real-time expected values against payout tables for Place/Buy wagers ($2, 3, 4, 5, 6, 8, 9, 10, 11, 12$), flagging bet signals when player edge exceeds $+2.0%\text{ EV}$.
Task 4.3 (Latency & Operational Field Validation): Confirm total pipeline latency remains under $45\text{ ms}$ on mobile hardware during live operation.


Milestone 4 (Week 16): Fully functional iOS application built, tested on TestFlight, displaying positive-EV wagers in real time within target betting lock-out windows.


4. Operational Risk & Mitigation Matrix

Identified RiskImpactMitigation StrategyCylindrical Glass Refraction / GlareHighPre-computed shader calibration grid compensating for acrylic lens distortion; Spatial ROI masking around green floor plate.Tight Bet Lock-out Windows ($< 1.0\text{ s}$)HighExtract 6DoF pose during pre-launch resting phase; trigger parallel simulation on initial $50\text{ ms}$ vibration onset.Contact Discontinuities in Gradient PassMediumImplement CMA-ES (Evolutionary Strategy) fallback if L-BFGS gradient optimization stalls on sharp corner contacts.Video Motion Blur / Low FPS SourcesMediumIntegrate RIFE frame interpolation and optical flow (CoTracker) when bootstrapping models from $30/60\text{ fps}$ YouTube footage.

See less
### Phase 1: High-Speed CV, Pose Extraction & HITL Framework (Weeks 1–4)
* **Task 1.1 (Camera Pipeline):** Lock exposure, white balance, and manual focus at $120\text{ fps}$ via `AVFoundation`. Implement zero-copy `CVPixelBuffer` transfers directly to Metal memory buffers.
* **Task 1.2 (Pop-Plate Homography):** OpenCV circular edge detection (`HoughCircles`) anchors world coordinate origin $(0,0,0)$ on the green felt plate.
* **Task 1.3 (Pose Estimation):** Train quantized YOLOv8-Keypoint model on Apple Neural Engine to identify die corners and face pips, feeding a Perspective-n-Point (PnP) solver for 6DoF pose extraction.
* **Task 1.4 (Human-in-the-Loop Active Learning UI):**
  * Swift/UIKit overlay rendering color-coded 3D bounding cubes over predicted poses.
  * Drag handles for keypoint corner corrections and 3-point pop-plate edge calibration.
  * $120\text{ fps}$ timeline scrubber and auto-accept thresholding ($\text{IoU} \ge 0.85$, PnP residual $\le 1.5\text{ px}$).
* **Task 1.5 (Data Collection):** Capture and annotate 50–100 video clips across varied roll speeds, resting poses, and edge impact occlusions.

> **Milestone 1 (Week 4):** Live camera feed tracks 6DoF dice coordinates and floor origin with $< 10\text{ ms}$ latency; HITL interface reduces clip annotation time to $< 30\text{ seconds}$ per video.

---

### Phase 2: Differentiable Rigid-Body Physics Engine (Weeks 5–8)
* **Task 2.1 (SDF Dome Geometry):** Analytical Signed Distance Fields (SDFs) representing glass cylinder walls, rubber bumper ring, and floor plate for $\mathcal{O}(1)$ collision queries.
* **Task 2.2 (Impulse Contact Solver):** Metal compute shaders executing Sequential Impulse dynamics with surface friction ($\mu$) and coefficient of restitution ($e$) under sub-step numerical integration ($\Delta t = 1.0\text{ ms}$).
* **Task 2.3 (Adjoint Differentiation):** Implement analytical forward/backward gradient passes ($\frac{\partial \text{Loss}}{\partial \mathbf{\theta}}$) to enable real-time parameter optimization.

> **Milestone 2 (Week 8):** GPU simulation engine executes 1,000 parallel 2.0-second dice rollouts in $12.5\text{ ms}$ on an A17/A18 Pro device.

---

### Phase 3: System Identification & Machine Calibration (Weeks 9–12)
* **Task 3.1 (Video Dataset Processing):** Process 30–50 recorded high-frame-rate video clips of target bubble craps machines through the inverse physics pipeline.
* **Task 3.2 (Global Parameter Convergence):** Execute L-BFGS adjoint optimization over training batches to fit global physical parameters ($\mu_{\text{felt}}, e_{\text{dice}}, \mathbf{J}_{\text{air}}$).
* **Task 3.3 (Spatial Surface Mapping):** Construct a $5 \times 5$ spatial friction grid ($\mu_{x,y}$) across the pop-plate to model non-uniform felt wear and pop impulse tilt.

> **Milestone 3 (Week 12):** Digital twin physics trajectory matches real video ground truth across the first 3 bounce impacts with $< 8\%$ cumulative spatial divergence.

---

### Phase 4: Edge Deployment, EV Engine & Field Verification (Weeks 13–16)
* **Task 4.1 (Monte Carlo Probability Aggregator):** Map terminal dice orientations across 2,000 rollouts into terminal outcome probabilities $P_2$ through $P_{12}$.
* **Task 4.2 (Expected Value & Kelly Engine):** Calculate real-time expected values against payout tables for Place/Buy wagers ($2, 3, 4, 5, 6, 8, 9, 10, 11, 12$), flagging bet signals when player edge exceeds $+2.0\%\text{ EV}$.
* **Task 4.3 (Latency & Operational Field Validation):** Confirm total pipeline latency remains under $45\text{ ms}$ on mobile hardware during live operation.

> **Milestone 4 (Week 16):** Fully functional iOS application built, tested on TestFlight, displaying positive-EV wagers in real time within target betting lock-out windows.

---

## 4. Operational Risk & Mitigation Matrix

| Identified Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Cylindrical Glass Refraction / Glare** | High | Pre-computed shader calibration grid compensating for acrylic lens distortion; Spatial ROI masking around green floor plate. |
| **Tight Bet Lock-out Windows ($< 1.0\text{ s}$)** | High | Extract 6DoF pose during pre-launch resting phase; trigger parallel simulation on initial $50\text{ ms}$ vibration onset. |
| **Contact Discontinuities in Gradient Pass** | Medium | Implement CMA-ES (Evolutionary Strategy) fallback if L-BFGS gradient optimization stalls on sharp corner contacts. |
| **Video Motion Blur / Low FPS Sources** | Medium | Integrate RIFE frame interpolation and 