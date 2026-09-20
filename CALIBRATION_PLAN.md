# Multi-Lens Extrinsic Calibration & Distortion Modeling Plan

## 1. Executive Summary & Problem Formulation

Multi-lens film cameras (such as the RETO 3D, Nishika N8000, and Nimslo) capture stereoscopic wiggle-gram triplets ($L_0, L_1, L_2$) simultaneously on a single film strip. While nominal specifications assume a rigid, perfectly collinear 1D array of identical lenses, physical reality introduces two primary sources of geometric distortion:

1. **Chassis Construction & Assembly Imperfections:**
   - **Non-Collinear Centers:** Due to injection molding tolerances and manual chassis assembly, the center lens $L_1$ does not lie strictly on the straight line connecting $L_0$ and $L_2$ (vertical sag $\Delta y_c$, depth offset $\Delta z_c$).
   - **Independent Angular Misalignments:** Each lens exhibits slight independent relative tilt (yaw, pitch, roll $\mathbf{R}_i$).
2. **Optical Lens Aberrations:**
   - Low-cost uncalibrated plastic/acrylic lenses create non-linear radial bowing and tangential decentering.

This document establishes the architecture for **joint extrinsic pose optimization via triplet correspondence cycle consistency** and presents a **comprehensive survey of lens distortion parameterizations** to guide future model selection.

---

## 2. Realistic Chassis Geometry & Extrinsic Parameterization

```
       Ideal (Collinear)               Realistic (Non-Collinear + Tilted)
   L0 ───────── L1 ───────── L2            L0 ───────┐
                                                      │  Δy_c (Sag)
                                                      └─── L1 (Rotated R1) ────── L2 (Rotated R2)
                                                             │
                                                             └── Δz_c (Depth offset)
```

### 2.1. 6-DoF Relative Pose Model
Rather than assuming a 1D scalar translation along the X-axis, the relative transformation between sub-frames is parameterized by full rigid 6-DoF Euclidean transforms $[ \mathbf{R} \mid \mathbf{t} ] \in \mathrm{SE}(3)$:

$$\mathbf{P}_1 = \mathbf{R}_{01} \mathbf{P}_0 + \mathbf{t}_{01}, \quad \mathbf{P}_2 = \mathbf{R}_{12} \mathbf{P}_1 + \mathbf{t}_{12}$$

* **Reference Frame:** $L_0$ is fixed as the origin coordinate system ($[ \mathbf{I} \mid \mathbf{0} ]$).
* **Scale Ambiguity:** The baseline magnitude $\|\mathbf{t}_{01}\|$ is normalized to $1.0$ (or nominal physical baseline $\approx 18.5\text{ mm}$), fixing the gauge freedom.
* **Non-Collinear Translation Vector:**
  $$\mathbf{t}_{01} = \begin{bmatrix} b_{x,01} \\ \Delta y_{01} \\ \Delta z_{01} \end{bmatrix}, \quad \mathbf{t}_{12} = \begin{bmatrix} b_{x,12} \\ \Delta y_{12} \\ \Delta z_{12} \end{bmatrix}$$
  where non-zero $\Delta y$ and $\Delta z$ account for vertical sag and optical depth discrepancy.

### 2.2. Scan Orientation & Peripheral Distortion Coupling
Real-world film scans arrive in **portrait or landscape formats** depending on camera handling during exposure and film lab scanner orientation (0°, 90°, 180°, 270°):

1. **Orientation Invariance (Chassis vs. Image Plane):**
   - The physical baseline and cross-baseline sag axes are fixed to the camera chassis, not raw image $(u, v)$ pixel coordinates.
   - Any cross-baseline compensation must operate in canonical chassis coordinates after orientation normalization (via [`StripOrientation`](file:///home/duty/workspace/leto-split/reto-core/src/geom.rs)).
2. **Peripheral Distortion vs. Sag Confounding:**
   - Multi-lens toy cameras utilize wide-angle optics ($f \approx 22\text{--}30\text{ mm}$) with severe unmodeled peripheral barrel distortion.
   - At image borders ($r \to r_{\max}$), radial distortion displaces points non-linearly in both baseline and cross-baseline directions.
   - **Hazard:** Imposing a tight, static cross-baseline tolerance or assuming a constant directional sag prior at the frame boundaries will falsely reject valid boundary matches.
3. **Adaptive Mitigation Strategy:**
   - **Central-Anchor Initialization:** Compute initial baseline disparity and relative orientation from the central optical region ($r \le 0.5 \cdot r_{\max}$), where distortion is minimal.
   - **Radial Slack Function:** Dynamically scale cross-baseline filtering tolerance toward image borders:
     $$\tau_{\text{cross}}(r) = \tau_0 \cdot \left(1 + \kappa \cdot \left(\frac{r}{r_{\max}}\right)^2\right)$$
   - **Boundary Feature Protection:** Prevent premature outlier pruning before joint pose and distortion optimization.

---

## 3. Extrinsic Optimization via Triplet Cycle Consistency

SuperPoint yields reliable keypoints across wide disparities, but residual mismatches occur around textureless regions, repetitive patterns, and film grain noise. We leverage the closed 3-view graph to filter outliers and refine relative extrinsics.

```
                  ┌───────────────────┐
                  │    Sub-frame 0    │
                  └─────────┬─────────┘
                           / \
                          /   \
        Pair Match M_01  /     \  Pair Match M_02
                        /       \
                       ▼         ▼
        ┌───────────────────┐   ┌───────────────────┐
        │    Sub-frame 1    ├───▶    Sub-frame 2    │
        └───────────────────┘   └───────────────────┘
                  Pair Match M_12
            (Cycle Invariant: M_01 ∘ M_12 ≡ M_02)
```

### 3.1. Pairwise Matching & Triplet Cycle Filter
1. **Compute Pairwise Matches Across All Pairs $\binom{N}{2}$:**
   For $N$ sequential lenses ($L_0, L_1, \dots, L_{N-1}$), compute mutual nearest-neighbor matches across all $\binom{N}{2}$ pairs (for $N=3$: $\mathcal{M}_{01}, \mathcal{M}_{12}, \mathcal{M}_{02}$; for $N=4$: $\mathcal{M}_{01}, \mathcal{M}_{12}, \mathcal{M}_{23}, \mathcal{M}_{02}, \mathcal{M}_{13}, \mathcal{M}_{03}$).
   - Direct observations on wide baselines (e.g. $\mathcal{M}_{02}$ for $N=3$, $\mathcal{M}_{03}$ for $N=4$) provide **$2\times\text{--}3\times$ higher depth resolution** ($\sigma_Z \propto \frac{Z^2}{B \cdot f}$) than single-step adjacent pairs.

2. **Cycle Closure & Transitive Consistency:**
   For each triplet combination $(i, j, k)$ with $0 \le i < j < k < N$:
   Form candidate tracks where $(\mathbf{p}_i, \mathbf{p}_j) \in \mathcal{M}_{ij}$ and $(\mathbf{p}_j, \mathbf{p}_k) \in \mathcal{M}_{jk}$.
   Validate consistency against direct long-baseline observation $(\mathbf{p}_i, \mathbf{p}_k^{(ik)}) \in \mathcal{M}_{ik}$:
   $$\epsilon_{\text{cycle}} = \|\mathbf{p}_k^{(jk)} - \mathbf{p}_k^{(ik)}\|_2 \le \epsilon_{\text{threshold}} \quad (\le 1.5\text{ px})$$
   Any candidate with conflicting transitive links is pruned.

### 3.2. Generalization to $N \ge 3$ 1D Adjacent Array & Group Composition
For a 1D sequence of $N$ rigidly mounted lenses, relative transformations must satisfy group composition across all $\binom{N}{3}$ triplet combinations:

$$\mathbf{T}_{ik} \equiv \mathbf{T}_{jk} \cdot \mathbf{T}_{ij} \iff \begin{bmatrix} \mathbf{R}_{ik} & \mathbf{t}_{ik} \\ \mathbf{0}^\top & 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R}_{jk} & \mathbf{t}_{jk} \\ \mathbf{0}^\top & 1 \end{bmatrix} \begin{bmatrix} \mathbf{R}_{ij} & \mathbf{t}_{ij} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

* **Essential Matrix / Epipolar Composition:**
  $$\mathbf{E}_{ik} \sim [\mathbf{t}_{ik}]_\times \mathbf{R}_{ik} = [\mathbf{R}_{jk} \mathbf{t}_{ij} + \mathbf{t}_{jk}]_\times (\mathbf{R}_{jk} \mathbf{R}_{ij})$$
* **Linear Translation Additivity (Small Rotation Limit $\mathbf{R} \approx \mathbf{I}$):**
  $$\mathbf{t}_{ik} = \mathbf{t}_{ij} + \mathbf{t}_{jk} \implies \text{disp}_{ik} \equiv \text{disp}_{ij} + \text{disp}_{jk}$$

### 3.3. Hierarchical Cascaded Binary Tree Reduction ($O(N \log N)$ Parallel Verification)
Rather than evaluating all $\binom{N}{3}$ combinations naively in serial, the 1D sequential layout maps naturally to a **Binary Reduction Tree / Segment Tree** of dyadic intervals:

```
                            [Level 2: Global Root Chord]
                                     (0, 3)
                           E_03 ≡ E_23 · E_12 · E_01
                                   /        \
                                  /          \
                [Level 1: Stride 2]          [Level 1: Stride 2]
                      (0, 2)                       (1, 3)
                   E_02 ≡ E_12 · E_01           E_13 ≡ E_23 · E_12
                       /     \                     /     \
                      /       \                   /       \
              [Level 0: Leaves (Stride 1 - Adjacent Pairs)]
                   (0, 1)          (1, 2)          (2, 3)
                    E_01            E_12            E_23
```

#### A. Tree Invariants & Parallel Properties
1. **Parallel Prefix Composition:**
   - **Leaves (Level 0, Stride 1):** Measure adjacent pairs $\mathcal{M}_{01}, \mathcal{M}_{12}, \mathcal{M}_{23}, \dots, \mathcal{M}_{N-2, N-1}$.
   - **Internal Nodes (Level $h$, Stride $2^h$):** Compose pairwise transforms $\mathbf{T}_{i, i+2^h} = \mathbf{T}_{i+2^{h-1}, i+2^h} \cdot \mathbf{T}_{i, i+2^{h-1}}$.
   - **Root Span $(0, N-1)$:** Enforces total global chord consistency $\mathbf{T}_{0, N-1} \equiv \prod_{i=0}^{N-2} \mathbf{T}_{i, i+1}$.
2. **Computational Complexity:**
   - **Total Comparisons:** $O(N \log N)$ structured dyadic comparisons (vs. $O(N^3)$ exhaustive triplets).
   - **Parallel Critical Path:** $O(\log N)$ tree depth latency executed concurrently across Rayon threads.
3. **Hierarchical Outlier Pruning:**
   If an adjacent leaf pair $(i, i+1)$ exhibits high epipolar residual, all transitive multi-view tracks passing through node $(i, i+1)$ are pruned in $O(1)$ without propagating errors to higher tree levels.

### 3.4. Trifocal Tensor & Multi-View Epipolar Formulation
For every validated triplet correspondence $(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2)$, the points must simultaneously satisfy:
1. **Pairwise Epipolar Constraints:**
   $$\mathbf{x}_1^\top \mathbf{F}_{01} \mathbf{x}_0 = 0, \quad \mathbf{x}_2^\top \mathbf{F}_{12} \mathbf{x}_1 = 0, \quad \mathbf{x}_2^\top \mathbf{F}_{02} \mathbf{x}_0 = 0$$
2. **Trifocal Point Transfer Invariant:**
   Given $\mathbf{x}_0$ and $\mathbf{x}_1$, the predicted position in frame 2 is uniquely constrained by the trifocal tensor $\mathcal{T}_i^{jk}$:
   $$x_2^k = x_0^i l_1^j \mathcal{T}_i^{jk}$$

### 3.5. Joint Levenberg-Marquardt Bundle Adjustment
Given $N$ robust multi-view tracks, we optimize the camera extrinsics $\mathbf{\Theta} = \{[\mathbf{R}_{0i} \mid \mathbf{t}_{0i}]\}_{i=1}^{N-1}$ and 3D triangulated points $\mathbf{X}_j$ by minimizing the robust Huber reprojection loss:

$$\mathcal{L}(\mathbf{\Theta}, \{\mathbf{X}_j\}) = \sum_{j=1}^M \sum_{v=0}^{N-1} \rho_{\text{Huber}}\left( \left\| \mathbf{x}_v^{(j)} - \pi\left(\mathbf{K}_v, \mathbf{R}_v, \mathbf{t}_v, \mathbf{X}_j\right) \right\|^2 \right) + \sum_{v=1}^{N-1} \lambda_{\text{prior}} \|\mathbf{R}_v - \mathbf{I}\|_F^2$$

### 3.6. Two-Stage Robust Filtering: Physics-First Optimization & Boundary Slack
SuperPoint correspondences provide high physical stability across the image plane. To prevent unmodeled wide-angle optical distortion from falsely corrupting the physical extrinsics or discarding valuable boundary matches, we execute a **Two-Stage Physics-Driven Pipeline**:

```
[Stage A: Central-Anchor Extrinsic Fitting]
SuperPoint Inliers in Central Zone (r ≤ 0.5·r_max) ──▶ Optimize Rigid Extrinsics [R_ij | t_ij]
                                                                  │
                                                                  ▼
[Stage B: Radial-Adaptive Outlier Filtering]
Evaluate Reprojection/Epipolar Residuals r_i against Radial Slack Tolerance τ(r)
      │
      ├─▶ Center Region (r ≈ 0): Tight physical tolerance (e.g. 1.0 px)
      └─▶ Boundary Region (r → r_max): Loosened tolerance τ(r) to accommodate uncalibrated barrel distortion
```

1. **Stage A: Central-Anchor Extrinsic Optimization:**
   - The central optical zone ($r \le 0.5 \cdot r_{\max}$) has minimal distortion ($\le 0.2\text{ px}$).
   - Keypoint correspondences in this region are strictly governed by rigid pinhole geometry ($[\mathbf{R}_{ij} \mid \mathbf{t}_{ij}]$).
### 3.7. Neural Confidence Ranking & Parallax-Aware Subpixel Gating

#### A. Neural Descriptor Matching Confidence as Primary Metric
Neural matchers (SuperPoint / LightGlue) compute deep semantic feature similarity invariant to substantial lighting, blur, and perspective distortion.
* **Confidence-Sorted Candidate Sets:** Match sets are ordered by descriptor confidence $C_{\text{match}} \in [0.0, 1.0]$. The top-tier highest-confidence matches anchor the initial rigid extrinsic estimation.
* **Outlier Filtering Priority:** Geometric verification filters candidate matches starting from the highest neural confidence down to the tail.

#### B. The Parallax Vulnerability in Monocular Subpixel Refinement
Independent monocular subpixel refinement (e.g. intensity gradient tensor / cornerSubPix) assumes that the local image patch is an identical rigid translation in both views. Under real stereo parallax:
1. **Perspective Foreshortening & Shear:** Slanted surfaces change aspect ratio between views $L_0$ and $L_1$.
2. **Asymmetric Corner Drift:** An independent gradient refinement on $L_0$ and $L_1$ can snap to slightly different sub-structures, introducing artificial disparity jitter ($0.2\text{--}0.8\text{ px}$).
3. **Aperture Sliding on Edges:** Along 1D dominant gradients, refinement slides freely along the edge direction.

```
       [Raw SuperPoint / LightGlue Matches (Confidence C_match)]
                                   │
                                   ▼
                   [Gradient Tensor Condition Check]
                   λ_min / λ_max ≥ γ_cond (True 2D Corner?)
                                  / \
                           YES   /   \   NO (Aperture / Flat)
                                /     \
                               ▼       ▼
       [Iterative Subpixel Refinement]  [Keep Robust Neural Coord]
                               │
               Drift ||Δp|| ≤ 1.0 px?
                               / \
                        YES   /   \   NO (Parallax Over-Drift)
                             /     \
                            ▼       ▼
           [Accept Subpixel Coord]  [Fallback to Neural Coord]
```

#### C. Robust Quality Gating Strategy
To prevent parallax-induced subpixel degradation while retaining continuous coordinate precision:
1. **Tensor Well-Conditioning Gate:**
   Compute eigenvalues $\lambda_{\min}, \lambda_{\max}$ of structure tensor $\mathbf{G}$.
   If $\lambda_{\min} < \tau_{\min}$ or $\lambda_{\min}/\lambda_{\max} < 0.15$ (indicating a 1D edge vulnerable to sliding under parallax), **skip refinement and keep the robust neural coordinate**.
2. **Configurable Parallax Drift Clamping (`max_subpixel_drift_px`):**
   If the refined subpixel displacement exceeds the configurable threshold ($\|\Delta \mathbf{p}\| > \delta_{\max}$, default: $2.5\text{ px}$, tunable across $1.0\text{--}3.5\text{ px}$ for experimental calibration), the subpixel shift is treated as parallax-corrupted and **reverts to the neural coordinate**.
3. **Joint Triplet Confidence Formulation:**
   $$\text{Conf}(\mathcal{T}) = \left( C_{01} \cdot C_{12} \cdot C_{02} \right)^{1/3} \cdot \exp\left(-\frac{\|\Delta \mathbf{p}_{0}\| + \|\Delta \mathbf{p}_{1}\| + \|\Delta \mathbf{p}_{2}\|}{3 \cdot \sigma_{\text{drift}}}\right)$$
   Penalizes high-drift triplets while rewarding strong neural descriptor consensus.

---

## 4. Lens Distortion Modeling Survey

A comprehensive survey of optical distortion parameterizations, evaluating their mathematical properties, stability, and suitability for multi-lens toy camera calibration.

### 4.1. Model Summary & Mathematical Formulations

#### 1. Fitzgibbon Division Model
* **Formula:**
  $$\mathbf{x}_u = \frac{\mathbf{x}_d - \mathbf{c}}{1 + \lambda_1 r^2 + \lambda_2 r^4} + \mathbf{c}, \quad \text{where } r = \|\mathbf{x}_d - \mathbf{c}\|$$
* **Pros:** Single-step direct undistortion without iterative root-finding. Enables low-degree polynomial systems in minimal RANSAC solvers.
* **Cons:** Cannot model tangential or asymmetric decentering distortion.
* **Best Used For:** Fast initial pose estimation inside RANSAC loops.

#### 2. Brown-Conrady Polynomial Model (OpenCV Standard)
* **Formula:**
  $$x_d = x_u (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \left[ 2 p_1 x_u y_u + p_2(r^2 + 2 x_u^2) \right]$$
  $$y_d = y_u (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \left[ p_1(r^2 + 2 y_u^2) + 2 p_2 x_u y_u \right]$$
* **Pros:** Standardized industry baseline (COLMAP, OpenCV). Decouples symmetric radial bowing ($k_1, k_2$) and lens element decentering/tilt ($p_1, p_2$).
* **Cons:** Forward mapping is distorted-from-undistorted; requires iterative Newton-Raphson to undistort. High-order terms ($k_3$) suffer from edge divergence (Runge phenomenon).
* **Best Used For:** Standard bundle adjustment self-calibration when distortion is moderate.

#### 3. Rational Polynomial Model (OpenCV 8-Parameter)
* **Formula:**
  $$\mathbf{x}_d = \mathbf{x}_u \cdot \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + \text{tangential}(p_1, p_2)$$
* **Pros:** Far higher expressive capacity for wide-angle and strongly curved lenses where simple polynomial Taylor series diverge.
* **Cons:** High risk of numerical singularity (denominator $\to 0$) if overparameterized without dense multi-view constraints.
* **Best Used For:** Ultra-wide angle lenses or severe non-linear barrel distortion.

#### 4. Kannala-Brandt Model (Equidistant Fisheye)
* **Formula:**
  $$\theta = \arctan(r_u), \quad \theta_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8), \quad \mathbf{x}_d = \frac{\theta_d}{r_u} \mathbf{x}_u$$
* **Pros:** Mathematically stable for field-of-view $\ge 100^\circ$. Avoids planar projection infinity singularities.
* **Cons:** Overkill for standard focal length toy cameras ($30\text{--}35\text{ mm}$ equivalent).
* **Best Used For:** Fisheye lenses and extreme wide-angle action cameras.

#### 5. Non-Parametric B-Spline Surfaces (`mrcal` / Generic Camera Model)
* **Formula:**
  $$\mathbf{x}_d = \mathbf{x}_u + \sum_{i=0}^M \sum_{j=0}^N \mathbf{c}_{ij} B_{i,k}(x_u) B_{j,k}(y_u)$$
* **Pros:** Local basis support prevents global runaway oscillations. Can capture completely arbitrary, asymmetric plastic mold defects and physical chassis warpage.
* **Cons:** High parameter count ($M \times N$ control points). Cannot be calibrated from a single 3-image shot; requires multi-strip offline calibration.
* **Best Used For:** Factory/offline pre-calibration baked into a static 2D displacement LUT.

#### 6. Neural Residual Coordinate Fields
* **Formula:**
  $$\mathbf{x}_d = f_{\text{geom}}(\mathbf{x}_u; \mathbf{\Theta}) + \mathrm{MLP}_{\mathbf{\Phi}}(\gamma(\mathbf{x}_u))$$
* **Pros:** Continuous implicit representation. Learns complex micro-aberrations and chromatic variations across sensor areas without manual grid resolution tuning.
* **Cons:** Impossible to fit from scratch on a single 3-frame triplet without severe overfitting. Requires deep learning framework (PyTorch/Candle/ONNX) or pre-baking.
* **Best Used For:** Research pipelines or generating ultra-dense baked LUTs offline.

---

### 4.2. Comparative Evaluation Matrix

| Parameterization Model | Parameter Count | Invertible Closed-Form? | Edge Stability (Runge) | Asymmetric Defect Handling | Per-Shot (3 Images) Feasible? | Runtime Latency in Rust |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fitzgibbon Division** | 1 – 2 | Yes (Direct) | High | Poor | **Yes** (Ideal for RANSAC) | $< 0.1\ \mu\text{s}$ |
| **Brown-Conrady** | 4 – 5 | No (Newton-Raphson) | Medium | Good (Decentering) | **Yes** (LM Refinement) | $\approx 0.5\ \mu\text{s}$ |
| **Rational Polynomial** | 8 | No (Iterative) | Medium-Low | Good | No (Overparameterized) | $\approx 1.2\ \mu\text{s}$ |
| **Kannala-Brandt** | 4 – 8 | No (Iterative) | High | Poor | Yes (If fisheye) | $\approx 0.8\ \mu\text{s}$ |
| **B-Spline (`mrcal`)** | $32\text{--}128$ | Via Grid Interpolation | Very High | Excellent | **Offline Only $\to$ Baked LUT** | $< 0.05\ \mu\text{s}$ (LUT) |
| **Neural Residual Field** | $1\text{k}\text{--}10\text{k}$ | Via Grid Inversion | High (Regularized) | Outstanding | **Offline Only $\to$ Baked LUT** | $< 0.05\ \mu\text{s}$ (LUT) |

---

## 5. Recommended Architecture & Implementation Plan

```
[Phase 1: Feature Match & Cycle Closure]
SuperPoint Pairwise (0-1, 1-2, 0-2) ──▶ Triplet Graph Consensus ──▶ Inlier Tracks

[Phase 2: Extrinsic 6-DoF Optimization]
Inlier Tracks ──▶ Levenberg-Marquardt SE(3) BA ──▶ Refined [R_01|t_01], [R_12|t_12]

[Phase 3: Distortion Decoupling]
Option A (Dynamic Online): Division / Brown-Conrady bounded LM self-calibration
Option B (Static Chassis Prior): Offline B-Spline / Neural field baked into 2D LUT
```

### 5.1. Module Implementation Plan for `reto-core`

1. **`reto_core::geom::triplet` (New submodule in `geom.rs` or `alignment.rs`):**
   - Implement `TripletTrack` struct storing $(\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_2)$ and photometric response.
   - Implement `filter_triplet_cycle_consistency(...)` with configurable pixel threshold $\epsilon_{\text{cycle}}$.
2. **`reto_core::calibration::extrinsic`:**
   - Define `ChassisExtrinsics` struct supporting independent 6-DoF relative poses $[\mathbf{R}_{01} \mid \mathbf{t}_{01}]$ and $[\mathbf{R}_{12} \mid \mathbf{t}_{12}]$.
   - Implement non-linear Levenberg-Marquardt solver using analytical Jacobians for trifocal / epipolar residuals.
3. **`reto_core::calibration::distortion`:**
   - Implement `DistortionModel` trait with pluggable implementations:
     - `DivisionModel` (fast algebraic evaluation).
     - `BrownConradyModel` (iterative solver).
     - `ChassisLutModel` (pre-baked 2D float displacement grid for RETO 3D / Nimslo).
4. **Diagnostic Taps:**
   - Expose intermediate reprojection error heatmaps and non-collinear chassis offsets ($\Delta y, \Delta z$) via CLI `--debug` tap.
