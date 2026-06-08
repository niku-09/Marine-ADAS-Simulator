# Marine ADAS Simulator — EKF Sensor Fusion & Autonomous Collision Avoidance

An interactive Python simulation of Advanced Driver Assistance Systems (ADAS) applied to marine vessels. Built to demonstrate sensor fusion, state estimation under uncertainty, real-time SLAM, and reactive collision avoidance in a maritime environment.

---

## What it simulates

The simulation models a vessel navigating a 30×30 m environment containing static obstacles (rocks, buoys, wrecks, anchored vessels) and dynamic AIS vessels. The vessel is equipped with three simulated sensors whose outputs are fused in real time:

- **GPS** — absolute position fix at ~1 Hz, with 8% random dropout and occasional multipath spikes
- **IMU** — high-frequency velocity and angular rate, with slowly growing bias (drift) over time
- **LiDAR** — 360° scan at 10 m range, Gaussian measurement noise

The EKF runs every frame, fusing all three sensors to produce an optimal pose estimate. The gap between ground truth and EKF estimate is live-plotted alongside a 3-sigma covariance ellipse that visibly grows during GPS dropout and contracts on re-acquisition.

---

## Technical approach

### Extended Kalman Filter (EKF)
State vector: `μ = [x, y, θ, v]`

**Predict step** — nonlinear motion model linearised via Jacobian `F = ∂f/∂μ`:
```
μ⁻ = f(μ, u)           # unicycle kinematics
P⁻ = F P Fᵀ + Q        # covariance propagation
```

**Update step** — sequential updates per sensor:
```
K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹    # Kalman gain
μ = μ⁻ + K(z − Hμ⁻)            # state correction
P = (I − KH) P⁻                 # covariance update
```

GPS and IMU use separate `H` matrices and `R` noise covariances, updated independently each frame. The 3-sigma uncertainty ellipse is derived from the eigendecomposition of `P[:2, :2]`.

### LiDAR (vectorised ray-marching)
All 360 beams are cast simultaneously as NumPy arrays. Each step advances all beams by 0.15 m and checks boundary conditions, the static occupancy grid, and dynamic vessel radii in a single vectorised pass. Complexity is O(beams × steps) but fully NumPy-vectorised — no Python-level beam loop.

### DWA Obstacle Avoidance (Dynamic Window Approach — simplified)
The forward LiDAR sector (±28 beams from heading) is inspected each frame. If the minimum range in that sector falls below a threat threshold:
- Clearance is compared port vs starboard (±85-beam windows)
- Steering correction is applied toward the clearer side, scaled by threat proximity
- Speed is reduced proportionally: `v_eff = v_cmd × clip(min_range / threshold, 0.15, 1.0)`

### SLAM (incremental occupancy map)
LiDAR endpoints are written into a 30×30 occupancy grid each frame (subsampled every 3rd beam for performance). Cells accumulate probability mass capped at 1.0. The map is blurred with a Gaussian kernel and displayed live in the SLAM panel. Note: only endpoints are marked occupied — free-space ray-clearing is not implemented.

---

## Repository contents

```
marine_adas_ekf.py       Main simulation — all logic and rendering (844 lines)
HANDOFF_marine_adas.md   Development context and architecture notes
README.md                This file
```

---

## Dependencies

| Package | Version | Notes |
|---|---|---|
| `numpy` | ≥ 1.21 | Required |
| `matplotlib` | ≥ 3.5 | Required |
| `scipy` | ≥ 1.7 | Optional — used for SLAM map Gaussian blur; degrades gracefully without it |

Install:
```bash
pip install numpy matplotlib scipy
```

---

## How to run

```bash
python marine_adas_ekf.py
```

Controls:

| Key | Action |
|---|---|
| `W` / `↑` | Throttle |
| `S` / `↓` | Brake |
| `A` / `←` | Turn port |
| `D` / `→` | Turn starboard |

---

## What you will see

**Main view (left panel)**
- Animated ocean texture with vessel wake trail
- 360° LiDAR beams colour-coded by range: red (< 35%) → orange (35–65%) → green (> 65% of max range)
- Actual trajectory (blue) vs EKF estimated trajectory (red dashed)
- Cyan dotted ellipse — EKF 3-sigma position uncertainty; grows during GPS dropout
- Yellow `+` markers — GPS fix positions as they arrive
- Yellow dashed rings — obstacle detection highlights, activated when LiDAR hits
- Rotating green radar sweep wedge
- HUD: speed (m/s and knots), position, heading, DWA alert flag

**Top-right — SLAM occupancy map**
Real-time map built from LiDAR returns. Hot colormap — brighter = more hits. Vessel position and trajectory overlaid.

**Mid-right — Sensor fusion status**
Live quality bars and status labels for GPS (with lock/search state), IMU (drift error), LiDAR (hit fraction), and EKF (position error vs ground truth).

**Bottom-right — EKF state vector**
Animated bars for `x`, `y`, `θ`, `v` normalised to map/max values. Position error readout colour-coded green / yellow / red by threshold.

---

## Known limitations

- SLAM uses endpoint-only marking — free space between vessel and hit point is not cleared (no inverse sensor model)
- DWA is a single-step reactive planner, not a full trajectory-rollout sampler
- Dynamic AIS vessels are not written into the static occupancy grid; LiDAR detects them via a radius check in the ray-marcher, but their positions are not tracked in the SLAM map
- IMU update uses a linear `H` matrix (valid approximation for small angular rates)
- No loop closure in SLAM

---

## Author

Nikhil Mohan  
B.Tech Mechatronics & Automation Engineering, VIT Chennai  
[niku-09.github.io](https://niku-09.github.io) · [LinkedIn](https://linkedin.com/in/nikhil-mohan-nm0909)
