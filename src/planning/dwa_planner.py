"""
src/planning/dwa_planner.py

Dynamic Window Approach (DWA) local planner with optional viability cost
injection via the trained continuous-angle U-Net model.

------------------------------------------------------------------------------
KEY DESIGN — WHY THIS EXISTS
------------------------------------------------------------------------------

DWA evaluates N_v × N_ω candidate (speed, angular-velocity) pairs every
control cycle and picks the lowest-cost trajectory.  Each trajectory has a
terminal heading θ_T.  Asking "can the robot escape heading θ_T from its
terminal position?" is exactly the query our continuous-angle model answers.

The challenge: we need this answer for all N_v × N_ω trajectories in one
control cycle (~50 ms budget).

  Without NN:  Oracle BFS per heading → ~189 ms × 16 bins = 3,024 ms.
               Real-time control is *impossible*.

  With NN:     All 16 heading bins batched into ONE GPU forward pass.
               → ~14 ms total.  DWA runs at > 50 Hz *after* precomputation.

Implementation strategy
-----------------------
1. ``precompute_viability(occ)`` — called ONCE per map (or whenever the map
   changes).  Builds a (N_bins, H, W) viability tensor in one GPU batch call
   and caches it as a dict: {angle_rad → (H, W) float32 array}.

2. ``plan(state, goal, occ, dist_tf)`` — called every control cycle.
   Simulates all (v, ω) trajectories on CPU (pure numpy, ~1 ms).  For each
   trajectory, looks up viability at the terminal pose from the cache in O(1).

Setting ``config.w_viability = 0.0`` produces a Vanilla DWA baseline with
zero NN overhead — identical to a standard DWA implementation.

------------------------------------------------------------------------------
COORDINATE CONVENTION
------------------------------------------------------------------------------
  • (x, y) are pixel coordinates: x increases East, y increases South.
  • θ (theta) is a heading in radians, math convention:
      0       → East   (positive-x direction)
      π/2     → North  (negative-y in image coords)
      π / -π  → West
      -π/2    → South  (positive-y in image coords)
  • Motion model:
      x_new = x + v · cos(θ) · dt
      y_new = y − v · sin(θ) · dt          ← minus because image y is flipped
      θ_new = θ + ω · dt
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State type
# ---------------------------------------------------------------------------

class DWAState(NamedTuple):
    """Immutable robot state in pixel coordinates."""
    x: float       # horizontal pixel position
    y: float       # vertical pixel position (image coords, y-down)
    theta: float   # heading in radians (math convention: 0=East, π/2=North)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DWAConfig:
    """
    Hyperparameters for the DWA planner.

    The single most important parameter for the comparison experiment is
    ``w_viability``:
      - 0.0  →  Vanilla DWA (no NN used after precomputation)
      - 4.0  →  DWA + Viability (recommended for trap avoidance)
    """

    # ---- kinematic limits ------------------------------------------------
    max_speed: float = 8.0      # pixels / step — max forward speed
    min_speed: float = 1.5      # pixels / step — robot always moves forward
    max_omega: float = 0.45     # rad / step    — max turn rate (~26 ° / step)

    # ---- trajectory simulation -------------------------------------------
    dt: float = 1.0             # step duration (abstract time unit)
    predict_steps: int = 15     # simulation horizon (steps)

    # ---- sampling grid ---------------------------------------------------
    n_v: int = 10               # number of speed samples
    n_omega: int = 21           # number of ω samples (odd → includes ω=0)

    # ---- viability precomputation ----------------------------------------
    n_heading_bins: int = 16    # heading angles for one batched GPU call
                                # 16 bins → 22.5 ° resolution

    # ---- cost weights ----------------------------------------------------
    w_goal_heading: float = 1.0  # penalise angular error toward goal
    w_goal_dist: float = 0.6     # reward closing distance to goal
    w_clearance: float = 2.0     # penalise proximity to obstacles
    w_speed: float = 1.5         # reward higher speed (MISSING in original — this was the bug)
                                 # without this the robot picks min_speed to reduce viability
                                 # cost and barely moves, so stuck detection fires on BOTH planners
    w_viability: float = 4.0     # penalise low viability at terminal pose
                                 # SET TO 0.0 FOR VANILLA DWA
                                 # effective range: [0, w_viability] since c_viability ∈ [0,1]

    # ---- operational thresholds ------------------------------------------
    goal_radius: float = 35.0   # pixels — robot is 30px; 35px gives 5px margin
    min_clearance: float = 3.0  # pixels — below this → trajectory invalid


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class DWAPlanner:
    """
    DWA local planner with optional viability cost via a trained U-Net.

    Typical usage
    -------------
    ::

        cfg_vanilla = DWAConfig(w_viability=0.0)
        cfg_via     = DWAConfig(w_viability=4.0)

        planner_vanilla  = DWAPlanner(model, "continuous_angle", 30, 20, 512, device, cfg_vanilla)
        planner_viability = DWAPlanner(model, "continuous_angle", 30, 20, 512, device, cfg_via)

        # Once per map (or when map changes):
        t_ms = planner_viability.precompute_viability(occ)  # ~14 ms GPU batch

        # Every control cycle:
        v, omega, info = planner_viability.plan(state, goal_xy, occ, dist_tf)
        state = DWAPlanner.apply_motion(state, v, omega)
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module],
        oracle_type: str,
        robot_L: int,
        robot_W: int,
        resolution: int,
        device: str,
        config: Optional[DWAConfig] = None,
    ) -> None:
        """
        Args:
            model:       Trained viability U-Net (continuous_angle or basic).
                         Pass None to disable viability cost entirely.
            oracle_type: "continuous_angle" (5-ch in, 1-ch out) or
                         "basic" (3-ch in, 4-ch out).
            robot_L:     Robot length in pixels (used to normalise channels).
            robot_W:     Robot width  in pixels.
            resolution:  Map resolution (pixels, square map assumed).
            device:      "cuda" or "cpu".
            config:      DWAConfig instance (or None → use defaults).
        """
        self.model = model
        self.oracle_type = oracle_type
        self.robot_L = robot_L
        self.robot_W = robot_W
        self.resolution = resolution
        self.device = device
        self.config = config or DWAConfig()

        # Precomputed viability cache: angle_rad → (H, W) float32 ndarray
        self._via_cache: Dict[float, np.ndarray] = {}
        self._cache_valid: bool = False

        # Sorted list of precomputed heading angles [0, 2π)
        cfg = self.config
        self._bins: List[float] = [
            2.0 * np.pi * i / cfg.n_heading_bins
            for i in range(cfg.n_heading_bins)
        ]

    # ------------------------------------------------------------------
    # Viability precomputation  (call once per map / environment change)
    # ------------------------------------------------------------------

    def precompute_viability(self, occ: np.ndarray) -> float:
        """
        Build viability maps for all heading bins via ONE batched GPU call.

        This is the core efficiency claim: N_bins heading queries cost the
        same as a single forward pass (~14 ms for N=16 on 512×512).

        Vs Oracle: 16 × ~189 ms = 3,024 ms → real-time DWA is impossible.

        Args:
            occ: (H, W) uint8 occupancy grid (1=free, 0=obstacle).

        Returns:
            Elapsed inference time in milliseconds.
        """
        self._cache_valid = False
        self._via_cache.clear()

        # Vanilla DWA — no NN needed
        if self.model is None or self.config.w_viability == 0.0:
            self._cache_valid = True
            return 0.0

        H, W = occ.shape
        bins = self._bins
        N = len(bins)

        # Build batch input: (N, 5, H, W) for continuous_angle
        # or (1, 3, H, W) for basic (we broadcast across bins after)
        if self.oracle_type == "continuous_angle":
            batch = np.zeros((N, 5, H, W), dtype=np.float32)
            batch[:, 0] = occ.astype(np.float32)
            batch[:, 1] = float(self.robot_L) / float(self.resolution)
            batch[:, 2] = float(self.robot_W) / float(self.resolution)
            for i, angle in enumerate(bins):
                batch[i, 3] = float(np.sin(angle))
                batch[i, 4] = float(np.cos(angle))
        else:
            # basic model: 3-channel, 4-channel output (N/S/E/W)
            # Run once, expand to all bins using min across directions
            batch = np.zeros((1, 3, H, W), dtype=np.float32)
            batch[0, 0] = occ.astype(np.float32)
            batch[0, 1] = float(self.robot_L) / float(self.resolution)
            batch[0, 2] = float(self.robot_W) / float(self.resolution)

        x_tensor = torch.from_numpy(batch).to(self.device)

        self.model.eval()
        if self.device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if self.oracle_type == "continuous_angle":
                # One batched forward pass → (N, 1, H, W)
                raw = torch.sigmoid(self.model(x_tensor))
                maps = raw[:, 0].cpu().numpy()  # (N, H, W) float32
            else:
                # basic model → (1, 4, H, W); use per-pixel min as conservative
                raw = torch.sigmoid(self.model(x_tensor))  # (1, 4, H, W)
                min_map = raw[0].min(dim=0).values.cpu().numpy()  # (H, W)
                maps = np.stack([min_map] * N, axis=0)             # (N, H, W)

        if self.device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._via_cache = {bins[i]: maps[i] for i in range(N)}
        self._cache_valid = True

        logger.debug(
            "precompute_viability: %d bins, %.1f ms, oracle_type=%s",
            N, elapsed_ms, self.oracle_type,
        )
        return elapsed_ms

    def _lookup_viability(self, x: float, y: float, theta: float) -> float:
        """
        O(1) viability lookup at pixel (x, y) for heading theta.

        Rounds theta to the nearest precomputed heading bin.

        Args:
            x:     Pixel x coordinate.
            y:     Pixel y coordinate.
            theta: Heading in radians (math convention).

        Returns:
            Viability probability in [0, 1].  Returns 1.0 (fully viable)
            if no cache is available (e.g. w_viability=0 vanilla mode).
        """
        if not self._cache_valid or not self._via_cache:
            return 1.0

        # Normalise theta to [0, 2π)
        theta_norm = theta % (2.0 * np.pi)

        # Find nearest bin (circular distance)
        best_bin = min(
            self._bins,
            key=lambda b: min(
                abs(theta_norm - b),
                2.0 * np.pi - abs(theta_norm - b),
            ),
        )
        via_map = self._via_cache[best_bin]
        H, W = via_map.shape
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        return float(via_map[yi, xi])

    # ------------------------------------------------------------------
    # Planning  (call every control cycle)
    # ------------------------------------------------------------------

    def plan(
        self,
        state: DWAState,
        goal: Tuple[float, float],
        occ: np.ndarray,
        dist_transform: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Dict]:
        """
        Select the best (v, ω) for the current robot state.

        The method evaluates all ``n_v × n_ω`` candidate trajectories and
        returns the one with the lowest composite cost.

        Args:
            state:          Current robot state.
            goal:           Goal position (x, y) in pixel coordinates.
            occ:            (H, W) occupancy grid.
            dist_transform: Precomputed distance transform of ``occ``.
                            If None, computed here (adds ~5 ms for 512×512).

        Returns:
            Tuple ``(best_v, best_omega, info_dict)`` where ``info_dict``
            contains debugging fields: best_cost, trajectory, viability_at_end.
        """
        cfg = self.config
        H, W = occ.shape
        goal_x, goal_y = float(goal[0]), float(goal[1])
        map_diag = float(np.hypot(H, W))

        if dist_transform is None:
            dist_transform = distance_transform_edt(occ).astype(np.float32)

        # Sample (v, ω) grid
        v_samples = np.linspace(cfg.min_speed, cfg.max_speed, cfg.n_v)
        omega_samples = np.linspace(-cfg.max_omega, cfg.max_omega, cfg.n_omega)

        best_cost: float = np.inf
        best_v: float = cfg.min_speed
        best_omega: float = 0.0
        best_traj: List[Tuple[float, float, float]] = [(state.x, state.y, state.theta)]
        best_via: float = 1.0

        for v in v_samples:
            for omega in omega_samples:
                traj, valid, min_clear = self._simulate(
                    state.x, state.y, state.theta,
                    v, omega, occ, dist_transform,
                )
                if not valid or not traj:
                    continue

                tx, ty, ttheta = traj[-1]

                # ---- cost components ----------------------------------------

                # 1. Goal heading: angular error between terminal heading and
                #    direction from terminal position toward goal.
                #    arctan2 sign: image coords → negate dy component.
                desired = np.arctan2(-(goal_y - ty), goal_x - tx)
                heading_err = abs(
                    np.arctan2(
                        np.sin(ttheta - desired),
                        np.cos(ttheta - desired),
                    )
                )
                c_goal_heading = heading_err / np.pi  # normalised [0, 1]

                # 2. Goal distance: Euclidean distance from terminal pose.
                c_goal_dist = np.hypot(tx - goal_x, ty - goal_y) / map_diag

                # 3. Obstacle clearance: penalise proximity (monotone, [0, ∞)).
                c_clearance = 1.0 / (1.0 + min_clear)

                # 4. Speed reward: normalised forward speed [0, 1].
                #    Classic DWA always includes this term to prevent the robot
                #    from stopping.  Without it, viability cost causes the robot
                #    to pick min_speed trajectories (barely moving) rather than
                #    committing to the correct fast path.
                c_speed = v / cfg.max_speed  # [0, 1], higher = faster

                # 5. Viability cost.
                #    The model output is nearly binary (trapped≈0.000, clear≈1.000).
                #    Linear cost (1 - via) is optimal: difference between trapped
                #    and clear terminal pose = w_viability × 1.0 ≈ 8 points,
                #    which dominates all other cost terms and forces the planner
                #    to strongly prefer exit trajectories.
                #    SET w_viability=0 FOR VANILLA DWA (no NN involvement at all).
                if cfg.w_viability > 0.0:
                    via = self._lookup_viability(tx, ty, ttheta)
                    c_viability = 1.0 - via      # 0 for clear, 1 for trapped
                else:
                    via = 1.0
                    c_viability = 0.0

                total = (
                    cfg.w_goal_heading * c_goal_heading
                    + cfg.w_goal_dist * c_goal_dist
                    + cfg.w_clearance * c_clearance
                    + cfg.w_viability * c_viability
                    - cfg.w_speed * c_speed
                )

                if total < best_cost:
                    best_cost = total
                    best_v = float(v)
                    best_omega = float(omega)
                    best_traj = traj
                    best_via = via

        info: Dict = {
            "best_cost": float(best_cost),
            "best_v": float(best_v),
            "best_omega": float(best_omega),
            "viability_at_terminal": float(best_via),
            "trajectory": best_traj,
        }
        return best_v, best_omega, info

    def _simulate(
        self,
        x: float,
        y: float,
        theta: float,
        v: float,
        omega: float,
        occ: np.ndarray,
        dist_tf: np.ndarray,
    ) -> Tuple[List[Tuple[float, float, float]], bool, float]:
        """
        Forward-simulate trajectory from (x, y, θ) using (v, ω).

        Uses Euler integration with the image-coordinate motion model:
          x_new = x + v · cos(θ) · dt
          y_new = y − v · sin(θ) · dt   (minus: image y increases South)
          θ_new = θ + ω · dt

        Args:
            x, y, theta: Current state.
            v:           Forward speed (pixels/step).
            omega:       Angular velocity (rad/step).
            occ:         Occupancy grid.
            dist_tf:     Precomputed distance transform.

        Returns:
            (trajectory, is_valid, min_clearance)
            trajectory:    List of (x, y, θ) tuples including initial pose.
            is_valid:      False if robot hits obstacle or goes out of bounds.
            min_clearance: Minimum distance to obstacle along trajectory.
        """
        cfg = self.config
        H, W = occ.shape
        traj: List[Tuple[float, float, float]] = [(x, y, theta)]
        min_clear = np.inf

        for _ in range(cfg.predict_steps):
            theta = theta + omega * cfg.dt
            x = x + v * np.cos(theta) * cfg.dt
            y = y - v * np.sin(theta) * cfg.dt  # image y-down convention

            # Bounds check
            if not (0.0 <= x < W and 0.0 <= y < H):
                return traj, False, 0.0

            xi = int(np.clip(round(x), 0, W - 1))
            yi = int(np.clip(round(y), 0, H - 1))

            # Collision check
            if occ[yi, xi] == 0:
                return traj, False, 0.0

            clearance = float(dist_tf[yi, xi])
            if clearance < cfg.min_clearance:
                return traj, False, clearance

            min_clear = min(min_clear, clearance)
            traj.append((x, y, theta))

        return traj, True, float(min_clear if np.isfinite(min_clear) else 0.0)

    # ------------------------------------------------------------------
    # Motion model  (static — used by caller to advance state)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_motion(
        state: DWAState,
        v: float,
        omega: float,
        dt: float = 1.0,
    ) -> DWAState:
        """
        Apply one step of the motion model.

        Args:
            state: Current robot state.
            v:     Forward speed (pixels/step).
            omega: Angular velocity (rad/step).
            dt:    Step duration.

        Returns:
            New DWAState after applying the action.
        """
        theta_new = state.theta + omega * dt
        x_new = state.x + v * np.cos(theta_new) * dt
        y_new = state.y - v * np.sin(theta_new) * dt
        return DWAState(x=x_new, y=y_new, theta=theta_new)

    # ------------------------------------------------------------------
    # Oracle timing helper  (for comparison in experiments)
    # ------------------------------------------------------------------

    @staticmethod
    def time_oracle_N_headings(
        occ: np.ndarray,
        robot_L: int,
        robot_W: int,
        n_headings: int = 16,
    ) -> float:
        """
        Time the Oracle (BFS) for ``n_headings`` evenly-spaced angles.

        This is the baseline the batched NN inference is compared against.
        Expected: ~189 ms × 16 = ~3,024 ms on a single CPU thread.

        Args:
            occ:       Occupancy grid.
            robot_L:   Robot length (pixels).
            robot_W:   Robot width (pixels).
            n_headings: Number of heading angles to query.

        Returns:
            Total elapsed time in milliseconds.
        """
        try:
            from src.oracle.extended_oracles import continuous_angle_viability
        except ImportError:
            logger.warning("Oracle not importable; returning 0.")
            return 0.0

        angles = [360.0 * i / n_headings for i in range(n_headings)]
        t0 = time.perf_counter()
        for angle in angles:
            _ = continuous_angle_viability(occ, robot_L, robot_W, angle)
        return (time.perf_counter() - t0) * 1000.0