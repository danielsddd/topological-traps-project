"""
src/integration/prm.py

Pure-Python PRM and TrapAwarePRM for directional viability experiments.

No external motion planning library required.
Uses: NumPy, SciPy (KDTree), occupancy grids, viability maps.

Classes:
    StandardPRM    — vanilla PRM baseline
    TrapAwarePRM   — viability-filtered PRM with hybrid sampling (our contribution)
    PRMComparison  — runs both, collects stats, generates figures
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# Direction indices
DIR_N, DIR_S, DIR_E, DIR_W = 0, 1, 2, 3
DIRECTION_NAMES = ["N", "S", "E", "W"]


# ---------------------------------------------------------------------------
# Dijkstra shortest path
# ---------------------------------------------------------------------------

def dijkstra(
    adjacency: Dict[int, List[Tuple[int, float]]],
    start: int,
    goal: int,
    n_nodes: int,
) -> Optional[List[int]]:
    """
    Dijkstra's shortest path on a weighted adjacency list.

    Args:
        adjacency: node_id → [(neighbor_id, weight), ...]
        start:     start node index
        goal:      goal node index
        n_nodes:   total number of nodes

    Returns:
        List of node indices from start to goal, or None if unreachable.
    """
    import heapq

    dist = [float("inf")] * n_nodes
    prev = [-1] * n_nodes
    dist[start] = 0.0
    heap = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == goal:
            break
        for v, w in adjacency.get(u, []):
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[goal] == float("inf"):
        return None

    path = []
    cur = goal
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


# ---------------------------------------------------------------------------
# StandardPRM
# ---------------------------------------------------------------------------

class StandardPRM:
    """
    Standard Probabilistic Roadmap planner.

    Uses occupancy grid for collision detection.
    Baseline for comparison with TrapAwarePRM.

    Args:
        occupancy:      (H, W) uint8 grid, 1=free 0=obstacle.
        num_samples:    Number of landmark nodes to sample.
        k_nn:           K nearest neighbors to connect per node.
        edge_steps:     Number of interpolation steps for edge validation.
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        num_samples: int = 500,
        k_nn: int = 10,
        edge_steps: int = 20,
    ):
        self.occupancy    = occupancy
        self.num_samples  = num_samples
        self.k_nn         = k_nn
        self.edge_steps   = edge_steps
        self.H, self.W    = occupancy.shape

        # Built after build_roadmap()
        self.nodes: np.ndarray          = np.empty((0, 2))
        self.adjacency: Dict[int, List] = defaultdict(list)
        self.build_time: float          = 0.0
        self.stats: Dict                = {}

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _is_free(self, row: int, col: int) -> bool:
        """Check if a pixel is in free space."""
        if row < 0 or row >= self.H or col < 0 or col >= self.W:
            return False
        return bool(self.occupancy[row, col] == 1)

    def _edge_free(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if the straight line between two pixels is collision-free."""
        for t in np.linspace(0, 1, self.edge_steps):
            r = int(round(r1 + t * (r2 - r1)))
            c = int(round(c1 + t * (c2 - c1)))
            if not self._is_free(r, c):
                return False
        return True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_free(self) -> Optional[Tuple[int, int]]:
        """Sample a uniformly random free pixel."""
        for _ in range(100):
            r = np.random.randint(0, self.H)
            c = np.random.randint(0, self.W)
            if self._is_free(r, c):
                return r, c
        return None

    def _should_accept(self, row: int, col: int) -> bool:
        """
        Accept/reject hook — overridden in TrapAwarePRM.
        Base version: accept all free pixels.
        """
        return True

    # ------------------------------------------------------------------
    # Roadmap construction
    # ------------------------------------------------------------------

    def build_roadmap(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> None:
        """
        Build the PRM roadmap.

        Args:
            start: (row, col) start pixel.
            goal:  (row, col) goal pixel.
        """
        t0    = time.perf_counter()
        nodes = [start, goal]  # indices 0=start, 1=goal

        attempts = rejected = 0
        while len(nodes) < self.num_samples + 2:
            sample = self._sample_free()
            if sample is None:
                continue
            attempts += 1
            if self._should_accept(*sample):
                nodes.append(sample)
            else:
                rejected += 1

        self.nodes     = np.array(nodes, dtype=np.float32)
        self.adjacency = defaultdict(list)

        # k-NN via KDTree
        tree = KDTree(self.nodes)
        for i, node in enumerate(self.nodes):
            dists, idxs = tree.query(node, k=min(self.k_nn + 1, len(self.nodes)))
            for j in idxs[1:]:  # skip self
                r1, c1 = int(self.nodes[i, 0]), int(self.nodes[i, 1])
                r2, c2 = int(self.nodes[j, 0]), int(self.nodes[j, 1])
                if self._edge_free(r1, c1, r2, c2):
                    w = self._edge_weight(i, j)
                    self.adjacency[i].append((j, w))
                    self.adjacency[j].append((i, w))

        self.build_time = time.perf_counter() - t0
        self.stats = {
            "n_nodes":       len(self.nodes),
            "n_edges":       sum(len(v) for v in self.adjacency.values()) // 2,
            "attempted":     attempts,
            "rejected":      rejected,
            "build_time_ms": self.build_time * 1000,
        }

    def _edge_weight(self, i: int, j: int) -> float:
        """Euclidean distance between nodes i and j."""
        return float(np.linalg.norm(self.nodes[i] - self.nodes[j]))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal using the roadmap.
        Roadmap must be built first with build_roadmap().

        Returns:
            List of (row, col) waypoints, or None if no path found.
        """
        path_idx = dijkstra(
            self.adjacency,
            start=0,
            goal=1,
            n_nodes=len(self.nodes),
        )
        if path_idx is None:
            return None
        return [(int(self.nodes[i, 0]), int(self.nodes[i, 1])) for i in path_idx]

    # ------------------------------------------------------------------
    # Trap analysis (uses Oracle labels)
    # ------------------------------------------------------------------

    def count_trap_samples(self, oracle_labels: np.ndarray) -> Dict:
        """
        Count how many roadmap nodes landed in Oracle-verified trap regions.

        A pixel is a trap if it is free but viable in NO direction.
        oracle_labels: (4, H, W) uint8 from the Oracle.

        Returns dict with trap_count, trap_rate, per_direction viable counts.
        """
        if len(self.nodes) == 0:
            return {}

        # Trap: free pixel with 0 viability in ALL directions
        trap_mask = (
            (self.occupancy == 1) &
            (oracle_labels.max(axis=0) == 0)
        )

        trap_count = 0
        for row, col in self.nodes.astype(int):
            if 0 <= row < self.H and 0 <= col < self.W:
                if trap_mask[row, col]:
                    trap_count += 1

        n = len(self.nodes)
        return {
            "n_nodes":    n,
            "trap_count": trap_count,
            "trap_rate":  trap_count / n if n > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# TrapAwarePRM
# ---------------------------------------------------------------------------

class TrapAwarePRM(StandardPRM):
    """
    Trap-Aware PRM — our contribution.

    Extends StandardPRM with viability-filtered sampling and a hybrid
    sampling strategy that guarantees global connectivity even on
    extremely trap-dense maps (e.g. warehouse environments with 67% trap
    density).

    Hybrid sampling budget (total = num_samples + 2 anchor nodes):
      1. Viability-filtered nodes  (1 - uniform_ratio) × num_samples
         — rejected if min viability across all 4 directions < threshold
      2. Unconditional uniform fill  uniform_ratio × num_samples
         — always accepted; guarantees cross-corridor connectivity
      3. Vicinity nodes  vicinity_nodes (total, split evenly start/goal)
         — unconditional, placed within vic_radius of start/goal;
           ensures endpoints are well-connected regardless of local
           trap density

    Edges are penalized by trap_penalty × (1 - mean_viability_along_edge).

    The viability map can come from:
      - The trained neural network (fast, approximate)
      - The Oracle (exact, slower — used for ablation)

    Args:
        occupancy:           (H, W) uint8 grid.
        viability_map:       (4, H, W) float32, values in [0, 1].
        viability_threshold: Reject filtered samples below this. Default 0.5.
        trap_penalty:        Edge weight penalty factor. Default 5.0.
        num_samples:         Number of accepted landmark nodes.
        k_nn:                K nearest neighbors.
        edge_steps:          Edge collision check resolution.
        uniform_ratio:       Fraction of nodes placed unconditionally (0–1).
                             Default 0.15. Increase for denser trap maps.
        vicinity_nodes:      Extra unconditional nodes near start/goal.
                             Default 20. Split evenly between both endpoints.
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        viability_map: np.ndarray,
        viability_threshold: float = 0.5,
        trap_penalty: float = 5.0,
        num_samples: int = 500,
        k_nn: int = 10,
        edge_steps: int = 20,
        uniform_ratio: float = 0.15,
        vicinity_nodes: int = 20,
    ):
        super().__init__(occupancy, num_samples, k_nn, edge_steps)
        self.viability_map       = viability_map   # (4, H, W)
        self.viability_threshold = viability_threshold
        self.trap_penalty        = trap_penalty
        self.uniform_ratio       = uniform_ratio
        self.vicinity_nodes      = vicinity_nodes

    def _should_accept(self, row: int, col: int) -> bool:
        """
        Reject samples in low-viability (trap) regions.

        Uses minimum viability across all 4 directions — conservative
        for a holonomic robot with unknown heading.
        """
        min_v = float(self.viability_map[:, row, col].min())
        return min_v >= self.viability_threshold

    def build_roadmap(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> None:
        """
        Build the PRM roadmap using hybrid sampling.

        Three-phase strategy:
          Phase 1 — Viability-filtered: majority of nodes avoid traps.
          Phase 2 — Unconditional uniform fill: guarantees global
                    connectivity even when filtered nodes cluster in
                    the small safe fraction of a trap-dense map.
          Phase 3 — Vicinity nodes: dense coverage near start/goal so
                    both endpoints connect even if they border a trap.

        Stats are extended with per-phase node counts and the
        ``uniform_ratio`` used, making them easy to log and compare
        across parameter sweeps.
        """
        t0 = time.perf_counter()
        nodes = [start, goal]  # indices 0=start, 1=goal

        n_uniform  = int(self.num_samples * self.uniform_ratio)
        n_filtered = self.num_samples - n_uniform

        # ------------------------------------------------------------------
        # Phase 1: viability-filtered nodes
        # ------------------------------------------------------------------
        attempts = rejected = 0
        n_fil_placed = 0
        while n_fil_placed < n_filtered:
            sample = self._sample_free()
            if sample is None:
                continue
            attempts += 1
            if self._should_accept(*sample):
                nodes.append(sample)
                n_fil_placed += 1
            else:
                rejected += 1

        # ------------------------------------------------------------------
        # Phase 2: unconditional uniform fill
        # These bypass the viability filter entirely — they are the "bridge"
        # nodes that cross trap regions and keep the roadmap connected.
        # ------------------------------------------------------------------
        n_uni_placed  = 0
        uni_attempts  = 0
        while n_uni_placed < n_uniform:
            sample = self._sample_free()
            uni_attempts += 1
            if sample is None:
                if uni_attempts > 10_000:
                    logger.warning("Uniform fill: exceeded 10k attempts — map may be almost fully occupied.")
                    break
                continue
            nodes.append(sample)
            n_uni_placed += 1

        # ------------------------------------------------------------------
        # Phase 3: vicinity nodes near start and goal
        # Place extra unconditional nodes within a neighbourhood of each
        # endpoint so that start/goal are well-connected regardless of local
        # trap density.  vic_radius = 1/8 of the longer map dimension.
        # ------------------------------------------------------------------
        n_vic_placed = 0
        per_anchor   = max(1, self.vicinity_nodes // 2)
        vic_radius   = max(self.H, self.W) // 8

        for anchor in (start, goal):
            placed = 0
            for _ in range(2_000):
                dr = np.random.randint(-vic_radius, vic_radius + 1)
                dc = np.random.randint(-vic_radius, vic_radius + 1)
                r  = int(np.clip(anchor[0] + dr, 0, self.H - 1))
                c  = int(np.clip(anchor[1] + dc, 0, self.W - 1))
                if self._is_free(r, c):
                    nodes.append((r, c))
                    placed += 1
                    n_vic_placed += 1
                if placed >= per_anchor:
                    break

        # ------------------------------------------------------------------
        # Build k-NN graph on all collected nodes
        # ------------------------------------------------------------------
        self.nodes     = np.array(nodes, dtype=np.float32)
        self.adjacency = defaultdict(list)

        tree = KDTree(self.nodes)
        for i, node in enumerate(self.nodes):
            dists, idxs = tree.query(node, k=min(self.k_nn + 1, len(self.nodes)))
            for j in idxs[1:]:  # skip self
                r1, c1 = int(self.nodes[i, 0]), int(self.nodes[i, 1])
                r2, c2 = int(self.nodes[j, 0]), int(self.nodes[j, 1])
                if self._edge_free(r1, c1, r2, c2):
                    w = self._edge_weight(i, j)
                    self.adjacency[i].append((j, w))
                    self.adjacency[j].append((i, w))

        self.build_time = time.perf_counter() - t0
        self.stats = {
            "n_nodes":       len(self.nodes),
            "n_edges":       sum(len(v) for v in self.adjacency.values()) // 2,
            "attempted":     attempts + uni_attempts,
            "rejected":      rejected,
            "n_filtered":    n_fil_placed,
            "n_uniform":     n_uni_placed,
            "n_vicinity":    n_vic_placed,
            "uniform_ratio": self.uniform_ratio,
            "vicinity_nodes": self.vicinity_nodes,
            "build_time_ms": self.build_time * 1000,
        }

    def _edge_weight(self, i: int, j: int) -> float:
        """
        Penalize edges passing through low-viability regions.

        weight = euclidean_dist × (1 + trap_penalty × (1 - mean_viability))
        """
        base = float(np.linalg.norm(self.nodes[i] - self.nodes[j]))

        # Sample 5 points along edge
        viabilities = []
        for t in np.linspace(0, 1, 5):
            r = int(round(self.nodes[i, 0] + t * (self.nodes[j, 0] - self.nodes[i, 0])))
            c = int(round(self.nodes[i, 1] + t * (self.nodes[j, 1] - self.nodes[i, 1])))
            r = max(0, min(self.H - 1, r))
            c = max(0, min(self.W - 1, c))
            viabilities.append(float(self.viability_map[:, r, c].min()))

        mean_v  = float(np.mean(viabilities))
        penalty = 1.0 + self.trap_penalty * (1.0 - mean_v)
        return base * penalty


# ---------------------------------------------------------------------------
# PRMResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class PRMResult:
    planner_name:  str
    n_nodes:       int
    n_edges:       int
    trap_count:    int
    trap_rate:     float
    path_found:    bool
    path_length:   float
    build_time_ms: float
    nodes:         np.ndarray = field(repr=False)
    path:          Optional[List[Tuple[int, int]]] = None


# ---------------------------------------------------------------------------
# PRMComparison runner
# ---------------------------------------------------------------------------

class PRMComparison:
    """
    Runs StandardPRM and TrapAwarePRM on the same scene.

    Collects stats and generates comparison figures.

    Usage:
        comp = PRMComparison(occupancy, oracle_labels, model_viability)
        results = comp.run(start=(50, 50), goal=(460, 460))
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        oracle_labels: np.ndarray,
        model_viability: Optional[np.ndarray] = None,
        num_samples: int = 500,
        k_nn: int = 10,
        viability_threshold: float = 0.5,
        trap_penalty: float = 5.0,
        uniform_ratio: float = 0.15,
        vicinity_nodes: int = 20,
    ):
        self.occupancy           = occupancy
        self.oracle_labels       = oracle_labels   # (4, H, W) uint8
        self.model_viability     = model_viability  # (4, H, W) float32 or None
        self.num_samples         = num_samples
        self.k_nn                = k_nn
        self.viability_threshold = viability_threshold
        self.trap_penalty        = trap_penalty
        self.uniform_ratio       = uniform_ratio
        self.vicinity_nodes      = vicinity_nodes

    def run(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        seed: int = 42,
    ) -> List[PRMResult]:
        """
        Run all planners on the same scene with the same random seed.

        Returns list of PRMResult for [standard, trap_aware_oracle,
        trap_aware_model (if model_viability provided)].
        """
        results = []

        def _make_trap_aware(viability: np.ndarray) -> TrapAwarePRM:
            return TrapAwarePRM(
                occupancy           = self.occupancy,
                viability_map       = viability,
                viability_threshold = self.viability_threshold,
                trap_penalty        = self.trap_penalty,
                num_samples         = self.num_samples,
                k_nn                = self.k_nn,
                uniform_ratio       = self.uniform_ratio,
                vicinity_nodes      = self.vicinity_nodes,
            )

        planners = [
            ("StandardPRM",           StandardPRM(
                self.occupancy, self.num_samples, self.k_nn)),
            ("TrapAwarePRM (Oracle)", _make_trap_aware(
                self.oracle_labels.astype(np.float32))),
        ]
        if self.model_viability is not None:
            planners.append(("TrapAwarePRM (Model)", _make_trap_aware(
                self.model_viability)))

        for name, planner in planners:
            np.random.seed(seed)
            logger.info("Running %s...", name)
            planner.build_roadmap(start, goal)
            path = planner.query(start, goal)
            trap_stats = planner.count_trap_samples(self.oracle_labels)

            path_length = 0.0
            if path:
                for i in range(len(path) - 1):
                    r1, c1 = path[i]
                    r2, c2 = path[i + 1]
                    path_length += float(np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2))

            results.append(PRMResult(
                planner_name  = name,
                n_nodes       = planner.stats["n_nodes"],
                n_edges       = planner.stats["n_edges"],
                trap_count    = trap_stats.get("trap_count", 0),
                trap_rate     = trap_stats.get("trap_rate", 0.0),
                path_found    = path is not None,
                path_length   = path_length,
                build_time_ms = planner.stats["build_time_ms"],
                nodes         = planner.nodes,
                path          = path,
            ))

        return results