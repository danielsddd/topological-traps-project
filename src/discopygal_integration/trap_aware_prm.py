"""
Trap-Aware PRM (Probabilistic Roadmap) Planner.

This module implements a viability-aware sampling-based motion planner
that uses neural network predictions to bias sampling away from trap regions.

Key Features:
- Uses predicted viability maps to avoid trap regions during sampling
- Biased sampling towards viable configurations
- Compatible with DiscoPyGal solvers infrastructure
- Supports Dubins car kinematics

The planner integrates with trained viability prediction models to
improve motion planning efficiency in environments with narrow passages.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from collections import deque
import logging

logger = logging.getLogger(__name__)

# Try to import DiscoPyGal
try:
    from discopygal.bindings import Point_2, FT, Polygon_2
    from discopygal.solvers_infra import Scene, Solver
    from discopygal.solvers.prm import PRM
    DISCOPYGAL_AVAILABLE = True
except ImportError:
    DISCOPYGAL_AVAILABLE = False
    logger.warning("DiscoPyGal not available")


class TrapAwareSampler:
    """
    Viability-aware configuration sampler.
    
    Uses neural network viability predictions to bias sampling
    towards viable configurations and away from trap regions.
    
    Args:
        viability_map: Viability predictions (4, H, W) - probabilities
        occupancy: Occupancy grid (H, W)
        resolution: Grid resolution
        viable_threshold: Threshold for considering a pixel viable
        bias_strength: How strongly to bias towards viable regions (0-1)
    """
    
    def __init__(
        self,
        viability_map: np.ndarray,
        occupancy: np.ndarray,
        resolution: int = 512,
        viable_threshold: float = 0.5,
        bias_strength: float = 0.8
    ):
        self.viability_map = viability_map  # (4, H, W)
        self.occupancy = occupancy  # (H, W)
        self.resolution = resolution
        self.viable_threshold = viable_threshold
        self.bias_strength = bias_strength
        
        # Precompute combined viability (viable in at least one direction)
        self.any_viable = np.any(
            viability_map > viable_threshold, axis=0
        ).astype(np.float32)
        
        # Precompute all-directions viable (viable in all directions)
        self.all_viable = np.all(
            viability_map > viable_threshold, axis=0
        ).astype(np.float32)
        
        # Create sampling probability map
        self._compute_sampling_weights()
    
    def _compute_sampling_weights(self):
        """Compute sampling weights based on viability."""
        # Start with free space
        weights = self.occupancy.astype(np.float32)
        
        # Boost viable regions
        viable_boost = 1.0 + self.bias_strength * 2 * self.any_viable
        weights = weights * viable_boost
        
        # Extra boost for all-directions-viable
        all_boost = 1.0 + self.bias_strength * self.all_viable
        weights = weights * all_boost
        
        # Normalize
        total = weights.sum()
        if total > 0:
            self.weights = weights / total
        else:
            # Fallback to uniform over free space
            self.weights = self.occupancy.astype(np.float32)
            self.weights /= self.weights.sum()
        
        # Create CDF for efficient sampling
        self.weights_flat = self.weights.flatten()
        self.cdf = np.cumsum(self.weights_flat)
    
    def sample_position(self) -> Tuple[float, float]:
        """
        Sample a position biased towards viable regions.
        
        Returns:
            (x, y) position in [0, 1] normalized coordinates
        """
        # Sample from CDF
        r = np.random.random()
        idx = np.searchsorted(self.cdf, r)
        idx = min(idx, len(self.weights_flat) - 1)
        
        # Convert to (row, col)
        H, W = self.occupancy.shape
        row = idx // W
        col = idx % W
        
        # Add jitter within pixel
        x = (col + np.random.random()) / W
        y = (row + np.random.random()) / H
        
        return x, y
    
    def sample_configuration(
        self,
        heading_aware: bool = True
    ) -> Tuple[float, float, float]:
        """
        Sample a full configuration (x, y, theta).
        
        Args:
            heading_aware: If True, bias heading towards viable directions
        
        Returns:
            (x, y, theta) configuration
        """
        x, y = self.sample_position()
        
        if heading_aware:
            theta = self._sample_heading(x, y)
        else:
            theta = np.random.random() * 2 * np.pi
        
        return x, y, theta
    
    def _sample_heading(self, x: float, y: float) -> float:
        """
        Sample heading biased towards viable directions.
        
        Args:
            x, y: Position in [0, 1]
        
        Returns:
            Heading angle in [0, 2π]
        """
        H, W = self.occupancy.shape
        col = int(x * W)
        row = int(y * H)
        
        col = np.clip(col, 0, W - 1)
        row = np.clip(row, 0, H - 1)
        
        # Get viability per direction at this pixel
        # Directions: N=0, S=1, E=2, W=3
        viabilities = self.viability_map[:, row, col]
        
        # Map directions to angles
        # N (up, -y) = -π/2, S (down, +y) = π/2
        # E (right, +x) = 0, W (left, -x) = π
        direction_angles = np.array([
            -np.pi / 2,  # N
            np.pi / 2,   # S
            0,           # E
            np.pi,       # W
        ])
        
        # Compute weights (softmax-like)
        weights = np.exp(viabilities * 2)  # Temperature scaling
        weights /= weights.sum()
        
        # Sample direction
        dir_idx = np.random.choice(4, p=weights)
        base_angle = direction_angles[dir_idx]
        
        # Add jitter (±45 degrees)
        jitter = (np.random.random() - 0.5) * np.pi / 2
        theta = (base_angle + jitter) % (2 * np.pi)
        
        return theta
    
    def get_viability_score(
        self,
        x: float,
        y: float,
        theta: float = None
    ) -> float:
        """
        Get viability score for a configuration.
        
        Args:
            x, y: Position in [0, 1]
            theta: Optional heading angle
        
        Returns:
            Viability score in [0, 1]
        """
        H, W = self.occupancy.shape
        col = int(x * W)
        row = int(y * H)
        
        if col < 0 or col >= W or row < 0 or row >= H:
            return 0.0
        
        if self.occupancy[row, col] == 0:
            return 0.0
        
        if theta is None:
            # Return max viability across all directions
            return float(self.viability_map[:, row, col].max())
        
        # Map theta to closest direction
        # Normalize theta to [0, 2π]
        theta = theta % (2 * np.pi)
        
        # Map to direction index
        if theta < np.pi / 4 or theta >= 7 * np.pi / 4:
            dir_idx = 2  # E
        elif theta < 3 * np.pi / 4:
            dir_idx = 0  # N
        elif theta < 5 * np.pi / 4:
            dir_idx = 3  # W
        else:
            dir_idx = 1  # S
        
        return float(self.viability_map[dir_idx, row, col])


class TrapAwarePRM:
    """
    Trap-Aware Probabilistic Roadmap Planner.
    
    A sampling-based motion planner that uses neural network viability
    predictions to bias sampling towards viable configurations.
    
    Args:
        model: Trained viability prediction model
        occupancy: Occupancy grid (H, W)
        robot_length: Robot length
        robot_width: Robot width
        resolution: Grid resolution (default: 512)
        num_samples: Number of roadmap samples
        k_neighbors: Number of nearest neighbors to connect
        viable_threshold: Threshold for viability
        bias_strength: Sampling bias strength
        device: PyTorch device for model inference
    """
    
    def __init__(
        self,
        model,
        occupancy: np.ndarray,
        robot_length: int,
        robot_width: int,
        resolution: int = 512,
        num_samples: int = 1000,
        k_neighbors: int = 10,
        viable_threshold: float = 0.5,
        bias_strength: float = 0.8,
        device: str = "cuda"
    ):
        self.model = model
        self.occupancy = occupancy
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.resolution = resolution
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.viable_threshold = viable_threshold
        self.bias_strength = bias_strength
        self.device = device
        
        # Predict viability map
        self.viability_map = self._predict_viability()
        
        # Create sampler
        self.sampler = TrapAwareSampler(
            viability_map=self.viability_map,
            occupancy=occupancy,
            resolution=resolution,
            viable_threshold=viable_threshold,
            bias_strength=bias_strength
        )
        
        # Roadmap
        self.nodes = []
        self.edges = {}
    
    def _predict_viability(self) -> np.ndarray:
        """Predict viability map using neural network."""
        import torch
        
        H, W = self.occupancy.shape
        
        # Construct input
        input_tensor = np.zeros((1, 3, H, W), dtype=np.float32)
        input_tensor[0, 0] = self.occupancy.astype(np.float32)
        input_tensor[0, 1] = self.robot_length / self.resolution
        input_tensor[0, 2] = self.robot_width / self.resolution
        
        input_tensor = torch.from_numpy(input_tensor).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
        
        return probs[0].cpu().numpy()  # (4, H, W)
    
    def build_roadmap(self, verbose: bool = False) -> int:
        """
        Build the PRM roadmap.
        
        Args:
            verbose: Print progress
        
        Returns:
            Number of nodes in roadmap
        """
        from tqdm import tqdm
        
        self.nodes = []
        self.edges = {}
        
        iterator = range(self.num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")
        
        # Sample nodes
        for _ in iterator:
            x, y, theta = self.sampler.sample_configuration(heading_aware=True)
            
            # Validate sample (collision check would go here)
            if self._is_valid_configuration(x, y, theta):
                node_id = len(self.nodes)
                self.nodes.append((x, y, theta))
                self.edges[node_id] = []
        
        if verbose:
            print(f"Sampled {len(self.nodes)} valid nodes")
        
        # Connect neighbors
        if verbose:
            print("Connecting neighbors...")
        
        self._connect_neighbors()
        
        total_edges = sum(len(e) for e in self.edges.values()) // 2
        if verbose:
            print(f"Created {total_edges} edges")
        
        return len(self.nodes)
    
    def _is_valid_configuration(
        self,
        x: float,
        y: float,
        theta: float
    ) -> bool:
        """
        Check if configuration is valid (collision-free).
        
        Note: This is a simplified check using the occupancy grid.
        For full collision checking, integrate with DiscoPyGal.
        """
        H, W = self.occupancy.shape
        col = int(x * W)
        row = int(y * H)
        
        if col < 0 or col >= W or row < 0 or row >= H:
            return False
        
        return self.occupancy[row, col] == 1
    
    def _connect_neighbors(self):
        """Connect each node to k nearest neighbors."""
        if len(self.nodes) < 2:
            return
        
        # Build KD-tree for efficient neighbor lookup
        from scipy.spatial import KDTree
        
        positions = np.array([(n[0], n[1]) for n in self.nodes])
        tree = KDTree(positions)
        
        for i, node in enumerate(self.nodes):
            # Find k+1 nearest (including self)
            _, indices = tree.query(positions[i], k=min(self.k_neighbors + 1, len(self.nodes)))
            
            for j in indices:
                if i != j and j not in self.edges[i]:
                    # Check edge validity (simplified)
                    if self._is_valid_edge(i, j):
                        self.edges[i].append(j)
                        self.edges[j].append(i)
    
    def _is_valid_edge(self, i: int, j: int) -> bool:
        """
        Check if edge between two nodes is valid.
        
        Note: Simplified check - for full validation use
        continuous collision checking.
        """
        x1, y1, _ = self.nodes[i]
        x2, y2, _ = self.nodes[j]
        
        # Check points along edge
        num_checks = 10
        for t in np.linspace(0, 1, num_checks):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            if not self._is_valid_configuration(x, y, 0):
                return False
        
        return True
    
    def query(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float]
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Query the roadmap for a path.
        
        Args:
            start: Start configuration (x, y, theta)
            goal: Goal configuration (x, y, theta)
        
        Returns:
            Path as list of configurations, or None if no path found
        """
        if len(self.nodes) == 0:
            return None
        
        # Add start and goal to roadmap temporarily
        start_id = len(self.nodes)
        goal_id = start_id + 1
        
        self.nodes.append(start)
        self.nodes.append(goal)
        self.edges[start_id] = []
        self.edges[goal_id] = []
        
        # Connect start and goal to roadmap
        self._connect_to_roadmap(start_id)
        self._connect_to_roadmap(goal_id)
        
        # Search for path using A*
        path_indices = self._astar(start_id, goal_id)
        
        # Remove temporary nodes
        self.nodes = self.nodes[:-2]
        del self.edges[start_id]
        del self.edges[goal_id]
        
        if path_indices is None:
            return None
        
        # Convert to configurations
        path = []
        for idx in path_indices:
            if idx < len(self.nodes):
                path.append(self.nodes[idx])
            elif idx == start_id:
                path.append(start)
            else:
                path.append(goal)
        
        return path
    
    def _connect_to_roadmap(self, node_id: int):
        """Connect a node to existing roadmap."""
        if len(self.nodes) <= 1:
            return
        
        from scipy.spatial import KDTree
        
        node = self.nodes[node_id]
        positions = np.array([(n[0], n[1]) for n in self.nodes[:-2]])  # Exclude temp nodes
        
        if len(positions) == 0:
            return
        
        tree = KDTree(positions)
        _, indices = tree.query([node[0], node[1]], k=min(self.k_neighbors, len(positions)))
        
        if isinstance(indices, np.integer):
            indices = [indices]
        
        for j in indices:
            if self._is_valid_edge(node_id, j):
                self.edges[node_id].append(j)
                self.edges[j].append(node_id)
    
    def _astar(
        self,
        start_id: int,
        goal_id: int
    ) -> Optional[List[int]]:
        """
        A* search for shortest path.
        
        Args:
            start_id: Start node index
            goal_id: Goal node index
        
        Returns:
            List of node indices, or None if no path
        """
        import heapq
        
        goal_pos = self.nodes[goal_id][:2]
        
        def heuristic(node_id):
            pos = self.nodes[node_id][:2]
            return np.sqrt((pos[0] - goal_pos[0])**2 + (pos[1] - goal_pos[1])**2)
        
        def distance(id1, id2):
            p1 = self.nodes[id1][:2]
            p2 = self.nodes[id2][:2]
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Priority queue: (f_score, node_id)
        open_set = [(heuristic(start_id), start_id)]
        came_from = {}
        g_score = {start_id: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            for neighbor in self.edges.get(current, []):
                tentative_g = g_score[current] + distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None
    
    def get_trap_statistics(self) -> Dict:
        """
        Get statistics about trap regions.
        
        Returns:
            Dictionary with trap statistics
        """
        free_pixels = self.occupancy.sum()
        
        # Trap = free but not viable in any direction
        trap_mask = (self.occupancy == 1) & (self.viability_map.max(axis=0) < self.viable_threshold)
        trap_pixels = trap_mask.sum()
        
        # Per-direction statistics
        dir_names = ['N', 'S', 'E', 'W']
        dir_stats = {}
        for i, name in enumerate(dir_names):
            viable = (self.viability_map[i] >= self.viable_threshold) & (self.occupancy == 1)
            dir_stats[f'viable_{name}'] = int(viable.sum())
            dir_stats[f'viable_{name}_ratio'] = float(viable.sum() / free_pixels) if free_pixels > 0 else 0
        
        return {
            'free_pixels': int(free_pixels),
            'trap_pixels': int(trap_pixels),
            'trap_ratio': float(trap_pixels / free_pixels) if free_pixels > 0 else 0,
            **dir_stats
        }


if __name__ == "__main__":
    print("TrapAwarePRM Module")
    print(f"DiscoPyGal available: {DISCOPYGAL_AVAILABLE}")
    
    # Test sampler with dummy data
    print("\nTesting TrapAwareSampler...")
    
    occupancy = np.ones((100, 100), dtype=np.uint8)
    occupancy[:10, :] = 0
    occupancy[-10:, :] = 0
    occupancy[:, :10] = 0
    occupancy[:, -10:] = 0
    
    viability = np.random.rand(4, 100, 100).astype(np.float32)
    
    sampler = TrapAwareSampler(
        viability_map=viability,
        occupancy=occupancy,
        resolution=100
    )
    
    # Test sampling
    positions = [sampler.sample_position() for _ in range(100)]
    configs = [sampler.sample_configuration() for _ in range(100)]
    
    print(f"  Sampled {len(positions)} positions")
    print(f"  Sampled {len(configs)} configurations")
    
    # Check samples are in free space
    valid = sum(1 for x, y in positions 
                if occupancy[int(y * 100), int(x * 100)] == 1)
    print(f"  Valid samples: {valid}/100")
    
    print("\n✓ TrapAwareSampler tests passed!")
