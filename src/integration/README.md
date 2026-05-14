# src/integration/

Probabilistic Roadmap Method (PRM) planners that use the trained viability
model to avoid topological traps during path planning.

Implemented entirely in pure NumPy — no external motion planning library
dependency.

---

## Files

| File | Description |
|---|---|
| `prm.py` | `StandardPRM`, `TrapAwarePRM`, `PRMResult`, `PRMComparison`. |

---

## StandardPRM

Baseline PRM: samples uniformly from free space, connects k-nearest
neighbours with straight-line edges (collision checked via rasterisation),
and queries shortest path with Dijkstra.

---

## TrapAwarePRM

Extends `StandardPRM` with a **viability-guided hybrid sampling strategy**:

### Sampling

1. **Viability-filtered** (default 85% of nodes): sample a random free
   pixel; accept only if the NN viability score exceeds `threshold`
   in at least one direction. Keeps the roadmap out of topological traps.
2. **Uniform** (default 15% of nodes): accept any free pixel regardless
   of viability. Preserves connectivity in highly trap-dense regions.
3. **Vicinity nodes**: a fixed number of unconditional nodes placed near
   start and goal to guarantee endpoint coverage.

### Edge weighting

Edges that cross low-viability pixels receive a `trap_penalty` multiplier
(default 5×) on their Euclidean length. This discourages trap-passing
paths without hard-forbidding them.

### Local planner

Straight-line holonomic collision check (pixel rasterisation). Kinematic
constraints (e.g. Dubins curves) are left as future work.

---

## Usage

```python
from src.integration.prm import StandardPRM, TrapAwarePRM

# Standard baseline
std = StandardPRM(occupancy=occ, num_samples=500, k_nn=10)
std_result = std.plan(start=(r0, c0), goal=(r1, c1))

# Trap-aware (requires NN viability map)
trap = TrapAwarePRM(
    viability_map=nn_output,   # (4, H, W) float32 — output of model.predict()
    occupancy=occ,
    num_samples=500,
    k_nn=10,
    threshold=0.5,             # min viability to accept a node
    trap_penalty=5.0,          # edge weight multiplier for trap-crossing edges
    uniform_ratio=0.15,        # fraction of nodes sampled without viability filter
    vicinity_nodes=20,         # unconditional nodes placed near start/goal
)
trap_result = trap.plan(start=(r0, c0), goal=(r1, c1))

print(trap_result.trap_rate)   # fraction of roadmap nodes in trap regions
print(trap_result.path_found)  # bool
print(trap_result.build_ms)    # roadmap construction time in ms
```

---

## Benchmark results

On a synthetic warehouse map (512×512, 67% trap density, 5 runs):

| Planner | Trap rate | Build time | Path found |
|---|---|---|---|
| StandardPRM | 0.673 | 641 ms | 100% |
| TrapAwarePRM (Oracle labels) | 0.123 | 1137 ms | 100% |
| TrapAwarePRM (NN, 12 ms pred) | **0.125** | 1132 ms | 100% |

The NN-guided planner matches Oracle quality (81.4% vs 81.8% trap reduction)
while replacing the 165 ms oracle with a 12 ms neural network inference.
