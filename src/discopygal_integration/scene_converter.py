"""
Scene Converter - HouseExpo JSON to DiscoPyGal Scene.

This module converts HouseExpo floor plan files to DiscoPyGal Scene format,
enabling use of DiscoPyGal's motion planning and collision detection.

Conversion Process:
1. Parse HouseExpo JSON wall segments
2. Create thin rectangle obstacles for each wall
3. Create rectangular robot with specified dimensions
4. Package into DiscoPyGal Scene object
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import DiscoPyGal - may not be available
try:
    from discopygal.bindings import Point_2, Polygon_2, FT
    from discopygal.solvers_infra import Scene, RobotPolygon, ObstaclePolygon
    DISCOPYGAL_AVAILABLE = True
except ImportError:
    DISCOPYGAL_AVAILABLE = False
    logger.warning("DiscoPyGal not available - scene conversion disabled")


def houseexpo_to_discopygal(
    json_path: str,
    robot_length: float,
    robot_width: float,
    start_pos: Tuple[float, float] = None,
    end_pos: Tuple[float, float] = None,
    scale: float = 1.0,
    wall_thickness: float = 0.1
):
    """
    Convert HouseExpo JSON floor plan to DiscoPyGal Scene.
    
    Args:
        json_path: Path to HouseExpo JSON file
        robot_length: Robot length in world units
        robot_width: Robot width in world units
        start_pos: (x, y) start position (optional)
        end_pos: (x, y) goal position (optional)
        scale: Scale factor for coordinates
        wall_thickness: Thickness of wall obstacles
    
    Returns:
        DiscoPyGal Scene object
    
    Raises:
        ImportError: If DiscoPyGal is not available
        FileNotFoundError: If JSON file not found
        ValueError: If JSON format is invalid
    """
    if not DISCOPYGAL_AVAILABLE:
        raise ImportError(
            "DiscoPyGal is not available. Install with: pip install discopygal-taucgl"
        )
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'walls' not in data:
        raise ValueError(f"No 'walls' key in {json_path}")
    
    walls = data['walls']
    
    # Create Scene
    scene = Scene()
    
    # Convert walls to obstacles
    # Each wall segment [x1, y1, x2, y2] becomes a thin rectangle obstacle
    for wall in walls:
        x1, y1, x2, y2 = [v * scale for v in wall]
        
        # Compute wall direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            continue
        
        # Normalized perpendicular for wall thickness
        thickness = wall_thickness * scale
        px = -dy / length * thickness / 2
        py = dx / length * thickness / 2
        
        # Rectangle corners
        points = [
            Point_2(FT(x1 - px), FT(y1 - py)),
            Point_2(FT(x2 - px), FT(y2 - py)),
            Point_2(FT(x2 + px), FT(y2 + py)),
            Point_2(FT(x1 + px), FT(y1 + py)),
        ]
        
        try:
            poly = Polygon_2(points)
            obstacle = ObstaclePolygon(poly)
            scene.add_obstacle(obstacle)
        except Exception as e:
            logger.warning(f"Failed to create wall obstacle: {e}")
            continue
    
    # Create rectangular robot
    half_l = robot_length / 2
    half_w = robot_width / 2
    robot_points = [
        Point_2(FT(-half_l), FT(-half_w)),
        Point_2(FT(half_l), FT(-half_w)),
        Point_2(FT(half_l), FT(half_w)),
        Point_2(FT(-half_l), FT(half_w)),
    ]
    robot_poly = Polygon_2(robot_points)
    
    # Set default start/end if not provided
    if start_pos is None:
        # Find a likely free position (center of bounding box)
        all_walls = np.array(walls)
        all_points = all_walls.reshape(-1, 2)
        center = all_points.mean(axis=0) * scale
        start_pos = tuple(center)
    
    if end_pos is None:
        # Default to offset from start
        end_pos = (start_pos[0] + 5 * scale, start_pos[1] + 5 * scale)
    
    robot = RobotPolygon(
        poly=robot_poly,
        start=Point_2(FT(start_pos[0]), FT(start_pos[1])),
        end=Point_2(FT(end_pos[0]), FT(end_pos[1]))
    )
    scene.add_robot(robot)
    
    return scene


def save_discopygal_scene(scene, output_path: str):
    """
    Save DiscoPyGal scene to JSON file.
    
    Args:
        scene: DiscoPyGal Scene object
        output_path: Output file path
    """
    if not DISCOPYGAL_AVAILABLE:
        raise ImportError("DiscoPyGal not available")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scene.to_file(output_path)


def batch_convert_houseexpo(
    json_dir: str,
    output_dir: str,
    robot_length: float,
    robot_width: float,
    num_maps: int = None,
    scale: float = 1.0
) -> Tuple[int, int]:
    """
    Batch convert HouseExpo maps to DiscoPyGal scenes.
    
    Args:
        json_dir: Directory containing JSON files
        output_dir: Output directory for scenes
        robot_length: Robot length
        robot_width: Robot width
        num_maps: Maximum number of maps (None = all)
        scale: Coordinate scale factor
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    from tqdm import tqdm
    
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = sorted(json_dir.glob("*.json"))
    if num_maps:
        json_files = json_files[:num_maps]
    
    successful = 0
    failed = 0
    
    for json_path in tqdm(json_files, desc="Converting"):
        try:
            scene = houseexpo_to_discopygal(
                str(json_path),
                robot_length,
                robot_width,
                scale=scale
            )
            
            output_path = output_dir / f"{json_path.stem}_scene.json"
            save_discopygal_scene(scene, str(output_path))
            successful += 1
            
        except Exception as e:
            logger.warning(f"Failed to convert {json_path}: {e}")
            failed += 1
    
    return successful, failed


def occupancy_to_obstacles(
    occupancy: np.ndarray,
    resolution: int = 512
) -> List:
    """
    Convert occupancy grid to list of obstacle polygons.
    
    This is an alternative to wall-based conversion, useful when
    starting from a rasterized occupancy grid.
    
    Args:
        occupancy: Binary occupancy grid (1=free, 0=obstacle)
        resolution: Grid resolution for coordinate scaling
    
    Returns:
        List of obstacle point lists
    
    Note:
        This is a simplified implementation. For better results,
        use contour detection and polygon simplification.
    """
    import cv2
    
    # Find contours of obstacle regions
    obstacle_mask = (1 - occupancy).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        obstacle_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    obstacles = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        # Convert to world coordinates
        points = []
        for pt in approx:
            x = pt[0][0] / resolution
            y = pt[0][1] / resolution
            points.append((x, y))
        
        obstacles.append(points)
    
    return obstacles


if __name__ == "__main__":
    print("Scene Converter Module")
    print(f"DiscoPyGal available: {DISCOPYGAL_AVAILABLE}")
    
    if not DISCOPYGAL_AVAILABLE:
        print("Install DiscoPyGal to use this module:")
        print("  pip install discopygal-taucgl")
