"""
Robot Geometry Utilities.

This module provides functions for computing robot geometry properties
needed for rotation checking and viability analysis.

Key Concepts:
- Robot is a rectangle with length (along heading) and width (perpendicular)
- Robot diagonal determines minimum clearance needed for rotation
- In-place rotation sweeps a circular area with diameter = diagonal
"""

import numpy as np
import math
from typing import Tuple, Optional


def get_robot_diagonal(length: int, width: int) -> float:
    """
    Calculate the robot's diagonal (rotation clearance).
    
    The diagonal is the distance from one corner to the opposite corner,
    which equals the diameter of the minimum circle containing the robot.
    A robot needs this much clearance to rotate 360° in place.
    
    Args:
        length: Robot length in pixels (along heading direction)
        width: Robot width in pixels (perpendicular to heading)
    
    Returns:
        Diagonal distance in pixels
    
    Example:
        >>> get_robot_diagonal(14, 9)
        16.643...  # sqrt(14² + 9²)
    """
    return math.sqrt(length**2 + width**2)


def get_rotation_kernel_size(
    length: int,
    width: int,
    margin: float = 1.0
) -> int:
    """
    Get kernel size for rotation checking.
    
    Returns the diameter of the circular kernel needed for
    morphological erosion to find rotation-safe regions.
    
    Args:
        length: Robot length in pixels
        width: Robot width in pixels  
        margin: Safety margin multiplier (default 1.0 = exact fit)
    
    Returns:
        Kernel diameter in pixels (odd number for symmetric kernel)
    
    Example:
        >>> get_rotation_kernel_size(14, 9, margin=1.2)
        21  # ceil(sqrt(14² + 9²) * 1.2), rounded to odd
    """
    diagonal = get_robot_diagonal(length, width)
    size = int(math.ceil(diagonal * margin))
    
    # Ensure odd number for symmetric kernel
    if size % 2 == 0:
        size += 1
    
    return size


def create_robot_footprint(
    length: int,
    width: int,
    angle: float = 0.0,
    canvas_size: Optional[int] = None
) -> np.ndarray:
    """
    Create a binary image of the robot footprint at a given angle.
    
    The robot is represented as a filled rectangle centered on the canvas.
    This is useful for visualization and collision checking.
    
    Args:
        length: Robot length in pixels
        width: Robot width in pixels
        angle: Rotation angle in degrees (0 = facing right/East)
        canvas_size: Size of output image (default: 2 * diagonal + margin)
    
    Returns:
        Binary image (canvas_size, canvas_size) with robot footprint
        Values: 1 = robot, 0 = empty
    """
    import cv2
    
    # Determine canvas size
    if canvas_size is None:
        diagonal = get_robot_diagonal(length, width)
        canvas_size = int(diagonal * 2.5)
        if canvas_size % 2 == 0:
            canvas_size += 1
    
    # Create canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    center = canvas_size // 2
    
    # Define rectangle corners (before rotation)
    half_l = length / 2
    half_w = width / 2
    
    corners = np.array([
        [-half_l, -half_w],
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w],
    ], dtype=np.float32)
    
    # Rotate corners
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float32)
    
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate to center and convert to integer pixel coords
    pixel_corners = (rotated_corners + center).astype(np.int32)
    
    # Draw filled polygon
    cv2.fillPoly(canvas, [pixel_corners], color=1)
    
    return canvas


def create_rotation_sweep(
    length: int,
    width: int,
    num_angles: int = 36,
    canvas_size: Optional[int] = None
) -> np.ndarray:
    """
    Create visualization of robot rotation sweep.
    
    Shows the circular envelope swept by the robot during rotation.
    This helps visualize why diagonal determines rotation clearance.
    
    Args:
        length: Robot length in pixels
        width: Robot width in pixels
        num_angles: Number of rotation angles to sample
        canvas_size: Canvas size (default: auto)
    
    Returns:
        Grayscale image showing all footprints overlaid
        Values: 0.0 to 1.0 (normalized)
    """
    if canvas_size is None:
        diagonal = get_robot_diagonal(length, width)
        canvas_size = int(diagonal * 2.5)
        if canvas_size % 2 == 0:
            canvas_size += 1
    
    # Accumulate footprints
    combined = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    
    for i in range(num_angles):
        angle = i * (360.0 / num_angles)
        footprint = create_robot_footprint(length, width, angle, canvas_size)
        combined += footprint.astype(np.float32)
    
    # Normalize to [0, 1]
    if combined.max() > 0:
        combined = combined / combined.max()
    
    return combined


def get_size_name(length: int, width: int) -> str:
    """
    Get human-readable name for a robot size.
    
    Args:
        length: Robot length
        width: Robot width
    
    Returns:
        Size name string (e.g., "XS", "M", "XL")
    """
    size_names = {
        (6, 4): "XS",
        (10, 6): "S",
        (14, 9): "M",
        (18, 11): "L",
        (22, 14): "XL",
    }
    return size_names.get((length, width), f"{length}x{width}")


def validate_robot_size(
    length: int,
    width: int,
    resolution: int = 512,
    max_ratio: float = 0.1
) -> Tuple[bool, str]:
    """
    Validate robot size is reasonable for the map resolution.
    
    Checks:
    - Both dimensions are positive
    - Length >= width (by convention)
    - Robot is not too large relative to map
    
    Args:
        length: Robot length in pixels
        width: Robot width in pixels
        resolution: Map resolution
        max_ratio: Maximum allowed diagonal/resolution ratio
    
    Returns:
        Tuple of (is_valid, message)
    """
    if length <= 0 or width <= 0:
        return False, f"Dimensions must be positive: ({length}, {width})"
    
    if length < width:
        return False, f"Length ({length}) should be >= width ({width})"
    
    diagonal = get_robot_diagonal(length, width)
    ratio = diagonal / resolution
    
    if ratio > max_ratio:
        return False, f"Robot too large: diagonal {diagonal:.1f} is {ratio:.1%} of resolution"
    
    return True, f"Valid: diagonal {diagonal:.1f} ({ratio:.1%} of map)"


def visualize_robot_sizes(
    robot_sizes: list,
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create visualization comparing different robot sizes.
    
    Args:
        robot_sizes: List of (length, width) tuples
        output_path: Path to save figure (optional)
        show: Whether to display the figure
    """
    import matplotlib.pyplot as plt
    
    n_sizes = len(robot_sizes)
    fig, axes = plt.subplots(2, n_sizes, figsize=(4 * n_sizes, 8))
    
    for i, (length, width) in enumerate(robot_sizes):
        # Top row: robot footprint
        footprint = create_robot_footprint(length, width, angle=0)
        axes[0, i].imshow(footprint, cmap='Blues')
        axes[0, i].set_title(f"{get_size_name(length, width)}\n{length}×{width}")
        axes[0, i].axis('off')
        
        # Bottom row: rotation sweep
        sweep = create_rotation_sweep(length, width, num_angles=36)
        axes[1, i].imshow(sweep, cmap='Blues')
        diagonal = get_robot_diagonal(length, width)
        axes[1, i].set_title(f"Rotation sweep\nDiag: {diagonal:.1f}px")
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("Footprint", fontsize=12)
    axes[1, 0].set_ylabel("Rotation Sweep", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test robot utilities
    print("Testing robot utilities...")
    
    # Test diagonal calculation
    robot_sizes = [(6, 4), (10, 6), (14, 9), (18, 11), (22, 14)]
    
    print("\nRobot sizes and diagonals:")
    for length, width in robot_sizes:
        diagonal = get_robot_diagonal(length, width)
        kernel_size = get_rotation_kernel_size(length, width, margin=1.2)
        valid, msg = validate_robot_size(length, width)
        
        print(f"  {get_size_name(length, width):3s} ({length:2d}×{width:2d}): "
              f"diagonal={diagonal:5.1f}px, kernel={kernel_size:2d}px - {msg}")
    
    # Test footprint creation
    print("\nCreating footprints...")
    for length, width in robot_sizes:
        footprint = create_robot_footprint(length, width, angle=45)
        assert footprint.shape[0] == footprint.shape[1], "Should be square"
        assert footprint.max() == 1, "Should have robot pixels"
        
    print("✓ All footprints created successfully")
    
    # Test rotation sweep
    print("\nCreating rotation sweeps...")
    for length, width in robot_sizes:
        sweep = create_rotation_sweep(length, width, num_angles=36)
        assert sweep.max() <= 1.0, "Should be normalized"
        
    print("✓ All rotation sweeps created successfully")
    
    # Visualize
    print("\nCreating visualization...")
    visualize_robot_sizes(robot_sizes, show=False)
    print("✓ Visualization complete")
    
    print("\n✓ All robot utility tests passed!")
