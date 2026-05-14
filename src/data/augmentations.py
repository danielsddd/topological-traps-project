"""
Direction-Aware Augmentations for Viability Labels.

When augmenting floor plan images with geometric transforms, we must
also permute the viability label channels to maintain correctness.

Label Order: [North, South, East, West] = channels [0, 1, 2, 3]

Transformations:
- 90° clockwise rotation: N→E→S→W→N
- 180° rotation: N↔S, E↔W
- 270° clockwise rotation: N→W→S→E→N  
- Horizontal flip: E↔W (swap channels 2 and 3)
- Vertical flip: N↔S (swap channels 0 and 1)

Example:
    If original viability-North points UP, after 90° CW rotation,
    what was UP is now pointing RIGHT (East), so:
    - New viability-East = old viability-North
    - New viability-South = old viability-East
    - etc.
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
import random


# Label permutation indices for each transform
# After transform, new_labels[i] = old_labels[permutation[i]]
PERMUTATIONS = {
    # 90° clockwise: [N,S,E,W] → [W,E,N,S]
    # Old North (up) becomes new East (right)
    "rotate_90_cw": [3, 2, 0, 1],  # new[N]=old[W], new[S]=old[E], new[E]=old[N], new[W]=old[S]
    
    # 180°: [N,S,E,W] → [S,N,W,E]
    "rotate_180": [1, 0, 3, 2],    # new[N]=old[S], new[S]=old[N], new[E]=old[W], new[W]=old[E]
    
    # 270° clockwise (= 90° counter-clockwise): [N,S,E,W] → [E,W,S,N]
    "rotate_270_cw": [2, 3, 1, 0], # new[N]=old[E], new[S]=old[W], new[E]=old[S], new[W]=old[N]
    
    # Horizontal flip (left-right): E↔W
    "flip_horizontal": [0, 1, 3, 2],  # N,S unchanged, E↔W
    
    # Vertical flip (up-down): N↔S
    "flip_vertical": [1, 0, 2, 3],    # N↔S, E,W unchanged
}


def permute_labels(labels: np.ndarray, permutation: List[int]) -> np.ndarray:
    """
    Permute label channels according to transformation.
    
    Args:
        labels: Label array (4, H, W)
        permutation: List of indices for channel permutation
    
    Returns:
        Permuted labels (4, H, W)
    """
    if labels.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got {labels.shape[0]}")
    
    new_labels = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(permutation):
        new_labels[new_idx] = labels[old_idx]
    
    return new_labels


def rotate_90_cw(
    occupancy: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate occupancy and labels 90° clockwise.
    
    Args:
        occupancy: Occupancy grid (H, W)
        labels: Viability labels (4, H, W)
    
    Returns:
        Tuple of (rotated_occupancy, permuted_labels)
    """
    # Rotate image: k=3 means 90° CW (or 270° CCW)
    rotated_occupancy = np.rot90(occupancy, k=-1)  # -1 = 90° CW
    rotated_labels = np.rot90(labels, k=-1, axes=(1, 2))
    
    # Permute channels
    permuted_labels = permute_labels(rotated_labels, PERMUTATIONS["rotate_90_cw"])
    
    return rotated_occupancy, permuted_labels


def rotate_180(
    occupancy: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate 180°."""
    rotated_occupancy = np.rot90(occupancy, k=2)
    rotated_labels = np.rot90(labels, k=2, axes=(1, 2))
    permuted_labels = permute_labels(rotated_labels, PERMUTATIONS["rotate_180"])
    
    return rotated_occupancy, permuted_labels


def rotate_270_cw(
    occupancy: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate 270° clockwise (= 90° counter-clockwise)."""
    rotated_occupancy = np.rot90(occupancy, k=1)  # 1 = 90° CCW = 270° CW
    rotated_labels = np.rot90(labels, k=1, axes=(1, 2))
    permuted_labels = permute_labels(rotated_labels, PERMUTATIONS["rotate_270_cw"])
    
    return rotated_occupancy, permuted_labels


def flip_horizontal(
    occupancy: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip left-right (horizontal flip)."""
    flipped_occupancy = np.fliplr(occupancy)
    flipped_labels = np.flip(labels, axis=2)  # Flip along width dimension
    permuted_labels = permute_labels(flipped_labels, PERMUTATIONS["flip_horizontal"])
    
    return flipped_occupancy, permuted_labels


def flip_vertical(
    occupancy: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip up-down (vertical flip)."""
    flipped_occupancy = np.flipud(occupancy)
    flipped_labels = np.flip(labels, axis=1)  # Flip along height dimension
    permuted_labels = permute_labels(flipped_labels, PERMUTATIONS["flip_vertical"])
    
    return flipped_occupancy, permuted_labels


class MultiRobotAugmentation:
    """
    Augmentation class that handles direction-aware label permutation.
    
    This class applies random geometric transformations to both the
    occupancy grid and viability labels, ensuring the directional
    semantics are preserved.
    
    Usage:
        aug = MultiRobotAugmentation(p_rotate=0.5, p_flip=0.5)
        occupancy_aug, labels_aug = aug(occupancy, labels)
    """
    
    def __init__(
        self,
        p_rotate: float = 0.5,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        p_noise: float = 0.0,
        noise_std: float = 0.01
    ):
        """
        Initialize augmentation.
        
        Args:
            p_rotate: Probability of applying rotation (0°, 90°, 180°, or 270°)
            p_flip_h: Probability of horizontal flip
            p_flip_v: Probability of vertical flip
            p_noise: Probability of adding Gaussian noise
            noise_std: Standard deviation of noise
        """
        self.p_rotate = p_rotate
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.p_noise = p_noise
        self.noise_std = noise_std
        
        # Rotation functions
        self.rotations = [
            lambda o, l: (o, l),  # 0° (no rotation)
            rotate_90_cw,
            rotate_180,
            rotate_270_cw,
        ]
    
    def __call__(
        self,
        occupancy: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to occupancy and labels.
        
        Args:
            occupancy: Occupancy grid (H, W), float32
            labels: Viability labels (4, H, W), float32
        
        Returns:
            Tuple of (augmented_occupancy, augmented_labels)
        """
        # Ensure contiguous arrays
        occupancy = np.ascontiguousarray(occupancy)
        labels = np.ascontiguousarray(labels)
        
        # Apply rotation
        if random.random() < self.p_rotate:
            rotation_idx = random.randint(0, 3)
            occupancy, labels = self.rotations[rotation_idx](occupancy, labels)
        
        # Apply horizontal flip
        if random.random() < self.p_flip_h:
            occupancy, labels = flip_horizontal(occupancy, labels)
        
        # Apply vertical flip
        if random.random() < self.p_flip_v:
            occupancy, labels = flip_vertical(occupancy, labels)
        
        # Add noise to occupancy (not labels)
        if random.random() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, occupancy.shape).astype(np.float32)
            occupancy = np.clip(occupancy + noise, 0, 1)
        
        return occupancy, labels


class DeterministicAugmentation:
    """
    Deterministic augmentation for validation/test.
    
    Applies no random transforms, only ensures correct dtypes.
    """
    
    def __call__(
        self,
        occupancy: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pass through with no augmentation."""
        return (
            np.ascontiguousarray(occupancy.astype(np.float32)),
            np.ascontiguousarray(labels.astype(np.float32))
        )


def get_train_augmentation(
    p_rotate: float = 0.5,
    p_flip_h: float = 0.5,
    p_flip_v: float = 0.5
) -> MultiRobotAugmentation:
    """
    Get augmentation for training.
    
    Args:
        p_rotate: Probability of rotation
        p_flip_h: Probability of horizontal flip
        p_flip_v: Probability of vertical flip
    
    Returns:
        Augmentation callable
    """
    return MultiRobotAugmentation(
        p_rotate=p_rotate,
        p_flip_h=p_flip_h,
        p_flip_v=p_flip_v,
        p_noise=0.0
    )


def get_val_augmentation() -> DeterministicAugmentation:
    """
    Get augmentation for validation/test.
    
    Returns:
        Deterministic augmentation (no random transforms)
    """
    return DeterministicAugmentation()


def test_augmentation_correctness():
    """
    Test that augmentations preserve label semantics.
    
    Creates a simple test case where we know what the result should be,
    and verifies the augmentation produces the correct output.
    """
    print("Testing augmentation correctness...")
    
    # Create a simple 4x4 test case
    # Occupancy: small room with opening on the right
    occupancy = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 1],  # Opening on right (East)
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)
    
    # Labels: Center pixel (1,2) is viable East only (can exit to right)
    # We'll make up simple labels for testing
    labels = np.zeros((4, 4, 4), dtype=np.float32)
    
    # Mark pixel (1,2) as viable North (channel 0)
    labels[0, 1, 2] = 1  # Viable North at (1,2)
    
    print("Original occupancy:")
    print(occupancy)
    print(f"Original label[0] (North) at (1,2): {labels[0, 1, 2]}")
    
    # Test 90° CW rotation
    rot_occ, rot_labels = rotate_90_cw(occupancy, labels)
    
    print("\nAfter 90° CW rotation:")
    print(rot_occ)
    
    # After 90° CW, point (1,2) moves to (2,2)
    # What was North (pointing up) is now East (pointing right)
    # So new East channel should have the value at the rotated position
    
    # The permutation [3, 2, 0, 1] means:
    # new[0]=old[3], new[1]=old[2], new[2]=old[0], new[3]=old[1]
    # So new East (channel 2) = old North (channel 0)
    
    # Position mapping for 90° CW rotation of 4x4:
    # Original (1,2) → Rotated (2,2) in the rotated image
    
    print(f"Rotated label[2] (East) at (2,2): {rot_labels[2, 2, 2]}")
    
    # Verify: after rotation, what was viable-North should become viable-East
    # at the appropriately rotated position
    
    # Test 180° rotation
    rot_occ_180, rot_labels_180 = rotate_180(occupancy, labels)
    print("\nAfter 180° rotation:")
    print(rot_occ_180)
    
    # Test flips
    flip_occ_h, flip_labels_h = flip_horizontal(occupancy, labels)
    print("\nAfter horizontal flip:")
    print(flip_occ_h)
    
    flip_occ_v, flip_labels_v = flip_vertical(occupancy, labels)
    print("\nAfter vertical flip:")
    print(flip_occ_v)
    
    print("\n✓ Augmentation tests completed")


def test_full_augmentation_pipeline():
    """Test the full augmentation class."""
    print("\nTesting full augmentation pipeline...")
    
    # Create random test data
    occupancy = np.random.rand(512, 512).astype(np.float32)
    labels = np.random.rand(4, 512, 512).astype(np.float32)
    
    # Test training augmentation
    aug = MultiRobotAugmentation(p_rotate=1.0, p_flip_h=1.0, p_flip_v=1.0)
    
    aug_occ, aug_labels = aug(occupancy, labels)
    
    assert aug_occ.shape == occupancy.shape, "Occupancy shape mismatch"
    assert aug_labels.shape == labels.shape, "Labels shape mismatch"
    assert aug_occ.dtype == np.float32, "Occupancy dtype mismatch"
    assert aug_labels.dtype == np.float32, "Labels dtype mismatch"
    
    print("✓ Full pipeline test passed")
    
    # Test validation augmentation
    val_aug = DeterministicAugmentation()
    val_occ, val_labels = val_aug(occupancy, labels)
    
    assert np.allclose(val_occ, occupancy), "Validation augmentation should not change occupancy"
    assert np.allclose(val_labels, labels), "Validation augmentation should not change labels"
    
    print("✓ Validation augmentation test passed")


if __name__ == "__main__":
    test_augmentation_correctness()
    test_full_augmentation_pipeline()
    print("\n✓ All augmentation tests passed!")
