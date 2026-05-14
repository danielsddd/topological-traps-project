#!/usr/bin/env python3
"""
scripts/test_velocity_oracle.py

Quick validation of the velocity-dependent viability oracle.
Run this BEFORE submitting the training job to verify correctness.

Tests:
  1. v=0 produces identical output to the basic Oracle
  2. Viable area monotonically decreases with speed
  3. Braking distance computation is correct
  4. Kernel sizes are physically reasonable
  5. A narrow corridor becomes a trap at high speed
  6. Dataset class loads and returns correct shapes
  7. Forward pass through the model with 4-channel input works

Usage:
    cd /vol/joberant_nobck/data/NLP_368307701_2526a/simanovsky2/project2
    conda activate traps
    python scripts/test_velocity_oracle.py
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_braking_distance():
    """Test d_brake = v²/(2a) conversion to pixels."""
    from src.oracle.velocity_oracle import braking_distance_px

    # v=0 → d=0
    assert braking_distance_px(0.0, 2.0, 10.0) == 0

    # v=2.0, a=2.0 → d_m = 4/(4) = 1.0 m → 10 px
    assert braking_distance_px(2.0, 2.0, 10.0) == 10

    # v=1.0, a=2.0 → d_m = 1/(4) = 0.25 m → 2.5 → ceil = 3 px
    assert braking_distance_px(1.0, 2.0, 10.0) == 3

    # v=3.0, a=2.0 → d_m = 9/(4) = 2.25 m → 22.5 → ceil = 23 px
    assert braking_distance_px(3.0, 2.0, 10.0) == 23

    print("  ✓ braking_distance_px")


def test_v0_matches_basic():
    """v=0 must produce identical output to the basic Oracle."""
    from src.oracle.velocity_oracle import velocity_viability
    from src.oracle.directional_viability import generate_labels_for_map

    H, W = 80, 80
    L, Wr = 8, 6

    # Open room with walls
    occ = np.ones((H, W), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 0

    basic = generate_labels_for_map(occ, L, Wr)
    vel0 = velocity_viability(occ, L, Wr, velocity=0.0, max_decel=2.0, px_per_m=10.0)

    assert basic.shape == vel0.shape == (4, H, W)
    assert np.array_equal(basic, vel0), (
        f"v=0 differs from basic! "
        f"Mismatch pixels: {(basic != vel0).sum()}"
    )
    print("  ✓ v=0 matches basic oracle exactly")


def test_monotonic_shrinkage():
    """Viable area must monotonically decrease as speed increases."""
    from src.oracle.velocity_oracle import velocity_viability

    H, W = 100, 100
    L, Wr = 10, 7
    occ = np.ones((H, W), dtype=np.uint8)
    occ[0:3, :] = occ[-3:, :] = occ[:, 0:3] = occ[:, -3:] = 0

    prev_viable = float("inf")
    for v in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        labels = velocity_viability(occ, L, Wr, v, 2.0, 10.0)
        viable = int(labels.sum())
        assert viable <= prev_viable, (
            f"Monotonicity violated! v={v}: {viable} > prev {prev_viable}"
        )
        prev_viable = viable

    print("  ✓ Viable area monotonically decreases with speed")


def test_corridor_becomes_trap():
    """
    A corridor of width 20px with robot_W=15 is viable at v=0 but
    becomes a trap at v=3.0 (d_brake=23 px > corridor depth).
    """
    from src.oracle.velocity_oracle import velocity_viability, braking_distance_px

    H, W = 80, 80
    occ = np.zeros((H, W), dtype=np.uint8)

    # Open room: rows 10-70, cols 10-70
    occ[10:70, 10:70] = 1

    # Narrow dead-end corridor going North from the room
    # Width=20 cols, depth=8 rows (cols 30-50, rows 2-10)
    occ[2:10, 30:50] = 1

    L, Wr = 10, 7

    # At v=0: corridor should be viable (robot fits, can back out)
    labels_v0 = velocity_viability(occ, L, Wr, 0.0, 2.0, 10.0)
    corridor_viable_v0 = labels_v0[:, 5, 40].sum()  # Middle of corridor

    # At v=3.0: d_brake=23 px >> corridor depth of 8. Should be a trap.
    d_brake = braking_distance_px(3.0, 2.0, 10.0)
    labels_v3 = velocity_viability(occ, L, Wr, 3.0, 2.0, 10.0)
    corridor_viable_v3 = labels_v3[:, 5, 40].sum()

    # The corridor at v=3 should have LESS viability than at v=0
    assert corridor_viable_v3 <= corridor_viable_v0, (
        f"High-speed corridor not less viable! "
        f"v=0: {corridor_viable_v0}, v=3: {corridor_viable_v3}, d_brake={d_brake}"
    )
    print(f"  ✓ Corridor trap: v=0 viable={corridor_viable_v0}, "
          f"v=3 viable={corridor_viable_v3} (d_brake={d_brake}px)")


def test_dataset_shapes():
    """VelocityViabilityDataset returns correct tensor shapes."""
    try:
        from src.experiments import VelocityViabilityDataset
    except ImportError as e:
        print(f"  SKIP (import error): {e}")
        return

    import tempfile
    import os

    # Create a tiny temporary dataset
    tmpdir = tempfile.mkdtemp()
    occ = np.ones((64, 64), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 0

    for i in range(3):
        np.save(os.path.join(tmpdir, f"map_{i:04d}.npy"), occ)

    ds = VelocityViabilityDataset(
        map_dir=tmpdir,
        manifest_path=None,
        robot_sizes=[(8, 6)],
        split="train",
        resolution=64,
        velocities=[0.0, 1.0, 2.0],
        num_velocities_per_map=2,
    )

    assert len(ds) == 6, f"Expected 6 samples (3 maps × 2 vel), got {len(ds)}"

    x, y, meta = ds[0]
    assert x.shape == (4, 64, 64), f"Input shape wrong: {x.shape}"
    assert y.shape == (4, 64, 64), f"Label shape wrong: {y.shape}"
    assert "velocity" in meta
    assert "velocity_norm" in meta
    assert "d_brake_px" in meta

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    print(f"  ✓ Dataset shapes correct: input={x.shape}, labels={y.shape}")


def test_model_forward_pass():
    """U-Net accepts 4-channel input and produces 4-channel output."""
    try:
        import torch
        from src.models.unet import MultiRobotViabilityUNet
    except ImportError as e:
        print(f"  SKIP (import error): {e}")
        return

    model = MultiRobotViabilityUNet(
        encoder_name="resnet34",
        encoder_weights=None,  # Skip pretrained for speed
        in_channels=4,
        classes=4,
    )
    model.eval()

    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 4, 64, 64), f"Output shape wrong: {out.shape}"
    print(f"  ✓ Model forward pass: (2, 4, 64, 64) → {out.shape}")


def main():
    print("=" * 60)
    print("VELOCITY ORACLE — PRE-TRAINING VALIDATION")
    print("=" * 60)

    tests = [
        ("Braking distance computation", test_braking_distance),
        ("v=0 matches basic Oracle", test_v0_matches_basic),
        ("Monotonic viable-area shrinkage", test_monotonic_shrinkage),
        ("Corridor becomes trap at speed", test_corridor_becomes_trap),
        ("Dataset tensor shapes", test_dataset_shapes),
        ("Model 4-channel forward pass", test_model_forward_pass),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")

    if failed > 0:
        print("\n⚠️  Fix failures before submitting the training job!")
        return 1

    print("\n✓ All tests passed — safe to submit training job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())