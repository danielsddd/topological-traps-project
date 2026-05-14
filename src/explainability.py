"""
Explainability — Grad-CAM saliency maps for the viability U-Net.

Implements Direction 4 of the project's research plan: producing
heatmaps that highlight which environmental features cause the model
to predict a "trap" (low viability) at a particular pixel and
direction.

Why Grad-CAM (and not vanilla saliency / Integrated Gradients)?
    Grad-CAM works on convolutional feature maps, gives spatially
    coherent class-discriminative explanations, and is cheap (one
    forward + one backward pass). For dense-prediction networks like
    a U-Net the standard recipe is:
        1. Pick a target conv layer ℓ (typically the encoder bottleneck
           or the last decoder block before the segmentation head).
        2. Define a scalar target — for segmentation we take a region
           of interest (RoI) of pixels for a chosen output channel
           (direction) and either sum the logits there or sum
           (1 − sigmoid(logits)) to highlight features causing TRAPS.
        3. Compute ∂target/∂A_ℓ for activations A_ℓ.
        4. Spatial-mean the gradients to get per-channel weights α_k.
        5. Heatmap = ReLU( Σ_k α_k · A_ℓ ), upsampled to input size.

This module exposes one function — `generate_saliency_map` — and a
small CLI for sanity-checking on a saved checkpoint and a single map.

The implementation is robust to:
    - The U-Net being wrapped inside `MultiRobotViabilityUNet.model`
      (it walks both the wrapper and the underlying smp.Unet).
    - Targets that ask for "trap features" (causes of low viability)
      and "viable features" (causes of high viability).
    - Single-channel models (continuous-angle / angle-cost-map modes)
      and multi-channel models (basic / cost-map modes).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer lookup
# ---------------------------------------------------------------------------

def list_conv_layers(model: nn.Module, max_items: int = 30) -> List[str]:
    """
    List candidate conv layers by qualified name.

    Useful when you don't know which `target_layer_name` to pass.

    Args:
        model:     The PyTorch model.
        max_items: Cap on number of names returned.

    Returns:
        List of dotted layer names that are nn.Conv2d.
    """
    names = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            names.append(n)
        if len(names) >= max_items:
            break
    return names


def _resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """
    Resolve a dotted module name (e.g. 'model.encoder.layer4.1.conv2').

    Args:
        model:       Root model.
        dotted_name: Dotted attribute path.

    Returns:
        The nn.Module at that path.

    Raises:
        AttributeError: if any part of the path is missing.
    """
    obj = model
    for part in dotted_name.split("."):
        if not hasattr(obj, part):
            # Try integer index for nn.Sequential / nn.ModuleList
            try:
                idx = int(part)
                obj = obj[idx]  # type: ignore[index]
                continue
            except (ValueError, TypeError):
                pass
            avail = [n for n, _ in obj.named_children()] if hasattr(obj, "named_children") else []
            raise AttributeError(
                f"Could not resolve '{dotted_name}': missing '{part}'. "
                f"Available children at this level: {avail}"
            )
        obj = getattr(obj, part)
    return obj


def _default_target_layer(model: nn.Module) -> str:
    """
    Pick a sensible default conv layer if the user did not specify one.

    For an SMP U-Net wrapped in MultiRobotViabilityUNet the encoder's
    last block tends to give the cleanest, most class-discriminative
    Grad-CAMs. We try a few well-known names in order.
    """
    candidates = [
        "model.encoder.layer4",      # ResNet34 last block
        "model.encoder.layer3",
        "model.decoder.blocks.0",    # SMP U-Net first decoder block (deepest)
        "model.encoder",             # last resort
    ]
    for c in candidates:
        try:
            _resolve_module(model, c)
            return c
        except AttributeError:
            continue

    # Fallback: the very last conv layer in the model
    last_conv_name = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv_name = n
    if last_conv_name is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last_conv_name


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Hook-based Grad-CAM extractor.

    Lifecycle:
        cam = GradCAM(model, target_layer_module)
        with cam:
            heatmap = cam.compute(input_tensor, target_scalar_fn)
        # hooks are removed on __exit__

    The class is single-use per __enter__; you can re-enter it for
    multiple inputs as long as you do not change the layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    # ---------- hook plumbing ----------

    def __enter__(self) -> "GradCAM":
        def fwd_hook(_m, _inp, out):
            # Detach is wrong here — we need the graph for backward.
            self._activations = out

        def bwd_hook(_m, _grad_in, grad_out):
            # grad_out is a tuple; we want the first element.
            self._gradients = grad_out[0]

        self._fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        # full_backward_hook is the modern (PyTorch ≥ 1.8) replacement
        # for register_backward_hook — needed for layers with multiple
        # outputs / inputs (residual blocks, decoder skip-connection
        # blocks). The signature matches.
        self._bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
        self._fwd_handle = None
        self._bwd_handle = None

    # ---------- main computation ----------

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_channel: int = 0,
        roi_mask: Optional[torch.Tensor] = None,
        explain: str = "trap",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a forward + backward pass and return the upsampled heatmap.

        Args:
            input_tensor:   (B, C, H, W) — must already be on the model's device.
                            B can be 1 (typical) or larger; output mirrors B.
            target_channel: Which output channel (direction) to explain.
                            For multi-channel models 0..C_out-1.
            roi_mask:       Optional (B, 1, H, W) or (H, W) binary mask
                            restricting the explained region. If None,
                            the entire output map is summed.
            explain:        Either:
                              "trap"   — explain causes of low viability
                                         (gradient w.r.t. -logit).
                              "viable" — explain causes of high viability
                                         (gradient w.r.t. +logit).

        Returns:
            (heatmaps, logits)
              heatmaps: (B, H, W) float32 in [0, 1]
              logits:   (B, C_out, H, W) float32 — the model output at the
                        time of explanation, on CPU as numpy (handy for
                        overlay plots).
        """
        if input_tensor.dim() != 4:
            raise ValueError(f"input_tensor must be 4-D (B,C,H,W); got {tuple(input_tensor.shape)}")
        if explain not in ("trap", "viable"):
            raise ValueError("explain must be 'trap' or 'viable'")

        self.model.eval()  # keep BatchNorm fixed
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)  # (B, C_out, H, W)
        if logits.dim() != 4:
            raise RuntimeError(
                f"Expected (B, C, H, W) logits; got {tuple(logits.shape)}"
            )
        B, C_out, H_out, W_out = logits.shape
        if not (0 <= target_channel < C_out):
            raise ValueError(f"target_channel={target_channel} out of range [0, {C_out})")

        # Build scalar target
        sign = -1.0 if explain == "trap" else 1.0
        target_logits = logits[:, target_channel, :, :]  # (B, H, W)

        if roi_mask is not None:
            if roi_mask.dim() == 2:
                roi_mask = roi_mask.unsqueeze(0).unsqueeze(0)
            if roi_mask.dim() == 3:
                roi_mask = roi_mask.unsqueeze(1)
            roi_mask = roi_mask.to(target_logits.device).float()
            if roi_mask.shape[-2:] != (H_out, W_out):
                roi_mask = F.interpolate(roi_mask, size=(H_out, W_out), mode="nearest")
            roi_mask = roi_mask.squeeze(1)  # (B, H, W)
            scalar = (target_logits * roi_mask).sum()
        else:
            scalar = target_logits.sum()

        scalar = scalar * sign

        # Backward
        self.model.zero_grad(set_to_none=True)
        scalar.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks captured no activations/gradients. "
                "Did the target layer participate in the forward pass?"
            )

        A = self._activations             # (B, K, h, w)
        dA = self._gradients              # (B, K, h, w)

        # Channel weights = global-average-pool of gradients
        weights = dA.mean(dim=(2, 3), keepdim=True)        # (B, K, 1, 1)
        cam = (weights * A).sum(dim=1, keepdim=True)       # (B, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input spatial size
        H_in, W_in = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = F.interpolate(cam, size=(H_in, W_in), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)                                # (B, H, W)

        # Per-sample [0, 1] normalisation
        out = []
        cam_np = cam.detach().cpu().numpy()
        for b in range(cam_np.shape[0]):
            x = cam_np[b]
            x_min, x_max = float(x.min()), float(x.max())
            if x_max - x_min > 1e-8:
                x = (x - x_min) / (x_max - x_min)
            else:
                x = np.zeros_like(x)
            out.append(x.astype(np.float32))
        heatmaps = np.stack(out, axis=0)  # (B, H, W)

        return heatmaps, logits.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Convenience wrapper used by other scripts and the CLI
# ---------------------------------------------------------------------------

def generate_saliency_map(
    model: nn.Module,
    input_batch: torch.Tensor,
    target_layer_name: Optional[str] = None,
    target_channel: int = 0,
    roi_mask: Optional[torch.Tensor] = None,
    explain: str = "trap",
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    One-call Grad-CAM for the viability U-Net.

    Args:
        model:             A trained model. Either MultiRobotViabilityUNet or
                           any nn.Module with (B, C, H, W) input and (B, C', H, W)
                           output.
        input_batch:       (B, C, H, W) tensor. CPU or GPU.
        target_layer_name: Dotted name of a Conv2d layer. If None, an
                           encoder-default is chosen automatically.
        target_channel:    Output channel (direction) to explain. 0..C'-1.
        roi_mask:          Optional (B, 1, H, W) mask to restrict the explained
                           region (e.g. only pixels predicted as traps).
        explain:           "trap" (default) or "viable".
        device:            Device override; if None, uses the model's device.

    Returns:
        Dict with:
            "heatmap"          : (B, H, W) float32 in [0, 1]
            "logits"           : (B, C', H, W) float32
            "target_layer"     : str — the layer name used
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    input_batch = input_batch.to(device)

    if target_layer_name is None:
        target_layer_name = _default_target_layer(model)
    target_layer = _resolve_module(model, target_layer_name)

    with GradCAM(model, target_layer) as cam:
        heatmaps, logits = cam.compute(
            input_batch,
            target_channel=target_channel,
            roi_mask=roi_mask,
            explain=explain,
        )

    return {
        "heatmap": heatmaps,
        "logits": logits,
        "target_layer": target_layer_name,
    }


# ---------------------------------------------------------------------------
# Visual overlay — for the report and the CLI demo
# ---------------------------------------------------------------------------

def overlay_heatmap_on_map(
    occupancy: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    cmap_name: str = "jet",
) -> np.ndarray:
    """
    Composite a [0, 1] heatmap on top of an occupancy grid.

    Args:
        occupancy: (H, W) — values in {0, 1}: 1 = free, 0 = obstacle.
        heatmap:   (H, W) — values in [0, 1].
        alpha:     Heatmap opacity.
        cmap_name: matplotlib colormap.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    if occupancy.shape != heatmap.shape:
        raise ValueError(
            f"shape mismatch: occupancy {occupancy.shape} vs heatmap {heatmap.shape}"
        )

    base = np.dstack([occupancy.astype(np.float32)] * 3)
    base = np.clip(base, 0.0, 1.0)
    # Modern API (matplotlib >= 3.6); fall back for older versions
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap(cmap_name)
    except (AttributeError, ImportError):
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
    heat_rgba = cmap(np.clip(heatmap.astype(np.float32), 0.0, 1.0))
    heat_rgb = heat_rgba[..., :3]

    out = (1.0 - alpha) * base + alpha * heat_rgb
    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _build_input_basic(
    occupancy: np.ndarray,
    L: int,
    W: int,
    resolution: int = 512,
) -> torch.Tensor:
    """3-channel input for the basic / cost-map model."""
    H, Wd = occupancy.shape
    x = np.zeros((1, 3, H, Wd), dtype=np.float32)
    x[0, 0] = occupancy.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)
    return torch.from_numpy(x)


def _build_input_angle(
    occupancy: np.ndarray,
    L: int,
    W: int,
    angle_deg: float,
    resolution: int = 512,
) -> torch.Tensor:
    """5-channel input for continuous-angle / angle-cost-map model."""
    # Lazy absolute import: keeps this file usable both as a module and
    # as a stand-alone CLI script invoked from the project root.
    from src.oracle.extended_oracles import angle_to_sincos  # type: ignore

    H, Wd = occupancy.shape
    s, c = angle_to_sincos(angle_deg)
    x = np.zeros((1, 5, H, Wd), dtype=np.float32)
    x[0, 0] = occupancy.astype(np.float32)
    x[0, 1] = float(L) / float(resolution)
    x[0, 2] = float(W) / float(resolution)
    x[0, 3] = s
    x[0, 4] = c
    return torch.from_numpy(x)


def _cli_main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM saliency demo for the viability U-Net."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pth checkpoint.")
    parser.add_argument("--map", type=str, required=True,
                        help="Path to a (H,W) .npy occupancy grid.")
    parser.add_argument("--robot-length", type=int, default=30)
    parser.add_argument("--robot-width", type=int, default=20)
    parser.add_argument("--target-channel", type=int, default=0,
                        help="Output channel (direction) to explain. "
                             "0=N, 1=S, 2=E, 3=W for basic mode.")
    parser.add_argument("--target-layer", type=str, default=None,
                        help="Dotted layer name, e.g. 'model.encoder.layer4'. "
                             "If omitted, an encoder-default is auto-picked.")
    parser.add_argument("--explain", choices=["trap", "viable"], default="trap")
    parser.add_argument("--angle", type=float, default=None,
                        help="If given, use 5-channel angle-conditioned input.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/explainability/saliency.png")
    parser.add_argument("--list-layers", action="store_true",
                        help="Print candidate conv layer names and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Late imports so the file remains import-light on the cluster
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.models.unet import MultiRobotViabilityUNet  # type: ignore

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load model -------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        model = MultiRobotViabilityUNet(**ckpt["config"]).to(device)
    else:
        # Best-effort fallback for legacy checkpoints
        is_angle = args.angle is not None
        in_ch = 5 if is_angle else 3
        out_ch = 1 if is_angle else 4
        logger.warning("No 'config' in checkpoint; defaulting in=%d out=%d", in_ch, out_ch)
        model = MultiRobotViabilityUNet(in_channels=in_ch, classes=out_ch).to(device)
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_dict_key])
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    if args.list_layers:
        for n in list_conv_layers(model, max_items=60):
            print(n)
        return

    # --- Load map ---------------------------------------------------------
    occ = np.load(args.map).astype(np.uint8)
    if occ.ndim != 2:
        raise ValueError(f"Expected 2-D occupancy grid, got {occ.shape}")
    logger.info("Map: %s shape=%s free=%.1f%%", args.map, occ.shape, occ.mean() * 100)

    # --- Build input ------------------------------------------------------
    if args.angle is not None:
        # Local helper using src.oracle.extended_oracles
        from src.oracle.extended_oracles import angle_to_sincos
        H, Wd = occ.shape
        s, c = angle_to_sincos(args.angle)
        inp = np.zeros((1, 5, H, Wd), dtype=np.float32)
        inp[0, 0] = occ.astype(np.float32)
        inp[0, 1] = float(args.robot_length) / 512.0
        inp[0, 2] = float(args.robot_width) / 512.0
        inp[0, 3] = s
        inp[0, 4] = c
        inp_t = torch.from_numpy(inp).to(device)
    else:
        inp_t = _build_input_basic(occ, args.robot_length, args.robot_width).to(device)

    # --- Run Grad-CAM -----------------------------------------------------
    result = generate_saliency_map(
        model=model,
        input_batch=inp_t,
        target_layer_name=args.target_layer,
        target_channel=args.target_channel,
        roi_mask=None,
        explain=args.explain,
        device=device,
    )
    heatmap = result["heatmap"][0]
    logits = result["logits"][0]
    via_pred = 1.0 / (1.0 + np.exp(-logits[args.target_channel]))
    logger.info("Heatmap range [%.3f, %.3f]; predicted viability mean=%.3f",
                heatmap.min(), heatmap.max(), via_pred.mean())
    logger.info("Used target layer: %s", result["target_layer"])

    # --- Visualise --------------------------------------------------------
    import matplotlib.pyplot as plt
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(occ, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Occupancy")
    axes[0].axis("off")

    axes[1].imshow(via_pred, cmap="viridis", vmin=0, vmax=1)
    title = f"Predicted viability (ch {args.target_channel}"
    if args.angle is not None:
        title += f", θ={args.angle:.0f}°"
    title += ")"
    axes[1].set_title(title)
    axes[1].axis("off")

    overlay = overlay_heatmap_on_map(occ.astype(np.float32), heatmap, alpha=0.55)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM ({args.explain})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    _cli_main()