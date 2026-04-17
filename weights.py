from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from .MC_network_torch import MCNetwork


# TensorFlow variable prefix -> PyTorch module attribute
TF_TO_TORCH_LAYER: Dict[str, str] = {
    "mc1": "conv1",
    "mc2l1": "conv2_1",
    "mc2l2": "conv2_2",
    "mc4l1": "conv4_1",
    "mc4l2": "conv4_2",
    "mc6l1": "conv6_1",
    "mc6l2": "conv6_2",
    "mc7l1": "conv7_1",
    "mc7l2": "conv7_2",
    "mc9l1": "conv9_1",
    "mc9l2": "conv9_2",
    "mc11l1": "conv11_1",
    "mc11l2": "conv11_2",
    "mc12": "conv12",
    "mc13": "conv13",
}


def _resolve_checkpoint_prefix(
    basketball_data_root: str = "BasketballData",
    metric: str = "psnr",
) -> str:
    root = Path(basketball_data_root)
    if not root.exists():
        raise FileNotFoundError(f"Basketball data folder not found: {root}")

    metric_norm = metric.lower()
    if metric_norm in {"psnr"}:
        token = "psnr"
    elif metric_norm in {"ms-ssim", "msssim", "ms_ssim"}:
        token = "ms-ssim"
    else:
        raise ValueError(
            "metric must be one of: 'psnr', 'ms-ssim', 'msssim', 'ms_ssim'"
        )

    candidates = sorted(root.rglob("model.ckpt.index"))
    filtered = [p for p in candidates if token in str(p).lower()]
    if not filtered:
        raise FileNotFoundError(
            f"No checkpoint index found for metric '{metric}' in {root}"
        )

    # Keep deterministic behavior and prefer the latest path lexicographically.
    index_file = filtered[-1]
    return str(index_file.with_suffix(""))


def _get_tf_checkpoint_reader(checkpoint_prefix: str):
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to read the .ckpt weights. "
            "Install tensorflow and retry."
        ) from exc

    # TF2 path
    if hasattr(tf.train, "load_checkpoint"):
        return tf.train.load_checkpoint(checkpoint_prefix)

    # TF1 fallback
    return tf.compat.v1.train.NewCheckpointReader(checkpoint_prefix)


def _get_tensor(reader, tensor_name: str) -> np.ndarray:
    if hasattr(reader, "get_tensor"):
        return reader.get_tensor(tensor_name)
    return reader.get_tensor(tensor_name)


def load_mc_weights(
    model: MCNetwork,
    checkpoint_prefix: Optional[str] = None,
    basketball_data_root: str = "BasketballData",
    metric: str = "psnr",
    map_location: Union[str, torch.device] = "cpu",
) -> MCNetwork:
    """
    Load MC network weights from TensorFlow checkpoint into a PyTorch MCNetwork.
    """
    if checkpoint_prefix is None:
        checkpoint_prefix = _resolve_checkpoint_prefix(
            basketball_data_root=basketball_data_root,
            metric=metric,
        )

    reader = _get_tf_checkpoint_reader(checkpoint_prefix)

    with torch.no_grad():
        for tf_prefix, torch_layer_name in TF_TO_TORCH_LAYER.items():
            layer = getattr(model, torch_layer_name)

            kernel = _get_tensor(reader, f"{tf_prefix}/kernel")
            # TF Conv2D kernel: [H, W, in_channels, out_channels]
            # Torch Conv2d weight: [out_channels, in_channels, H, W]
            kernel = np.transpose(kernel, (3, 2, 0, 1))
            layer.weight.copy_(
                torch.from_numpy(kernel).to(dtype=layer.weight.dtype, device=map_location)
            )

            bias = _get_tensor(reader, f"{tf_prefix}/bias")
            layer.bias.copy_(
                torch.from_numpy(bias).to(dtype=layer.bias.dtype, device=map_location)
            )

    return model


def build_mc_with_weights(
    checkpoint_prefix: Optional[str] = None,
    basketball_data_root: str = "BasketballData",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> MCNetwork:
    model = MCNetwork().to(device)
    return load_mc_weights(
        model=model,
        checkpoint_prefix=checkpoint_prefix,
        basketball_data_root=basketball_data_root,
        metric=metric,
        map_location=device,
    )
