import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from .motion_torch import MotionNetwork


TF_TO_TORCH_MOTION_LAYER: Dict[str, str] = {
    "conv2d": "conv1",
    "conv2d_1": "conv2",
    "conv2d_2": "conv3",
    "conv2d_3": "conv4",
    "conv2d_4": "conv5",
}


def _normalize_checkpoint_prefix(path_str: str) -> str:
    """Normalize a checkpoint file or prefix into the TensorFlow prefix path."""
    path = Path(path_str)
    if path.suffix == ".index":
        return str(path.with_suffix(""))
    return str(path)


def _resolve_motion_checkpoint_prefix(search_root: str = ".") -> str:
    """Resolve the TensorFlow checkpoint prefix for the optical flow model."""
    env_prefix = os.environ.get("MOTION_CHECKPOINT_PREFIX")
    if env_prefix:
        return _normalize_checkpoint_prefix(env_prefix)

    root = Path(search_root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint search root not found: {root}")

    candidates = sorted(root.rglob("*.index"))
    preferred = [
        path
        for path in candidates
        if path.name in {"model.index", "model.ckpt.index"}
    ]
    for path in reversed(preferred):
        prefix = _normalize_checkpoint_prefix(str(path))
        reader = _get_tf_checkpoint_reader(prefix)
        if _checkpoint_has_motion_weights(reader):
            return prefix

    for path in reversed(candidates):
        prefix = _normalize_checkpoint_prefix(str(path))
        reader = _get_tf_checkpoint_reader(prefix)
        if _checkpoint_has_motion_weights(reader):
            return prefix

    raise FileNotFoundError(
        f"No TensorFlow checkpoint with motion weights found while searching in {root}"
    )


def _get_tf_checkpoint_reader(checkpoint_prefix: str):
    """Create a TensorFlow checkpoint reader."""
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to read the motion checkpoint."
        ) from exc

    if hasattr(tf.train, "load_checkpoint"):
        return tf.train.load_checkpoint(checkpoint_prefix)

    return tf.compat.v1.train.NewCheckpointReader(checkpoint_prefix)


def _checkpoint_has_tensor(reader, tensor_name: str) -> bool:
    """Check whether a tensor exists in the checkpoint reader."""
    if hasattr(reader, "has_tensor"):
        return reader.has_tensor(tensor_name)

    try:
        reader.get_tensor(tensor_name)
        return True
    except Exception:
        return False


def _checkpoint_has_motion_weights(reader) -> bool:
    """Check whether a checkpoint contains motion-network tensors."""
    prefixes = [
        "flow_motion/flow_cnn_0/conv2d/kernel",
        "flow_cnn_0/flow_cnn_0/conv2d/kernel",
        "flow_cnn_0/conv2d/kernel",
    ]
    return any(_checkpoint_has_tensor(reader, prefix) for prefix in prefixes)


def _resolve_tensor_prefix(reader, layer_index: int, tf_layer_name: str) -> str:
    """Resolve the full TensorFlow tensor prefix for one motion layer."""
    prefixes = [
        f"flow_motion/flow_cnn_{layer_index}/{tf_layer_name}",
        f"flow_cnn_{layer_index}/{tf_layer_name}",
    ]
    for prefix in prefixes:
        if _checkpoint_has_tensor(reader, prefix + "/kernel"):
            return prefix

    raise KeyError(
        f"Could not find motion layer '{tf_layer_name}' for block {layer_index}"
    )


def load_motion_weights(
    model: MotionNetwork,
    checkpoint_prefix: Optional[str] = None,
    motion_root: str = ".",
    map_location: Union[str, torch.device] = "cpu",
) -> MotionNetwork:
    """Load TensorFlow optical-flow weights into the PyTorch MotionNetwork."""
    if checkpoint_prefix is None:
        checkpoint_prefix = _resolve_motion_checkpoint_prefix(search_root=motion_root)
    else:
        checkpoint_prefix = _normalize_checkpoint_prefix(checkpoint_prefix)

    reader = _get_tf_checkpoint_reader(checkpoint_prefix)

    with torch.no_grad():
        for block_index, block in enumerate(model.blocks):
            for tf_layer_name, torch_layer_name in TF_TO_TORCH_MOTION_LAYER.items():
                tf_prefix = _resolve_tensor_prefix(reader, block_index, tf_layer_name)
                layer = getattr(block, torch_layer_name)

                kernel = reader.get_tensor(tf_prefix + "/kernel")
                kernel = np.transpose(kernel, (3, 2, 0, 1))
                layer.weight.copy_(
                    torch.from_numpy(kernel).to(
                        dtype=layer.weight.dtype,
                        device=map_location,
                    )
                )

                bias = reader.get_tensor(tf_prefix + "/bias")
                layer.bias.copy_(
                    torch.from_numpy(bias).to(
                        dtype=layer.bias.dtype,
                        device=map_location,
                    )
                )

    return model


def build_motion_with_weights(
    checkpoint_prefix: Optional[str] = None,
    motion_root: str = ".",
    device: Union[str, torch.device] = "cpu",
) -> MotionNetwork:
    """Build a MotionNetwork and load TensorFlow weights into it."""
    model = MotionNetwork().to(device)
    return load_motion_weights(
        model=model,
        checkpoint_prefix=checkpoint_prefix,
        motion_root=motion_root,
        map_location=device,
    )
