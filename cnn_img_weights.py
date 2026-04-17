import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy import fftpack

from .CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis


def _normalize_checkpoint_prefix(path_str: str) -> str:
    path = Path(path_str)
    if path.suffix == ".index":
        return str(path.with_suffix(""))
    return str(path)


def _resolve_checkpoint_prefix(
    search_root: str = ".",
    metric: str = "psnr",
) -> str:
    env_prefix = os.environ.get("OPENDVC_CHECKPOINT_PREFIX")
    if env_prefix:
        return _normalize_checkpoint_prefix(env_prefix)

    root = Path(search_root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint search root not found: {root}")

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
    matches = [path for path in candidates if token in str(path).lower()]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for metric '{metric}' while searching in {root}"
        )
    return _normalize_checkpoint_prefix(str(matches[-1]))


def _get_tf_checkpoint_reader(checkpoint_prefix: str):
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to read the OpenDVC checkpoint."
        ) from exc

    if hasattr(tf.train, "load_checkpoint"):
        return tf.train.load_checkpoint(checkpoint_prefix)
    return tf.compat.v1.train.NewCheckpointReader(checkpoint_prefix)


def _checkpoint_has_tensor(reader, tensor_name: str) -> bool:
    if hasattr(reader, "has_tensor"):
        return reader.has_tensor(tensor_name)

    try:
        reader.get_tensor(tensor_name)
        return True
    except Exception:
        return False


def _irdft_matrix(shape) -> np.ndarray:
    shape = tuple(int(s) for s in shape)
    size = int(np.prod(shape))
    rank = len(shape)
    matrix = np.identity(size, dtype=np.float64).reshape((size,) + shape)
    for axis in range(rank):
        matrix = fftpack.rfft(matrix, axis=axis + 1)
        slices = (rank + 1) * [slice(None)]
        if shape[axis] % 2 == 1:
            slices[axis + 1] = slice(1, None)
        else:
            slices[axis + 1] = slice(1, -1)
        matrix[tuple(slices)] *= np.sqrt(2.0)
    matrix /= np.sqrt(size)
    return np.reshape(matrix, (size, size))


def _reconstruct_kernel_from_rdft(
    kernel_rdft: np.ndarray,
    kernel_shape: Tuple[int, int, int, int],
) -> np.ndarray:
    spatial_shape = kernel_shape[:2]
    matrix = _irdft_matrix(spatial_shape).astype(kernel_rdft.dtype, copy=False)
    kernel = np.matmul(matrix, kernel_rdft)
    return np.reshape(kernel, kernel_shape)


def _infer_mv_dims(reader) -> Tuple[int, int]:
    shape_map = reader.get_variable_to_shape_map()
    analysis_bias_name = "MV_analysis/layer_0/signal_conv2d/bias"
    if analysis_bias_name not in shape_map:
        analysis_bias_name = "analysis/layer_0/signal_conv2d/bias"

    analysis_kernel_name = "MV_analysis/layer_3/signal_conv2d/kernel_rdft"
    if analysis_kernel_name not in shape_map:
        analysis_kernel_name = "analysis/layer_3/signal_conv2d/kernel_rdft"

    n = int(shape_map[analysis_bias_name][0])
    rdft_shape = shape_map[analysis_kernel_name]
    m = int(rdft_shape[1] // n)
    return n, m


def _load_signal_conv_block(
    reader,
    layer,
    tf_prefix: str,
    map_location: Union[str, torch.device],
) -> None:
    op = layer.op
    kernel_rdft = reader.get_tensor(tf_prefix + "/kernel_rdft")

    kernel_h, kernel_w = layer.kernel_size
    if layer.corr:
        in_channels = op.in_channels
        out_channels = op.out_channels
    else:
        in_channels = op.in_channels
        out_channels = op.out_channels

    kernel = _reconstruct_kernel_from_rdft(
        kernel_rdft,
        (kernel_h, kernel_w, in_channels, out_channels),
    )

    if layer.corr:
        kernel = np.transpose(kernel, (3, 2, 0, 1))
    else:
        kernel = np.transpose(kernel, (2, 3, 0, 1))

    op.weight.copy_(
        torch.from_numpy(kernel).to(dtype=op.weight.dtype, device=map_location)
    )

    if op.bias is not None:
        bias = reader.get_tensor(tf_prefix + "/bias")
        op.bias.copy_(
            torch.from_numpy(bias).to(dtype=op.bias.dtype, device=map_location)
        )

    if layer.activation is not None:
        gdn = layer.activation
        beta = reader.get_tensor(tf_prefix + "/gdn/reparam_beta")
        gamma = reader.get_tensor(tf_prefix + "/gdn/reparam_gamma")

        gdn.beta_reparam.copy_(
            torch.from_numpy(beta).to(
                dtype=gdn.beta_reparam.dtype,
                device=map_location,
            )
        )
        gdn.gamma_reparam.copy_(
            torch.from_numpy(gamma).to(
                dtype=gdn.gamma_reparam.dtype,
                device=map_location,
            )
        )


def load_mv_analysis_weights(
    model: MVAnalysis,
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    map_location: Union[str, torch.device] = "cpu",
) -> MVAnalysis:
    if checkpoint_prefix is None:
        checkpoint_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        checkpoint_prefix = _normalize_checkpoint_prefix(checkpoint_prefix)

    reader = _get_tf_checkpoint_reader(checkpoint_prefix)

    with torch.no_grad():
        for layer_index in range(4):
            layer = getattr(model, f"layer_{layer_index}")
            tf_prefix = f"MV_analysis/layer_{layer_index}/signal_conv2d"
            if not _checkpoint_has_tensor(reader, tf_prefix + "/kernel_rdft"):
                tf_prefix = f"analysis/layer_{layer_index}/signal_conv2d"
            _load_signal_conv_block(reader, layer, tf_prefix, map_location)

    return model


def load_mv_synthesis_weights(
    model: MVSynthesis,
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    map_location: Union[str, torch.device] = "cpu",
) -> MVSynthesis:
    if checkpoint_prefix is None:
        checkpoint_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        checkpoint_prefix = _normalize_checkpoint_prefix(checkpoint_prefix)

    reader = _get_tf_checkpoint_reader(checkpoint_prefix)

    with torch.no_grad():
        for layer_index in range(4):
            layer = getattr(model, f"layer_{layer_index}")
            tf_prefix = f"MV_synthesis/layer_{layer_index}/signal_conv2d"
            if not _checkpoint_has_tensor(reader, tf_prefix + "/kernel_rdft"):
                tf_prefix = f"synthesis/layer_{layer_index}/signal_conv2d"
            _load_signal_conv_block(reader, layer, tf_prefix, map_location)

    return model


def build_mv_analysis_with_weights(
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> MVAnalysis:
    resolved_prefix = checkpoint_prefix
    if resolved_prefix is None:
        resolved_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        resolved_prefix = _normalize_checkpoint_prefix(resolved_prefix)

    reader = _get_tf_checkpoint_reader(resolved_prefix)
    n, m = _infer_mv_dims(reader)
    model = MVAnalysis(num_filters=n, M=m).to(device)
    return load_mv_analysis_weights(
        model=model,
        checkpoint_prefix=resolved_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        map_location=device,
    )


def build_mv_synthesis_with_weights(
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> MVSynthesis:
    resolved_prefix = checkpoint_prefix
    if resolved_prefix is None:
        resolved_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        resolved_prefix = _normalize_checkpoint_prefix(resolved_prefix)

    reader = _get_tf_checkpoint_reader(resolved_prefix)
    n, m = _infer_mv_dims(reader)
    model = MVSynthesis(num_filters=n, M=m).to(device)
    return load_mv_synthesis_weights(
        model=model,
        checkpoint_prefix=resolved_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        map_location=device,
    )


def load_res_analysis_weights(
    model: ResAnalysis,
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    map_location: Union[str, torch.device] = "cpu",
) -> ResAnalysis:
    if checkpoint_prefix is None:
        checkpoint_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        checkpoint_prefix = _normalize_checkpoint_prefix(checkpoint_prefix)

    reader = _get_tf_checkpoint_reader(checkpoint_prefix)

    with torch.no_grad():
        for layer_index in range(4):
            layer = getattr(model, f"layer_{layer_index}")
            tf_prefix = f"analysis/layer_{layer_index}/signal_conv2d"
            _load_signal_conv_block(reader, layer, tf_prefix, map_location)

    return model


def load_res_synthesis_weights(
    model: ResSynthesis,
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    map_location: Union[str, torch.device] = "cpu",
) -> ResSynthesis:
    if checkpoint_prefix is None:
        checkpoint_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        checkpoint_prefix = _normalize_checkpoint_prefix(checkpoint_prefix)

    reader = _get_tf_checkpoint_reader(checkpoint_prefix)

    with torch.no_grad():
        for layer_index in range(4):
            layer = getattr(model, f"layer_{layer_index}")
            tf_prefix = f"synthesis/layer_{layer_index}/signal_conv2d"
            _load_signal_conv_block(reader, layer, tf_prefix, map_location)

    return model


def build_res_analysis_with_weights(
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> ResAnalysis:
    resolved_prefix = checkpoint_prefix
    if resolved_prefix is None:
        resolved_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        resolved_prefix = _normalize_checkpoint_prefix(resolved_prefix)

    reader = _get_tf_checkpoint_reader(resolved_prefix)
    n, m = _infer_mv_dims(reader)
    model = ResAnalysis(num_filters=n, M=m).to(device)
    return load_res_analysis_weights(
        model=model,
        checkpoint_prefix=resolved_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        map_location=device,
    )


def build_res_synthesis_with_weights(
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = ".",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> ResSynthesis:
    resolved_prefix = checkpoint_prefix
    if resolved_prefix is None:
        resolved_prefix = _resolve_checkpoint_prefix(
            search_root=open_dvc_root,
            metric=metric,
        )
    else:
        resolved_prefix = _normalize_checkpoint_prefix(resolved_prefix)

    reader = _get_tf_checkpoint_reader(resolved_prefix)
    n, m = _infer_mv_dims(reader)
    model = ResSynthesis(num_filters=n, M=m).to(device)
    return load_res_synthesis_weights(
        model=model,
        checkpoint_prefix=resolved_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        map_location=device,
    )


__all__ = [
    "load_mv_analysis_weights",
    "load_mv_synthesis_weights",
    "build_mv_analysis_with_weights",
    "build_mv_synthesis_with_weights",
    "load_res_analysis_weights",
    "load_res_synthesis_weights",
    "build_res_analysis_with_weights",
    "build_res_synthesis_with_weights",
]
