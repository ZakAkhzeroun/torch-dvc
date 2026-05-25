import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn


def _to_2tuple(value) -> Tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("Expected a 2-element iterable.")
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


def _same_padding_for_kernel(kernel_size, corr, strides_up=(1, 1)):
    kernel_size = _to_2tuple(kernel_size)
    strides_up = _to_2tuple(strides_up)

    if corr:
        padding = [(k // 2, (k - 1) // 2) for k in kernel_size]
    else:
        padding = [((k - 1) // 2, k // 2) for k in kernel_size]

    return tuple(
        ((pad[0] - 1) // stride + 1, (pad[1] - 1) // stride + 1)
        for pad, stride in zip(padding, strides_up)
    )


def _padding_for_torch(padding_pairs):
    (pad_h0, pad_h1), (pad_w0, pad_w1) = padding_pairs
    return pad_w0, pad_w1, pad_h0, pad_h1


def _slice_along_hw(x: torch.Tensor, top: int, bottom: int, left: int, right: int):
    height_slice = slice(top, None if bottom == 0 else -bottom)
    width_slice = slice(left, None if right == 0 else -right)
    return x[:, :, height_slice, width_slice]


class GDN(nn.Module):
    """FP32 GDN/IGDN kept unchanged for training stability."""

    def __init__(
        self,
        channels: int,
        inverse: bool = False,
        rectify: bool = False,
        gamma_init: float = 0.1,
        beta_min: float = 1e-6,
        reparam_offset: float = 2 ** -18,
    ):
        super().__init__()
        self.channels = int(channels)
        self.inverse = bool(inverse)
        self.rectify = bool(rectify)
        self.beta_min = float(beta_min)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset ** 2
        beta_init = torch.sqrt(torch.ones(self.channels) + pedestal)
        gamma_init_matrix = gamma_init * torch.eye(self.channels)
        gamma_init_matrix = torch.sqrt(gamma_init_matrix + pedestal)

        self.beta_reparam = nn.Parameter(beta_init)
        self.gamma_reparam = nn.Parameter(gamma_init_matrix)

    def _reparameterize_beta(self):
        pedestal = self.reparam_offset ** 2
        bound = math.sqrt(self.beta_min + pedestal)
        beta = torch.maximum(
            self.beta_reparam,
            torch.tensor(bound, dtype=self.beta_reparam.dtype, device=self.beta_reparam.device),
        )
        return beta.pow(2) - pedestal

    def _reparameterize_gamma(self):
        pedestal = self.reparam_offset ** 2
        gamma = torch.maximum(
            self.gamma_reparam,
            torch.tensor(
                self.reparam_offset,
                dtype=self.gamma_reparam.dtype,
                device=self.gamma_reparam.device,
            ),
        )
        return gamma.pow(2) - pedestal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("GDN expects a 4D NCHW tensor.")

        if self.rectify:
            x = F.relu(x)

        beta = self._reparameterize_beta()
        gamma = self._reparameterize_gamma().transpose(0, 1).contiguous()
        gamma = gamma.view(self.channels, self.channels, 1, 1)
        norm_pool = F.conv2d(x.pow(2), gamma, beta)

        if self.inverse:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.rsqrt(norm_pool)

        return x * norm_pool


class SignalConv2D(nn.Module):
    """
    Quantized variant of the minimal tfc.SignalConv2D port.

    GDN/IGDN remains fp32; only conv operators are quantized.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        corr: bool = True,
        strides_down=1,
        strides_up=1,
        padding: str = "valid",
        use_bias: bool = False,
        activation: nn.Module = None,
        weight_bit_width: int = 8,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _to_2tuple(kernel_size)
        self.corr = bool(corr)
        self.strides_down = _to_2tuple(strides_down)
        self.strides_up = _to_2tuple(strides_up)
        self.padding = str(padding).lower()
        if self.padding == "same":
            self.padding = "same_zeros"
        self.use_bias = bool(use_bias)
        self.activation = activation

        if any(stride != 1 for stride in self.strides_up) and any(stride != 1 for stride in self.strides_down):
            raise NotImplementedError("This port does not support simultaneous upsampling and downsampling.")
        if self.padding not in ("valid", "same_zeros"):
            raise NotImplementedError("Only 'valid' and 'same_zeros' padding are implemented.")

        if self.corr:
            if any(stride != 1 for stride in self.strides_up):
                raise NotImplementedError("Cross-correlation with strides_up is not used in CNN_img.py.")
            self.op = qnn.QuantConv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.strides_down,
                padding=0,
                bias=self.use_bias,
                weight_bit_width=weight_bit_width,
                return_quant_tensor=False,
            )
        else:
            if any(stride != 1 for stride in self.strides_down):
                raise NotImplementedError("Convolution with strides_down is not used in CNN_img.py.")
            self.op = qnn.QuantConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.strides_up,
                padding=0,
                output_padding=0,
                bias=self.use_bias,
                weight_bit_width=weight_bit_width,
                return_quant_tensor=False,
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.op.weight)
        if self.op.bias is not None:
            nn.init.zeros_(self.op.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.corr and self.padding == "same_zeros":
            padding_pairs = _same_padding_for_kernel(self.kernel_size, corr=True, strides_up=(1, 1))
            x = F.pad(x, _padding_for_torch(padding_pairs))
            x = self.op(x)
        elif self.corr:
            x = self.op(x)
        else:
            original_height, original_width = x.shape[-2:]
            if self.padding == "same_zeros":
                padding_pairs = _same_padding_for_kernel(self.kernel_size, corr=False, strides_up=self.strides_up)
                x = F.pad(x, _padding_for_torch(padding_pairs))
                pad_top, pad_bottom = padding_pairs[0]
                pad_left, pad_right = padding_pairs[1]
            else:
                pad_top = pad_bottom = pad_left = pad_right = 0

            x = self.op(x)

            if self.padding == "same_zeros":
                crop_top = pad_top * self.strides_up[0] + self.kernel_size[0] // 2
                crop_left = pad_left * self.strides_up[1] + self.kernel_size[1] // 2
                target_height = original_height * self.strides_up[0]
                target_width = original_width * self.strides_up[1]
                crop_bottom = x.shape[-2] - (crop_top + target_height)
                crop_right = x.shape[-1] - (crop_left + target_width)
                x = _slice_along_hw(x, crop_top, crop_bottom, crop_left, crop_right)

        if self.activation is not None:
            x = self.activation(x)
        return x


class MVAnalysis(nn.Module):
    def __init__(self, num_filters: int, M: int, in_channels: int = 2, weight_bit_width: int = 8):
        super().__init__()
        self.layer_0 = SignalConv2D(
            in_channels,
            num_filters,
            (3, 3),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters),
            weight_bit_width=weight_bit_width,
        )
        self.layer_1 = SignalConv2D(
            num_filters,
            num_filters,
            (3, 3),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters),
            weight_bit_width=weight_bit_width,
        )
        self.layer_2 = SignalConv2D(
            num_filters,
            num_filters,
            (3, 3),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters),
            weight_bit_width=weight_bit_width,
        )
        self.layer_3 = SignalConv2D(
            num_filters,
            M,
            (3, 3),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=False,
            activation=None,
            weight_bit_width=weight_bit_width,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.layer_0(tensor)
        tensor = self.layer_1(tensor)
        tensor = self.layer_2(tensor)
        tensor = self.layer_3(tensor)
        return tensor


class MVSynthesis(nn.Module):
    def __init__(self, num_filters: int, M: int, out_channels: int = 2, weight_bit_width: int = 8):
        super().__init__()
        self.layer_0 = SignalConv2D(
            M,
            num_filters,
            (3, 3),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters, inverse=True),
            weight_bit_width=weight_bit_width,
        )
        self.layer_1 = SignalConv2D(
            num_filters,
            num_filters,
            (3, 3),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters, inverse=True),
            weight_bit_width=weight_bit_width,
        )
        self.layer_2 = SignalConv2D(
            num_filters,
            num_filters,
            (3, 3),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters, inverse=True),
            weight_bit_width=weight_bit_width,
        )
        self.layer_3 = SignalConv2D(
            num_filters,
            out_channels,
            (3, 3),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
            weight_bit_width=weight_bit_width,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.layer_0(tensor)
        tensor = self.layer_1(tensor)
        tensor = self.layer_2(tensor)
        tensor = self.layer_3(tensor)
        return tensor


class ResAnalysis(nn.Module):
    def __init__(self, num_filters: int, M: int, in_channels: int = 3, weight_bit_width: int = 8):
        super().__init__()
        self.layer_0 = SignalConv2D(
            in_channels,
            num_filters,
            (5, 5),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters),
            weight_bit_width=weight_bit_width,
        )
        self.layer_1 = SignalConv2D(
            num_filters,
            num_filters,
            (5, 5),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters),
            weight_bit_width=weight_bit_width,
        )
        self.layer_2 = SignalConv2D(
            num_filters,
            num_filters,
            (5, 5),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters),
            weight_bit_width=weight_bit_width,
        )
        self.layer_3 = SignalConv2D(
            num_filters,
            M,
            (5, 5),
            corr=True,
            strides_down=2,
            padding="same_zeros",
            use_bias=False,
            activation=None,
            weight_bit_width=weight_bit_width,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.layer_0(tensor)
        tensor = self.layer_1(tensor)
        tensor = self.layer_2(tensor)
        tensor = self.layer_3(tensor)
        return tensor


class ResSynthesis(nn.Module):
    def __init__(self, num_filters: int, M: int, out_channels: int = 3, weight_bit_width: int = 8):
        super().__init__()
        self.layer_0 = SignalConv2D(
            M,
            num_filters,
            (5, 5),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters, inverse=True),
            weight_bit_width=weight_bit_width,
        )
        self.layer_1 = SignalConv2D(
            num_filters,
            num_filters,
            (5, 5),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters, inverse=True),
            weight_bit_width=weight_bit_width,
        )
        self.layer_2 = SignalConv2D(
            num_filters,
            num_filters,
            (5, 5),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=GDN(num_filters, inverse=True),
            weight_bit_width=weight_bit_width,
        )
        self.layer_3 = SignalConv2D(
            num_filters,
            out_channels,
            (5, 5),
            corr=False,
            strides_up=2,
            padding="same_zeros",
            use_bias=True,
            activation=None,
            weight_bit_width=weight_bit_width,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.layer_0(tensor)
        tensor = self.layer_1(tensor)
        tensor = self.layer_2(tensor)
        tensor = self.layer_3(tensor)
        return tensor


MV_analysis = MVAnalysis
MV_synthesis = MVSynthesis
Res_analysis = ResAnalysis
Res_synthesis = ResSynthesis


__all__ = [
    "GDN",
    "SignalConv2D",
    "MVAnalysis",
    "MVSynthesis",
    "ResAnalysis",
    "ResSynthesis",
    "MV_analysis",
    "MV_synthesis",
    "Res_analysis",
    "Res_synthesis",
]
