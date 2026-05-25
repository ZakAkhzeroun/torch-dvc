import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn


def quant_friendly_interpolate(x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """
    Quantization-friendly 2D upsample.

    Uses nearest-neighbor repeat, which maps cleanly to integer arithmetic
    and avoids floating-point resampling kernels such as grid_sample.
    """
    if scale_factor < 1 or int(scale_factor) != scale_factor:
        raise ValueError("scale_factor must be a positive integer.")
    scale = int(scale_factor)
    if scale == 1:
        return x
    x = x.repeat_interleave(scale, dim=2)
    x = x.repeat_interleave(scale, dim=3)
    return x


class MCNetwork(nn.Module):
    """
    Brevitas QAT-friendly MC network.

    Quantization strategy:
    - Quantized convolutions + quantized ReLU activations.
    - Residual additions pass through explicit quantize/dequantize boundaries
      to stabilize shared scale handling before and after add operations.
    """

    def __init__(self, weight_bit_width: int = 8, act_bit_width: int = 8):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=False)

        self.conv1 = qnn.QuantConv2d(8, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv2_1 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv2_2 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv4_1 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv4_2 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv6_1 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv6_2 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv7_1 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv7_2 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv9_1 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv9_2 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv11_1 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv11_2 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.conv12 = qnn.QuantConv2d(64, 64, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)
        self.conv13 = qnn.QuantConv2d(64, 3, 3, 1, 1, weight_bit_width=weight_bit_width, bias=True)

        self.relu = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=False)
        self.pre_add_quant = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=False)
        self.post_add_quant = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, qnn.QuantConv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _qadd(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Re-quantize both branches before summation, then requantize the sum.
        a_q = self.pre_add_quant(a)
        b_q = self.pre_add_quant(b)
        return self.post_add_quant(a_q + b_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        m1 = self.conv1(x)

        l1 = self.relu(m1)
        l1 = self.conv2_1(l1)
        l2 = self.relu(l1)
        l2 = self.conv2_2(l2)
        m2 = self._qadd(m1, l2)

        m3 = F.avg_pool2d(m2, kernel_size=2, stride=2, padding=0)
        m3 = self.post_add_quant(m3)

        l1 = self.relu(m3)
        l1 = self.conv4_1(l1)
        l2 = self.relu(l1)
        l2 = self.conv4_2(l2)
        m4 = self._qadd(m3, l2)

        m5 = F.avg_pool2d(m4, kernel_size=2, stride=2, padding=0)
        m5 = self.post_add_quant(m5)

        l1 = self.relu(m5)
        l1 = self.conv6_1(l1)
        l2 = self.relu(l1)
        l2 = self.conv6_2(l2)
        m6 = self._qadd(m5, l2)

        l1 = self.relu(m6)
        l1 = self.conv7_1(l1)
        l2 = self.relu(l1)
        l2 = self.conv7_2(l2)
        m7 = self._qadd(m6, l2)

        m8 = quant_friendly_interpolate(m7, scale_factor=2)
        m8 = self.post_add_quant(m8)
        m8 = self._qadd(m4, m8)

        l1 = self.relu(m8)
        l1 = self.conv9_1(l1)
        l2 = self.relu(l1)
        l2 = self.conv9_2(l2)
        m9 = self._qadd(m8, l2)

        m10 = quant_friendly_interpolate(m9, scale_factor=2)
        m10 = self.post_add_quant(m10)
        m10 = self._qadd(m2, m10)

        l1 = self.relu(m10)
        l1 = self.conv11_1(l1)
        l2 = self.relu(l1)
        l2 = self.conv11_2(l2)
        m11 = self._qadd(m10, l2)

        m12 = self.conv12(m11)
        m12 = self.relu(m12)

        return self.conv13(m12)
