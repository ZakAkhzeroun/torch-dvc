import torch
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn


def quant_friendly_interpolate(x: torch.Tensor, size) -> torch.Tensor:
    """
    Quantization-friendly nearest-neighbor upsample via repeat to target size.

    Uses integer repeats and crops to exact spatial size to avoid shape drift
    across odd-sized pyramids built with ceil-mode pooling.
    """
    out_h, out_w = int(size[0]), int(size[1])
    in_h, in_w = x.shape[-2], x.shape[-1]
    if out_h < in_h or out_w < in_w:
        raise ValueError("Target size must be >= input size for upsample.")
    if out_h == in_h and out_w == in_w:
        return x

    scale_h = max((out_h + in_h - 1) // in_h, 1)
    scale_w = max((out_w + in_w - 1) // in_w, 1)
    x = x.repeat_interleave(scale_h, dim=2)
    x = x.repeat_interleave(scale_w, dim=3)
    return x[:, :, :out_h, :out_w]


def quant_friendly_avg_pool2d_stride2(x: torch.Tensor) -> torch.Tensor:
    """
    Quantization-friendly 2x2 average pooling with stride 2.

    Implemented as sum over each 2x2 window multiplied by 0.25.
    In fixed-point hardware, division by 4 is typically a right shift.
    """
    x_sum = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False) * 4.0
    return x_sum * 0.25


def dense_image_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp an image using flow in pixel units, matching tf.contrib.image.dense_image_warp."""
    batch, _, height, width = image.shape

    y_base = torch.arange(height, device=image.device, dtype=image.dtype)
    x_base = torch.arange(width, device=image.device, dtype=image.dtype)
    try:
        grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(y_base, x_base)

    grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)

    sample_y = grid_y - flow[:, 0, :, :]
    sample_x = grid_x - flow[:, 1, :, :]

    if height > 1:
        sample_y = (2.0 * sample_y / float(height - 1)) - 1.0
    else:
        sample_y = torch.zeros_like(sample_y)

    if width > 1:
        sample_x = (2.0 * sample_x / float(width - 1)) - 1.0
    else:
        sample_x = torch.zeros_like(sample_x)

    sampling_grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class FlowRefinementBlock(nn.Module):
    """Quantized flow refinement block for one pyramid level."""

    def __init__(self, weight_bit_width: int = 8, act_bit_width: int = 8):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=False)
        self.relu = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=False)

        self.conv1 = qnn.QuantConv2d(8, 32, kernel_size=7, padding=3, weight_bit_width=weight_bit_width, bias=True)
        self.conv2 = qnn.QuantConv2d(32, 64, kernel_size=7, padding=3, weight_bit_width=weight_bit_width, bias=True)
        self.conv3 = qnn.QuantConv2d(64, 32, kernel_size=7, padding=3, weight_bit_width=weight_bit_width, bias=True)
        self.conv4 = qnn.QuantConv2d(32, 16, kernel_size=7, padding=3, weight_bit_width=weight_bit_width, bias=True)
        self.conv5 = qnn.QuantConv2d(16, 2, kernel_size=7, padding=3, weight_bit_width=weight_bit_width, bias=True)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, qnn.QuantConv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, im1_warp: torch.Tensor, im2: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([im1_warp, im2, flow], dim=1)
        x = self.input_quant(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.conv5(x)


class MotionNetwork(nn.Module):
    """Quantized variant of the OpenDVC motion network."""

    def __init__(self, weight_bit_width: int = 8, act_bit_width: int = 8, warp_grad_through_flow: bool = True):
        super().__init__()
        self.warp_grad_through_flow = bool(warp_grad_through_flow)
        self.blocks = nn.ModuleList(
            [FlowRefinementBlock(weight_bit_width=weight_bit_width, act_bit_width=act_bit_width) for _ in range(5)]
        )
        self.pre_add_quant = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=False)
        self.post_add_quant = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=False)

    def _qadd(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_q = self.pre_add_quant(a)
        b_q = self.pre_add_quant(b)
        return self.post_add_quant(a_q + b_q)

    def convnet(self, im1_warp: torch.Tensor, im2: torch.Tensor, flow: torch.Tensor, layer: int) -> torch.Tensor:
        return self.blocks[layer](im1_warp, im2, flow)

    def loss(self, flow_coarse: torch.Tensor, im1: torch.Tensor, im2: torch.Tensor, layer: int):
        target_size = (im1.shape[-2], im1.shape[-1])
        flow = quant_friendly_interpolate(flow_coarse, size=target_size)
        flow = self.post_add_quant(flow)

        # Optional safety mode: disable flow-grid gradient path for warp ops.
        warp_flow = flow if self.warp_grad_through_flow else flow.detach()
        im1_warped = dense_image_warp(im1, warp_flow)
        residual = self.convnet(im1_warped, im2, flow, layer)
        flow_fine = self._qadd(residual, flow)

        warp_flow_fine = flow_fine if self.warp_grad_through_flow else flow_fine.detach()
        im1_warped_fine = dense_image_warp(im1, warp_flow_fine)
        loss_layer = torch.mean((im1_warped_fine - im2) ** 2)
        return loss_layer, flow_fine

    def optical_flow(self, im1_4: torch.Tensor, im2_4: torch.Tensor):
        im1_3 = quant_friendly_avg_pool2d_stride2(im1_4)
        im1_2 = quant_friendly_avg_pool2d_stride2(im1_3)
        im1_1 = quant_friendly_avg_pool2d_stride2(im1_2)
        im1_0 = quant_friendly_avg_pool2d_stride2(im1_1)

        im2_3 = quant_friendly_avg_pool2d_stride2(im2_4)
        im2_2 = quant_friendly_avg_pool2d_stride2(im2_3)
        im2_1 = quant_friendly_avg_pool2d_stride2(im2_2)
        im2_0 = quant_friendly_avg_pool2d_stride2(im2_1)

        batch = im1_4.shape[0]
        flow_zero = torch.zeros(
            batch,
            2,
            im1_0.shape[-2],
            im1_0.shape[-1],
            device=im1_4.device,
            dtype=im1_4.dtype,
        )

        loss_0, flow_0 = self.loss(flow_zero, im1_0, im2_0, 0)
        loss_1, flow_1 = self.loss(flow_0, im1_1, im2_1, 1)
        loss_2, flow_2 = self.loss(flow_1, im1_2, im2_2, 2)
        loss_3, flow_3 = self.loss(flow_2, im1_3, im2_3, 3)
        loss_4, flow_4 = self.loss(flow_3, im1_4, im2_4, 4)

        return flow_4, loss_0, loss_1, loss_2, loss_3, loss_4

    def forward(self, im1_4: torch.Tensor, im2_4: torch.Tensor):
        return self.optical_flow(im1_4, im2_4)
