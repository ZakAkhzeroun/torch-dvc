import torch
import torch.nn as nn
import torch.nn.functional as F


def avg_pool2d_same(x: torch.Tensor) -> torch.Tensor:
    """Apply TensorFlow-style average pooling with SAME padding."""
    return F.avg_pool2d(
        x,
        kernel_size=2,
        stride=2,
        ceil_mode=True,
        count_include_pad=False,
    )


def tf_buggy_resize_bilinear(x: torch.Tensor, size) -> torch.Tensor:
    """Match TensorFlow 1.x tf.image.resize_images bilinear sampling."""
    _, _, in_height, in_width = x.shape
    out_height, out_width = size

    if in_height == 1:
        y_coords = torch.zeros(out_height, device=x.device, dtype=x.dtype)
    else:
        y_coords = torch.arange(out_height, device=x.device, dtype=x.dtype)
        y_coords = y_coords * (float(in_height) / float(out_height))
        y_coords = (2.0 * y_coords / float(in_height - 1)) - 1.0

    if in_width == 1:
        x_coords = torch.zeros(out_width, device=x.device, dtype=x.dtype)
    else:
        x_coords = torch.arange(out_width, device=x.device, dtype=x.dtype)
        x_coords = x_coords * (float(in_width) / float(out_width))
        x_coords = (2.0 * x_coords / float(in_width - 1)) - 1.0

    try:
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords)

    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


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
    """Refine optical flow at one pyramid level."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(16, 2, kernel_size=7, padding=3)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, im1_warp: torch.Tensor, im2: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([im1_warp, im2, flow], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.conv5(x)


class MotionNetwork(nn.Module):
    """PyTorch implementation of the TensorFlow motion network."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FlowRefinementBlock() for _ in range(5)])

    def convnet(
        self,
        im1_warp: torch.Tensor,
        im2: torch.Tensor,
        flow: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Run the refinement CNN for one pyramid layer."""
        return self.blocks[layer](im1_warp, im2, flow)

    def loss(
        self,
        flow_coarse: torch.Tensor,
        im1: torch.Tensor,
        im2: torch.Tensor,
        layer: int,
    ):
        """Refine the flow at one pyramid level and compute reconstruction loss."""
        target_size = (im1.shape[-2], im2.shape[-1])
        flow = tf_buggy_resize_bilinear(flow_coarse, target_size)
        im1_warped = dense_image_warp(im1, flow)
        residual = self.convnet(im1_warped, im2, flow, layer)
        flow_fine = residual + flow

        im1_warped_fine = dense_image_warp(im1, flow_fine)
        loss_layer = torch.mean((im1_warped_fine - im2) ** 2)

        return loss_layer, flow_fine

    def optical_flow(self, im1_4: torch.Tensor, im2_4: torch.Tensor):
        """Compute full-resolution optical flow and per-level losses."""
        im1_3 = avg_pool2d_same(im1_4)
        im1_2 = avg_pool2d_same(im1_3)
        im1_1 = avg_pool2d_same(im1_2)
        im1_0 = avg_pool2d_same(im1_1)

        im2_3 = avg_pool2d_same(im2_4)
        im2_2 = avg_pool2d_same(im2_3)
        im2_1 = avg_pool2d_same(im2_2)
        im2_0 = avg_pool2d_same(im2_1)

        batch, _, height, width = im1_4.shape
        flow_zero = torch.zeros(
            batch,
            2,
            height // 16,
            width // 16,
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
        """Alias for the optical flow computation."""
        return self.optical_flow(im1_4, im2_4)
