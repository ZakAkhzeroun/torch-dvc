import torch
import torch.nn as nn
import torch.nn.functional as F


def tf_buggy_resize_bilinear(x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """Match TensorFlow 1.x tf.image.resize_images bilinear sampling."""
    _, _, in_height, in_width = x.shape
    out_height = in_height * scale_factor
    out_width = in_width * scale_factor

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


class MCNetwork(nn.Module):
    """PyTorch implementation of the MC network architecture."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv6_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv7_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv9_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv11_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv11_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv13 = nn.Conv2d(64, 3, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m1 = self.conv1(x)

        l1 = F.relu(m1)
        l1 = self.conv2_1(l1)
        l2 = F.relu(l1)
        l2 = self.conv2_2(l2)
        m2 = m1 + l2

        m3 = F.avg_pool2d(m2, kernel_size=2, stride=2, padding=0)

        l1 = F.relu(m3)
        l1 = self.conv4_1(l1)
        l2 = F.relu(l1)
        l2 = self.conv4_2(l2)
        m4 = m3 + l2

        m5 = F.avg_pool2d(m4, kernel_size=2, stride=2, padding=0)

        l1 = F.relu(m5)
        l1 = self.conv6_1(l1)
        l2 = F.relu(l1)
        l2 = self.conv6_2(l2)
        m6 = m5 + l2

        l1 = F.relu(m6)
        l1 = self.conv7_1(l1)
        l2 = F.relu(l1)
        l2 = self.conv7_2(l2)
        m7 = m6 + l2

        m8 = tf_buggy_resize_bilinear(m7, scale_factor=2)
        m8 = m4 + m8

        l1 = F.relu(m8)
        l1 = self.conv9_1(l1)
        l2 = F.relu(l1)
        l2 = self.conv9_2(l2)
        m9 = m8 + l2

        m10 = tf_buggy_resize_bilinear(m9, scale_factor=2)
        m10 = m2 + m10

        l1 = F.relu(m10)
        l1 = self.conv11_1(l1)
        l2 = F.relu(l1)
        l2 = self.conv11_2(l2)
        m11 = m10 + l2

        m12 = self.conv12(m11)
        m12 = F.relu(m12)

        return self.conv13(m12)
