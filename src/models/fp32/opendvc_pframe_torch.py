from typing import Dict, Optional, Union

import torch
import torch.nn as nn

try:
    from .CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis
    from .MC_network_torch import MCNetwork
    from .cnn_img_weights import (
        build_mv_analysis_with_weights,
        build_mv_synthesis_with_weights,
        build_res_analysis_with_weights,
        build_res_synthesis_with_weights,
    )
    from .motion_torch import MotionNetwork, dense_image_warp
    from .motion_weights import build_motion_with_weights
    from .weights import build_mc_with_weights
except ImportError:
    from src.models.fp32.CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis
    from src.models.fp32.MC_network_torch import MCNetwork
    from src.models.fp32.cnn_img_weights import (
        build_mv_analysis_with_weights,
        build_mv_synthesis_with_weights,
        build_res_analysis_with_weights,
        build_res_synthesis_with_weights,
    )
    from src.models.fp32.motion_torch import MotionNetwork, dense_image_warp
    from src.models.fp32.motion_weights import build_motion_with_weights
    from src.models.fp32.weights import build_mc_with_weights


class OpenDVCPFrameModel(nn.Module):
    """
    PyTorch recreation of the neural forward path in OpenDVC_test_P-frame.py.

    This model reproduces the network computation graph:
    1. Optical flow estimation
    2. Motion analysis transform
    3. Motion synthesis transform
    4. Motion compensation network
    5. Residual analysis transform
    6. Residual synthesis transform
    7. Final frame reconstruction

    Note:
    The original TensorFlow script also applies entropy bottlenecks for motion
    and residual latents. Those entropy coding steps are not implemented in
    torch_dvc, so this module treats the latent hats as identity by default:
    `flow_latent_hat = flow_latent` and `res_latent_hat = res_latent`.
    """

    def __init__(
        self,
        motion_net: MotionNetwork,
        mv_analysis: MVAnalysis,
        mv_synthesis: MVSynthesis,
        mc_net: MCNetwork,
        res_analysis: ResAnalysis,
        res_synthesis: ResSynthesis,
    ):
        super().__init__()
        self.motion_net = motion_net
        self.mv_analysis = mv_analysis
        self.mv_synthesis = mv_synthesis
        self.mc_net = mc_net
        self.res_analysis = res_analysis
        self.res_synthesis = res_synthesis

    def forward(
        self,
        y0_com: torch.Tensor,
        y1_raw: torch.Tensor,
        flow_latent_hat: Optional[torch.Tensor] = None,
        res_latent_hat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the OpenDVC P-frame forward pass on NCHW tensors in [0, 1].

        Args:
            y0_com: Reference frame, shape [B, 3, H, W].
            y1_raw: Target raw frame, shape [B, 3, H, W].
            flow_latent_hat: Optional motion latent after quantization/decode.
            res_latent_hat: Optional residual latent after quantization/decode.
        """
        flow_tensor, _, _, _, _, _ = self.motion_net(y0_com, y1_raw)

        flow_latent = self.mv_analysis(flow_tensor)
        if flow_latent_hat is None:
            flow_latent_hat = flow_latent

        flow_hat = self.mv_synthesis(flow_latent_hat)

        y1_warp = dense_image_warp(y0_com, flow_hat)

        mc_input = torch.cat([flow_hat, y0_com, y1_warp], dim=1)
        y1_mc = self.mc_net(mc_input)

        res = y1_raw - y1_mc
        res_latent = self.res_analysis(res)
        if res_latent_hat is None:
            res_latent_hat = res_latent

        res_hat = self.res_synthesis(res_latent_hat)
        y1_com = torch.clamp(res_hat + y1_mc, 0.0, 1.0)

        return {
            "flow_tensor": flow_tensor,
            "flow_latent": flow_latent,
            "flow_latent_hat": flow_latent_hat,
            "flow_hat": flow_hat,
            "y1_warp": y1_warp,
            "y1_mc": y1_mc,
            "res": res,
            "res_latent": res_latent,
            "res_latent_hat": res_latent_hat,
            "res_hat": res_hat,
            "y1_com": y1_com,
        }


def build_opendvc_pframe_model(
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = "OpenDVC_model/PSNR_256_model",
    basketball_data_root: str = "OpenDVC_model/PSNR_256_model",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> OpenDVCPFrameModel:
    """
    Build a PyTorch OpenDVC P-frame model with weights loaded from TensorFlow.

    The motion and image-transform weights are read from the OpenDVC checkpoint.
    The MC network weights are loaded using the existing MC weight builder.
    """
    motion_net = build_motion_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        motion_root=open_dvc_root,
        device=device,
    )
    mv_analysis = build_mv_analysis_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        device=device,
    )
    mv_synthesis = build_mv_synthesis_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        device=device,
    )
    mc_net = build_mc_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        basketball_data_root=basketball_data_root,
        metric=metric,
        device=device,
    )

    res_analysis = build_res_analysis_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        device=device,
    )
    res_synthesis = build_res_synthesis_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        device=device,
    )

    model = OpenDVCPFrameModel(
        motion_net=motion_net,
        mv_analysis=mv_analysis,
        mv_synthesis=mv_synthesis,
        mc_net=mc_net,
        res_analysis=res_analysis,
        res_synthesis=res_synthesis,
    ).to(device)
    model.eval()
    return model


__all__ = [
    "OpenDVCPFrameModel",
    "build_opendvc_pframe_model",
]
