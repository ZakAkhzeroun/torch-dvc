from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .CNN_img_torch import MVSynthesis, ResSynthesis
from .MC_network_torch import MCNetwork
from .quant_motion import dense_image_warp


class OpenDVCPFrameDecoderQAT(nn.Module):
    """
    Quantization-aware decoder path for OpenDVC P-frames.

    Quantized neural layers:
    - MV synthesis
    - MC network
    - Residual synthesis

    Intentionally kept fp32:
    - dense_image_warp (grid_sample based)
    - tensor concat
    - final add and clamp
    """

    def __init__(
        self,
        mv_synthesis: MVSynthesis,
        mc_net: MCNetwork,
        res_synthesis: ResSynthesis,
    ):
        super().__init__()
        self.mv_synthesis = mv_synthesis
        self.mc_net = mc_net
        self.res_synthesis = res_synthesis

    def forward(
        self,
        y0_com: torch.Tensor,
        flow_latent_hat: torch.Tensor,
        res_latent_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        flow_hat = self.mv_synthesis(flow_latent_hat)
        y1_warp = dense_image_warp(y0_com, flow_hat)
        mc_input = torch.cat([flow_hat, y0_com, y1_warp], dim=1)
        y1_mc = self.mc_net(mc_input)
        res_hat = self.res_synthesis(res_latent_hat)
        y1_com = torch.clamp(res_hat + y1_mc, 0.0, 1.0)
        return {
            "flow_hat": flow_hat,
            "y1_warp": y1_warp,
            "y1_mc": y1_mc,
            "res_hat": res_hat,
            "y1_com": y1_com,
        }


def build_opendvc_pframe_decoder_qat(
    num_filters: int = 128,
    latent_channels: int = 128,
    weight_bits: int = 16,
    act_bits: int = 16,
    device: torch.device = torch.device("cpu"),
) -> OpenDVCPFrameDecoderQAT:
    mv_synthesis = MVSynthesis(num_filters=num_filters, M=latent_channels, weight_bit_width=weight_bits)
    mc_net = MCNetwork(weight_bit_width=weight_bits, act_bit_width=act_bits)
    res_synthesis = ResSynthesis(num_filters=num_filters, M=latent_channels, weight_bit_width=weight_bits)
    model = OpenDVCPFrameDecoderQAT(
        mv_synthesis=mv_synthesis,
        mc_net=mc_net,
        res_synthesis=res_synthesis,
    ).to(device)
    model.eval()
    return model


def load_fp32_state_into_qat_decoder(
    qat_decoder: nn.Module,
    fp32_state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, int, Dict[str, str]]:
    """
    Load compatible fp32 decoder weights into QAT decoder.

    Returns:
        loaded_count, skipped_count, skipped_reason_by_key
    """
    qat_state = qat_decoder.state_dict()
    loaded = 0
    skipped = {}
    for key, value in fp32_state_dict.items():
        if key not in qat_state:
            skipped[key] = "missing_in_qat_decoder"
            continue
        if qat_state[key].shape != value.shape:
            skipped[key] = "shape_mismatch"
            continue
        qat_state[key].copy_(value)
        loaded += 1
    qat_decoder.load_state_dict(qat_state, strict=False)
    return loaded, len(skipped), skipped


__all__ = [
    "OpenDVCPFrameDecoderQAT",
    "build_opendvc_pframe_decoder_qat",
    "load_fp32_state_into_qat_decoder",
]
