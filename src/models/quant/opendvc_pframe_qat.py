from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis
from .MC_network_torch import MCNetwork
from .quant_motion import MotionNetwork, dense_image_warp


class IdentityEntropyModel(nn.Module):
    """
    Placeholder entropy model for reconstruction-only QAT debugging.

    Entropy likelihood computation is intentionally left optional in this QAT
    path to keep fragile rate modeling in fp32 until a dedicated quantized
    entropy path is added.
    """

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return x, None

    def aux_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)


class OpenDVCPFrameQATModel(nn.Module):
    """
    Quantization-aware OpenDVC P-frame model.

    Safety choices:
    - Keep warping/grid-sample flow path in fp32.
    - Keep final clamp in fp32.
    - Keep residual/flow/latent signed tensors unconstrained by unsigned ReLU
      quantization in top-level math.
    """

    def __init__(
        self,
        num_filters: int = 128,
        latent_channels: int = 128,
        weight_bits: int = 16,
        act_bits: int = 16,
        warp_grad_through_flow: bool = False,
        entropy_factory: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        self.warp_grad_through_flow = bool(warp_grad_through_flow)
        self.motion_net = MotionNetwork(
            weight_bit_width=weight_bits,
            act_bit_width=act_bits,
            warp_grad_through_flow=warp_grad_through_flow,
        )
        self.mv_analysis = MVAnalysis(num_filters=num_filters, M=latent_channels, weight_bit_width=weight_bits)
        self.mv_synthesis = MVSynthesis(num_filters=num_filters, M=latent_channels, weight_bit_width=weight_bits)
        self.mc_net = MCNetwork(weight_bit_width=weight_bits, act_bit_width=act_bits)
        self.res_analysis = ResAnalysis(num_filters=num_filters, M=latent_channels, weight_bit_width=weight_bits)
        self.res_synthesis = ResSynthesis(num_filters=num_filters, M=latent_channels, weight_bit_width=weight_bits)

        if entropy_factory is None:
            self.entropy_bottleneck_mv = IdentityEntropyModel()
            self.entropy_bottleneck_res = IdentityEntropyModel()
        else:
            self.entropy_bottleneck_mv = entropy_factory(latent_channels)
            self.entropy_bottleneck_res = entropy_factory(latent_channels)

    def forward(
        self,
        y0_com: torch.Tensor,
        y1_raw: torch.Tensor,
        flow_latent_hat: Optional[torch.Tensor] = None,
        res_latent_hat: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        flow_tensor, loss_0, loss_1, loss_2, loss_3, loss_4 = self.motion_net(y0_com, y1_raw)

        flow_latent = self.mv_analysis(flow_tensor)
        if flow_latent_hat is None:
            flow_latent_hat, mv_likelihoods = self.entropy_bottleneck_mv(flow_latent, training=self.training)
        else:
            mv_likelihoods = None

        flow_hat = self.mv_synthesis(flow_latent_hat)
        # Safety mode for older runtimes: optionally avoid grid-sample backward
        # through flow coordinates by detaching flow before warping.
        warp_flow_hat = flow_hat if self.warp_grad_through_flow else flow_hat.detach()
        y1_warp = dense_image_warp(y0_com, warp_flow_hat)
        mc_input = torch.cat([flow_hat, y0_com, y1_warp], dim=1)
        y1_mc = self.mc_net(mc_input)

        res = y1_raw - y1_mc
        res_latent = self.res_analysis(res)
        if res_latent_hat is None:
            res_latent_hat, res_likelihoods = self.entropy_bottleneck_res(res_latent, training=self.training)
        else:
            res_likelihoods = None

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
            "mv_likelihoods": mv_likelihoods,
            "res_likelihoods": res_likelihoods,
            "motion_loss_0": loss_0,
            "motion_loss_1": loss_1,
            "motion_loss_2": loss_2,
            "motion_loss_3": loss_3,
            "motion_loss_4": loss_4,
        }

    def aux_loss(self) -> torch.Tensor:
        mv_aux = self.entropy_bottleneck_mv.aux_loss() if hasattr(self.entropy_bottleneck_mv, "aux_loss") else 0.0
        res_aux = self.entropy_bottleneck_res.aux_loss() if hasattr(self.entropy_bottleneck_res, "aux_loss") else 0.0
        if not torch.is_tensor(mv_aux):
            mv_aux = torch.tensor(float(mv_aux))
        if not torch.is_tensor(res_aux):
            res_aux = torch.tensor(float(res_aux))
        return mv_aux + res_aux


def load_fp32_state_into_qat(
    qat_model: nn.Module,
    fp32_state_dict: Dict[str, torch.Tensor],
):
    """
    Load compatible fp32 weights into a QAT model.

    Returns:
        loaded_keys: list of parameter names copied
        skipped: list of tuples (name, reason)
    """
    qat_state = qat_model.state_dict()
    loaded_keys = []
    skipped = []

    for key, value in fp32_state_dict.items():
        if key not in qat_state:
            skipped.append((key, "missing_in_qat"))
            continue
        if qat_state[key].shape != value.shape:
            skipped.append((key, "shape_mismatch"))
            continue
        qat_state[key].copy_(value)
        loaded_keys.append(key)

    qat_model.load_state_dict(qat_state, strict=False)
    return loaded_keys, skipped


__all__ = [
    "IdentityEntropyModel",
    "OpenDVCPFrameQATModel",
    "load_fp32_state_into_qat",
]
