import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.quant.CNN_img_torch import MVSynthesis, ResSynthesis
from src.models.quant.MC_network_torch import MCNetwork
from src.models.quant.opendvc_pframe_qat import OpenDVCPFrameQATModel
from src.models.quant.quant_motion import MotionNetwork, dense_image_warp


def stage(name):
    print("[stage] {}".format(name), flush=True)


def run_mc():
    stage("MC init")
    m = MCNetwork(weight_bit_width=16, act_bit_width=16).train()
    x = torch.rand(1, 8, 256, 256)
    stage("MC forward")
    y = m(x)
    stage("MC backward")
    y.mean().backward()


def run_motion():
    stage("Motion init")
    m = MotionNetwork(weight_bit_width=16, act_bit_width=16, warp_grad_through_flow=True).train()
    a = torch.rand(1, 3, 256, 256)
    b = torch.rand(1, 3, 256, 256)
    stage("Motion forward")
    out = m(a, b)
    stage("Motion backward")
    (out[0].mean() + sum(v for v in out[1:])).backward()


def run_motion_no_warp_grid_grad():
    stage("Motion(no-warp-grid-grad) init")
    m = MotionNetwork(weight_bit_width=16, act_bit_width=16, warp_grad_through_flow=False).train()
    a = torch.rand(1, 3, 256, 256)
    b = torch.rand(1, 3, 256, 256)
    stage("Motion(no-warp-grid-grad) forward")
    out = m(a, b)
    stage("Motion(no-warp-grid-grad) backward")
    (out[0].mean() + sum(v for v in out[1:])).backward()


def run_decoder_path():
    stage("MV/Res synth + MC init")
    mv = MVSynthesis(num_filters=128, M=128, weight_bit_width=16).train()
    rs = ResSynthesis(num_filters=128, M=128, weight_bit_width=16).train()
    mc = MCNetwork(weight_bit_width=16, act_bit_width=16).train()

    y0 = torch.rand(1, 3, 256, 256)
    flow_lat = torch.rand(1, 128, 16, 16)
    res_lat = torch.rand(1, 128, 16, 16)

    stage("MV synthesis forward")
    flow = mv(flow_lat)
    stage("dense warp forward")
    y1_warp = dense_image_warp(y0, flow)
    stage("MC forward")
    y1_mc = mc(torch.cat([flow, y0, y1_warp], dim=1))
    stage("Res synthesis forward")
    res = rs(res_lat)
    y = torch.clamp(y1_mc + res, 0.0, 1.0)

    stage("Decoder path backward")
    y.mean().backward()


def run_pframe_qat():
    stage("Pframe QAT init")
    m = OpenDVCPFrameQATModel(
        num_filters=128,
        latent_channels=128,
        weight_bits=16,
        act_bits=16,
        entropy_factory=None,
    ).train()
    y0 = torch.rand(1, 3, 256, 256)
    y1 = torch.rand(1, 3, 256, 256)

    stage("Pframe QAT forward")
    out = m(y0, y1)
    stage("Pframe QAT backward")
    torch.mean((out["y1_com"] - y1) ** 2).backward()


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    print("Python:", sys.version, flush=True)
    print("Torch:", torch.__version__, flush=True)

    tests = [
        ("mc", run_mc),
        ("motion_no_warp_grid_grad", run_motion_no_warp_grid_grad),
        ("motion", run_motion),
        ("decoder_path", run_decoder_path),
        ("pframe_qat", run_pframe_qat),
    ]

    for name, fn in tests:
        stage("BEGIN {}".format(name))
        fn()
        stage("END {}".format(name))

    print("All stages passed", flush=True)


if __name__ == "__main__":
    main()
