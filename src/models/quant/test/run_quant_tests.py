import sys
import traceback
import argparse
from pathlib import Path

import torch

# Allow running this script directly from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.quant.CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis
from src.models.quant.decoder_qat import build_opendvc_pframe_decoder_qat
from src.models.quant.MC_network_torch import MCNetwork
from src.models.quant.opendvc_pframe_qat import OpenDVCPFrameQATModel
from src.models.quant.quant_motion import MotionNetwork


def _assert_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise AssertionError("{} contains non-finite values".format(name))


def test_mc_network():
    model = MCNetwork(weight_bit_width=8, act_bit_width=8)
    model.eval()

    x = torch.rand(2, 8, 64, 64)
    with torch.no_grad():
        y = model(x)

    if y.shape != (2, 3, 64, 64):
        raise AssertionError("MC output shape mismatch: got {}".format(tuple(y.shape)))
    _assert_finite("MC output", y)


def test_motion_network():
    model = MotionNetwork(weight_bit_width=8, act_bit_width=8)
    model.eval()

    im1 = torch.rand(2, 3, 64, 64)
    im2 = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(im1, im2)

    if not isinstance(out, tuple) or len(out) != 6:
        raise AssertionError("Motion output must be tuple(flow_4, loss_0..loss_4)")

    flow_4 = out[0]
    if flow_4.shape != (2, 2, 64, 64):
        raise AssertionError("Motion flow shape mismatch: got {}".format(tuple(flow_4.shape)))
    _assert_finite("Motion flow", flow_4)

    for idx, loss_val in enumerate(out[1:], start=0):
        if loss_val.ndim != 0:
            raise AssertionError("loss_{} must be scalar tensor".format(idx))
        _assert_finite("loss_{}".format(idx), loss_val)


def test_cnn_img_modules():
    num_filters = 128
    latent_channels = 128

    mv_analysis = MVAnalysis(num_filters=num_filters, M=latent_channels)
    mv_synthesis = MVSynthesis(num_filters=num_filters, M=latent_channels)
    res_analysis = ResAnalysis(num_filters=num_filters, M=latent_channels)
    res_synthesis = ResSynthesis(num_filters=num_filters, M=latent_channels)

    mv_analysis.eval()
    mv_synthesis.eval()
    res_analysis.eval()
    res_synthesis.eval()

    flow = torch.rand(2, 2, 64, 64)
    frame = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        mv_lat = mv_analysis(flow)
        if mv_lat.shape != (2, latent_channels, 4, 4):
            raise AssertionError("MVAnalysis output shape mismatch: got {}".format(tuple(mv_lat.shape)))
        _assert_finite("MVAnalysis output", mv_lat)

        flow_rec = mv_synthesis(mv_lat)
        if flow_rec.shape != flow.shape:
            raise AssertionError("MVSynthesis output shape mismatch: got {}".format(tuple(flow_rec.shape)))
        _assert_finite("MVSynthesis output", flow_rec)

        res_lat = res_analysis(frame)
        if res_lat.shape != (2, latent_channels, 4, 4):
            raise AssertionError("ResAnalysis output shape mismatch: got {}".format(tuple(res_lat.shape)))
        _assert_finite("ResAnalysis output", res_lat)

        frame_rec = res_synthesis(res_lat)
        if frame_rec.shape != frame.shape:
            raise AssertionError("ResSynthesis output shape mismatch: got {}".format(tuple(frame_rec.shape)))
        _assert_finite("ResSynthesis output", frame_rec)


def test_qat_pframe_forward_backward():
    model = OpenDVCPFrameQATModel(
        num_filters=128,
        latent_channels=128,
        weight_bits=16,
        act_bits=16,
        entropy_factory=None,
    )
    # Keep eval mode for stability at random init; gradients still flow.
    model.eval()

    y0 = torch.rand(2, 3, 64, 64)
    y1 = torch.rand(2, 3, 64, 64)

    out = model(y0, y1)
    if out["y1_com"].shape != y1.shape:
        raise AssertionError("QAT P-frame output shape mismatch: got {}".format(tuple(out["y1_com"].shape)))
    _assert_finite("QAT y1_com", out["y1_com"])

    loss = torch.mean((out["y1_com"].float() - y1.float()) ** 2)
    loss.backward()
    if not any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise AssertionError("No gradients produced in QAT backward pass")


def test_qat_decoder_forward_backward():
    model = build_opendvc_pframe_decoder_qat(
        num_filters=128,
        latent_channels=128,
        weight_bits=16,
        act_bits=16,
        device=torch.device("cpu"),
    )
    model.train()

    y0 = torch.rand(2, 3, 64, 64)
    flow_latent_hat = torch.rand(2, 128, 4, 4)
    res_latent_hat = torch.rand(2, 128, 4, 4)
    out = model(y0, flow_latent_hat, res_latent_hat)

    if out["y1_com"].shape != y0.shape:
        raise AssertionError("QAT decoder output shape mismatch: got {}".format(tuple(out["y1_com"].shape)))
    _assert_finite("QAT decoder y1_com", out["y1_com"])

    loss = torch.mean((out["y1_com"].float() - y0.float()) ** 2)
    loss.backward()
    if not any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise AssertionError("No gradients produced in QAT decoder backward pass")


def _build_tests():
    return [
        ("MCNetwork", test_mc_network),
        ("MotionNetwork", test_motion_network),
        ("CNN_img Modules", test_cnn_img_modules),
        ("QAT P-frame", test_qat_pframe_forward_backward),
        ("QAT Decoder", test_qat_decoder_forward_backward),
    ]


def run_all_tests(selected=None) -> int:
    tests = _build_tests()
    if selected is not None:
        selected_l = selected.strip().lower()
        if selected_l == "decoder":
            tests = [(name, fn) for name, fn in tests if "decoder" in name.lower()]
        elif selected_l == "full":
            pass
        else:
            raise ValueError("Unknown test selection '{}'. Use 'full' or 'decoder'.".format(selected))

    failures = []
    for name, fn in tests:
        try:
            fn()
            print("[PASS] {}".format(name))
        except Exception:
            failures.append(name)
            print("[FAIL] {}".format(name))
            traceback.print_exc()

    if failures:
        print("\nQuant test summary: FAILED ({}/{})".format(len(failures), len(tests)))
        print("Failed modules: {}".format(", ".join(failures)))
        return 1

    print("\nQuant test summary: PASSED ({}/{})".format(len(tests), len(tests)))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tests", default="full", choices=["full", "decoder"])
    args = parser.parse_args()
    sys.exit(run_all_tests(selected=args.tests))
