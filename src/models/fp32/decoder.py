import argparse
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from .MC_network_torch import MCNetwork
    from .CNN_img_torch import MVSynthesis, ResSynthesis
    from .cnn_img_weights import (
        build_mv_synthesis_with_weights,
        build_res_synthesis_with_weights,
    )
    from .motion_torch import dense_image_warp
    from .weights import build_mc_with_weights
except ImportError:
    from src.models.fp32.MC_network_torch import MCNetwork
    from src.models.fp32.CNN_img_torch import MVSynthesis, ResSynthesis
    from src.models.fp32.cnn_img_weights import (
        build_mv_synthesis_with_weights,
        build_res_synthesis_with_weights,
    )
    from src.models.fp32.motion_torch import dense_image_warp
    from src.models.fp32.weights import build_mc_with_weights


class OpenDVCPFrameDecoder(nn.Module):
    """
    Torch decoder counterpart to OpenDVC_test_P-frame_decoder.py.

    This reproduces the neural decoding path after entropy decompression:
    1. MV synthesis
    2. Motion warping
    3. Motion compensation network
    4. Residual synthesis
    5. Final reconstruction

    Bitstream entropy decoding is not implemented in torch_dvc, so this class
    expects already-decoded latent tensors (`flow_latent_hat`, `res_latent_hat`).
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


def build_opendvc_pframe_decoder(
    checkpoint_prefix: Optional[str] = None,
    open_dvc_root: str = "OpenDVC_model/PSNR_256_model",
    basketball_data_root: str = "OpenDVC_model/PSNR_256_model",
    metric: str = "psnr",
    device: Union[str, torch.device] = "cpu",
) -> OpenDVCPFrameDecoder:
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
    res_synthesis = build_res_synthesis_with_weights(
        checkpoint_prefix=checkpoint_prefix,
        open_dvc_root=open_dvc_root,
        metric=metric,
        device=device,
    )

    model = OpenDVCPFrameDecoder(
        mv_synthesis=mv_synthesis,
        mc_net=mc_net,
        res_synthesis=res_synthesis,
    ).to(device)
    model.eval()
    return model


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load images for decoder.py.") from exc

    image = Image.open(str(path)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_image(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save images for decoder.py.") from exc

    image = np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image).save(str(path))


def _to_nchw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def _to_hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)


def _load_latent(path: Path, expected_batch: int = 1) -> np.ndarray:
    latent = np.load(str(path))
    if latent.ndim != 4:
        raise ValueError(
            "Expected latent tensor with 4 dimensions, got shape {} from {}".format(
                latent.shape, path
            )
        )
    if latent.shape[0] != expected_batch:
        raise ValueError(
            "Expected batch size {}, got {} in {}".format(
                expected_batch, latent.shape[0], path
            )
        )
    return latent.astype(np.float32, copy=False)


def _read_bitstream_lengths(path: Path):
    with path.open("rb") as ff:
        mv_len = np.frombuffer(ff.read(2), dtype=np.uint16)
        if len(mv_len) != 1:
            raise ValueError("Invalid bitstream header in {}".format(path))
        mv_bytes = int(mv_len[0])
        string_mv = ff.read(mv_bytes)
        string_res = ff.read()
    return mv_bytes, len(string_res), string_mv, string_res


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ref", default="ref.png")
    parser.add_argument("--com", default="dec.png")
    parser.add_argument("--bin", default=None)
    parser.add_argument("--flow-latent", default=None)
    parser.add_argument("--res-latent", default=None)
    parser.add_argument(
        "--latents-npz",
        default=None,
        help="Optional .npz containing flow_latent_hat and res_latent_hat arrays.",
    )
    parser.add_argument(
        "--metric",
        default="psnr",
        choices=["psnr", "ms-ssim", "msssim", "ms_ssim"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-prefix", default=None)
    args = parser.parse_args()

    ref_path = Path(args.ref)
    com_path = Path(args.com)
    com_path.parent.mkdir(parents=True, exist_ok=True)

    ref = _load_image(ref_path)
    if ref.shape[0] % 16 != 0 or ref.shape[1] % 16 != 0:
        raise ValueError(
            "Input height and width must be divisible by 16, got {}.".format(
                ref.shape[:2]
            )
        )

    flow_latent = None
    res_latent = None

    if args.latents_npz is not None:
        data = np.load(args.latents_npz)
        if "flow_latent_hat" not in data or "res_latent_hat" not in data:
            raise KeyError(
                "Expected keys 'flow_latent_hat' and 'res_latent_hat' in {}".format(
                    args.latents_npz
                )
            )
        flow_latent = data["flow_latent_hat"].astype(np.float32, copy=False)
        res_latent = data["res_latent_hat"].astype(np.float32, copy=False)

    if args.flow_latent is not None:
        flow_latent = _load_latent(Path(args.flow_latent))
    if args.res_latent is not None:
        res_latent = _load_latent(Path(args.res_latent))

    if args.bin is not None:
        mv_bytes, res_bytes, _, _ = _read_bitstream_lengths(Path(args.bin))
        print("Read bitstream {}".format(Path(args.bin).resolve()))
        print("Motion bytes: {}".format(mv_bytes))
        print("Residual bytes: {}".format(res_bytes))
        if flow_latent is None or res_latent is None:
            raise NotImplementedError(
                "Bitstream entropy decoding is not implemented in torch_dvc yet. "
                "Provide --flow-latent and --res-latent, or --latents-npz with "
                "decoded latent hats."
            )

    if flow_latent is None or res_latent is None:
        raise ValueError(
            "Decoder requires latent hats. Provide --flow-latent and --res-latent, "
            "or --latents-npz."
        )

    device = torch.device(args.device)
    model = build_opendvc_pframe_decoder(
        checkpoint_prefix=args.checkpoint_prefix,
        open_dvc_root=".",
        basketball_data_root=".",
        metric=args.metric,
        device=device,
    )

    y0 = _to_nchw(ref).to(device=device, dtype=torch.float32)
    flow_latent_t = torch.from_numpy(flow_latent).to(device=device, dtype=torch.float32)
    res_latent_t = torch.from_numpy(res_latent).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(y0, flow_latent_t, res_latent_t)

    decoded = _to_hwc(outputs["y1_com"])
    _save_image(com_path, decoded)

    print("Reference: {}".format(ref_path.resolve()))
    print("Output: {}".format(com_path.resolve()))
    print("Metric/checkpoint family: {}".format(args.metric))
    print("Decoded frame shape: {}".format(decoded.shape))


if __name__ == "__main__":
    main()


__all__ = [
    "OpenDVCPFrameDecoder",
    "build_opendvc_pframe_decoder",
]
