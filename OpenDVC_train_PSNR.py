import argparse
import gc
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

import load
from src.models.fp32.CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis
from src.models.fp32.MC_network_torch import MCNetwork
from src.models.fp32.motion_torch import MotionNetwork, dense_image_warp


class FallbackEntropyBottleneck(nn.Module):
    """Simple differentiable entropy model fallback when CompressAI is unavailable."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if training:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            x_hat = x + noise
        else:
            x_hat = torch.round(x)

        scale = torch.nn.functional.softplus(self.log_scale) + 1e-6
        centered = torch.abs(x - x_hat)
        likelihood = torch.exp(-centered / scale) / (2.0 * scale)
        likelihood = torch.clamp(likelihood, min=1e-9)
        return x_hat, likelihood

    def aux_loss(self) -> torch.Tensor:
        return self.log_scale.sum() * 0.0


class EntropyBottleneckTorch(nn.Module):
    """
    Replacement for tfc.EntropyBottleneck.

    Uses CompressAI EntropyBottleneck when available, with a local fallback.
    """

    def __init__(self, channels: int):
        super().__init__()
        self._use_compressai = False
        try:
            from compressai.entropy_models import EntropyBottleneck as CEntropyBottleneck

            self.backend = CEntropyBottleneck(channels)
            self._use_compressai = True
        except Exception:
            self.backend = FallbackEntropyBottleneck(channels)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_compressai:
            x_hat, likelihood = self.backend(x, training=training)
            return x_hat, torch.clamp(likelihood, min=1e-9)
        return self.backend(x, training=training)

    def aux_loss(self) -> torch.Tensor:
        if self._use_compressai and hasattr(self.backend, "loss"):
            return self.backend.loss()
        if hasattr(self.backend, "aux_loss"):
            return self.backend.aux_loss()
        return next(self.parameters()).new_zeros(())


class OpenDVCTrainPSNRModel(nn.Module):
    def __init__(self, num_filters: int, latent_channels: int):
        super().__init__()
        self.motion = MotionNetwork()
        self.mv_analysis = MVAnalysis(num_filters=num_filters, M=latent_channels)
        self.mv_synthesis = MVSynthesis(num_filters=num_filters, M=latent_channels)
        self.entropy_bottleneck_mv = EntropyBottleneckTorch(latent_channels)
        self.mc = MCNetwork()
        self.res_analysis = ResAnalysis(num_filters=num_filters, M=latent_channels)
        self.res_synthesis = ResSynthesis(num_filters=num_filters, M=latent_channels)
        self.entropy_bottleneck_res = EntropyBottleneckTorch(latent_channels)

    def forward(self, y0_com: torch.Tensor, y1_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        flow_tensor, _, _, _, _, _ = self.motion(y0_com, y1_raw)

        flow_latent = self.mv_analysis(flow_tensor)
        flow_latent_hat, mv_likelihoods = self.entropy_bottleneck_mv(flow_latent, training=self.training)
        flow_hat = self.mv_synthesis(flow_latent_hat)

        y1_warp = dense_image_warp(y0_com, flow_hat)
        mc_input = torch.cat([flow_hat, y0_com, y1_warp], dim=1)
        y1_mc = self.mc(mc_input)

        res = y1_raw - y1_mc
        res_latent = self.res_analysis(res)
        res_latent_hat, res_likelihoods = self.entropy_bottleneck_res(res_latent, training=self.training)
        res_hat = self.res_synthesis(res_latent_hat)

        y1_com = torch.clamp(res_hat + y1_mc, 0.0, 1.0)

        return {
            "y1_com": y1_com,
            "y1_warp": y1_warp,
            "y1_mc": y1_mc,
            "mv_likelihoods": mv_likelihoods,
            "res_likelihoods": res_likelihoods,
        }

    def aux_loss(self) -> torch.Tensor:
        return self.entropy_bottleneck_mv.aux_loss() + self.entropy_bottleneck_res.aux_loss()


def _bpp_from_likelihoods(likelihoods: torch.Tensor, height: int, width: int, batch_size: int) -> torch.Tensor:
    return torch.sum(-torch.log2(torch.clamp(likelihoods, min=1e-9))) / (height * width * batch_size)


def _to_nchw(batch_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(batch_hwc).to(device=device, dtype=torch.float32) / 255.0
    return tensor.permute(0, 3, 1, 2).contiguous()


def _to_hwc(batch_nchw: torch.Tensor) -> np.ndarray:
    return batch_nchw.detach().permute(0, 2, 3, 1).cpu().numpy()


def _safe_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None


def _discover_training_folders(train_data_root: str) -> np.ndarray:
    root = Path(train_data_root)
    if not root.exists():
        raise FileNotFoundError("Training data root not found: {}".format(root))

    folders = sorted([str(p) for p in root.iterdir() if p.is_dir()])
    if not folders:
        raise ValueError("No sequence folders found under {}".format(root))
    return np.asarray(folders, dtype=object)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected boolean value, got '{}'".format(value))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--l", type=int, default=1024, choices=[256, 512, 1024, 2048])
    parser.add_argument("--N", type=int, default=128, choices=[128])
    parser.add_argument("--M", type=int, default=128, choices=[128])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=700000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=20000)
    parser.add_argument("--train-data-root", default="training_data")
    parser.add_argument("--weights-root", default="OpenDVC_model")
    parser.add_argument("--testing", type=_parse_bool, default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    i_qp_map = {256: 37, 512: 32, 1024: 27, 2048: 22}
    i_qp = i_qp_map[args.l]

    channel = 3
    folder = None if args.testing else _discover_training_folders(args.train_data_root)
    device = torch.device(args.device)

    model = OpenDVCTrainPSNRModel(num_filters=args.N, latent_channels=args.M).to(device)
    model.train()

    main_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    aux_params = list(model.entropy_bottleneck_mv.parameters()) + list(model.entropy_bottleneck_res.parameters())
    aux_optimizer = torch.optim.Adam(aux_params, lr=args.lr * 10.0)

    save_path = os.path.join(args.weights_root, "PSNR_{}_model".format(args.l))
    os.makedirs(save_path, exist_ok=True)
    summary_writer = _safe_summary_writer(save_path)

    latest_path = os.path.join(save_path, "model_latest.pt")

    iteration = 0
    while True:
        if iteration <= 100000:
            frames = 2
            if iteration <= 20000:
                phase = "mv"
            elif iteration <= 40000:
                phase = "mc"
            else:
                phase = "all"
        else:
            frames = 7
            phase = "all"

        if iteration <= 300000:
            lr = args.lr
        elif iteration <= 600000:
            lr = args.lr / 10.0
        else:
            lr = args.lr / 100.0

        for param_group in main_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in aux_optimizer.param_groups:
            param_group["lr"] = lr * 10.0

        if args.testing:
            data = np.random.randint(
                0,
                256,
                size=(frames, args.batch_size, args.height, args.width, channel),
                dtype=np.uint8,
            ).astype(np.float32)
        else:
            data = np.zeros([frames, args.batch_size, args.height, args.width, channel], dtype=np.float32)
            data = load.load_data(
                data,
                frames,
                args.batch_size,
                args.height,
                args.width,
                channel,
                folder,
                i_qp,
            )

        for ff in range(frames - 1):
            if ff == 0:
                f0_com = data[0]
                f1_raw = data[1]
            else:
                f0_com = f1_decoded * 255.0
                f1_raw = data[ff + 1]

            y0 = _to_nchw(f0_com, device)
            y1 = _to_nchw(f1_raw, device)

            outputs = model(y0, y1)
            train_bpp_mv = _bpp_from_likelihoods(outputs["mv_likelihoods"], args.height, args.width, args.batch_size)
            train_bpp_res = _bpp_from_likelihoods(outputs["res_likelihoods"], args.height, args.width, args.batch_size)

            total_mse = torch.mean((outputs["y1_com"] - y1) ** 2)
            warp_mse = torch.mean((outputs["y1_warp"] - y1) ** 2)
            mc_mse = torch.mean((y1 - outputs["y1_mc"]) ** 2)

            train_loss_total = args.l * total_mse + (train_bpp_mv + train_bpp_res)
            train_loss_mv = args.l * warp_mse + train_bpp_mv
            train_loss_mc = args.l * mc_mse + train_bpp_mv

            if phase == "mv":
                train_loss = train_loss_mv
            elif phase == "mc":
                train_loss = train_loss_mc
            else:
                train_loss = train_loss_total

            main_optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            main_optimizer.step()

            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = model.aux_loss()
            if aux_loss.requires_grad:
                aux_loss.backward()
                aux_optimizer.step()

            f1_decoded = _to_hwc(outputs["y1_com"])  # in [0, 1]
            psnr = 10.0 * torch.log10(1.0 / torch.clamp(total_mse, min=1e-12))

            print("Training_OpenDVC Iteration:", iteration)
            iteration += 1

            if summary_writer is not None and iteration % args.log_interval == 0:
                summary_writer.add_scalar("psnr", float(psnr.detach().cpu()), iteration)
                summary_writer.add_scalar(
                    "bits_total",
                    float((train_bpp_mv + train_bpp_res).detach().cpu()),
                    iteration,
                )

            if iteration % args.save_interval == 0:
                checkpoint_path = os.path.join(save_path, "model_{}.pt".format(iteration))
                torch.save(
                    {
                        "iteration": iteration,
                        "args": vars(args),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": main_optimizer.state_dict(),
                        "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                torch.save(
                    {
                        "iteration": iteration,
                        "args": vars(args),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": main_optimizer.state_dict(),
                        "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                    },
                    latest_path,
                )

        if iteration > args.max_iters:
            break

        del data
        del f0_com
        del f1_raw
        del f1_decoded
        gc.collect()

    if summary_writer is not None:
        summary_writer.close()


if __name__ == "__main__":
    main()
