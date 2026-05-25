import argparse
import gc
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import load
from src.models.fp32.CNN_img_torch import MVAnalysis, MVSynthesis, ResAnalysis, ResSynthesis
from src.models.fp32.MC_network_torch import MCNetwork
from src.models.fp32.motion_torch import MotionNetwork, dense_image_warp
from src.models.quant.opendvc_pframe_qat import OpenDVCPFrameQATModel, load_fp32_state_into_qat


try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None


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


class OpenDVCTrainMSModel(nn.Module):
    def __init__(
        self,
        num_filters: int,
        latent_channels: int,
        mode: str = "fp32",
        weight_bits: int = 16,
        act_bits: int = 16,
        warp_grad_through_flow: bool = False,
        use_likelihoods: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.use_likelihoods = use_likelihoods

        if mode == "qat":
            entropy_factory = EntropyBottleneckTorch if use_likelihoods else None
            self.qat_model = OpenDVCPFrameQATModel(
                num_filters=num_filters,
                latent_channels=latent_channels,
                weight_bits=weight_bits,
                act_bits=act_bits,
                warp_grad_through_flow=warp_grad_through_flow,
                entropy_factory=entropy_factory,
            )
        else:
            self.motion = MotionNetwork()
            self.mv_analysis = MVAnalysis(num_filters=num_filters, M=latent_channels)
            self.mv_synthesis = MVSynthesis(num_filters=num_filters, M=latent_channels)
            self.entropy_bottleneck_mv = EntropyBottleneckTorch(latent_channels)
            self.mc = MCNetwork()
            self.res_analysis = ResAnalysis(num_filters=num_filters, M=latent_channels)
            self.res_synthesis = ResSynthesis(num_filters=num_filters, M=latent_channels)
            self.entropy_bottleneck_res = EntropyBottleneckTorch(latent_channels)

    def forward(self, y0_com: torch.Tensor, y1_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode == "qat":
            out = self.qat_model(y0_com, y1_raw)
            return {
                "y1_com": out["y1_com"],
                "mv_likelihoods": out["mv_likelihoods"],
                "res_likelihoods": out["res_likelihoods"],
            }

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
            "mv_likelihoods": mv_likelihoods,
            "res_likelihoods": res_likelihoods,
        }

    def aux_loss(self) -> torch.Tensor:
        if self.mode == "qat":
            return self.qat_model.aux_loss()
        return self.entropy_bottleneck_mv.aux_loss() + self.entropy_bottleneck_res.aux_loss()


def _bpp_from_likelihoods(likelihoods: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if likelihoods is None:
        return None
    batch_size = int(target.shape[0])
    height = int(target.shape[-2])
    width = int(target.shape[-1])
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


def _collect_aux_params(model: nn.Module):
    # Collect parameters from modules that expose aux_loss(); deduplicate by id.
    aux_params = []
    seen = set()
    for module in model.modules():
        if module is model:
            continue
        if not hasattr(module, "aux_loss"):
            continue
        for param in module.parameters():
            pid = id(param)
            if pid in seen or not param.requires_grad:
                continue
            seen.add(pid)
            aux_params.append(param)
    return aux_params


def _require_likelihood(likelihood: Optional[torch.Tensor], name: str):
    if likelihood is None:
        raise RuntimeError(
            "Expected '{}' likelihood tensor, but model returned None. "
            "Use --recon-only-debug=true for reconstruction-only debug runs.".format(name)
        )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--l", type=int, default=32, choices=[8, 16, 32, 64])
    parser.add_argument("--N", type=int, default=128, choices=[128])
    parser.add_argument("--M", type=int, default=128, choices=[128])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=300000)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=20000)
    parser.add_argument("--train-data-root", default="training_data")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Explicit training dataset directory (overrides --train-data-root if set).",
    )
    parser.add_argument("--weights-root", default="OpenDVC_model")
    parser.add_argument("--testing", type=_parse_bool, default=False)
    parser.add_argument("--mode", default="fp32", choices=["fp32", "qat"])
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--act-bits", type=int, default=16)
    parser.add_argument(
        "--qat-warp-grad-through-flow",
        type=_parse_bool,
        default=False,
        help="QAT motion safety toggle. If false, disables gradient through warp grid flow path.",
    )
    parser.add_argument("--qat-lr-scale", type=float, default=0.1)
    parser.add_argument(
        "--use-likelihoods",
        type=_parse_bool,
        default=True,
        help="Include entropy likelihood-based bpp terms in the loss.",
    )
    parser.add_argument(
        "--recon-only-debug",
        type=_parse_bool,
        default=False,
        help="Debug-only mode: disable bpp terms and train with MS-SSIM term only.",
    )
    parser.add_argument("--fp32-init-checkpoint", default=None)
    parser.add_argument("--resume", default=None, help="Resume full training state from checkpoint.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if ms_ssim is None:
        raise ImportError(
            "`pytorch-msssim` is required for MS-SSIM training. Install with: pip install pytorch-msssim"
        )

    if args.resume is not None and args.fp32_init_checkpoint is not None:
        raise ValueError("Use either --resume or --fp32-init-checkpoint, not both.")
    if args.recon_only_debug and args.use_likelihoods:
        print("WARNING: --recon-only-debug=true, disabling likelihood rate terms for this run.")
        args.use_likelihoods = False

    i_level_map = {8: 2, 16: 3, 32: 5, 64: 7}
    i_level = i_level_map[args.l]

    channel = 3
    data_root = args.dataset_dir if args.dataset_dir is not None else args.train_data_root
    folder = None if args.testing else _discover_training_folders(data_root)
    device = torch.device(args.device)

    model = OpenDVCTrainMSModel(
        num_filters=args.N,
        latent_channels=args.M,
        mode=args.mode,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
        warp_grad_through_flow=args.qat_warp_grad_through_flow,
        use_likelihoods=args.use_likelihoods,
    ).to(device)
    model.train()

    if args.mode == "qat":
        print(
            "QAT mode enabled: weight_bits={} act_bits={} base_lr={} (lr_scale={}) warp_grad_through_flow={}".format(
                args.weight_bits,
                args.act_bits,
                args.lr * args.qat_lr_scale,
                args.qat_lr_scale,
                args.qat_warp_grad_through_flow,
            )
        )
        if not args.qat_warp_grad_through_flow:
            print(
                "WARNING: QAT warp flow-grid gradient is disabled for runtime stability on this environment."
            )
    if args.recon_only_debug:
        print("WARNING: Reconstruction-only debug mode active. bpp terms are disabled.")

    if args.mode == "qat" and args.fp32_init_checkpoint:
        ckpt = torch.load(args.fp32_init_checkpoint, map_location=device)
        fp32_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        loaded_keys, skipped = load_fp32_state_into_qat(model.qat_model, fp32_state)
        print("QAT init from FP32 checkpoint loaded={} skipped={}".format(len(loaded_keys), len(skipped)))

    base_lr = args.lr if args.mode == "fp32" else args.lr * args.qat_lr_scale
    main_optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    aux_params = _collect_aux_params(model)
    aux_optimizer = torch.optim.Adam(aux_params, lr=base_lr * 10.0) if aux_params else None

    run_name = "MS-SSIM_{}_{}".format(args.l, args.mode)
    if args.mode == "qat":
        run_name = "{}_w{}_a{}".format(run_name, args.weight_bits, args.act_bits)
    save_path = os.path.join(args.weights_root, run_name + "_model")
    os.makedirs(save_path, exist_ok=True)
    summary_writer = _safe_summary_writer(save_path)
    latest_path = os.path.join(save_path, "model_latest.pt")
    mode_latest_path = os.path.join(save_path, "model_{}_latest.pt".format(args.mode))

    iteration = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            main_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if aux_optimizer is not None and "aux_optimizer_state_dict" in ckpt and ckpt["aux_optimizer_state_dict"] is not None:
            aux_optimizer.load_state_dict(ckpt["aux_optimizer_state_dict"])
        iteration = int(ckpt.get("iteration", 0))
        print("Resumed training from {} at iteration {}".format(args.resume, iteration))

    while True:
        if iteration >= args.max_iters:
            break
        frames = 7
        lr = args.lr if iteration <= 200000 else args.lr / 10.0
        for param_group in main_optimizer.param_groups:
            param_group["lr"] = lr if args.mode == "fp32" else lr * args.qat_lr_scale
        if aux_optimizer is not None:
            for param_group in aux_optimizer.param_groups:
                param_group["lr"] = (lr if args.mode == "fp32" else lr * args.qat_lr_scale) * 10.0

        if args.testing:
            data = np.random.randint(
                0,
                256,
                size=(frames, args.batch_size, args.height, args.width, channel),
                dtype=np.uint8,
            ).astype(np.float32)
        else:
            data = np.zeros([frames, args.batch_size, args.height, args.width, channel], dtype=np.float32)
            data = load.load_data_ssim(
                data,
                frames,
                args.batch_size,
                args.height,
                args.width,
                channel,
                folder,
                i_level,
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
            if args.use_likelihoods:
                _require_likelihood(outputs["mv_likelihoods"], "mv_likelihoods")
                _require_likelihood(outputs["res_likelihoods"], "res_likelihoods")
                train_bpp_mv = _bpp_from_likelihoods(outputs["mv_likelihoods"], y1)
                train_bpp_res = _bpp_from_likelihoods(outputs["res_likelihoods"], y1)
            else:
                train_bpp_mv = y1.new_zeros(())
                train_bpp_res = y1.new_zeros(())
            frame_msssim = ms_ssim(outputs["y1_com"], y1, data_range=1.0, size_average=True)
            train_loss_total = args.l * (1.0 - frame_msssim) + (train_bpp_mv + train_bpp_res)

            main_optimizer.zero_grad(set_to_none=True)
            train_loss_total.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            main_optimizer.step()

            aux_loss_value = None
            if aux_optimizer is not None:
                aux_optimizer.zero_grad(set_to_none=True)
                aux_loss = model.aux_loss()
                if aux_loss.requires_grad:
                    aux_loss.backward()
                    aux_optimizer.step()
                    aux_loss_value = float(aux_loss.detach().cpu())

            # Detach recurrent reference to avoid backprop through the full sequence
            # and keep memory stable across 7-frame loops.
            f1_decoded = _to_hwc(outputs["y1_com"].detach())  # in [0, 1]

            iteration += 1

            if summary_writer is not None and iteration % args.log_interval == 0:
                total_bpp = train_bpp_mv + train_bpp_res
                current_lr = float(main_optimizer.param_groups[0]["lr"])
                print(
                    "[iter {}] mode={} bits(w/a)={}/{} loss_total={:.6f} ms-ssim={:.6f} "
                    "bpp_mv={:.6f} bpp_res={:.6f} bpp_total={:.6f} lr={:.8f} aux_loss={}".format(
                        iteration,
                        args.mode,
                        args.weight_bits if args.mode == "qat" else "fp32",
                        args.act_bits if args.mode == "qat" else "fp32",
                        float(train_loss_total.detach().cpu()),
                        float(frame_msssim.detach().cpu()),
                        float(train_bpp_mv.detach().cpu()),
                        float(train_bpp_res.detach().cpu()),
                        float(total_bpp.detach().cpu()),
                        current_lr,
                        "n/a" if aux_loss_value is None else "{:.6f}".format(aux_loss_value),
                    )
                )
                summary_writer.add_scalar("loss_total", float(train_loss_total.detach().cpu()), iteration)
                summary_writer.add_scalar("ms-ssim", float(frame_msssim.detach().cpu()), iteration)
                summary_writer.add_scalar("bpp_mv", float(train_bpp_mv.detach().cpu()), iteration)
                summary_writer.add_scalar("bpp_res", float(train_bpp_res.detach().cpu()), iteration)
                summary_writer.add_scalar(
                    "bits_total",
                    float(total_bpp.detach().cpu()),
                    iteration,
                )
                summary_writer.add_scalar("lr_main", current_lr, iteration)
                if aux_loss_value is not None:
                    summary_writer.add_scalar("aux_loss", aux_loss_value, iteration)
                summary_writer.add_text("run/mode", args.mode, iteration)
                summary_writer.add_text("run/bitwidth", "w{} a{}".format(args.weight_bits, args.act_bits), iteration)

            if iteration % args.save_interval == 0:
                checkpoint = {
                    "iteration": iteration,
                    "args": vars(args),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": main_optimizer.state_dict(),
                    "aux_optimizer_state_dict": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                }
                checkpoint_path = os.path.join(save_path, "model_{}_{}.pt".format(args.mode, iteration))
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, latest_path)
                torch.save(checkpoint, mode_latest_path)

        del data
        del f0_com
        del f1_raw
        del f1_decoded
        gc.collect()

    if summary_writer is not None:
        summary_writer.close()


if __name__ == "__main__":
    main()
