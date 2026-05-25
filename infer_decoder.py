import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.fp32 import build_opendvc_pframe_decoder
from src.models.quant.decoder_qat import (
    build_opendvc_pframe_decoder_qat,
    load_fp32_state_into_qat_decoder,
)


def _load_image(path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to run infer_decoder.py.") from exc

    img = Image.open(str(path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _save_image(path, array):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to run infer_decoder.py.") from exc

    array = np.clip(np.round(array * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(str(path))


def _to_nchw(array):
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def _to_hwc(tensor):
    return tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)


def _pad_to_multiple_of_16(arr):
    h, w = arr.shape[:2]
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16

    if pad_h == 0 and pad_w == 0:
        return arr, (h, w)

    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (h, w)


def _find_latents(lat_dir):
    return sorted(lat_dir.glob("*.npz"))


def _find_reference_image(ref_dir, lat_files):
    if not lat_files:
        raise ValueError("No latent files found.")

    first_lat_stem = lat_files[0].stem
    if first_lat_stem.startswith("f") and first_lat_stem[1:].isdigit():
        first_idx = int(first_lat_stem[1:])
        if first_idx > 1:
            candidate = ref_dir / ("f{:03d}.png".format(first_idx - 1))
            if candidate.exists():
                return candidate

    candidates = sorted(ref_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError("No reference images found in {}".format(ref_dir))
    return candidates[0]


def _load_latent_pair(path):
    data = np.load(str(path))
    if "flow_latent_hat" in data:
        flow_latent = data["flow_latent_hat"]
    elif "flow_latent" in data:
        flow_latent = data["flow_latent"]
    else:
        raise KeyError("Missing flow latent in {}".format(path))

    if "res_latent_hat" in data:
        res_latent = data["res_latent_hat"]
    elif "res_latent" in data:
        res_latent = data["res_latent"]
    else:
        raise KeyError("Missing residual latent in {}".format(path))

    if flow_latent.ndim != 4 or res_latent.ndim != 4:
        raise ValueError("Latents in {} must be 4D tensors.".format(path))

    return flow_latent.astype(np.float32, copy=False), res_latent.astype(np.float32, copy=False)


def _normalize_family(family):
    token = family.strip().lower()
    if token in ("psnr",):
        return "psnr", "PSNR"
    if token in ("ms-ssim", "msssim", "ms_ssim", "mmsi"):
        return "ms-ssim", "MS-SSIM"
    raise ValueError("Unsupported weights family: {}".format(family))


def _resolve_checkpoint_prefix(args):
    if args.checkpoint_prefix is not None:
        return args.checkpoint_prefix, args.metric

    metric_from_family, folder_family = _normalize_family(args.weights_family)
    metric = metric_from_family if args.metric is None else args.metric

    if args.weights_resolution is None:
        default_res = 256 if folder_family == "PSNR" else 8
        resolution = default_res
    else:
        resolution = int(args.weights_resolution)

    ckpt_dir = Path(args.weights_root) / "{}_{}_model".format(folder_family, resolution)
    ckpt_prefix = ckpt_dir / "model.ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError("Weights directory not found: {}".format(ckpt_dir))
    if not (ckpt_dir / "model.ckpt.index").exists():
        raise FileNotFoundError("Checkpoint index not found: {}".format(ckpt_dir / "model.ckpt.index"))

    return str(ckpt_prefix), metric


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lat-dir", default="latres")
    parser.add_argument("--ref-dir", default="imgres")
    parser.add_argument("--ref-image", default=None)
    parser.add_argument("--out-dir", default="decimgres")
    parser.add_argument("--metric", default=None, choices=["psnr", "ms-ssim", "msssim", "ms_ssim"])
    parser.add_argument("--weights-family", default="psnr")
    parser.add_argument("--weights-resolution", type=int, default=None)
    parser.add_argument("--weights-root", default="OpenDVC_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--mode", default="fp32", choices=["fp32", "qat"])
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--act-bits", type=int, default=16)
    parser.add_argument("--fp32-init-checkpoint", default=None)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    lat_dir = Path(args.lat_dir)
    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_files = _find_latents(lat_dir)
    if not lat_files:
        raise FileNotFoundError("No .npz files found in {}".format(lat_dir))
    if args.max_frames is not None:
        lat_files = lat_files[: args.max_frames]

    if args.ref_image is not None:
        ref_path = Path(args.ref_image)
        if not ref_path.exists():
            raise FileNotFoundError("Reference image not found: {}".format(ref_path))
    else:
        ref_path = _find_reference_image(ref_dir, lat_files)

    if args.mode == "fp32":
        checkpoint_prefix, metric = _resolve_checkpoint_prefix(args)
        decoder_model = build_opendvc_pframe_decoder(
            checkpoint_prefix=checkpoint_prefix,
            metric=metric,
            device=args.device,
        )
    else:
        metric = args.metric if args.metric is not None else "psnr"
        decoder_model = build_opendvc_pframe_decoder_qat(
            num_filters=args.num_filters,
            latent_channels=args.latent_channels,
            weight_bits=args.weight_bits,
            act_bits=args.act_bits,
            device=torch.device(args.device),
        )
        if args.fp32_init_checkpoint is not None:
            ckpt = torch.load(args.fp32_init_checkpoint, map_location=args.device)
            fp32_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            loaded, skipped, _ = load_fp32_state_into_qat_decoder(decoder_model, fp32_state)
            print("QAT decoder init from FP32 checkpoint loaded={} skipped={}".format(loaded, skipped))

    decoder_model.eval()

    ref = _load_image(ref_path)
    ref_padded, ref_hw = _pad_to_multiple_of_16(ref)
    prev_rec_t = _to_nchw(ref_padded).to(args.device)

    _save_image(out_dir / ref_path.name, ref)

    with torch.no_grad():
        for lat_path in lat_files:
            flow_latent_np, res_latent_np = _load_latent_pair(lat_path)

            flow_latent_t = torch.from_numpy(flow_latent_np).to(args.device)
            res_latent_t = torch.from_numpy(res_latent_np).to(args.device)

            out = decoder_model(prev_rec_t, flow_latent_t, res_latent_t)

            y1_com = _to_hwc(out["y1_com"])
            y1_com = y1_com[: ref_hw[0], : ref_hw[1], :]

            out_name = lat_path.stem + ".png"
            _save_image(out_dir / out_name, y1_com)

            prev_rec_t = out["y1_com"].detach()

    print("Decoding finished.")
    print("Saved decoded images to: {}".format(out_dir.resolve()))
    print("Decoder mode: {}".format(args.mode))


if __name__ == "__main__":
    main()
