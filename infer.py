import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.fp32 import build_opendvc_pframe_model


def _load_image(path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to run infer.py.") from exc

    img = Image.open(str(path)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _save_image(path, array):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to run infer.py.") from exc

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

    padded = np.pad(
        arr,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )
    return padded, (h, w)


def _find_frames(input_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(input_dir.glob(ext))
    return sorted(paths)


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
    parser.add_argument("--input-dir", default="BasketballPass")
    parser.add_argument("--img-out", default="imgres")
    parser.add_argument("--lat-out", default="latres")
    parser.add_argument("--metric", default=None, choices=["psnr", "ms-ssim", "msssim", "ms_ssim"])
    parser.add_argument("--weights-family", default="psnr")
    parser.add_argument("--weights-resolution", type=int, default=None)
    parser.add_argument("--weights-root", default="OpenDVC_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    img_out = Path(args.img_out)
    lat_out = Path(args.lat_out)
    img_out.mkdir(parents=True, exist_ok=True)
    lat_out.mkdir(parents=True, exist_ok=True)

    frame_paths = _find_frames(input_dir)
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames to run P-frame inference.")
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
        if len(frame_paths) < 2:
            raise ValueError("max-frames must allow at least 2 frames.")

    checkpoint_prefix, metric = _resolve_checkpoint_prefix(args)

    model = build_opendvc_pframe_model(
        checkpoint_prefix=checkpoint_prefix,
        metric=metric,
        device=args.device,
    )
    model.eval()

    # First frame is the reference and copied directly as reconstruction.
    ref = _load_image(frame_paths[0])
    ref_padded, ref_hw = _pad_to_multiple_of_16(ref)
    ref_t = _to_nchw(ref_padded).to(args.device)
    _save_image(img_out / frame_paths[0].name, ref)

    prev_rec_t = ref_t

    with torch.no_grad():
        for idx in range(1, len(frame_paths)):
            cur_path = frame_paths[idx]
            cur = _load_image(cur_path)

            if cur.shape[0] != ref.shape[0] or cur.shape[1] != ref.shape[1]:
                raise ValueError(
                    "All frames must have same resolution; got {} for {} and {} for {}".format(
                        ref.shape[:2], frame_paths[0].name, cur.shape[:2], cur_path.name
                    )
                )

            cur_padded, cur_hw = _pad_to_multiple_of_16(cur)
            if cur_hw != ref_hw:
                raise ValueError("Unexpected frame size mismatch after padding.")

            y1_raw_t = _to_nchw(cur_padded).to(args.device)
            out = model(prev_rec_t, y1_raw_t)

            y1_com = _to_hwc(out["y1_com"])
            y1_com = y1_com[: cur_hw[0], : cur_hw[1], :]
            _save_image(img_out / cur_path.name, y1_com)

            np.savez_compressed(
                str(lat_out / (cur_path.stem + ".npz")),
                flow_latent=out["flow_latent"].detach().cpu().numpy(),
                flow_latent_hat=out["flow_latent_hat"].detach().cpu().numpy(),
                res_latent=out["res_latent"].detach().cpu().numpy(),
                res_latent_hat=out["res_latent_hat"].detach().cpu().numpy(),
            )

            prev_rec_t = out["y1_com"].detach()

    print("Inference finished.")
    print("Saved reconstructed images to: {}".format(img_out.resolve()))
    print("Saved latent tensors to: {}".format(lat_out.resolve()))


if __name__ == "__main__":
    main()
