from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError("Pillow is required by load.py.") from exc


def _read_rgb_image(path):
    arr = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32)
    return arr

def load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP):

    for b in range(batch_size):

        path = Path(folder[np.random.randint(len(folder))])

        first = _read_rgb_image(path / ("im1_bpg444_QP" + str(I_QP) + ".png"))
        max_left = max(first.shape[1] - Width, 0)
        bb = np.random.randint(0, max_left + 1) if max_left > 0 else 0

        for f in range(frames):

            if f == 0:
                img = first
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = _read_rgb_image(path / ("im" + str(f + 1) + ".png"))
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data


def load_data_ssim(data, frames, batch_size, Height, Width, Channel, folder, I_level):

    for b in range(batch_size):

        path = Path(folder[np.random.randint(len(folder))])

        first = _read_rgb_image(path / ("im1_level" + str(I_level) + "_ssim.png"))
        max_left = max(first.shape[1] - Width, 0)
        bb = np.random.randint(0, max_left + 1) if max_left > 0 else 0

        for f in range(frames):

            if f == 0:
                img = first
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = _read_rgb_image(path / ("im" + str(f + 1) + ".png"))
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data
