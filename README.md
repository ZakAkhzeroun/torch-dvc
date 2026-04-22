# torch_dvc

## Python version
- Recommended: **Python 3.6**
- Reason: the project depends on `tensorflow==1.12.0`, which is typically used with Python 3.6.

## Required libraries
From `requirements.txt`:
- `numpy==1.19.5`
- `scipy==1.1.0`
- `Pillow==8.4.0`
- `tensorflow==1.12.0`
- `torch==1.10.2`

## Set up environment with conda

### Option 1 (recommended): use the provided installer
```bash
bash install_requirements.sh opendvc
```

Then activate the environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opendvc
```

### Option 2: manual conda setup
```bash
conda create -y -n opendvc python=3.6
conda activate opendvc
conda install -y numpy==1.19.5 scipy==1.1.0 pillow==8.4.0 tensorflow==1.12.0
conda install -y -c pytorch pytorch==1.10.2 cpuonly
```

## Run the model

### 1) P-frame inference (reconstructed frames + latents)
```bash
python infer.py \
  --input-dir BasketballPass \
  --img-out imgres \
  --lat-out latres \
  --weights-root OpenDVC_model \
  --weights-family psnr \
  --device cpu
```

Outputs:
- Reconstructed frames in `imgres/`
- Latents (`.npz`) in `latres/`

### 2) Decoder inference from saved latents
```bash
python infer_decoder.py \
  --lat-dir latres \
  --ref-dir imgres \
  --out-dir decimgres \
  --weights-root OpenDVC_model \
  --weights-family psnr \
  --device cpu
```

Outputs:
- Decoded frames in `decimgres/`
