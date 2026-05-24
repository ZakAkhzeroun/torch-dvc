# Training data layout

Place each training sequence in its own subdirectory under `training_data/`.

Expected per-sequence files:
- `im1_bpg444_QP22.png`, `im1_bpg444_QP27.png`, `im1_bpg444_QP32.png`, `im1_bpg444_QP37.png`
- `im1_level2_ssim.png`, `im1_level3_ssim.png`, `im1_level5_ssim.png`, `im1_level7_ssim.png`
- `im2.png`, `im3.png`, ... up to the number of frames used for training.

Example:
- `training_data/sequence_0001/im1_bpg444_QP27.png`
- `training_data/sequence_0001/im1_level5_ssim.png`
- `training_data/sequence_0001/im2.png`

Both training scripts now scan all immediate subdirectories in `training_data/` by default.
