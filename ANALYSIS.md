# Compatibility Analysis for `OpenDVC_train_MS-SSIM.py` and `OpenDVC_train_PSNR.py`

## Scope
This analysis checks whether the two new training scripts can run and integrate with the current `torch_dvc` codebase **without modifying code now**. It lists required changes to make them usable for training and compatible with the existing project workflow.

## Result
In the current repository state, both scripts are **not directly runnable/integrated**. They are structurally close to the existing model code, but there are blocking issues in imports, data pipeline dependencies, runtime dependencies, and checkpoint interoperability.

## Findings and Required Changes

## 1) Broken imports to project modules (blocking)
Both training files import modules using paths that do not exist in this repo layout:
- `from torch_dvc.CNN_img_torch ...`
- `from torch_dvc.MC_network_torch ...`
- `from torch_dvc.motion_torch ...`

Current files are located under `src/models/fp32/`.

Evidence:
- `OpenDVC_train_PSNR.py:11-13`
- `OpenDVC_train_MS-SSIM.py:11-13`
- Existing module locations: `src/models/fp32/CNN_img_torch.py`, `src/models/fp32/MC_network_torch.py`, `src/models/fp32/motion_torch.py`

Required change:
- Update imports to the actual package path used by this repo (for example, `src.models.fp32...` or a robust relative import strategy).

## 2) Missing `load` module used by both trainers (blocking)
Both scripts require `import load` and call:
- `load.load_data(...)`
- `load.load_data_ssim(...)`

There is no `load.py` (or equivalent module) in this repository.

Evidence:
- `OpenDVC_train_PSNR.py:10, 196`
- `OpenDVC_train_MS-SSIM.py:10, 188`
- Repository search shows no definition of `load_data` / `load_data_ssim`

Required change:
- Add/port a dataset loader module that provides `load_data` and `load_data_ssim`, or replace this API with a PyTorch `Dataset`/`DataLoader` pipeline.

## 3) Required data artifact `folder.npy` is absent (blocking)
Both scripts expect `folder.npy` at runtime:
- `folder = np.load("folder.npy", allow_pickle=True)`

`folder.npy` is not present at repo root.

Evidence:
- `OpenDVC_train_PSNR.py:155`
- `OpenDVC_train_MS-SSIM.py:164`

Required change:
- Provide `folder.npy` generation/loading workflow, document its format, and ensure paths are configurable (CLI arg recommended).

## 4) Inference/training checkpoint format mismatch (high impact)
Training scripts save PyTorch checkpoints (`.pt`) with `state_dict`.
Existing inference (`infer.py`, `infer_decoder.py`) loads TensorFlow-style `model.ckpt` via TensorFlow readers in weight builders.

Evidence:
- Trainers save `.pt`: `OpenDVC_train_PSNR.py:247-256`, `OpenDVC_train_MS-SSIM.py:242-251`
- Inference resolves `model.ckpt`: `infer.py` and `infer_decoder.py` `_resolve_checkpoint_prefix`
- TF checkpoint loading in weight helpers: `src/models/fp32/weights.py`, `cnn_img_weights.py`, `motion_weights.py`

Required change:
- Add a PyTorch checkpoint loading path for inference, or add a conversion/export step from trainer `.pt` to current expected inference format.
- Decide canonical training output format for this repo.

## 5) Dependency gaps for training scripts (blocking unless installed)
Training scripts require dependencies not listed in current `requirements.txt`:
- `compressai` (optional in code, but needed for real entropy bottleneck behavior)
- `pytorch-msssim` (mandatory for MS-SSIM trainer)
- TensorBoard writer dependency (`torch.utils.tensorboard` -> usually requires `tensorboard` package)

Evidence:
- `OpenDVC_train_MS-SSIM.py`: explicit `ImportError` if `pytorch_msssim` missing
- Entropy bottleneck wrapper attempts `compressai.entropy_models.EntropyBottleneck`
- `requirements.txt` currently lists only numpy/scipy/Pillow/tensorflow/torch

Required change:
- Update environment/dependency spec for training mode (requirements extras or separate training requirements file).
- Validate versions against your Python/Torch stack.

## 6) Package execution context is fragile (high impact)
Because scripts are at repo root and import `torch_dvc.*`, running from inside `/home/alizakaria/work/torch_dvc` is likely to fail module resolution for `torch_dvc` (depends on parent path setup).

Required change:
- Make import paths execution-context safe (relative imports or standardized module entrypoint).
- Document exact launch command (e.g., `python -m ...`) once imports are fixed.

## 7) No integration entrypoints/documentation for training (medium)
Repo README documents inference only; no training instructions, dataset prep, or expected artifacts.

Required change:
- Add training section to README including:
  - dataset expectations
  - how to generate/provide `folder.npy`
  - command examples for PSNR/MS-SSIM
  - output checkpoint usage (and compatibility with inference)

## 8) Fallback entropy bottleneck is a compatibility fallback, not equivalent behavior (medium)
If `compressai` is unavailable, scripts use `FallbackEntropyBottleneck`, which is a simplified differentiable approximation. This may train, but rate modeling will not match OpenDVC/CompressAI behavior.

Required change:
- Decide whether fallback is acceptable for your target results.
- If not, make `compressai` mandatory for production training runs.

## 9) Minor robustness improvements recommended (non-blocking)
- Expose `folder.npy` path and data root as CLI arguments instead of hardcoding.
- Add resume-from-checkpoint option.
- Add deterministic seed controls for reproducibility.
- Validate image size constraints in data loader (OpenDVC path commonly expects dimensions divisible by 16).

## Compatibility Summary
- **Can they be used immediately for training?** No, not without changes.
- **Main blockers:** import paths, missing `load` module, missing `folder.npy`, dependency gaps.
- **Main integration gap:** trainer outputs (`.pt`) are not consumed by current inference path (`model.ckpt` TensorFlow loaders).

## Suggested Change Order
1. Fix module imports and execution entrypoint strategy.
2. Add/port data loader (`load_data`, `load_data_ssim`) and `folder.npy` pipeline.
3. Add/install missing training dependencies.
4. Define checkpoint interoperability (PyTorch-native inference loading or conversion).
5. Document complete training workflow in README.
