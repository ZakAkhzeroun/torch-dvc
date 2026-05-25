import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.quant.opendvc_pframe_qat import OpenDVCPFrameQATModel


def main() -> int:
    model = OpenDVCPFrameQATModel(
        num_filters=128,
        latent_channels=128,
        weight_bits=16,
        act_bits=16,
        entropy_factory=None,
    )
    # Keep eval mode for random-init stability; gradients still flow.
    model.eval()

    y0 = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    y1 = torch.rand(2, 3, 64, 64, dtype=torch.float32)

    out = model(y0, y1)
    y1_com = out["y1_com"]
    assert y1_com.shape == y1.shape, "Unexpected reconstruction shape: {}".format(tuple(y1_com.shape))

    # Keep reconstruction loss in fp32 for numerical stability.
    recon_loss = torch.mean((y1_com.float() - y1.float()) ** 2)
    recon_loss.backward()

    grad_ok = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "No gradients were produced by backward pass"

    print("QAT dummy test passed")
    print("recon_loss={:.6f}".format(float(recon_loss.detach().cpu())))
    print("mv_likelihoods={}".format(out["mv_likelihoods"]))
    print("res_likelihoods={}".format(out["res_likelihoods"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
