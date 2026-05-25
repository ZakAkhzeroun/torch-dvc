import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.quant.decoder_qat import build_opendvc_pframe_decoder_qat


def main() -> int:
    model = build_opendvc_pframe_decoder_qat(
        num_filters=128,
        latent_channels=128,
        weight_bits=16,
        act_bits=16,
        device=torch.device("cpu"),
    )
    model.train()

    y0 = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    flow_latent_hat = torch.rand(2, 128, 4, 4, dtype=torch.float32)
    res_latent_hat = torch.rand(2, 128, 4, 4, dtype=torch.float32)

    out = model(y0, flow_latent_hat, res_latent_hat)
    y1_com = out["y1_com"]
    assert y1_com.shape == y0.shape, "Unexpected decoder output shape: {}".format(tuple(y1_com.shape))
    assert torch.isfinite(y1_com).all(), "Decoder output contains non-finite values"

    recon_loss = torch.mean((y1_com.float() - y0.float()) ** 2)
    recon_loss.backward()
    grad_ok = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_ok, "No gradients were produced by backward pass"

    print("QAT decoder dummy test passed")
    print("recon_loss={:.6f}".format(float(recon_loss.detach().cpu())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
