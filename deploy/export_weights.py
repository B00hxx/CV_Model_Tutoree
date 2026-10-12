"""Deploy step 1: dump the trained weights to a plain NumPy archive.

Vitis AI 3.0 ships PyTorch 1.x, so we hand it plain arrays instead of a
checkpoint written by torch 2.x. Run in the training image (repo root):

    docker compose run --rm -v "$PWD/deploy:/app/deploy" --entrypoint python train deploy/export_weights.py

Writes outputs/deploy/gtsrb_float.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_model  # noqa: E402


def main():
    model, ckpt = load_model(Path("outputs/best_model.pt"), torch.device("cpu"))
    out = Path("outputs/deploy/gtsrb_float.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
    print(f"epoch {ckpt['epoch']}, val acc {ckpt['val_acc']*100:.2f}% -> {out}")


if __name__ == "__main__":
    main()
