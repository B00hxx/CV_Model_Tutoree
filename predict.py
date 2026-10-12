"""Step 3: classify individual images with the saved model.

    python predict.py data/gtsrb/Test/00000.png data/gtsrb/Test/00001.png --top-k 3
    python predict.py my_sign.jpg --ckpt outputs/best_model.pt
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from src.classes import CLASS_NAMES
from src.data import build_transforms
from src.utils import get_device, load_model


def main():
    parser = argparse.ArgumentParser(description="Predict GTSRB classes for image files")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_model(args.ckpt, device)
    _, eval_tf = build_transforms(ckpt["config"]["img_size"])

    batch = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in args.images]).to(device)
    with torch.no_grad():
        probs = model(batch).softmax(1).cpu()
    top_p, top_i = probs.topk(args.top_k, dim=1)

    for path, ps, idx in zip(args.images, top_p, top_i):
        print(f"\n{path}")
        for p, i in zip(ps.tolist(), idx.tolist()):
            print(f"  {p*100:6.2f}%  [{i:2d}] {CLASS_NAMES[i]}")


if __name__ == "__main__":
    main()
