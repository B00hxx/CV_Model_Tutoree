"""Deploy step 2: quantize TrafficSignCNN to INT8 with Vitis AI and export an .xmodel.

Run inside the Vitis AI 3.0 PyTorch container, from the repo root:

    python deploy/quantize.py --mode float           # float accuracy of the deploy model (no Vitis AI needed)
    python deploy/quantize.py --mode calib           # collect activation ranges on training images
    python deploy/quantize.py --mode test            # INT8 accuracy on the test set
    python deploy/quantize.py --mode test --deploy   # export outputs/deploy/quantize_result/TrafficSignCNN_int.xmodel

Self-contained on purpose: it only imports src/model.py, because src/data.py uses
Python 3.9+ syntax and the Vitis AI container runs Python 3.8.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import TrafficSignCNN  # noqa: E402

# Must match src/data.py
MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)
IMG_SIZE = 48


class SignRows(Dataset):
    """GTSRB rows -> ROI crop -> eval transform (same as src/data.py, no augmentation)."""

    def __init__(self, root, rows):
        self.root = Path(root)
        self.rows = rows
        self.tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(self.root / r["Path"]).convert("RGB")
        img = img.crop((int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]) + 1, int(r["Roi.Y2"]) + 1))
        return self.tf(img), int(r["ClassId"])


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_deploy_model(npz_path):
    model = TrafficSignCNN(num_classes=43)
    # Identical to AdaptiveAvgPool2d((3, 3)) for a 6x6 map (48x48 input), and a plain
    # 2x2 average pool maps directly onto the DPU.
    model.pool = nn.AvgPool2d(2)
    state = {k: torch.from_numpy(v) for k, v in np.load(npz_path).items()}
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def accuracy(model, loader):
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["float", "calib", "test"], required=True)
    p.add_argument("--deploy", action="store_true", help="test mode: export the xmodel (batch 1)")
    p.add_argument("--data-dir", type=Path, default=Path("archive"))
    p.add_argument("--weights", type=Path, default=Path("outputs/deploy/gtsrb_float.npz"))
    p.add_argument("--out", type=Path, default=Path("outputs/deploy/quantize_result"))
    p.add_argument("--calib-images", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    model = build_deploy_model(args.weights)
    test_rows = read_rows(args.data_dir / "Test.csv")
    # Calibrate on training images only; the test set stays untouched for scoring
    calib_rows = random.Random(0).sample(read_rows(args.data_dir / "Train.csv"), args.calib_images)

    def loader(rows, bs):
        return DataLoader(SignRows(args.data_dir, rows), batch_size=bs, num_workers=args.num_workers)

    if args.mode == "float":
        print(f"float deploy model, test acc: {accuracy(model, loader(test_rows, args.batch_size))*100:.2f}%")
        return

    from pytorch_nndct.apis import torch_quantizer

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    quantizer = torch_quantizer(args.mode, model, (dummy,), output_dir=str(args.out),
                                device=torch.device("cpu"))
    qmodel = quantizer.quant_model

    if args.mode == "calib":
        acc = accuracy(qmodel, loader(calib_rows, args.batch_size))
        quantizer.export_quant_config()
        print(f"calibrated on {len(calib_rows)} images (acc on them {acc*100:.2f}%) -> {args.out}")
    elif args.deploy:
        # xmodel export needs a batch-1 forward pass in test mode
        accuracy(qmodel, loader(test_rows[:1], 1))
        quantizer.export_xmodel(output_dir=str(args.out), deploy_check=False)
        print(f"exported {args.out}/TrafficSignCNN_int.xmodel")
    else:
        print(f"INT8 (simulated) test acc: {accuracy(qmodel, loader(test_rows, args.batch_size))*100:.2f}%")


if __name__ == "__main__":
    main()
