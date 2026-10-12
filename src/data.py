"""Dataset, transforms and DataLoaders for the Kaggle GTSRB layout.

Expected layout (straight from the Kaggle zip):
    data/gtsrb/
        Train.csv   -> Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path   (Path = Train/<cls>/<cls>_<track>_<frame>.png)
        Test.csv    -> same columns                                           (Path = Test/<id>.png)
        Train/  Test/  Meta/

Important detail: the training set is made of *tracks* of ~30 frames of the
same physical sign. A random image-level train/val split leaks near-duplicate
frames into validation and gives you a fake 99.9% val accuracy. We split by
track instead, so validation is an honest estimate of test performance.
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import Config

# Per-channel stats of GTSRB (RGB). Using dataset stats instead of ImageNet's.
MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


class GTSRBDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], transform, use_roi: bool = True):
        self.root = Path(root)
        self.rows = rows
        self.transform = transform
        self.use_roi = use_roi
        self.targets = [int(r["ClassId"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def load_image(self, idx: int) -> Image.Image:
        r = self.rows[idx]
        img = Image.open(self.root / r["Path"]).convert("RGB")
        if self.use_roi:
            # Roi.X2/Y2 are inclusive pixel coords -> +1 for PIL's exclusive box
            box = (int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]) + 1, int(r["Roi.Y2"]) + 1)
            img = img.crop(box)
        return img

    def __getitem__(self, idx: int):
        return self.transform(self.load_image(idx)), self.targets[idx]


def build_transforms(img_size: int):
    # NO horizontal flip: flipping "Turn left" makes it "Turn right" (label noise).
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def track_key(path: str) -> str:
    """'Train/20/00020_00007_00015.png' -> '00020_00007' (class + track id)."""
    stem = Path(path).stem
    return "_".join(stem.split("_")[:2])


def split_by_track(rows: list[dict], val_frac: float, seed: int):
    tracks = defaultdict(list)
    for r in rows:
        tracks[track_key(r["Path"])].append(r)

    # Stratify: hold out val_frac of the tracks *of each class*
    by_class = defaultdict(list)
    for key in tracks:
        by_class[key.split("_")[0]].append(key)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for keys in by_class.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * val_frac))
        for i, k in enumerate(keys):
            (val_rows if i < n_val else train_rows).extend(tracks[k])
    return train_rows, val_rows


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo Normalize for plotting. x: (C,H,W) or (N,C,H,W)."""
    mean = torch.tensor(MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def build_loaders(cfg: Config):
    train_tf, eval_tf = build_transforms(cfg.img_size)
    all_train = read_rows(cfg.data_dir / "Train.csv")
    test_rows = read_rows(cfg.data_dir / "Test.csv")
    train_rows, val_rows = split_by_track(all_train, cfg.val_frac, cfg.seed)

    train_ds = GTSRBDataset(cfg.data_dir, train_rows, train_tf, cfg.use_roi)
    val_ds = GTSRBDataset(cfg.data_dir, val_rows, eval_tf, cfg.use_roi)
    test_ds = GTSRBDataset(cfg.data_dir, test_rows, eval_tf, cfg.use_roi)

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }
