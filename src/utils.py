"""Seeding, device selection, checkpoint save/load, early stopping."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from .model import TrafficSignCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # fixed input size -> let cuDNN pick fastest kernels
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, cfg, epoch: int, val_acc: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the state_dict (not the whole model object) + everything needed to rebuild it
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    c = ckpt["config"]
    model = TrafficSignCNN(num_classes=c["num_classes"], dropout=c["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if metric > self.best:
            self.best = metric
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience
