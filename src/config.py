"""Central configuration: every hyperparameter and path lives here.

Override anything from the command line, e.g.
    python train.py --epochs 30 --batch-size 256 --img-size 48
"""
import argparse
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: Path = Path("data/gtsrb")      # folder that holds Train/, Test/, Train.csv, Test.csv
    output_dir: Path = Path("outputs")       # checkpoints, plots, metrics

    # Data
    img_size: int = 48          # GTSRB images are ~30-250 px; 48x48 is the classic sweet spot
    use_roi: bool = True        # crop to the sign bounding box given in the CSVs
    val_frac: float = 0.15      # fraction of *tracks* held out for validation
    num_workers: int = 4

    # Training
    epochs: int = 25
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.3
    patience: int = 6           # early stopping on validation accuracy
    seed: int = 42
    amp: bool = True            # mixed precision on GPU

    num_classes: int = 43

    @property
    def ckpt_path(self) -> Path:
        return self.output_dir / "best_model.pt"

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def parse_args(argv=None) -> Config:
    """Builds an argparse parser from the dataclass fields so the two never drift apart."""
    cfg = Config()
    parser = argparse.ArgumentParser(description="GTSRB CNN")
    for f in fields(Config):
        default = getattr(cfg, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args(argv)
    return Config(**vars(args))
