"""Step 1: train the CNN, keep the best checkpoint by validation accuracy.

Writes to outputs/:
    best_model.pt        state_dict + config (reload with src.utils.load_model)
    model_exported.pt2   torch.export program (torch.export.load(p).module(), no source code needed)
    history.json, config.json,
    training_curves.png, lr_schedule.png, class_distribution.png

    python train.py
    python train.py --epochs 40 --lr 2e-3
"""
import torch
from torch import nn

from src.config import parse_args
from src.data import build_loaders
from src.engine import evaluate, train_one_epoch
from src.model import TrafficSignCNN, count_params
from src.plots import plot_class_distribution, plot_history, plot_lr
from src.utils import EarlyStopping, get_device, load_model, save_checkpoint, save_json, set_seed


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    amp = cfg.amp and device.type == "cuda"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    loaders = build_loaders(cfg)
    print(f"train {len(loaders['train'].dataset)} | val {len(loaders['val'].dataset)} | "
          f"test {len(loaders['test'].dataset)} images")

    out = cfg.output_dir
    save_json(cfg.to_dict(), out / "config.json")
    plot_class_distribution(loaders["train"].dataset.targets, loaders["test"].dataset.targets,
                            out / "class_distribution.png", cfg.num_classes)

    model = TrafficSignCNN(cfg.num_classes, cfg.dropout).to(device)
    print(f"Model params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # plain CE for val so it's comparable across runs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(loaders["train"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    stopper = EarlyStopping(cfg.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                          scheduler, scaler, device, amp, history["lr"])
        va_loss, va_acc, _, _ = evaluate(model, loaders["val"], eval_criterion, device, amp)

        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"),
                        (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        improved, stop = stopper.step(va_acc)
        if improved:
            save_checkpoint(cfg.ckpt_path, model, cfg, epoch, va_acc)
        print(f"epoch {epoch:2d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc*100:6.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc*100:6.2f}%" + ("  * saved" if improved else ""))
        if stop:
            print(f"Early stopping: no val improvement for {cfg.patience} epochs.")
            break

    save_json(history, out / "history.json")
    plot_history(history, out / "training_curves.png")
    plot_lr(history, out / "lr_schedule.png")

    # Also export the best model with torch.export: loadable without this repo's source code
    # via torch.export.load(path).module(); batch size stays dynamic.
    best, ckpt = load_model(cfg.ckpt_path, torch.device("cpu"))
    example = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    exported = torch.export.export(best, (example,),
                                   dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
    torch.export.save(exported, out / "model_exported.pt2")

    print(f"\nBest val acc {ckpt['val_acc']*100:.2f}% at epoch {ckpt['epoch']}")
    print(f"Saved {cfg.ckpt_path} and {out / 'model_exported.pt2'}")


if __name__ == "__main__":
    main()
