"""Step 2: test the saved model on the official GTSRB test set (12,630 images).

Writes to outputs/:
    test_metrics.json, classification_report.txt,
    confusion_matrix.png, per_class_metrics.png, confidence.png, misclassified.png

    python evaluate.py
    python evaluate.py --output-dir outputs/run2
"""
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, top_k_accuracy_score
from torch import nn

from src.classes import CLASS_NAMES
from src.config import parse_args
from src.data import build_loaders, denormalize
from src.engine import evaluate
from src.plots import (plot_confidence, plot_confusion_matrix, plot_misclassified,
                       plot_per_class_metrics)
from src.utils import get_device, load_model, save_json


def main():
    cfg = parse_args()
    device = get_device()
    model, ckpt = load_model(cfg.ckpt_path, device)
    print(f"Loaded {cfg.ckpt_path} (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']*100:.2f}%)")

    test_loader = build_loaders(cfg)["test"]
    loss, acc, targets, probs = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    y_true = targets.numpy()
    probs = probs.numpy()
    y_pred = probs.argmax(1)

    out = cfg.output_dir
    labels = list(range(cfg.num_classes))
    metrics = {
        "test_loss": loss,
        "accuracy": acc,
        "top3_accuracy": top_k_accuracy_score(y_true, probs, k=3, labels=labels),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "n_errors": int((y_pred != y_true).sum()),
        "n_test": int(len(y_true)),
    }
    metrics["ece"] = plot_confidence(probs, y_true, out / "confidence.png")
    plot_confusion_matrix(y_true, y_pred, out / "confusion_matrix.png", cfg.num_classes)
    plot_per_class_metrics(y_true, y_pred, out / "per_class_metrics.png", cfg.num_classes)

    # Misclassified examples, most confident mistakes first (the interesting ones)
    wrong = np.where(y_pred != y_true)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:24]
    ds = test_loader.dataset
    imgs = torch.stack([denormalize(ds[i][0]) for i in wrong]) if len(wrong) else torch.empty(0)
    plot_misclassified(imgs, y_true[wrong], y_pred[wrong], probs[wrong].max(1),
                       out / "misclassified.png")

    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(report)
    save_json(metrics, out / "test_metrics.json")

    print(report)
    print("\n".join(f"{k:>14}: {v:.4f}" if isinstance(v, float) else f"{k:>14}: {v}"
                    for k, v in metrics.items()))
    print(f"\nPlots and reports written to {out}/")


if __name__ == "__main__":
    main()
