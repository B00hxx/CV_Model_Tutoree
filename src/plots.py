"""All figures. Each function takes data in, writes a PNG out."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (WSL / headless)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .classes import CLASS_NAMES

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, path: Path):
    """Loss and accuracy curves: train vs val. Shows convergence and over/underfitting."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train", lw=2)
    ax1.plot(epochs, history["val_loss"], label="val", lw=2)
    ax1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax1.legend()

    ax2.plot(epochs, np.array(history["train_acc"]) * 100, label="train", lw=2)
    ax2.plot(epochs, np.array(history["val_acc"]) * 100, label="val", lw=2)
    best = int(np.argmax(history["val_acc"]))
    ax2.scatter(best + 1, history["val_acc"][best] * 100, s=80, zorder=3, color="black",
                label=f"best val {history['val_acc'][best]*100:.2f}%")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="%")
    ax2.legend()
    _save(fig, path)


def plot_lr(history: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history["lr"], lw=2)
    ax.set(title="Learning rate (OneCycle)", xlabel="step", ylabel="lr")
    _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: Path, num_classes: int = 43):
    """Row-normalized: each row sums to 1, so the diagonal is per-class recall."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes)).astype(float)
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([f"{i} {n}" for i, n in enumerate(CLASS_NAMES)], rotation=90, fontsize=7)
    ax.set_yticklabels([f"{i} {n}" for i, n in enumerate(CLASS_NAMES)], fontsize=7)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix (row-normalized) — test set")
    # Annotate off-diagonal errors only, so the confusions jump out
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=5, color="crimson")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_per_class_metrics(y_true, y_pred, path: Path, num_classes: int = 43):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    order = np.argsort(f1)  # worst classes first — that's what you want to look at
    fig, ax = plt.subplots(figsize=(10, 11))
    y = np.arange(num_classes)
    ax.barh(y, f1[order], color="steelblue", label="F1")
    ax.scatter(p[order], y, marker="|", s=120, color="black", label="precision")
    ax.scatter(r[order], y, marker="x", s=30, color="darkorange", label="recall")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CLASS_NAMES[i]} (n={support[i]})" for i in order], fontsize=7)
    lo = max(0.0, min(f1.min(), p.min(), r.min()) - 0.05)
    ax.set_xlim(lo, 1.005)
    ax.invert_yaxis()  # worst class at the top
    ax.set(title="Per-class metrics on test set (worst F1 at top)", xlabel="score")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_class_distribution(train_targets, test_targets, path: Path, num_classes: int = 43):
    tr = np.bincount(train_targets, minlength=num_classes)
    te = np.bincount(test_targets, minlength=num_classes)
    x = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.2, tr, width=0.4, label="train")
    ax.bar(x + 0.2, te, width=0.4, label="test")
    ax.set_xticks(x)
    ax.set(title="Class distribution (imbalanced -> report macro F1, not just accuracy)",
           xlabel="ClassId", ylabel="images")
    ax.legend()
    _save(fig, path)


def plot_confidence(probs, y_true, path: Path):
    """Confidence of correct vs wrong predictions + reliability (calibration) curve."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 26)
    ax1.hist(conf[correct], bins=bins, alpha=0.7, label=f"correct ({correct.sum()})", density=True)
    ax1.hist(conf[~correct], bins=bins, alpha=0.7, label=f"wrong ({(~correct).sum()})", density=True)
    ax1.set(title="Prediction confidence", xlabel="max softmax prob", ylabel="density")
    ax1.legend()

    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
    acc_bin = [correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    conf_bin = [conf[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    ece = sum(abs(a - c) * (idx == b).mean() for b, (a, c) in enumerate(zip(acc_bin, conf_bin))
              if not np.isnan(a))
    ax2.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax2.plot(conf_bin, acc_bin, "o-", lw=2, label=f"model (ECE={ece:.3f})")
    ax2.set(title="Reliability diagram", xlabel="confidence", ylabel="accuracy",
            xlim=(0, 1), ylim=(0, 1))
    ax2.legend()
    _save(fig, path)
    return float(ece)


def plot_misclassified(images, y_true, y_pred, conf, path: Path, max_n: int = 24):
    """images: (N,3,H,W) tensor already denormalized to [0,1]."""
    n = min(max_n, len(images))
    if n == 0:
        return
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for k, ax in enumerate(np.atleast_1d(axes).ravel()[:n]):
        ax.imshow(images[k].permute(1, 2, 0).numpy())
        ax.set_title(f"T: {CLASS_NAMES[y_true[k]]}\nP: {CLASS_NAMES[y_pred[k]]} ({conf[k]:.2f})",
                     fontsize=6.5, color="crimson")
    fig.suptitle("Misclassified test images (T=true, P=predicted)", fontsize=11)
    _save(fig, path)
