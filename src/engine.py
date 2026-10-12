"""One-epoch train / evaluate loops. Kept free of plotting and I/O."""
import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp: bool,
                    lr_log: list | None = None):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # OneCycleLR steps per batch
        if lr_log is not None:
            lr_log.append(scheduler.get_last_lr()[0])

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool = False):
    """Returns loss, accuracy and the raw targets / probabilities for metrics."""
    model.eval()
    total_loss, seen = 0.0, 0
    all_probs, all_targets = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        seen += x.size(0)
        all_probs.append(logits.float().softmax(1).cpu())
        all_targets.append(y.cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    acc = (probs.argmax(1) == targets).float().mean().item()
    return total_loss / seen, acc, targets, probs
