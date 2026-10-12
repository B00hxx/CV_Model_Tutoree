"""A compact VGG-style CNN (~0.6M params). Expect ~98%+ on the GTSRB test set.

Input 3x48x48:
    block1 -> 32 x 24 x 24
    block2 -> 64 x 12 x 12
    block3 -> 128 x 6 x 6
    head   -> 43 logits
"""
import torch
from torch import nn


def conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    # Two 3x3 convs = 5x5 receptive field with fewer params; BN before ReLU;
    # bias=False because BN already has a learnable shift.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout / 2),
    )


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = 43, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, dropout),
            conv_block(32, 64, dropout),
            conv_block(64, 128, dropout),
        )
        # Adaptive pool makes the head independent of the exact input size
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
