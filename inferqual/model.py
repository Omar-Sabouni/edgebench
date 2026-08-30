"""The deterministic vision workload used by InferQual."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_CLASSES = 10
DEFAULT_SEED = 2026


class TinyVisionNet(nn.Module):
    """A small CNN that is realistic enough to expose backend differences."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, OUTPUT_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def build_model(seed: int = DEFAULT_SEED) -> TinyVisionNet:
    """Build identical weights without disturbing the caller's random state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyVisionNet()
    return model.cpu().eval()


def make_input(seed: int, index: int = 0) -> npt.NDArray[np.float32]:
    """Return one deterministic, contiguous NCHW input."""
    generator = np.random.default_rng(np.random.SeedSequence([seed, index]))
    values = generator.normal(size=INPUT_SHAPE).astype(np.float32)
    return np.ascontiguousarray(values)


def make_inputs(seed: int, count: int) -> list[npt.NDArray[np.float32]]:
    """Return a deterministic input set with independent cases."""
    if count < 1:
        raise ValueError("count must be positive")
    return [make_input(seed, index) for index in range(count)]
