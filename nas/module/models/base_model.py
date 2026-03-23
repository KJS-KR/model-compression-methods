"""
NAS 실험용 CNN 모델 (Baseline)

Neural Architecture Search는 최적의 아키텍처를 자동으로 탐색하는 기법입니다.
이 모델은 사람이 설계한 고정 아키텍처로, NAS가 찾은 아키텍처와
비교하기 위한 Baseline으로 사용됩니다.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    NAS 실험용 Baseline CNN 모델 (~1.18M parameters)

    Pruning/Distillation/Quantization과 동일한 구조를 사용하여
    동일 조건에서 압축 기법 간 비교가 가능합니다.

    구조:
    - 4개의 Convolutional layers (128 -> 64 -> 64 -> 32 filters)
    - 2개의 MaxPooling layers
    - Fully connected classifier (2048 -> 512 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 3 -> 128 -> 64
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # Block 2: 64 -> 64 -> 32
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )

        # Classifier
        # Input: 32 * 8 * 8 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (batch, 32, 8, 8) -> (batch, 2048)
        x = self.classifier(x)
        return x
