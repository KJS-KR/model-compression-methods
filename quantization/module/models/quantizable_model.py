"""
Quantization 전용 CNN 모델

Static Quantization과 QAT(Quantization-Aware Training)를 위해서는
모델에 QuantStub/DeQuantStub을 추가하고, fuse_modules로
Conv+ReLU, Linear+ReLU를 융합해야 합니다.

기본 CNN과 동일한 구조이지만 양자화 지원을 위한 수정이 포함되어 있습니다.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quant


class QuantizableCNN(nn.Module):
    """
    양자화 가능한 CNN 모델 (~1.18M parameters)

    CNN과 동일한 구조에 다음이 추가됨:
    - QuantStub: 입력을 float → quantized 텐서로 변환
    - DeQuantStub: 출력을 quantized → float 텐서로 변환
    - fuse_model(): Conv+ReLU, Linear+ReLU 레이어 융합

    Static PTQ / QAT 모두 이 모델을 사용합니다.
    """

    def __init__(self, num_classes: int = 10):
        super(QuantizableCNN, self).__init__()

        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # Feature extractor (CNN과 동일)
        self.features = nn.Sequential(
            # Block 1: 3 -> 128 -> 64
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 -> 64 -> 32
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier (CNN과 동일, 단 Dropout 제거 — 양자화 호환성)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """
        Conv+ReLU, Linear+ReLU 레이어를 융합합니다.

        융합하면 하나의 연산으로 합쳐져서:
        - 양자화 시 정확도 손실 감소
        - 추론 속도 향상
        - 메모리 사용량 감소

        주의: Static PTQ에서는 eval() 모드에서 호출해야 합니다.
              QAT에서는 train() 모드에서 호출합니다.
        """
        torch.ao.quantization.fuse_modules(self, [
            ['features.0', 'features.1'],      # Conv2d(3,128) + ReLU
            ['features.2', 'features.3'],      # Conv2d(128,64) + ReLU
            ['features.5', 'features.6'],      # Conv2d(64,64) + ReLU
            ['features.7', 'features.8'],      # Conv2d(64,32) + ReLU
            ['classifier.0', 'classifier.1'],  # Linear(2048,512) + ReLU
        ], inplace=True)
