"""
NAS 탐색 공간 및 FlexibleCNN 모델

Neural Architecture Search는 주어진 탐색 공간(Search Space)에서
최적의 아키텍처를 자동으로 찾는 기법입니다.

이 모듈은 다음을 정의합니다:
1. SEARCH_SPACE: 탐색 가능한 아키텍처 구성 요소
2. FlexibleCNN: 다양한 아키텍처 구성을 받아 동적으로 생성되는 CNN
3. 유틸리티 함수: 랜덤 아키텍처 샘플링, 탐색 공간 크기 계산 등
"""

import random
import copy

import torch
import torch.nn as nn


# =============================================================================
# 1. 탐색 공간 정의
# =============================================================================

SEARCH_SPACE = {
    # Conv 레이어별 필터 수 후보
    'filters': [16, 32, 64, 128],
    # Conv 레이어별 커널 크기 후보
    'kernel_sizes': [3, 5],
    # FC hidden units 후보
    'fc_hidden': [128, 256, 512, 1024],
    # Conv 레이어 수 (고정: 4개, Baseline과 동일한 구조 유지)
    'num_conv_layers': 4,
    # MaxPool 위치 (고정: 2번째, 4번째 Conv 뒤, 0-indexed)
    'pool_after': [1, 3],
}


def get_search_space_size() -> int:
    """
    탐색 공간의 전체 크기 계산

    각 Conv 레이어마다:
    - filters: 4가지 선택 (16, 32, 64, 128)
    - kernel_sizes: 2가지 선택 (3, 5)
    FC hidden: 4가지 선택

    전체 = (4 * 2)^4 * 4 = 16,384 가지 아키텍처
    """
    n_filters = len(SEARCH_SPACE['filters'])
    n_kernels = len(SEARCH_SPACE['kernel_sizes'])
    n_fc = len(SEARCH_SPACE['fc_hidden'])
    n_conv = SEARCH_SPACE['num_conv_layers']

    return (n_filters * n_kernels) ** n_conv * n_fc


def sample_architecture(seed: int = None) -> dict:
    """
    탐색 공간에서 랜덤 아키텍처 하나를 샘플링

    Args:
        seed: 재현성을 위한 시드 (None이면 랜덤)

    Returns:
        dict: 아키텍처 구성
        예시: {'filters': [64, 32, 128, 16], 'kernel_sizes': [3, 5, 3, 3], 'fc_hidden': 256}
    """
    if seed is not None:
        random.seed(seed)

    arch = {
        'filters': [
            random.choice(SEARCH_SPACE['filters'])
            for _ in range(SEARCH_SPACE['num_conv_layers'])
        ],
        'kernel_sizes': [
            random.choice(SEARCH_SPACE['kernel_sizes'])
            for _ in range(SEARCH_SPACE['num_conv_layers'])
        ],
        'fc_hidden': random.choice(SEARCH_SPACE['fc_hidden']),
    }
    return arch


def architecture_to_string(arch: dict) -> str:
    """아키텍처를 읽기 쉬운 문자열로 변환"""
    filters_str = '-'.join(str(f) for f in arch['filters'])
    kernels_str = '-'.join(str(k) for k in arch['kernel_sizes'])
    return f"F[{filters_str}]_K[{kernels_str}]_FC{arch['fc_hidden']}"


def mutate_architecture(arch: dict) -> dict:
    """
    아키텍처의 구성 요소 하나를 랜덤으로 변경 (돌연변이)

    변경 가능 요소:
    - filters[i]: 4개 Conv 레이어 중 하나의 필터 수 변경
    - kernel_sizes[i]: 4개 Conv 레이어 중 하나의 커널 크기 변경
    - fc_hidden: FC hidden units 변경

    총 9가지 가능한 mutation point 중 하나를 랜덤 선택
    """
    new_arch = copy.deepcopy(arch)

    mutation_points = (
        [f'filter_{i}' for i in range(SEARCH_SPACE['num_conv_layers'])] +
        [f'kernel_{i}' for i in range(SEARCH_SPACE['num_conv_layers'])] +
        ['fc']
    )
    point = random.choice(mutation_points)

    if point.startswith('filter_'):
        idx = int(point.split('_')[1])
        current = new_arch['filters'][idx]
        candidates = [f for f in SEARCH_SPACE['filters'] if f != current]
        new_arch['filters'][idx] = random.choice(candidates)
    elif point.startswith('kernel_'):
        idx = int(point.split('_')[1])
        current = new_arch['kernel_sizes'][idx]
        candidates = [k for k in SEARCH_SPACE['kernel_sizes'] if k != current]
        if candidates:
            new_arch['kernel_sizes'][idx] = random.choice(candidates)
    elif point == 'fc':
        current = new_arch['fc_hidden']
        candidates = [f for f in SEARCH_SPACE['fc_hidden'] if f != current]
        new_arch['fc_hidden'] = random.choice(candidates)

    return new_arch


# =============================================================================
# 2. FlexibleCNN 모델
# =============================================================================

class FlexibleCNN(nn.Module):
    """
    탐색 공간의 아키텍처 구성을 받아 동적으로 생성되는 CNN

    Baseline CNN과 동일한 전체 구조:
    - 4개의 Conv 블록 (Conv2d + ReLU, 2/4번째 뒤 MaxPool)
    - AdaptiveAvgPool2d로 공간 차원 통일 (4x4)
    - FC classifier (fc_hidden -> num_classes)

    AdaptiveAvgPool2d를 사용하여 다양한 커널 크기에서도
    FC 입력 크기가 일정하도록 보장합니다.

    Args:
        arch: 아키텍처 구성 딕셔너리
        num_classes: 출력 클래스 수 (기본 10)
    """

    def __init__(self, arch: dict, num_classes: int = 10):
        super(FlexibleCNN, self).__init__()

        self.arch = arch
        filters = arch['filters']
        kernel_sizes = arch['kernel_sizes']
        fc_hidden = arch['fc_hidden']
        pool_after = SEARCH_SPACE['pool_after']

        # Feature extractor 동적 구성
        layers = []
        in_channels = 3  # CIFAR-10 RGB

        for i in range(SEARCH_SPACE['num_conv_layers']):
            out_channels = filters[i]
            k = kernel_sizes[i]
            padding = k // 2  # same padding

            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=k, padding=padding))
            layers.append(nn.ReLU())

            if i in pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels

        # AdaptiveAvgPool2d로 공간 차원 통일 (커널 크기 변화에 대응)
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        self.features = nn.Sequential(*layers)

        # Classifier
        fc_input = filters[-1] * 4 * 4  # 마지막 필터 수 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
