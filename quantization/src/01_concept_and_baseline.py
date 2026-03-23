"""
Part 1: Quantization 개념 및 Baseline 학습

이 스크립트에서 다루는 내용:
1. Quantization의 핵심 개념
2. CIFAR-10 데이터셋 로드
3. CNN 모델 학습 (Baseline)
4. 모델 크기 측정
5. 가중치 분포 시각화 및 파라미터 분석

Quantization 핵심 아이디어:
- 모델의 가중치와 활성화(activation)를 FP32에서 INT8 등 낮은 비트로 변환
- 모델 크기 ~4배 감소, 추론 속도 향상
- 정확도 손실을 최소화하는 것이 핵심 과제

양자화 수식:
===========
  q = round(x / scale) + zero_point
  x_dequant = (q - zero_point) * scale

여기서:
- scale = (x_max - x_min) / (q_max - q_min)
- zero_point = round(q_min - x_min / scale)

Quantization 분류:
=================
1. Dynamic Quantization (동적 양자화)
   - 가중치: 사전에 INT8로 변환
   - 활성화: 추론 시 동적으로 양자화
   - 가장 간단, calibration 불필요

2. Static Quantization (정적 양자화) = PTQ (Post-Training Quantization)
   - 가중치 + 활성화 모두 사전에 INT8로 변환
   - calibration 데이터 필요 (활성화 범위 측정)
   - 더 나은 성능, 별도 모델 구조 필요 (QuantStub/DeQuantStub)

3. Quantization-Aware Training (QAT)
   - 학습 중 양자화를 시뮬레이션 (Fake Quantization)
   - Straight-Through Estimator (STE)로 역전파
   - 가장 높은 정확도, 학습 비용 추가
"""

import sys
import os
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'module'))
from models import CNN


# =============================================================================
# 1. 환경 설정
# =============================================================================

def get_device() -> str:
    """사용 가능한 가속기 확인"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
print(f"Using device: {DEVICE}")

# 하이퍼파라미터
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# 재현성을 위한 시드 설정
SEED = 42
torch.manual_seed(SEED)


# =============================================================================
# 2. 데이터 로드 및 전처리
# =============================================================================

def get_data_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 데이터셋 로드

    정규화 값 설명:
    - mean=[0.485, 0.456, 0.406]: ImageNet 기준 RGB 채널별 평균
    - std=[0.229, 0.224, 0.225]: ImageNet 기준 RGB 채널별 표준편차
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )

    return train_loader, test_loader


# =============================================================================
# 3. 학습 및 평가 함수
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str
) -> list[float]:
    """
    일반적인 Cross-Entropy 학습

    Returns:
        각 에포크의 평균 loss 리스트
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


def test(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """
    모델 평가

    Returns:
        테스트 정확도 (%)
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 4. 모델 크기 측정
# =============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """
    모델의 저장 크기 측정 (MB)

    양자화의 주요 이점 중 하나는 모델 크기 감소입니다:
    - FP32: 파라미터당 4 bytes
    - INT8: 파라미터당 1 byte → ~4배 감소
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    tmp_path = tmp.name
    tmp.close()

    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)

    return size_mb


# =============================================================================
# 5. 가중치 분포 시각화
# =============================================================================

def visualize_weight_distribution(model: nn.Module, save_path: str = 'results/weight_distribution.png'):
    """
    모델의 가중치 분포 시각화

    양자화 전 가중치 분포를 확인하면:
    - 분포가 균일할수록 양자화 오차가 작음
    - outlier가 많으면 양자화 정확도 손실 가능
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weight Distribution by Layer (Before Quantization)', fontsize=14)

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    for idx, (name, layer) in enumerate(conv_layers):
        ax = axes[idx // 2][idx % 2]
        weights = layer.weight.data.cpu().numpy().flatten()

        ax.hist(weights, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f'{name}\n(shape: {list(layer.weight.shape)}, total: {weights.size:,})')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')

        # 통계 정보 표시
        stats_text = f'mean: {weights.mean():.4f}\nstd: {weights.std():.4f}\nmin: {weights.min():.4f}\nmax: {weights.max():.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"가중치 분포 시각화 저장: {save_path}")


def analyze_layer_parameters(model: nn.Module):
    """레이어별 파라미터 분석"""
    print("\n[레이어별 파라미터 분석]")
    print(f"{'Layer':<30} {'Shape':<25} {'Parameters':<15} {'비율':<10}")
    print("-" * 80)

    total_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        layer_info.append((name, list(param.shape), num_params))

    for name, shape, num_params in layer_info:
        ratio = num_params / total_params * 100
        print(f"{name:<30} {str(shape):<25} {num_params:>12,} {ratio:>8.1f}%")

    print("-" * 80)
    print(f"{'Total':<30} {'':<25} {total_params:>12,} {'100.0%':>10}")

    return total_params


# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    print("=" * 60)
    print("Part 1: Quantization - 개념 및 Baseline 학습")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] CIFAR-10 데이터셋 로드 중...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f"    - 학습 데이터: {len(train_loader.dataset):,} 샘플")
    print(f"    - 테스트 데이터: {len(test_loader.dataset):,} 샘플")

    # 모델 학습
    print("\n[2] CNN 모델 학습 (Baseline)")
    print("-" * 40)

    torch.manual_seed(SEED)
    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"    파라미터 수: {count_parameters(model):,}")

    train_losses = train(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    baseline_accuracy = test(model, test_loader, DEVICE)

    # 모델 저장
    os.makedirs('data/trained_models', exist_ok=True)
    torch.save(model.state_dict(), 'data/trained_models/baseline_model.pth')
    print("    Baseline 모델 저장: data/trained_models/baseline_model.pth")

    # 모델 크기 측정
    print("\n[3] 모델 크기 측정")
    print("-" * 40)
    model_size = get_model_size_mb(model)
    print(f"    FP32 모델 크기: {model_size:.2f} MB")
    print(f"    파라미터당 바이트: 4 bytes (float32)")
    print(f"    INT8 양자화 시 예상 크기: ~{model_size / 4:.2f} MB")

    # 레이어별 파라미터 분석
    print("\n[4] 레이어별 파라미터 분석")
    print("-" * 40)
    total_params = analyze_layer_parameters(model)

    # 가중치 분포 시각화
    print("\n[5] 가중치 분포 시각화")
    print("-" * 40)
    visualize_weight_distribution(model)

    # 양자화 관련 가중치 통계
    print("\n[6] 양자화 관련 가중치 통계")
    print("-" * 40)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight.data.cpu()
            w_min, w_max = weights.min().item(), weights.max().item()
            w_range = w_max - w_min
            # INT8 양자화 시 step size (scale)
            scale = w_range / 255  # uint8 기준
            print(f"    {name}:")
            print(f"      range: [{w_min:.4f}, {w_max:.4f}], "
                  f"dynamic range: {w_range:.4f}, "
                  f"INT8 step: {scale:.6f}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"    모델: CNN ({total_params:,} parameters)")
    print(f"    Baseline 정확도: {baseline_accuracy:.2f}%")
    print(f"    FP32 모델 크기: {model_size:.2f} MB")
    print(f"\n    이 모델에 다양한 Quantization 기법을 적용하여")
    print(f"    정확도를 유지하면서 모델 크기와 추론 속도를 개선하는 것이 목표입니다!")

    return {
        'baseline_accuracy': baseline_accuracy,
        'train_losses': train_losses,
        'total_params': total_params,
        'model_size_mb': model_size
    }


if __name__ == "__main__":
    results = main()
